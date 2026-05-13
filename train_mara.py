# train_mara.py
# -----------------------------------------------------------------------------
import os
import math
import time
import random
import numpy as np
import sys
import bisect
import shlex
import subprocess
import argparse
import yaml
from typing import Dict, Any, Optional
from contextlib import nullcontext
from datetime import timedelta, datetime

# ===== Torch & FSDP2 Imports =====
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh

# FSDP2 imports
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

# State dict utilities for checkpointing
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

# Use absolute path to ensure we get common_fsdp2, not common
common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)  # insert at the beginning to prioritize

from tokenizer_abstraction import get_tokenizer
from model_v2 import Transformer, ModelArgs
from configure_optimizers import configure_optimizers, summarize_optimizer_settings, MUON_FAMILY, DION_FAMILY, FSDP2_MUON_FAMILY, VALID_OPTIMIZER_TYPES
import logger
from dataloader import PercentageDataLoader, DataMixSchedule
from diagnostics import LayerDiagnostics
from tail_truncation import ProgressiveTailTruncation

import os
os.environ.setdefault('TORCH_KERNEL_CACHE_PATH', '/tmp/torch_kernels')

# -----------------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.*')
warnings.filterwarnings('ignore', message='You are using `torch.load` with `weights_only=False`', category=FutureWarning)

# -------------------------- Fatal Error Helper --------------------------
def fatal_error(message: str, exit_code: int = 1):
    """Print a clean error message and exit gracefully (no ugly stack trace).

    In distributed mode, only rank 0 prints the message, then all ranks exit together.
    """
    import time as _time

    # Determine if we're in distributed mode and get rank
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # Only rank 0 prints the error
    if rank == 0:
        print(f"\n{'='*60}")
        print("CONFIGURATION ERROR:")
        print(f"{'='*60}")
        for line in message.strip().split('\n'):
            print(f"  {line}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

    # In distributed mode, synchronize so all ranks exit together
    if dist.is_initialized():
        try:
            # First barrier ensures rank 0 has printed before anyone exits
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass  # If distributed is already torn down, just continue to exit

        # Small delay to let torchrun see all processes exiting together
        # This prevents the "sending SIGTERM" spam
        _time.sleep(0.5)

    # Use os._exit to avoid torch's ChildFailedError noise
    os._exit(exit_code)

# -------------------------- Validation --------------------------
def calc_group_loss(model, loader, *, eval_iters, device, ddp, dtype, device_type):
    """
    Return the mean loss for the group that `loader` is currently locked to.
    All ranks run the same number of iterations; the result is averaged across
    ranks before it is returned (so every rank gets the final scalar).

    Note: With FSDP2, we must call through the sharded model (not _orig_mod)
    because inputs need to interact with DTensor parameters properly.
    """
    tot = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=dtype):
                _, loss = model(x, y)
            tot += loss.detach()

    tot /= eval_iters                       # average over micro-batches
    if ddp:
        dist.all_reduce(tot, op=dist.ReduceOp.AVG)   # average across ranks
    return tot.item()                       # every rank now holds the same value

def do_validation(model, val_loader, device, eval_iters,
                  step, ddp_rank, val_log_file,
                  total_tokens_processed,
                  ddp, ddp_world_size, data_type, device_type):

    logger.print_and_log(f"[R{ddp_rank}] running validation at step {step}")

    t0 = time.time()
    model.eval()
    group_names = val_loader.group_names(active_only=True)
    group_losses = {}

    dtype = torch.bfloat16 if data_type == "bf16" else torch.float16 if data_type == "fp16" else torch.float32

    for g in group_names:
        val_loader.set_val_group(g, eval_iters = eval_iters)
        loss = calc_group_loss(model, val_loader, eval_iters=eval_iters, device=device, ddp=ddp, dtype=dtype, device_type=device_type)
        group_losses[g] = loss

        if ddp:
            dist.barrier()

    val_loader.set_val_group(None)

    if ddp_rank == 0:
        n_groups = len(group_names)
        # Get the percentages from val_loader groups
        group_percentages = {g.name: g.percentage for g in val_loader.groups}
        
        # Since this is validation, we can either:
        # 1. Use the configured percentages directly
        # 2. Or assume equal weighting for validation
        # Let's use the configured percentages normalized to sum to 1
        total_pct = sum(group_percentages[name] for name in group_names)
        if total_pct > 0:
            effective_probs = {name: group_percentages[name] / total_pct for name in group_names}
        else:
            # Fallback to equal weights if all percentages are 0
            effective_probs = {name: 1.0 / n_groups for name in group_names}
        
        # Format output with both loss and effective probability
        txt = " | ".join(f"{name[:7]}: {group_losses[name]:.4f} (P={effective_probs[name]:.3f})" 
                        for name in group_names)
        
        # Weight average by effective sampling probability
        avg = sum(group_losses[name] * effective_probs[name] for name in group_names)

        # calculate perplexity for display
        ppl = math.exp(avg)
        
        line = (f"st: {step:5d} | tok: {total_tokens_processed:11d} | {txt} | AVG: {avg:.4f} [{ppl:.4f}]")
        logger.print_and_log(line, True, val_log_file, silent=True)
        logger.print_and_log(line)

    logger.print_and_log(f"[R{ddp_rank}] validation done in {time.time()-t0:.1f}s")

    model.train()  # Return to training mode
    
    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Synchronize to ensure all operations are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# -------------------------- Learning Rate Schedule --------------------------
def get_lr(it: int, settings):
    """
    Unified LR getter that routes to the appropriate schedule based on settings.
    """
    schedule_type = getattr(settings, 'lr_schedule_type', 'restarts')  # Default to restarts for backwards compat

    if schedule_type == 'restarts':
        return get_lr_with_restarts(
            it, settings.max_lr, settings.min_lr, settings.warmup_steps,
            settings.max_steps, settings.restart_steps, settings.restart_gamma
        )
    elif schedule_type == 'cosine':
        return get_lr_with_restarts(
            it, settings.max_lr, settings.min_lr, settings.warmup_steps,
            settings.max_steps
        )
    elif schedule_type == 'plateau':
        # Dual plateau configuration
        return get_lr_with_dual_plateau(
            it, settings.max_lr, settings.min_lr, settings.warmup_steps,
            settings.max_steps,
            settings.first_plat_lr,
            settings.decay_to_first_plat_pct,
            settings.first_plat_len_pct,
            settings.decay_to_second_pct,
            settings.second_plat_lr,
            settings.second_plat_len_pct
        )
    else:
        fatal_error(f"Unknown lr_schedule_type: '{schedule_type}'\nValid options: 'cosine', 'restarts', 'plateau'")


def interpolate_lr_mod(schedule, step):
    """Linear interpolation of lr_mod schedule. Returns scale factor (1.0 = normal)."""
    if step <= schedule[0][0]:
        return schedule[0][1]
    if step >= schedule[-1][0]:
        return schedule[-1][1]
    for i in range(len(schedule) - 1):
        s0, v0 = schedule[i]
        s1, v1 = schedule[i + 1]
        if s0 <= step <= s1:
            t = (step - s0) / (s1 - s0)
            return v0 + t * (v1 - v0)
    return schedule[-1][1]


def parse_aux_heads_config(aux_cfg):
    """Parse the auxiliary_heads YAML block.

    Accepts:
      auxiliary_heads:
        enabled: true
        heads:
          - layer: 23
            weight: 0.05                       # scalar = constant
          - layer: 46
            weight: [[0, 0.001], [3000, 0.10]] # [[step, val], ...] = linear interp

    Returns:
        (sorted_layer_list, {layer_idx: schedule}) where schedule is a list of
        (step, weight) tuples suitable for interpolate_lr_mod. Scalar weights
        are normalized to [(0, value)] for uniform handling. Returns
        ([], {}) if disabled or absent.
    """
    if not aux_cfg or not aux_cfg.get('enabled', False):
        return [], {}
    heads = aux_cfg.get('heads') or []
    if not isinstance(heads, list) or len(heads) == 0:
        fatal_error("auxiliary_heads.heads must be a non-empty list when enabled.")
    layers = []
    schedules = {}
    for entry in heads:
        if not isinstance(entry, dict) or 'layer' not in entry or 'weight' not in entry:
            fatal_error(f"auxiliary_heads.heads entry must have 'layer' and 'weight' keys: {entry}")
        li = entry['layer']
        if not isinstance(li, int):
            fatal_error(f"auxiliary_heads.heads[*].layer must be an int, got {type(li).__name__}: {li}")
        w = entry['weight']
        if isinstance(w, (int, float)):
            sched = [(0, float(w))]
        elif isinstance(w, list) and len(w) > 0 and all(
            isinstance(wp, (list, tuple)) and len(wp) == 2 and isinstance(wp[0], int)
            and isinstance(wp[1], (int, float)) for wp in w
        ):
            sched = sorted([(int(p[0]), float(p[1])) for p in w], key=lambda x: x[0])
        else:
            fatal_error(
                f"auxiliary_heads.heads[*].weight must be a number or [[step, val], ...]: {w}"
            )
        if li in schedules:
            fatal_error(f"auxiliary_heads.heads has duplicate entry for layer {li}")
        schedules[li] = sched
        layers.append(li)
    return sorted(layers), schedules


def parse_lr_mods(lr_mods_config, model):
    """Parse lr_mods config and build param-to-schedule mapping.

    Entry formats:
      [name, schedule]              -- name is 'emb' or 'out'
      [all, type, schedule]         -- all layers; type is 'attn', 'ffn', or 'all'
      [start, end, type, schedule]  -- layer range; type is 'attn', 'ffn', or 'all'

    Last matching rule wins for each parameter.
    Returns list of (param, schedule) tuples.
    """
    import re
    param_schedules = {}  # id(param) -> (param, schedule)

    def _is_norm(n):
        return n.endswith('bias') or ('norm' in n.lower() and n.endswith('weight'))

    def _match_type(n, ptype):
        is_attn = 'attention.' in n and 'norm' not in n
        is_ffn = 'feed_forward.' in n
        if ptype == 'all':
            return is_attn or is_ffn
        elif ptype == 'attn':
            return is_attn
        elif ptype == 'ffn':
            return is_ffn
        return False

    for entry in lr_mods_config:
        if isinstance(entry[0], str) and len(entry) == 2:
            # [name, schedule] — emb or head
            name, schedule = entry
            for n, p in model.named_parameters():
                if name == 'emb' and 'tok_embeddings' in n:
                    param_schedules[id(p)] = (p, schedule)
                elif name == 'out' and n.startswith('output.'):
                    param_schedules[id(p)] = (p, schedule)

        elif isinstance(entry[0], str) and len(entry) == 3:
            # [all, type, schedule] — all layers with type filter
            _, ptype, schedule = entry
            for n, p in model.named_parameters():
                if _is_norm(n):
                    continue
                if _match_type(n, ptype):
                    param_schedules[id(p)] = (p, schedule)

        else:
            # [start, end, type, schedule] — layer range
            start, end, ptype, schedule = entry
            for n, p in model.named_parameters():
                m = re.match(r'layers\.(\d+)\.', n)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                if layer_idx < start or layer_idx > end:
                    continue
                if _match_type(n, ptype):
                    param_schedules[id(p)] = (p, schedule)

    return list(param_schedules.values())


def parse_wd_rules(wd_config, model):
    """Parse weight_decay rules list. Mirrors lr_mods format.

    Entry formats:
      [name, value_or_schedule]       — name is 'emb', 'out', or 'all'
      [start, end, value_or_schedule] — layer range (all non-norm params)

    Last matching rule wins. Norms always get WD=0.
    Returns list of (param, value_or_schedule) tuples.
    """
    import re
    param_wds = {}  # id(param) -> (param, value_or_schedule)

    def _is_norm(n):
        return n.endswith('bias') or ('norm' in n.lower() and n.endswith('weight'))

    for entry in wd_config:
        if isinstance(entry[0], str):
            # [name, value_or_schedule]
            name, wd_val = entry
            for n, p in model.named_parameters():
                if _is_norm(n):
                    continue
                if name == 'emb' and 'tok_embeddings' in n:
                    param_wds[id(p)] = (p, wd_val)
                elif name == 'out' and n.startswith('output.'):
                    param_wds[id(p)] = (p, wd_val)
                elif name == 'all' and 'layers.' in n:
                    param_wds[id(p)] = (p, wd_val)
        else:
            # [start, end, value_or_schedule]
            start, end, wd_val = entry
            for n, p in model.named_parameters():
                m = re.match(r'layers\.(\d+)\.', n)
                if not m:
                    continue
                layer_idx = int(m.group(1))
                if layer_idx < start or layer_idx > end:
                    continue
                if _is_norm(n):
                    continue
                param_wds[id(p)] = (p, wd_val)

    return list(param_wds.values())


def get_lr_with_restarts(it: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int, restart_steps=(), gamma: float = 1.0):
    """
    Cosine decay with linear warm-up **and** user-specified warm restarts.

    Args
    ----
    it            : current global step (0-based)
    max_lr,       : peak and floor learning-rates
    min_lr
    warmup_steps  : linear ramp length at the very start
    max_steps     : total training steps *excluding* future restarts
    restart_steps : iterable of global steps where LR should jump to a (possibly decayed) peak
    gamma         : multiplicative factor applied to the new peak LR at each restart (γ=1 → full reset, γ<1 → diminishing peaks)
    """
    # ---------- 1. optional warm-up  ----------
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # ---------- 2. locate the current cycle ----------
    idx = bisect.bisect_right(restart_steps, it)
    cycle_start = restart_steps[idx-1] if idx else warmup_steps
    cycle_end   = restart_steps[idx]   if idx < len(restart_steps) else max_steps

    # ---------- 3. compute per-cycle cosine ----------
    cycle_len     = cycle_end - cycle_start
    cycle_pos     = it - cycle_start
    decay_ratio   = min(1.0, cycle_pos / cycle_len)
    coeff         = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # ---------- 4. per-cycle η_max may decay by γ^idx ----------
    cycle_peak_lr = max_lr * (gamma ** idx)

    return min_lr + coeff * (cycle_peak_lr - min_lr)

def get_lr_with_dual_plateau(it: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int,
                             first_plat_lr: float, decay_to_first_plat_pct: float, first_plat_len_pct: float,
                             decay_to_second_pct: float, second_plat_lr: float, second_plat_len_pct: float):
    """
    Dual plateau schedule: Warmup → Decay to first plateau → Hold first → Decay to second → Hold second → Final decay

    Args:
        first_plat_lr: LR for first plateau
        decay_to_first_plat_pct: Percentage of max_steps to decay from max to first plateau
        first_plat_len_pct: Percentage of max_steps to hold first plateau
        decay_to_second_pct: Percentage of max_steps to decay from first to second plateau
        second_plat_lr: LR for second plateau
        second_plat_len_pct: Percentage of max_steps to hold second plateau

    Note: Final decay percentage is calculated as remainder
    """

    # Calculate phase boundaries
    decay_to_first_end = warmup_steps + int(decay_to_first_plat_pct * max_steps)
    first_plat_end = decay_to_first_end + int(first_plat_len_pct * max_steps)
    decay_to_second_end = first_plat_end + int(decay_to_second_pct * max_steps)
    second_plat_end = decay_to_second_end + int(second_plat_len_pct * max_steps)

    # Phase 1: Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # Phase 2: Cosine decay from max_lr to first_plat_lr
    if it < decay_to_first_end:
        progress = (it - warmup_steps) / (decay_to_first_end - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return first_plat_lr + coeff * (max_lr - first_plat_lr)

    # Phase 3: First plateau
    if it < first_plat_end:
        return first_plat_lr

    # Phase 4: Cosine decay from first_plat_lr to second_plat_lr
    if it < decay_to_second_end:
        progress = (it - first_plat_end) / (decay_to_second_end - first_plat_end)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return second_plat_lr + coeff * (first_plat_lr - second_plat_lr)

    # Phase 5: Second plateau
    if it < second_plat_end:
        return second_plat_lr

    # Phase 6: Final cosine decay to min_lr
    progress = (it - second_plat_end) / (max_steps - second_plat_end)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (second_plat_lr - min_lr)

def log_lr_schedule(settings, logger):
    """
    Print info about the learning rate schedule
    """
    schedule_type = getattr(settings, 'lr_schedule_type', 'restarts')
    logger.print_and_log(f"Learning Rate Schedule:")
    logger.print_and_log(f"  ] Schedule Type = {schedule_type}")
    logger.print_and_log(f"  ] Max LR        = {settings.max_lr:.6e}")
    logger.print_and_log(f"  ] Min LR        = {settings.min_lr:.6e}")
    logger.print_and_log(f"  ] Warmup Steps  = {settings.warmup_steps:,}")

    if schedule_type == 'cosine':
        pass  # warmup + cosine decay — no extra params to log

    elif schedule_type == 'restarts':
        logger.print_and_log(f"  ] Restart Steps = {settings.restart_steps}")
        logger.print_and_log(f"  ] Restart Gamma = {settings.restart_gamma:.2f}")

    elif schedule_type == 'plateau':

        # Dual plateau configuration
        logger.print_and_log(f"  ] Dual Plateau Configuration:")
        logger.print_and_log(f"  ]   First Plateau -> LR: {settings.first_plat_lr:.6e} ({settings.first_plat_lr/settings.max_lr*100:.1f}% of max)")
        logger.print_and_log(f"  ]   Second Plateau-> LR: {settings.second_plat_lr:.6e} ({settings.second_plat_lr/settings.max_lr*100:.1f}% of max)")

        # Calculate phase boundaries
        decay_to_first_end = settings.warmup_steps + int(settings.decay_to_first_plat_pct * settings.max_steps)
        first_plat_end = decay_to_first_end + int(settings.first_plat_len_pct * settings.max_steps)
        decay_to_second_end = first_plat_end + int(settings.decay_to_second_pct * settings.max_steps)
        second_plat_end = decay_to_second_end + int(settings.second_plat_len_pct * settings.max_steps)

        # Calculate final decay percentage
        final_decay_pct = 1.0 - settings.decay_to_first_plat_pct - settings.first_plat_len_pct - \
                            settings.decay_to_second_pct - settings.second_plat_len_pct - \
                            (settings.warmup_steps / settings.max_steps)

        logger.print_and_log(f"  ] Schedule phases:")
        logger.print_and_log(f"  ]   1. Warmup [{settings.warmup_steps/settings.max_steps*100:5.2f}%]:           steps 0-{settings.warmup_steps:,} → {settings.max_lr:.2e}")
        logger.print_and_log(f"  ]   2. Decay to first [{settings.decay_to_first_plat_pct*100:5.2f}%]:   steps {settings.warmup_steps:,}-{decay_to_first_end:,} → {settings.first_plat_lr:.2e}")
        logger.print_and_log(f"  ]   3. First plateau [{settings.first_plat_len_pct*100:5.2f}%]:    steps {decay_to_first_end:,}-{first_plat_end:,} @ {settings.first_plat_lr:.2e}")
        logger.print_and_log(f"  ]   4. Decay to second [{settings.decay_to_second_pct*100:5.2f}%]:  steps {first_plat_end:,}-{decay_to_second_end:,} → {settings.second_plat_lr:.2e}")
        logger.print_and_log(f"  ]   5. Second plateau [{settings.second_plat_len_pct*100:5.2f}%]:   steps {decay_to_second_end:,}-{second_plat_end:,} @ {settings.second_plat_lr:.2e}")
        logger.print_and_log(f"  ]   6. Final decay [{final_decay_pct*100:5.2f}%]:      steps {second_plat_end:,}-{settings.max_steps:,} → {settings.min_lr:.2e}")

class ActivationProbe:
    """RMS capture of forward activations at probe points per TransformerBlock.

    Per-layer captures: h_in_rms (block input), attn_out_rms (attention output),
    h_mid_rms (post_attn_norm output, KEEL with layer_id > 0 only),
    ffn_out_rms (FFN/MoE output), h_out_rms (block output).
    Top-level captures: final_norm_in_rms, final_norm_out_rms.

    Designed for sample-mode use:
        probe = ActivationProbe(model); probe.attach()
        with torch.no_grad():
            model(x, y)
        data = probe.detach_and_get()

    Hooks are removed by detach_and_get(). RMS values are computed eagerly on
    device during the forward and bulk-synced to host on detach.
    """

    def __init__(self, model):
        # Unwrap torch.compile root if present (per-submodule compile leaves the
        # root unwrapped, but be defensive).
        self.model = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.handles = []
        self._per_layer = {}   # {idx: {key: 0-d Tensor}}
        self._final = {}        # {key: 0-d Tensor}

    @staticmethod
    def _rms_tensor(t):
        if isinstance(t, tuple):
            t = t[0]
        return t.detach().float().pow(2).mean().sqrt()

    def _stash(self, idx, key, t):
        self._per_layer.setdefault(idx, {})[key] = self._rms_tensor(t)

    def attach(self):
        for idx, blk in enumerate(self.model.layers):
            i = idx  # capture by value
            self.handles.append(
                blk.register_forward_pre_hook(lambda m, a, _i=i: self._stash(_i, 'h_in_rms', a[0]))
            )
            self.handles.append(
                blk.register_forward_hook(lambda m, a, out, _i=i: self._stash(_i, 'h_out_rms', out))
            )

            if getattr(blk, 'use_gdn', False) and hasattr(blk, 'gdn_attn'):
                attn_mod = blk.gdn_attn
            elif hasattr(blk, 'attention'):
                attn_mod = blk.attention
            else:
                attn_mod = None
            if attn_mod is not None:
                self.handles.append(
                    attn_mod.register_forward_hook(
                        lambda m, a, out, _i=i: self._stash(_i, 'attn_out_rms', out)
                    )
                )

            if hasattr(blk, 'post_attn_norm'):
                self.handles.append(
                    blk.post_attn_norm.register_forward_hook(
                        lambda m, a, out, _i=i: self._stash(_i, 'h_mid_rms', out)
                    )
                )

            if getattr(blk, 'moe_enabled', False) and hasattr(blk, 'moe'):
                ffn_mod = blk.moe
            elif hasattr(blk, 'feed_forward'):
                ffn_mod = blk.feed_forward
            else:
                ffn_mod = None
            if ffn_mod is not None:
                self.handles.append(
                    ffn_mod.register_forward_hook(
                        lambda m, a, out, _i=i: self._stash(_i, 'ffn_out_rms', out)
                    )
                )

        # Top-level final norm: pre captures the activation entering the head,
        # post captures the output that feeds the LM head.
        norm = getattr(self.model, 'norm', None)
        if norm is not None:
            self.handles.append(
                norm.register_forward_pre_hook(
                    lambda m, a: self._final.__setitem__('final_norm_in_rms', self._rms_tensor(a[0]))
                )
            )
            self.handles.append(
                norm.register_forward_hook(
                    lambda m, a, out: self._final.__setitem__('final_norm_out_rms', self._rms_tensor(out))
                )
            )

    def detach_and_get(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        layers = {}
        for idx, d in self._per_layer.items():
            layers[idx] = {k: float(v.item()) for k, v in d.items()}
        final = {k: float(v.item()) for k, v in self._final.items()}
        self._per_layer = {}
        self._final = {}
        return {'layers_by_idx': layers, **final}


def _clip_grad_norm_mixed_mesh(model, max_norm, norm_type=2.0):
    """clip_grad_norm_ that works with params on different DTensor meshes.

    Standard clip_grad_norm_ uses torch.stack on per-param norms, which fails
    when params are DTensors on different meshes (e.g. dp_mesh vs edp_mesh in EP).

    Instead: compute local norm² per param, all-reduce across all ranks, then clip.
    This is correct because:
      - FSDP sharded params: each rank has 1/N of the grad, sum of partial norms² = full norm²
      - EP expert params: each rank has unique experts, sum gives total expert norm²
    """
    from torch.distributed.tensor import DTensor

    total_local_norm_sq = 0.0
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad)
        g = p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
        total_local_norm_sq += torch.linalg.vector_norm(g, norm_type).item() ** norm_type

    if not grads:
        return 0.0

    # All-reduce local norms² to get global norm²
    norm_tensor = torch.tensor(total_local_norm_sq, device=grads[0].device)
    dist.all_reduce(norm_tensor)
    total_norm = norm_tensor.item() ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

    return total_norm


# -------------------------- Training Loop --------------------------
def train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_local_rank, ddp_world_size, start_step,
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule,
        diagnostics: LayerDiagnostics = None,
        truncator: ProgressiveTailTruncation = None,
        wd_overrides: dict = None,
        lr_scale_overrides: dict = None,
        awd=None,
        moe_balance_hook=None,
    ):

    def sync_val_loader():
        """Sync val_loader group percentages from train_loader so validation tests all currently-mixed groups."""
        current_pcts = {g.name: g.percentage for g in train_loader.groups}
        val_loader.set_percentages_silent(current_pcts)

    # Parse auxiliary-head schedules once. Empty when the feature is disabled,
    # which keeps the per-step hot path free of dict lookups in that case.
    # Startup config printout happens at the top of train() — not here.
    _, aux_head_schedules = parse_aux_heads_config(getattr(settings, 'auxiliary_heads', None))
    aux_heads_enabled = bool(aux_head_schedules)

    # Do a baseline validation before training
    if start_step == 1:
        sync_val_loader()
        do_validation(model, val_loader, device, settings.eval_iters, 0, ddp_rank, settings.val_log_file, total_tokens_processed, ddp, ddp_world_size, settings.data_type, device_type)

    for step in range(start_step, settings.max_steps):
        t0 = time.time()
        last_step = (step == settings.max_steps - 1)

        # Progressive tail truncation — decide how many layers to run this step
        # Force full-depth on validation steps so diagnostics + val loss are clean
        is_val_step = (step % settings.val_step == 0 or last_step)
        active_layers = truncator.get_truncation_point(step) if (truncator and not is_val_step) else None
        # Normalize full-depth → None so torch.compile keeps its fast path
        if active_layers is not None and active_layers >= model_cfg.n_layers:
            active_layers = None
        is_truncated = active_layers is not None
        trunc_loss_w = truncator.get_loss_weight(active_layers or model_cfg.n_layers) if truncator else 1.0

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        # Per-step aux loss bookkeeping. Schedule weights are evaluated once
        # per step (constant within a step, lerp across waypoints). Per-head
        # unweighted CE values accumulate across micro-batches for logging.
        aux_weights_now = (
            {li: interpolate_lr_mod(sched, step) for li, sched in aux_head_schedules.items()}
            if aux_heads_enabled else {}
        )
        aux_loss_accum = (
            {li: torch.zeros((), device=device) for li in aux_head_schedules}
            if aux_heads_enabled else {}
        )
        grad_accum_steps = grad_accum_schedule[step]
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(step=step)
            x, y = x.to(device), y.to(device)
            # Avoid gradient synchronization on intermediate micro-steps
            # (big speedup with FSDP/DDP when grad_accum_steps > 1)
            sync_ctx = nullcontext()
            if ddp and micro_step < grad_accum_steps - 1:
                # torch.compile may wrap the module; handle both cases robustly
                if hasattr(model, "no_sync"):
                    sync_ctx = model.no_sync()
                elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod, "no_sync"):
                    sync_ctx = model._orig_mod.no_sync()

            with sync_ctx:
                # Optionally bypass torch.compile on truncated steps to avoid
                # caching a separate compiled graph for every unique truncation
                # point.  Set bypass_compile: false to go through compile anyway.
                fwd_model = model
                if is_truncated and truncator.bypass_compile and hasattr(model, '_orig_mod'):
                    fwd_model = model._orig_mod
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if settings.data_type=="bf16" else torch.float16 if settings.data_type=="fp16" else torch.float32):
                    logits, loss = fwd_model(x, y, active_layers=active_layers)

                # Auxiliary prediction heads: fold weighted aux losses into the
                # main loss so backward() drives gradients through both. Per-head
                # unweighted values accumulate for logging.
                if aux_heads_enabled:
                    raw_for_aux = fwd_model._orig_mod if hasattr(fwd_model, '_orig_mod') else fwd_model
                    aux_tensors = getattr(raw_for_aux, '_last_aux_loss_tensors', None) or {}
                    if aux_tensors:
                        for li, t in aux_tensors.items():
                            w = aux_weights_now.get(li, 0.0)
                            if w != 0.0:
                                loss = loss + w * t
                            aux_loss_accum[li] = aux_loss_accum[li] + (t.detach().float() / grad_accum_steps)

                loss = loss * trunc_loss_w / grad_accum_steps
                loss_accum += loss.detach().float()
                loss.backward()

        # Capture gradient norms before optimizer.step() clears them
        # Only capture on steps right before validation to avoid overhead.
        # Skip truncated steps — skipped layers have zero gradients which
        # would corrupt the diagnostics snapshot.
        if diagnostics is not None and not is_truncated and (step % settings.val_step == 0 or last_step):
            diagnostics.capture_gradients()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            for _li in aux_loss_accum:
                dist.all_reduce(aux_loss_accum[_li], op=dist.ReduceOp.AVG)

        # Moonlight-style: unified LR for all param groups
        # With rms_scale=True, the 0.2*sqrt(max(A,B)) scaling in Muon
        # provides the appropriate per-layer ratio automatically
        scheduled_lr = get_lr(step, settings)
        for param_group in optimizer.param_groups:
            lr = param_group.get('lr')
            if isinstance(lr, torch.Tensor):
                lr.fill_(scheduled_lr)
            else:
                # Use tensor to stay compatible with TorchAO 8-bit optimizers
                param_group['lr'] = torch.tensor(scheduled_lr)

        lr = scheduled_lr
        
        clip_value = settings.clip_warmup if step < settings.warmup_steps else settings.clip_standard
        norm = _clip_grad_norm_mixed_mesh(model, clip_value)

        # Apply per-parameter LR scaling from lr_mods (via side-dict)
        if lr_mod_entries and lr_scale_overrides is not None:
            for param, schedule in lr_mod_entries:
                scale = interpolate_lr_mod(schedule, step)
                lr_scale_overrides[id(param)] = scale

            # For standalone Adam: apply scales via param_group lr
            if settings.optimizer_type not in FSDP2_MUON_FAMILY:
                for param_group in optimizer.param_groups:
                    wd_group = param_group.get('wd_group', 'default')
                    if wd_group == 'norm_bias':
                        continue  # norms always unscaled
                    for p in param_group['params']:
                        scale = lr_scale_overrides.get(id(p), 1.0)
                        if scale != 1.0:
                            new_lr = scheduled_lr * scale
                            group_lr = param_group.get('lr')
                            if isinstance(group_lr, torch.Tensor):
                                group_lr.fill_(new_lr)
                            else:
                                param_group['lr'] = torch.tensor(new_lr)
                            break

        # AWD: compute gradient norms and update multipliers (every check_interval steps)
        awd_updated = False
        if awd is not None and not is_truncated:
            awd_updated = awd.compute_and_update(step)

        # Apply per-param weight decay from rules (via side-dict)
        if wd_entries and wd_overrides is not None:
            for param, wd_val in wd_entries:
                if isinstance(wd_val, list):
                    wd_overrides[id(param)] = interpolate_lr_mod(wd_val, step)
                else:
                    wd_overrides[id(param)] = float(wd_val)

        # AWD: apply multipliers on top of base WD
        if awd is not None:
            awd.apply_multipliers()
            if awd_updated and ddp_rank == 0:
                logger.print_and_log(f"  {awd.format_log_line(step)}")

        # Debug: log current lr_mod and wd values per rule target
        if (lr_mod_entries or wd_entries) and ddp_rank == 0:
            def _rule_label(e, has_type=False):
                if isinstance(e[0], str) and len(e) == 2:
                    return e[0]
                elif isinstance(e[0], str) and len(e) == 3:
                    return f"{e[0]}.{e[1]}"
                elif has_type:
                    return f"{e[0]}-{e[1]}.{e[2]}"
                else:
                    return f"{e[0]}-{e[1]}"

            def _next_target(sched, step):
                """Return (step, value) of next schedule point, or None if at/past end."""
                if not isinstance(sched, list) or len(sched) <= 1:
                    return None
                if step >= sched[-1][0]:
                    return None
                for i in range(len(sched) - 1):
                    if step < sched[i + 1][0]:
                        return (int(sched[i + 1][0]), sched[i + 1][1])
                return None

            def _fmt_val(sched, step):
                """Format current value + next target if scheduled."""
                if isinstance(sched, list):
                    val = interpolate_lr_mod(sched, step)
                    nxt = _next_target(sched, step)
                    if nxt:
                        return f"{val:.4f} ({nxt[0]}\u2192{nxt[1]})"
                    return f"{val:.4f}"
                return f"{float(sched):.4f}"

        # if lr_mod_entries and ddp_rank == 0:
        #     parts = [f"[{_rule_label(e, has_type=True)}]: {_fmt_val(e[-1], step)}" for e in settings.lr_mods]
        #     logger.print_and_log(f"  dbg lr: {' '.join(parts)}")
        # if wd_entries and ddp_rank == 0:
        #     parts = [f"[{_rule_label(e)}]: {_fmt_val(e[-1], step)}" for e in settings.weight_decay]
        #     logger.print_and_log(f"  dbg wd: {' '.join(parts)}")

        # MoE: update expert_bias from accumulated tokens_per_expert before step
        if moe_balance_hook is not None:
            moe_balance_hook()

        # Snapshot weights for update ratio diagnostic (before step modifies them)
        is_diag_step = diagnostics is not None and not is_truncated and (step % settings.val_step == 0 or last_step)
        if is_diag_step:
            diagnostics.snapshot_weights()

        optimizer.step()

        # Compute update norms from pre/post step diff
        if is_diag_step:
            diagnostics.capture_updates()
        # Synchronization slows down training, so rough (unsynchronized) timings are fine!
        #if device_type == "cuda":
        #    torch.cuda.synchronize()

        if ddp_rank == 0:
            dt = time.time() - t0
            tokens_per_step = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)
            total_tokens_processed += tokens_per_step
            tokens_per_sec = tokens_per_step / dt

            mfu = compute_mfu(tokens_per_sec, flops_per_token, ddp_world_size, settings.data_type) * 100
        
            ppl = math.exp(loss_accum.item())
            trunc_tag = f" | trunc: {active_layers if is_truncated else model_cfg.n_layers}/{model_cfg.n_layers}" if (truncator and truncator.enabled) else ""
            bal_tag = f" | bal: {moe_stats[0]['avg_cv']:.3f}" if moe_stats[0] else ""
            drp_tag = f" | drp: {moe_stats[0]['drop_pct']:.2f}%" if moe_stats[0] and moe_stats[0]['drop_pct'] > 0 else ""
            aux_tag = ""
            aux_silent = ""
            if aux_loss_accum:
                aux_pairs = sorted((li, v.item()) for li, v in aux_loss_accum.items())
                aux_tag = " | " + " ".join(f"aux_l{li}: {v:.4f}" for li, v in aux_pairs)
                aux_silent = "|" + "|".join(f"aux_l{li}={v:.6f}" for li, v in aux_pairs)
            logger.print_and_log(
                f"st: {step:5d} | ls: {loss_accum.item():.6f} | ppl: {ppl:.2f} | lr: {lr:.4e} | nrm: {norm:.4f} [{clip_value:.1f}] | dt: {dt:.2f}s | t_tk: {total_tokens_processed:11,d} | tok/s: {tokens_per_sec:.0f} | MFU: {mfu:.0f}%{bal_tag}{drp_tag}{trunc_tag}{aux_tag}",
            )

            logger.print_and_log(
                f"{step:5d}|{loss_accum.item():.6f}|{ppl:.2f}|{lr:.4e}|{norm:.4f}|{dt:.2f}|{total_tokens_processed:11d}|{tokens_per_sec:.0f}{aux_silent}",
                True, settings.train_log_file, silent=True
            )

        if step in settings.restart_steps:
            logger.print_and_log(f"Warm restart: LR {lr:.4e} → clip {clip_value:.2f}")  # TODO: Why am I showing the clip value here?

        if step % settings.val_step == 0 or last_step:
            sync_val_loader()
            do_validation(model, val_loader, device, settings.eval_iters, step, ddp_rank, settings.val_log_file, total_tokens_processed, ddp, ddp_world_size, settings.data_type, device_type)

            # Log current data mix if using data annealing
            train_loader.log_schedule_status(step, ddp_rank, logger.print_and_log)

            # Log output-head LR when batch-adjusted scaling is active
            if settings.output_lr_batch_adjust is not None and ddp_rank == 0:
                out_sched = None
                for entry in settings.lr_mods or []:
                    if isinstance(entry[0], str) and len(entry) == 2 and entry[0] == 'out':
                        out_sched = entry[1]
                        break
                if out_sched:
                    current_mult = interpolate_lr_mod(out_sched, step)
                    nxt = next(((s, m) for s, m in out_sched if s > step), None)
                    nxt_str = f"  (next: step {nxt[0]:,} \u2192 mult {nxt[1]:.4f})" if nxt else "  (final stage)"
                    logger.print_and_log(
                        f"  Output head LR: mult={current_mult:.4f} | eff_lr={scheduled_lr * current_mult:.4e} | body_lr={scheduled_lr:.4e}{nxt_str}"
                    )

            # Compute output-head feedback gain on a fresh val batch.
            # Runs the model's eval branch (non-CCE) so h and logits are
            # exposed in the autograd graph; uses torch.autograd.grad so
            # model.grad fields are not touched. One extra val batch consumed.
            if diagnostics is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    fb_x, fb_y = val_loader.next_batch(step=step)
                    # Slice down: eval branch materializes [B*T, vocab] logits (no CCE
                    # fusion), which OOMs at full B*T after train+val have saturated memory.
                    # 1x256 gives ~33M grad_logits elements per rank — plenty for an RMS ratio.
                    fb_x = fb_x[:1, :256].to(device)
                    fb_y = fb_y[:1, :256].to(device)
                    with torch.autocast(device_type=device_type,
                                        dtype=torch.bfloat16 if settings.data_type=="bf16"
                                              else torch.float16 if settings.data_type=="fp16"
                                              else torch.float32):
                        diagnostics.compute_feedback_gain(fb_x, fb_y)
                    del fb_x, fb_y
                except Exception as e:
                    if ddp_rank == 0:
                        logger.print_and_log(f"  feedback_gain compute failed: {type(e).__name__}: {e}")

            # Persistent activation diagnostics: forward-activation RMS profile
            # captured via hooks on a tiny val batch. Eval mode disables
            # activation checkpointing inside TransformerBlock.forward, so each
            # block's intermediates are materialized once and visible to hooks.
            activation_data = None
            if diagnostics is not None:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ap_x, ap_y = val_loader.next_batch(step=step)
                    ap_x = ap_x[:1, :256].to(device)
                    ap_y = ap_y[:1, :256].to(device)
                    probe = ActivationProbe(model)
                    probe.attach()
                    try:
                        model.eval()
                        with torch.no_grad(), torch.autocast(
                            device_type=device_type,
                            dtype=torch.bfloat16 if settings.data_type == "bf16"
                                  else torch.float16 if settings.data_type == "fp16"
                                  else torch.float32,
                        ):
                            model(ap_x, ap_y)
                        activation_data = probe.detach_and_get()
                    finally:
                        # Ensure hooks always come off even if forward raised.
                        if probe.handles:
                            for _h in probe.handles:
                                _h.remove()
                            probe.handles = []
                        model.train()
                    del ap_x, ap_y
                except Exception as e:
                    activation_data = None
                    if ddp_rank == 0:
                        logger.print_and_log(f"  activation_probe failed: {type(e).__name__}: {e}")

            # Log layer diagnostics after validation
            if diagnostics is not None:
                awd_diag = awd.get_diagnostics_data() if awd is not None else None
                moe_diag = moe_stats[0] if moe_stats[0] else None
                snapshot = diagnostics.log_diagnostics(
                    step, settings.nas_path, total_tokens_processed,
                    awd_data=awd_diag, moe_data=moe_diag,
                    activation_data=activation_data,
                )
                diagnostics.print_summary(snapshot, logger, awd_data=awd_diag, moe_data=moe_diag)

            # Log MoE expert utilization (only on rank 0, only when MoE is active)
            if moe_stats[0] and ddp_rank == 0:
                _ms = moe_stats[0]
                has_drops = _ms.get('total_dropped', 0) > 0
                _dc = _ms.get('drop_counts', [])
                logger.print_and_log(f"  === MoE Expert Utilization @ step {step} (CV = std/mean, lower = better) ===")
                if has_drops:
                    logger.print_and_log(f"  {'layer':>7s} | {'min%':>6s} {'max%':>6s} {'CV':>7s} | {'bias range':>20s} | {'dropped':>8s}")
                    logger.print_and_log(f"  {'-------':>7s} | {'------':>6s} {'------':>6s} {'-------':>7s} | {'--------------------':>20s} | {'--------':>8s}")
                else:
                    logger.print_and_log(f"  {'layer':>7s} | {'min%':>6s} {'max%':>6s} {'CV':>7s} | {'bias range':>20s}")
                    logger.print_and_log(f"  {'-------':>7s} | {'------':>6s} {'------':>6s} {'-------':>7s} | {'--------------------':>20s}")
                for i, (lid, pct, cv, bias) in enumerate(_ms['per_layer']):
                    b_min, b_max = min(bias), max(bias)
                    line = f"  L{lid:>5d} | {min(pct):6.1f} {max(pct):6.1f} {cv:7.4f} | [{b_min:+.4f}, {b_max:+.4f}]"
                    if has_drops:
                        dc = _dc[i] if i < len(_dc) else 0
                        line += f" | {dc:>8d}"
                    logger.print_and_log(line)
                summary = f"  {'avg':>7s} | {'':>6s} {'':>6s} {_ms['avg_cv']:7.4f} |"
                if has_drops:
                    summary += f" {'':>20s} | {_ms['total_dropped']:>8d}"
                logger.print_and_log(summary)

        if step > 0 and (step % settings.save_step == 0 or last_step):
            train_loader.log_detailed_dataloader_status(step, ddp_rank)
            save_model(model, optimizer, model_cfg, step, ddp_rank, ddp_local_rank, train_loader, total_tokens_processed, settings, awd=awd)

# -------------------------- Checkpointing and Resuming --------------------------
# Helper function to get the seeds for all random number generators
def get_rank_rng_state():
    checkpoint_data = {}
    checkpoint_data["random_state"] = random.getstate()
    checkpoint_data["numpy_state"] = np.random.get_state()
    checkpoint_data["rng_state"] = torch.get_rng_state().cpu().numpy()
    if torch.cuda.is_available():
        checkpoint_data["cuda_rng_state"] = torch.cuda.get_rng_state().cpu().numpy()
    return checkpoint_data

# Save the model, optimizer state, RNG state, trainloader state, etc.
def save_model(model, optimizer, model_config, step, ddp_rank, ddp_local_rank, train_loader, total_tokens_processed, settings, awd=None):
    # DEBUG: Print rank info at the very start
    logger.print_and_log("SAVE_MODEL: Starting checkpoint save...")
    os.makedirs(settings.local_checkpoint_dir, exist_ok=True)               # Ensure the checkpoint directory exists
    dist.barrier()

    # CRITICAL: Clear any gradient state before saving
    # This ensures no stale gradient shards persist
    logger.print_and_log("  ] Clearing gradients before save...")
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)

    # Get the underlying model (handle torch.compile wrapper)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # FSDP2: Use get_model_state_dict to gather full state dict to rank 0
    logger.print_and_log("  ] Getting full model state dict on CPU...")
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(raw_model, options=options)

    # ── EP Expert Consolidation ──
    # With Expert Parallel, each rank holds only num_local_experts.  get_model_state_dict
    # returns each rank's LOCAL expert params.  Gather all experts via all_gather and save
    # to a SEPARATE file (not the main checkpoint) so that:
    #   - Resume can restore the correct per-rank expert weights (the main checkpoint
    #     only has rank 0's experts, so set_model_state_dict silently gives all ranks
    #     rank 0's experts — this file fixes that).
    #   - Inference can load all experts into a single-GPU model with ep_degree=1.
    ep_degree = getattr(model_config, 'ep_degree', 1)
    _ep_experts_dict = None
    if ep_degree > 1 and getattr(model_config, 'moe_enabled', False):
        ep_group = None
        for layer in raw_model.layers:
            if getattr(layer, 'moe_enabled', False) and layer.moe._ep_group is not None:
                ep_group = layer.moe._ep_group
                break
        if ep_group is not None:
            # Guard: _local_tensor gives the full local expert only when edp_mesh=1
            # (no FSDP sharding within experts). With edp_mesh>1, _local_tensor is
            # an FSDP shard and the all_gather would produce garbage.
            edp_size = ddp_world_size // ep_degree
            assert edp_size == 1, (
                f"EP expert consolidation requires edp_mesh size 1 (ep_degree == world_size). "
                f"Got edp_size={edp_size} (world_size={ddp_world_size}, ep_degree={ep_degree}). "
                f"Save/resume with edp_mesh > 1 is not yet supported."
            )

            device = f'cuda:{ddp_local_rank}'
            # Build expert key→param mapping from the MODEL (available on all ranks),
            # not from full_sd (which is empty on non-rank-0 with full_state_dict=True).
            expert_params = {}
            for layer in raw_model.layers:
                if not getattr(layer, 'moe_enabled', False):
                    continue
                lid = layer.layer_id
                for pname in ('w1', 'w2', 'w3'):
                    key = f'layers.{lid}.moe.experts.{pname}'
                    param = getattr(layer.moe.experts, pname)
                    # Get local tensor data (handles DTensor from FSDP wrapping)
                    local_t = param._local_tensor if hasattr(param, '_local_tensor') else param.data
                    expert_params[key] = local_t

            logger.print_and_log(f"  ] Consolidating {len(expert_params)} expert params from {ep_degree} EP ranks...")
            _ep_experts_dict = {}
            for key, local_t in expert_params.items():
                local_t = local_t.to(device).contiguous()
                gathered = [torch.empty_like(local_t) for _ in range(ep_degree)]
                dist.all_gather(gathered, local_t, group=ep_group)
                if ddp_rank == 0:
                    _ep_experts_dict[key] = torch.cat(gathered, dim=0).cpu()
                del gathered
            del expert_params
            torch.cuda.empty_cache()
            if ddp_rank != 0:
                _ep_experts_dict = None  # Only rank 0 needs the consolidated dict
            elif _ep_experts_dict:
                first_key = next(iter(_ep_experts_dict))
                logger.print_and_log(f"  ] Expert consolidation complete ({first_key.split('.')[-1]} shape: {_ep_experts_dict[first_key].shape})")

    # Save and dump the model (only rank 0 has the full state dict)
    if ddp_rank == 0:
        # Save model configuration as a dictionary (your new format)
        model_config_dict = {
            'dim': model_config.dim,
            'n_layers': model_config.n_layers,
            'n_heads': model_config.n_heads,
            'n_kv_heads': model_config.n_kv_heads,
            'vocab_size': model_config.vocab_size,
            'inner_dim': model_config.inner_dim,
#            'multiple_of': model_config.multiple_of,
            'norm_eps': model_config.norm_eps,
            'max_seq_len': model_config.max_seq_len,
            'dropout': model_config.dropout,
            'pad_id': model_config.pad_id,
            'use_activation_checkpointing': model_config.use_activation_checkpointing,
            'qk_norm_mode': model_config.qk_norm_mode,
            'tie_word_embeddings': getattr(model_config, 'tie_word_embeddings', True),
            'rope_theta': getattr(model_config, 'rope_theta', 500000.0),
            'use_keel': getattr(model_config, 'use_keel', False),
            'keel_alpha': getattr(model_config, 'keel_alpha', None),
            # MoE configuration
            'moe_enabled': getattr(model_config, 'moe_enabled', False),
            'moe_num_experts': getattr(model_config, 'moe_num_experts', 8),
            'moe_top_k': getattr(model_config, 'moe_top_k', 2),
            'moe_num_shared_experts': getattr(model_config, 'moe_num_shared_experts', 1),
            'moe_score_func': getattr(model_config, 'moe_score_func', 'sigmoid'),
            'moe_score_before_experts': getattr(model_config, 'moe_score_before_experts', True),
            'moe_route_norm': getattr(model_config, 'moe_route_norm', False),
            'moe_route_scale': getattr(model_config, 'moe_route_scale', 1.0),
            'moe_load_balance_coeff': getattr(model_config, 'moe_load_balance_coeff', 1e-3),
            'moe_interleave_step': getattr(model_config, 'moe_interleave_step', 1),
            'moe_n_dense_layers': getattr(model_config, 'moe_n_dense_layers', 0),
            'moe_n_tail_dense_layers': getattr(model_config, 'moe_n_tail_dense_layers', 0),
            'moe_capacity_factor': getattr(model_config, 'moe_capacity_factor', 0.0),
            'moe_inner_dim': getattr(model_config, 'moe_inner_dim', None),
            'ep_degree': getattr(model_config, 'ep_degree', 1),
            # GDN hybrid attention
            'gdn_enabled': getattr(model_config, 'gdn_enabled', False),
            'gdn_interleave_step': getattr(model_config, 'gdn_interleave_step', 4),
            'n_gdn_heads': getattr(model_config, 'n_gdn_heads', None),
            'gdn_head_dim': getattr(model_config, 'gdn_head_dim', None),
            'gdn_v_expand': getattr(model_config, 'gdn_v_expand', 2.0),
            'gdn_short_conv_kernel': getattr(model_config, 'gdn_short_conv_kernel', 4),
            'gdn_mode': getattr(model_config, 'gdn_mode', 'chunk'),
        }
        checkpoint_data = {
            "model": full_sd,
            "config": model_config_dict,
            "step": step,
            "total_tokens_processed": total_tokens_processed,
            "tok_kind": settings.tok_kind,
            "tok_path": settings.tok_path,
            "special_tokens": getattr(settings, 'special_tokens', None),
            "optimizer_type": settings.optimizer_type,
            "max_lr": settings.max_lr,
            "cpu_offload": getattr(settings, 'cpu_offload', False),
            "checkpoint_version": "3.0",            # FSDP2 checkpoint format
        }
        checkpoint_path = os.path.join(settings.local_checkpoint_dir, f"model_step_{step:06d}.pt")
        logger.print_and_log(f"  ] Saving model checkpoint to {checkpoint_path}...")
        torch.save(checkpoint_data, checkpoint_path)
        del full_sd
        torch.cuda.empty_cache()

        # Save consolidated EP expert weights immediately after main checkpoint
        # (minimizes crash window where main ckpt exists but EP file doesn't).
        if _ep_experts_dict is not None:
            ep_path = os.path.join(settings.local_checkpoint_dir, f"ep_experts_step_{step:06d}.pt")
            torch.save(_ep_experts_dict, ep_path)
            logger.print_and_log(f"  ] EP expert weights saved ({len(_ep_experts_dict)} params, {ep_degree} ranks consolidated)")
            del _ep_experts_dict

    dist.barrier()  # Ensure all ranks wait for rank 0 to finish saving model

    # FSDP2: Save optimizer state as shards (each rank saves its own shard)
    # get_optimizer_state_dict returns the local shard for this rank
    # Note: unwrap AdamC wrappers — they delegate to a _base_optimizer but don't
    # inherit from torch.optim.Optimizer, which the FSDP checkpoint API requires.
    logger.print_and_log("  ] Saving optimizer state in shards...")
    optim_options = StateDictOptions(full_state_dict=False)  # Keep sharded
    save_optim = getattr(optimizer, '_base_optimizer', optimizer)
    optim_sd = get_optimizer_state_dict(raw_model, save_optim, options=optim_options)
    shard_optim_path = os.path.join(settings.local_checkpoint_dir, f"optimizer_step_{step:06d}_rank_{ddp_rank}.pt")
    logger.print_and_log(f"  ] [R{ddp_rank}] Saving shard of optimizer state to {shard_optim_path}", False)
    torch.save(optim_sd, shard_optim_path)

    torch.cuda.empty_cache()

    rng_and_loader_states = get_rank_rng_state()
    rng_and_loader_states["train_loader_state"] = train_loader.get_state()

    # logger.print_and_log(f"  ] [R{ddp_rank}] Saving DataLoader State. {train_loader.current_shard_info()}", False)
    # More readable multi-line version
    logger.print_and_log(f"  ] [R{ddp_rank}] Saving DataLoader State:", False)
    for group in train_loader.groups:
        if group.loaded_shard and group.is_active:
            shard_name = os.path.basename(group.loaded_shard.path)
            pos = group.loaded_shard.position
            total = len(group.loaded_shard.tokens)
            pct_used = (pos / total * 100) if total > 0 else 0
            logger.print_and_log(f"      {group.name:15s}: {shard_name:30s} @ {pct_used:5.1f}% used", False)

    filename = os.path.join(settings.local_checkpoint_dir, f"rng_state_step_{step:06d}_rank_{ddp_rank}.pt")
    # logger.print_and_log(f"  ] [R{ddp_rank}] Saving RNG state to {filename}", False)
    torch.save(rng_and_loader_states, filename)
    # logger.print_and_log(f"  ] [R{ddp_rank}] RNG state saved successfully", False)

    # Save AWD state (only rank 0 — it's identical across ranks)
    if awd is not None and ddp_rank == 0:
        awd_path = os.path.join(settings.local_checkpoint_dir, f"awd_state_step_{step:06d}.pt")
        torch.save(awd.state_dict(), awd_path)
        logger.print_and_log(f"  ] AWD state saved")

    # Save MoE expert_bias buffers explicitly (rank 0 only — identical across ranks).
    # FSDP2's get_model_state_dict may silently drop non-DTensor persistent buffers,
    # so we save them separately as a failsafe.
    if getattr(model_config, 'moe_enabled', False) and ddp_rank == 0:
        bias_dict = {}
        for layer in raw_model.layers:
            if getattr(layer, 'moe_enabled', False) and layer.moe.expert_bias is not None:
                bias_dict[f'layers.{layer.layer_id}.moe.expert_bias'] = layer.moe.expert_bias.cpu().clone()
        if bias_dict:
            bias_path = os.path.join(settings.local_checkpoint_dir, f"moe_bias_step_{step:06d}.pt")
            torch.save(bias_dict, bias_path)
            logger.print_and_log(f"  ] MoE expert_bias saved ({len(bias_dict)} layers)")

    logger.print_and_log(f"  ] [R{ddp_rank}] Save Complete", False)
    dist.barrier()

    # Start migration of checkpoint to the remote storage after successful save:
    # Run rsync once per node (using local_rank == 0 to identify one process per node)
    if ddp_local_rank == 0:
        trigger_checkpoint_sync(settings, ddp_rank, step)

# ---------------------------------------------------------------------------
# Checkpoint sync helper (also used as NAS recovery callback)
# ---------------------------------------------------------------------------
def trigger_checkpoint_sync(settings, ddp_rank, step=None):
    """Rsync local checkpoints to NAS. Called after saves and on NAS recovery."""
    try:
        os.makedirs(settings.nas_path, exist_ok=True)
        subprocess.Popen(
            ["nohup", "rsync", "-av", "--remove-source-files",
             f"{settings.local_checkpoint_dir}/", settings.nas_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        if step is not None:
            logger.print_and_log(f"  ] [R{ddp_rank}] Started checkpoint migration (step {step}) to NAS: {settings.nas_path}", False)
        else:
            logger.print_and_log(f"  ] [R{ddp_rank}] NAS recovery: retrying checkpoint sync to {settings.nas_path}", False)
    except Exception as e:
        logger.print_and_log(f"  ] [R{ddp_rank}] WARNING: checkpoint sync failed: {e}", False)

# Resume the state of our model and training environment for all of our processes
def resume_training(model, optimizer, train_loader, ddp_rank, settings, grad_accum_schedule, awd=None):
    logger.print_and_log(f"Resuming training from {settings.resume_checkpoint_path}:")

    torch.cuda.empty_cache()        # Clear out GPU memory to reduce fragmentation
    dist.barrier()                  # Ensure all processes wait until everything is saved

    # Get the underlying model (handle torch.compile wrapper)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # ---------------------------------------------------------------------------
    # STEP 1: LOAD AND DISTRIBUTE MODEL STATE
    # ---------------------------------------------------------------------------
    logger.print_and_log("  ] Loading model checkpoint...")
    checkpoint = torch.load(settings.resume_checkpoint_path, map_location='cpu')
    logger.print_and_log("  ] Torch.load complete.")

    # Extract important metadata before we start freeing memory
    state_dict = checkpoint["model"]
    shard_step = checkpoint["step"]
    total_tokens_processed = checkpoint["total_tokens_processed"]

    # Determine what optimizer the checkpoint was trained with
    checkpoint_optimizer_type = checkpoint.get('optimizer_type', None)
    if checkpoint_optimizer_type is None:
        # Legacy checkpoint: infer from use_adamc flag
        if checkpoint.get('use_adamc', False):
            checkpoint_optimizer_type = 'adamc'  # could have been 8bit, but close enough
        else:
            checkpoint_optimizer_type = 'adamw'

    # Fix old references to "hidden_dim" to "inner_dim"
    if "config" in checkpoint:
        config = checkpoint["config"]
        if "hidden_dim" in config and "inner_dim" not in config:
            config["inner_dim"] = config["hidden_dim"]
            del config["hidden_dim"]
            logger.print_and_log("  ] Converted legacy hidden_dim to inner_dim in checkpoint")

    # FSDP2: Use set_model_state_dict to load and distribute weights
    logger.print_and_log("  ] Distributing model weights across ranks...")
    # Detect new modules added since the checkpoint (e.g. auxiliary heads when
    # the intervention is being turned on at resume). Missing keys are tolerated
    # so freshly-initialized weights survive the load; unexpected keys would
    # still indicate a real config mismatch and we want to know about those.
    model_keys = set(raw_model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing_in_ckpt = sorted(model_keys - ckpt_keys)
    has_new_aux = any(k.startswith('aux_heads.') for k in missing_in_ckpt)
    use_strict = not has_new_aux
    if has_new_aux:
        n_aux = sum(1 for k in missing_in_ckpt if k.startswith('aux_heads.'))
        logger.print_and_log(
            f"  ] Detected {n_aux} new aux-head param(s) absent from checkpoint — "
            f"loading non-strict so they keep their fresh init."
        )
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=use_strict)
    set_model_state_dict(raw_model, state_dict, options=options)

    # Important: Free up memory after model is distributed to GPUs
    del state_dict
    del checkpoint
    torch.cuda.empty_cache()

    dist.barrier()  # Ensure all processes have completed model loading

    # ---------------------------------------------------------------------------
    # STEP 2: RESTORE RNG AND DATALOADER STATE
    # ---------------------------------------------------------------------------
    # These are small and won't cause memory problems
    pth = os.path.dirname(settings.resume_checkpoint_path)
    filename = os.path.join(pth, f"rng_state_step_{shard_step:06d}_rank_{ddp_rank}.pt")

    # Detect world_size mismatch: if this rank's RNG file doesn't exist, the checkpoint
    # was saved with fewer ranks. RNG and dataloader state are rank-specific and not
    # meaningful across topology changes, so skip restoration entirely.
    if os.path.exists(filename):
        logger.print_and_log("  ] Setting random seeds and restoring DataLoader state...")
        # weights_only=False needed because RNG states contain NumPy arrays
        rng_states = torch.load(filename, weights_only=False)

        # Set random states
        random.setstate(rng_states["random_state"])
        np.random.set_state(rng_states["numpy_state"])
        torch.set_rng_state(torch.ByteTensor(rng_states["rng_state"]))
        if "cuda_rng_state" in rng_states:
            torch.cuda.set_rng_state(torch.ByteTensor(rng_states["cuda_rng_state"]))

        # Restore DataLoader state
        train_loader_state = rng_states["train_loader_state"]
        train_loader.set_state(train_loader_state)
        if hasattr(settings, 'resume_data_reset'):
            reset_mode = settings.resume_data_reset
        else:
            reset_mode = "continue"
        train_loader.reset(reset_mode)
        logger.print_and_log(f"  ] [R{ddp_rank}] DataLoader State Restored: {train_loader.current_shard_info()} [reset mode: {reset_mode}]", False)

        del rng_states
        del train_loader_state
    else:
        logger.print_and_log(
            f"  ] RNG state file not found for rank {ddp_rank} — "
            f"topology change detected (checkpoint has fewer ranks). "
            f"Skipping RNG/DataLoader restore; dataloader will start from fresh shards."
        )

    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # STEP 3: LOAD OPTIMIZER STATE (sharded or full)
    # ---------------------------------------------------------------------------
    # Check for a consolidated full optimizer state dict first (from consolidate_optimizer.py).
    # This enables resuming on a different world_size than the checkpoint was saved with.
    optim_full_path = os.path.join(pth, f"optimizer_step_{shard_step:06d}_full.pt")
    optim_shard_path = os.path.join(pth, f"optimizer_step_{shard_step:06d}_rank_{ddp_rank}.pt")
    use_full_optim = os.path.exists(optim_full_path)

    if use_full_optim:
        logger.print_and_log(f"  ] Found consolidated optimizer state: {os.path.basename(optim_full_path)}")
    else:
        logger.print_and_log("  ] Restoring optimizer shards ...")
        logger.print_and_log(f"  ] [R{ddp_rank}] loading optimiser shard {optim_shard_path}", False)

    if settings.optimizer_type != checkpoint_optimizer_type:
        logger.print_and_log(f"  ] [R{ddp_rank}] WARNING: Optimizer changed: {checkpoint_optimizer_type} → {settings.optimizer_type}")
        logger.print_and_log(f"  ] [R{ddp_rank}] Skipping optimizer state - starting with fresh optimizer state")
    else:
        load_optim = getattr(optimizer, '_base_optimizer', optimizer)

        # Save param_group keys before restore — set_optimizer_state_dict may strip
        # required keys (lr, weight_decay, etc.) that TorchAO needs at step() time
        pg_defaults = [{k: v for k, v in pg.items() if k != 'params'} for pg in load_optim.param_groups]

        def _inject_empty_state_for_new_params(saved_sd: dict, log_prefix: str = ""):
            """Inject empty optimizer state for params present in the current
            optimizer but absent from the saved state dict (e.g., aux-head
            params added at resume). Optimizers initialize per-param state
            lazily on the first step(), so an empty dict is the correct
            starting point."""
            model_fqns = set(name for name, _ in raw_model.named_parameters())
            state_dict = saved_sd.get('state', {})
            saved_fqns = set(state_dict.keys())
            missing = model_fqns - saved_fqns
            if not missing:
                return
            aux_missing = sorted(fqn for fqn in missing if fqn.startswith('aux_heads.'))
            other_missing = sorted(missing - set(aux_missing))
            if other_missing:
                preview = ", ".join(other_missing[:5])
                ellipsis = ", ..." if len(other_missing) > 5 else ""
                logger.print_and_log(
                    f"  ] {log_prefix}WARNING: optimizer state missing "
                    f"{len(other_missing)} non-aux-head param(s): {preview}{ellipsis}"
                )
            if aux_missing:
                logger.print_and_log(
                    f"  ] {log_prefix}Injecting empty optimizer state for "
                    f"{len(aux_missing)} new aux-head param(s) (lazy init on first step)"
                )
            for fqn in missing:
                state_dict[fqn] = {}
            saved_sd['state'] = state_dict

        if use_full_optim:
            # Full state dict — all ranks load the same file, FSDP distributes automatically.
            # Strip param_group metadata: the current optimizer already has the correct
            # param_groups. Only the state tensors (momentum, exp_avg, etc.) need
            # redistribution. Custom param_group keys (MuonFSDP2's momentum/beta2 vs
            # Adam's betas) cause KeyErrors in _unflatten_state_dict if left in.
            optim_sd = torch.load(optim_full_path, map_location="cpu")
            # Sync param_group metadata with the current optimizer.  The saved dict's
            # param_groups may have extra keys injected during consolidation (e.g.
            # 'betas' on Muon groups) that cause KeyErrors in _unflatten_state_dict.
            # Keep each group's 'params' (FQN list — needed for state→param mapping)
            # but replace all other keys with the current optimizer's values.
            assert len(optim_sd['param_groups']) == len(load_optim.param_groups), (
                f"Param group count mismatch: saved={len(optim_sd['param_groups'])}, "
                f"current={len(load_optim.param_groups)}"
            )
            for saved_pg, current_pg in zip(optim_sd['param_groups'], load_optim.param_groups):
                saved_params = saved_pg['params']
                saved_pg.clear()
                saved_pg['params'] = saved_params
                for k, v in current_pg.items():
                    if k != 'params':
                        saved_pg[k] = v
            _inject_empty_state_for_new_params(optim_sd)
            optim_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            set_optimizer_state_dict(raw_model, load_optim, optim_sd, options=optim_options)
            del optim_sd
            logger.print_and_log("  ] Full optimizer state distributed across ranks.")
        else:
            # Sharded state dict — each rank loads its own shard (requires same world_size)
            optim_shard = torch.load(optim_shard_path, map_location="cpu")
            _inject_empty_state_for_new_params(optim_shard, log_prefix=f"[R{ddp_rank}] ")
            optim_options = StateDictOptions(full_state_dict=False)
            set_optimizer_state_dict(raw_model, load_optim, optim_shard, options=optim_options)
            del optim_shard

        # Re-add any param_group keys that were stripped during restore
        for pg, defaults in zip(load_optim.param_groups, pg_defaults):
            for k, v in defaults.items():
                if k not in pg:
                    pg[k] = v

        # Update weight_decay per group if changed (scalar mode only; rules mode uses _wd per step)
        if isinstance(settings.weight_decay, (int, float)):
            for param_group in optimizer.param_groups:
                if param_group.get('wd_group') == 'norm_bias':
                    continue
                old_wd = param_group.get('weight_decay', 0.0)
                if old_wd != settings.weight_decay:
                    param_group['weight_decay'] = settings.weight_decay
                    logger.print_and_log(f"  ] Updated weight_decay: {old_wd} \u2192 {settings.weight_decay}", r0_only=True)
                    break  # all non-norm groups share the same scalar

        # Also update betas if they've changed
        for param_group in optimizer.param_groups:
            param_group['betas'] = (settings.beta1, settings.beta2)

    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # STEP 3b: RESTORE AWD STATE (if available)
    # ---------------------------------------------------------------------------
    if awd is not None:
        awd_path = os.path.join(pth, f"awd_state_step_{shard_step:06d}.pt")
        if os.path.exists(awd_path):
            awd.load_state_dict(torch.load(awd_path, weights_only=True))
            logger.print_and_log(f"  ] AWD state restored", r0_only=True)
        else:
            logger.print_and_log(f"  ] AWD state not found — starting fresh", r0_only=True)

    # ---------------------------------------------------------------------------
    # STEP 3c: RESTORE MoE expert_bias (if available)
    # ---------------------------------------------------------------------------
    # FSDP2's set_model_state_dict may not restore non-DTensor persistent buffers
    # correctly. Load the explicitly saved expert_bias and verify/overwrite.
    if getattr(raw_model, 'layers', None) is not None:
        bias_path = os.path.join(pth, f"moe_bias_step_{shard_step:06d}.pt")
        if os.path.exists(bias_path):
            bias_dict = torch.load(bias_path, weights_only=True)
            restored = 0
            for layer in raw_model.layers:
                if getattr(layer, 'moe_enabled', False) and layer.moe.expert_bias is not None:
                    key = f'layers.{layer.layer_id}.moe.expert_bias'
                    if key in bias_dict:
                        saved_bias = bias_dict[key]
                        current = layer.moe.expert_bias
                        if not torch.allclose(current.cpu().float(), saved_bias.float(), atol=1e-6):
                            logger.print_and_log(
                                f"  ] WARNING: expert_bias L{layer.layer_id} was NOT correctly "
                                f"restored by set_model_state_dict (was zeros={current.abs().sum().item() < 1e-8}, "
                                f"saved range=[{saved_bias.min():.4f}, {saved_bias.max():.4f}])"
                            )
                        layer.moe.expert_bias.copy_(saved_bias.to(layer.moe.expert_bias.device))
                        restored += 1
            logger.print_and_log(f"  ] MoE expert_bias restored ({restored} layers)", r0_only=True)
        else:
            # No explicit bias file — check if set_model_state_dict handled it
            has_moe = False
            for layer in raw_model.layers:
                if getattr(layer, 'moe_enabled', False) and layer.moe.expert_bias is not None:
                    has_moe = True
                    bias = layer.moe.expert_bias
                    if bias.abs().sum().item() < 1e-8:
                        logger.print_and_log(
                            f"  ] WARNING: expert_bias L{layer.layer_id} is all zeros after resume "
                            f"(no moe_bias file found — older checkpoint?)"
                        )
            if has_moe:
                logger.print_and_log(f"  ] MoE expert_bias: relying on set_model_state_dict (no explicit save found)", r0_only=True)

    # ---------------------------------------------------------------------------
    # STEP 3d: RESTORE EP expert weights (if available)
    # ---------------------------------------------------------------------------
    # With Expert Parallel, the main checkpoint only has rank 0's local experts.
    # set_model_state_dict gives ALL ranks rank 0's expert weights (wrong for ranks 1+).
    # The consolidated ep_experts file has all experts — slice per-rank and overwrite.
    ep_experts_path = os.path.join(pth, f"ep_experts_step_{shard_step:06d}.pt")
    has_ep = any(
        getattr(layer, 'moe_enabled', False) and layer.moe.ep_degree > 1
        for layer in getattr(raw_model, 'layers', [])
    )
    if os.path.exists(ep_experts_path):
        # Guard: _local_tensor overwrite only works when edp_mesh=1
        if has_ep:
            first_moe = next(l for l in raw_model.layers if getattr(l, 'moe_enabled', False))
            edp_size = ddp_world_size // first_moe.moe.ep_degree
            assert edp_size == 1, (
                f"EP expert restore requires edp_mesh size 1 (ep_degree == world_size). "
                f"Got edp_size={edp_size}. Not yet supported."
            )

        ep_experts = torch.load(ep_experts_path, map_location='cpu', weights_only=True)
        restored_count = 0
        ep_local_rank = 0
        num_local = 1
        for layer in raw_model.layers:
            if not getattr(layer, 'moe_enabled', False):
                continue
            num_local = layer.moe.num_local_experts
            ep_deg = layer.moe.ep_degree
            ep_local_rank = ddp_rank % ep_deg
            expected_experts = ep_deg * num_local
            start = ep_local_rank * num_local
            end = start + num_local

            for pname in ('w1', 'w2', 'w3'):
                key = f'layers.{layer.layer_id}.moe.experts.{pname}'
                if key in ep_experts:
                    full_param = ep_experts[key]  # [num_experts, ...]
                    assert full_param.shape[0] == expected_experts, (
                        f"EP expert shape mismatch for {key}: file has {full_param.shape[0]} experts, "
                        f"expected {expected_experts} (ep_degree={ep_deg} × num_local={num_local})"
                    )
                    my_slice = full_param[start:end]  # [num_local, ...]
                    # Overwrite the DTensor's underlying local data
                    target = getattr(layer.moe.experts, pname)
                    local_data = target._local_tensor if hasattr(target, '_local_tensor') else target.data
                    local_data.copy_(my_slice.to(local_data.device))
                    restored_count += 1

        del ep_experts
        torch.cuda.empty_cache()
        logger.print_and_log(f"  ] EP expert weights restored ({restored_count} params, rank {ddp_rank} got experts [{ep_local_rank * num_local}:{(ep_local_rank + 1) * num_local}])")
    elif has_ep:
        # Missing consolidated file for an EP model — hard error, not a warning.
        # Without this file, all ranks have rank 0's experts (silent corruption).
        raise FileNotFoundError(
            f"EP expert weights file not found: ep_experts_step_{shard_step:06d}.pt\n"
            f"All ranks would get rank 0's experts — training would diverge.\n"
            f"Re-save checkpoint with updated save_model to fix."
        )

    # ---------------------------------------------------------------------------
    # STEP 4: Set next training step and resume confirmation
    # ---------------------------------------------------------------------------
    start_step = shard_step + 1
    lrc = get_lr(start_step, settings)
    gac = grad_accum_schedule[start_step]
    logger.print_and_log(f"  ] Resumed @ step {start_step}:  LR: {lrc:.6e} Grad Accum Steps: {gac}")

    dist.barrier()  # Final sync before returning

    return start_step, total_tokens_processed

def setup_ddp(ep_degree=1):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        timeout = timedelta(minutes=30)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)

        init_process_group(backend='nccl', timeout=timeout, device_id=torch.device(device))

        if ep_degree > 1:
            assert ddp_world_size % ep_degree == 0, (
                f"world_size ({ddp_world_size}) must be divisible by ep_degree ({ep_degree})"
            )
            efsdp_size = ddp_world_size // ep_degree

            # 2D mesh for expert parallel: (efsdp, ep)
            mesh_2d = init_device_mesh("cuda", (efsdp_size, ep_degree), mesh_dim_names=("fsdp", "ep"))
            ep_mesh = mesh_2d["ep"]       # 1D sub-mesh for all-to-all
            edp_mesh = mesh_2d["fsdp"]    # 1D sub-mesh for expert FSDP

            # 1D mesh for dense FSDP (all GPUs)
            dp_mesh = init_device_mesh("cuda", (ddp_world_size,))
        else:
            # Standard 1D mesh — no EP
            dp_mesh = init_device_mesh("cuda", (ddp_world_size,))
            ep_mesh = None
            edp_mesh = None
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        dp_mesh = None  # No mesh needed for single-GPU
        ep_mesh = None
        edp_mesh = None
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, dp_mesh, ep_mesh, edp_mesh

def suggest_fsdp_dimensions(target_embd: int, target_intermediate: int, world_size: int, max_options: int = 3):
    """
    Suggest good model dimensions that are compatible with FSDP sharding.

    Requirements:
    - cfg_embd and cfg_intermediate must be divisible by world_size
    - cfg_embd should be divisible by 64 (tensor core efficiency)
    - cfg_embd = cfg_heads × head_dim (typically head_dim = 64 or 128)
    - cfg_heads must be divisible by cfg_kv_heads (for GQA)

    Returns a list of dicts with suggested configurations.
    """
    import math

    # LCM of world_size and 64 for optimal tensor dimensions
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    embd_lcm = lcm(world_size, 64)
    intermediate_lcm = lcm(world_size, 64)

    suggestions = []

    # Find cfg_embd values near target that are multiples of embd_lcm
    for multiplier in range(max(1, (target_embd - embd_lcm * 2) // embd_lcm),
                           (target_embd + embd_lcm * 3) // embd_lcm + 1):
        embd = multiplier * embd_lcm
        if embd < 512:  # Skip unreasonably small values
            continue

        # Try head_dim = 64 first (most common)
        for head_dim in [64, 128]:
            if embd % head_dim != 0:
                continue

            heads = embd // head_dim

            # Find valid kv_heads (must divide heads, and ideally divide world_size for efficiency)
            valid_kv_heads = []
            for kv in [world_size, world_size // 2 if world_size % 2 == 0 else None,
                       heads // 4, heads // 5, heads // 6, heads // 7, heads // 8]:
                if kv is not None and kv > 0 and heads % kv == 0 and kv <= heads:
                    if kv not in valid_kv_heads:
                        valid_kv_heads.append(kv)

            if not valid_kv_heads:
                continue

            # Pick best kv_heads (prefer world_size for FSDP efficiency, then reasonable ratios)
            kv_heads = valid_kv_heads[0]
            gqa_ratio = heads // kv_heads

            # Find cfg_intermediate near target
            int_multiplier = round(target_intermediate / intermediate_lcm)
            intermediate = int_multiplier * intermediate_lcm

            # Calculate size difference from target
            embd_diff = abs(embd - target_embd) / target_embd * 100
            int_diff = abs(intermediate - target_intermediate) / target_intermediate * 100 if target_intermediate > 0 else 0

            suggestions.append({
                'embd': embd,
                'heads': heads,
                'kv_heads': kv_heads,
                'head_dim': head_dim,
                'gqa_ratio': f"{gqa_ratio}:1",
                'intermediate': intermediate,
                'embd_diff': embd_diff,
                'total_diff': embd_diff + int_diff,
            })

    # Sort by total difference from target and remove duplicates
    suggestions.sort(key=lambda x: (x['total_diff'], x['embd']))
    seen = set()
    unique = []
    for s in suggestions:
        key = (s['embd'], s['heads'], s['kv_heads'])
        if key not in seen:
            seen.add(key)
            unique.append(s)
            if len(unique) >= max_options:
                break

    return unique


def check_params(cfg, settings=None, world_size=1):
    """
    Checks for common parameter mismatches in a ModelArgs-like config object:
      - Ensures that cfg.dim is divisible by cfg.n_heads
      - Checks n_kv_heads <= n_heads if needed
      - If using DION Muon optimizer, ensures dimensions are divisible by world_size

    Exits gracefully with a helpful message if any check fails.
    """
    # 1. Check that cfg.dim is divisible by cfg.n_heads
    if cfg.dim % cfg.n_heads != 0:
        ratio = cfg.dim / cfg.n_heads
        fatal_error(
            f"Parameter mismatch: The embedding dimension 'dim' ({cfg.dim}) must be divisible by "
            f"the number of attention heads 'n_heads' ({cfg.n_heads}).\n"
            f"Currently, {cfg.dim} / {cfg.n_heads} = {ratio:.2f}, which is not an integer.\n\n"
            "How to fix:\n"
            " - Pick 'dim' such that dim % n_heads == 0, e.g. if you keep n_heads=14, try dim=1008 or 1120.\n"
            " - Or pick 'n_heads' that divides your 'dim', e.g. if you keep dim=1024, try n_heads=8 or 16."
        )

    # 2. (Optional) If you're using separate n_kv_heads, check it
    if cfg.n_kv_heads is not None:
        if cfg.n_kv_heads > cfg.n_heads:
            fatal_error(
                f"Parameter mismatch: 'n_kv_heads' ({cfg.n_kv_heads}) cannot exceed 'n_heads' ({cfg.n_heads}).\n"
                "How to fix:\n"
                " - Either remove n_kv_heads or ensure n_kv_heads <= n_heads."
            )

    # 3. If using DION Muon optimizer with multi-GPU, check dimension divisibility
    if settings is not None and getattr(settings, 'optimizer_type', 'adamw') in DION_FAMILY and world_size > 1:
        # DION Muon uses all-to-all communication which requires dimensions divisible by world_size
        errors = []

        # Check embedding dimension
        if cfg.dim % world_size != 0:
            errors.append(f"- cfg_embd ({cfg.dim}) is not divisible by world_size ({world_size})")

        # Check intermediate/FFN dimension
        if cfg.inner_dim is not None and cfg.inner_dim % world_size != 0:
            errors.append(f"- cfg_intermediate ({cfg.inner_dim}) is not divisible by world_size ({world_size})")

        if errors:
            # Generate smart suggestions
            target_intermediate = cfg.inner_dim if cfg.inner_dim is not None else cfg.dim * 4
            suggestions = suggest_fsdp_dimensions(cfg.dim, target_intermediate, world_size, max_options=3)

            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n\nSuggested configurations (divisible by world_size AND 64 for efficiency):\n"
                suggestion_text += "-" * 70 + "\n"
                for i, s in enumerate(suggestions, 1):
                    diff_note = f"({s['embd_diff']:.1f}% from target)" if s['embd_diff'] > 0.1 else "(closest match)"
                    suggestion_text += f"  Option {i}: {diff_note}\n"
                    suggestion_text += f"    cfg_embd: {s['embd']:<6}  cfg_heads: {s['heads']:<3}  cfg_kv_heads: {s['kv_heads']:<2}  (head_dim={s['head_dim']}, GQA {s['gqa_ratio']})\n"
                    suggestion_text += f"    cfg_intermediate: {s['intermediate']}\n"
                suggestion_text += "-" * 70

            fatal_error(
                f"DION Muon optimizer requires model dimensions to be divisible by world_size ({world_size}).\n"
                f"The following dimensions are incompatible:\n" +
                "\n".join(errors) +
                suggestion_text + "\n\n"
                "Why these constraints?\n"
                f" - Divisible by {world_size}: Required for FSDP weight sharding across GPUs\n"
                " - Divisible by 64: Optimal for tensor core operations (performance)\n"
                " - cfg_heads divisible by cfg_kv_heads: Required for GQA (grouped query attention)"
            )

    # 4. EP / MoE validation
    ep_degree = getattr(cfg, 'ep_degree', 1)
    if ep_degree > 1:
        if not getattr(cfg, 'moe_enabled', False):
            fatal_error(
                f"ep_degree={ep_degree} but moe_enabled is not set.\n"
                "Expert Parallel only makes sense with MoE layers.\n\n"
                "How to fix:\n"
                " - Set moe_enabled: true, or remove ep_degree from config"
            )
        if cfg.moe_num_experts % ep_degree != 0:
            fatal_error(
                f"Expert Parallel: num_experts ({cfg.moe_num_experts}) must be divisible by "
                f"ep_degree ({ep_degree}).\n"
                f"Currently: {cfg.moe_num_experts} % {ep_degree} = {cfg.moe_num_experts % ep_degree}\n\n"
                "How to fix:\n"
                f" - Set moe_num_experts to a multiple of {ep_degree} (e.g. {ep_degree})"
            )
        if ep_degree > cfg.moe_num_experts:
            fatal_error(
                f"ep_degree ({ep_degree}) exceeds moe_num_experts ({cfg.moe_num_experts}).\n"
                "Each EP rank must own at least 1 expert.\n\n"
                "How to fix:\n"
                f" - Set ep_degree <= {cfg.moe_num_experts}"
            )

    return

def summarize_model(model: Transformer) -> dict:
    """
    Returns a dictionary containing useful information
    about the structure and dimensions of the model.
    """
    cfg = model.params
    # With EP, expert params are local (num_local_experts) — compute the global
    # total by adding back the missing (ep_degree - 1) copies of expert weights.
    local_params = sum(p.numel() for p in model.parameters())
    ep_expert_extra = 0
    ep_degree = getattr(cfg, 'ep_degree', 1)
    if ep_degree > 1 and hasattr(model, 'layers'):
        for layer in model.layers:
            if getattr(layer, 'moe_enabled', False):
                experts = layer.moe.experts
                expert_numel = sum(p.numel() for p in [experts.w1, experts.w2, experts.w3])
                ep_expert_extra += expert_numel * (ep_degree - 1)

    summary = {
        "vocab_size": cfg.vocab_size,
        "num_layers": cfg.n_layers,
        "model_dim": cfg.dim,
        "num_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
#        "multiple_of": cfg.multiple_of,
        "norm_eps": cfg.norm_eps,
        "max_seq_len": cfg.max_seq_len,
        "dropout": cfg.dropout,
        "layer_info": [],
        "total_params": local_params + ep_expert_extra,
        "local_params": local_params if ep_degree > 1 else None,
    }

    # Loop each TransformerBlock to extract layer-specific info
    for i, block in enumerate(model.layers):
        is_moe = getattr(block, 'moe_enabled', False)
        is_gdn = getattr(block, 'use_gdn', False)

        # Extract attention head info — GDN layers use gdn_attn, not attention
        if is_gdn:
            attn_heads = block.gdn_attn.num_heads
            attn_head_dim = block.gdn_attn.head_dim
        else:
            attn_heads = block.attention.n_local_heads
            attn_head_dim = block.attention.head_dim

        if is_moe:
            # MoE layer: expert weights are 3D nn.Parameter (num_local_experts, hidden_dim, dim)
            expert_inner_dim = block.moe.experts.w1.shape[1]
            num_experts_global = block.moe.num_experts  # global count (not local)
            num_local_experts = block.moe.num_local_experts
            top_k = block.moe.router.top_k
            shared = block.moe.shared_experts is not None
            layer_info = {
                "layer_id": i,
                "attention_norm_dim": block.attention_norm.weight.shape[0],
                "ffn_norm_dim": block.ffn_norm.weight.shape[0],
                "num_heads": attn_heads,
                "head_dim": attn_head_dim,
                "feedforward_intermediate_dim": expert_inner_dim,
                "moe": True,
                "gdn": is_gdn,
                "num_experts": num_experts_global,
                "num_local_experts": num_local_experts,
                "top_k": top_k,
                "shared_experts": shared,
            }
        else:
            # Dense layer: w1.weight.shape[0] is the intermediate dim
            ff_intermediate_dim = block.feed_forward.w1.weight.shape[0]
            layer_info = {
                "layer_id": i,
                "attention_norm_dim": block.attention_norm.weight.shape[0],
                "ffn_norm_dim": block.ffn_norm.weight.shape[0],
                "num_heads": attn_heads,
                "head_dim": attn_head_dim,
                "feedforward_intermediate_dim": ff_intermediate_dim,
                "moe": False,
                "gdn": is_gdn,
            }
        summary["layer_info"].append(layer_info)

    # Use first dense layer's intermediate dim for top-level summary; fall back to first layer
    dense_layers = [li for li in summary["layer_info"] if not li.get("moe")]
    if dense_layers:
        feedforward_dim = dense_layers[0]["feedforward_intermediate_dim"]
    elif summary["layer_info"]:
        feedforward_dim = summary["layer_info"][0]["feedforward_intermediate_dim"]
    else:
        feedforward_dim = "N/A"
    summary["intermediate_dim"] = feedforward_dim

    return summary


def _apply_per_submodule_compile(model, compile_mode, logger):
    """Apply torch.compile per-submodule to each TransformerBlock.

    All layers are compiled per-submodule (not whole-block) because:
    - Inline activation checkpointing (checkpoint() in forward) causes
      dynamo guard invalidation / recompilation when whole-block compiled
    - Per-submodule is also better for tail truncation — each submodule's
      graph is constant regardless of how many layers run

    For MoE layers: experts, router, and shared_experts are each compiled
    individually. FSDP2 hooks fire in __call__ outside the compiled forward.
    """
    import torch._dynamo

    # Each layer has distinct compiled submodules. Dynamo's cache needs one
    # entry per unique module at each call site. Default limit (8) is too
    # small when num_layers > 8, causing spurious recompile warnings.
    torch._dynamo.config.cache_size_limit = max(16, len(model.layers) + 4)

    has_moe = any(getattr(l, 'moe_enabled', False) for l in model.layers)
    if has_moe:
        # Required for dynamic token routing shapes (histc output → split sizes)
        torch._dynamo.config.capture_scalar_outputs = True

    n_dense = 0
    n_moe = 0
    n_gdn = 0

    for layer in model.layers:
        is_gdn = getattr(layer, 'use_gdn', False)
        is_moe = getattr(layer, 'moe_enabled', False)

        for attr_name in list(layer._modules.keys()):
            submod = layer._modules[attr_name]
            if attr_name == 'moe' and is_moe:
                # Inside MoE: compile each child (experts, router, shared_experts)
                for moe_attr in list(submod._modules.keys()):
                    moe_child = submod._modules[moe_attr]
                    if moe_child is not None:
                        setattr(submod, moe_attr,
                                torch.compile(moe_child, mode=compile_mode))
            else:
                # GDN attn, softmax attention, norms, dense FFN — all compile fine
                setattr(layer, attr_name,
                        torch.compile(submod, mode=compile_mode))

        if is_gdn:
            n_gdn += 1
        if is_moe:
            n_moe += 1
        if not is_moe:
            n_dense += 1

    n_attn = len(model.layers) - n_gdn  # softmax attention layers
    logger.print_and_log(f"Compiling model (per-submodule, mode={compile_mode})...")
    parts = []
    if n_gdn:
        parts.append(f"{n_gdn} GDN")
    if n_attn:
        parts.append(f"{n_attn} attn")
    if n_moe:
        parts.append(f"{n_moe} MoE")
    if n_dense:
        parts.append(f"{n_dense} dense-FFN")
    logger.print_and_log(f"  ] Layers: {' + '.join(parts)} (all per-submodule)")


def create_and_shard_model(model_cfg, dp_mesh, ep_mesh, edp_mesh, device, settings, logger):
    """
    Create model on meta device, apply FSDP2 sharding, materialize, and initialize weights.

    This is the recommended FSDP2 workflow per torchtitan docs:
    1. Create model on meta device (instant, no memory)
    2. Shard FIRST while still on meta (no memory allocated yet)
    3. Materialize - each GPU only allocates its shard
    4. Initialize weights with synchronized RNG

    With Expert Parallel (ep_degree > 1):
    - Expert weights get inner FSDP on edp_mesh (prevents outer dp_mesh from touching them)
    - Dense weights (attention, norms, shared_experts, router) get outer FSDP on dp_mesh
    - EP mesh attached to MoE modules for all-to-all dispatch/combine

    Args:
        model_cfg: ModelArgs configuration
        dp_mesh: Device mesh for dense FSDP (all GPUs)
        ep_mesh: EP sub-mesh for all-to-all (None if no EP)
        edp_mesh: Expert FSDP sub-mesh (None if no EP)
        device: Target device (e.g., 'cuda:0')
        settings: Training settings with dtype config
        logger: Logger for status messages

    Returns:
        Sharded, materialized, initialized model
    """
    logger.print_and_log(f"Creating model...")

    # Create on meta device (instant, no memory)
    logger.print_and_log(f"  ] Creating on meta device (instant)")
    with torch.device('meta'):
        model = Transformer(model_cfg)

    # Attach EP mesh to MoE modules (before FSDP wrapping)
    if ep_mesh is not None:
        for layer in model.layers:
            if getattr(layer, 'moe_enabled', False):
                layer.moe.set_ep_mesh(ep_mesh)

    # Determine target dtype
    target_dtype = (
        torch.bfloat16 if settings.FSDP_param_dtype == 'bf16'
        else torch.float16 if settings.FSDP_param_dtype == 'fp16'
        else torch.float32
    )

    # Create mixed precision policy
    reduce_dtype = (
        torch.bfloat16 if settings.FSDP_reduce_dtype == 'bf16'
        else torch.float16 if settings.FSDP_reduce_dtype == 'fp16'
        else torch.float32
    )
    try:
        # PyTorch 2.4+: output_dtype ensures FSDP module outputs match param_dtype
        mp_policy = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=target_dtype,
        )
    except TypeError:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=reduce_dtype,
        )

    reshard_after_forward = getattr(settings, 'reshard_after_forward', True)
    cpu_offload = getattr(settings, 'cpu_offload', False)
    offload_policy = CPUOffloadPolicy(pin_memory=True) if cpu_offload else None
    logger.print_and_log(f"  ] Applying FSDP2 sharding (reshard_after_forward={reshard_after_forward})")
    if cpu_offload:
        logger.print_and_log(f"  ] CPU offload enabled (pin_memory=True)")

    # Two-level FSDP wrapping: inner experts on edp_mesh, outer layers on dp_mesh
    for layer in model.layers:
        # Inner: expert weights on edp_mesh (prevents outer dp_mesh from touching them)
        if ep_mesh is not None and getattr(layer, 'moe_enabled', False):
            fully_shard(layer.moe.experts, mesh=edp_mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, offload_policy=offload_policy)
        # Outer: entire layer on dp_mesh (attention, norms, shared_experts, router gate)
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, offload_policy=offload_policy)
    # Auxiliary prediction heads each get their own FSDP wrap. They fire only
    # at their tap-layer depth, so an independent unshard/reshard cycle keeps
    # them off the all-gather schedule for the rest of the body forward pass.
    if getattr(model, 'aux_heads', None) is not None and len(model.aux_heads) > 0:
        for aux_head in model.aux_heads.values():
            fully_shard(aux_head, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, offload_policy=offload_policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False, offload_policy=offload_policy)

    # Materialize: CPU offload requires params on CPU; FSDP handles H2D transfers
    mat_device = "cpu" if cpu_offload else device
    logger.print_and_log(f"  ] Materializing to {mat_device} in {target_dtype}")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    model = model.to_empty(device=mat_device)
    torch.set_default_dtype(old_dtype)

    # CPU offload: FSDP manages params (H2D on demand) but buffers stay on CPU.
    # Move all buffers to GPU — they're small (RoPE freqs, expert_bias, etc.)
    if cpu_offload:
        n_moved = 0
        for name, buf in model.named_buffers():
            if buf is not None and buf.device.type == "cpu":
                # Walk the module hierarchy to set the buffer on the owning module
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], buf.to(device))
                n_moved += 1
        logger.print_and_log(f"  ] Moved {n_moved} buffers to {device}")

    # Initialize weights with synchronized RNG across all ranks
    logger.print_and_log(f"  ] Initializing weights (synchronized seed=42)")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if hasattr(model, 'init_weights'):
        model.init_weights()
    else:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    logger.print_and_log(f"  ] Model ready")
    return model


def _compute_active_params(model, summary, moe_layer_info, logger):
    """Compute and log active params per token for MoE models.

    In MoE, only top_k out of num_experts fire per token, so the
    "active" param count is smaller than the total. This metric
    shows the effective model size a single token sees each forward pass.
    """
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    top_k = moe_layer_info['top_k']
    num_experts = moe_layer_info['num_experts']

    # Non-expert params: embeddings, norms, attention, router gate, shared experts, output head
    # Expert params: w1/w2/w3 3D tensors (scaled by top_k/num_experts)
    active = 0
    for layer in raw.layers:
        if getattr(layer, 'moe_enabled', False):
            moe = layer.moe
            ep_degree = getattr(moe, 'ep_degree', 1)
            # Expert params: only top_k/num_experts fraction active per token
            expert_numel = sum(p.numel() for p in [moe.experts.w1, moe.experts.w2, moe.experts.w3])
            expert_numel *= ep_degree  # local → global
            active += expert_numel * top_k / num_experts
            # Everything else in the MoE module is always active (router, shared experts)
            for p in moe.parameters():
                if p is not moe.experts.w1 and p is not moe.experts.w2 and p is not moe.experts.w3:
                    active += p.numel()
            # Attention + norms (always active)
            attn_mod = layer.gdn_attn if getattr(layer, 'use_gdn', False) else layer.attention
            active += sum(p.numel() for p in attn_mod.parameters())
            active += layer.attention_norm.weight.numel() + layer.ffn_norm.weight.numel()
        else:
            active += sum(p.numel() for p in layer.parameters())

    # Embeddings + output head + final norm
    active += raw.tok_embeddings.weight.numel()
    if hasattr(raw, 'output') and raw.output is not None:
        if raw.output.weight is not raw.tok_embeddings.weight:
            active += raw.output.weight.numel()
    if hasattr(raw, 'norm'):
        active += raw.norm.weight.numel()

    active = int(active)
    total = summary['total_params']
    pct = active / total * 100 if total > 0 else 0
    logger.print_and_log(f"  ] ACTIVE/TOKEN = {active:,} ({pct:.1f}% of total)")


def print_model_summary(model, model_cfg, settings, logger):
    """Print model architecture summary."""
    summary = summarize_model(model)
    logger.print_and_log(f"Model Summary:")
    logger.print_and_log(f"  ] TOTAL PARAMS = {summary['total_params']:,}")
    if summary.get('local_params') is not None:
        logger.print_and_log(f"  ] PER-RANK     = {summary['local_params']:,} (EP distributes experts)")
    logger.print_and_log(f"  ] VOCAB        = {summary['vocab_size']:,}")
    logger.print_and_log(f"  ] LAYERS       = {summary['num_layers']:,}")
    logger.print_and_log(f"  ] DIM          = {summary['model_dim']:,}")
    logger.print_and_log(f"  ] HEADS        = {summary['num_heads']:,}")
    if summary['n_kv_heads'] is not None and summary['n_kv_heads'] != summary['num_heads']:
        logger.print_and_log(f"  ] KV HEADS     = {summary['n_kv_heads']:,} (GQA)")
    logger.print_and_log(f"  ] CONTEXT      = {summary['max_seq_len']:,}")
    logger.print_and_log(f"  ] INTERMEDIATE = {summary['intermediate_dim']:,}")
    logger.print_and_log(f"  ] NORM EPS     = {summary['norm_eps']}")
    logger.print_and_log(f"  ] QK NORM MODE = {settings.qk_norm_mode if settings.qk_norm_mode else 'None'}")
    logger.print_and_log(f"  ] TIE WORD EMB = {model_cfg.tie_word_embeddings}")
    logger.print_and_log(f"  ] ROPE THETA   = {model_cfg.rope_theta:,.0f}")
    logger.print_and_log(f"  ] DROPOUT      = {model_cfg.dropout}")
    logger.print_and_log(f"  ] ACT CKPT     = {model_cfg.use_activation_checkpointing}")
    if model_cfg.use_keel:
        alpha = model_cfg.keel_alpha if model_cfg.keel_alpha else model_cfg.n_layers * 2
        logger.print_and_log(f"  ] KEEL         = ENABLED (alpha={alpha})")
    if getattr(model_cfg, 'attn_res_enabled', False):
        bs = getattr(model_cfg, 'attn_res_block_size', 8)
        n_blocks = model_cfg.n_layers // bs
        remainder = model_cfg.n_layers % bs
        label = f"{n_blocks} blocks of {bs} layers"
        if remainder:
            label += f" + 1 tail block of {remainder} layers"
        logger.print_and_log(f"  ] ATTN RES     = ENABLED ({label})")

    # MoE summary
    moe_layers = [li for li in summary["layer_info"] if li.get("moe")]
    if moe_layers:
        n_moe = len(moe_layers)
        n_dense = summary["num_layers"] - n_moe
        li0 = moe_layers[0]
        logger.print_and_log(f"  ] MoE LAYERS   = {n_moe} MoE + {n_dense} dense")
        ep_degree = getattr(model_cfg, 'ep_degree', 1)
        if ep_degree > 1:
            logger.print_and_log(f"  ] EXPERTS      = {li0['num_experts']} total ({li0['num_local_experts']} local, EP={ep_degree}), top_k = {li0['top_k']}")
        else:
            logger.print_and_log(f"  ] EXPERTS      = {li0['num_experts']}, top_k = {li0['top_k']}")
        logger.print_and_log(f"  ] EXPERT DIM   = {li0['feedforward_intermediate_dim']:,}")
        logger.print_and_log(f"  ] SHARED EXP   = {'yes' if li0['shared_experts'] else 'no'}")

        # Active params per token: only top_k experts fire, not all num_experts.
        # This shows the effective model size a single token "sees" each forward pass.
        _compute_active_params(model, summary, li0, logger)

    # For FLOPs: use effective FFN dim (MoE: top_k * expert_dim, dense: intermediate_dim)
    if moe_layers:
        li0 = moe_layers[0]
        effective_ffn_dim = li0['top_k'] * li0['feedforward_intermediate_dim']
    else:
        effective_ffn_dim = summary['intermediate_dim']

    _, flops_per_token = compute_transformer_flops_per_token(
        settings.cfg_embd, settings.cfg_layers, settings.cfg_heads,
        effective_ffn_dim, settings.T, settings.cfg_voc_sz
    )
    return flops_per_token


# Compute the FLOPs per token for a given model configuration
def compute_transformer_flops_per_token(
    dim: int,
    n_layers: int,
    n_heads: int,
    ffn_dim: int,
    seq_length: int,
    vocab_size: int,
    *,
    include_backward: bool = True,
    backprop_factor: float = 3.0,
):
    """Return per‑token FLOPs for forward and (optionally) training."""
    if dim % n_heads:
        fatal_error(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
    head_dim = dim // n_heads

    flops_qkv        = 3 * 2 * dim * dim
    flops_rotary     = n_heads * (head_dim // 2) * 6
    flops_qk         = n_heads * 2 * head_dim * seq_length
    flops_softmax    = n_heads * 2 * seq_length            # per‑token
    flops_attn_v     = n_heads * 2 * head_dim * seq_length
    flops_out_proj   = 2 * dim * dim
    norm_cost_attn   = 20_000
    attn_total       = sum((flops_qkv, flops_rotary, flops_qk,
                            flops_softmax, flops_attn_v, flops_out_proj,
                            norm_cost_attn))

    flops_w1, flops_w3 = (2 * dim * ffn_dim for _ in range(2))
    flops_activation   = 5 * ffn_dim
    flops_w2           = 2 * ffn_dim * dim
    norm_cost_ffn      = 20_000
    ffn_total          = flops_w1 + flops_w3 + flops_activation + flops_w2 + norm_cost_ffn

    per_block   = attn_total + ffn_total
    forward     = n_layers * per_block + 20_000 + 2 * dim * vocab_size
    training    = forward * backprop_factor if include_backward else forward
    return forward, training

# An NVIDIA 3090 can do ~ 142 TFLOPs in bfloat16 or fp16 - we assume each RANK can do 142 TFLOPs
def _device_peak_flops(device_idx: int, dtype: str = "fp16") -> float:
    """
    Rough peak FLOPs for one GPU in the chosen precision.
    """
    prop  = torch.cuda.get_device_properties(device_idx)
    sm    = prop.multi_processor_count          # 82 for 3090
    clock = prop.clock_rate * 1e3               # ~1.695 GHz
    cuda_cores_per_sm = {
        6: 64,   # Pascal
        7: 64,   # Volta / Turing
        8: 128,  # Ampere & later
    }.get(prop.major, 64)

    fp32_flops = 2 * sm * cuda_cores_per_sm * clock  # ~35.6 TFLOPS

    if dtype in ("fp16", "bf16"):
        # Ampere tensor cores: 256 fp16 FMA / SM / clk  -> 4× fp32 cores
        factor = 4 if prop.major >= 8 else 2 if prop.major >= 7 else 1
        return fp32_flops * factor  # 35.6 * 4 = ~142 TFLOPS
    return fp32_flops

def default_peak_table(dtype: str = "fp16") -> float:
    """Fallback when torch cannot query the GPU (e.g. CPU-only run)."""
    # These are for RTX 3090:
    return {"fp32": 35.6e12, "fp16": 142e12, "bf16": 142e12}[dtype]

def compute_mfu(tokens_per_second: float,
                flops_per_token: float,
                ddp_world_size: int,
                data_type: str = "fp16") -> float:
    """
    Model-FLOPs-Utilization (MFU).

    Automatically derives the peak TFLOPs of *each* GPU instead of assuming an
    A100.  Works for mixed clusters of different GPUs as well.
    """
    try:
        # In DDP, each rank can only see its own GPU as device 0
        my_peak = _device_peak_flops(0, data_type)
        theoretical = my_peak * ddp_world_size
    except Exception:
        # Fallback
        theoretical = default_peak_table(data_type) * ddp_world_size
        
    achieved = tokens_per_second * flops_per_token
    return achieved / theoretical

def print_grad_accum_change_points(schedule, *,B: int, T: int, world_size: int, logger):
    """
    Pretty-print the global steps where gradient-accumulation changes, *plus*
    one extra line that marks the start of the constant-GA plateau if that
    segment never triggered a change by itself.
    """
    logger.print_and_log("  ]       Gradient Accumulation Schedule:")
    tok_per_micro = B * T * world_size
    header = f"  ]       step  {'GA':>4}  {'effective batch':>18}   {'% of train':>10}"
    logger.print_and_log(header)

    prev = schedule[0]
    last_boundary = 0                     # last step we printed

    for k, ga in enumerate(schedule):
        if k == 0 or ga != prev:          # "real" change points
            eff_batch = ga * tok_per_micro
            pct = 100 * k / len(schedule)
            line = f"{k:10d}  {ga:4d}  {eff_batch:18,d}  {pct:10.2f}%"
            logger.print_and_log("  ] " + line)
            prev = ga
            last_boundary = k

def build_user_defined_schedule(ga_schedule, max_steps, tok_per_micro, ddp_rank):
    """
    Build gradient accumulation schedule from user-defined [step, batch_size] pairs.
    Uses step function: holds each GA value until the next defined step.
    """
    # Initialize schedule array with zeros (will be filled)
    grad_accum_schedule = [0] * max_steps

    # Sort schedule by step (in case user didn't provide them in order)
    sorted_schedule = sorted(ga_schedule, key=lambda x: x[0])

    # Calculate GA values and print validation table
    schedule_points = []
    for step, desired_batch in sorted_schedule:
        desired_ga = desired_batch / tok_per_micro
        actual_ga = max(1, round(desired_ga))  # Round to nearest int, minimum 1
        actual_batch = actual_ga * tok_per_micro
        schedule_points.append({
            'step': step,
            'ga': actual_ga,
            'desired_batch': desired_batch,
            'actual_batch': actual_batch,
            'diff': actual_batch - desired_batch
        })

    # Build step function schedule
    for i, point in enumerate(schedule_points):
        start_step = point['step']
        # End step is either the next schedule point or the end of training
        end_step = schedule_points[i + 1]['step'] if i + 1 < len(schedule_points) else max_steps

        # Fill in the GA value for this range
        for step in range(start_step, end_step):
            if step < max_steps:
                grad_accum_schedule[step] = point['ga']

    return grad_accum_schedule

def build_automatic_schedule(settings, tok_per_micro, ddp_rank):
    """
    Build gradient accumulation schedule automatically based on target/min batch sizes.
    This is the legacy automatic mode.
    """
    max_grad_accum_steps = settings.target_batch_size // tok_per_micro

    # Calculate minimum GA steps if min_batch_size is specified
    min_grad_accum_steps = 1
    if hasattr(settings, 'min_batch_size') and settings.min_batch_size:
        min_grad_accum_steps = max(1, settings.min_batch_size // tok_per_micro)

    total_ramp_steps = int(settings.max_steps * settings.ramp_percent)

    # Calculate the GA values we'll use (powers of 2 up to max, but starting from min)
    ga_values = []
    current = 1
    while current <= max_grad_accum_steps:
        if current >= min_grad_accum_steps:  # Only include values >= minimum
            ga_values.append(current)
        current *= 2

    # Ensure we include the exact max value
    if not ga_values or ga_values[-1] != max_grad_accum_steps:
        ga_values.append(max_grad_accum_steps)

    num_stages = len(ga_values)

    # Build the schedule array directly
    grad_accum_schedule = [max_grad_accum_steps] * settings.max_steps  # Start with all at max

    if total_ramp_steps > 0 and num_stages > 1:
        # Calculate transition points
        segment_length = total_ramp_steps // (num_stages - 1)

        for i, ga_value in enumerate(ga_values[:-1]):  # Skip the last (max) value
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_stages - 2 else total_ramp_steps
            grad_accum_schedule[start:end] = [ga_value] * (end - start)

    return grad_accum_schedule

def build_training_schedule(settings, ddp_world_size, ddp_rank, train_loader):
    tok_per_micro = settings.B * settings.T * ddp_world_size

    # Calculate total_ramp_steps (needed for logging, even with user-defined schedule)
    total_ramp_steps = 0
    if hasattr(settings, 'ramp_percent'):
        total_ramp_steps = int(settings.max_steps * settings.ramp_percent)

    # ╭──────────────────────────────────────────────────────────────────────────╮
    # │ Check if user provided explicit gradient schedule                        │
    # ╰──────────────────────────────────────────────────────────────────────────╯
    if hasattr(settings, 'ga_schedule') and settings.ga_schedule:
        # User-defined schedule mode
        grad_accum_schedule = build_user_defined_schedule(
            settings.ga_schedule,
            settings.max_steps,
            tok_per_micro,
            ddp_rank
        )
        # Set total_batch_size based on final GA value
        settings.total_batch_size = grad_accum_schedule[-1] * tok_per_micro
    else:
        # Automatic schedule mode (legacy)
        grad_accum_schedule = build_automatic_schedule(
            settings,
            tok_per_micro,
            ddp_rank
        )
        settings.total_batch_size = (settings.target_batch_size // tok_per_micro) * tok_per_micro

    # ╭──────────────────────────────────────────────────────────────────────────╮
    # │ 3.  Build learning rate restart schedule                                 │
    # ╰──────────────────────────────────────────────────────────────────────────╯
    if getattr(settings, 'auto_restart_points', False):
        settings.restart_steps = [
            int(settings.max_steps * 0.10),
            int(settings.max_steps * 0.25),
            int(settings.max_steps * 0.50)
        ]

    # ╭──────────────────────────────────────────────────────────────────────────╮
    # │ 4.  Console summaries                                                    │
    # ╰──────────────────────────────────────────────────────────────────────────╯
    if ddp_rank == 0:
        logger.print_and_log("Training schedule summary:")
        print_grad_accum_change_points(grad_accum_schedule,B=settings.B,T=settings.T,world_size=ddp_world_size,logger=logger)
        
        logger.print_and_log(f"  ] Final Batch Size         = {settings.total_batch_size:,}")
        logger.print_and_log(f"  ] Max Steps                = {settings.max_steps:,}")
        # Only log ramp steps if using automatic schedule (which uses ramp_percent)
        if hasattr(settings, 'ramp_percent') and not (hasattr(settings, 'ga_schedule') and settings.ga_schedule):
            logger.print_and_log(f"  ] GA Ramp Steps            = {total_ramp_steps:,} ({settings.ramp_percent*100:.0f}%)")
        logger.print_and_log(f"  ] Grad Clip Warmup         = {settings.clip_warmup:.1f}")
        logger.print_and_log(f"  ] Grad Clip Standard       = {settings.clip_standard:.1f}")

        # Calculate EXACT tokens to be processed
        tokens_per_micro = settings.B * settings.T * ddp_world_size
        total_tokens_to_process = sum(ga * tokens_per_micro for ga in grad_accum_schedule)
        effective_epochs = total_tokens_to_process / train_loader.active_tokens

        logger.print_and_log(f"  ] Total Tokens in Dataset  = {train_loader.total_tokens:,}")
        logger.print_and_log(f"  ] Active Tokens in Dataset = {train_loader.active_tokens:,}")
        logger.print_and_log(f"  ] Total Tokens to Process  = {total_tokens_to_process:,}")
        logger.print_and_log(f"  ] Effective Epochs (exact) = {effective_epochs:.2f}")

    return grad_accum_schedule

class Settings:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):

        # Initialize from the provided dictionary (from YAML or CLI)
        if config_dict:
            for key, value in config_dict.items():
                # Convert lists to tuples for groups if needed
                if key == "groups" and isinstance(value, list):
                    value = [tuple(g) if isinstance(g, list) else g for g in value]
                setattr(self, key, value)
        
        # Set derived values
        if not hasattr(self, 'min_lr') or self.min_lr is None:
            self.min_lr = self.max_lr * 0.1

        # Learning rate schedule defaults
        if not hasattr(self, 'restart_steps'):
            self.restart_steps = ()
        if not hasattr(self, 'restart_gamma'):
            self.restart_gamma = 1.0

        # If config has nas_root, use it; otherwise, default to "./log/"
        if not hasattr(self, 'nas_root'):
            self.nas_root = "./log/"
        
        if not hasattr(self, 'nas_path') or self.nas_path is None:
            self.nas_path = os.path.join(self.nas_root, self.run_name) + "/"
        
        if hasattr(self, 'resume_training') and self.resume_training:
            # Make sure there is a resume step
            if not hasattr(self, 'resume_step') or self.resume_step is None:
                fatal_error("resume_training is True but resume_step is not set.\nPlease set resume_step in your config file.")
            
            if (not hasattr(self, 'resume_checkpoint_path') or self.resume_checkpoint_path is None):
                self.resume_checkpoint_path = f"{self.nas_path}model_step_{self.resume_step:06d}.pt"
        
        # if config has local_checkpoint_root, use it; otherwise, default to "~/checkpoints/"
        if not hasattr(self, 'local_checkpoint_root'):
            self.local_checkpoint_root = "~/checkpoints/"

        if not hasattr(self, 'local_checkpoint_dir') or self.local_checkpoint_dir is None:
            expanded = os.path.expanduser(self.local_checkpoint_root)
            self.local_checkpoint_dir = os.path.join(expanded, self.run_name)

        # Ensure qk_norm_mode has a default value
        if not hasattr(self, 'qk_norm_mode'):
            self.qk_norm_mode = None

        # Default to tying for backward compatibility / memory efficiency
        if not hasattr(self, 'tie_word_embeddings'):
            self.tie_word_embeddings = True

        # --- Optimizer type validation ---
        # Detect legacy optimizer flags and fail with a clear migration message
        legacy_flags = ['use_muon', 'use_adamc', 'use_8bit_adam', 'use_adafactor']
        found_legacy = [f for f in legacy_flags if hasattr(self, f)]
        if found_legacy:
            fatal_error(
                "Legacy optimizer flags detected: " + ", ".join(found_legacy) + "\n\n"
                "These have been replaced by a single 'optimizer_type' field.\n"
                "Please update your YAML config. Examples:\n"
                "  optimizer_type: adamc_8bit     # was use_8bit_adam: torchao + use_adamc: true\n"
                "  optimizer_type: normuon_fsdp2  # was use_muon: normuon_fsdp2\n"
                "  optimizer_type: adamw           # was use_8bit_adam: no + use_adamc: false\n\n"
                "Valid types: " + ", ".join(sorted(VALID_OPTIMIZER_TYPES))
            )

        if not hasattr(self, 'optimizer_type'):
            self.optimizer_type = "adamw"

        if self.optimizer_type not in VALID_OPTIMIZER_TYPES:
            fatal_error(
                f"Unknown optimizer_type: '{self.optimizer_type}'\n\n"
                "Valid types: " + ", ".join(sorted(VALID_OPTIMIZER_TYPES))
            )

        # --- Weight decay normalization: scalar or list of rules ---
        if hasattr(self, 'weight_decay'):
            wd = self.weight_decay
            if isinstance(wd, (int, float)):
                self.weight_decay = float(wd)
            elif isinstance(wd, list):
                pass  # rules list — validated at parse time
            else:
                fatal_error(f"weight_decay must be a number or list of rules, got {type(wd).__name__}")
        else:
            self.weight_decay = 0.0

        # --- lr_mods: optional per-layer LR modifiers ---
        if not hasattr(self, 'lr_mods'):
            self.lr_mods = None

        # --- output_lr_batch_adjust: auto output-head LR scaling based on effective batch ---
        # formula: output_lr = body_lr * base_mult * (ref_batch / current_batch) ** exponent
        # ref_batch defaults to grad_accum_schedule[0] * tok_per_micro (the starting eff batch)
        if not hasattr(self, 'output_lr_batch_adjust'):
            self.output_lr_batch_adjust = None
        elif self.output_lr_batch_adjust is not None:
            cfg = self.output_lr_batch_adjust
            if not isinstance(cfg, dict):
                fatal_error(f"output_lr_batch_adjust must be a dict, got {type(cfg).__name__}")
            for key in ('base_mult', 'exponent'):
                if key not in cfg:
                    fatal_error(f"output_lr_batch_adjust missing required field: '{key}'")
                if not isinstance(cfg[key], (int, float)):
                    fatal_error(f"output_lr_batch_adjust.{key} must be a number, got {type(cfg[key]).__name__}")
            if 'ref_batch' in cfg and cfg['ref_batch'] is not None and not isinstance(cfg['ref_batch'], (int, float)):
                fatal_error(f"output_lr_batch_adjust.ref_batch must be a number, got {type(cfg['ref_batch']).__name__}")
            # Hard conflict: output_lr_batch_adjust fully replaces any manual [out, ...] lr_mod.
            if self.lr_mods:
                for entry in self.lr_mods:
                    if isinstance(entry[0], str) and len(entry) == 2 and entry[0] == 'out':
                        fatal_error(
                            "Config conflict: output_lr_batch_adjust and lr_mods [out, ...] cannot both be set.\n"
                            "output_lr_batch_adjust computes the 'out' schedule automatically from the GA schedule — "
                            "it fully replaces any manual [out, ...] rule.\n"
                            "Remove one of them."
                        )

        # --- adaptive_wd: optional adaptive weight decay ---
        if not hasattr(self, 'adaptive_wd'):
            self.adaptive_wd = None
        elif isinstance(self.adaptive_wd, dict):
            if not self.adaptive_wd.get('enabled', False):
                self.adaptive_wd = None

        # --- auxiliary_heads: intermediate-depth prediction heads ---
        # Validated lightly here; full parse happens in parse_aux_heads_config
        # at trainer setup time (needs n_layers from model_cfg to range-check).
        if not hasattr(self, 'auxiliary_heads'):
            self.auxiliary_heads = None
        elif isinstance(self.auxiliary_heads, dict):
            if not self.auxiliary_heads.get('enabled', False):
                self.auxiliary_heads = None

        # --- GDN hybrid attention ---
        if not hasattr(self, 'gdn_enabled'):
            self.gdn_enabled = False
        if not hasattr(self, 'gdn_interleave_step'):
            self.gdn_interleave_step = 4
        if not hasattr(self, 'n_gdn_heads'):
            self.n_gdn_heads = None
        if not hasattr(self, 'gdn_head_dim'):
            self.gdn_head_dim = None
        if not hasattr(self, 'gdn_v_expand'):
            self.gdn_v_expand = 2.0
        if not hasattr(self, 'gdn_short_conv_kernel'):
            self.gdn_short_conv_kernel = 4
        if not hasattr(self, 'gdn_mode'):
            self.gdn_mode = 'chunk'

    def handle_arguments(self, args: argparse.Namespace):
        """Update settings based on command line arguments."""
        if args.run_name:
            self.run_name = args.run_name
            self.nas_path = f"./log/{self.run_name}/"
            self.resume_checkpoint_path = f"{self.nas_path}model_step_{self.resume_step:06d}.pt"
            self.local_checkpoint_dir = os.path.expanduser(f"~/checkpoints/{self.run_name}")

        if args.resume_training is not None:
            self.resume_training = args.resume_training

        if args.resume_step is not None:
            self.resume_step = args.resume_step
            self.resume_checkpoint_path = f"{self.nas_path}model_step_{self.resume_step:06d}.pt"

        # Verify that we have a valid run name and max_steps
        if not self.run_name:
            print("Run name is required. Please specify --run-name <name> or set it in the config file.")
            sys.exit(1)

        if not self.max_steps:
            print("max_steps is required. Please specify it in the config file.")
            sys.exit(1)
            
    def __str__(self):
        return "\n".join(f"{key}: {value}" for key, value in vars(self).items())

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Settings':
        """Load settings from a YAML file.

        Also stashes the raw source text and the source path on the instance
        so the run-directory snapshot can be a verbatim copy of the original
        config file rather than a machine-generated dump that loses comments,
        key order, and formatting.
        """
        with open(yaml_path, 'r') as f:
            raw_text = f.read()
        config_dict = yaml.safe_load(raw_text)
        instance = cls(config_dict)
        instance._source_yaml_text = raw_text
        instance._source_yaml_path = yaml_path
        return instance

    def to_yaml(self, yaml_path: str):
        """Save current settings to a YAML file.

        Writes a verbatim copy of the original source YAML (preserving
        comments, key order, and formatting), then appends a trailing block
        of any keys present on the live settings object but absent from the
        source. That block captures runtime-derived fields (cfg_voc_sz,
        total_batch_size, restart_steps, ...) and CLI-injected overrides
        (resume_step, ...) so dashboards and post-hoc analysis always see
        a complete record of what the run actually used.

        Falls back to a raw dump of the effective settings dict for
        programmatically-constructed instances that have no source file.
        """
        source_text = getattr(self, '_source_yaml_text', None)
        if source_text is not None:
            try:
                cmdline = shlex.join(sys.argv)
            except (AttributeError, TypeError):
                cmdline = " ".join(sys.argv)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                f"# Saved:   {timestamp}\n"
                f"# Command: {cmdline}\n"
                f"\n"
            )

            try:
                source_dict = yaml.safe_load(source_text) or {}
            except yaml.YAMLError:
                source_dict = {}
            derived = {}
            for key, value in vars(self).items():
                if key.startswith('_'):
                    continue
                if key in source_dict:
                    continue
                if key == "groups" and isinstance(value, list):
                    value = [[g[0], g[1]] if isinstance(g, tuple) else g for g in value]
                derived[key] = value

            with open(yaml_path, 'w') as f:
                f.write(header)
                f.write(source_text)
                if not source_text.endswith("\n"):
                    f.write("\n")
                if derived:
                    f.write("\n# --- Derived fields (computed at runtime, not in source config) ---\n")
                    try:
                        yaml.dump(derived, f, default_flow_style=False, sort_keys=False)
                    except yaml.YAMLError as e:
                        f.write(f"# (failed to dump derived fields: {e})\n")
            return

        # Fallback: raw dump of the effective settings (legacy behavior).
        config_dict = {}
        for key, value in vars(self).items():
            if key.startswith('_'):
                continue  # skip private stash fields like _source_yaml_text
            # Convert tuples to lists for YAML serialization
            if key == "groups" and isinstance(value, list):
                value = [[g[0], g[1]] if isinstance(g, tuple) else g for g in value]
            config_dict[key] = value

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Arcane LLM Trainer')
    parser.add_argument('--config', type=str, default=None,help='Path to YAML configuration file')
    
    # Allow overriding specific settings from command line
    parser.add_argument('--run-name', type=str, default=None,help='Override run name from config')
    parser.add_argument('--resume-training', action='store_true', default=None,help='Resume training from checkpoint')
    parser.add_argument('--resume-step', type=int, default=None,help='Step to resume from')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # ----------------------- Load settings -----------------------
    if args.config:
        settings = Settings.from_yaml(args.config)        
        settings.handle_arguments(args) # Handle command line overrides
    else:
        print("Configuration file required. Use --config <path> to specify.")
        # Exit early if no config is provided
        sys.exit(1)

    # ----------------------- Initialize DDP & Logger -----------------------
    # EP degree: auto-default to world_size when MoE is enabled and ep_degree not set.
    # For single-node, EP=world_size is optimal (fast intra-node all-to-all).
    # Set ep_degree: 1 in config to explicitly disable EP.
    ep_degree = getattr(settings, 'ep_degree', None)
    if ep_degree is None:
        if getattr(settings, 'moe_enabled', False):
            ep_degree = int(os.environ.get('WORLD_SIZE', 1))
        else:
            ep_degree = 1
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type, mesh, ep_mesh, edp_mesh = setup_ddp(ep_degree)

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # torch.set_float32_matmul_precision('high')  # or 'medium' - This isn't helpful for bf16/fp16
    
    # Use MASTER_ADDR from environment (same as DDP uses)
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    logger_port = int(getattr(settings, 'server_port', 29600))  # Separate port from DDP

    logger._instance.set_server_host(master_addr)
    logger._instance.set_server_port(int(logger_port))
    logger._instance.set_logdir(settings.nas_path)
    logger._instance.set_resume(settings.resume_training)
    logger._instance.set_default_logfile(settings.gen_log_file)
    logger._instance.set_rank(ddp_rank)

    # Register NAS recovery callback to retry checkpoint sync (one per node)
    if ddp_local_rank == 0:
        logger._instance.on_nas_recovery(
            lambda: trigger_checkpoint_sync(settings, ddp_rank)
        )

    logger.print_and_log(f"======================================================")
    logger.print_and_log(f"Arcane LLM Trainer v0.9")
    logger.print_and_log("")
    logger.print_and_log(f"Run Name: {settings.run_name}")
    logger.print_and_log(f"======================================================")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ----------------------- Create the data loaders -----------------------
    # Build data schedule from groups (if any group has per-step waypoints).
    #
    # When resuming, the dataloader must be initialized with the mix at the
    # *resume step*, not at step 0. Otherwise any group that is 0% at step 0
    # but active mid-schedule (e.g. a dataset that ramps in at step 11,000)
    # will be placed in `deprecated_groups` at construction time, and
    # `set_state` will silently drop its saved shard position — the group
    # effectively resumes from a fresh shard instead of continuing.
    data_schedule = DataMixSchedule.from_groups(settings.groups)
    resume_step = settings.resume_step if settings.resume_training else None
    mix_step = resume_step if resume_step is not None else 0
    if data_schedule is not None:
        initial_groups = data_schedule.get_initial_groups(step=mix_step)
    else:
        initial_groups = settings.groups

    # Skip shard init for train_loader when resuming - shards will be loaded via set_state()
    train_loader = PercentageDataLoader(B=settings.B, T=settings.T, rank=ddp_rank, world_size=ddp_world_size, split="train",
                                        data_root=settings.data_root_path, validation=False, groups=initial_groups,
                                        skip_shard_init=settings.resume_training, data_schedule=data_schedule,
                                        resume_step=resume_step)

    val_loader = PercentageDataLoader(B=settings.B, T=settings.T, rank=ddp_rank, world_size=ddp_world_size, split="val",
                                      data_root=settings.data_root_path, validation=True, groups=initial_groups,
                                      data_schedule=data_schedule, resume_step=resume_step)

    # ----------------------- Build the GA & training schedule -----------------------
    grad_accum_schedule = build_training_schedule(settings, ddp_world_size, ddp_rank, train_loader)

    # ----------------------- Auto output-head LR schedule (batch-adjusted) -----------
    # Computes a [out, [[step, mult], ...]] lr_mods entry from the GA schedule so that
    # the output head LR shrinks as effective batch size grows.
    if settings.output_lr_batch_adjust is not None:
        cfg = settings.output_lr_batch_adjust
        tok_per_micro = settings.B * settings.T * ddp_world_size
        ref_batch = cfg.get('ref_batch')
        if ref_batch is None:
            ref_batch = grad_accum_schedule[0] * tok_per_micro
        ref_batch = float(ref_batch)
        base_mult = float(cfg['base_mult'])
        exponent = float(cfg['exponent'])

        out_schedule = []
        last_ga = None
        for step, ga in enumerate(grad_accum_schedule):
            if ga != last_ga:
                eff_batch = ga * tok_per_micro
                mult = base_mult * (ref_batch / eff_batch) ** exponent
                out_schedule.append([step, mult])
                last_ga = ga

        if settings.lr_mods is None:
            settings.lr_mods = []
        settings.lr_mods.append(['out', out_schedule])

        if ddp_rank == 0:
            logger.print_and_log("Output Head LR Batch Adjustment:")
            logger.print_and_log(f"  ] base_mult = {base_mult}")
            logger.print_and_log(f"  ] exponent  = {exponent}")
            logger.print_and_log(f"  ] ref_batch = {int(ref_batch):,}")
            logger.print_and_log(f"  ] formula   = body_lr * {base_mult} * ({int(ref_batch):,} / eff_batch) ** {exponent}")
            logger.print_and_log(f"  ]      step       eff_batch     mult")
            for step, mult in out_schedule:
                eff_batch = grad_accum_schedule[step] * tok_per_micro
                logger.print_and_log(f"  ] {step:9,d}  {eff_batch:14,d}  {mult:7.4f}")

    # Print info about the learning rate schedule
    log_lr_schedule(settings, logger)

    # ----------------------- Get the tokenizer from tokenizer_abstraction -----------------------
    special_tokens_path = getattr(settings, 'special_tokens', None)

    enc = get_tokenizer(
        settings.tok_kind,
        path=settings.tok_path,
        special_tokens=special_tokens_path,
    )

    def round_up(x: int, multiple: int = 128) -> int:
        return (x + multiple - 1) // multiple * multiple
    settings.cfg_voc_sz = round_up(len(enc), 1024)          # GPU friendly size: 100_278 -> 100_352 for TikToken cl100k_base
    logger.print_and_log(f"Tokenizer Info:")
    logger.print_and_log(f"  ] Tokenizer vocab = {len(enc):,}  →  rounded to {settings.cfg_voc_sz:,}")
    logger.print_and_log(f"  ] Tokenizer Kind = {settings.tok_kind}")
    logger.print_and_log(f"  ] Tokenizer Path = {settings.tok_path}")
    if special_tokens_path:
        logger.print_and_log(f"  ] Special tokens = {special_tokens_path}")

    # ----------------------- Auxiliary prediction heads -----------------------
    # Parse auxiliary_heads YAML block. We only need the layer list here to
    # build the model; per-head weight schedules are re-parsed inside train_loop.
    _aux_head_layers, _ = parse_aux_heads_config(getattr(settings, 'auxiliary_heads', None))
    for _li in _aux_head_layers:
        if _li < 0 or _li >= settings.cfg_layers:
            fatal_error(
                f"auxiliary_heads.heads layer {_li} out of range for cfg_layers={settings.cfg_layers}"
            )

    # ----------------------- Create the model -----------------------
    model_cfg = ModelArgs(
        dim=settings.cfg_embd,
        inner_dim=settings.cfg_intermediate,
        n_heads=settings.cfg_heads,
        n_kv_heads=getattr(settings, 'cfg_kv_heads', None),  # GQA support (optional)
        n_layers=settings.cfg_layers,
        vocab_size=settings.cfg_voc_sz,
        max_seq_len=settings.T,
        dropout=settings.dropout,
        use_activation_checkpointing=settings.use_activation_checkpointing,
        norm_eps=settings.norm_eps,
        pad_id=enc.pad_id,
        qk_norm_mode=settings.qk_norm_mode,
        tie_word_embeddings=getattr(settings, 'tie_word_embeddings', True),
        rope_theta=getattr(settings, 'rope_theta', 500000.0),
        use_keel=getattr(settings, 'use_keel', False),
        keel_alpha=getattr(settings, 'keel_alpha', None),
        # MoE configuration
        moe_enabled=getattr(settings, 'moe_enabled', False),
        moe_num_experts=getattr(settings, 'moe_num_experts', 8),
        moe_top_k=getattr(settings, 'moe_top_k', 2),
        moe_num_shared_experts=getattr(settings, 'moe_num_shared_experts', 1),
        moe_score_func=getattr(settings, 'moe_score_func', 'sigmoid'),
        moe_score_before_experts=getattr(settings, 'moe_score_before_experts', True),
        moe_route_norm=getattr(settings, 'moe_route_norm', False),
        moe_route_scale=getattr(settings, 'moe_route_scale', 1.0),
        moe_load_balance_coeff=getattr(settings, 'moe_load_balance_coeff', 1e-3),
        moe_interleave_step=getattr(settings, 'moe_interleave_step', 1),
        moe_n_dense_layers=getattr(settings, 'moe_n_dense_layers', 0),
        moe_n_tail_dense_layers=getattr(settings, 'moe_n_tail_dense_layers', 0),
        moe_capacity_factor=getattr(settings, 'moe_capacity_factor', 0.0),
        moe_inner_dim=getattr(settings, 'moe_inner_dim', None),
        ep_degree=ep_degree,
        # GDN hybrid attention
        gdn_enabled=getattr(settings, 'gdn_enabled', False),
        gdn_interleave_step=getattr(settings, 'gdn_interleave_step', 4),
        n_gdn_heads=getattr(settings, 'n_gdn_heads', None),
        gdn_head_dim=getattr(settings, 'gdn_head_dim', None),
        gdn_v_expand=getattr(settings, 'gdn_v_expand', 2.0),
        gdn_short_conv_kernel=getattr(settings, 'gdn_short_conv_kernel', 4),
        gdn_mode=getattr(settings, 'gdn_mode', 'chunk'),
        # AttnRes (Block Attention Residuals)
        attn_res_enabled=getattr(settings, 'attn_res_enabled', False),
        attn_res_block_size=getattr(settings, 'attn_res_block_size', 8),
        # Auxiliary prediction heads
        aux_head_layers=_aux_head_layers,
    )

    # ----------------------- Save Settings Config File -----------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(settings.nas_path, f"config_{timestamp}.yaml")
    if ddp_rank == 0:
        settings.to_yaml(config_path)
        logger.print_and_log(f"Configuration saved to {config_path}")

    check_params(model_cfg, settings=settings, world_size=ddp_world_size)

    # ----------------------- Create and shard model -----------------------
    model = create_and_shard_model(model_cfg, mesh, ep_mesh, edp_mesh, device, settings, logger)

    # ----------------------- Print model summary -----------------------
    flops_per_token = 0
    if ddp_rank == 0:
        flops_per_token = print_model_summary(model, model_cfg, settings, logger)

    # ----------------------- Log MoE configuration -----------------------
    if model_cfg.moe_enabled and ddp_rank == 0:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        n_moe = sum(1 for l in raw.layers if getattr(l, 'moe_enabled', False))
        n_dense = model_cfg.n_layers - n_moe
        logger.print_and_log(f"MoE Config:")
        n_head_dense = getattr(model_cfg, 'moe_n_dense_layers', 0)
        n_tail_dense = getattr(model_cfg, 'moe_n_tail_dense_layers', 0)
        if n_tail_dense:
            logger.print_and_log(f"  ] {n_moe} MoE layers, {n_head_dense} head-dense, {n_tail_dense} tail-dense (synth)")
        else:
            logger.print_and_log(f"  ] {n_moe} MoE layers, {n_dense} dense layers")
        logger.print_and_log(f"  ] Experts        = {model_cfg.moe_num_experts}, top_k = {model_cfg.moe_top_k}")
        logger.print_and_log(f"  ] Shared experts = {model_cfg.moe_num_shared_experts}")
        logger.print_and_log(f"  ] Score func     = {model_cfg.moe_score_func}")
        logger.print_and_log(f"  ] Load balance   = {model_cfg.moe_load_balance_coeff}")
        if getattr(model_cfg, 'moe_capacity_factor', 0) > 0:
            logger.print_and_log(f"  ] Capacity       = {model_cfg.moe_capacity_factor} (train-only token dropping)")
        if model_cfg.moe_inner_dim:
            logger.print_and_log(f"  ] Expert hidden  = {model_cfg.moe_inner_dim}")
        if ep_degree > 1:
            auto_tag = " (auto=world_size)" if getattr(settings, 'ep_degree', None) is None else ""
            logger.print_and_log(f"  ] EP degree      = {ep_degree} (local experts = {model_cfg.moe_num_experts // ep_degree}){auto_tag}")

    # ----------------------- Log GDN configuration -----------------------
    if getattr(model_cfg, 'gdn_enabled', False) and ddp_rank == 0:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        n_gdn = sum(1 for l in raw.layers if getattr(l, 'use_gdn', False))
        n_attn = model_cfg.n_layers - n_gdn
        gdn_heads = model_cfg.n_gdn_heads or model_cfg.n_heads
        gdn_hdim = model_cfg.gdn_head_dim or 256
        logger.print_and_log(f"GDN Hybrid Config:")
        logger.print_and_log(f"  ] {n_gdn} GDN layers, {n_attn} full-attention layers")
        logger.print_and_log(f"  ] Interleave    = every {model_cfg.gdn_interleave_step}th layer is full-attention")
        logger.print_and_log(f"  ] GDN heads     = {gdn_heads} (head_dim={gdn_hdim})")
        logger.print_and_log(f"  ] V expand      = {model_cfg.gdn_v_expand}")
        logger.print_and_log(f"  ] Conv kernel   = {model_cfg.gdn_short_conv_kernel}")
        logger.print_and_log(f"  ] Mode          = {model_cfg.gdn_mode}")
        logger.print_and_log(f"  ] Full-attn gate = sigmoid (gated softmax attention)")

    # ----------------------- Log Auxiliary Heads configuration -----------------------
    if _aux_head_layers and ddp_rank == 0:
        _, _aux_print_schedules = parse_aux_heads_config(getattr(settings, 'auxiliary_heads', None))
        # ~ params per head: RMSNorm(d) + Linear(d, V) = d + d*V
        _params_per_head = model_cfg.dim + model_cfg.dim * settings.cfg_voc_sz
        _total_aux_params = _params_per_head * len(_aux_head_layers)
        logger.print_and_log(f"Auxiliary Heads Config:")
        logger.print_and_log(
            f"  ] Taps          = {len(_aux_head_layers)} heads at layers "
            f"{', '.join(f'L{li}' for li in _aux_head_layers)}"
        )
        logger.print_and_log(
            f"  ] Architecture  = RMSNorm + Linear(d={model_cfg.dim}, V={settings.cfg_voc_sz:,})"
        )
        logger.print_and_log(
            f"  ] Param overhead= ~{_total_aux_params/1e6:.1f}M total "
            f"({_params_per_head/1e6:.1f}M per head x {len(_aux_head_layers)})"
        )
        logger.print_and_log(f"  ] Optimizer     = Muon group, full LR (no output_lr_batch_adjust scaling)")
        logger.print_and_log(f"  ] FSDP          = each head gets its own fully_shard wrap")
        logger.print_and_log(f"  ] Schedules     (linear interpolation between waypoints):")
        for _li in _aux_head_layers:
            _sched = _aux_print_schedules[_li]
            if len(_sched) == 1:
                logger.print_and_log(f"  ]   L{_li:>3d}: constant {_sched[0][1]}")
            else:
                _wp = " -> ".join(f"{v} @ step {s:,}" for s, v in _sched)
                logger.print_and_log(f"  ]   L{_li:>3d}: {_wp}")

    # ----------------------- Log training configuration -----------------------
    if ddp_rank == 0:
        logger.print_and_log(f"Training Config:")
        logger.print_and_log(f"  ] Data Type      = {settings.data_type}")
        logger.print_and_log(f"  ] FSDP Param     = {settings.FSDP_param_dtype}")
        logger.print_and_log(f"  ] FSDP Reduce    = {settings.FSDP_reduce_dtype}")
        logger.print_and_log(f"  ] Micro Batch    = {settings.B}")
        logger.print_and_log(f"  ] Context (T)    = {settings.T}")
        logger.print_and_log(f"  ] Val Step       = {settings.val_step}")
        logger.print_and_log(f"  ] Save Step      = {settings.save_step}")
        logger.print_and_log(f"  ] Eval Iters     = {settings.eval_iters}")
        logger.print_and_log(f"  ] Data Root      = {settings.data_root_path}")
        if getattr(settings, 'cpu_offload', False):
            logger.print_and_log(f"  ] CPU Offload    = ON (params, grads, optimizer states on CPU)")
            if settings.optimizer_type in ('adamw_8bit', 'adamc_8bit'):
                logger.print_and_log(f"  ] WARNING: cpu_offload + {settings.optimizer_type} may cause device mismatch errors (known torchao bug)")

    # ----------------------- Configure the optimizer -----------------------
    logger.print_and_log(f"Optimizer Configuration...")
    start_step = 1
    optimizer = configure_optimizers(
        model=model,
        optimizer_type=settings.optimizer_type,
        weight_decay=settings.weight_decay if isinstance(settings.weight_decay, (int, float)) else 0.0,
        learning_rate=settings.max_lr,
        betas=(getattr(settings, 'beta1', 0.9), getattr(settings, 'beta2', 0.95)),
        device_type=device_type,
        muon_momentum=getattr(settings, 'muon_momentum', 0.95),
        muon_ns_steps=getattr(settings, 'muon_ns_steps', 5),
        normuon_beta2=getattr(settings, 'normuon_beta2', 0.95),
        dion_kwargs=getattr(settings, 'dion_kwargs', None),
        distributed_mesh=mesh if ddp else None,
        adafactor_beta2=getattr(settings, 'adafactor_beta2', None),
        cautious_weight_decay=getattr(settings, 'cautious_weight_decay', False),
        muonsphere_radius_scale=getattr(settings, 'muonsphere_radius_scale', 2.0),
        muonsphere_power_iters=getattr(settings, 'muonsphere_power_iters', 10),
        dion2_fraction=getattr(settings, 'dion2_fraction', 0.25),
        dion2_ef_decay=getattr(settings, 'dion2_ef_decay', 0.95),
        adam16bit_state_dtype=getattr(settings, 'adam16bit_state_dtype', 'mixed'),
        muon_adam_state_dtype=getattr(settings, 'muon_adam_state_dtype', 'fp32'),
    )

    summarize_optimizer_settings(settings, ddp_world_size, grad_accum_schedule[0], logger, model=model)

    # Side-dicts for per-param overrides — shared by reference between train_loop and optimizer
    wd_overrides = {}           # id(param) -> float  (WD rules)
    lr_scale_overrides = {}     # id(param) -> float  (lr_mods)
    optimizer.wd_overrides = wd_overrides
    optimizer.lr_scale_overrides = lr_scale_overrides  # only used by Muon; AdamC ignores this

    # ----------------------- Parse LR modifiers -----------------------
    lr_mod_entries = None
    if settings.lr_mods:
        if settings.optimizer_type in DION_FAMILY:
            logger.print_and_log("WARNING: lr_mods not supported with DION optimizers — ignoring", r0_only=True)
        else:
            # For standalone Adam: filter out attn/ffn rules (can't differentiate within one group)
            effective_rules = settings.lr_mods
            if settings.optimizer_type not in FSDP2_MUON_FAMILY:
                effective_rules = []
                for e in settings.lr_mods:
                    if isinstance(e[0], str) and len(e) == 2:
                        effective_rules.append(e)  # emb/head — always supported
                    elif isinstance(e[0], str) and len(e) == 3:
                        if e[1] == 'all':
                            effective_rules.append(e)  # [all, all, ...] — maps to all groups
                        else:
                            logger.print_and_log(
                                f"WARNING: lr_mods rule [all, {e[1]}] ignored "
                                f"(attn/ffn differentiation requires Muon optimizer)", r0_only=True)
                    elif e[2] == 'all':
                        effective_rules.append(e)  # [start, end, all, ...] — maps to default group
                    else:
                        logger.print_and_log(
                            f"WARNING: lr_mods rule '{e[2]}' for layers {e[0]}-{e[1]} ignored "
                            f"(attn/ffn differentiation requires Muon optimizer)", r0_only=True)

            if effective_rules:
                lr_mod_entries = parse_lr_mods(effective_rules, model)
                if ddp_rank == 0:
                    logger.print_and_log(f"  ] LR mods applied to {len(lr_mod_entries)} parameters")

    # ----------------------- Parse WD rules -----------------------
    wd_entries = None
    has_wd_schedules = False
    if isinstance(settings.weight_decay, list):
        wd_entries = parse_wd_rules(settings.weight_decay, model)
        has_wd_schedules = any(isinstance(wd_val, list) for _, wd_val in wd_entries)
        if ddp_rank == 0:
            logger.print_and_log(f"  ] WD rules applied to {len(wd_entries)} parameters")

    # ----------------------- Initialize AWD (Adaptive Weight Decay) -----------------------
    awd = None
    if settings.adaptive_wd is not None:
        from adaptive_wd import AdaptiveWD
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        base_wd = settings.weight_decay if isinstance(settings.weight_decay, (int, float)) else 0.0
        awd = AdaptiveWD(raw_model, settings.adaptive_wd, ddp_rank, ddp_world_size, ddp, base_wd, wd_overrides)
        if ddp_rank == 0:
            logger.print_and_log(f"Adaptive WD: {len(awd.groups)} groups, check every {awd.check_interval} steps")

    # ----------------------- MoE load balancing hook -----------------------
    moe_balance_hook = None
    moe_stats = [None]  # populated each step by balance hook (list for closure mutation)
    if getattr(model_cfg, 'moe_enabled', False) and getattr(model_cfg, 'moe_load_balance_coeff', None):
        def _update_expert_bias(_model=model, _ddp=ddp):
            """Pre-optimizer-step: update expert_bias from accumulated tokens_per_expert.

            Returns dict with balance stats for logging:
                avg_cv: average coefficient of variation across MoE layers
                per_layer: list of (layer_id, tpe_pct, cv, bias) per MoE layer
            """
            raw = _model._orig_mod if hasattr(_model, "_orig_mod") else _model
            tpe_list = []
            layer_ids = []
            drop_counts = []
            for layer in raw.layers:
                if getattr(layer, 'moe_enabled', False) and layer.moe.load_balance_coeff is not None:
                    tpe_list.append(layer.moe.tokens_per_expert.clone())
                    layer_ids.append(layer.layer_id)
                    drop_counts.append(getattr(layer.moe, '_tokens_dropped_accum', 0))
                    layer.moe._tokens_dropped_accum = 0  # zero alongside tokens_per_expert
            if not tpe_list:
                moe_stats[0] = None
                return
            tpe_stacked = torch.vstack(tpe_list)
            if _ddp:
                dist.all_reduce(tpe_stacked, op=dist.ReduceOp.SUM)

            # Compute balance stats before updating bias
            per_layer = []
            cvs = []
            for i, lid in enumerate(layer_ids):
                tpe = tpe_stacked[i].float()
                total = tpe.sum()
                pct = (tpe / total * 100).cpu().tolist() if total > 0 else [0.0] * tpe.shape[0]
                mean = tpe.mean()
                cv = (tpe.std() / mean).item() if mean > 0 else 0.0
                cvs.append(cv)
                per_layer.append((lid, pct, cv))

            # Update expert_bias and zero counters
            idx = 0
            with torch.no_grad():
                for layer in raw.layers:
                    if getattr(layer, 'moe_enabled', False) and layer.moe.load_balance_coeff is not None:
                        tpe = tpe_stacked[idx].float()
                        idx += 1
                        delta = layer.moe.load_balance_coeff * torch.sign(tpe.mean() - tpe)
                        delta = delta - delta.mean()
                        layer.moe.expert_bias.add_(delta)
                        layer.moe.tokens_per_expert.zero_()

            # Attach bias values to per_layer (after update)
            idx = 0
            for layer in raw.layers:
                if getattr(layer, 'moe_enabled', False) and layer.moe.load_balance_coeff is not None:
                    bias = layer.moe.expert_bias.cpu().tolist()
                    lid, pct, cv = per_layer[idx]
                    per_layer[idx] = (lid, pct, cv, bias)
                    idx += 1

            total_dropped = sum(drop_counts)
            # All-reduce drops to match the all-reduced tpe denominator
            if _ddp and total_dropped > 0:
                drop_t = torch.tensor([total_dropped], dtype=torch.float64, device=tpe_stacked.device)
                dist.all_reduce(drop_t, op=dist.ReduceOp.SUM)
                total_dropped = int(drop_t.item())
            # tokens_per_expert accumulates pre-drop counts, so sum is total assignments
            total_assignments = tpe_stacked.sum().item()
            moe_stats[0] = {
                'avg_cv': sum(cvs) / len(cvs) if cvs else 0.0,
                'per_layer': per_layer,
                'drop_counts': drop_counts,
                'total_dropped': total_dropped,
                'drop_pct': (total_dropped / total_assignments * 100) if total_assignments > 0 else 0.0,
            }
        moe_balance_hook = _update_expert_bias
        if ddp_rank == 0:
            parts = [f"bias_update (coeff={model_cfg.moe_load_balance_coeff})"]
            if getattr(model_cfg, 'moe_aux_balance_coeff', 0) > 0:
                parts.append(f"aux_balance_loss={model_cfg.moe_aux_balance_coeff}")
            if getattr(model_cfg, 'moe_bias_before_score', False):
                parts.append("bias_before_score=True")
            logger.print_and_log(f"MoE load balancing: {', '.join(parts)}")

    # ----------------------- Optionally Resume Training  -----------------------
    total_tokens_processed = 0
    if settings.resume_training:
        start_step, total_tokens_processed = resume_training(model, optimizer, train_loader, ddp_rank, settings, grad_accum_schedule, awd=awd)

    # ----------------------- Optionally compile the model -----------------------
    if settings.compile_model:
        # Suppress noisy-but-harmless dynamo warnings from FLA internals
        # (lru_cache tracing + cuda_utils.get_device_properties)
        warnings.filterwarnings("ignore", message=".*lru_cache.*", module="torch._dynamo")
        warnings.filterwarnings("ignore", message=".*cuda_utils.get_device_properties.*", module="torch._dynamo")
        _apply_per_submodule_compile(model, settings.compile_mode, logger)

    dist.barrier()  # Ensure all processes are synchronized before starting training

    # ----------------------- Create layer diagnostics tracker -----------------------
    diagnostics = LayerDiagnostics(model, ddp_rank, ddp_world_size, ddp)
    if diagnostics._init_message:
        logger.print_and_log(f"[Diagnostics] {diagnostics._init_message}")
    logger.print_and_log(f"Layer diagnostics enabled - will log to {settings.nas_path}diagnostics.jsonl")

    # ----------------------- Progressive tail truncation -----------------------
    trunc_cfg = getattr(settings, 'truncation', None) or {}
    truncator = ProgressiveTailTruncation(n_layers=model_cfg.n_layers, config=trunc_cfg)
    # Note: with per-submodule compile, there is no top-level _orig_mod wrapper.
    # Each submodule is individually compiled, so truncation (running fewer layers)
    # doesn't change any compiled graph — bypass_compile is unnecessary.
    if ddp_rank == 0:
        if truncator.enabled:
            logger.print_and_log(f"Tail Truncation:")
            logger.print_and_log(f"  ] Status       = ENABLED")
            logger.print_and_log(f"  ] Depth Power  = {truncator.depth_power} (1=uniform, 2=shallow-biased, 3=strongly shallow)")
            logger.print_and_log(f"  ] Loss Weight  = {truncator.loss_weight}")
            logger.print_and_log(f"  ] Bypass Comp  = {truncator.bypass_compile}")
            logger.print_and_log(f"  ] Safe Frac    = {truncator.fmt_schedule(truncator._safe_schedule)}")
            logger.print_and_log(f"  ] Trunc Prob   = {truncator.fmt_schedule(truncator._prob_schedule)}")
            # Show zone sizes at final scheduled safe_fraction
            final_safe_frac = truncator._safe_schedule[-1][1]
            safe_layer = int(model_cfg.n_layers * final_safe_frac)
            zone_size = model_cfg.n_layers - safe_layer
            logger.print_and_log(f"  ] Safe Zone    = layers 0-{safe_layer - 1} ({safe_layer} layers)")
            logger.print_and_log(f"  ] Trunc Zone   = layers {safe_layer}-{model_cfg.n_layers - 1} ({zone_size} layers)")
        else:
            logger.print_and_log(f"Tail Truncation: DISABLED")

    # ----------------------- Train the miraculous beast! -----------------------
    train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_local_rank, ddp_world_size, start_step,
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule,
        diagnostics=diagnostics,
        truncator=truncator,
        wd_overrides=wd_overrides,
        lr_scale_overrides=lr_scale_overrides,
        awd=awd,
        moe_balance_hook=moe_balance_hook,
    )

    if ddp:
        torch.cuda.synchronize()
        dist.barrier()
        destroy_process_group()

    logger.print_and_log("[R{ddp_rank}] Training complete", False)