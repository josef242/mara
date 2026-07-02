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

# Use absolute path to ensure we get common_fsdp2, not common.
# doc-mask branch: pair with the common_fsdp2-docmask WORKTREE so the branch pair
# (mara_fsdp2-docmask + common_fsdp2-docmask) is self-consistent regardless of what
# the main trees have checked out. Falls back to ../common_fsdp2 off-branch.
common_path = '../common_fsdp2-docmask'
if not os.path.isdir(common_path):
    common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)  # insert at the beginning to prioritize

from tokenizer_abstraction import get_tokenizer
from model_v2 import Transformer, ModelArgs, precompute_freqs_cis
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
def calc_group_loss(model, loader, *, eval_iters, device, ddp, dtype, device_type,
                    scaffold_mode=False, active_layers=None, scs_deepest_tap=None):
    """
    Return the mean loss for the group that `loader` is currently locked to.
    All ranks run the same number of iterations; the result is averaged across
    ranks before it is returned (so every rank gets the final scalar).

    Note: With FSDP2, we must call through the sharded model (not _orig_mod)
    because inputs need to interact with DTensor parameters properly.

    Under SCS scaffold mode, the model's forward returns (None, None) and we
    pull the deepest active aux head's CE from _last_aux_loss_tensors — same
    semantic as the training loop's `ls:` field, so val and train stay
    directly comparable across the scaffold-to-full transition.
    """
    tot = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=dtype):
                _, loss = model(
                    x, y, active_layers=active_layers, scaffold_mode=scaffold_mode,
                )
            if scaffold_mode:
                raw_for_aux = model._orig_mod if hasattr(model, '_orig_mod') else model
                aux_tensors = getattr(raw_for_aux, '_last_aux_loss_tensors', {}) or {}
                # Hard fatal rather than continue/early-return: scaffold mode
                # dispatch is deterministic across ranks (same scs_deepest_tap
                # everywhere), so a missing aux loss on some rank but not
                # others would desync the collective below into a hang. If we
                # hit this it's a real bug and we want it loud.
                if scs_deepest_tap is None or scs_deepest_tap not in aux_tensors:
                    raise RuntimeError(
                        f"calc_group_loss: scaffold_mode=True but scs_deepest_tap="
                        f"{scs_deepest_tap} not in aux_tensors {sorted(aux_tensors.keys())}. "
                        f"Aux head capture path may be misconfigured."
                    )
                loss = aux_tensors[scs_deepest_tap]
            tot += loss.detach()

    tot /= eval_iters                       # average over micro-batches
    if ddp:
        dist.all_reduce(tot, op=dist.ReduceOp.AVG)   # average across ranks
    return tot.item()                       # every rank now holds the same value

def do_validation(model, val_loader, device, eval_iters,
                  step, ddp_rank, val_log_file,
                  total_tokens_processed,
                  ddp, ddp_world_size, data_type, device_type,
                  scaffold_mode=False, active_layers=None, scs_deepest_tap=None):

    logger.print_and_log(f"[R{ddp_rank}] running validation at step {step}")

    t0 = time.time()
    model.eval()
    group_names = val_loader.group_names(active_only=True)
    group_losses = {}

    dtype = torch.bfloat16 if data_type == "bf16" else torch.float16 if data_type == "fp16" else torch.float32

    for g in group_names:
        val_loader.set_val_group(g, eval_iters = eval_iters)
        loss = calc_group_loss(
            model, val_loader, eval_iters=eval_iters, device=device, ddp=ddp,
            dtype=dtype, device_type=device_type,
            scaffold_mode=scaffold_mode, active_layers=active_layers,
            scs_deepest_tap=scs_deepest_tap,
        )
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
def _lr_schedule_fingerprint(settings):
    """Identity of the LR schedule, for the self-anchoring controller's resume guard. The 'auto'
    LR-track reference r=K_anchor*lr(t)/lr_anchor relies on the SAME schedule for plant and reference;
    if any of these change across a resume after the anchor is latched, the lr cancellation is invalid
    and the controller would fight the new curve. Compared in resume_training()."""
    _sched = str(getattr(settings, 'lr_schedule_type', 'restarts'))
    fp = [
        _sched,
        round(float(getattr(settings, 'max_lr', 0.0)), 12),
        round(float(getattr(settings, 'min_lr', 0.0)), 12),
        int(getattr(settings, 'max_steps', 0) or 0),
        int(getattr(settings, 'warmup_steps', 0) or 0),
        tuple(getattr(settings, 'restart_steps', ()) or ()),
        round(float(getattr(settings, 'restart_gamma', 1.0) or 1.0), 12),
    ]
    # 'plateau' (get_lr_with_dual_plateau) is shaped by 6 extra settings; capture them too, else a
    # changed plateau param would slip past the resume guard and the auto controller would fight the
    # new curve. (cosine/restarts ignore these — harmless to include as defaults.)
    if _sched == 'plateau':
        fp += [round(float(getattr(settings, _k, 0.0) or 0.0), 12) for _k in (
            'first_plat_lr', 'decay_to_first_plat_pct', 'first_plat_len_pct',
            'decay_to_second_pct', 'second_plat_lr', 'second_plat_len_pct')]
    return tuple(fp)


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


def get_zloss_alpha(step: int, settings) -> float:
    """Effective z-loss coefficient at absolute global `step`.

    Pure function of (global step + static z_loss config) -> resume-safe with
    NO new checkpoint state, exactly like get_lr: it keys off the same global
    `step` loop variable that round-trips through checkpoint save/load (on
    resume, start_step = checkpoint['step'] + 1), so alpha_eff at a given
    global step is identical across kill/resume.

    Returns 0.0 when z-loss is disabled. With warmup disabled the full target
    alpha applies immediately (the non-annealed behavior). Otherwise, with
    progress = clamp((step - start_step) / duration_steps, 0, 1):
        cosine (default): alpha * (1 - cos(pi * progress)) / 2
        linear:           alpha * progress
    Before start_step -> 0.0; at/after start_step + duration_steps -> alpha.
    """
    z = getattr(settings, 'z_loss', None)
    if z is None:
        return 0.0
    alpha = float(z.get('alpha', 0.0))
    # --- warmup (ramp 0 -> alpha) ---
    warmup = z.get('warmup') or {}
    if warmup.get('enabled', False):
        start_step = int(warmup.get('start_step', 0))
        duration = int(warmup.get('duration_steps', 1))
        shape = warmup.get('shape', 'cosine')
        if step <= start_step:
            alpha = 0.0
        elif duration > 0:
            progress = (step - start_step) / duration
            if progress < 1.0:
                progress = max(0.0, progress)
                if shape == 'linear':
                    alpha = alpha * progress
                else:  # cosine: zero-slope ends — gentlest onset, no kink
                    alpha = alpha * (1.0 - math.cos(math.pi * progress)) / 2.0
    # --- warmdown (ramp alpha -> 0), half-open [start, start+duration) per
    # Guardrail 4: alpha must be EXACTLY 0 from start+duration onward, before
    # row-centering's s goes nonzero at the same global step. Applied as a
    # multiplier on the (post-warmup) alpha so the two ramps compose cleanly. ---
    warmdown = z.get('warmdown') or {}
    if warmdown.get('enabled', False):
        wd_start = int(warmdown.get('start_step', 0))
        wd_dur = int(warmdown.get('duration_steps', 1))
        wd_shape = warmdown.get('shape', 'cosine')
        if step >= wd_start + max(1, wd_dur):
            return 0.0                       # fully warmed down -> exactly 0
        if step >= wd_start and wd_dur > 0:
            u = (step - wd_start) / wd_dur   # in [0, 1)
            if wd_shape == 'linear':
                alpha = alpha * (1.0 - u)
            else:  # cosine_down(u) = 0.5*(1+cos(pi*u))
                alpha = alpha * 0.5 * (1.0 + math.cos(math.pi * u))
    return alpha


def _row_center_cfg(settings):
    """Normalize the row_center_head config to a dict with keys
    {enabled: bool, warmup: dict|None}. Accepts BOTH the flat bool form
    (`row_center_head: true`, steady-state full projection from step 0) and the
    nested dict form (`row_center_head: {enabled, warmup: {...}}`, staged warmup).
    Returns {'enabled': False} when off."""
    rc = getattr(settings, 'row_center_head', False)
    if isinstance(rc, bool):
        return {'enabled': rc, 'warmup': None}
    if isinstance(rc, dict):
        return {'enabled': bool(rc.get('enabled', False)), 'warmup': rc.get('warmup')}
    return {'enabled': False, 'warmup': None}


def _head_gauge_cfg(settings):
    """Normalize the head_gauge_projection config (dn4 head-hygiene) to
    {enabled: bool, init_row_center: bool}. Accepts flat bool
    (`head_gauge_projection: true`) or nested dict. Off by default. This is the
    APPLIED-UPDATE gauge projection (in-optimizer), NOT the legacy row_center_head
    (post-step weight + exp_avg surgery) -- the two are mutually exclusive."""
    hg = getattr(settings, 'head_gauge_projection', None)
    if isinstance(hg, bool):
        return {'enabled': hg, 'init_row_center': hg}
    if isinstance(hg, dict):
        return {'enabled': bool(hg.get('enabled', False)),
                'init_row_center': bool(hg.get('init_row_center', False))}
    return {'enabled': False, 'init_row_center': False}


class GPMTracker:
    """Gradient Productivity Metric — live, on the status line.

    Measures: when grad-norm spikes ABOVE its local trend, does loss drop MORE than its
    local trend on the NEXT step? Positive => big gradients here are productive (strong
    learning signal); ~0 => spikes are noise; negative => spikes are anti-productive.

    Detrends BOTH nrm and dloss against a rolling median (so we measure LOCAL fluctuation
    coupling, not the trivial shared early-vs-late training envelope), then takes a
    Spearman rank correlation (robust to heavy-tailed nrm spikes). Lag-1: nrm[t] vs
    dloss[t]=ls[t]-ls[t+1] (the spike precedes the drop — Josef's observed effect).

    Two windows: GPM-S (short, responsive) and GPM-L (long, stable baseline). The GAP
    between them is itself the signal: S>L => productivity rising (breakthrough igniting);
    S<L => dipping (plateau/saturation). Validated offline (tools/gpm.py): KeelHaul ~+0.25
    vs mf ~+0.13 vs DN2 ~+0.07 at W=51 — the tangent-projected regime has ~2-3.5x more
    productive gradients (radial shock-absorber removed => norm spikes are all loss-relevant).

    Trailing by 1 step (needs ls[t+1] to score nrm[t]); negligible compute (Spearman over
    <=W points once per logged step). rank0 only.
    """
    def __init__(self, w_short=15, w_long=101):
        from collections import deque
        self.ws, self.wl = w_short, w_long
        self.buf = deque(maxlen=w_long + 2)  # (nrm, ls) history, sized to the long window

    def update(self, nrm, ls):
        """Append this step's (grad-norm, loss). Call once per logged step, rank0."""
        self.buf.append((float(nrm), float(ls)))

    def seed_from_log(self, gen_log_path, before_step):
        """Pre-fill the rolling buffer from a gen_log on resume, so GPM-L doesn't reset to a
        cold short-memory window after a checkpoint restart (the deque isn't checkpointed —
        but the log IS the persistent history). Loads the (nrm, ls) of training lines with
        step < before_step, keeping the last w_long+2 of them: exactly what the live buffer
        would have held had the run never paused. No-ops cleanly on a fresh run (missing log,
        before_step<=1, or no parseable lines). Call once, right after construction, rank0.

        Returns the number of points seeded (0 if it no-op'd)."""
        try:
            if not gen_log_path or before_step is None or before_step <= 1:
                return 0
            import os as _os, re as _re
            if not _os.path.exists(gen_log_path):
                return 0
            # step / ls / nrm in the same order the trainer writes them on a training line.
            pat = _re.compile(r"st:\s*(\d+).*?ls:\s*([0-9.]+).*?nrm:\s*([0-9.]+)")
            rows = {}
            with open(gen_log_path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = pat.search(line)
                    if m:
                        s = int(m.group(1))
                        if s < before_step:
                            rows[s] = (float(m.group(3)), float(m.group(2)))  # (nrm, ls)
            if not rows:
                return 0
            # last w_long+2 steps in order -> exactly the deque's maxlen window
            tail = [rows[s] for s in sorted(rows)][-(self.wl + 2):]
            self.buf.clear()
            self.buf.extend(tail)
            return len(tail)
        except Exception:
            # Seeding is a best-effort warm-start; never let it break a resume.
            return 0

    @staticmethod
    def _spearman(a, b):
        n = len(a)
        if n < 4:
            return None
        def ranks(v):
            order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
            while i < n:
                j = i
                while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        ra, rb = ranks(a), ranks(b)
        ma, mb = sum(ra) / n, sum(rb) / n
        cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
        va = sum((x - ma) ** 2 for x in ra) ** 0.5
        vb = sum((x - mb) ** 2 for x in rb) ** 0.5
        return cov / (va * vb) if va > 0 and vb > 0 else 0.0

    @staticmethod
    def _resid(x):
        """Residual vs centered rolling-median over the WHOLE passed window (the window IS
        the locality here, since we pass exactly the last W points)."""
        import statistics as _st
        m = _st.median(x)
        return [v - m for v in x]

    def _gpm(self, W):
        # use the last W+1 points; dloss[i]=ls[i]-ls[i+1], aligned with nrm[i]
        pts = list(self.buf)[-(W + 1):]
        if len(pts) < 5:
            return None
        nr = [p[0] for p in pts]; lsv = [p[1] for p in pts]
        dloss = [lsv[i] - lsv[i + 1] for i in range(len(pts) - 1)]
        nr_al = nr[:len(dloss)]
        return self._spearman(self._resid(nr_al), self._resid(dloss))

    def status_tag(self):
        """' | gpm: +0.31/+0.25' (GPM-S/GPM-L) for the status line, or ' | gpm: pending' until
        there is enough history (<5 points). No trend arrow: trending is read on the Dashboard
        (it plots both curves); the status line stays compact. Format is byte-identical to the
        retrofit injector so historical and live runs parse the same (the Dashboard's numeric
        regex simply skips 'pending')."""
        gs, gl = self._gpm(self.ws), self._gpm(self.wl)
        if gs is None and gl is None:
            return " | gpm: pending"
        def f(v):
            return f"{v:+.2f}" if v is not None else " -- "
        return f" | gpm: {f(gs)}/{f(gl)}"


def get_row_center_s(step, settings):
    """Row-center warmup scalar s(t) in [0,1] at absolute global `step`. Pure
    function of (step + static config) -> resume-safe like get_zloss_alpha.

    s scales the TARGET-GAUGE schedule: mu_target = (1-s)*mu0, so s=0 leaves the
    gauge at its captured start value (projection is a no-op) and s=1 is fully
    centered (steady-state projection to mean->0). Returns:
      - 0.0 when row-centering is disabled
      - 1.0 when enabled with NO warmup (flat-bool / steady-state: always full)
      - the cosine/linear ramp 0->1 over [start, start+duration) when warmup on,
        1.0 at/after start+duration, 0.0 before start.
    cosine_up(u) = 0.5*(1 - cos(pi*u))."""
    cfg = _row_center_cfg(settings)
    if not cfg['enabled']:
        return 0.0
    warmup = cfg['warmup'] or {}
    if not warmup.get('enabled', False):
        return 1.0                            # steady-state: full projection
    start_step = int(warmup.get('start_step', 0))
    duration = int(warmup.get('duration_steps', 1))
    shape = warmup.get('shape', 'cosine')
    if step < start_step:
        return 0.0
    if step >= start_step + max(1, duration):
        return 1.0
    u = (step - start_step) / duration        # in [0, 1)
    if shape == 'linear':
        return u
    return 0.5 * (1.0 - math.cos(math.pi * u))


def _find_rowcenter_zloss_overlap(settings):
    """Return the first global step where BOTH z-loss alpha_eff > 0 AND
    row-center s > 0 (i.e. they're concurrently active), or None if disjoint.
    Step-active check (Guardrail 2): scans the union of the schedules' plausible
    active ranges rather than trusting the static flags."""
    z = getattr(settings, 'z_loss', None)
    cfg = _row_center_cfg(settings)
    if z is None or not cfg['enabled']:
        return None
    # Candidate step bounds from both schedules. z-loss can be active from its
    # warmup start (or step 0 if no warmup) through its warmdown end (or
    # unbounded). row-center s>0 from its warmup start (or step 0 if steady-state)
    # onward. We only need to find ANY overlap, so scan a bounded window that
    # covers both schedules' transition regions plus a margin.
    bounds = []
    wu = z.get('warmup') or {}
    wd = z.get('warmdown') or {}
    rcw = cfg['warmup'] or {}
    for blk in (wu, wd, rcw):
        if blk.get('enabled', False):
            bounds.append(int(blk.get('start_step', 0)))
            bounds.append(int(blk.get('start_step', 0)) + int(blk.get('duration_steps', 1)))
    if not bounds:
        # Both steady-state on (flat row_center + z-loss no schedule): they
        # overlap everywhere -> report step 0.
        if get_zloss_alpha(0, settings) > 0 and get_row_center_s(0, settings) > 0:
            return 0
        return None
    lo = max(0, min(bounds) - 2)
    hi = max(bounds) + 2
    # If z-loss has no warmdown it stays active indefinitely after warmup; if
    # row-center is steady-state (no warmup) it's active from 0. The bounded scan
    # over the transition region catches the staged-handoff case; extend hi a bit
    # past the row-center warmup end to be safe.
    for step in range(lo, hi + 1):
        if get_zloss_alpha(step, settings) > 0.0 and get_row_center_s(step, settings) > 0.0:
            return step
    return None


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

    Each waypoint's step must be a non-negative int and steps within a single
    schedule must be strictly increasing (duplicates would cause a divide-by-
    zero in interpolate_lr_mod's lerp). Weights must lie in [0, 1] — values
    outside that range are rejected because higher-level logic (SCS scaffold
    detection at λ >= 1.0, aux-head weighting into total_loss) assumes the
    normalised range.

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
        # Strict-monotonic step validation: duplicates would divide by zero
        # in interpolate_lr_mod; negative steps are nonsensical.
        for _i, (_s, _v) in enumerate(sched):
            if _s < 0:
                fatal_error(
                    f"auxiliary_heads.heads[layer={li}].weight has negative step {_s}"
                )
            if _i > 0 and _s == sched[_i - 1][0]:
                fatal_error(
                    f"auxiliary_heads.heads[layer={li}].weight has duplicate step {_s} "
                    f"(would divide by zero in interpolation)"
                )
        # Weight bounds: SCS scaffold detection and the trainer's `if w != 0`
        # gate both assume weights in [0, 1].
        for _s, _v in sched:
            if not (0.0 <= _v <= 1.0):
                fatal_error(
                    f"auxiliary_heads.heads[layer={li}].weight value {_v} at step {_s} "
                    f"outside [0, 1] — reject (use schedule weights in the normalised range)"
                )
        if li in schedules:
            fatal_error(f"auxiliary_heads.heads has duplicate entry for layer {li}")
        schedules[li] = sched
        layers.append(li)
    return sorted(layers), schedules


# ─────────────────────────────────────────────────────────────────────────────
# Scaffolded Cascading Supervision (SCS) helpers
# ─────────────────────────────────────────────────────────────────────────────
def deepest_active_tap(aux_head_schedules, step, threshold=1.0):
    """Return the deepest aux head layer whose schedule weight at `step` is
    >= threshold, or None if no head is currently at or above threshold.

    Used by the SCS dispatch (compute_inactive_layers=false) to decide whether
    to enter scaffold mode and at which depth to truncate the forward."""
    if not aux_head_schedules:
        return None
    deepest = None
    for li, sched in aux_head_schedules.items():
        if interpolate_lr_mod(sched, step) >= threshold:
            if deepest is None or li > deepest:
                deepest = li
    return deepest


def compute_scs_activation_events(aux_head_schedules, n_layers, threshold=1.0):
    """Scan the aux head schedules to determine the step at which each layer
    first becomes "active" under SCS, where:

      * Layers from 0 through the shallowest aux tap are always active from
        step 0 (the network's initial trainable region).
      * A compartment between two taps (prev_tap+1 .. next_tap) activates at
        the first step where the deepest tap at λ >= threshold reaches the
        deeper tap of the pair.
      * The "cascade-complete compartment" — layers from the deepest aux tap
        + 1 through n_layers - 1 — activates at the first step where no aux
        head is at λ >= threshold (this is also when the main LM head turns on).

    Returns dict[layer_idx -> activation_step]. Layers that never activate
    within the schedule window get marked as activating at max_step + 1.
    """
    sorted_layers = sorted(aux_head_schedules.keys())
    if not sorted_layers:
        return {i: 0 for i in range(n_layers)}

    max_step = max(
        max(s for s, _ in sched)
        for sched in aux_head_schedules.values()
    )

    activation = {}
    first_tap = sorted_layers[0]
    for li in range(first_tap + 1):
        activation[li] = 0

    prev_deepest = first_tap
    for step in range(0, max_step + 2):
        cur = deepest_active_tap(aux_head_schedules, step, threshold=threshold)
        if cur is None:
            # Cascade complete — main head + remaining tail come online here.
            for li in range(prev_deepest + 1, n_layers):
                activation.setdefault(li, step)
            break
        if cur > prev_deepest:
            for li in range(prev_deepest + 1, cur + 1):
                activation.setdefault(li, step)
            prev_deepest = cur

    for li in range(n_layers):
        activation.setdefault(li, max_step + 1)
    return activation


def scs_compartment_lr_scale(activation_step, warmup_steps, init_mult, step):
    """Per-step LR multiplier for a layer in a freshly-activated compartment.

    Special case: activation_step == 0 (the initial always-active compartment
        — layers 0..first_tap that train from the very start of the run) is
        not a "freshly activated" compartment; it returns 1.0 unconditionally
        so the body trains at full LR from step 0.
    Before activation_step: 0.0 (frozen — when paired with scaffold mode this
        means the layer's params don't get touched by WD either, since the
        optimizer multiplies WD by effective_lr = lr * lr_scale).
    At activation_step .. activation_step + warmup_steps: linear ramp from
        init_mult to 1.0.
    After activation_step + warmup_steps: 1.0 (full LR).
    """
    if activation_step == 0:
        # Initial compartment — trained from step 0, no soft start.
        return 1.0
    if step < activation_step:
        return 0.0
    if warmup_steps <= 0 or step >= activation_step + warmup_steps:
        return 1.0
    t = (step - activation_step) / warmup_steps
    return init_mult + (1.0 - init_mult) * t


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

    # Coverage guard (silent-failure protection). When weight_decay is a RULES list, the base WD is
    # forced to 0.0, so ANY non-norm param NOT matched by a rule silently gets WD=0 — which is never
    # intentional (e.g. a lone [all, ...] body rule leaves emb+head at 0). Require full coverage:
    # every non-norm param must be matched by SOME rule. An explicit [emb, 0.0] counts (it's in
    # param_wds) — only the SILENT, rule-less zero is rejected. Norms are intentionally WD=0, skipped.
    uncovered = [n for n, p in model.named_parameters()
                 if not _is_norm(n) and id(p) not in param_wds]
    if uncovered:
        buckets = {}
        for n in uncovered:
            if 'tok_embeddings' in n:
                b = "emb (add a [emb, <wd>] rule)"
            elif n.startswith('output.') or n.endswith('output.weight'):
                b = "head (add a [out, <wd>] rule)"
            elif 'layers.' in n:
                b = "body (add [all, <wd>] or a layer-range rule)"
            else:
                b = "other"
            buckets[b] = buckets.get(b, 0) + 1
        summary = "; ".join(f"{b}: {c}" for b, c in buckets.items())
        fatal_error(
            f"weight_decay RULES leave {len(uncovered)} non-norm param(s) uncovered, so they would "
            f"SILENTLY get WD=0 (the rules-mode base WD is 0.0) — never intentional. Cover them -> "
            f"{summary}. To deliberately zero a group, set it explicitly (e.g. [emb, 0.0]). "
            f"First uncovered: {uncovered[:4]}")

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


def _clip_group_of(name):
    """Bucket a param into a coarse group for clip telemetry (Probe B). Mirrors the
    Muon-vs-Adam split that matters for the clip-throttle question: the Muon BODY is
    clip-invariant (NS discards magnitude), the magnitude-sensitive Adam groups are not.
      'body'  — Muon matrices (attn/ffn 2D weights)        [clip-invariant]
      'head'  — output projection                          [magnitude-sensitive]
      'emb'   — token embeddings                           [magnitude-sensitive]
      'other' — norms / router / GDN-small / biases        [magnitude-sensitive]

    NOTE: per-submodule torch.compile (`_apply_per_submodule_compile`) inserts an
    `_orig_mod` segment into param paths — e.g. `layers.0.attention.wq.weight` becomes
    `layers.0.attention._orig_mod.wq.weight`. We strip ALL `_orig_mod.` segments before
    matching, else every compiled body matrix mis-buckets as 'other' (the gn_body=0 bug,
    2026-06-25). The body grads are present and correctly trained — this only affected the
    telemetry's bucketing, not the clip/nrm math (which sum every param regardless of group)."""
    n = name.replace('_orig_mod.', '')   # undo per-submodule compile name mangling
    if n.endswith('.weight') and any(s in n for s in (
            'attention.wq', 'attention.wk', 'attention.wv', 'attention.wo',
            'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3',
            'gdn_attn.q_proj', 'gdn_attn.k_proj', 'gdn_attn.v_proj', 'gdn_attn.o_proj',
            'shared_experts.w1', 'shared_experts.w2', 'shared_experts.w3', 'g_proj')):
        return 'body'
    if n.startswith('output.') or n.endswith('output.weight'):
        return 'head'
    if 'tok_embeddings' in n:
        return 'emb'
    return 'other'


def _clip_grad_norm_mixed_mesh(model, max_norm, norm_type=2.0, group_telemetry=False):
    """clip_grad_norm_ that works with params on different DTensor meshes.

    Standard clip_grad_norm_ uses torch.stack on per-param norms, which fails
    when params are DTensors on different meshes (e.g. dp_mesh vs edp_mesh in EP).

    Instead: compute local norm² per param, all-reduce across all ranks, then clip.
    This is correct because:
      - FSDP sharded params: each rank has 1/N of the grad, sum of partial norms² = full norm²
      - EP expert params: each rank has unique experts, sum gives total expert norm²

    Perf: norm² is accumulated ON-GPU (no per-param .item() sync); a single all-reduce
    handles the total (+ 4 group sums when group_telemetry=True — same #all-reduces).

    Returns total_norm (float). When group_telemetry=True, returns
    (total_norm, {'body':n, 'head':n, 'emb':n, 'other':n, 'clip_coef':c}) — per-group
    global grad norms (Probe B), computed in the SAME pass at negligible extra cost.
    """
    from torch.distributed.tensor import DTensor

    grads = []
    dev = None
    if group_telemetry:
        # 5 on-GPU accumulators: [total, body, head, emb, other] local norm²
        acc = None
        _gidx = {'body': 1, 'head': 2, 'emb': 3, 'other': 4}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            grads.append(p.grad)
            g = p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
            if acc is None:
                dev = g.device
                acc = torch.zeros(5, device=dev, dtype=torch.float64)
            nsq = torch.linalg.vector_norm(g, norm_type).double() ** norm_type
            acc[0] += nsq
            acc[_gidx[_clip_group_of(name)]] += nsq
        if not grads:
            return 0.0, {'body': 0.0, 'head': 0.0, 'emb': 0.0, 'other': 0.0, 'clip_coef': 1.0}
        dist.all_reduce(acc)
        vals = acc.tolist()
        total_norm = vals[0] ** (1.0 / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        groups = {'body': vals[1] ** (1.0 / norm_type), 'head': vals[2] ** (1.0 / norm_type),
                  'emb': vals[3] ** (1.0 / norm_type), 'other': vals[4] ** (1.0 / norm_type),
                  'clip_coef': min(1.0, clip_coef)}
        return total_norm, groups

    # fast path (telemetry off): single on-GPU accumulator, one .item() sync total
    acc = None
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad)
        g = p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
        if acc is None:
            acc = torch.zeros(1, device=g.device, dtype=torch.float64)
        acc += torch.linalg.vector_norm(g, norm_type).double() ** norm_type

    if not grads:
        return 0.0

    dist.all_reduce(acc)
    total_norm = acc.item() ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

    return total_norm


def _global_param_norms(params, ddp, norm_type=2.0):
    """Global (FSDP-reduced) weight-norm and grad-norm over the given params.

    Uses the SAME local-norm**2 -> all_reduce -> sqrt method as
    _clip_grad_norm_mixed_mesh, so the returned head grad-norm is reduced
    consistently with the total grad-norm it will be divided by (otherwise a
    local-shard numerator over a global denominator is meaningless under FSDP).
    Call AFTER backward and BEFORE clipping so grads are unscaled.

    Returns (weight_norm, grad_norm) as python floats.
    """
    from torch.distributed.tensor import DTensor

    w_local_sq = 0.0
    g_local_sq = 0.0
    dev = None
    for p in params:
        if p is None:
            continue
        pw = p._local_tensor if isinstance(p, DTensor) else p
        dev = pw.device
        w_local_sq += torch.linalg.vector_norm(pw, norm_type).item() ** norm_type
        if p.grad is not None:
            pg = p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
            g_local_sq += torch.linalg.vector_norm(pg, norm_type).item() ** norm_type

    if dev is None:
        return 0.0, 0.0
    if ddp:
        t = torch.tensor([w_local_sq, g_local_sq], device=dev)
        dist.all_reduce(t)
        w_local_sq, g_local_sq = t[0].item(), t[1].item()
    return w_local_sq ** (1.0 / norm_type), g_local_sq ** (1.0 / norm_type)


def _head_param(raw_model):
    """The main LM readout head param = raw_model.output.weight, identified by
    OBJECT (not name): under weight tying output.weight IS tok_embeddings.weight,
    which named_parameters() reports only under the embedding name — so a
    name-based match would miss it. Returns the param (or None)."""
    out = getattr(raw_model, "output", None)
    return getattr(out, "weight", None) if out is not None else None


def _row_center_head_step(model, optimizer, want_exp_avg=True):
    """Project the CE-invisible common-mode gauge out of the LM head in place
    (and out of the Adam first moment, so it can't regrow). Function-preserving:
    subtracts the same scalar h.mu from every vocab logit -> CE/softmax/sampling
    unchanged. Call AFTER optimizer.step(), under eager (the projection does an
    all-reduce + a bf16 stochastic-rounding randint, both of which dislike being
    traced — same reason the 16-bit Adam step is kept out of compile).

    Returns the telemetry dict from row_center_head_ (or None if no head found).
    `want_exp_avg=False` projects only the weight (used for the pre-first-forward
    projection on init/resume, where stripping momentum isn't needed yet)."""
    from row_center import row_center_head_
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    p = _head_param(raw)
    if p is None:
        return None
    exp_avg = None
    if want_exp_avg:
        st = optimizer.state.get(p, {})
        exp_avg = st.get("exp_avg", None)  # None before the head's first step
    return row_center_head_(p, exp_avg=exp_avg, vocab_dim=0)


def _row_center_capture_gauge(model, optimizer):
    """Capture the warmup start gauges (mu0, mbar0) from the live head + Adam
    first moment. Returns the capture_gauge dict (fp32 CPU tensors), or None if
    no head. Called once at warmup start_step (then checkpoint-persisted)."""
    from row_center import capture_gauge
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    p = _head_param(raw)
    if p is None:
        return None
    exp_avg = optimizer.state.get(p, {}).get("exp_avg", None)
    return capture_gauge(p, exp_avg=exp_avg, vocab_dim=0)


def _row_center_warmup_step(model, optimizer, s, mu0, mbar0):
    """Per-step TARGET-GAUGE warmup projection (Stage 2). Pins the stored gauge to
    (1-s)*mu0 (and exp_avg to (1-s)*mbar0). mu0/mbar0 are the start-of-warmup
    captures (NOT recomputed). Returns telemetry or None if no head."""
    from row_center import row_center_head_warmup_
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    p = _head_param(raw)
    if p is None:
        return None
    exp_avg = optimizer.state.get(p, {}).get("exp_avg", None)
    return row_center_head_warmup_(p, s, mu0, exp_avg=exp_avg, mbar0=mbar0, vocab_dim=0)


def _centered_geometry_step(model):
    """Centered head-geometry health metrics (Item B) at val cadence. Returns the
    centered_geometry dict (||W_c||, s1_c, spec_conc_c, eff_rank, small-sigma) or
    None if no head. centered_geometry always subtracts the current row-mean, so it
    reads true at any ramp position (Nexus #163). Cheap [D,D] eigh — val cadence
    only, not per step."""
    from row_center import centered_geometry
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    p = _head_param(raw)
    if p is None:
        return None
    return centered_geometry(p, vocab_dim=0)


def _logz_c_at_val(model, tokens, max_tok=4096, tok_chunk=1024):
    """DATA-SIDE centered logZ_c on a small batch, at VAL CADENCE only.

    logZ_c = mean_token logsumexp(h @ (W - mu)^T) — the centered log-partition.
    This is NOT free in the training forward: CCE fuses the loss and never
    materializes the [N,V] logits, so we re-run a bounded [<=max_tok, V] matmul +
    logsumexp here (once per val_step, capped tokens) rather than per step. Logged
    REGARDLESS of z-loss (z-loss only ever surfaced RAW logZ, and only when on).

    Captures the post-final-norm hidden via a forward hook on model.norm, then
    centers the head (W - mu, mu the global row-mean) and reduces in token chunks
    to bound VRAM. Returns logZ_c mean (float) or None. Caller runs at val cadence;
    all ranks (no collective here — head/h are full on each rank in this codebase's
    eval path; if vocab-sharded, this would need the same all-reduce as the probe,
    but the live head is replicated for the metric)."""
    import torch as _t
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    p = _head_param(raw)
    if p is None:
        return None
    cap = {}

    def _hook(_m, _inp, out):
        cap['h'] = out.detach()

    # Bound the token count: take a single <=max_tok window (no need for the full
    # batch — this is a trend metric, not the exact full-batch mean).
    toks = tokens[:, :max_tok] if tokens.size(1) > max_tok else tokens
    handle = raw.norm.register_forward_hook(_hook)
    try:
        with _t.no_grad():
            if _t.cuda.is_available():
                with _t.autocast("cuda", dtype=_t.bfloat16):
                    model(toks)
            else:
                model(toks)
    finally:
        handle.remove()
    if 'h' not in cap:
        return None
    h = cap['h'].reshape(-1, cap['h'].size(-1)).float()
    W = p.detach().float() if not hasattr(p, '_local_tensor') else p._local_tensor.detach().float()
    mu = W.mean(dim=0)
    Wc = W - mu.unsqueeze(0)
    N = h.shape[0]
    acc = 0.0
    cnt = 0
    for s in range(0, N, tok_chunk):
        hc = h[s:s + tok_chunk].to(W.device)
        logits_c = hc @ Wc.t()                      # [c, V] centered
        acc += _t.logsumexp(logits_c, dim=-1).sum().item()
        cnt += hc.shape[0]
    return acc / max(1, cnt)


# -------------------------- Training Loop --------------------------
def train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_local_rank, ddp_world_size, start_step,
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule,
        diagnostics: LayerDiagnostics = None,
        truncator: ProgressiveTailTruncation = None,
        wd_overrides: dict = None,
        lr_scale_overrides: dict = None,
        awd=None,
        body_lr_ctrl=None,
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
    # z-loss gate (single source of truth for the micro-step fold + log emit).
    # settings.z_loss is None exactly when disabled (normalized in Settings).
    zloss_enabled = getattr(settings, 'z_loss', None) is not None

    # row-center gate (gauge subtraction on the LM head, post-step projection).
    # Accepts flat-bool (steady-state) or nested-dict (staged warmup) config.
    _rc_cfg = _row_center_cfg(settings)
    row_center_enabled = _rc_cfg['enabled']
    _rc_warmup = _rc_cfg['warmup'] or {}
    row_center_warmup_on = bool(_rc_warmup.get('enabled', False))
    row_center_warmup_start = int(_rc_warmup.get('start_step', 0)) if row_center_warmup_on else None
    # head_gauge_projection gate (dn4): in-optimizer applied-update gauge projection.
    # Independent of row_center_head (which is the legacy, mutually-exclusive path).
    _hg_cfg = _head_gauge_cfg(settings)
    head_gauge_enabled = _hg_cfg['enabled']
    head_gauge_init_rc = _hg_cfg['init_row_center']
    # Warmup target gauges (mu0/mbar0), captured once at warmup start and
    # checkpoint-persisted (Guardrail 1). Restored from checkpoint if mid-warmup.
    # Held as fp32 tensors; None until captured. _rc_captured guards single capture.
    rc_warmup_mu0 = getattr(train_loop, '_rc_restore_mu0', None)
    rc_warmup_mbar0 = getattr(train_loop, '_rc_restore_mbar0', None)
    rc_warmup_captured = rc_warmup_mu0 is not None

    # Guardrail 5 (advisory): transition health guard. Loud WARNING log ONLY — no
    # auto-pause / auto-checkpoint / auto-intervene (Josef keeps manual control).
    # Off unless transition_health_guard: true in config. Thresholds from the hard
    # branch's nrm=5.61 event: nrm>5 once, nrm>3 for N repeated steps, eff_rank_c<7,
    # spec_conc_c>0.45.
    health_guard_on = bool(getattr(settings, 'transition_health_guard', False))
    # Suppress the grad-norm guard for the first N steps (config:
    # health_guard_warmup_steps, default 100). LR warmup => high nrm is expected and
    # noisy at the start of any from-scratch / fresh-resume run; the guard is for the
    # steady state, not the ramp.
    health_guard_warmup_steps = int(getattr(settings, 'health_guard_warmup_steps', 100))
    _hg_nrm_run = 0   # consecutive steps with nrm > 3

    # Gradient Productivity Metric (GPM) on the status line — opt-in via track_gpm: true.
    # Live "are big gradients productive right now?" read (GPM-S/GPM-L; gap = rising/falling).
    _gpm_cfg = getattr(settings, 'track_gpm', False)
    gpm_tracker = None
    if _gpm_cfg:
        _gpm_ws = int(getattr(settings, 'gpm_window_short', 15))
        _gpm_wl = int(getattr(settings, 'gpm_window_long', 101))
        gpm_tracker = GPMTracker(w_short=_gpm_ws, w_long=_gpm_wl)
        # Warm-start the rolling buffer from the gen_log on resume so GPM-L doesn't reset to a
        # cold short-memory window after a checkpoint restart (no "seam" at the resume point).
        # The deque isn't checkpointed, but the gen_log holds the full (nrm, ls) history.
        # IMPORTANT: settings.gen_log_file is a BARE filename ('gen_log.txt'); the logger writes
        # it under its logdir (set to settings.nas_path via logger._instance.set_logdir(), L4906)
        # as nas_path/gen_log.txt (logger.py L299). seed_from_log needs that resolved path —
        # passing the bare name made os.path.exists() check the trainer CWD and silently no-op
        # (priming bug v1, 2026-06-25). Note: `logger` here is the MODULE; get_dir() lives on
        # logger._instance, so resolve via settings.nas_path directly (the value the logdir was
        # set from) rather than logger.get_dir() — module has no get_dir (priming bug v2).
        if start_step > 1:
            _glf = getattr(settings, 'gen_log_file', None)
            _gpaths = []
            if _glf:
                _nasp = getattr(settings, 'nas_path', None)
                if _nasp:
                    _gpaths.append(os.path.join(_nasp, _glf))
                try:
                    _gpaths.append(os.path.join(logger._instance.get_dir(), _glf))
                except Exception:
                    pass
                _gpaths.append(_glf)  # fallback: bare/relative (covers absolute paths too)
            _seeded = 0
            for _gp in _gpaths:
                _seeded = gpm_tracker.seed_from_log(_gp, start_step)
                if _seeded:
                    break
            if ddp_rank == 0:
                if _seeded:
                    logger.print_and_log(f"  ] GPM: warm-started buffer with {_seeded} pre-resume "
                                         f"points from gen_log (no restart seam).")
                else:
                    logger.print_and_log(f"  ] GPM: no warm-start (gen_log not found at "
                                         f"{_gpaths or '[none]'}; cold start).")

    # Probe B — per-group clip telemetry (opt-in via track_clip_groups: true). Logs the
    # global grad-norm split by group + the clip coefficient, so we can SEE who the global
    # clip is firing on (Muon body is clip-invariant; Adam groups are magnitude-sensitive —
    # see docs/PROBE_A_clip_replay_RESULTS.md). Folded into the existing clip pass → ~free.
    _clip_groups_cfg = bool(getattr(settings, 'track_clip_groups', False))

    # Tangent-projection STRENGTH f (partial projection): f=1 strips all of the radial Muon-update
    # component (flat ‖W‖, original behavior); f<1 leaves (1-f) of it, so ‖W‖ grows at (1-f) of its
    # natural rate; f=0 = no projection. Scalar (constant) or [[step, val], ...] schedule (linear
    # interp). Written into the Muon body groups' 'tangent_project_strength' each step below; the
    # f-sweep dial for the "is bounded growth better than flat ‖W‖" experiment. Default 1.0 =
    # existing configs unchanged.
    _tp_on = bool(getattr(settings, 'tangent_project', False))
    _tps_cfg = getattr(settings, 'tangent_project_strength', 1.0)
    _tps_is_sched = isinstance(_tps_cfg, list)

    # Body relative-step pdr = ||dW||/||W|| (effective-LR metric for the tangent-projection
    # annealing experiment, Math Brief #6). Emitted as its own line on diagnostics/val steps
    # (snapshot cadence — NOT per step). _body_param_ids_for_pdr lets the pdr line report the
    # REAL applied body-LR mult (read from lr_scale_overrides for a body param). Body = attn/ffn
    # Muon matrices, matching _clip_group_of (handles torch.compile's _orig_mod naming).
    _body_param_ids_for_pdr = {
        id(p) for n, p in (model._orig_mod if hasattr(model, '_orig_mod') else model).named_parameters()
        if _clip_group_of(n) == 'body'
    }

    # FFN-only pdr controller (kv3, docs/KV3_CONTROLLER_DESIGN.md): the body subset it
    # actuates = the dense feed_forward Muon matrices (w1/w2/w3). Built like
    # _body_param_ids_for_pdr but FFN-only; strips _orig_mod for torch.compile naming. The
    # controller object (body_lr_ctrl) is constructed in main() and threaded in; here we
    # resolve which params its held multiplier is written to. Inert when disabled.
    # id -> NAME bridge: the resume-stable key for the shadow controller's per-matrix shadow norm S.
    # The name is the _orig_mod-stripped param name — exactly the key load_state_dict restores S under,
    # so S survives a process restart (id() does not). The trainer rebuilds this map every run.
    _ffn_id_to_name = {
        id(p): n.replace('_orig_mod.', '')
        for n, p in (model._orig_mod if hasattr(model, '_orig_mod') else model).named_parameters()
        if (lambda _n: _n.endswith('.weight') and any(
            s in _n for s in ('feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'))
            )(n.replace('_orig_mod.', ''))
    }
    _ffn_param_ids_for_ctrl = set(_ffn_id_to_name.keys())
    # Attn id-set for the decomposed [body-pdr] readout = body minus ffn (for dense layers this is
    # exactly attention.wq/wk/wv/wo). Lets the line report attn and ffn lr_mult SEPARATELY now that
    # they can be on different schedules (kv2 attn-ramp vs ffn-glide; kv3 ffn-controller vs attn=1.0).
    _attn_param_ids_for_pdr = _body_param_ids_for_pdr - _ffn_param_ids_for_ctrl
    # acts_on_attn: the controller broadcasts its (FFN-computed) m to the attention matrices too, so the
    # attn pdr rides the same free-growth glide as the FFN (recovers the picket-hyg attn<->ffn lock-step
    # that projecting-but-not-compensating attn breaks). The m is applied to this UNION; the radial-budget
    # WD λ is NOT (it stays FFN-only at _ffn_param_ids_for_ctrl) so attn keeps its flat base WD.
    _acts_on_attn = bool(getattr(body_lr_ctrl, 'acts_on_attn', False)) if body_lr_ctrl is not None else False
    _pdr_m_ids = (_ffn_param_ids_for_ctrl | _attn_param_ids_for_pdr) if _acts_on_attn \
        else _ffn_param_ids_for_ctrl

    # Controller is DENSE-FFN-only: it ACTUATES feed_forward.w1/w2/w3 but OBSERVES the diagnostics
    # FFN-aggregate pdr, which under MoE is the EXPERT params — a measure/actuate mismatch (and the
    # actuated set would be empty). Fail loudly rather than silently steer with no authority.
    if body_lr_ctrl is not None and body_lr_ctrl.enabled:
        if getattr(model_cfg, 'moe_enabled', False):
            raise RuntimeError(
                "ffn_pdr_controller is dense-FFN-only (actuates feed_forward.w1/w2/w3) but the model "
                "is MoE — its observed FFN pdr would be expert params it cannot actuate. Disable the "
                "controller or run a dense-FFN config.")
        if not _ffn_param_ids_for_ctrl:
            raise RuntimeError(
                "ffn_pdr_controller enabled but found 0 dense feed_forward.w1/w2/w3 params to "
                "actuate — the controller would have no authority. Check the model topology.")
        # Shadow modes need an FSDP/DTensor body: the single-device Muon path (SingelDeviceWork) has
        # no tangent-projection block, so radial_stats is never produced and the controller would have
        # no telemetry to steer on. Fail loudly rather than silently no-op.
        if getattr(body_lr_ctrl, 'ref_mode', '') in ('auto_shadow_growth', 'auto_shadow_partial'):
            from torch.distributed.tensor import DTensor as _DTcheck
            _raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            _bad = [n for n, p in _raw_m.named_parameters()
                    if id(p) in _ffn_param_ids_for_ctrl and not isinstance(p, _DTcheck)]
            if _bad:
                raise RuntimeError(
                    f"ffn_pdr_controller shadow mode requires an FSDP/DTensor body, but {len(_bad)} FFN "
                    "matrices are plain tensors — the single-device Muon path has no tangent-projection "
                    "block, so no radial telemetry is produced. Run under FSDP2 (world_size>1).")
        if ddp_rank == 0:
            _attn_n = len(_pdr_m_ids) - len(_ffn_param_ids_for_ctrl)
            logger.print_and_log(
                f"  FFN pdr controller: actuating {len(_ffn_param_ids_for_ctrl)} dense FFN matrices"
                + (f" + {_attn_n} attn matrices (acts_on_attn: same m, FFN-only WD)" if _acts_on_attn else ""))

    # Shadow-norm controller (auto_shadow_growth/partial): per-step radial accumulators that observe()
    # consumes at diagnostic cadence. _shadow_dR holds Σ ΔR_free per matrix NAME; _shadow_eta holds Σ η
    # (the window's WD-shrink driver for S); _shadow_R / _shadow_gamma hold the latest ‖W‖ / γ per matrix.
    _shadow_ctrl_on = (body_lr_ctrl is not None and body_lr_ctrl.enabled
                       and getattr(body_lr_ctrl, 'ref_mode', '')
                       in ('auto_shadow_growth', 'auto_shadow_partial'))
    _shadow_dR: dict = {}        # name -> Σ ΔR_free since last observe
    _shadow_R: dict = {}         # name -> latest ‖W‖
    _shadow_gamma: dict = {}     # name -> latest γ
    _shadow_eta = {'sum': 0.0}   # Σ scheduled_lr since last observe (dict for scope-safe mutation)

    # ── SCS (Scaffolded Cascading Supervision) per-loop state ───────────────
    # Active when `auxiliary_heads.compute_inactive_layers: false`. The trainer
    # then truncates forward + backward at the deepest tap currently at
    # λ >= 1.0 (skipping the main LM head) and ramps each freshly-activated
    # compartment's LR from new_layer_lr_multiplier up to 1.0 over
    # new_layer_warmup_steps. Inactive layers (and output.weight during
    # scaffold) get lr_scale=0 so WD doesn't quietly decay them either.
    _ah_cfg = getattr(settings, 'auxiliary_heads', None) or {}
    scs_enabled = aux_heads_enabled and not _ah_cfg.get('compute_inactive_layers', True)
    scs_warmup_steps = int(_ah_cfg.get('new_layer_warmup_steps', 0)) if scs_enabled else 0
    scs_init_mult = float(_ah_cfg.get('new_layer_lr_multiplier', 1.0)) if scs_enabled else 1.0
    scs_activation_steps = (
        compute_scs_activation_events(aux_head_schedules, model_cfg.n_layers)
        if scs_enabled else {}
    )
    # The cascade-complete step is the activation step of the deepest layer
    # (the last compartment to come online). Used to drive the output-head
    # warmup ramp post-cascade. 0 when SCS disabled.
    _cascade_complete_step = (
        scs_activation_steps.get(model_cfg.n_layers - 1, 0)
        if scs_enabled else 0
    )
    # Precomputed compartment ranges for the per-step scs_lr: debug line —
    # list of (start_layer, end_layer, activation_step). Consecutive layers
    # with the same activation step group into a single compartment.
    scs_compartment_ranges = []
    if scs_enabled:
        _prev_act = None
        _range_start = 0
        for _li in range(model_cfg.n_layers):
            _st = scs_activation_steps[_li]
            if _st != _prev_act:
                if _prev_act is not None:
                    scs_compartment_ranges.append((_range_start, _li - 1, _prev_act))
                _range_start = _li
                _prev_act = _st
        if _prev_act is not None:
            scs_compartment_ranges.append(
                (_range_start, model_cfg.n_layers - 1, _prev_act)
            )
    # ── SCS compatibility guards ────────────────────────────────────────────
    # The SCS freeze relies on the optimiser scaling WD by lr_scale (lr_mods
    # side-dict). NorMuon (FSDP2_MUON_FAMILY) is wired for this; the
    # standalone Adam fallback at the lr_mods application site broadcasts a
    # single param's scale to the whole group and silently freezes
    # everything (or scales by the wrong amount). Fail fast.
    if scs_enabled and settings.optimizer_type not in FSDP2_MUON_FAMILY:
        fatal_error(
            f"SCS (compute_inactive_layers=false) requires an optimizer in "
            f"FSDP2_MUON_FAMILY (per-param lr_scale via side-dict). Got "
            f"optimizer_type='{settings.optimizer_type}'. Either pick a "
            f"normuon_fsdp2-family optimizer or set "
            f"auxiliary_heads.compute_inactive_layers: true."
        )
    # MuonSphere does a spectral retraction (param.mul_(scale_factor)) BEFORE
    # the lr_scale lookup, so SCS's "freeze via lr_scale=0" can't actually
    # freeze MuonSphere-grouped params — they keep getting rescaled to stay
    # on the spectral sphere every step. Refuse the combination.
    if scs_enabled and settings.optimizer_type in {"muonsphere_fsdp2", "normuon_sphere_fsdp2"}:
        fatal_error(
            f"SCS (compute_inactive_layers=false) is incompatible with "
            f"MuonSphere optimizers ('muonsphere_fsdp2', 'normuon_sphere_fsdp2'): "
            f"spectral retraction bypasses the lr_scale freeze. Use a non-Sphere "
            f"variant (normuon_fsdp2 / muon_fsdp2) for SCS runs."
        )
    # When weight-tying is on, output.weight IS tok_embeddings.weight (the
    # same Parameter object). Freezing output.weight during scaffold would
    # also freeze tok_embeddings, breaking the "tok_embeddings is always
    # active" invariant. Refuse the combination explicitly.
    if scs_enabled and getattr(settings, 'tie_word_embeddings', True):
        fatal_error(
            "SCS (compute_inactive_layers=false) is incompatible with "
            "tie_word_embeddings=true: the shared Parameter would be frozen "
            "(and decayed!) during scaffold along with the LM head. Set "
            "tie_word_embeddings: false to use SCS."
        )
    # SCS + resume_training is plausible (resume a baseline checkpoint with
    # SCS enabled to scaffold from a pre-trained starting point) but the
    # tail layers and main head carry their existing trained weights /
    # optimizer state — scaffold won't touch them, but it also won't reset
    # them. Make the operator aware so a "scaffold from a fresh start"
    # expectation isn't silently violated.
    if scs_enabled and getattr(settings, 'resume_training', False) and ddp_rank == 0:
        logger.print_and_log(
            "[SCS] NOTE: resume_training=true with SCS enabled. Inactive tail "
            "layers + main head are preserved at their checkpointed values "
            "during scaffold (no fresh init). If you intended scaffold to "
            "start from a fresh tail, reset those weights manually."
        )
    # SCS is meaningless if scaffold isn't on at step 0 — the schedule fires
    # cascade-complete immediately, every layer gets activation_step=0
    # (treated as always-active), and SCS is silently a no-op. Catch this
    # misconfig at startup rather than letting the user run a 200k-step
    # baseline thinking they got SCS.
    if scs_enabled and deepest_active_tap(aux_head_schedules, 0) is None:
        fatal_error(
            "SCS (compute_inactive_layers=false) requires at least one aux "
            "head at lambda >= 1.0 at step 0, but no head's schedule reaches "
            "that threshold at step 0. Either set the shallowest head's "
            "weight schedule to start at 1.0, or disable SCS."
        )
    # Map layer_idx -> list of Parameter objects in that layer (for fast
    # per-step lr_scale_overrides updates). Also collect output-head params
    # separately so we can freeze them via the same hook during scaffold mode.
    scs_layer_params = {i: [] for i in range(model_cfg.n_layers)}
    scs_output_params = []
    # Aux-head params per tap layer. Aux heads above the active range during
    # scaffold get no forward (no grad), but their WD still applies unless
    # we freeze them via lr_scale=0. Without this they silently decay across
    # the long scaffolding phases.
    scs_aux_head_params = {li: [] for li in aux_head_schedules} if scs_enabled else {}
    # Set of param ids that lr_mods manages — SCS must NOT overwrite their
    # lr_scale post-cascade (e.g. output_lr_batch_adjust auto-injects an
    # [out, schedule] entry; SCS clobbering it to 1.0 every step would
    # silently disable the output-head LR scaling).
    _lr_mod_managed_param_ids = (
        {id(p) for p, _ in (lr_mod_entries or [])}
        if scs_enabled else set()
    )
    if scs_enabled:
        import re as _re
        _raw_for_scan = model._orig_mod if hasattr(model, '_orig_mod') else model
        _layer_re = _re.compile(r'layers\.(\d+)\.')
        _aux_re = _re.compile(r'aux_heads\.(\d+)\.')
        for _name, _p in _raw_for_scan.named_parameters():
            _m = _layer_re.match(_name)
            if _m:
                scs_layer_params[int(_m.group(1))].append(_p)
                continue
            _ma = _aux_re.match(_name)
            if _ma:
                _aux_li = int(_ma.group(1))
                if _aux_li in scs_aux_head_params:
                    scs_aux_head_params[_aux_li].append(_p)
                continue
            if _name.startswith('output.') or _name == 'norm.weight':
                # Main LM head Linear weight AND final RMSNorm — both should
                # freeze during scaffold (the scaffold path bypasses them).
                scs_output_params.append(_p)
        if ddp_rank == 0:
            _starts_at = sorted({s for s in scs_activation_steps.values()})
            logger.print_and_log(
                f"SCS enabled: activation events @ steps {_starts_at}, "
                f"warmup {scs_warmup_steps} steps from init_mult={scs_init_mult}"
            )

    # SCS owns body-layer lr_scale during scaffold and warmup phases. Any
    # lr_mods entry targeting body params would be silently overridden
    # whenever SCS asserts a scale != 1.0. Surface this at startup so the
    # user knows their lr_mods rule is muted during those phases. Must run
    # AFTER scs_layer_params is populated (the named-parameters scan above).
    if scs_enabled and ddp_rank == 0:
        _lr_mod_in_body = []
        if lr_mod_entries:
            _body_param_ids = {
                id(p) for li in range(model_cfg.n_layers) for p in scs_layer_params[li]
            }
            _lr_mod_in_body = [p for p, _ in lr_mod_entries if id(p) in _body_param_ids]
        if _lr_mod_in_body:
            logger.print_and_log(
                f"[SCS] WARNING: {len(_lr_mod_in_body)} body-layer lr_mod entries "
                f"are present. SCS owns body-layer lr_scale during scaffold and "
                f"warmup phases — those lr_mod entries will be silently overridden "
                f"in those phases (active only post-cascade for params whose "
                f"current SCS scale is 1.0)."
            )

    # Row-center the head BEFORE the first forward pass (incl. the baseline
    # validation below). Project weight only here — the Adam 1st moment is empty
    # before the head's first step, and is handled per-step thereafter.
    # Function-preserving, so this never perturbs the baseline val number.
    #
    # REQ 1 (load-bearing): when warmup is enabled, SUPPRESS this hard projection.
    # Under warmup the SCHEDULE is the sole path to centered — hard-projecting here
    # would wipe the gauge cold (the exact optimizer shock the staged transition
    # exists to avoid) and then "warm up" from an already-centered head, defeating
    # the redo. The per-step warmup projection (target-gauge, starting at s=0 =
    # no-op) takes over at start_step.
    if row_center_enabled and not row_center_warmup_on:
        _rc0 = _row_center_head_step(model, optimizer, want_exp_avg=False)
        if _rc0 is not None and ddp_rank == 0:
            logger.print_and_log(
                f"[row-center] pre-first-forward projection: "
                f"||mu(W)|| {_rc0['mu_w_pre']:.4g} -> {_rc0['mu_w_post']:.2e} "
                f"(proj_ratio {_rc0['proj_ratio']:.4f})"
            )
    elif row_center_enabled and row_center_warmup_on and ddp_rank == 0:
        logger.print_and_log(
            f"[row-center] warmup enabled (start {row_center_warmup_start}, "
            f"{int(_rc_warmup.get('duration_steps', 0))} steps, "
            f"{_rc_warmup.get('shape', 'cosine')}) — hard pre-forward projection "
            f"SUPPRESSED; schedule is the sole path to centered."
        )

    # dn4 head-hygiene: one-time WEIGHT-ONLY row-center at init (gauge-clean start),
    # under its OWN gate -- NOT the legacy row_center_head path (no exp_avg surgery).
    # Pairs with the in-optimizer applied-update projection to keep the head gauge-free
    # from step 0. Skipped on resume past step 1 (the stored head is already centered).
    if head_gauge_init_rc and start_step == 1 and not getattr(settings, 'resume_training', False):
        _hg0 = _row_center_head_step(model, optimizer, want_exp_avg=False)
        if _hg0 is not None and ddp_rank == 0:
            logger.print_and_log(
                f"[head-gauge] init row-center: ||mu(W)|| "
                f"{_hg0['mu_w_pre']:.4g} -> {_hg0['mu_w_post']:.2e}")

    # Do a baseline validation before training
    if start_step == 1:
        sync_val_loader()
        _scs_dt0 = deepest_active_tap(aux_head_schedules, 0) if scs_enabled else None
        _scs_sm0 = _scs_dt0 is not None
        do_validation(
            model, val_loader, device, settings.eval_iters, 0, ddp_rank,
            settings.val_log_file, total_tokens_processed,
            ddp, ddp_world_size, settings.data_type, device_type,
            scaffold_mode=_scs_sm0,
            active_layers=(_scs_dt0 + 1) if _scs_sm0 else None,
            scs_deepest_tap=_scs_dt0,
        )

    # [doc-mask] in-stream separator check state (fatal by ~step 20 on a wrong bos id —
    # the silent-no-op failure mode; skipped entirely when the feature is off or resuming
    # past the check window).
    _dm_stream_check = (settings.doc_attn_mask_enabled or settings.doc_pos_reset) and start_step <= 20
    _dm_bos_seen = 0
    _dm_windows_seen = 0

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

        # ── SCS dispatch (forward-shape decisions) ──────────────────────────
        # If any aux head is at λ >= 1.0, truncate forward + backward at that
        # depth and skip the main LM head. The deepest active aux head's CE
        # becomes the effective LM loss. When the cascade releases (no head at
        # λ >= 1.0), scaffold_mode flips off and the network trains end-to-end.
        # The lr_scale_overrides writes for SCS happen further down, AFTER the
        # lr_mod_entries loop, so SCS's freeze can't be silently overwritten.
        scs_deepest_tap = None
        scaffold_mode = False
        scs_active_layers = None
        if scs_enabled:
            scs_deepest_tap = deepest_active_tap(aux_head_schedules, step, threshold=1.0)
            if scs_deepest_tap is not None:
                scaffold_mode = True
                scs_active_layers = scs_deepest_tap + 1
                # SCS takes precedence over the existing truncator (they're
                # incompatible — truncator weights the loss, SCS drops main).
                active_layers = scs_active_layers
                is_truncated = True
                trunc_loss_w = 1.0

        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Two loss accumulators per step:
        #   main_loss_accum  -- the model's actual NTP loss against targets;
        #                       what `ls:` in the log reflects, and what
        #                       perplexity is computed against. Comparable to
        #                       val_loss and to baseline runs.
        #   total_loss_accum -- the value the optimizer minimises: main loss
        #                       plus Σ_i w_i * aux_loss_i. Equal to main when
        #                       aux is disabled or all schedule weights are 0.
        # Initialised as 0-D tensors (not Python floats) so dist.all_reduce
        # works even when no micro-step contributes a value (rare but
        # possible under degenerate scaffold configs).
        main_loss_accum = torch.zeros((), device=device)
        total_loss_accum = torch.zeros((), device=device)
        # Z-loss bookkeeping (RAW, pre-alpha) for the live LM readout head.
        # zloss_alpha_eff is a pure function of the absolute global step ->
        # resume-safe, no checkpoint state. Accumulators stay zero when z-loss
        # is disabled, so logging gates cleanly on zloss_enabled.
        zloss_accum = torch.zeros((), device=device)
        logZ_accum = torch.zeros((), device=device)
        logZ_p95_last = 0.0          # last micro-step's logZ 95th pctile (snapshot)
        zloss_alpha_eff = get_zloss_alpha(step, settings)
        zloss_diag = None            # snapshot dict for diagnostics.jsonl (None when z-loss off)
        rc_diag = None               # row-center snapshot for diagnostics.jsonl (None when off)
        # Per-step aux loss bookkeeping. Schedule weights are evaluated once
        # per step (constant within a step, lerp across waypoints). Per-head
        # unweighted CE values accumulate across micro-batches for logging.
        aux_weights_now = (
            {li: interpolate_lr_mod(sched, step) for li, sched in aux_head_schedules.items()}
            if aux_heads_enabled else {}
        )
        # Lazy-init: only populate entries for aux heads that actually fire
        # this step. Under SCS scaffold, taps above the active range don't
        # capture; pre-initialising would log misleading `aux_lN: 0.0000` for
        # heads that aren't training. Keys are added on first contribution.
        aux_loss_accum: dict = {}
        grad_accum_steps = grad_accum_schedule[step]
        # REPLICATED-DATA probe (Math Agent test #3): when WD_REPLICATED_DATA=1,
        # broadcast rank 0's microbatch to ALL ranks so every rank computes its
        # gradient over IDENTICAL data. Combined with WD_INSITU_PROBE=1 this is the
        # clean single-variable cut: same data across ranks, everything else (FSDP
        # path, all-gathered bf16 params, reduce-scatter) unchanged. If the -0.0129
        # lean SURVIVES => it is FSDP machinery/precision, NOT cross-rank data
        # composition. If it VANISHES/changes => it is the data distribution. Env-gated;
        # zero effect on normal runs. (One-shot — WD_INSITU_PROBE exits after this step.)
        _repl_data = os.environ.get('WD_REPLICATED_DATA') == '1' and ddp
        # TOKEN CAPTURE (WD_DUMP_TOKENS=1): dump rank0's FIRST microbatch tokens as
        # RAW BINARY int tensors for exact offline replay (Math Agent #1, splits the
        # residual into real-stream-data vs FSDP/bf16-path). BLACK BOX: tokens are
        # never decoded, detokenized, printed, or logged — only torch.save'd as ints
        # and fed straight back into the model. The dataset is unfiltered; we treat
        # all token content as opaque in every code path. Shapes only in any log.
        _dump_tokens = os.environ.get('WD_DUMP_TOKENS') == '1'
        # Microbatch PROGRESS log — gated on any probe flag so normal training stays
        # quiet. Shows N/GA + per-microbatch cadence so long GA (e.g. 366 in fp32) is
        # not a black box. rank0 only, every _prog_every (default 25) + the last one.
        _prog = bool(os.environ.get('WD_INSITU_PROBE') or os.environ.get('WD_REDUCE_PROBE')
                     or os.environ.get('WD_DUMP_TOKENS') or os.environ.get('WD_REPLICATED_DATA'))
        _prog_every = int(os.environ.get('WD_PROG_EVERY', '25'))
        _prog_t0 = time.time()
        # PROBE-FAST (WD_PROBE_FAST=1 or WD_PROBE_MAX_MB=N): cap grad-accum microbatches for
        # the toggle-matrix probes. The body-ramp lean is a POPULATION property present in every
        # microbatch (not accumulation-dependent), so cos(.grad,W) over a few microbatches ≈ the
        # full-GA value. Slashes runtime AND avoids the world=1 optimizer.step OOM (paired with
        # the pre-step fast capture below). Default cap = 8 when WD_PROBE_FAST set, else full.
        if os.environ.get('WD_PROBE_MAX_MB'):
            grad_accum_steps = min(grad_accum_steps, int(os.environ['WD_PROBE_MAX_MB']))
        elif os.environ.get('WD_PROBE_FAST') == '1':
            grad_accum_steps = min(grad_accum_steps, 8)
        for micro_step in range(grad_accum_steps):
            if _prog and ddp_rank == 0 and (micro_step % _prog_every == 0 or micro_step == grad_accum_steps - 1):
                _el = time.time() - _prog_t0
                _rate = (_el / max(micro_step, 1))
                _eta = _rate * (grad_accum_steps - micro_step)
                logger.print_and_log(f"[PROBE-PROG] microbatch {micro_step+1}/{grad_accum_steps}  "
                                     f"elapsed={_el:.0f}s  ~{_rate:.1f}s/mb  eta~{_eta:.0f}s")
            x, y = train_loader.next_batch(step=step)
            x, y = x.to(device), y.to(device)
            # [doc-mask] stream sanity: the mask/pos-reset keyed on a WRONG bos id is a silent
            # complete no-op (the exact failure the review caught — the data separator is the
            # tokenizer-native id 1, not <|bos|>=32000). Count separators over the first ~20
            # steps and fatal if the configured id never appears in the stream.
            if _dm_stream_check and step <= 20:
                _dm_bos_seen += int((x == settings.doc_bos_token_id).sum())
                _dm_windows_seen += x.shape[0]
                if step == 20 and micro_step == grad_accum_steps - 1:
                    # Judge on the GLOBAL count: ranks read DISJOINT data, so one rank can
                    # locally see zero separators (books-heavy draw) while others see many.
                    # A single-rank fatal_error would strand the other ranks in the step's
                    # collectives (barrier vs all_reduce mismatch -> NCCL timeout); the
                    # all-reduced verdict fires identically on every rank.
                    if ddp:
                        _dm_t = torch.tensor([_dm_bos_seen, _dm_windows_seen],
                                             device=device, dtype=torch.int64)
                        dist.all_reduce(_dm_t)
                        _dm_bos_seen, _dm_windows_seen = int(_dm_t[0]), int(_dm_t[1])
                    if _dm_bos_seen == 0:
                        fatal_error(
                            f"doc_attn_mask: bos_token_id={settings.doc_bos_token_id} NEVER appeared "
                            f"in {_dm_windows_seen} windows (all ranks) — the mask/position-reset is "
                            f"a silent no-op. The llama stream separates documents with id 1.")
                    logger.print_and_log(
                        f"  ] [doc-mask] stream check OK: {_dm_bos_seen} separators in "
                        f"{_dm_windows_seen} windows (~{_dm_bos_seen / max(_dm_windows_seen, 1):.2f} "
                        f"docs-started/window at T={settings.T}, all ranks)")
                    _dm_stream_check = False
            if _repl_data:
                # broadcast rank0's batch to all ranks (token-id int tensors)
                dist.broadcast(x, src=0)
                dist.broadcast(y, src=0)
            if _dump_tokens and micro_step == 0 and ddp_rank == 0:
                _tokpath = os.environ.get('WD_TOKENS_OUT', 'rank0_tokens.pt')
                # raw binary int tensors only; NO decode, NO content logging
                torch.save({'x': x.detach().cpu(), 'y': y.detach().cpu(),
                            'step': step, 'shape': tuple(x.shape)}, _tokpath)
                logger.print_and_log(f"[WD-DUMP] saved rank0 microbatch tokens (shape {tuple(x.shape)}, OPAQUE) -> {_tokpath}")
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
                    logits, main_loss = fwd_model(
                        x, y, active_layers=active_layers, scaffold_mode=scaffold_mode,
                    )

                # Build total_loss = main + Σ w_i * aux_i (when any aux head
                # has a nonzero weight at this step). Per-head unweighted CE
                # accumulates for logging regardless of weight.
                #
                # In scaffold_mode, main_loss is None (model skipped the main
                # LM head); total_loss starts as the deepest active aux head's
                # weighted contribution and aggregates from there. main_loss
                # for logging gets set to the deepest active aux head's
                # unweighted CE — the "effective LM loss" of the partial
                # network — so `ls:` stays interpretable.
                total_loss = main_loss
                main_loss_for_log = None  # set below; either model's main or deepest aux CE
                if aux_heads_enabled:
                    raw_for_aux = fwd_model._orig_mod if hasattr(fwd_model, '_orig_mod') else fwd_model
                    aux_tensors = getattr(raw_for_aux, '_last_aux_loss_tensors', None) or {}
                    if aux_tensors:
                        for li, t in aux_tensors.items():
                            w = aux_weights_now.get(li, 0.0)
                            if w != 0.0:
                                total_loss = (w * t) if total_loss is None else total_loss + w * t
                            _add = t.detach().float() / grad_accum_steps
                            aux_loss_accum[li] = aux_loss_accum[li] + _add if li in aux_loss_accum else _add
                        if scaffold_mode and scs_deepest_tap is not None and scs_deepest_tap in aux_tensors:
                            # The effective LM loss in scaffold mode = deepest
                            # active aux head's CE (the model is literally
                            # training to predict tokens through that head).
                            main_loss_for_log = aux_tensors[scs_deepest_tap].detach().float()

                if main_loss_for_log is None:
                    main_loss_for_log = main_loss.detach().float() if main_loss is not None else None

                if total_loss is None:
                    # Should not happen — scaffold_mode requires at least one
                    # aux head firing, and non-scaffold paths always return
                    # main_loss from the model. Guard for clarity.
                    raise RuntimeError(
                        "total_loss is None at step {} — no main loss and no aux losses".format(step)
                    )

                # Z-loss: select the SAME head that drives main_loss_for_log —
                # the deepest active aux tap under SCS scaffold, else the
                # model's main head — and fold alpha_eff * zloss into the
                # objective. Shallow scaffolding aux taps never contribute.
                # Added BEFORE the trunc/grad-accum scaling below so the z term
                # gets the identical normalisation as the rest of total_loss.
                # The raw (pre-alpha) value is accumulated for logging; the
                # headline ls:/ppl stay pure CE (they read main_loss_accum).
                if zloss_enabled:
                    raw_for_z = fwd_model._orig_mod if hasattr(fwd_model, '_orig_mod') else fwd_model
                    if scaffold_mode and scs_deepest_tap is not None:
                        z_sel = (getattr(raw_for_z, '_last_aux_zloss', None) or {}).get(scs_deepest_tap)
                        logz_sel = (getattr(raw_for_z, '_last_aux_logz', None) or {}).get(scs_deepest_tap)
                    else:
                        z_sel = getattr(raw_for_z, '_last_zloss', None)
                        logz_sel = getattr(raw_for_z, '_last_logz', None)
                    if z_sel is not None:
                        if zloss_alpha_eff != 0.0:
                            total_loss = total_loss + zloss_alpha_eff * z_sel
                        zloss_accum += z_sel.detach().float() / grad_accum_steps
                        if logz_sel is not None:
                            logZ_accum += logz_sel.detach().float() / grad_accum_steps
                        # logZ p95 (tail) — snapshot the last micro-step's value
                        # (a percentile doesn't average meaningfully; main head
                        # only — aux taps don't surface it). rms is derived from
                        # zloss_accum at log time (rms = sqrt(mean logZ**2)).
                        _p95 = getattr(raw_for_z, '_last_logz_p95', None)
                        if _p95 is not None:
                            logZ_p95_last = _p95.detach().float().item()

                # Apply the same truncation weighting + grad-accum normalisation
                # to both. backward() runs on total_loss so the optimiser sees
                # the combined objective; main_loss_for_log is recorded for
                # the `ls:` field.
                total_loss = total_loss * trunc_loss_w / grad_accum_steps
                if main_loss_for_log is not None:
                    main_loss_accum += main_loss_for_log * trunc_loss_w / grad_accum_steps
                total_loss_accum += total_loss.detach().float()
                total_loss.backward()

        # Capture gradient norms before optimizer.step() clears them
        # Only capture on steps right before validation to avoid overhead.
        # Skip diagnostics on truncator-truncated steps (skipped layers
        # have zero gradients which would corrupt the snapshot). SCS
        # scaffold IS the steady state for thousands of steps, so we DO
        # capture there — active-layer entries are valid, frozen-tail
        # entries are zero/empty and the dashboard already tolerates
        # missing per-layer data.
        if diagnostics is not None and (not is_truncated or scaffold_mode) and (step % settings.val_step == 0 or last_step):
            diagnostics.capture_gradients()

        if ddp:
            dist.all_reduce(main_loss_accum, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_loss_accum, op=dist.ReduceOp.AVG)
            for _li in aux_loss_accum:
                dist.all_reduce(aux_loss_accum[_li], op=dist.ReduceOp.AVG)
            if zloss_enabled:
                dist.all_reduce(zloss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(logZ_accum, op=dist.ReduceOp.AVG)

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

        # Tangent-projection strength f(step): write the current value into every Muon body group
        # so the optimizer's projection uses it this step. f is a global scalar (uniform across
        # ranks and matrices), so this is just a few dict writes — like the per-step LR write above.
        # Skipped entirely when projection is off; constant config writes a constant (default 1.0
        # = no behavior change). The group's strength is read in muon_fsdp2's projection block.
        if _tp_on:
            _f_now = interpolate_lr_mod(_tps_cfg, step) if _tps_is_sched else float(_tps_cfg)
            for _pg in optimizer.param_groups:
                if _pg.get('tangent_project', False):
                    _pg['tangent_project_strength'] = _f_now

        clip_value = settings.clip_warmup if step < settings.warmup_steps else settings.clip_standard

        # Head metrics for the z-loss probe: weight_norm + GLOBAL (FSDP-reduced)
        # grad_norm of the output head, measured BEFORE clipping so they're
        # unscaled and consistent with total grad-norm (also pre-clip). The
        # ratio head.grad_norm/total tells us whether z-loss is actually
        # touching the readout. Computed only when z-loss is enabled.
        head_w_norm = head_g_norm = None
        if zloss_enabled:
            _raw_hm = model._orig_mod if hasattr(model, '_orig_mod') else model
            head_w_norm, head_g_norm = _global_param_norms(
                [_head_param(_raw_hm)], ddp
            )

        if _clip_groups_cfg:
            norm, _clip_groups = _clip_grad_norm_mixed_mesh(model, clip_value, group_telemetry=True)
        else:
            norm = _clip_grad_norm_mixed_mesh(model, clip_value)
            _clip_groups = None

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

        # ── SCS lr_scale_overrides (must run AFTER lr_mod_entries so SCS's
        #    freeze of output.weight / inactive layers can't be silently
        #    clobbered by an `[out, schedule]` rule like the one
        #    output_lr_batch_adjust auto-injects).
        if scs_enabled:
            # Per-layer compartment warmup + freeze. When SCS has nothing
            # custom to assert (scale==1.0 AND not in scaffold), defer to
            # any lr_mods entry that manages the param. During scaffold or
            # warmup ramp (scale != 1.0), SCS always wins.
            for _li in range(model_cfg.n_layers):
                _scale = scs_compartment_lr_scale(
                    scs_activation_steps[_li], scs_warmup_steps, scs_init_mult, step,
                )
                if scaffold_mode and _li >= scs_active_layers:
                    _scale = 0.0  # inactive tail — freeze (no fwd, no WD decay)
                _defer = (_scale == 1.0 and not scaffold_mode)
                for _p in scs_layer_params[_li]:
                    if _defer and id(_p) in _lr_mod_managed_param_ids:
                        continue
                    lr_scale_overrides[id(_p)] = _scale
            # Main output head + final norm: treated as the "cascade-complete
            # compartment" — they activate at the same step that brings the
            # tail layers online, and ramp from init_mult to 1.0 over
            # new_layer_warmup_steps just like any other newly-activated
            # compartment. Without this ramp the random-init output head
            # would shock the trained body via backprop on the first
            # non-scaffold step (large dL/dh through layers 0..deepest_tap).
            # During scaffold proper, scs_compartment_lr_scale returns 0.0
            # because step < cascade-complete step, so freeze still holds.
            # Defers to lr_mods only post-warmup (scale==1.0) to avoid
            # clobbering output_lr_batch_adjust style rules.
            _out_scale = scs_compartment_lr_scale(
                _cascade_complete_step, scs_warmup_steps, scs_init_mult, step,
            )
            _out_defer = (_out_scale == 1.0 and not scaffold_mode)
            for _p in scs_output_params:
                if _out_defer and id(_p) in _lr_mod_managed_param_ids:
                    continue
                lr_scale_overrides[id(_p)] = _out_scale
            # Aux heads:
            #   * Above active range during scaffold: frozen (no fwd, no
            #     grad, but WD would still apply without the freeze).
            #   * Current schedule weight == 0: also frozen — no gradient
            #     contributes back through them (the trainer's `if w != 0`
            #     gate skips the multiply-add into total_loss), so the only
            #     effect they'd see is silent WD decay.
            #   * Otherwise: full LR.
            for _li, _params in scs_aux_head_params.items():
                _w_now = aux_weights_now.get(_li, 0.0)
                _frozen = (
                    (scaffold_mode and _li > scs_deepest_tap)
                    or _w_now == 0.0
                )
                _aux_scale = 0.0 if _frozen else 1.0
                for _p in _params:
                    lr_scale_overrides[id(_p)] = _aux_scale

        # FFN pdr controller (kv3): write the controller's HELD FFN multiplier into the
        # lr_scale side-dict for the FFN body params. Placed AFTER the SCS block so SCS can't
        # clobber it, and before optimizer.step. The held value is refreshed at diagnostic
        # cadence by body_lr_ctrl.observe(...) near the [body-pdr] line. Inert when disabled
        # (current_multiplier() == 1.0). When enabled, the controller OWNS the FFN lr_scale
        # (it runs after the lr_mods write at 1909, so it wins for FFN params by design).
        if body_lr_ctrl is not None and body_lr_ctrl.enabled and lr_scale_overrides is not None:
            _m_ffn = body_lr_ctrl.current_multiplier()
            for _pid in _pdr_m_ids:                       # FFN, + attn when acts_on_attn (same m)
                lr_scale_overrides[_pid] = _m_ffn

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

        # Shadow-norm controller: write the radial-budget body WD λ into wd_overrides for the FFN
        # body matrices. AFTER the WD-rules + AWD writers so the controller OWNS body-FFN WD (the
        # radial-budget law replaces the hand WD taper). None until the first shadow observe -> skip
        # (the config's base WD stands until the controller is live). Held value, like the m write.
        if _shadow_ctrl_on and wd_overrides is not None:
            _lam_ffn = body_lr_ctrl.current_wd()
            if _lam_ffn is not None:
                for _pid in _ffn_param_ids_for_ctrl:
                    wd_overrides[_pid] = _lam_ffn

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

        # Snapshot weights for update ratio diagnostic (before step modifies them).
        # Same gating as capture_gradients above — scaffold counts as a valid
        # capture step even though is_truncated is True.
        is_diag_step = diagnostics is not None and (not is_truncated or scaffold_mode) and (step % settings.val_step == 0 or last_step)
        if is_diag_step:
            diagnostics.snapshot_weights()

        # ── IN-SITU ΔW DECOMPOSITION (WD-waste investigation; env-gated, one-shot) ──
        # Snapshot per-matrix W before optimizer.step(); after the step measure the
        # ACTUAL ΔW on the genuine FSDP2 path with warm buffers, decompose into
        # radial (<ΔW,W>/||W||) vs tangential, subtract the known decoupled-WD term
        # (eff_lr*wd*||W||), and dump JSON. Resolves the cold-replay 38x gap with
        # zero replication assumptions. Set WD_INSITU_PROBE=1 to enable; exits after
        # one captured step. NO effect on normal runs.
        # ── STAGE A: bf16 ALL-REDUCE radial-gradient probe (env-gated, one-shot) ──
        # Stage B (offline single-card) EXONERATED the fused-CCE kernel, bf16-CCE
        # accumulation, long context (T=12288), and act-ckpt recompute: all give
        # cos(g,W) ~ +0.0001 sign-random, NOT the in-situ -0.0129. STAGE A then ruled
        # out the bf16 reduce-scatter too (fp32-reduce gave IDENTICAL -0.01288). So the
        # lean is PRE-reduction. This probe splits the last fork: compares
        # cos(g_reduced, W) [GLOBAL, the real reduce-scattered .grad] vs
        # cos(g_local, W) [PER-RANK, a local grad-accum under set_requires_gradient_sync
        # (False) => NO cross-rank reduction, NO aggregation in the cos]. If EVERY rank's
        # LOCAL grad already leans <<0, the anomaly is intrinsic to a single rank's
        # sharded computation (all-gathered bf16 params / its data shard) and the
        # single-card probe should have caught it but didn't => REAL MYSTERY. If LOCAL~0
        # but REDUCED<<0, it emerges only from combining 8 different-data partials.
        # Set WD_REDUCE_PROBE=1. One-shot: dumps JSON, exits before optimizer.step().
        if os.environ.get('WD_REDUCE_PROBE') == '1':
            import torch.distributed as _rdist
            from torch.distributed.tensor import DTensor as _RDT
            _raw_rp = model._orig_mod if hasattr(model, '_orig_mod') else model
            def _rp_local(_t):
                return _t._local_tensor if isinstance(_t, _RDT) else _t
            def _rp_gsum(_t, _ref):
                if isinstance(_ref, _RDT) and _rdist.is_available() and _rdist.is_initialized():
                    _t = _t.clone(); _rdist.all_reduce(_t, group=_ref.device_mesh.get_group())
                return _t
            def _rp_isbody(_n):
                return any(_n.endswith(s) for s in ('wo.weight', 'w2.weight', 'wq.weight',
                           'wk.weight', 'wv.weight', 'w1.weight', 'w3.weight'))
            def _rp_cos_map(cross_rank):
                # cross_rank=True: aggregate dot/norms ACROSS ranks (correct for the
                #   real reduce-scattered DTensor grad — gives the GLOBAL cos).
                # cross_rank=False: per-rank LOCAL cos, NO all-reduce — answers "does
                #   ONE rank's own gradient (its data shard, all-gathered bf16 params)
                #   already lean?" Critical: for the un-synced local grad we must NOT
                #   sum across ranks (that would re-combine the 8 partials we are
                #   deliberately keeping separate). Each rank reports its own shard's cos.
                _out = {}
                for _n, _p in _raw_rp.named_parameters():
                    if not _rp_isbody(_n) or _p.grad is None:
                        continue
                    _W = _rp_local(_p).detach().float(); _G = _rp_local(_p.grad).detach().float()
                    if cross_rank:
                        _dot = _rp_gsum((_W*_G).sum(), _p).item()
                        _wn = _rp_gsum((_W*_W).sum(), _p).clamp_min(0).sqrt().item()
                        _gn = _rp_gsum((_G*_G).sum(), _p).clamp_min(0).sqrt().item()
                    else:
                        _dot = (_W*_G).sum().item()
                        _wn = (_W*_W).sum().clamp_min(0).sqrt().item()
                        _gn = (_G*_G).sum().clamp_min(0).sqrt().item()
                    _out[_n] = (_dot/(_wn*_gn)) if (_wn > 0 and _gn > 0) else 0.0
                return _out
            # (1) REDUCED: the current .grad is the real reduce-scattered grad from
            # the grad-accum backward above (averaged over grad_accum_steps micro-
            # batches, reduce-scattered across ranks). GLOBAL cos (cross-rank agg).
            _cos_reduced = _rp_cos_map(cross_rank=True)
            # (2) LOCAL: re-run the SAME number of microbatches (grad_accum_steps)
            # ALL inside no_sync() so they accumulate locally with NO cross-rank
            # reduction. This holds the microbatch count + loss normalisation
            # constant — the ONLY difference vs (1) is the reduce-scatter. (Data
            # differs by the next grad_accum_steps batches from the loader, but the
            # radial-lean question is a population property, not batch-specific, and
            # Stage B showed it's batch-insensitive at +0.0001 across panels.)
            for _p in _raw_rp.parameters():
                _p.grad = None
            # Local pass microbatch count: capped (default 16) to bound the EXTRA
            # memory of this second grad-accum pass (the reduced pass already ran).
            # The lean is a SYSTEMATIC component present in every microbatch, so fewer
            # microbatches leaves the expected cos ≈ unchanged (only the variance grows);
            # 16 keeps per-matrix noise low enough to distinguish -0.013 from +0.0001
            # cleanly. The REDUCED arm is still the full real grad_accum_steps, measured
            # in the same run, as an internal calibration. Set WD_REDUCE_LOCAL_MB to tune.
            _ga = min(grad_accum_steps, int(os.environ.get('WD_REDUCE_LOCAL_MB', '16')))
            # FSDP2 has NO no_sync() context — it uses set_requires_gradient_sync().
            # Disable gradient sync so the grad-accum below stays LOCAL (per-rank,
            # NO reduce-scatter). The model is wrapped with fully_shard => it's an
            # FSDPModule exposing this method. Re-enable after.
            _has_grsync = hasattr(model, 'set_requires_gradient_sync')
            _orig_target = model._orig_mod if hasattr(model, '_orig_mod') else None
            if _has_grsync:
                model.set_requires_gradient_sync(False)
            elif _orig_target is not None and hasattr(_orig_target, 'set_requires_gradient_sync'):
                _orig_target.set_requires_gradient_sync(False); _has_grsync = True
            try:
                _fm = model
                if is_truncated and truncator.bypass_compile and hasattr(model, '_orig_mod'):
                    _fm = model._orig_mod
                for _ms in range(_ga):
                    _xr, _yr = train_loader.next_batch(step=step)
                    _xr, _yr = _xr.to(device), _yr.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if settings.data_type=="bf16" else torch.float16 if settings.data_type=="fp16" else torch.float32):
                        _lr_logits, _lr_loss = _fm(_xr, _yr, active_layers=active_layers, scaffold_mode=scaffold_mode)
                    (_lr_loss * trunc_loss_w / _ga).backward()
            finally:
                if _has_grsync:
                    (model if hasattr(model, 'set_requires_gradient_sync') else _orig_target
                     ).set_requires_gradient_sync(True)
            # LOCAL: per-rank, NO cross-rank aggregation. Each rank's value is the cos
            # over ITS OWN shard of W vs ITS OWN local grad. We summarise each rank's
            # median and gather all ranks to rank 0 — so we see if EVERY rank's local
            # gradient leans (=> intrinsic to the sharded local computation) or only the
            # aggregate does.
            import statistics as _rst, json as _rjson
            _cos_local = _rp_cos_map(cross_rank=False)
            def _rp_med(_d):
                _v = list(_d.values())
                return {'median': _rst.median(_v), 'mean': _rst.mean(_v),
                        'negfrac': sum(1 for _c in _v if _c < 0)/len(_v), 'n': len(_v)} if _v else {}
            _ml_self = _rp_med(_cos_local)              # THIS rank's local summary
            _mr = _rp_med(_cos_reduced)                 # global reduced summary (same on all ranks)
            # gather every rank's local summary to rank 0
            _ml_all = [None] * (ddp_world_size if ddp else 1)
            if ddp:
                _rdist.all_gather_object(_ml_all, (ddp_rank, _ml_self))
            else:
                _ml_all = [(0, _ml_self)]
            if ddp_rank == 0:
                logger.print_and_log(
                    f"[WD-REDUCE] cos(g_REDUCED,W) GLOBAL: median={_mr.get('median'):+.5f} "
                    f"mean={_mr.get('mean'):+.5f} negfrac={_mr.get('negfrac',0)*100:.0f}% n={_mr.get('n')}")
                for _rk, _m in sorted([x for x in _ml_all if x is not None]):
                    logger.print_and_log(
                        f"[WD-REDUCE] cos(g_LOCAL,W) rank{_rk}: median={_m.get('median'):+.5f} "
                        f"mean={_m.get('mean'):+.5f} negfrac={_m.get('negfrac',0)*100:.0f}% n={_m.get('n')}")
                logger.print_and_log(
                    "[WD-REDUCE] Read: LOCAL(per-rank)<<0 => the lean is INTRINSIC to a single rank's "
                    "sharded gradient (all-gathered bf16 params / data shard), NOT the cross-rank combine "
                    "=> the single-card probe SHOULD have caught it but didn't => REAL MYSTERY. "
                    "LOCAL~0 but REDUCED<<0 => the lean emerges only from summing 8 different-data partials.")
                _routp = os.environ.get('WD_REDUCE_OUT', 'wd_reduce.json')
                with open(_routp, 'w') as _rf:
                    _rjson.dump({'step': step, 'reduced_global': _mr,
                                 'local_per_rank': {str(_rk): _m for _rk, _m in _ml_all if _m is not None},
                                 'reduced_per_matrix': _cos_reduced,
                                 'local_per_matrix_rank0': _cos_local}, _rf, indent=1)
                logger.print_and_log(f"[WD-REDUCE] wrote {_routp} — exiting one-shot")
            if ddp:
                _rdist.barrier()
            sys.exit(0)

        _insitu = os.environ.get('WD_INSITU_PROBE') == '1'
        _insitu_snap = None
        if _insitu:
            from torch.distributed.tensor import DTensor as _DT
            _raw_is = model._orig_mod if hasattr(model, '_orig_mod') else model
            # map param id -> (eff_lr, wd) from the optimizer groups for the WD term
            _effmap = {}
            for _g in optimizer.param_groups:
                _lr = float(_g.get('lr', 0.0)); _w = float(_g.get('weight_decay', 0.0))
                for _p in _g['params']:
                    _effmap[id(_p)] = (_lr, _w)
            _insitu_snap = {}
            for _n, _p in _raw_is.named_parameters():
                if _p.dim() == 2:
                    _loc = _p._local_tensor if isinstance(_p, _DT) else _p
                    _lr, _w = _effmap.get(id(_p), (0.0, 0.0))
                    _ls = lr_scale_overrides.get(id(_p), 1.0) if lr_scale_overrides else 1.0
                    _insitu_snap[_n] = (_loc.detach().float().clone(), _lr * _ls, _w)

        # PRE-vs-POST-STEP grad probe (WD_PREPOST_PROBE=1): the DECISIVE test of whether the
        # "-0.0129 lean" is the RAW CE gradient or the Muon/Newton-Schulz-TRANSFORMED grad.
        # Muon's step() scatters the NS-orthogonalized update back INTO p.grad (muon_fsdp2:357),
        # so the in-situ probe (which reads .grad AFTER step) measures the POST-NS grad, not the
        # raw one. Here we capture cos(.grad,W) BOTH before AND after optimizer.step on the SAME
        # step, same params, so the only difference is the Muon transform. One-shot.
        if os.environ.get('WD_PREPOST_PROBE') == '1':
            import torch.distributed as _fdist
            from torch.distributed.tensor import DTensor as _FDT
            import statistics as _fst, json as _fjson
            _raw_f = model._orig_mod if hasattr(model, '_orig_mod') else model
            def _f_gsum(_t, _ref):
                if isinstance(_ref, _FDT) and _fdist.is_available() and _fdist.is_initialized():
                    _t = _t.clone(); _fdist.all_reduce(_t, group=_ref.device_mesh.get_group())
                return _t
            def _f_isbody(_n):
                return any(_n.endswith(s) for s in ('wo.weight','w2.weight','wq.weight',
                           'wk.weight','wv.weight','w1.weight','w3.weight'))
            def _f_cosmap():
                _o = {}
                for _n, _p in _raw_f.named_parameters():
                    if not _f_isbody(_n) or _p.grad is None:
                        continue
                    _W = (_p._local_tensor if isinstance(_p, _FDT) else _p).detach().float()
                    _G = (_p.grad._local_tensor if isinstance(_p.grad, _FDT) else _p.grad).detach().float()
                    _d = _f_gsum((_W*_G).sum(), _p).item()
                    _wn = _f_gsum((_W*_W).sum(), _p).clamp_min(0).sqrt().item()
                    _gn = _f_gsum((_G*_G).sum(), _p).clamp_min(0).sqrt().item()
                    _o[_n] = (_d/(_wn*_gn)) if (_wn>0 and _gn>0) else 0.0
                return _o
            def _f_summ(_d):
                _v = list(_d.values())
                return {'median': _fst.median(_v), 'mean': _fst.mean(_v),
                        'negfrac': sum(1 for c in _v if c<0)/len(_v), 'n': len(_v)} if _v else {}
            # snapshot W (pre-step) for the post-step cos to use the SAME reference
            _W_snap = {_n: (_p._local_tensor if isinstance(_p, _FDT) else _p).detach().float().clone()
                       for _n, _p in _raw_f.named_parameters() if _f_isbody(_n)}
            _cos_pre = _f_cosmap()              # RAW reduced loss gradient vs W
            optimizer.step()                    # Muon scatters NS-update into .grad
            # post-step: cos(.grad_now, W_pre) — .grad is now the Muon-transformed update.
            # ALSO (Math Agent Probe 1): compute the TANGENT-PROJECTED update U⊥ = U − W⟨U,W⟩/‖W‖²
            # (global all-reduced coeff) and its cos — MEASUREMENT ONLY, no param mutation. If
            # cos(U⊥,W)→0 and ‖U⊥‖/‖U‖≈1, the projection fix is validated before any real run.
            _cos_post = {}; _cos_proj = {}; _normratio = {}
            for _n, _p in _raw_f.named_parameters():
                if not _f_isbody(_n) or _p.grad is None or _n not in _W_snap:
                    continue
                _W = _W_snap[_n]
                _G = (_p.grad._local_tensor if isinstance(_p.grad, _FDT) else _p.grad).detach().float()
                _dot = _f_gsum((_W*_G).sum(), _p).item()
                _wn = _f_gsum((_W*_W).sum(), _p).clamp_min(0).sqrt().item()
                _gn = _f_gsum((_G*_G).sum(), _p).clamp_min(0).sqrt().item()
                _cos_post[_n] = (_dot/(_wn*_gn)) if (_wn>0 and _gn>0) else 0.0
                # projected update U⊥ = U − c·W,  c = ⟨U,W⟩/‖W‖²  (global)
                _wsq = _wn*_wn
                _c = (_dot/_wsq) if _wsq>0 else 0.0
                _Gp = _G - _c*_W
                _dp = _f_gsum((_W*_Gp).sum(), _p).item()
                _gpn = _f_gsum((_Gp*_Gp).sum(), _p).clamp_min(0).sqrt().item()
                _cos_proj[_n] = (_dp/(_wn*_gpn)) if (_wn>0 and _gpn>0) else 0.0
                _normratio[_n] = (_gpn/_gn) if _gn>0 else 0.0
            if ddp_rank == 0:
                _sp, _so, _spr = _f_summ(_cos_pre), _f_summ(_cos_post), _f_summ(_cos_proj)
                _nr = list(_normratio.values()); _nrmed = _fst.median(_nr) if _nr else 0.0
                logger.print_and_log(f"[PREPOST] cos(RAW grad,W)  PRE-step : median={_sp.get('median'):+.5f} negfrac={_sp.get('negfrac',0)*100:.0f}% n={_sp.get('n')}")
                logger.print_and_log(f"[PREPOST] cos(NS update,W) POST-step: median={_so.get('median'):+.5f} negfrac={_so.get('negfrac',0)*100:.0f}% n={_so.get('n')}")
                logger.print_and_log(f"[PREPOST] cos(PROJECTED,W) U⊥      : median={_spr.get('median'):+.5f} negfrac={_spr.get('negfrac',0)*100:.0f}%  ‖U⊥‖/‖U‖ median={_nrmed:.5f}")
                logger.print_and_log("[PREPOST] PROBE 1 PASS if cos(PROJECTED,W)→0 and ‖U⊥‖/‖U‖≈1 (projection removes radial, keeps ~all the update).")
                _foutp = os.environ.get('WD_PREPOST_OUT', 'wd_prepost.json')
                with open(_foutp, 'w') as _ff:
                    _fjson.dump({'step': step, 'mb': grad_accum_steps,
                                 'pre_summary': _sp, 'post_summary': _so, 'proj_summary': _spr,
                                 'normratio_median': _nrmed,
                                 'pre_per_matrix': _cos_pre, 'post_per_matrix': _cos_post,
                                 'proj_per_matrix': _cos_proj}, _ff, indent=1)
                logger.print_and_log(f"[PREPOST] wrote {_foutp} — exiting")
            if ddp:
                _fdist.barrier()
            sys.exit(0)

        optimizer.step()

        # Shadow-norm controller: accumulate the per-step radial budget EVERY step, before the
        # optimizer overwrites radial_stats next step (spec B2). ΔR_free = η·γ·‖W‖ per matrix, with
        # η = the non-controller body LR (scheduled_lr; m excluded). radial_stats is id-keyed and
        # GLOBAL (all-reduced), so this is bit-identical across ranks. FFN body matrices only.
        if _shadow_ctrl_on:
            _rs = getattr(optimizer, 'radial_stats', None)
            if _rs:
                _shadow_eta['sum'] += scheduled_lr
                for _pid, (_wn, _g) in _rs.items():
                    _nm = _ffn_id_to_name.get(_pid)
                    if _nm is None:
                        continue
                    _shadow_R[_nm] = _wn
                    _shadow_gamma[_nm] = _g
                    _shadow_dR[_nm] = _shadow_dR.get(_nm, 0.0) + scheduled_lr * _g * _wn

        if _insitu:
            import torch.distributed as _dist
            from torch.distributed.tensor import DTensor as _DT
            _raw_is = model._orig_mod if hasattr(model, '_orig_mod') else model
            # WARM stage-trace: replay the NorMuon stages on the WARM momentum buffer
            # (post-step, the buffer that PRODUCED this update) to find WHERE the
            # radial bias cos(.,W) enters: grad -> warm-momentum -> NS -> scaling ->
            # normuon. Tests the momentum-curvature hypothesis (cold replay CAN'T —
            # cold momentum == grad). Uses the real optimizer.state buffers.
            try:
                from muon_fsdp2 import (zeropower_via_newtonschulz5 as _ns,
                                        apply_scaling as _ascale, apply_normuon as _anorm)
                _ns_steps = int(getattr(settings, 'muon_ns_steps', 5) or 5)
                _b2 = float(getattr(settings, 'normuon_beta2', 0.95) or 0.95)
                _rmss = bool(getattr(settings, 'muon_rms_scale', False))
                _have_stage = True
            except Exception:
                _have_stage = False
            _rows = []
            for _n, _p in _raw_is.named_parameters():
                if _n not in _insitu_snap:
                    continue
                _W0, _efflr, _wd = _insitu_snap[_n]
                _W1 = (_p._local_tensor if isinstance(_p, _DT) else _p).detach().float()
                _total = _W1 - _W0
                _wd_dW = (-(_efflr * _wd)) * _W0
                _muon_dW = _total - _wd_dW
                def _gsum(_t, _ref=_p):
                    _v = _t
                    if isinstance(_ref, _DT) and _dist.is_available() and _dist.is_initialized():
                        _v = _v.clone(); _dist.all_reduce(_v, group=_ref.device_mesh.get_group())
                    return _v
                def _cos(_a, _b):  # global cos(a,b) over the param's mesh
                    _d = _gsum((_a*_b).sum()).item()
                    _an = _gsum((_a*_a).sum()).clamp_min(0).sqrt().item()
                    _bn = _gsum((_b*_b).sum()).clamp_min(0).sqrt().item()
                    return (_d/(_an*_bn)) if (_an>0 and _bn>0) else 0.0
                _wsq = _gsum((_W0*_W0).sum()); _wn = _wsq.clamp_min(0).sqrt().item()
                _tot_dot = _gsum((_total*_W0).sum()).item()
                _mu_dot = _gsum((_muon_dW*_W0).sum()).item()
                _mu_sq = _gsum((_muon_dW*_muon_dW).sum()); _mun = _mu_sq.clamp_min(0).sqrt().item()
                _row = {
                    'name': _n, 'w_norm': _wn,
                    'total_dW_radial': (_tot_dot/_wn) if _wn>0 else 0.0,
                    'muon_dW_norm': _mun,
                    'muon_radial': (_mu_dot/_wn) if _wn>0 else 0.0,
                    'muon_cos': (_mu_dot/(_mun*_wn)) if (_mun>0 and _wn>0) else 0.0,
                    'wd_radial': -(_efflr*_wd)*_wn,
                    'eff_lr': _efflr, 'wd': _wd,
                }
                # WARM stage-trace (Muon body params only — those with momentum_buffer)
                if _have_stage:
                    _st = optimizer.state.get(_p, {})
                    _mb = _st.get('momentum_buffer')
                    if _mb is not None and _p.grad is not None and _p.dim() == 2:
                        try:
                            _Wl = _W0  # pre-step weight (local), the buffer's reference
                            _g = (_p.grad._local_tensor if isinstance(_p.grad, _DT) else _p.grad).detach().float()
                            _m = (_mb._local_tensor if isinstance(_mb, _DT) else _mb).detach().float()
                            _row['cos_grad_W'] = _cos(_g, _Wl)
                            _row['cos_warmmom_W'] = _cos(_m, _Wl)            # <-- momentum-curvature shows HERE
                            _u = _ns(_m.clone(), _ns_steps).type_as(_m).float()
                            _row['cos_afterNS_W'] = _cos(_u, _Wl)
                            _u = _ascale(_u, _rmss)
                            _row['cos_afterscale_W'] = _cos(_u, _Wl)
                            _smb = _st.get('second_momentum_buffer')
                            if _smb is not None:
                                _sm = (_smb._local_tensor if isinstance(_smb, _DT) else _smb).detach().float().clone()
                                _u = _anorm(_u, _sm, _b2)
                                _row['cos_afternormuon_W'] = _cos(_u, _Wl)
                        except Exception as _e:
                            _row['stage_err'] = f"{type(_e).__name__}"
                _rows.append(_row)
            if ddp_rank == 0:
                import json as _json
                _outp = os.environ.get('WD_INSITU_OUT', 'wd_insitu.json')
                with open(_outp, 'w') as _f:
                    _json.dump({'step': step, 'per_matrix': _rows}, _f, indent=1)
                logger.print_and_log(f"[WD-INSITU] wrote {_outp} ({len(_rows)} matrices) — exiting after one-shot capture")
            if ddp: dist.barrier()
            sys.exit(0)

        # Compute update norms from pre/post step diff
        if is_diag_step:
            diagnostics.capture_updates()

        # Row-center the LM head (gauge subtraction). AFTER capture_updates() so
        # the update-norm diagnostic reflects the true optimizer update, not the
        # gauge projection (logged separately as rc_*). Function-preserving on the
        # output; strips the gauge from the Adam 1st moment too. Eager + no_grad
        # (inside the helper) — kept out of compile.
        rc_tel = None
        rc_s_eff = 0.0
        if row_center_enabled:
            if row_center_warmup_on:
                rc_s_eff = get_row_center_s(step, settings)
                if step >= row_center_warmup_start:
                    # Capture the start gauge ONCE (first in-window step) unless it
                    # was restored from a mid-warmup checkpoint (Guardrail 1).
                    if not rc_warmup_captured:
                        _cap = _row_center_capture_gauge(model, optimizer)
                        if _cap is not None:
                            rc_warmup_mu0 = _cap["mu0"]
                            rc_warmup_mbar0 = _cap["mbar0"]
                            rc_warmup_captured = True
                            if ddp_rank == 0:
                                _mb = rc_warmup_mbar0.norm().item() if rc_warmup_mbar0 is not None else 0.0
                                logger.print_and_log(
                                    f"[row-center] warmup start @ {step}: captured "
                                    f"mu0={rc_warmup_mu0.norm().item():.4f} mbar0={_mb:.4f}"
                                )
                    if rc_warmup_captured:
                        rc_tel = _row_center_warmup_step(
                            model, optimizer, rc_s_eff, rc_warmup_mu0, rc_warmup_mbar0)
            else:
                rc_s_eff = 1.0
                rc_tel = _row_center_head_step(model, optimizer, want_exp_avg=True)
        # Synchronization slows down training, so rough (unsynchronized) timings are fine!
        #if device_type == "cuda":
        #    torch.cuda.synchronize()

        # Token accounting MUST advance identically on EVERY rank: the FFN pdr controller
        # (kv3) reads total_tokens_processed as its reference clock (observe -> reference(tok_m)),
        # and a rank-0-only counter would make the controller's per-rank FFN lr_scale writes
        # diverge and desync the sharded optimizer. The increment is deterministic and identical
        # across ranks (config/schedule-derived, no per-rank data), so no collective is needed.
        # (Previously this lived inside the rank-0 gate — harmless when only rank 0 logged it.)
        tokens_per_step = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)
        total_tokens_processed += tokens_per_step

        if ddp_rank == 0:
            dt = time.time() - t0
            tokens_per_sec = tokens_per_step / dt

            mfu = compute_mfu(tokens_per_sec, flops_per_token, ddp_world_size, settings.data_type) * 100
        
            main_loss_val = main_loss_accum.item()
            total_loss_val = total_loss_accum.item()
            # Perplexity is the language-model quality metric — compute against
            # the main NTP loss so it stays comparable to baseline / val_loss.
            ppl = math.exp(main_loss_val)
            # Suppress trunc_tag during SCS scaffold — `scs_tag` already
            # surfaces the active depth and a duplicated `trunc: 19/70`
            # alongside `scs: L18/70` reads as two unrelated systems.
            trunc_tag = f" | trunc: {active_layers if is_truncated else model_cfg.n_layers}/{model_cfg.n_layers}" if (truncator and truncator.enabled and not scaffold_mode) else ""
            bal_tag = f" | bal: {moe_stats[0]['avg_cv']:.3f}" if moe_stats[0] else ""
            drp_tag = f" | drp: {moe_stats[0]['drop_pct']:.2f}%" if moe_stats[0] and moe_stats[0]['drop_pct'] > 0 else ""
            # SCS scaffold tag — surfaces the active tap depth so it's obvious
            # at a glance which phase the run is in. Shows the tap depth
            # during scaffold; shows "done" post-cascade so operators joining
            # an SCS-enabled run mid-stream can still tell it's an SCS run.
            # Empty for non-SCS configs.
            if scaffold_mode:
                scs_tag = f" | scs: L{scs_deepest_tap}/{model_cfg.n_layers}"
            elif scs_enabled:
                scs_tag = " | scs: done"
            else:
                scs_tag = ""
            # Total (optimiser) loss only shown when it differs from main —
            # i.e., when at least one aux head has a nonzero current weight.
            tot_tag = ""
            tot_silent = ""
            if aux_heads_enabled and total_loss_val != main_loss_val:
                tot_tag = f" | tot_ls: {total_loss_val:.6f}"
                tot_silent = f"|tot_ls={total_loss_val:.6f}"
            aux_tag = ""
            aux_silent = ""
            if aux_loss_accum:
                aux_pairs = sorted((li, v.item()) for li, v in aux_loss_accum.items())
                aux_tag = " | " + " ".join(f"aux_l{li}: {v:.4f}" for li, v in aux_pairs)
                aux_silent = "|" + "|".join(f"aux_l{li}={v:.6f}" for li, v in aux_pairs)
            # Z-loss columns: zloss (raw mean(logZ**2), pre-alpha), logZ
            # (mean logZ), logZ_rms (sqrt mean logZ**2), logZ_p95 (tail), z_a
            # (effective alpha this step), and the HEAD-mechanism probe columns
            # (hd_wn = head weight norm, hd_gn = head grad norm GLOBAL/pre-clip,
            # hd_gr = head_grad/total_grad ratio — does z-loss touch the
            # readout?). Display-only diagnostics — the headline ls:/ppl above
            # stay PURE CE. Appended last so non-zloss log parsers are
            # unaffected. rms = sqrt(zloss); ratio uses `norm` (total grad-norm,
            # also pre-clip), so numerator and denominator are reduced
            # consistently.
            z_tag = ""
            z_silent = ""
            if zloss_enabled:
                zloss_val = zloss_accum.item()
                logZ_val = logZ_accum.item()
                logZ_rms = zloss_val ** 0.5 if zloss_val > 0 else 0.0
                hd_gr = (head_g_norm / norm) if (head_g_norm is not None and norm > 0) else 0.0
                z_tag = (f" | zloss: {zloss_val:.4f} | logZ: {logZ_val:.4f}"
                         f" | logZ_rms: {logZ_rms:.4f} | logZ_p95: {logZ_p95_last:.4f}"
                         f" | z_a: {zloss_alpha_eff:.4e}"
                         f" | hd_wn: {(head_w_norm or 0.0):.4f} | hd_gn: {(head_g_norm or 0.0):.4e}"
                         f" | hd_gr: {hd_gr:.4f}")
                z_silent = (f"|zloss={zloss_val:.6f}|logZ={logZ_val:.6f}"
                            f"|logZ_rms={logZ_rms:.6f}|logZ_p95={logZ_p95_last:.6f}"
                            f"|z_a={zloss_alpha_eff:.6e}"
                            f"|hd_wn={(head_w_norm or 0.0):.6f}|hd_gn={(head_g_norm or 0.0):.6e}"
                            f"|hd_gr={hd_gr:.6f}")
                # Snapshot the latest per-step z-loss stats so the val-cadence
                # diagnostics record (diagnostics.jsonl) can carry a structured
                # z_loss block. Per-step resolution stays in train_log.txt.
                zloss_diag = {
                    'zloss': zloss_val, 'logZ': logZ_val, 'logZ_rms': logZ_rms,
                    'logZ_p95': logZ_p95_last, 'z_a': zloss_alpha_eff,
                    'hd_wn': (head_w_norm or 0.0), 'hd_gn': (head_g_norm or 0.0),
                    'hd_gr': hd_gr,
                }
            # Row-center columns: rc_muW (PRE-projection ||mu(W)|| — the real
            # diagnostic: per-step gauge regrowth rate, i.e. how hard Adam is
            # fighting the projection), rc_muWp (POST ||mu(W)|| — ~0 by
            # construction; nonzero means something's broken), rc_mbar (1st-
            # moment gauge stripped this step), rc_ratio (||1 mu^T||_F/||W||_F).
            # Display-only; the headline ls/ppl stay untouched (function-
            # preserving op). Appended last so log parsers are unaffected.
            rc_tag = ""
            rc_silent = ""
            if row_center_enabled and rc_tel is not None:
                _mbar = rc_tel['m_bar'] if rc_tel['m_bar'] is not None else 0.0
                # proj_ratio only in the steady-state dict; warmup dict has 's'.
                _ratio = rc_tel.get('proj_ratio')
                _ratio_tag = f" | rc_ratio: {_ratio:.4f}" if _ratio is not None else ""
                _ratio_silent = f"|rc_ratio={_ratio:.6f}" if _ratio is not None else ""
                rc_tag = (f" | rc_muW: {rc_tel['mu_w_pre']:.4e} | rc_muWp: {rc_tel['mu_w_post']:.2e}"
                          f" | rc_mbar: {_mbar:.4e}{_ratio_tag}")
                rc_silent = (f"|rc_muW={rc_tel['mu_w_pre']:.6e}|rc_muWp={rc_tel['mu_w_post']:.6e}"
                             f"|rc_mbar={_mbar:.6e}{_ratio_silent}")
                rc_diag = {
                    'muW_pre': rc_tel['mu_w_pre'], 'muW_post': rc_tel['mu_w_post'],
                    'm_bar': _mbar, 'proj_ratio': _ratio,
                    's': rc_tel.get('s'),
                }
            # Guardrail 3: log the EFFECTIVE schedule scalars every step during a
            # staged transition (z-loss warmdown OR row-center warmup active or
            # pending) so the temporal separation is VISIBLE — we can confirm at a
            # glance that alpha hit exactly 0 before s went nonzero.
            sched_tag = ""
            if zloss_enabled or (row_center_enabled and row_center_warmup_on):
                sched_tag = f" | z_a_eff: {zloss_alpha_eff:.3e} | rc_s: {rc_s_eff:.4f}"
            # GPM: feed the tracker every step (keeps its rolling windows populated), but it's no
            # longer shown on the status line — the value goes to the train_log below for the
            # Dashboard's gpm chart. The per-group clip fields (gn_body/gn_head/gn_emb/gn_other/
            # clip_c) were retired from the line entirely (still computed under track_clip_groups).
            gpm_train_tag = ""
            if gpm_tracker is not None:
                gpm_tracker.update(norm, main_loss_val)
                gpm_train_tag = gpm_tracker.status_tag()   # ' | gpm: +0.31/+0.25' or ' | gpm: pending'
            # NOTE: body pdr is NOT on the per-step line — it's a val_step-cadence quantity,
            # emitted as its own [body-pdr] line in the diagnostics block below.
            logger.print_and_log(
                f"st: {step:5d} | ls: {main_loss_val:.6f} | ppl: {ppl:.2f} | lr: {lr:.4e} | nrm: {norm:.4f} [{clip_value:.1f}] | dt: {dt:.2f}s | t_tk: {total_tokens_processed:11,d} | tok/s: {tokens_per_sec:.0f} | MFU: {mfu:.0f}%{bal_tag}{drp_tag}{trunc_tag}{scs_tag}{tot_tag}{aux_tag}{z_tag}{rc_tag}{sched_tag}",
            )

            _sched_silent = f"|z_a_eff={zloss_alpha_eff:.6e}|rc_s={rc_s_eff:.6f}" if sched_tag else ""
            logger.print_and_log(
                f"{step:5d}|{main_loss_val:.6f}|{ppl:.2f}|{lr:.4e}|{norm:.4f}|{dt:.2f}|{total_tokens_processed:11d}|{tokens_per_sec:.0f}{tot_silent}{aux_silent}{z_silent}{rc_silent}{_sched_silent}{gpm_train_tag}",
                True, settings.train_log_file, silent=True
            )

            # Guardrail 5: transition health guard — grad-norm part (per logged
            # step). WARNING only; never auto-acts. _hg_nrm_run tracks repeats.
            # Suppressed during the warmup window (health_guard_warmup_steps, default
            # 100): a from-scratch / freshly-resumed run always has high grad-norm
            # while LR ramps from ~0, so warning every step there is pure noise that
            # trains you to ignore the line before a REAL spike appears.
            if health_guard_on and step >= health_guard_warmup_steps:
                if norm > 3.0:
                    _hg_nrm_run += 1
                else:
                    _hg_nrm_run = 0
                if norm > 5.0:
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {step}] nrm={norm:.2f} > 5 (single-step "
                        f"spike; cf. hard-branch nrm=5.61@18540). Advisory only — no auto-action.")
                elif _hg_nrm_run >= 3:
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {step}] nrm={norm:.2f} > 3 for {_hg_nrm_run} "
                        f"consecutive steps. Advisory only — no auto-action.")

            # SCS effective-LR debug line — shows per-compartment, output-head,
            # and per-aux-head lr_scale values at this step. Lets you eyeball
            # whether warmup ramps are being applied as expected. Easy to grep
            # away later once we're confident the schedule is firing right.
            if scs_enabled:
                _comp_parts = []
                for _start, _end, _act in scs_compartment_ranges:
                    _s = scs_compartment_lr_scale(_act, scs_warmup_steps, scs_init_mult, step)
                    if scaffold_mode and _start >= scs_active_layers:
                        _s = 0.0
                    _comp_parts.append(f"L{_start}-{_end}={_s:.3f}")
                _out_dbg = scs_compartment_lr_scale(
                    _cascade_complete_step, scs_warmup_steps, scs_init_mult, step,
                )
                _aux_dbg_parts = []
                for _li in sorted(aux_head_schedules):
                    _wn = aux_weights_now.get(_li, 0.0)
                    _frz = (
                        (scaffold_mode and scs_deepest_tap is not None and _li > scs_deepest_tap)
                        or _wn == 0.0
                    )
                    _aux_dbg_parts.append(f"L{_li}={0.0 if _frz else 1.0:.1f}")
                logger.print_and_log(
                    f"  ] scs_lr: {' '.join(_comp_parts)} | out={_out_dbg:.3f} | "
                    f"aux {' '.join(_aux_dbg_parts)}"
                )

            # Per-step aux head schedule weights (λ_i). Useful for sanity-
            # checking tot_ls = Σ λ_i × aux_l_i. Inactive taps (λ == 0)
            # are omitted to keep the line short.
            if aux_heads_enabled:
                _lam_parts = [
                    f"L{_li}={_w:.4f}"
                    for _li, _w in sorted(aux_weights_now.items())
                    if _w != 0.0
                ]
                if _lam_parts:
                    logger.print_and_log(f"  ] lambdas: {' '.join(_lam_parts)}")

        if step in settings.restart_steps:
            logger.print_and_log(f"Warm restart: LR {lr:.4e} → clip {clip_value:.2f}")  # TODO: Why am I showing the clip value here?

        if step % settings.val_step == 0 or last_step:
            sync_val_loader()
            do_validation(
                model, val_loader, device, settings.eval_iters, step, ddp_rank,
                settings.val_log_file, total_tokens_processed,
                ddp, ddp_world_size, settings.data_type, device_type,
                scaffold_mode=scaffold_mode,
                active_layers=scs_active_layers,
                scs_deepest_tap=scs_deepest_tap,
            )

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
            # Skipped during SCS scaffold — the main output head is frozen and
            # not in the active forward path, so the metric is undefined and
            # a full-depth forward would unshard frozen tail params (waste).
            if diagnostics is not None and not scaffold_mode:
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
            # Skipped during SCS scaffold — the probe forward would unshard
            # frozen tail layers + main head and capture meaningless RMS
            # values through them. Resumes cleanly post-cascade.
            activation_data = None
            if diagnostics is not None and not scaffold_mode:
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

            # Centered head-geometry health metrics (Item B). The CANONICAL
            # head-health metrics post-row-centering (raw hd_wn is gauge-
            # contaminated). Also the dn1-collapse early-warning: rising
            # spectral_concentration_c + eroding small singular values was the
            # death signature. Cheap [D,D] gram->eigh; computed only here at val
            # cadence. MUST run on all ranks (the gram all-reduce is collective).
            # centered_geometry ALWAYS subtracts the current row-mean now (Nexus
            # #163), so it reads the TRUE centered geometry at any ramp position —
            # no already_centered flag needed (a mid-ramp weight carries a residual
            # gauge that, unsubtracted, faked a rank-~1.6 collapse).
            cg_diag = _centered_geometry_step(model)

            # DATA-SIDE logZ_c at val cadence (bounded forward; NOT per-step — see
            # _logz_c_at_val). Logged regardless of z-loss so we always have the
            # centered log-partition trend (z-loss only ever surfaced RAW logZ).
            logz_c_val = None
            _was_training = model.training
            try:
                _vx, _vy = val_loader.next_batch(step=step)
                model.eval()
                logz_c_val = _logz_c_at_val(model, _vx.to(device))
            except Exception as _e:
                if ddp_rank == 0:
                    logger.print_and_log(f"  logZ_c@val failed: {type(_e).__name__}: {_e}")
            finally:
                if _was_training:
                    model.train()   # restore the mode we found
            # Fold the data-side logZ_c into the centered-geom record so it lands
            # in diagnostics.jsonl['centered_geom'] regardless of z-loss state.
            if logz_c_val is not None and cg_diag is not None:
                cg_diag['logZ_c'] = logz_c_val

            # Guardrail 5: transition health guard — geometry part (val cadence).
            # WARNING only; never auto-acts. Thresholds from the family calibration
            # (Nexus #156/#157): eff_rank_c < 7 warn, spec_conc_c > 0.45.
            if health_guard_on and cg_diag is not None and ddp_rank == 0:
                _er = cg_diag.get('effective_rank_c')
                _sc = cg_diag.get('spectral_concentration_c')
                if _er is not None and _er < 7.0:
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {step}] eff_rank_c={_er:.2f} < 7 "
                        f"(spec_conc_c={_sc:.3f}) — approaching low-rank collapse "
                        f"(dead dn1@14k=4.3). Advisory only; investigate, do NOT engage z-loss.")
                elif _sc is not None and _sc > 0.45:
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {step}] spec_conc_c={_sc:.3f} > 0.45 "
                        f"(eff_rank_c={_er:.2f}). Advisory only — watch for persistence.")

            # Log layer diagnostics after validation
            if diagnostics is not None:
                awd_diag = awd.get_diagnostics_data() if awd is not None else None
                moe_diag = moe_stats[0] if moe_stats[0] else None
                # dn4 head-hygiene ||Ubar|| pre/post must join the centered-geom record
                # BEFORE log_diagnostics() writes the jsonl line — folding it after the
                # write (as the [head-gauge] print block below once did) left the values
                # gen_log-only, invisible to every diagnostics.jsonl consumer.
                if head_gauge_enabled and cg_diag is not None:
                    cg_diag['head_ubar_pre'] = getattr(optimizer, '_last_head_ubar_pre', None)
                    cg_diag['head_ubar_post'] = getattr(optimizer, '_last_head_ubar_post', None)
                snapshot = diagnostics.log_diagnostics(
                    step, settings.nas_path, total_tokens_processed,
                    awd_data=awd_diag, moe_data=moe_diag,
                    activation_data=activation_data,
                    zloss_data=zloss_diag,
                    rc_data=rc_diag,
                    cg_data=cg_diag,
                )
                diagnostics.print_summary(snapshot, logger, awd_data=awd_diag, moe_data=moe_diag)
                # Body relative-step pdr = ||dW||/||W|| (Math Brief #6 annealing experiment).
                # Emitted as its OWN line on diagnostics/val steps — it's a snapshot-cadence
                # quantity (computed once per val_step), so it does NOT belong on the per-step
                # status line where it would imply per-step resolution it doesn't have. Shows the
                # measured median pdr (attn/ffn split) alongside the commanded body-LR mult, so the
                # anneal is legible: watch pdr bend down after the lr_mult starts dropping (~st 2680).
                if ddp_rank == 0:
                    try:
                        _att = [l.attn.param_delta_ratio for l in snapshot.layers if l.attn.param_delta_ratio is not None]
                        _ffn = [l.ffn.param_delta_ratio for l in snapshot.layers if l.ffn.param_delta_ratio is not None]
                        _all = sorted(_att + _ffn)
                        if _all:
                            _med = _all[len(_all) // 2]
                            _amed = sorted(_att)[len(_att) // 2] if _att else float('nan')
                            _fmed = sorted(_ffn)[len(_ffn) // 2] if _ffn else float('nan')
                            # Commanded LR multipliers this step, DECOMPOSED attn vs ffn — read
                            # straight from the side-dict the optimizer actually uses (so it's the
                            # REAL applied value, whether set by lr_mods OR the kv3 controller).
                            # attn and ffn can now be on different schedules, so a single body mult
                            # would be ambiguous. Each group is uniform internally, so one
                            # representative param per group is exact. Default 1.0 if uncovered.
                            def _rep_mult(_ids):
                                if lr_scale_overrides is not None:
                                    for _pid in _ids:
                                        return lr_scale_overrides.get(_pid, 1.0)
                                return 1.0
                            _amult = _rep_mult(_attn_param_ids_for_pdr)
                            _fmult = _rep_mult(_ffn_param_ids_for_ctrl)
                            logger.print_and_log(
                                f"  [body-pdr] pdr={_med:.3e} (attn={_amed:.3e} ffn={_fmed:.3e}) "
                                f"| lr_mult attn={_amult:.3f} ffn={_fmult:.3f}")
                    except Exception:
                        pass
                    # Tangent-projection strength f at this val step. f is written into the Muon
                    # body groups every step (~line 1994); surface it here so the ramp is legible --
                    # ||W|| only visibly flattens as f->1 (much later), so this is the DIRECT readout.
                    if _tp_on:
                        _f_log = interpolate_lr_mod(_tps_cfg, step) if _tps_is_sched else float(_tps_cfg)
                        logger.print_and_log(
                            f"  [tangent] f={_f_log:.4f}  (radial growth removed {100*_f_log:.1f}%; "
                            f"||W|| grows at {100*(1.0 - _f_log):.1f}% of natural rate)")
                # FFN pdr controller (kv3): feed the FFN-median pdr at this diagnostic cadence.
                # MUST run on ALL ranks with the identical all-reduced param_delta_ratio so
                # body_lr_ctrl.m stays bit-identical across ranks — the per-step actuator write
                # applies m on every rank, and any cross-rank drift would desync the optimizer.
                # (snapshot is built on all ranks; param_delta_ratio is post-all_reduce.) Log r0 only.
                if body_lr_ctrl is not None and body_lr_ctrl.enabled:
                    # Current tangent-projection f (all ranks, deterministic). The 'auto' controller
                    # latches its self-anchored reference only once f>=1 (the true freeze point); 0.0
                    # when projection is off. scheduled_lr (the live cosine body LR) feeds the LR-track
                    # reference r=K_anchor*lr/lr_anchor. Both are ignored in 'knots' mode.
                    _f_now_ctrl = ((interpolate_lr_mod(_tps_cfg, step) if _tps_is_sched else float(_tps_cfg))
                                   if _tp_on else 0.0)
                    try:
                        # Drop None AND NaN (x == x is False for NaN) so one bad layer can't make
                        # the median NaN; the controller additionally rejects non-finite _fmed_c.
                        _ffn_pdr = [l.ffn.param_delta_ratio for l in snapshot.layers
                                    if l.ffn.param_delta_ratio is not None
                                    and l.ffn.param_delta_ratio == l.ffn.param_delta_ratio]
                        _fmed_c = sorted(_ffn_pdr)[len(_ffn_pdr) // 2] if _ffn_pdr else None
                        _radial = ({_nm: (_shadow_R[_nm], _shadow_dR.get(_nm, 0.0),
                                          _shadow_gamma.get(_nm, 0.0)) for _nm in _shadow_R}
                                   if _shadow_ctrl_on else None)
                        _eta_acc = _shadow_eta['sum'] if _shadow_ctrl_on else None
                        if _shadow_ctrl_on:
                            # Reset the ΔR_free window BEFORE observe (the snapshot above already
                            # captured it) so a raising observe() can't leave a dirty window that
                            # double-counts into S next time. Latest R/γ are kept (overwritten/step).
                            _shadow_dR.clear()
                            _shadow_eta['sum'] = 0.0
                        body_lr_ctrl.observe(step, total_tokens_processed / 1e6, _fmed_c,
                                             scheduled_lr=scheduled_lr, f_now=_f_now_ctrl,
                                             radial=_radial, eta_accum=_eta_acc)
                        # Stamp the LR-schedule fingerprint WITH the anchor (all ranks, deterministic): it
                        # must describe the schedule the anchor was actually captured against — NOT the
                        # startup schedule, since a pre-anchor resume could have changed it. The resume
                        # guard compares against this. Runs on every rank so the checkpointed value matches.
                        if getattr(body_lr_ctrl, '_just_latched', False):
                            body_lr_ctrl.lr_fingerprint = _lr_schedule_fingerprint(settings)
                        if ddp_rank == 0:
                            # auto-mode: announce the anchor latch + any (loose) sanity warning, once.
                            if getattr(body_lr_ctrl, '_just_latched', False):
                                logger.print_and_log(
                                    f"  [ffn-ctrl] AUTO anchor LATCHED @ {step}: "
                                    f"K_anchor={body_lr_ctrl.K_anchor:.3e} lr_anchor={body_lr_ctrl.lr_anchor:.3e} "
                                    f"(geomean of {body_lr_ctrl.anchor_samples} post-freeze samples) — "
                                    f"reference now r=K_anchor*lr/lr_anchor")
                                if body_lr_ctrl.anchor_warn:
                                    logger.print_and_log(
                                        f"  [HEALTH WARNING @ {step}] FFN-ctrl auto-anchor: {body_lr_ctrl.anchor_warn}")
                            _ln = body_lr_ctrl.log_line()
                            if _ln:
                                logger.print_and_log(_ln)
                            # Surface guardrails as prominent HEALTH WARNING lines (parallel to the
                            # geometry guard) so an out-of-authority controller isn't buried inline.
                            if body_lr_ctrl.alarm:
                                logger.print_and_log(
                                    f"  [HEALTH WARNING @ {step}] FFN-ctrl base-LR-too-high: m pinned "
                                    f"at floor {body_lr_ctrl.m_floor} while pdr_ffn > "
                                    f"{body_lr_ctrl.alarm_pdr_ratio}*r for {body_lr_ctrl.alarm_consecutive}+ "
                                    f"samples — lower base body/FFN LR (advisory).")
                            elif body_lr_ctrl.upper_alarm:
                                logger.print_and_log(
                                    f"  [HEALTH WARNING @ {step}] FFN-ctrl NO UPWARD AUTHORITY: m pegged at "
                                    f"m_max={body_lr_ctrl.m_max} while the unclamped demand wants more "
                                    f"(m_raw={body_lr_ctrl._m_ff_raw:.2f}) — body is BELOW target (cooler than "
                                    f"reference). Amplification is deliberately forbidden, so this is "
                                    f"INFORMATIONAL: anchor/reference too high, base LR too low, or the body "
                                    f"cooling faster than asked. Worrying only if the run also underfits.")
                            elif body_lr_ctrl.inspect:
                                logger.print_and_log(
                                    f"  [HEALTH WARNING @ {step}] FFN-ctrl m={body_lr_ctrl.m:.3f} < "
                                    f"{body_lr_ctrl.authority_low_m} before the merge region — inspect (advisory).")
                            # Prolonged staleness = the controller has stopped controlling (pdr
                            # measurement broken / disengaged). Surface it at HEALTH-WARNING severity
                            # so a silently-frozen m isn't hidden behind an inline STALE suffix.
                            if body_lr_ctrl._dropped >= 3:
                                logger.print_and_log(
                                    f"  [HEALTH WARNING @ {step}] FFN-ctrl STALLED: {body_lr_ctrl._dropped} "
                                    f"consecutive dropped pdr samples — m HELD at {body_lr_ctrl.m:.3f}, "
                                    f"controller not actuating. Check FFN pdr measurement.")
                            # auto mode never latched, well past the freeze point = body running UNCONTROLLED
                            # with no closed loop. Surface loudly (the inline 'AUTO pre-anchor' line is easy
                            # to miss). Validation already requires f->1, so this catches the residual causes
                            # (pdr never measured, anchor_step set beyond where f actually reaches 1, etc.).
                            if (getattr(body_lr_ctrl, 'ref_mode', '') == 'auto' and not body_lr_ctrl.anchor_set
                                    and step > body_lr_ctrl.anchor_step
                                    + max(3000, body_lr_ctrl.anchor_samples * int(getattr(settings, 'val_step', 100)) * 5)):
                                logger.print_and_log(
                                    f"  [HEALTH WARNING @ {step}] FFN-ctrl AUTO has NOT anchored "
                                    f"{step - body_lr_ctrl.anchor_step} steps past anchor_step="
                                    f"{body_lr_ctrl.anchor_step} — m HELD at 1.0, body UNCONTROLLED. Check that "
                                    f"tangent_project_strength reaches 1.0 and FFN pdr is being measured "
                                    f"(collected {len(body_lr_ctrl._anchor_buf)}/{body_lr_ctrl.anchor_samples}).")
                    except Exception as _e:
                        # Do NOT silently swallow: a raising observe() means m is HELD (stale) and
                        # the controller has stopped controlling. Inputs are all-reduce-identical so
                        # this raises identically on every rank (no divergence) — log it loudly.
                        if ddp_rank == 0:
                            logger.print_and_log(
                                f"  [ffn-ctrl] observe FAILED @ {step}: {type(_e).__name__}: {_e} — "
                                f"m HELD at {body_lr_ctrl.m:.3f}, controller STALLED")
                    # Auto-anchor sanity FATAL: the captured anchor is implausible vs the (loose) bands.
                    # All ranks see identical all-reduced inputs, so this fires identically — and it is
                    # OUTSIDE the try so fatal_error (which exits) is not swallowed by the except above.
                    if getattr(body_lr_ctrl, 'anchor_fatal', None):
                        fatal_error(f"ffn_pdr_controller auto-anchor capture is implausible: "
                                    f"{body_lr_ctrl.anchor_fatal}")
                if cg_diag is not None and ddp_rank == 0:
                    _lzc = cg_diag.get('logZ_c')
                    _lzc_tag = f" logZ_c={_lzc:.2f}" if _lzc is not None else ""
                    logger.print_and_log(
                        f"  [centered-geom]{_lzc_tag} ||W_c||={cg_diag['Wc_fro']:.2f} "
                        f"gauge||mu||={cg_diag.get('mu_w_norm', 0.0):.3e} "
                        f"({100 * cg_diag.get('gauge_frac', 0.0):.1f}% of head) "
                        f"s1_c={cg_diag['s1_c']:.2f} "
                        f"spec_conc_c={cg_diag['spectral_concentration_c']:.4f} "
                        f"eff_rank={cg_diag['effective_rank_c']:.1f} "
                        f"small_sig(p1/p5/p10)={cg_diag['small_sigma_p1']:.3f}/"
                        f"{cg_diag['small_sigma_p5']:.3f}/{cg_diag['small_sigma_p10']:.3f}"
                    )

                # dn4 head-hygiene: surface the per-step head gauge magnitude removed
                # from the head's Adam update (||Ubar|| pre; post ~0 confirms the SR
                # write-back landed). Already folded into centered_geom above (before
                # the jsonl write); this is just the gen_log surface.
                if head_gauge_enabled:
                    _ub_pre = getattr(optimizer, '_last_head_ubar_pre', None)
                    _ub_post = getattr(optimizer, '_last_head_ubar_post', None)
                    if ddp_rank == 0 and _ub_pre is not None:
                        _pt = f" -> {_ub_post:.2e}" if _ub_post is not None else ""
                        logger.print_and_log(
                            f"  [head-gauge] ||Ubar||(removed from head update)={_ub_pre:.3e}{_pt}")

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
            # Row-center warmup targets to persist (Guardrail 1): only meaningful
            # once captured (warmup started). start_step/duration recorded so a
            # forensic tool can see which schedule produced the checkpoint.
            _rc_ws = None
            if row_center_enabled and row_center_warmup_on and rc_warmup_captured:
                _rc_ws = {
                    "mu0": rc_warmup_mu0, "mbar0": rc_warmup_mbar0,
                    "start_step": row_center_warmup_start,
                    "duration": int(_rc_warmup.get('duration_steps', 0)),
                }
            save_model(model, optimizer, model_cfg, step, ddp_rank, ddp_local_rank, train_loader, total_tokens_processed, settings, awd=awd, body_lr_ctrl=body_lr_ctrl, rc_warmup_state=_rc_ws)

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
def save_model(model, optimizer, model_config, step, ddp_rank, ddp_local_rank, train_loader, total_tokens_processed, settings, awd=None, body_lr_ctrl=None, rc_warmup_state=None):
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
            # Auxiliary prediction heads — so neo_common's filter-based ModelArgs
            # rebuild at inference time picks them up and the state-dict load
            # is clean (no "unexpected keys" warning for aux_heads.*.{norm,linear}.weight).
            'aux_head_layers': list(getattr(model_config, 'aux_head_layers', []) or []),
        }
        # SCS knobs — recorded so a forensic tool / dashboard can tell
        # whether a checkpoint was produced under scaffold and with which
        # warmup schedule. Resume still reads these from YAML (settings),
        # not from the checkpoint, but persisting them here makes a
        # silent settings drift visible after the fact.
        _scs_cfg_for_save = getattr(settings, 'auxiliary_heads', None) or {}
        scs_settings_snapshot = {
            'compute_inactive_layers': _scs_cfg_for_save.get('compute_inactive_layers', True),
            'new_layer_warmup_steps': _scs_cfg_for_save.get('new_layer_warmup_steps', 0),
            'new_layer_lr_multiplier': _scs_cfg_for_save.get('new_layer_lr_multiplier', 1.0),
        } if isinstance(_scs_cfg_for_save, dict) else None

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
            "scs_settings": scs_settings_snapshot,
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

    # Save FFN pdr controller state (rank 0 only — bit-identical across ranks). Mirrors AWD.
    if body_lr_ctrl is not None and getattr(body_lr_ctrl, 'enabled', False) and ddp_rank == 0:
        ctrl_path = os.path.join(settings.local_checkpoint_dir, f"bodylr_state_step_{step:06d}.pt")
        torch.save(body_lr_ctrl.state_dict(), ctrl_path)
        logger.print_and_log(f"  ] FFN-ctrl state saved")

    # Save row-center WARMUP targets (Guardrail 1). The target-gauge schedule
    # anchors to the gauge captured at warmup start (mu0/mbar0); a 200-step warmup
    # spans checkpoints, so on resume we MUST restore these rather than recapture
    # from the now-partially-centered head (which would silently re-anchor the ramp
    # to a smaller gauge and stall it). Global D-vectors, identical across ranks ->
    # rank 0 only. Only written while a capture exists (i.e. warmup has started).
    if rc_warmup_state is not None and rc_warmup_state.get("mu0") is not None and ddp_rank == 0:
        rc_path = os.path.join(settings.local_checkpoint_dir, f"rowcenter_warmup_step_{step:06d}.pt")
        torch.save({
            "mu0": rc_warmup_state["mu0"],          # fp32 CPU [D]
            "mbar0": rc_warmup_state.get("mbar0"),  # fp32 CPU [D] or None
            "start_step": rc_warmup_state.get("start_step"),
            "duration": rc_warmup_state.get("duration"),
        }, rc_path)
        logger.print_and_log(f"  ] row-center warmup targets saved")

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
def resume_training(model, optimizer, train_loader, ddp_rank, settings, grad_accum_schedule, awd=None, body_lr_ctrl=None):
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

    # REMOVAL-direction reconciliation (aux heads turned OFF at resume): when the
    # auxiliary_heads block is removed from config, the model no longer constructs
    # aux_heads.* but the checkpoint still contains those (~251.7M) params -> they'd
    # be UNEXPECTED keys and strict load would RuntimeError. Drop the orphaned
    # aux_heads.* keys from the loaded state_dict so the slimmed model loads cleanly
    # (and strict stays valid, still catching any OTHER unexpected key). Scoped to
    # aux_heads.* and only fires when the model has no aux-head params at all, so
    # it's a no-op on normal resumes and on the turn-ON path below.
    model_keys = set(raw_model.state_dict().keys())
    model_has_aux = any(k.startswith('aux_heads.') for k in model_keys)
    if not model_has_aux:
        orphan_aux = [k for k in state_dict if k.startswith('aux_heads.')]
        if orphan_aux:
            for k in orphan_aux:
                del state_dict[k]
            logger.print_and_log(
                f"  ] aux heads removed from config — dropped {len(orphan_aux)} "
                f"orphaned aux_heads.* param(s) from the checkpoint load (the head "
                f"weights remain in the checkpoint file; just not loaded)."
            )

    # Detect new modules added since the checkpoint (e.g. auxiliary heads when
    # the intervention is being turned on at resume). Missing keys are tolerated
    # so freshly-initialized weights survive the load; unexpected keys would
    # still indicate a real config mismatch and we want to know about those.
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
                # Under SCS, scaffold-frozen params (the optimiser's
                # early-return skips them entirely) never had state created
                # on save — missing state on resume is expected for those.
                # Downgrade WARNING → NOTE so an operator resuming a
                # scaffold run doesn't think the checkpoint is corrupt.
                _scs_on = (
                    isinstance(getattr(settings, 'auxiliary_heads', None), dict)
                    and settings.auxiliary_heads.get('enabled', False)
                    and not settings.auxiliary_heads.get('compute_inactive_layers', True)
                )
                _kind = "NOTE (SCS scaffold)" if _scs_on else "WARNING"
                logger.print_and_log(
                    f"  ] {log_prefix}{_kind}: optimizer state missing "
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

        def _drop_removed_aux_state(saved_sd: dict, log_prefix: str = ""):
            """Symmetric to _inject_empty_state_for_new_params, for the aux-heads-
            REMOVED case: when the current model has no aux_heads.* params but the
            saved optimizer state does, strip those orphaned entries from both
            'state' and each param_group's 'params' FQN list. Otherwise
            set_optimizer_state_dict would try to map state onto params that no
            longer exist (or trip the param_groups<->params consistency checks).
            No-op when the model still has aux heads (normal resume / turn-ON)."""
            model_fqns = set(name for name, _ in raw_model.named_parameters())
            if any(f.startswith('aux_heads.') for f in model_fqns):
                return  # model still has aux heads -> nothing to drop
            st = saved_sd.get('state', {})
            orphan = [k for k in st if k.startswith('aux_heads.')]
            for k in orphan:
                del st[k]
            n_pg = 0
            for pg in saved_sd.get('param_groups', []):
                if 'params' in pg:
                    before = len(pg['params'])
                    pg['params'] = [p for p in pg['params'] if not (isinstance(p, str) and p.startswith('aux_heads.'))]
                    n_pg += before - len(pg['params'])
            if orphan or n_pg:
                logger.print_and_log(
                    f"  ] {log_prefix}aux heads removed — dropped {len(orphan)} "
                    f"orphaned optimizer-state entr(ies) + {n_pg} param-group ref(s)."
                )

        if use_full_optim:
            # Full state dict — all ranks load the same file, FSDP distributes automatically.
            # Strip param_group metadata: the current optimizer already has the correct
            # param_groups. Only the state tensors (momentum, exp_avg, etc.) need
            # redistribution. Custom param_group keys (MuonFSDP2's momentum/beta2 vs
            # Adam's betas) cause KeyErrors in _unflatten_state_dict if left in.
            optim_sd = torch.load(optim_full_path, map_location="cpu")
            # Drop orphaned aux-head state/refs FIRST (aux-heads-removed case) so the
            # param_group 'params' lists below match the slimmed model.
            _drop_removed_aux_state(optim_sd)
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
            _drop_removed_aux_state(optim_shard, log_prefix=f"[R{ddp_rank}] ")
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

    # STEP 3c: RESTORE FFN pdr controller state (if available). Mirrors AWD: all ranks load the
    # identical state. Missing file is non-fatal in knots mode (controller restarts at m=1.0), but in
    # 'auto' mode a missing file on a POST-ANCHOR resume is FATAL (the self-anchored reference is lost).
    if body_lr_ctrl is not None and getattr(body_lr_ctrl, 'enabled', False):
        ctrl_path = os.path.join(pth, f"bodylr_state_step_{shard_step:06d}.pt")
        _ref_mode = getattr(body_lr_ctrl, 'ref_mode', 'knots')
        _ctrl_auto = _ref_mode == 'auto'
        _ctrl_shadow = _ref_mode in ('auto_shadow_growth', 'auto_shadow_partial')
        # no-handoff modes (partial, OR growth with freeze_handoff:false) never latch a frozen tail;
        # the shadow S integral carries the ENTIRE history, so a missing S after f>0 is fatal — vs the
        # growth+handoff case, whose distinct hazard is re-capturing r_freeze on an already-frozen body.
        _no_handoff = (_ref_mode == 'auto_shadow_partial') or \
            (_ref_mode == 'auto_shadow_growth' and not bool(getattr(body_lr_ctrl, 'freeze_handoff', True)))
        # f at the resume step — shadow resume guards key off the PROJECTION schedule (engagement is
        # f-gated), not warmup_step. Reconstructed from the same source the train loop uses.
        _tps_cfg_r = getattr(settings, 'tangent_project_strength', 1.0)
        _f_resume = ((interpolate_lr_mod(_tps_cfg_r, shard_step) if isinstance(_tps_cfg_r, list)
                      else float(_tps_cfg_r)) if bool(getattr(settings, 'tangent_project', False)) else 0.0)
        if os.path.exists(ctrl_path):
            body_lr_ctrl.load_state_dict(torch.load(ctrl_path, weights_only=True))
            # LR-schedule guard: the auto anchor (r=K_anchor*lr/lr_anchor) AND the shadow frozen tail
            # (r=r_freeze*lr/lr_freeze) both ride lr(t) LITERALLY. If the schedule changed since the
            # latch, the cancellation is invalid and the controller fights the new curve. Fatal.
            if (_ctrl_auto and body_lr_ctrl.anchor_set) or (_ctrl_shadow and body_lr_ctrl.frozen):
                _cur_fp = _lr_schedule_fingerprint(settings)
                _anc_fp = body_lr_ctrl.lr_fingerprint
                if _anc_fp is None:
                    # Latched but no fingerprint recorded (hand-edited / pre-feature checkpoint): can't
                    # verify the schedule is unchanged. Warn loudly rather than silently skip the guard.
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {shard_step}] FFN-ctrl resume: reference is LATCHED but NO LR "
                        f"fingerprint was recorded — cannot verify the LR schedule is unchanged. If any LR "
                        f"setting changed, the LR-track reference is invalid and the controller may fight "
                        f"the new curve.", r0_only=True)
                elif tuple(_anc_fp) != _cur_fp:
                    fatal_error(
                        f"FFN-ctrl ({_ref_mode}): the LR schedule changed across resume (latched under "
                        f"{tuple(_anc_fp)}, now {_cur_fp}) — the LR-track reference cancellation is "
                        f"invalid and the controller would fight the new curve. Restore the original LR "
                        f"settings.")
            # freeze_handoff guard: the flag is CONFIG-authoritative (not restored from the checkpoint),
            # but it gates the post-f=1 LAW. Flipping it across a resume silently switches that law —
            # true->false abandons the latched LR-track tail for continuous R/S (a kink at the seam);
            # false->true RE-CAPTURES r_freeze at the resume step (rebasing the experiment, the exact
            # hazard the missing-state fatals forbid). Once the controller has ENGAGED (shadow_active /
            # frozen / f>0), a mismatch is fatal; pre-engagement (f=0, no history) it's a benign warn.
            if _ctrl_shadow:
                _ck_fho = getattr(body_lr_ctrl, '_ckpt_freeze_handoff', None)
                _engaged = (getattr(body_lr_ctrl, 'shadow_active', False)
                            or getattr(body_lr_ctrl, 'frozen', False) or _f_resume > 0.0)
                if _ck_fho is not None and bool(_ck_fho) != bool(body_lr_ctrl.freeze_handoff):
                    if _engaged:
                        fatal_error(
                            f"FFN-ctrl ({_ref_mode}): freeze_handoff changed across resume "
                            f"(checkpointed {bool(_ck_fho)}, config now {bool(body_lr_ctrl.freeze_handoff)}) "
                            f"while the controller is ENGAGED (f={_f_resume:.3f}). This silently switches the "
                            f"post-f=1 control law (LR-track tail <-> continuous R/S) — true->false kinks the "
                            f"seam, false->true re-captures r_freeze at the resume step and rebases the run. "
                            f"Restore the original freeze_handoff value.")
                    else:
                        logger.print_and_log(
                            f"  [HEALTH WARNING @ {shard_step}] FFN-ctrl resume: freeze_handoff differs from "
                            f"the checkpoint ({bool(_ck_fho)} -> {bool(body_lr_ctrl.freeze_handoff)}) but the "
                            f"controller has not engaged yet (f=0) — allowed (no history to rebase).",
                            r0_only=True)
                elif _ck_fho is None and _engaged:
                    logger.print_and_log(
                        f"  [HEALTH WARNING @ {shard_step}] FFN-ctrl resume: checkpoint predates the "
                        f"freeze_handoff field — cannot verify it is unchanged. If it was flipped, the "
                        f"post-f=1 law switches silently.", r0_only=True)
            logger.print_and_log(f"  ] FFN-ctrl state restored", r0_only=True)
        elif _ctrl_auto and shard_step >= (body_lr_ctrl.anchor_step
                + body_lr_ctrl.anchor_samples * int(getattr(settings, 'val_step', 100))):
            # PAST the earliest possible anchor-latch point with the anchor state MISSING: we cannot tell
            # whether the anchor was already latched, and re-capturing on a (now possibly frozen) body would
            # silently rebase the experiment. Fatal, not a silent re-capture (Math Q11). Before this point
            # the run could not have collected enough samples to latch, so a missing file there falls through
            # to the generic post-warmup warn below and simply re-anchors fresh.
            fatal_error(
                f"FFN-ctrl auto-mode resume @ {shard_step} is past the earliest anchor-latch point "
                f"(anchor_step {body_lr_ctrl.anchor_step} + {body_lr_ctrl.anchor_samples} samples) but the "
                f"controller state file is MISSING ({os.path.basename(ctrl_path)}). Cannot determine whether "
                f"the anchor was latched; refusing to risk a silent re-capture on a possibly-frozen body. "
                f"Restore the bodylr_state_*.pt checkpoint.")
        elif _ctrl_shadow and _ref_mode == 'auto_shadow_growth' and not _no_handoff and _f_resume >= 1.0 - 1e-6:
            # POST-FREEZE with shadow state MISSING (handoff ON): re-entering the ramp would re-capture
            # r_freeze on an already-frozen body, rebasing the run. Fatal, not a silent re-capture (spec §8).
            fatal_error(
                f"FFN-ctrl auto_shadow_growth resume @ {shard_step} is POST-FREEZE (f={_f_resume:.3f}) but "
                f"the controller state is MISSING ({os.path.basename(ctrl_path)}) — refusing to silently "
                f"re-capture r_freeze on an already-frozen body (would rebase the run). Restore the "
                f"bodylr_state_*.pt checkpoint.")
        elif _ctrl_shadow and _no_handoff and _f_resume > 0.0:
            # No-handoff (partial, or growth+freeze_handoff:false): never freezes; S carries the ENTIRE
            # accumulated history. A fresh S would zero the integral -> m snaps to 1.0, erasing all
            # controller history. Fatal (spec §8).
            fatal_error(
                f"FFN-ctrl {_ref_mode} (no-handoff) resume @ {shard_step} has f={_f_resume:.3f}>0 but the "
                f"controller state is MISSING ({os.path.basename(ctrl_path)}) — a fresh S would zero the "
                f"accumulated shadow integral (m -> 1.0, erasing all controller history). Restore the "
                f"bodylr_state_*.pt checkpoint.")
        elif shard_step > body_lr_ctrl.warmup_step:
            # Resuming PAST warmup with no controller state = the learned K and annealed m are lost;
            # the controller restarts at m=1.0 and re-warms the FFN LR back up over several thousand
            # steps, perturbing the very anneal the run is testing. Surface at HEALTH-WARNING severity.
            # (Auto pre-anchor lands here too: benign — it simply re-anchors later.)
            logger.print_and_log(
                f"  [HEALTH WARNING @ {shard_step}] FFN-ctrl state not found on a post-warmup resume "
                f"— controller RESET to m=1.0; the FFN angular-LR anneal will transiently REHEAT.",
                r0_only=True)
        else:
            logger.print_and_log(f"  ] FFN-ctrl state not found — starting fresh", r0_only=True)

    # ---------------------------------------------------------------------------
    # Restore row-center WARMUP targets (Guardrail 1). If a checkpoint was saved
    # mid-warmup, the captured mu0/mbar0 are restored and stashed on train_loop so
    # the schedule re-anchors to the ORIGINAL start gauge — NOT recaptured from the
    # now-partially-centered head (which would shrink the anchor and stall the
    # ramp). Stash on the function object since train_loop reads it at loop top.
    # Clear any stale stash first so a fresh (non-mid-warmup) resume starts clean.
    train_loop._rc_restore_mu0 = None
    train_loop._rc_restore_mbar0 = None
    rc_warmup_path = os.path.join(pth, f"rowcenter_warmup_step_{shard_step:06d}.pt")
    if os.path.exists(rc_warmup_path):
        _rcw = torch.load(rc_warmup_path, weights_only=True)
        train_loop._rc_restore_mu0 = _rcw.get("mu0")
        train_loop._rc_restore_mbar0 = _rcw.get("mbar0")
        _mu0n = train_loop._rc_restore_mu0.norm().item() if train_loop._rc_restore_mu0 is not None else 0.0
        logger.print_and_log(
            f"  ] row-center warmup targets restored (mu0={_mu0n:.4f}, "
            f"start={_rcw.get('start_step')}, dur={_rcw.get('duration')}) "
            f"— resuming mid-warmup, schedule re-anchored to original gauge",
            r0_only=True)

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
    if start_step >= settings.max_steps:
        # resuming a checkpoint saved at the final step: nothing left to train — exit
        # gracefully instead of IndexError-ing on the schedule lookup below
        logger.print_and_log(
            f"  ] Resumed @ step {start_step}: run already complete "
            f"(max_steps={settings.max_steps}); training loop will no-op.")
        dist.barrier()
        return start_step, total_tokens_processed
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

    # A schedule that starts past step 0 would leave grad_accum=0 for the leading
    # steps: the micro-batch loop never runs, optimizer.step() fires on empty grads,
    # and the log shows ls: 0.000000 — silent no-training. Fail at build time.
    if not sorted_schedule or sorted_schedule[0][0] != 0:
        fatal_error(
            f"ga_schedule must start at step 0 (first waypoint is "
            f"{sorted_schedule[0][0] if sorted_schedule else 'MISSING'}) — steps before the "
            f"first waypoint would train NOTHING (grad_accum=0) while logging ls: 0.")

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

        # --- ffn_pdr_controller: FFN-only pdr feedback controller (kv3) ---
        # Restores the relative-LR anneal via FFN angular-step control. Collapse to None
        # unless enabled; light validation (full semantics in body_lr_controller.py).
        # docs/KV3_CONTROLLER_DESIGN.md. MUST NOT coexist with AWD (AWD moves ||W|| = pdr's
        # denominator) — they're mutually exclusive controllers on the body.
        if not hasattr(self, 'ffn_pdr_controller'):
            self.ffn_pdr_controller = None
        elif isinstance(self.ffn_pdr_controller, dict):
            if not self.ffn_pdr_controller.get('enabled', False):
                self.ffn_pdr_controller = None
            else:
                _c = self.ffn_pdr_controller
                _mf = _c.get('m_floor', 0.30)
                if not isinstance(_mf, (int, float)) or not (0.0 < float(_mf) <= 1.0):
                    fatal_error(f"ffn_pdr_controller.m_floor must be in (0,1], got {_mf!r}")
                _mx = float(_c.get('m_max', 1.0))
                if _mx > 1.0:
                    fatal_error("ffn_pdr_controller.m_max must be <= 1.0 (no body-LR amplification)")
                if float(_mf) > _mx:
                    fatal_error(f"ffn_pdr_controller.m_floor ({_mf}) must be <= m_max ({_mx}) — "
                                "else the output clamp pins m above m_max.")
                for _ak in ('k_ema_alpha', 'pdr_ema_alpha'):
                    _av = float(_c.get(_ak, 0.15))
                    if not (0.0 < _av <= 1.0):
                        fatal_error(f"ffn_pdr_controller.{_ak} must be in (0,1], got {_av}")
                if not (0.0 <= float(_c.get('rate_down', 0.05)) < 1.0):
                    fatal_error("ffn_pdr_controller.rate_down must be in [0,1)")
                if float(_c.get('rate_up', 0.02)) < 0.0:
                    fatal_error("ffn_pdr_controller.rate_up must be >= 0")
                _ws = int(_c.get('warmup_step', 1500))
                _mxs = int(getattr(self, 'max_steps', 0) or 0)
                if _ws < 0 or (_mxs and _ws >= _mxs):
                    fatal_error(f"ffn_pdr_controller.warmup_step ({_ws}) must be in [0, max_steps) — "
                                f"else the controller stays warmup-gated for the whole run.")
                if int(_c.get('alarm_consecutive', 3)) < 1:
                    fatal_error("ffn_pdr_controller.alarm_consecutive must be >= 1")
                if float(_c.get('integral_clamp', 0.5)) < 0.0:
                    fatal_error("ffn_pdr_controller.integral_clamp must be >= 0")
                if float(_c.get('alarm_pdr_ratio', 1.1)) <= 0.0:
                    fatal_error("ffn_pdr_controller.alarm_pdr_ratio must be > 0")
                if float(_c.get('upper_alarm_margin', 0.05)) < 0.0:
                    fatal_error("ffn_pdr_controller.upper_alarm_margin must be >= 0")
                _ref = _c.get('reference')
                if _ref is not None and not isinstance(_ref, dict):
                    fatal_error("ffn_pdr_controller.reference must be a dict")
                if isinstance(_ref, dict):
                    _mode = _ref.get('mode', 'knots')
                    _shadow_modes = ('auto_shadow_growth', 'auto_shadow_partial')
                    if _mode not in ('knots', 'auto') + _shadow_modes:
                        fatal_error("ffn_pdr_controller.reference.mode must be one of 'knots', 'auto', "
                                    f"'auto_shadow_growth', 'auto_shadow_partial', got {_mode!r}")
                    if _mode == 'knots':
                        if not _ref.get('knots'):
                            fatal_error("ffn_pdr_controller.reference.knots is required for mode 'knots' "
                                        "(the reference pdr curve).")
                    elif _mode == 'auto':
                        # 'auto': self-anchored LR-track. No knots; needs the freeze point (anchor_step)
                        # and tangent projection (it anchors the body's own pdr once f->1).
                        _as = _ref.get('anchor_step')
                        if _as is None:
                            fatal_error("ffn_pdr_controller.reference.anchor_step is required for mode 'auto' "
                                        "(an int, or 'auto' to derive it from the tangent_project_strength freeze point).")
                        if isinstance(_as, str) and _as.strip().lower() == 'auto':
                            # Footgun-killer: derive the freeze step from the f-schedule instead of hand-coupling it
                            # (and risking a silent mismatch if the schedule is retimed). = the EARLIEST step where f
                            # reaches its terminal value (so redundant trailing knots like [[..,1.0],[later,1.0]] still
                            # resolve to the true freeze, not the last knot). Resolved IN-PLACE so the controller and
                            # every check below see a plain int. Assumes a monotone-up f-schedule (the grow-then-clamp
                            # shape); the terminal-f==1.0 guard below still gates whether auto is meaningful at all.
                            _tps_sched = getattr(self, 'tangent_project_strength', None)
                            if not (isinstance(_tps_sched, list) and _tps_sched and
                                    all(isinstance(k, (list, tuple)) and len(k) == 2 for k in _tps_sched)):
                                fatal_error("ffn_pdr_controller.reference.anchor_step: 'auto' needs a "
                                            "tangent_project_strength SCHEDULE [[step,val],...] to derive the freeze "
                                            "step from (got a scalar or malformed schedule). Use an explicit int instead.")
                            _term = float(_tps_sched[-1][1])
                            _as = next((int(k[0]) for k in _tps_sched if float(k[1]) >= _term - 1e-9),
                                       int(_tps_sched[-1][0]))
                            _ref['anchor_step'] = _as   # write the resolved int back into the live config
                            print(f"[ffn-ctrl] anchor_step: auto -> resolved to {_as} "
                                  f"(tangent_project_strength reaches its terminal f={_term:g} there).")
                        if int(_as) <= _ws:
                            fatal_error(f"ffn_pdr_controller.reference.anchor_step ({_as}) must be > "
                                        f"warmup_step ({_ws}) — engage before the freeze point.")
                        if _mxs and int(_as) >= _mxs:
                            fatal_error(f"ffn_pdr_controller.reference.anchor_step ({_as}) must be < "
                                        f"max_steps ({_mxs}).")
                        if int(_ref.get('anchor_samples', 8)) < 1:
                            fatal_error("ffn_pdr_controller.reference.anchor_samples must be >= 1.")
                        if not getattr(self, 'tangent_project', False):
                            fatal_error("ffn_pdr_controller.reference.mode 'auto' requires tangent_project: true "
                                        "(it self-anchors the reference at the body-freeze point f->1).")
                        # auto must be able to FREEZE: the controller only anchors once f reaches 1.0. If the
                        # tangent_project_strength schedule terminates below 1.0, f never freezes, m is held
                        # at 1 forever, and the controller never latches (silent uncontrolled run). Require
                        # the terminal f to be ~1.0. (Malformed schedules fall through to the dedicated
                        # tangent_project_strength validation below.)
                        _tps = getattr(self, 'tangent_project_strength', 1.0)
                        _tps_term = None
                        if isinstance(_tps, (int, float)) and not isinstance(_tps, bool):
                            _tps_term = float(_tps)
                        elif isinstance(_tps, list) and _tps and isinstance(_tps[-1], (list, tuple)) \
                                and len(_tps[-1]) == 2:
                            try:
                                _tps_term = float(_tps[-1][1])
                            except (TypeError, ValueError):
                                _tps_term = None
                        if _tps_term is not None and _tps_term < 1.0 - 1e-6:
                            fatal_error(
                                f"ffn_pdr_controller.reference.mode 'auto' needs the body to FREEZE, but "
                                f"tangent_project_strength terminates at f={_tps_term} (<1.0) — f never reaches "
                                f"1.0, so the controller would hold m=1 forever and never anchor. End the "
                                f"tangent_project_strength schedule at 1.0.")
                        # auto's reference r=K_anchor*lr(t)/lr_anchor rides lr LITERALLY, so it assumes a
                        # MONOTONE-decaying LR. A 'restarts' schedule re-warms lr at each restart_step; any
                        # restart at/after the freeze would spike r, peg m at m_max, and disable the
                        # controller for that whole window. Forbid post-freeze restarts under auto.
                        if str(getattr(self, 'lr_schedule_type', 'restarts')) == 'restarts':
                            _rs = [int(s) for s in (getattr(self, 'restart_steps', ()) or ())]
                            _bad = [s for s in _rs if s >= int(_as)]
                            if _bad:
                                fatal_error(
                                    f"ffn_pdr_controller.reference.mode 'auto' assumes a monotone-decaying LR, "
                                    f"but lr_schedule_type='restarts' re-warms at restart_steps {_bad} >= "
                                    f"anchor_step ({_as}) — each post-freeze restart would spike the LR-track "
                                    f"reference, peg m at m_max, and disable the controller. Use "
                                    f"lr_schedule_type: cosine (or move all restarts before anchor_step).")
                    elif _mode in _shadow_modes:
                        # shadow-norm modes (Math Q12): m=median(R/S); body WD=radial-budget law. No knots,
                        # no anchor — the reference is constructed online. Needs tangent projection
                        # (radial_stats is produced ONLY by its block) and an FSDP/DTensor body (the
                        # single-device Muon path has no tangent block — gated at controller-wire time).
                        if not getattr(self, 'tangent_project', False):
                            fatal_error(f"ffn_pdr_controller.reference.mode '{_mode}' requires "
                                        "tangent_project: true (it reads the per-step radial telemetry the "
                                        "tangent-projection block produces).")
                        _rho = float(_c.get('rho', 0.20))
                        if not (0.0 < _rho <= 1.0):
                            fatal_error(f"ffn_pdr_controller.rho must be in (0,1], got {_rho}")
                        _lmax = float(_c.get('lambda_max', 0.02)); _lmin = float(_c.get('lambda_min', 0.002))
                        if not (0.0 <= _lmin <= _lmax):
                            fatal_error(f"ffn_pdr_controller.lambda_min ({_lmin}) must be in "
                                        f"[0, lambda_max ({_lmax})].")
                        _mmf = float(_c.get('m_min_full', 0.20))
                        if not (0.0 < _mmf <= _mx):
                            fatal_error(f"ffn_pdr_controller.m_min_full ({_mmf}) must be in (0, m_max ({_mx})].")
                        _gg = float(_c.get('glide_gain', 1.0))
                        if not (0.0 < _gg <= 5.0):
                            fatal_error(f"ffn_pdr_controller.glide_gain ({_gg}) must be in (0, 5] "
                                        "(1.0 = pure R/S; >1 steepens the cut; Math's preferred band [1.0,1.3]).")
                        _fho = _c.get('freeze_handoff', True)
                        if not isinstance(_fho, bool):
                            fatal_error(f"ffn_pdr_controller.freeze_handoff must be a boolean (true/false), "
                                        f"got {_fho!r}")
                        _aoa = _c.get('acts_on_attn', False)
                        if not isinstance(_aoa, bool):
                            fatal_error(f"ffn_pdr_controller.acts_on_attn must be a boolean (true/false), "
                                        f"got {_aoa!r}")
                        if _mode == 'auto_shadow_partial' and _fho is False:
                            print("[ffn-ctrl] note: freeze_handoff:false is a no-op for auto_shadow_partial "
                                  "(partial mode never hands off regardless). It only matters for "
                                  "auto_shadow_growth.")
                        for _stale in ('anchor_step', 'anchor_samples', 'anchor_warn_band',
                                       'anchor_fatal_band', 'anchor_abs_warn', 'knots'):
                            if _stale in _ref:
                                fatal_error(f"ffn_pdr_controller.reference.{_stale} does not apply to mode "
                                            f"'{_mode}' (the reference is constructed online from R/S, not "
                                            "anchored/fitted). Remove it.")
                        if 'warmup_step' in _c:
                            fatal_error("ffn_pdr_controller.warmup_step does not apply to shadow modes "
                                        "(engagement is f-gated, not step-gated). Remove it.")
                        if 'm_floor' in _c:
                            fatal_error("ffn_pdr_controller.m_floor is inert in shadow modes — the f-aware "
                                        "floor m_min(f)=1-f*(1-m_min_full) replaces it. Use m_min_full.")
                        # FOOTGUN BANNER: a body WD *schedule* that targets the FFN matrices is SILENTLY
                        # overridden by the controller's radial-budget λ (the controller owns FFN body WD).
                        # A flat base WD is fine (it stands for attn + the brief pre-engagement FFN window);
                        # a SCHEDULE on 'all' or a layer-range [start,end,sched] is the trap — warn loudly.
                        _wd_rules = getattr(self, 'weight_decay', None)
                        if isinstance(_wd_rules, list):
                            for _r in _wd_rules:
                                if not isinstance(_r, (list, tuple)) or len(_r) < 2:
                                    continue
                                _hits_ffn = (len(_r) == 2 and _r[0] == 'all') or \
                                            (len(_r) == 3 and not isinstance(_r[0], str))
                                if _hits_ffn and isinstance(_r[-1], list):  # value is a SCHEDULE, not a scalar
                                    print(
                                        "\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                                        "  !! WARNING: a body weight_decay SCHEDULE is set, but the shadow\n"
                                        f"  !!          controller ('{_mode}') OWNS the FFN body WD via the\n"
                                        "  !!          radial-budget law. This schedule is SILENTLY OVERRIDDEN\n"
                                        "  !!          for feed_forward.w1/w2/w3 — it affects ONLY attention /\n"
                                        "  !!          non-FFN body. To tune FFN WD use the controller knobs\n"
                                        "  !!          (rho, lambda_max, lambda_min), NOT a WD schedule.\n"
                                        f"  !!          offending rule: {list(_r)[:2]}...\n"
                                        "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                                    break
                        # terminal-f policy (Math): growth REQUIRES f->1 (grow-then-clamp); partial ALLOWS f<1.
                        _tps = getattr(self, 'tangent_project_strength', 1.0)
                        _term = None
                        if isinstance(_tps, (int, float)) and not isinstance(_tps, bool):
                            _term = float(_tps)
                        elif isinstance(_tps, list) and _tps and isinstance(_tps[-1], (list, tuple)) \
                                and len(_tps[-1]) == 2:
                            try:
                                _term = float(_tps[-1][1])
                            except (TypeError, ValueError):
                                _term = None
                        if _mode == 'auto_shadow_growth' and _term is not None and _term < 1.0 - 1e-6:
                            fatal_error(
                                "ffn_pdr_controller.reference.mode 'auto_shadow_growth' is the grow-then-clamp "
                                f"recipe and REQUIRES the body to fully freeze, but tangent_project_strength "
                                f"terminates at f={_term} (<1.0). For an intentional partial-projection run, "
                                "use reference.mode: auto_shadow_partial.")
                    _bf = _ref.get('blend_from')
                    if _bf is not None:
                        if float(_bf.get('start_tok_m', 0)) >= float(_bf.get('end_tok_m', 1)):
                            fatal_error("ffn_pdr_controller.reference.blend_from.start_tok_m must be < "
                                        "end_tok_m (else the smoothstep blend degenerates to a hard step).")
                if self.adaptive_wd is not None:
                    fatal_error("ffn_pdr_controller and adaptive_wd are mutually exclusive "
                                "(AWD moves ||W||, the denominator of pdr) — enable only one.")
                # SCS (auxiliary_heads with compute_inactive_layers:false) freezes FFN params via
                # lr_scale=0 during scaffold; the controller's per-step FFN lr_scale write (which
                # runs after the SCS block) would un-freeze them. Mutually exclude, like AWD.
                _ah = getattr(self, 'auxiliary_heads', None)
                if isinstance(_ah, dict) and _ah.get('enabled', False) \
                        and not _ah.get('compute_inactive_layers', True):
                    fatal_error("ffn_pdr_controller and SCS (auxiliary_heads compute_inactive_layers"
                                ":false) are mutually exclusive — the controller would un-freeze "
                                "SCS-frozen FFN params.")
        elif self.ffn_pdr_controller is not None:
            fatal_error(f"ffn_pdr_controller must be a dict, got {type(self.ffn_pdr_controller).__name__}")

        # --- tangent_project_strength: partial-projection f (scalar in [0,1] OR schedule) ---
        # f=1 (default) = full projection (flat ||W||); f<1 lets ||W|| grow at (1-f) of its natural
        # rate; f=0 = no projection. Scalar (constant), or [[step, val], ...] for a schedule (e.g.
        # allow growth early, clamp late). The f-sweep dial. Only meaningful when tangent_project on.
        if not hasattr(self, 'tangent_project_strength'):
            self.tangent_project_strength = 1.0
        else:
            _ts = self.tangent_project_strength
            if isinstance(_ts, bool):
                fatal_error("tangent_project_strength must be a number or schedule, not a bool")
            elif isinstance(_ts, (int, float)):
                if not (0.0 <= float(_ts) <= 1.0):
                    fatal_error(f"tangent_project_strength scalar must be in [0,1], got {_ts}")
            elif isinstance(_ts, list):
                if not _ts:
                    fatal_error("tangent_project_strength schedule is empty")
                _prev = None
                for _kn in _ts:
                    if not (isinstance(_kn, list) and len(_kn) == 2):
                        fatal_error(f"tangent_project_strength schedule entries must be [step, val], got {_kn!r}")
                    if not (0.0 <= float(_kn[1]) <= 1.0):
                        fatal_error(f"tangent_project_strength schedule values must be in [0,1], got {_kn[1]}")
                    if _prev is not None and float(_kn[0]) <= _prev:
                        fatal_error(f"tangent_project_strength schedule steps must be strictly ascending, got {_kn[0]} after {_prev}")
                    _prev = float(_kn[0])
            else:
                fatal_error(f"tangent_project_strength must be a number or [[step,val],...], got {type(_ts).__name__}")

        # --- auxiliary_heads: intermediate-depth prediction heads ---
        # Validated lightly here; full parse happens in parse_aux_heads_config
        # at trainer setup time (needs n_layers from model_cfg to range-check).
        if not hasattr(self, 'auxiliary_heads'):
            self.auxiliary_heads = None
        elif isinstance(self.auxiliary_heads, dict):
            if not self.auxiliary_heads.get('enabled', False):
                self.auxiliary_heads = None
            else:
                ah = self.auxiliary_heads
                # Scaffolded Cascading Supervision (SCS) optional fields.
                # compute_inactive_layers: when False, the trainer truncates
                # forward/backward at the deepest tap currently at λ >= 1.0 and
                # skips the main LM head. Inactive layers (and output.weight)
                # are frozen via lr_scale_overrides so WD doesn't decay them.
                cil = ah.get('compute_inactive_layers', True)
                if not isinstance(cil, bool):
                    fatal_error(
                        f"auxiliary_heads.compute_inactive_layers must be bool, got {type(cil).__name__}"
                    )
                if not cil:
                    # Validate the warmup knobs that come with SCS.
                    wms = ah.get('new_layer_warmup_steps', 0)
                    if not isinstance(wms, int) or wms < 0:
                        fatal_error(
                            f"auxiliary_heads.new_layer_warmup_steps must be a non-negative int, got {wms!r}"
                        )
                    nlm = ah.get('new_layer_lr_multiplier', 1.0)
                    if not isinstance(nlm, (int, float)) or not (0.0 <= float(nlm) <= 1.0):
                        fatal_error(
                            f"auxiliary_heads.new_layer_lr_multiplier must be a float in [0, 1], got {nlm!r}"
                        )

        # --- z_loss: optional confidence penalty on logsumexp (log-partition) ---
        # Mirrors adaptive_wd/auxiliary_heads normalization: default None, and
        # collapse to None unless explicitly enabled so downstream code only
        # checks `settings.z_loss is not None`. The block is read by
        # get_zloss_alpha(step, settings) — a pure function of the global step,
        # so nothing here needs to persist in the checkpoint.
        if not hasattr(self, 'z_loss'):
            self.z_loss = None
        elif isinstance(self.z_loss, dict):
            if not self.z_loss.get('enabled', False):
                self.z_loss = None
            else:
                zl = self.z_loss
                a = zl.get('alpha', None)
                if not isinstance(a, (int, float)) or float(a) < 0.0:
                    fatal_error(
                        f"z_loss.alpha must be a non-negative number, got {a!r}"
                    )
                # backend: precision/memory tradeoff for the option-D z-loss
                # gradient (CCE 25.4.3 has no return_lse; logZ is reconstructed
                # as CE_none + target_logit, a bf16 catastrophic cancellation).
                #   'fp32_accum' (default): CCE accum_e/c_fp32 in the backward ->
                #       grad cosine ~0.999 vs fp32 truth, ~+0.45 GB at the head.
                #   'bf16': lightest memory, grad cosine ~0.990 (fine for a small
                #       annealed regularizer).
                bk = zl.get('backend', 'fp32_accum')
                if bk not in ('bf16', 'fp32_accum'):
                    fatal_error(
                        f"z_loss.backend must be 'bf16' or 'fp32_accum', got {bk!r}"
                    )
                # dn4 Lever 2: penalized quantity. 'raw' (default) = mean(logZ**2);
                # 'centered' = mean(relu(logZ_c - tau)**2), the gauge-invariant deadband
                # ceiling (DN4_HEAD_HYGIENE_SPEC, Math-approved). tau required when centered.
                _ztgt = zl.get('target', 'raw')
                if _ztgt not in ('raw', 'centered'):
                    fatal_error(f"z_loss.target must be 'raw' or 'centered', got {_ztgt!r}")
                if _ztgt == 'centered':
                    _ztau = zl.get('tau', None)
                    if not isinstance(_ztau, (int, float)) or isinstance(_ztau, bool):
                        fatal_error("z_loss.target: centered requires a numeric z_loss.tau "
                                    "(the deadband ceiling on logZ_c; e.g. 120-128).")
                    if getattr(self, 'tie_word_embeddings', True):
                        fatal_error(
                            "z_loss.target: centered requires an UNTIED head: it shapes logZ_c via "
                            "mu(output.weight), and on a tied head output.weight IS the input "
                            "embedding, so the centered ceiling would also regularize the embeddings. "
                            "Set tie_word_embeddings: false or use target: raw.")
                warmup = zl.get('warmup')
                if warmup is not None:
                    if not isinstance(warmup, dict):
                        fatal_error(
                            f"z_loss.warmup must be a dict, got {type(warmup).__name__}"
                        )
                    if warmup.get('enabled', False):
                        ss = warmup.get('start_step', 0)
                        if not isinstance(ss, int) or ss < 0:
                            fatal_error(
                                f"z_loss.warmup.start_step must be a non-negative int, got {ss!r}"
                            )
                        ds = warmup.get('duration_steps', None)
                        if not isinstance(ds, int) or ds <= 0:
                            fatal_error(
                                f"z_loss.warmup.duration_steps must be a positive int, got {ds!r}"
                            )
                        shp = warmup.get('shape', 'cosine')
                        if shp not in ('cosine', 'linear'):
                            fatal_error(
                                f"z_loss.warmup.shape must be 'cosine' or 'linear', got {shp!r}"
                            )
                # warmdown: ramp alpha -> 0 (staged transition Stage 1). Same
                # validation shape as warmup. alpha hits exactly 0 at
                # start_step + duration_steps (half-open), before row-centering's
                # s goes nonzero — see get_zloss_alpha + the overlap assert below.
                warmdown = zl.get('warmdown')
                if warmdown is not None:
                    if not isinstance(warmdown, dict):
                        fatal_error(
                            f"z_loss.warmdown must be a dict, got {type(warmdown).__name__}"
                        )
                    if warmdown.get('enabled', False):
                        ss = warmdown.get('start_step', 0)
                        if not isinstance(ss, int) or ss < 0:
                            fatal_error(
                                f"z_loss.warmdown.start_step must be a non-negative int, got {ss!r}"
                            )
                        ds = warmdown.get('duration_steps', None)
                        if not isinstance(ds, int) or ds <= 0:
                            fatal_error(
                                f"z_loss.warmdown.duration_steps must be a positive int, got {ds!r}"
                            )
                        shp = warmdown.get('shape', 'cosine')
                        if shp not in ('cosine', 'linear'):
                            fatal_error(
                                f"z_loss.warmdown.shape must be 'cosine' or 'linear', got {shp!r}"
                            )
        elif self.z_loss is not None:
            fatal_error(f"z_loss must be a dict, got {type(self.z_loss).__name__}")

        # --- row_center_head: gauge subtraction on the LM readout head ---
        # Accepts BOTH forms:
        #   flat bool   row_center_head: true      -> steady-state full projection
        #   nested dict row_center_head: {enabled: true, warmup: {start_step,
        #               duration_steps, shape}}    -> staged target-gauge warmup
        # When on, the trainer subtracts the vocab-row mean mu from the main head
        # every step (and from the Adam first moment), removing the CE-invisible
        # common-mode gauge. Function-preserving on the MODEL OUTPUT (not on the
        # optimizer trajectory — hence the staged warmup for mid-run resumes).
        # See common_fsdp2/row_center.py. Assumes UNTIED + bias-free head.
        if not hasattr(self, 'row_center_head'):
            self.row_center_head = False
        rc = self.row_center_head
        if isinstance(rc, dict):
            if not isinstance(rc.get('enabled', False), bool):
                fatal_error("row_center_head.enabled must be a bool")
            rcw = rc.get('warmup')
            if rcw is not None:
                if not isinstance(rcw, dict):
                    fatal_error(f"row_center_head.warmup must be a dict, got {type(rcw).__name__}")
                if rcw.get('enabled', False):
                    ss = rcw.get('start_step', 0)
                    if not isinstance(ss, int) or ss < 0:
                        fatal_error(f"row_center_head.warmup.start_step must be a non-negative int, got {ss!r}")
                    ds = rcw.get('duration_steps', None)
                    if not isinstance(ds, int) or ds <= 0:
                        fatal_error(f"row_center_head.warmup.duration_steps must be a positive int, got {ds!r}")
                    shp = rcw.get('shape', 'cosine')
                    if shp not in ('cosine', 'linear'):
                        fatal_error(f"row_center_head.warmup.shape must be 'cosine' or 'linear', got {shp!r}")
            rc_enabled = bool(rc.get('enabled', False))
        elif isinstance(rc, bool):
            rc_enabled = rc
        else:
            fatal_error(f"row_center_head must be a bool or dict, got {type(rc).__name__}")
        # Escape hatch reserved for a deliberate ablation only.
        if not hasattr(self, 'allow_row_center_with_z_loss'):
            self.allow_row_center_with_z_loss = False
        # Guardrail 5: advisory transition health guard (WARNING logs only, no
        # auto-action). Off by default — Josef flips it on for a staged transition.
        if not hasattr(self, 'transition_health_guard'):
            self.transition_health_guard = False
        elif not isinstance(self.transition_health_guard, bool):
            fatal_error(f"transition_health_guard must be a bool, got {type(self.transition_health_guard).__name__}")
        if rc_enabled:
            if self.tie_word_embeddings:
                fatal_error(
                    "row_center_head requires an UNTIED output head: subtracting "
                    "the row-mean from a tied head also shifts the input "
                    "embeddings, which is NOT function-preserving. Set "
                    "tie_word_embeddings: false or disable row_center_head."
                )
            # STEP-ACTIVE OVERLAP ASSERT (Guardrail 2): the incompatibility is
            # ACTIVE overlap, not static co-enablement. row_center s and z-loss
            # alpha can both be CONFIGURED on (staged transition) as long as their
            # schedules are temporally disjoint. Fail only if some global step has
            # BOTH s(step) > 0 AND alpha_eff(step) > 0 (then raw z-loss would be
            # acting as centered z-loss). Scan the union of the schedules' active
            # ranges. Override allows a deliberate ablation.
            if self.z_loss is not None and not self.allow_row_center_with_z_loss:
                overlap_step = _find_rowcenter_zloss_overlap(self)
                if overlap_step is not None:
                    a = get_zloss_alpha(overlap_step, self)
                    s = get_row_center_s(overlap_step, self)
                    fatal_error(
                        f"row_center_head and z_loss are ACTIVE at the same step "
                        f"{overlap_step} (alpha_eff={a:.3e}, s={s:.3f}). Centering a "
                        f"head while raw-logZ z-loss is nonzero makes it CENTERED "
                        f"z-loss (a real regularizer, not the inert gauge). Stage "
                        f"them disjoint (z-loss warmdown ends at/before row-center "
                        f"warmup start), or set allow_row_center_with_z_loss: true."
                    )

        # --- head_gauge_projection (dn4 head-hygiene): in-optimizer projection of the
        # CE-invisible common-mode gauge out of the LM head's APPLIED Adam update each
        # step. See docs/DN4_HEAD_HYGIENE_SPEC.md. ---
        if _head_gauge_cfg(self)['enabled']:
            if self.tie_word_embeddings:
                fatal_error(
                    "head_gauge_projection requires an UNTIED output head: projecting the "
                    "head update's row-mean on a tied head also moves the input embeddings "
                    "(not function-preserving). Set tie_word_embeddings: false.")
            if getattr(self, 'muon_adam_state_dtype', 'fp32') != 'fp32':
                fatal_error(
                    "head_gauge_projection needs muon_adam_state_dtype: fp32 — any other value "
                    "selects the FUSED 16-bit Adam path, which applies update+WD internally so "
                    "the applied update U is never exposed to the projection (silent no-op).")
            if rc_enabled:
                fatal_error(
                    "head_gauge_projection and row_center_head are MUTUALLY EXCLUSIVE (competing "
                    "head-gauge implementations; the legacy row_center_head also projects exp_avg "
                    "and does post-step weight surgery). Enable exactly one.")

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

        # --- doc_attn_mask (branch doc-mask): confine attention to (causal AND
        # same-document) within packed windows via a FlexAttention BlockMask;
        # optionally restart RoPE positions at each BOS. Both OFF by default —
        # flags-off behavior is byte-identical to main. ---
        _dm = getattr(self, 'doc_attn_mask', None) or {}
        if _dm is True:
            _dm = {'enabled': True}
        if not isinstance(_dm, dict):
            fatal_error("doc_attn_mask must be a dict {enabled, reset_positions, bos_token_id} or true")
        _dm_known = {'enabled', 'reset_positions', 'bos_token_id', 'allow_reset_without_mask'}
        _dm_typos = set(_dm) - _dm_known
        if _dm_typos:
            # silent-ignore would default the feature OFF — the worst direction for a typo
            fatal_error(f"doc_attn_mask: unknown key(s) {sorted(_dm_typos)} — known: {sorted(_dm_known)}")
        self.doc_attn_mask_enabled = bool(_dm.get('enabled', False))
        self.doc_pos_reset = bool(_dm.get('reset_positions', False))
        # NOTE: the stream's actual document separator is the tokenizer-native BOS id 1
        # (verified against the llama .npy shards 2026-07-02) — NOT the <|bos|>=32000
        # special token, which never occurs in pretokenized data. A wrong id makes both
        # features a silent no-op; the trainer also stream-checks this in-loop (fatal by
        # ~step 20 if the configured id never appears).
        self.doc_bos_token_id = int(_dm.get('bos_token_id', 1))
        if self.doc_pos_reset and not self.doc_attn_mask_enabled \
                and not bool(_dm.get('allow_reset_without_mask', False)):
            fatal_error(
                "doc_attn_mask: reset_positions without enabled trains CROSS-document attention "
                "over aliased, non-monotonic RoPE positions — a geometry matching neither the "
                "baseline nor the masked treatment. Set enabled: true, or opt in explicitly with "
                "allow_reset_without_mask: true if this ablation is truly intended.")
        if self.doc_attn_mask_enabled or self.doc_pos_reset:
            if self.gdn_enabled:
                fatal_error(
                    "doc_attn_mask/reset_positions are not supported with gdn_enabled — GDN's "
                    "recurrent state crosses document boundaries (FLA varlen state reset not "
                    "implemented). Disable one.")
            if float(getattr(self, 'dropout', 0.0)) > 0.0:
                fatal_error(
                    "doc_attn_mask requires dropout: 0.0 — the FlexAttention path has no "
                    "attention-dropout argument, so dropout>0 would silently change semantics "
                    "between masked and unmasked layers.")
            if getattr(self, 'attn_res_enabled', False):
                fatal_error(
                    "doc_attn_mask with attn_res_enabled is untested (block-residual retrieval "
                    "mixes representations across the window between attention calls). Validate "
                    "separately before combining.")
            if self.doc_attn_mask_enabled and not bool(getattr(self, 'compile_model', False)):
                fatal_error(
                    "doc_attn_mask requires compile_model: true — uncompiled flex_attention falls "
                    "back to a score-materializing math path (multi-GB transients per layer, OOM "
                    "at production shapes). Compile, or disable the mask while bisecting.")
            if bool(getattr(self, 'resume_training', False)):
                logger.print_and_log(
                    "  ] [doc-mask] WARNING: resuming with doc_attn_mask/reset_positions set — "
                    "these flags MUST match the run being resumed (no automatic mismatch guard "
                    "yet); flipping them mid-run silently changes attention semantics.")

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
        # PROBE (WD_NO_TF32=1): force TRUE IEEE fp32 matmul (TF32 has only ~10 mantissa bits,
        # so a "fp32" run with TF32 on is NOT full precision). Needed to truly rule precision
        # out of the body-ramp lean (Math Agent #5). Unset => normal TF32 behaviour.
        if os.environ.get('WD_NO_TF32') == '1':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            logger.print_and_log("  ] WD_NO_TF32: TF32 DISABLED — true IEEE fp32 matmul")
    
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
        # Document attention masking (branch doc-mask)
        doc_attn_mask=settings.doc_attn_mask_enabled,
        doc_pos_reset=settings.doc_pos_reset,
        bos_token_id=settings.doc_bos_token_id,
    )
    if settings.doc_attn_mask_enabled or settings.doc_pos_reset:
        logger.print_and_log(
            f"  ] [doc-mask] attention mask: {'ON (causal AND same-doc, FlexAttention)' if settings.doc_attn_mask_enabled else 'off'}"
            f" | RoPE reset at BOS: {'ON' if settings.doc_pos_reset else 'off'}"
            f" | BOS id: {settings.doc_bos_token_id}")

    # ----------------------- Save Settings Config File -----------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(settings.nas_path, f"config_{timestamp}.yaml")
    if ddp_rank == 0:
        settings.to_yaml(config_path)
        logger.print_and_log(f"Configuration saved to {config_path}")

    check_params(model_cfg, settings=settings, world_size=ddp_world_size)

    # ----------------------- Create and shard model -----------------------
    model = create_and_shard_model(model_cfg, mesh, ep_mesh, edp_mesh, device, settings, logger)
    # [freqs-check] RoPE buffer integrity after the meta->fully_shard->to_empty->init_weights
    # flow: to_empty() clobbers value-carrying buffers and init_weights never refilled them
    # (found during the doc-mask review; see doc_pos_reset). Verify against a fresh table.
    _rc, _rs = precompute_freqs_cis(
        model_cfg.dim // model_cfg.n_heads, model_cfg.max_seq_len,
        getattr(model_cfg, 'rope_theta', 500000.0))
    _bc = model.freqs_cos.detach().cpu().float()
    _bs = model.freqs_sin.detach().cpu().float()
    # tolerance wide enough for a bf16-cast of CORRECT values; corruption (zeros /
    # stale block contents) is orders of magnitude outside it
    _cos_ok = torch.allclose(_bc, _rc, atol=1e-2)
    _sin_ok = torch.allclose(_bs, _rs, atol=1e-2)
    logger.print_and_log(
        f"  ] [freqs-check] cos ok: {_cos_ok} | sin ok: {_sin_ok} | dtype {model.freqs_cos.dtype} "
        f"| cos absmax {_bc.abs().max():.4f} (ref {_rc.abs().max():.4f}) "
        f"| sin absmax {_bs.abs().max():.4f} | sin==ref_COS: {torch.allclose(_bs, _rc, atol=1e-2)} "
        f"| row1[:4] {_bc[1, :4].tolist()} ref {_rc[1, :4].tolist()}")

    # Z-loss is a trainer/loss-path concern, not a ModelArgs/architecture knob,
    # so set the backend flag on the raw module post-build (before per-submodule
    # compile at the bottom of setup; that leaves the root unwrapped, so the
    # flag persists). _zloss_fp32_accum: None=off (loss path byte-for-byte
    # identical to baseline) | False=bf16 backend | True=fp32_accum backend.
    _raw_for_zloss = model._orig_mod if hasattr(model, "_orig_mod") else model
    if settings.z_loss is None:
        _raw_for_zloss._zloss_fp32_accum = None
    else:
        _raw_for_zloss._zloss_fp32_accum = (
            settings.z_loss.get('backend', 'fp32_accum') == 'fp32_accum'
        )
        # dn4 Lever 2: penalized quantity (raw mean(logZ**2) | centered deadband).
        _raw_for_zloss._zloss_target = settings.z_loss.get('target', 'raw')
        _raw_for_zloss._zloss_tau = float(settings.z_loss.get('tau', 0.0))

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

    # ----------------------- Log SCS (Scaffolded Cascading Supervision) -----
    _ah_print = getattr(settings, 'auxiliary_heads', None) or {}
    if _aux_head_layers and not _ah_print.get('compute_inactive_layers', True) and ddp_rank == 0:
        _, _scs_sched = parse_aux_heads_config(getattr(settings, 'auxiliary_heads', None))
        _scs_activation = compute_scs_activation_events(_scs_sched, model_cfg.n_layers)
        _scs_wms = int(_ah_print.get('new_layer_warmup_steps', 0))
        _scs_nlm = float(_ah_print.get('new_layer_lr_multiplier', 1.0))
        # Determine cascade-complete step: the activation step of the layer
        # one past the deepest aux tap (i.e. the first non-aux-controlled layer).
        _deepest_tap_layer = max(_aux_head_layers)
        _cascade_complete = (
            _scs_activation[_deepest_tap_layer + 1]
            if _deepest_tap_layer + 1 < model_cfg.n_layers
            else None
        )
        logger.print_and_log(f"Scaffolded Cascading Supervision (SCS):")
        logger.print_and_log(f"  ] compute_inactive_layers = false  (scaffold mode while any tap is at λ >= 1.0)")
        logger.print_and_log(
            f"  ] warmup              = {_scs_wms} steps ramping from {_scs_nlm} -> 1.0 per compartment"
        )
        if _cascade_complete is not None:
            logger.print_and_log(
                f"  ] cascade complete    = step {_cascade_complete:,} (main LM head + tail layers come online)"
            )
            logger.print_and_log(
                f"  ] output head warmup = step {_cascade_complete:,} -> {_cascade_complete + _scs_wms:,} "
                f"(ramps {_scs_nlm} -> 1.0, mirrors tail-compartment warmup)"
            )
        # Group consecutive layers with the same activation step into compartments
        # for a compact display.
        logger.print_and_log(f"  ] Compartments       (layer range -> activation step):")
        _prev_step = None
        _range_start = 0
        for _li in range(model_cfg.n_layers):
            _st = _scs_activation[_li]
            if _st != _prev_step:
                if _prev_step is not None:
                    _end_label = _li - 1
                    logger.print_and_log(
                        f"  ]   L{_range_start:>3d}..L{_end_label:>3d}: activates @ step {_prev_step:,}"
                    )
                _range_start = _li
                _prev_step = _st
        # Trailing range
        logger.print_and_log(
            f"  ]   L{_range_start:>3d}..L{model_cfg.n_layers - 1:>3d}: activates @ step {_prev_step:,}"
        )
        logger.print_and_log(f"  ] During scaffold: main head + final norm frozen via lr_scale=0; inactive tail likewise.")

    # ----------------------- Log z-loss (log-partition regularization) --------
    _zl = getattr(settings, 'z_loss', None)
    if _zl is not None and ddp_rank == 0:
        _zl_alpha = float(_zl.get('alpha', 0.0))
        _zl_backend = _zl.get('backend', 'fp32_accum')
        _zl_grad = '~0.999 cos / ~0.05 norm-rel' if _zl_backend == 'fp32_accum' \
            else '~0.990 cos / ~0.12 norm-rel'
        logger.print_and_log(f"Z-Loss (log-partition regularization):")
        if _zl.get('target', 'raw') == 'centered':
            _zl_tau = float(_zl.get('tau', 0.0))
            logger.print_and_log(
                f"  ] objective      = CE + alpha * mean(relu(logZ_c - tau)^2)   [CENTERED deadband, dn4 Lever 2]"
            )
            logger.print_and_log(
                f"  ] centered/tau   = logZ_c = logZ - h.mu (gauge-invariant; zero common-mode gradient); deadband ceiling tau = {_zl_tau:.2f}"
            )
        else:
            logger.print_and_log(
                f"  ] objective      = CE + alpha * mean(logZ^2) on the live LM readout head   [RAW]"
            )
        logger.print_and_log(f"  ] alpha (target) = {_zl_alpha:.3e}")
        logger.print_and_log(
            f"  ] backend        = {_zl_backend}  (option-D reconstruction; grad {_zl_grad} vs fp32 truth)"
        )
        _zw = _zl.get('warmup') or {}
        if _zw.get('enabled', False):
            _zw_start = int(_zw.get('start_step', 0))
            _zw_dur = int(_zw.get('duration_steps', 1))
            _zw_shape = _zw.get('shape', 'cosine')
            logger.print_and_log(
                f"  ] alpha warmup   = {_zw_shape} ramp 0 -> {_zl_alpha:.3e} "
                f"over steps {_zw_start:,}..{_zw_start + _zw_dur:,} (absolute global step, resume-safe)"
            )
        else:
            logger.print_and_log(f"  ] alpha warmup   = disabled (full alpha applied immediately)")
        logger.print_and_log(
            f"  ] headline ls/ppl stay PURE CE; zloss/logZ/logZ_rms/logZ_p95/z_a + head metrics (hd_*) logged separately"
        )

    # ----------------------- Log head gauge projection (dn4 Lever 1) ----------
    _hg = _head_gauge_cfg(settings)
    if _hg['enabled'] and ddp_rank == 0:
        logger.print_and_log(f"Head Gauge Projection (dn4 head-hygiene, applied-update):")
        logger.print_and_log(
            f"  ] operation      = U <- U - 1 mu(U)^T on the LM head's Adam update each step (mu = global vocab-row mean)"
        )
        logger.print_and_log(
            f"  ] removes the CE-invisible common-mode gauge from the APPLIED update (NOT exp_avg, NOT post-step weight surgery)"
        )
        logger.print_and_log(
            f"  ] fp32 row-mean + stochastic-rounding bf16 write-back; gauge-invariant -> CE/softmax UNCHANGED, gauge cannot accumulate"
        )
        logger.print_and_log(
            f"  ] init_row_center = {_hg['init_row_center']} (one-time weight-only gauge clean at step 1)"
        )
        logger.print_and_log(
            f"  ] telemetry      = [head-gauge] ||Ubar|| per val step (post ~0 confirms the write-back); + centered geom (logZ_c, ||W_c||)"
        )

    # ----------------------- Log tangent projection (Muon body radial-growth control) ----------
    if getattr(settings, 'tangent_project', False) and ddp_rank == 0:
        _tps = getattr(settings, 'tangent_project_strength', 1.0)
        logger.print_and_log(f"Tangent Projection (Muon body radial-growth control):")
        logger.print_and_log(
            f"  ] operation      = strip f of the radial (||W) component of each Muon body update "
            f"-> ||W|| grows at (1-f) of its natural rate  (f=1 flat/frozen, f=0 off)"
        )
        if isinstance(_tps, list):
            logger.print_and_log(f"  ] strength f     = SCHEDULE {_tps}  (linear interp by step)")
        else:
            logger.print_and_log(f"  ] strength f     = {float(_tps):.3f} (constant)")
        logger.print_and_log(
            f"  ] preserve_norm  = {bool(getattr(settings, 'tangent_project_preserve_norm', False))}"
            f"  ; live f printed on the [tangent] line every val step"
        )

    # ----------------------- Log row-center head (gauge subtraction) ----------
    # Gate on the NORMALISED enabled flag, not the raw config: row_center_head is now a
    # dict ({enabled: false}) and a non-empty dict is TRUTHY, so `getattr(...)` alone
    # printed this banner even when disabled — a lying boot log (esp. dangerous since
    # row-centering has killed runs). Use the same _row_center_cfg()['enabled'] the
    # feature itself reads.
    if _row_center_cfg(settings)['enabled'] and ddp_rank == 0:
        logger.print_and_log(f"Row-Center Head (gauge subtraction):")
        logger.print_and_log(
            f"  ] operation      = W <- W - 1 mu^T (mu = global vocab-row mean), full projection from step 0"
        )
        logger.print_and_log(
            f"  ] function-preserving: shifts every token's logits by the scalar h.mu -> CE/softmax/sampling UNCHANGED"
        )
        logger.print_and_log(
            f"  ] also projects Adam 1st moment (exp_avg) by ITS OWN row-mean each step so the gauge can't regrow"
        )
        logger.print_and_log(
            f"  ] NOT centered z-loss (z_loss must be off); main LM head only; assumes untied + bias-free head"
        )
        if getattr(settings, 'allow_row_center_with_z_loss', False):
            logger.print_and_log(
                f"  ] [!] allow_row_center_with_z_loss=TRUE — z-loss runs as CENTERED z-loss (deliberate ablation)"
            )
        logger.print_and_log(
            f"  ] telemetry      = rc_muW_pre/post, rc_mbar, rc_proj_fro, rc_proj_ratio (diagnostics.jsonl at val cadence)"
        )

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
        tangent_project=getattr(settings, 'tangent_project', False),
        tangent_project_preserve_norm=getattr(settings, 'tangent_project_preserve_norm', False),
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

    # dn4 head-hygiene: register the LM head for in-optimizer applied-update gauge
    # projection. Built from the POST-FSDP-wrap param so id() matches the object the
    # optimizer iterates. Assert EXACTLY ONE match in the non-fused Adam path: 0 ⇒ a
    # silent no-op (id captured pre-wrap / wrong param), >1 ⇒ a wiring bug.
    if _head_gauge_cfg(settings)['enabled']:
        _raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        _head_w = _head_param(_raw)
        if _head_w is None:
            raise RuntimeError("head_gauge_projection enabled but the model has no output head.")
        if getattr(getattr(_raw, 'output', None), 'bias', None) is not None:
            raise RuntimeError("head_gauge_projection assumes a BIAS-FREE head (the bias mean is "
                               "its own gauge and would need separate handling).")
        if not hasattr(optimizer, 'head_gauge_ids'):
            raise RuntimeError(f"head_gauge_projection requires a Muon-family optimizer; got "
                               f"{type(optimizer).__name__} (no head_gauge_ids hook).")
        optimizer.head_gauge_ids = {id(_head_w)}
        # _head_gauge_verify stays OFF (default) on the hot path: the per-step projection
        # logs ||Ubar|| (the gauge it removed); the post-write ~0 check (an extra all-reduce)
        # was confirmed by the unit test + smoke and isn't recomputed every step in production.
        _matched = [(gi, p) for gi, g in enumerate(optimizer.param_groups)
                    for p in g['params'] if id(p) in optimizer.head_gauge_ids]
        if len(_matched) != 1:
            raise RuntimeError(f"head_gauge_projection: expected EXACTLY 1 optimizer param to match "
                               f"the head, got {len(_matched)} (0 ⇒ silent no-op, id() likely captured "
                               f"before FSDP wrap; >1 ⇒ wiring bug).")
        _gi, _hp = _matched[0]
        if _hp.dim() != 2:
            raise RuntimeError(f"head_gauge_projection: matched head param is {_hp.dim()}-D, expected 2-D [V,D].")
        if optimizer.param_groups[_gi].get('use_muon', False):
            raise RuntimeError("head_gauge_projection: matched head is in a Muon group, expected the Adam path.")
        if getattr(optimizer, '_use_16bit_adam', False):
            raise RuntimeError("head_gauge_projection: optimizer is in 16-bit Adam mode; U is not exposed.")
        if ddp_rank == 0:
            logger.print_and_log(f"[head-gauge] registered LM head {tuple(_hp.shape)} for applied-update "
                                 f"gauge projection (Adam path group {_gi}; init_row_center="
                                 f"{_head_gauge_cfg(settings)['init_row_center']}).")

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

    # ----------------------- FFN pdr controller (kv3) -----------------------
    # Restores the relative-LR anneal tangent projection removed, by controlling FFN body
    # angular step size (pdr) toward a reference. Inert unless ffn_pdr_controller.enabled.
    # docs/KV3_CONTROLLER_DESIGN.md. Reads/writes the SAME lr_scale_overrides the optimizer uses.
    body_lr_ctrl = None
    if getattr(settings, 'ffn_pdr_controller', None) is not None:
        from body_lr_controller import BodyLRController
        body_lr_ctrl = BodyLRController(settings.ffn_pdr_controller)
        # LR-schedule fingerprint: the 'auto' LR-track reference rides this exact schedule (both r and
        # the plant ∝ lr, so the lr motion cancels in the control error). Captured here, checkpointed,
        # and re-checked on resume — a changed schedule after anchoring would break the cancellation and
        # the controller would fight the new curve. (Knots mode: stored but inert.)
        body_lr_ctrl.lr_fingerprint = _lr_schedule_fingerprint(settings)
        if ddp_rank == 0 and body_lr_ctrl.enabled:
            if getattr(body_lr_ctrl, 'force_m', None) is not None:
                logger.print_and_log(
                    f"  ## DEBUG force_m={body_lr_ctrl.force_m} -- FFN m is PINNED, controller feedback "
                    f"BYPASSED. OPEN-LOOP ACTUATOR PROBE ONLY; do NOT use for a real run. ##")
            if body_lr_ctrl.ref_mode == 'auto':
                logger.print_and_log(
                    f"FFN pdr controller: ENABLED — AUTO self-anchored LR-track "
                    f"(warmup_step={body_lr_ctrl.warmup_step}, anchor_step={body_lr_ctrl.anchor_step}, "
                    f"anchor_samples={body_lr_ctrl.anchor_samples}, m_floor={body_lr_ctrl.m_floor}, "
                    f"m_max={body_lr_ctrl.m_max}, FF-only={not body_lr_ctrl.pid.active}); reference is "
                    f"discovered at the freeze point — no hand-fit knots.")
            elif body_lr_ctrl.ref_mode in ('auto_shadow_growth', 'auto_shadow_partial'):
                if body_lr_ctrl.ref_mode == 'auto_shadow_partial':
                    _ho = "NO-HANDOFF (partial: R/S law throughout, body never fully freezes)"
                elif body_lr_ctrl.freeze_handoff:
                    _ho = "f=1 -> LR-track handoff"
                else:
                    _ho = "NO-HANDOFF (R/S continues past f=1 -> continuous anneal, no kink)"
                _scope = ("FFN+attn (acts_on_attn: same m to attn, FFN-only WD)"
                          if body_lr_ctrl.acts_on_attn else "FF-only")
                logger.print_and_log(
                    f"FFN pdr controller: ENABLED — SHADOW-NORM [{body_lr_ctrl.ref_mode}]: "
                    f"m = median(R/S)^g (glide_gain g={body_lr_ctrl.glide_gain}) constructed online "
                    f"(no anchor/knots/warmup); body WD = radial-budget "
                    f"lambda=clamp({body_lr_ctrl.lam_min},{body_lr_ctrl.lam_max}, "
                    f"rho={body_lr_ctrl.rho}*(1-f)*gamma_EMA); f-aware floor m_min_full={body_lr_ctrl.m_min_full}, "
                    f"m_max={body_lr_ctrl.m_max}, {_ho}, {_scope}.")
            else:
                logger.print_and_log(
                    f"FFN pdr controller: ENABLED — knots reference "
                    f"(warmup_step={body_lr_ctrl.warmup_step}, "
                    f"m_floor={body_lr_ctrl.m_floor}, FF-only={not body_lr_ctrl.pid.active})")
            # observe() fires at val_step, so val_step IS the controller's cadence (there is no separate
            # `cadence` field). The EMA alphas + rate limits are per-observe-sample, tuned for val_step ~100;
            # warn if val_step is far off (a large val_step stretches the EMA time-constant -> sluggish control).
            _vs = int(getattr(settings, 'val_step', 100))
            if not (25 <= _vs <= 250):
                logger.print_and_log(
                    f"  WARNING: val_step ({_vs}) is far from the ~100 the FFN-ctrl EMA alphas/rate-limits "
                    f"were tuned for — observe() runs at val_step, so this re-times the controller. Retune "
                    f"k_ema_alpha/pdr_ema_alpha/rate_* or move val_step toward ~100.")
            if 'cadence' in (settings.ffn_pdr_controller or {}):
                logger.print_and_log(
                    "  note: ffn_pdr_controller.cadence is deprecated and IGNORED — observe() runs at val_step.")

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
        start_step, total_tokens_processed = resume_training(model, optimizer, train_loader, ddp_rank, settings, grad_accum_schedule, awd=awd, body_lr_ctrl=body_lr_ctrl)

    # ----------------------- Optionally compile the model -----------------------
    if settings.compile_model:
        # Suppress noisy-but-harmless dynamo warnings from FLA internals
        # (lru_cache tracing + cuda_utils.get_device_properties)
        warnings.filterwarnings("ignore", message=".*lru_cache.*", module="torch._dynamo")
        warnings.filterwarnings("ignore", message=".*cuda_utils.get_device_properties.*", module="torch._dynamo")
        _apply_per_submodule_compile(model, settings.compile_mode, logger)

    dist.barrier()  # Ensure all processes are synchronized before starting training

    # ----------------------- Create layer diagnostics tracker -----------------------
    diagnostics = LayerDiagnostics(model, ddp_rank, ddp_world_size, ddp,
                                   track_subgroups=bool(getattr(settings, 'track_subgroup_pdr', False)))
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
        body_lr_ctrl=body_lr_ctrl,
        moe_balance_hook=moe_balance_hook,
    )

    if ddp:
        torch.cuda.synchronize()
        dist.barrier()
        destroy_process_group()

    logger.print_and_log("[R{ddp_rank}] Training complete", False)