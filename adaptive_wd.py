# adaptive_wd.py
"""
Adaptive Weight Decay (AWD) for ultra-deep architectures.

Monitors per-layer gradient concentration and dynamically adjusts weight decay
for layers that accumulate disproportionate gradient energy. AWD *multiplies*
the base WD (from weight_decay rules or scalar) — it never replaces it.

Metric modes:
  - "g_norm":         absolute gradient norm (for emb/out)
  - "ratio":          g_norm / mean_layer_g_norm (for transformer layers)
  - "growth_rate":    w_norm growth / mean_layer_w_norm growth (relative weight accumulation)
  - "out_emb_growth": out w_norm growth / emb w_norm growth (out/emb divergence)
  - "w_rms_target":   setpoint controller — drives each component toward a target w_rms

Usage:
    awd = AdaptiveWD(model, config, ddp_rank, ddp_world_size)

    # In training loop, after clip_grad_norm_ but before optimizer.step():
    awd.compute_and_update(step)    # reads gradients/weights, updates EMA + multipliers
    # ... set base WD in wd_overrides from rules ...
    awd.apply_multipliers()         # multiply wd_overrides by AWD multipliers
"""

import math
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any


class AdaptiveWD:
    """
    Adaptive Weight Decay controller.

    Tracks per-component gradient norms and/or weight norms,
    maintains EMA-smoothed values, and produces WD multipliers.
    """

    def __init__(
        self,
        model,
        config: dict,
        ddp_rank: int,
        ddp_world_size: int,
        ddp: bool = True,
        base_wd: float = 0.0,
        wd_overrides: dict = None,
    ):
        """
        Args:
            model: Raw model (unwrapped from torch.compile / FSDP).
            config: The adaptive_wd dict from YAML config.
            ddp_rank: This process's rank.
            ddp_world_size: Total number of processes.
            ddp: Whether we're in distributed mode.
            base_wd: Scalar WD fallback for params without wd_overrides entry.
            wd_overrides: Shared side-dict (keyed by id(param)) for per-param WD.
        """
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.ddp = ddp
        self._base_wd = base_wd
        self.wd_overrides = wd_overrides if wd_overrides is not None else {}

        # Parse top-level config
        self.check_interval = config.get('check_interval', 50)
        self.smoothing = config.get('smoothing', 0.9)

        # Parse group configs
        self.groups = []
        for g in config.get('groups', []):
            self.groups.append({
                'target': g['target'],
                'sublayer': g.get('sublayer', 'all'),
                'metric': g.get('metric', 'g_norm'),
                'target_value': g.get('target_value'),
                'target_ratio': g.get('target_ratio'),
                'tolerance': g.get('tolerance'),
                'engage_above': g.get('engage_above'),
                'ease_below': g.get('ease_below'),
                'min_wd_multiplier': g.get('min_wd_multiplier', 0.2),
                'max_wd_multiplier': g.get('max_wd_multiplier', 15.0),
                'aggression': g.get('aggression', 0.1),
                'recovery': g.get('recovery', 0.95),
                # PID gains for w_rms_target setpoint control.
                # Defaults tuned via closed-loop plant simulation against actual
                # training diagnostics.  The key insight: normalized error is small
                # (e.g. 0.67 when w_rms is 1.5× target), so Kp must be large to
                # generate meaningful multiplier force.  High Ki + large integral_max
                # lets the controller reach the steady-state multiplier (~11× for
                # target=0.09, base_wd=0.1).  pid_smoothing=0 avoids the control
                # lag that was the #1 failure mode of the old I-only controller.
                # Kd≈0 is optimal — plant dynamics are too slow (100-step intervals)
                # for derivative action to help.
                'kp': g.get('kp', 90.0),             # Proportional gain
                'ki': g.get('ki', 50.0),              # Integral gain
                'kd': g.get('kd', 0.0),               # Derivative gain (on PV, not error)
                'integral_max': g.get('integral_max', 50.0),  # Anti-windup clamp
                'pid_smoothing': g.get('pid_smoothing', 0.0),  # PV EMA (0=raw, fastest response)
            })

        # Check which metric families are needed
        metrics_used = {g['metric'] for g in self.groups}
        self._needs_grad = bool(metrics_used & {'g_norm', 'ratio'})
        self._needs_wnorm = bool(metrics_used & {'growth_rate', 'out_emb_growth', 'w_rms_target'})
        self._needs_growth = bool(metrics_used & {'growth_rate', 'out_emb_growth'})

        # Build component → param mapping from model structure
        self._component_names: List[str] = []
        self._component_params: Dict[str, List[torch.nn.Parameter]] = {}
        self._component_num_params: Dict[str, int] = {}
        self._param_to_component: Dict[int, str] = {}
        self._build_component_map(model)

        # Resolve which groups target which components
        self._group_components: List[List[str]] = []
        self._component_to_groups: Dict[str, List[int]] = {}
        self._match_groups()

        # Per-component state
        self._ema: Dict[str, float] = {}
        self._multipliers: Dict[str, float] = {}
        self._last_mean_g_norm: float = 1.0

        # Cold-start: init w_rms_target components at min_wd_multiplier
        # for the LEGACY (non-PID) controller. PID computes mult directly
        # from error state, so no seeding needed — default 1.0 is correct.
        # NOTE: With PID enabled (kp/ki/kd set), this block is a no-op
        # because PID overwrites the multiplier on first check_interval.
        for gidx, group in enumerate(self.groups):
            if group['metric'] == 'w_rms_target':
                # Skip PID-configured groups — they don't need cold-start
                if group.get('kp', 0) > 0 or group.get('ki', 0) > 0:
                    continue
                min_mult = group['min_wd_multiplier']
                if min_mult != 1.0:
                    for comp in self._group_components[gidx]:
                        if comp not in self._multipliers:
                            self._multipliers[comp] = min_mult

        # Weight norm tracking state
        self._prev_w_norms: Dict[str, float] = {}  # w_norm at previous check
        self._last_mean_layer_growth: float = 0.0
        self._last_emb_growth: float = 0.0

        # PID controller state (for w_rms_target metric)
        self._pid_integral: Dict[str, float] = {}    # accumulated integral error
        self._pid_prev_pv: Dict[str, float] = {}     # previous PV for D term (avoids setpoint kick)

    def _build_component_map(self, model):
        """Walk model structure and build component → param lists."""
        # Embeddings
        if hasattr(model, 'tok_embeddings'):
            self._component_names.append('emb')
            self._component_params['emb'] = [model.tok_embeddings.weight]

        # Output head (if not tied)
        if hasattr(model, 'output') and model.output is not None:
            # Check for weight tying
            tied = model.tok_embeddings.weight is model.output.weight
            if not tied:
                self._component_names.append('out')
                self._component_params['out'] = [model.output.weight]

        # Transformer layers
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                # Attention: GDN projections or softmax wq/wk/wv/wo
                attn_name = f'L{i}.attn'
                if getattr(layer, 'use_gdn', False):
                    gdn = layer.gdn_attn
                    attn_params = [gdn.q_proj.weight, gdn.k_proj.weight,
                                   gdn.v_proj.weight, gdn.o_proj.weight,
                                   gdn.g_proj.weight]
                else:
                    attn = layer.attention
                    attn_params = [attn.wq.weight, attn.wk.weight,
                                   attn.wv.weight, attn.wo.weight]
                    if hasattr(attn, 'g_proj'):
                        attn_params.append(attn.g_proj.weight)
                self._component_names.append(attn_name)
                self._component_params[attn_name] = attn_params

                # FFN: w1, w2, w3 (dense) or MoE expert/shared params
                ffn_name = f'L{i}.ffn'
                if getattr(layer, 'moe_enabled', False):
                    # MoE layer: track expert 3D params + shared expert 2D params
                    expert_params = [layer.moe.experts.w1, layer.moe.experts.w2, layer.moe.experts.w3]
                    ep_degree = getattr(layer.moe, 'ep_degree', 1)
                    ffn_params = list(expert_params)
                    # With EP, each rank holds only local experts — scale numel
                    # by ep_degree so w_rms_target uses the global param count
                    # (all_reduce(SUM) aggregates norms across all EP ranks)
                    expert_numel = sum(p.numel() for p in expert_params) * ep_degree
                    shared_numel = 0
                    if layer.moe.shared_experts is not None:
                        se = layer.moe.shared_experts
                        shared_params = [se.w1.weight, se.w2.weight, se.w3.weight]
                        ffn_params.extend(shared_params)
                        shared_numel = sum(p.numel() for p in shared_params)
                    self._component_names.append(ffn_name)
                    self._component_params[ffn_name] = ffn_params
                    self._component_num_params[ffn_name] = expert_numel + shared_numel
                else:
                    ff = layer.feed_forward
                    ffn_params = [ff.w1.weight, ff.w2.weight, ff.w3.weight]
                    self._component_names.append(ffn_name)
                    self._component_params[ffn_name] = ffn_params

        # Build reverse mapping and param counts
        for name, params in self._component_params.items():
            if name not in self._component_num_params:
                self._component_num_params[name] = sum(p.numel() for p in params)
            for p in params:
                self._param_to_component[id(p)] = name

    def _match_groups(self):
        """Resolve group targets to component name lists."""
        for gidx, group in enumerate(self.groups):
            target = group['target']
            sublayer = group['sublayer']
            components = []

            if target == 'emb':
                if 'emb' in self._component_params:
                    components = ['emb']
            elif target == 'out':
                if 'out' in self._component_params:
                    components = ['out']
            elif isinstance(target, list) and len(target) == 2:
                start, end = int(target[0]), int(target[1])
                for i in range(start, end + 1):
                    if sublayer in ('all', 'attn'):
                        name = f'L{i}.attn'
                        if name in self._component_params:
                            components.append(name)
                    if sublayer in ('all', 'ffn'):
                        name = f'L{i}.ffn'
                        if name in self._component_params:
                            components.append(name)

            self._group_components.append(components)

            for comp in components:
                if comp not in self._component_to_groups:
                    self._component_to_groups[comp] = []
                self._component_to_groups[comp].append(gidx)

    def _compute_w_norms(self, device) -> Dict[str, float]:
        """Compute weight norms for each component. Batched all_reduce for FSDP2 sharded params."""
        n = len(self._component_names)
        local_sq = torch.zeros(n, device=device)
        for i, name in enumerate(self._component_names):
            for p in self._component_params[name]:
                local_sq[i] += p.detach().float().pow(2).sum()

        if self.ddp and self.ddp_world_size > 1:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

        norms = local_sq.sqrt()
        return {name: norms[i].item() for i, name in enumerate(self._component_names)}

    def compute_and_update(self, step: int) -> bool:
        """
        Compute metrics and update multipliers if step is a check interval.

        Must be called AFTER gradient clipping, while gradients are still present.
        Returns True if multipliers were updated.
        """
        # Cold start: fire immediately on first call (e.g. resume with no prior AWD state)
        if step % self.check_interval != 0 and len(self._ema) > 0:
            return False

        n = len(self._component_names)
        if n == 0:
            return False

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # =====================================================================
        # Gradient norms (for g_norm / ratio metrics)
        # =====================================================================
        g_norms = None
        mean_g = 1.0
        if self._needs_grad:
            local_sq = torch.zeros(n, device=device)
            for i, name in enumerate(self._component_names):
                for p in self._component_params[name]:
                    if p.grad is not None:
                        local_sq[i] += p.grad.detach().float().pow(2).sum()

            if self.ddp and self.ddp_world_size > 1:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

            g_norms = local_sq.sqrt()

            layer_indices = [i for i, name in enumerate(self._component_names)
                             if name.startswith('L')]
            if layer_indices:
                self._last_mean_g_norm = g_norms[layer_indices].mean().item()
            else:
                self._last_mean_g_norm = 1.0

            mean_g = self._last_mean_g_norm if self._last_mean_g_norm > 1e-10 else 1.0

        # =====================================================================
        # Weight norms (for growth_rate / out_emb_growth / w_rms_target)
        # =====================================================================
        w_norms: Dict[str, float] = {}
        growths: Dict[str, float] = {}
        mean_layer_growth = 0.0
        emb_growth = 0.0
        growth_valid = False

        if self._needs_wnorm:
            w_norms = self._compute_w_norms(device)

            if self._needs_growth:
                if self._prev_w_norms:
                    for name in self._component_names:
                        prev = self._prev_w_norms.get(name)
                        if prev is not None:
                            growths[name] = w_norms[name] - prev

                    layer_growths = [growths[name] for name in self._component_names
                                     if name.startswith('L') and name in growths]
                    if layer_growths:
                        mean_layer_growth = sum(layer_growths) / len(layer_growths)
                        self._last_mean_layer_growth = mean_layer_growth

                    emb_growth = growths.get('emb', 0.0)
                    self._last_emb_growth = emb_growth

                    growth_valid = True

                # Store current w_norms for next check
                self._prev_w_norms = w_norms

        # =====================================================================
        # EMA + multiplier updates
        # =====================================================================
        alpha = self.smoothing

        for i, name in enumerate(self._component_names):
            for gidx in self._component_to_groups.get(name, []):
                group = self.groups[gidx]
                metric_type = group['metric']

                # --- Compute raw metric value ---
                if metric_type == 'g_norm':
                    raw = g_norms[i].item() if g_norms is not None else 0.0
                elif metric_type == 'ratio':
                    raw = (g_norms[i].item() / mean_g) if g_norms is not None else 0.0
                elif metric_type in ('growth_rate', 'out_emb_growth'):
                    if not growth_valid:
                        continue
                    raw_growth = growths.get(name, 0.0)
                    if metric_type == 'growth_rate':
                        if mean_layer_growth <= 0:
                            continue
                        raw = raw_growth / mean_layer_growth
                    else:
                        if emb_growth <= 0:
                            continue
                        raw = raw_growth / emb_growth
                elif metric_type == 'w_rms_target':
                    # =============================================================
                    # PID setpoint controller (self-contained)
                    #
                    # Skips the global EMA below — PID uses its own lighter-smoothed
                    # process variable. Computes multiplier DIRECTLY from error state
                    # rather than incremental multiplicative updates.
                    # =============================================================
                    if not w_norms:
                        continue
                    w_norm = w_norms.get(name, 0.0)
                    num_params = self._component_num_params.get(name, 1)
                    raw = w_norm / math.sqrt(num_params)

                    target_val = group['target_value']
                    if target_val is None or target_val <= 0:
                        continue

                    # PID process variable — default is raw (no smoothing) for
                    # fastest response.  Increase pid_smoothing only if measurement
                    # noise causes controller oscillation.
                    pid_alpha = group.get('pid_smoothing', 0.0)
                    pid_pv_key = f"{name}:pid_pv"
                    pid_raw_key = f"{name}:pid_raw"
                    self._ema[pid_raw_key] = raw  # always store raw for diagnostics
                    if pid_pv_key not in self._ema:
                        self._ema[pid_pv_key] = raw
                    elif pid_alpha > 0:
                        self._ema[pid_pv_key] = pid_alpha * self._ema[pid_pv_key] + (1 - pid_alpha) * raw
                    else:
                        self._ema[pid_pv_key] = raw  # smoothing=0: use raw directly
                    pv = self._ema[pid_pv_key]

                    # Normalized error: positive = above target = need more WD
                    error = (pv - target_val) / target_val

                    # PID gains
                    kp = group['kp']
                    ki = group['ki']
                    kd = group['kd']
                    integral_max = group['integral_max']

                    # --- I term: accumulated error with anti-windup ---
                    pid_key = f"{name}:{gidx}"
                    prev_integral = self._pid_integral.get(pid_key, 0.0)
                    new_integral = max(-integral_max, min(integral_max, prev_integral + error))

                    # --- D term: derivative on PV (not error) to avoid setpoint kick ---
                    # If target_value changes on resume, prev_pv is still a valid
                    # measurement — unlike prev_error which embeds the old target.
                    prev_pv = self._pid_prev_pv.get(pid_key, pv)  # init to current (no kick)
                    d_pv = (pv - prev_pv) / target_val  # normalized rate of PV change

                    # --- PID output ---
                    mult = 1.0 + kp * error + ki * new_integral + kd * d_pv

                    # --- Anti-windup: freeze integral when output is saturated ---
                    min_mult = group['min_wd_multiplier']
                    max_mult = group['max_wd_multiplier']
                    if mult > max_mult and error > 0:
                        new_integral = prev_integral
                    elif mult < min_mult and error < 0:
                        new_integral = prev_integral

                    # Store PID state
                    self._pid_integral[pid_key] = new_integral
                    self._pid_prev_pv[pid_key] = pv

                    # Clamp and store multiplier
                    mult = max(min_mult, min(mult, max_mult))
                    self._multipliers[name] = mult
                    continue  # w_rms_target fully handled by PID

                else:
                    continue

                # =============================================================
                # Threshold-based metrics only below this point
                # (g_norm, ratio, growth_rate, out_emb_growth)
                # =============================================================

                # --- Global EMA update ---
                ema_key = f"{name}:{gidx}"
                if ema_key not in self._ema:
                    self._ema[ema_key] = raw
                else:
                    self._ema[ema_key] = alpha * self._ema[ema_key] + (1 - alpha) * raw
                smoothed = self._ema[ema_key]

                # --- Multiplier adjustment ---
                mult = self._multipliers.get(name, 1.0)

                target = group['target_ratio'] if metric_type in ('ratio', 'growth_rate', 'out_emb_growth') else group['target_value']
                if target is None:
                    continue
                engage = group['engage_above']
                ease = group['ease_below']
                if engage is None or ease is None:
                    continue
                if smoothed > engage:
                    excess = smoothed - target
                    mult *= (1.0 + group['aggression'] * excess)
                elif smoothed < ease:
                    mult *= group['recovery']
                else:
                    # Dead zone: recover toward 1.0 (neutral)
                    recovery = group['recovery']
                    if mult < 1.0:
                        mult = min(mult / recovery, 1.0)
                    elif mult > 1.0:
                        mult *= recovery

                mult = max(group['min_wd_multiplier'], min(mult, group['max_wd_multiplier']))
                self._multipliers[name] = mult

        return True

    def apply_multipliers(self):
        """
        Apply AWD multipliers to wd_overrides dict.

        Must be called AFTER the existing WD rules loop populates wd_overrides.
        For params without an override entry, falls back to self._base_wd.
        """
        for comp_name, mult in self._multipliers.items():
            if mult == 1.0:
                continue
            for param in self._component_params[comp_name]:
                pid = id(param)
                base_wd = self.wd_overrides.get(pid, self._base_wd)
                if base_wd > 0:
                    self.wd_overrides[pid] = base_wd * mult

    def has_active_multipliers(self) -> bool:
        """Return True if any multiplier is not 1.0."""
        return any(m != 1.0 for m in self._multipliers.values())

    def format_log_line(self, step: int) -> str:
        """
        Format AWD status line.

        Shows components where mult != 1.0, plus always the terminal
        (last) component of each group for context.
        """
        shown = set()
        parts = []

        for gidx, group in enumerate(self.groups):
            components = self._group_components[gidx]
            if not components:
                continue

            metric_type = group['metric']
            terminal = components[-1]

            for comp in components:
                mult = self._multipliers.get(comp, 1.0)

                if mult != 1.0 or comp == terminal:
                    if comp in shown:
                        continue
                    shown.add(comp)

                    ema_key = f"{comp}:{gidx}"
                    ema = self._ema.get(ema_key, 0.0)

                    if metric_type == 'ratio':
                        part = f"{comp} ratio={ema:.1f} mult={mult:.2f}x"
                    elif metric_type == 'growth_rate':
                        part = f"{comp} growth={ema:.2f} mult={mult:.2f}x"
                    elif metric_type == 'out_emb_growth':
                        part = f"{comp} out/emb={ema:.2f} mult={mult:.2f}x"
                    elif metric_type == 'w_rms_target':
                        pid_pv_key = f"{comp}:pid_pv"
                        pv = self._ema.get(pid_pv_key, ema)
                        target_val = self.groups[gidx].get('target_value', 0)
                        pid_i_key = f"{comp}:{gidx}"
                        integ = self._pid_integral.get(pid_i_key, 0.0)
                        part = f"{comp} pv={pv:.4f} tgt={target_val:.4f} I={integ:+.2f} mult={mult:.2f}x"
                    else:  # g_norm
                        part = f"{comp} g_norm={ema:.2f} mult={mult:.2f}x"

                    if mult != 1.0:
                        sample_param = self._component_params[comp][0]
                        eff_wd = self.wd_overrides.get(id(sample_param), self._base_wd)
                        part += f" (wd={eff_wd:.4f})"

                    parts.append(part)

        return f"awd @ step {step}: {' | '.join(parts)}" if parts else f"awd @ step {step}: (all nominal)"

    def get_diagnostics_data(self) -> dict:
        """
        Return per-component AWD state for diagnostics JSONL.

        Returns dict like:
            {"emb": {"mult": 1.0, "eff_wd": 0.03, "metrics": {"g_norm": 0.05}},
             "out": {"mult": 1.8, "eff_wd": 0.054, "metrics": {"g_norm": 2.82}},
             "L65.attn": {"mult": 2.3, "eff_wd": 0.069, "metrics": {"ratio": 12.1}},
             ...}
        """
        data = {}
        for name in self._component_names:
            gidxs = self._component_to_groups.get(name, [])
            if not gidxs:
                continue

            mult = self._multipliers.get(name, 1.0)
            sample_param = self._component_params[name][0]
            eff_wd = self.wd_overrides.get(id(sample_param), self._base_wd)

            metrics = {}
            for gidx in gidxs:
                metric_type = self.groups[gidx]['metric']
                ema_key = f"{name}:{gidx}"
                ema = self._ema.get(ema_key, 0.0)
                metrics[metric_type] = round(ema, 6)

                # Include PID breakdown for w_rms_target components
                if metric_type == 'w_rms_target':
                    pid_pv_key = f"{name}:pid_pv"
                    pid_raw_key = f"{name}:pid_raw"
                    pid_key = f"{name}:{gidx}"
                    target_val = self.groups[gidx].get('target_value', 0)
                    pv = self._ema.get(pid_pv_key, 0.0)
                    error = (pv - target_val) / target_val if target_val > 0 else 0.0
                    metrics['w_rms_raw'] = round(self._ema.get(pid_raw_key, 0.0), 6)
                    metrics['pid_pv'] = round(pv, 6)
                    metrics['pid_error'] = round(error, 4)
                    metrics['pid_integral'] = round(self._pid_integral.get(pid_key, 0.0), 4)
                    metrics['pid_prev_pv'] = round(self._pid_prev_pv.get(pid_key, 0.0), 4)

            data[name] = {
                'mult': round(mult, 4),
                'eff_wd': round(eff_wd, 6),
                'metrics': metrics,
            }
        return data

    # v1: ema (plain keys), multipliers
    # v2: ema (compound keys name:gidx), multipliers, prev_w_norms
    # v3: + PID state (pid_integral, pid_prev_pv)
    _STATE_VERSION = 3

    def state_dict(self) -> dict:
        """Return AWD state for checkpointing."""
        return {
            'version': self._STATE_VERSION,
            'ema': dict(self._ema),
            'multipliers': dict(self._multipliers),
            'prev_w_norms': dict(self._prev_w_norms),
            'pid_integral': dict(self._pid_integral),
            'pid_prev_pv': dict(self._pid_prev_pv),
        }

    def load_state_dict(self, state: dict):
        """Restore AWD state from checkpoint."""
        version = state.get('version', 1)
        self._multipliers = state.get('multipliers', {})
        self._prev_w_norms = state.get('prev_w_norms', {})
        if version >= 2:
            self._ema = state.get('ema', {})
        else:
            # v1 used plain component names as EMA keys — incompatible with
            # v2's compound keys (name:gidx). Discard and cold-start EMAs.
            self._ema = {}
        if version >= 3:
            self._pid_integral = state.get('pid_integral', {})
            self._pid_prev_pv = state.get('pid_prev_pv', {})
        else:
            # v2 had no PID state — cold-start PID terms.
            # Multipliers are preserved so the output won't jump.
            self._pid_integral = {}
            self._pid_prev_pv = {}