# kv3 — FFN pdr controller: LOCKED design spec

Status: **design converged** (Code ↔ Math, Briefs #7 + Q8 + Math's two verdicts). This is the
authoritative build spec. kv2 keeps running open-loop (it's leading; do not disturb). This is kv3.

## One-line statement
> Control FFN angular step size (pdr) with a feedforward controller while leaving the
> structurally self-regularized attention path alone.

Tangent projection = radial hygiene (flat ‖W‖). FFN pdr controller = restores the relative-LR anneal
the projection removed. Attention is left free because QK-norm already gives wq/wk forward
scale-invariance ("tangent projection for free").

## The controller (Math's final spec)
- **Scope**: FFN body matrices only (w1/w2/w3). Attention held at `m=1.0`.
- **Mode**: **feedforward-only** for run 1. `m = r(t) / K_ema`, where `K_ema = EMA(pdr_ffn / m)` in
  **log space** (α ≈ 0.1–0.2 @ 100-step cadence). This is feedback through the *measured plant gain*,
  not an open-loop schedule. PI present behind config but **gains = 0**; enable integral only if live
  tracking shows persistent bias feedforward can't remove. No D ever.
- **Plant** (verified): `pdr = K(t)·m`, linear within a step; `K = group_lr·‖update‖/‖W‖`, measurable.
- **Output limits**: `m ∈ [m_floor, 1.0]`, `m_floor = 0.25–0.35`. `m_max = 1.0` (never amplify above base).
- **Rate limit**: asymmetric — cool ≤5%/sample, reheat ≤1–2%/sample (don't chase noise upward).
- **Cadence**: update every ~100 steps (diagnostic cadence); commit one multiplier per update.
- **Warmup gate**: `m=1.0`, loop frozen, until step 1500 (LR-cap). Then engage.
- **Anti-windup**: freeze integral when output pinned at a rail in the error's direction (PI is off in
  run 1, but wire it correctly).

## The reference r(t) — smooth `dn2_merge` (Math Q8 §7)
**Smooth α-blend, NOT a piecewise cliff:**
```
r_ffn(t) = (1-α(t))·r_kv2_early(t) + α(t)·r_DN2_ffn(t)
α(t) = smoothstep(0 at 197M → 1 at ~575M)        # 3u²-2u³
```
- `r_kv2_early`: near-hold with gentle decline (~3.42e-3 @197M → ~3.10 @400M → ~2.90 @600M) — preserve
  kv2's productive early plasticity.
- `r_DN2_ffn`: DN2's *actual* FFN-median pdr glide (peak 2.94e-3 @393M → 2.28 @800M → 1.26 @26B).
- α=0 near LR-cap (no abrupt command at step 1500), rising smoothly to 1 by ~575M, then follow DN2 down.

**Why not the alternatives** (data-grounded): literal DN2-FFN tracking = m≈0.48 slam at LR-cap (throws
away kv2's working early regime — kv2 ffn 3.42e-3 vs DN2 1.66e-3 @197M). `math_glide` = too gentle late.
DN2's FFN was NEVER hot (f/b ≈ 1.0–1.10) — the hot FFN is projection-specific to kv2/KH, so the
1.15–1.25×-body fallback is wrong. Merge preserves early lead, lands on DN2's proven late band.

## Subgroup pdr diagnostics — ADD IMMEDIATELY (Math item 5)
Split the attention/FFN aggregate into:
- `pdr(wq/wk)` vs `pdr(wv/wo)` — tests the QK-norm hypothesis (only wq/wk get forward
  scale-invariance; wv/wo don't). If attn is quiet only because wq/wk are quiet while wv/wo differ,
  the grouping needs revisiting.
- `pdr(w1/w3)` vs `pdr(w2)` — tells whether to eventually control SwiGLU-in-proj only, w2 only, or
  all FFN. (kv3 controls all-FFN first regardless, but instrument from day one.)

Cheap change in diagnostics.py (per-projection grouping in `_get_attention_params`/`_get_ffn_params`
snapshot keys). Does not affect kv2 (running old code); lands on kv3.

## Known coupling: lr_scale also scales weight decay
The Muon optimizer's `effective_lr = lr · lr_scale` multiplies **both** the update and the WD term.
So cooling `m` cools the FFN's WD by the same factor — `m` is an **LR-dominant**, not pure-LR, lever.
This is benign and not a new confound vs kv2: (a) kv2's open-loop `lr_mods` had the *identical*
coupling; (b) WD=0.002 is tiny; (c) both effects *lower* pdr (less update **and** less ‖W‖ shrink →
higher ‖W‖ → lower pdr), so they're aligned with the controller's goal. If a future run wants pure-LR
control, it would need a separate LR-only override path (not the WD-coupled lr_scale). Flagged by the
failure-modes critic; documented rather than re-architected for run 1.

## Guardrails (live checks for kv3)
- **Multiplier authority**: `m_ffn < 0.5` *before* the merge region → inspect. `m_ffn = m_floor` AND
  `pdr_ffn > 1.1·r(t)` for several samples → base FFN/body LR too high (controller out of authority) →
  then (and only then) lower base body/FFN LR or global max_lr.
- **Do NOT lower global max_lr preemptively** despite sim m_final≈0.62. The problem is FFN-specific;
  a global cut would needlessly cool attention/head/emb/norms/routers. Let the controller reveal it.
  Rule of thumb: `m_ffn ≈ 0.6–0.8` late is fine.
- **Attention preservation**: if attn pdr falls materially below its healthy ~2.25–2.40e-3 band while
  FFN-only control is active, the grouping is wrong (shouldn't happen except via model coupling).
- **Loss guard**: judge by *keeping kv2's early lead* while preventing the late non-decaying FFN pdr
  shelf — NOT by matching DN2 loss early (kv2 is ahead).
- **Tracking**: judge in log space; a few percent error is fine (the target is a healthy schedule, not
  a physical constant).

## Simulation validation (done — tools/pdr_controller_sim.py)
FF-only, smooth `dn2_merge`, 10% noise, worst-case rising K: track_rms 0.036, no freeze/alarm across
15 scenario cells (3 refs × 5 K-futures), oscillation is pure noise-jitter (1 reversal @0% noise),
+20% K jump → +20% transient then recovery, m settles ~0.63. Caveat (Math): proves controller *math*
is stable vs exogenous K; cannot prove closed-loop K matches recorded — hence the 5 K-scenarios.

## Build checklist (for when we implement kv3)
1. Extract the ~35-line PID kernel from `adaptive_wd.py` (w_rms_target branch) into a standalone
   `PIDController` (error → clamped I + saturation-freeze → D-on-PV → log-output → clamp), reuse the
   versioned-`.pt`-on-rank-0 checkpoint convention.
2. `BodyLRController` (FFN-only): PV = `pdr_ffn` (injected, not from `_compute_w_norms`); feedforward
   `m = r/K_ema`; writes `lr_scale_overrides[id(p)]` for FFN body params; PI present, gains 0.
3. Wire at the lr_mod write site ([train_mara.py:1909-1912]) — replace the schedule's value source for
   FFN params; keep attn on the existing path at m=1.0. Must run after clip, before any SCS block.
4. Add subgroup pdr diagnostics (above).
5. `kv3.yaml`: clone kv2, swap `lr_mods` → controller config, keep max_lr/projection/WD unchanged.
6. Long-term (separate from-scratch branch, NOT kv3): structural FFN normalization as the architectural
   analog of QK-norm — candidates: RMSNorm on the SwiGLU product before w2, learned branch gain on a
   normalized FFN output, weight-normed FFN matrices. SwiGLU has no harmless insertion point, so this
   is a research branch, not a midstream fix.

— Code + Math (relayed by Josef)
