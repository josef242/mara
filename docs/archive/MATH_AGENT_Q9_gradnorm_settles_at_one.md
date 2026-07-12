# Math Agent Q9 — why does the global gradient norm settle at ≈1.0 under tangent projection?

## The question (the thing we actually want to understand)
Both tangent-projected runs (kv2, KeelHaul-v1) drive their **global pre-clip gradient norm `nrm`**
to a steady-state hover at **≈1.0**, while the non-projected run (DN2) settles 3× lower (~0.3) on a
completely different curve. We can explain why projection makes `nrm` *plateau* (flat ‖W‖). We
**cannot** explain why the plateau sits at **≈1.0 specifically** rather than any other constant. Is
there a real attractor pinning it near unity, and what sets that level?

`nrm` definition: the global L2 norm of the **full raw gradient** (Σ over all params), measured
**pre-clip, pre-optimizer** — i.e. the clip quantity, before Newton-Schulz/NorMuon touch it. (Logged
as the `nrm:` field on the per-step train line.)

## Headline data — settled `nrm` (steps ≥ 4000, post the GA-batch ramp)
`nrm` is right-skewed (occasional spikes to 15+), so the **trimmed-mean** (drop top/bottom 2%) is the
honest center; raw mean and median bracket it.

| run | **trimmed-mean** | median | mean | std | n | projection | body ‖W‖ |
|-----|------------------|--------|------|-----|---|------------|----------|
| **kv2**       | **1.011** | 0.89 | 1.05 | 0.49 | 1730 | ON | flat ~490 |
| **KeelHaul-v1** | **1.030** | 0.91 | 1.07 | 0.53 | 1082 | ON | flat ~490 |
| **DN2 (dreadnought_v2)** | 0.304 | 0.26 | 0.32 | 0.25 | 14203 | OFF | grows 720→1345 |

## kv2 ≈ KeelHaul: near-identical descent to the 1.0 plateau
Windowed-mean `nrm` per 1000 steps. kv2 and KeelHaul are the **same model + same projection**,
differing ONLY in the body-LR anneal (KeelHaul none, kv2 an lr_mod schedule) — yet their `nrm`
trajectories are nearly identical, which isolates the settling as a **projection-family property**:

| step | kv2 | KeelHaul | DN2 |
|------|-----|----------|-----|
| 0–1k | 6.47 | 6.56 | 6.48 |
| 1–2k | 2.63 | 2.65 | 1.75 |
| 2–3k | 2.05 | 2.05 | 0.98 |
| 3–4k | 1.70 | 1.68 | 0.72 |
| **4–5k** | **1.08** | **1.07** | 0.25 |
| **5–6k** | **1.01** | **1.09** | 0.22 |
| 6k+ | (run length) | (run length) | 0.20 → 0.17 → … → 0.80 @18k |

Per-500-step, |kv2 − KeelHaul| ≤ 0.06 the entire way down, and **0.006** at the 1.0 plateau.

## What we CAN explain — the scale-invariance half
The body is scale-invariant (RMSNorm everywhere), so for a scale-invariant loss `‖∇_W L‖ ∝ 1/‖W‖`.
Tested directly via `nrm × ‖W‖` (should be ~constant if that law holds):

**DN2 fits it cleanly** in its body-dominated window:

| step | body ‖W‖ | nrm (med) | nrm × ‖W‖ |
|------|----------|-----------|-----------|
| 5000 | 927 | 0.206 | **191** |
| 6000 | 1010 | 0.190 | **192** |
| 8000 | 1145 | 0.163 | **186** |

Constant to ~3%. So as DN2's body grows, its grad-norm falls as 1/‖W‖ toward ~0.2. (It rises again
past 11k — exactly when DN2's auxiliary heads switch on and inject gradient: 10k→12k product 262→372.)

**Projection pins it.** kv2/KeelHaul hold ‖W‖ flat (~490), so the 1/‖W‖ decline *cannot happen* and
`nrm` stops falling. That explains why it **plateaus**.

## What we CANNOT explain — why the plateau is at ≈1.0
The scale-invariance argument explains *flat*, not the *level*. Two things make ≈1.0 a real puzzle:
1. Within the flat-‖W‖ window (steps 3k→5k), kv2's `nrm` still **declines** 1.48→0.93 while ‖W‖ is
   flat — so `nrm × ‖W‖` *falls* (721→459). The plateau level is therefore set by the *training
   dynamics* (variance, curvature) at the frozen ‖W‖, **not** by the 1/‖W‖ law. Why those dynamics
   land at ~1.0 is open.
2. It's suggestively round and **reproducible across two runs** (1.011, 1.030) — which is why it
   smells like an attractor rather than a coincidence, but we have no mechanism for the value.

Candidate angles we considered (none conclusive — starting points for you):
- **Newton-Schulz feedback?** Muon's *update* is NS-orthogonalized to ~unit spectral scale
  (independent of grad magnitude). Does the optimizer's unit-scale step somehow drive the *gradient*
  to equilibrate at a related O(1) scale? (We don't see the causal chain — NS normalizes the grad
  *away*, so why would the grad settle at 1?)
- **Clip?** kv2 clip=2.0, KeelHaul similar — not binding at 1.0 (so it's not a clip floor), though
  clipping shaped the noisy early approach (`nrm` started ~6.5).
- **A curvature/Hessian scale** of the projected (flat-‖W‖) body equilibrium that happens to be O(1)?
- **Coincidence of model scale / the ‖W‖ value at which projection froze it** — i.e. not universal,
  just where *this* arch + ‖W‖≈490 + batch lands?

## Caveats for a clean read
- kv2/KeelHaul vs DN2 differ in MANY ways (arch 75L/T2048/B8 vs 69L/T8192/B3, WD 0.002 vs 0.02, clip
  2.0 vs 1.0) — so DN2 alone does **not** isolate projection as the cause. The **kv2 ≈ KeelHaul**
  near-identity (differ only in anneal) is what makes the projected-family result robust.
- A clean isolation is coming for free: **dn3** (the grow-then-clamp experiment) takes DN2 and ramps
  projection in (f: 0→1). **Falsifiable prediction:** as f→1 freezes DN2's body, its `nrm` should
  climb off the 1/‖W‖ track and flatten into a plateau. If it plateaus near the *same* O(1) level,
  that's strong evidence the value is intrinsic to projection; if it plateaus elsewhere, the ≈1.0 is
  model/scale-specific. (dn3 launches in a few days.)

## What we're asking
1. Is there a mechanism pinning the projected-body gradient norm to **≈1.0** (an attractor), or is
   that value incidental to this model/‖W‖/batch?
2. What sets the *level* of the plateau, given the scale-invariance law only explains its *flatness*?
3. Does the NS/Muon update geometry feed back on the gradient scale in a way that would prefer O(1)?

— Code (relayed by Josef)
