# Math Agent Brief #6 — Did tangent projection remove an implicit LR-annealing schedule? (KH effective-LR is high and never decays)

## The question
KeelHaul (tangent-projection body-ramp fix, WD 0.002) vs DN2/`dreadnought_v2` (normal NorMuon, the
body-norm ramp we projected out, WD 0.02). KH's raw gradient norm `nrm` settles **~3× higher** than
DN2 at matched tokens, and KH's **val loss crosses from ahead → behind around 500M tokens** and the
gap widens. Josef's hunch: **KH is behaving like it has a much higher (and non-decaying) learning
rate.** The diagnostics below support that strongly. We want your read on the mechanism and the fix
before changing the recipe.

## The data (matched tokens, from each run's diagnostics.jsonl + gen_log; body = attn+ffn matrices)

**Body weight norm — the projection works as designed (KH flat, DN2 ramps):**
| tok(M) | KH ‖W‖ | DN2 ‖W‖ | KH/DN |
|---|---|---|---|
| 100 | 3093 | 3439 | 0.90 |
| 400 | 3120 | 3809 | 0.82 |
| 800 | 3169 | 5071 | **0.62** |

KH ‖W‖ is essentially constant (+2.5% over 800M). DN2 ramps **+47%** — the classic body-norm growth.

**Relative gradient g/w = g_norm/w_norm (weight-norm removed): KH stays high, DN2 lower:**
| tok(M) | KH g/w | DN2 g/w | KH/DN |
|---|---|---|---|
| 200 | 8.1e-5 | 3.5e-5 | 2.3× |
| 500 | 7.7e-5 | 4.9e-5 | 1.6× |
| 800 | 5.9e-5 | 2.7e-5 | 2.2× |

So the 3× raw-`nrm` gap is NOT a weight-norm artifact — it survives (in fact grows) after dividing
by ‖W‖. (We initially suspected DN2's growing ‖W‖ was artificially suppressing its raw nrm; the g/w
table refutes that — KH's *relative* gradient is genuinely ~2× higher.)

**Relative STEP size pdr = ‖ΔW‖/‖W‖ — the effective LR in a scale-invariant body (THE test):**
| tok(M) | KH pdr | DN2 pdr | KH/DN |
|---|---|---|---|
| 100 | 1.44e-3 | 0.78e-3 | 1.83× |
| 400 | 2.96e-3 | 2.68e-3 | 1.10× |
| 600 | 3.41e-3 | 2.55e-3 | 1.34× |
| 800 | 3.41e-3 | 2.30e-3 | **1.49×** |

**The trajectories are the key:** DN2's relative step **peaks ~400M (2.7e-3) then DECAYS**
(2.7→2.5→2.4→2.3e-3). KH's relative step **keeps climbing and plateaus high** (3.0→3.1→3.4→3.4e-3,
never decays). DN2 anneals; KH does not.

**Loss — crossover, not a constant offset (KH−DN2; positive = KH worse):**
| tok(M) | dVal | dTrain |
|---|---|---|
| 100 | −0.05 | −0.10 |
| 300 | −0.16 | −0.06 |
| 400 | −0.20 | −0.07 |  ← KH max lead
| 500 | 0.00 | +0.08 |  ← crossover
| 600 | +0.14 | +0.14 |
| 800 | **+0.21** | +0.12 |  ← DN2 ahead, widening

KH **leads to ~450M, DN2 overtakes ~500M, gap widens to +0.21 nats by 800M.** This is the signature
of a too-high, non-annealing LR: wins the sprint (faster early), loses the marathon (no consolidation).

## Our mechanistic hypothesis (please confirm / refute / refine)
In a pre-norm, scale-invariant body, the per-direction effective learning rate scales like
‖ΔW‖/‖W‖. The Muon update has ~fixed magnitude (NS + apply_scaling + NorMuon renorm — see Brief #5),
so **as ‖W‖ grows, ‖ΔW‖/‖W‖ shrinks: the body NORM RAMP IS AN IMPLICIT LR-DECAY/ANNEALING
SCHEDULE.** DN2 gets this for free (its ramping ‖W‖ anneals the effective LR → relative step peaks
then decays → body consolidates). KH's tangent projection pins ‖W‖ flat, which **removes the
annealing**: the effective body LR stays high and even rises, so KH behaves like a high, non-decaying
LR — explaining the higher `nrm`/g/w, the persistently large pdr, and the win-early-lose-late loss
crossover.

So the body-norm ramp may NOT be pure waste (as the WD investigation implicitly assumed). Part of it
was a useful **annealing mechanism**. The projection threw out the baby (annealing) with the
bathwater (uncontrolled norm growth + WD waste).

## Questions
1. **Is ‖ΔW‖/‖W‖ the right "effective LR" proxy** for this scale-invariant pre-norm body, and does
   the data (DN2 peaks-then-decays vs KH climbs-then-plateaus) support "ramp = implicit LR anneal"?
2. **Is the loss crossover (KH wins to 450M, loses after) consistent with a non-annealing high LR**,
   vs other explanations (WD 10× lower under-regularizing; projection removing useful gradient signal;
   plain variance)? Note WD is also 10× lower in KH (0.002 vs 0.02) — is the gap the missing
   annealing, the lower WD, or both? They're entangled in this comparison.
3. **The proposed fix: pair tangent projection with an EXPLICIT body-LR decay** that reproduces the
   annealing the ramp used to provide — recovering DN2's late consolidation without the norm growth.
   - What schedule? Match DN2's empirical pdr decay (peak ~2.7e-3 at 400M → 2.3e-3 at 800M, ~15%
     decay)? Or a principled cosine/inverse-sqrt on the body LR?
   - Alternatively: is the right knob to **partially** project (leave some radial component so ‖W‖
     grows in a *controlled* way, keeping a milder natural anneal)? Or **lower the body LR globally**
     (KH pdr is ~1.5× DN2 — would a ~1.5× body-LR cut roughly match DN2's effective-LR trajectory)?
4. Does WD interact here — with the ramp gone, is some WD actually *desirable* now (as a different
   route to shrink ‖W‖ and re-introduce anneal), contradicting the "WD is wrong for this regime"
   stance? Or does WD-driven shrink behave differently from ramp-driven growth for annealing purposes?

## What we'll do based on your answer
- If "ramp = implicit anneal" holds → add an explicit body-LR decay to the projected run (one new
  schedule), keep projection + low WD. Re-run from scratch, expect to match-or-beat DN2 with a flat
  body norm AND proper late consolidation.
- If the gap is mostly the 10× lower WD → raise body WD modestly instead (cheaper test).
- If it's projection removing useful signal → reconsider partial projection.

## Reference (all source-grounded, this repo)
- Data tool: `tools/kh_dn2_diag_compare.py` (reads both runs' diagnostics.jsonl + gen_log).
- Effective-LR / magnitude background: `docs/MATH_AGENT_BRIEF_5_norm_clip_vs_muon.md`,
  `docs/PROBE_A_clip_replay_RESULTS.md` (Muon discards raw-grad magnitude; update mag ≈ fixed×lr).
- Projection + ramp mechanism: `docs/MATH_AGENT_BRIEF_3_newton_schulz.md`, `docs/WD_WASTE_ANALYSIS.md`.
- KH config: tangent_project true, preserve_norm false, WD 0.002, clip 2.0, lr 3.5e-4 (cosine).
  DN2: normal NorMuon, WD 0.02, same lr family.
