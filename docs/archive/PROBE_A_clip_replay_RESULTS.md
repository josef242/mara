# Probe A (clip-replay) — RESULTS, KeelHaul step 3500

Ran `tools/clip_replay_probe.py` on `model_step_003500.pt` (rig-31, RTX 4080, fp32 math, n=48
body matrices + 48 Adam-group params, real W from checkpoint, muon hyperparams beta=0.95 ns_steps=5
normuon_beta2=0.99). Read-only: no optimizer.step, no weight writes, no data. Pushes c·G through the
REAL muon_fsdp2 transform (momentum → NS → apply_scaling → NorMuon → tangent proj) and real
adam_update, for clip scales c ∈ {1.0, 0.75, 0.5, 0.25, 0.1}.

## Results

### BODY (Muon) — clip is a NO-OP on the body
| c | cos(U_c,U_1) cold | ‖U_c‖/‖U_1‖ cold | cos(U_c,U_1) warm | cos(U_c,W) |
|-----|------|------|------|------|
| 1.0 | 1.00000 | 1.0000 | 1.00000 | ~0 (1e-11) |
| 0.75| 0.99990 | 1.0000 | 0.99979 | ~0 |
| 0.5 | 1.00000 | 1.0000 | 0.99965 | ~0 |
| 0.25| 1.00000 | 1.0000 | 0.99939 | ~0 |
| 0.1 | 0.99978 | 0.9995 | 0.99923 | ~0 |

- **Direction unchanged** (cos≈1.0) and **magnitude unchanged** (ratio≈1.0) across ALL clip scales,
  cold AND warm. Empirically confirms NS scale-invariance: polar(cG)=polar(G). The body does not
  care about the clip.
- **Warm-momentum direction shift: max 0.0008** (cos drops 1.0→0.9992 only at the extreme c=0.1).
  The momentum-staleness effect the Math Agent flagged is REAL but NEGLIGIBLE at these scales.
- cos(U_c,W)≈0 confirms tangent projection holds regardless of clip (orthogonal to W).

### ADAM groups (head/emb/norms/router) — clip THROTTLES them, monotonically
| c | ‖D_c‖/‖D_1‖ | cos(D_c,D_1) |
|-----|------|------|
| 1.0 | 1.0000 | 1.00000 |
| 0.75| 0.9846 | 0.99999 |
| 0.5 | 0.9656 | 0.99995 |
| 0.25| 0.9431 | 0.99990 |
| 0.1 | 0.9279 | 0.99987 |

- Adam update **shrinks monotonically** with the clip coefficient (direction preserved). At a 10×
  clip (c=0.1) the Adam update is ~93% of unclipped; at KeelHaul's actual operating point
  (nrm~1.5–2.2 → c≈0.45–0.67) the loss is only **~3–5%**.

## Verdict / interpretation

**The Math Agent's structural hypothesis is CONFIRMED IN SIGN:** clipping is a no-op on the Muon
body (magnitude AND direction) and DOES throttle the Adam groups. The asymmetry is exactly as
predicted — the large Muon-body raw gradient sets the global clip coefficient, and the
magnitude-sensitive Adam params (not the body) pay for it.

**BUT the magnitude is modest.** Adam's m/√v normalization is largely scale-cancelling (clipping
shrinks numerator via m and, with lag, denominator via v), so the net throttle is only ~7% even at
10× clip, and ~3–5% at KeelHaul's real nrm. So clip-throttling of Adam is a **real contributor** to
the loss lag, but probably **not the whole story** — the effect at the actual operating point is
small. Worth fixing (it's free), but we should keep looking for a larger cause (LR schedule, data
mix, batch size, SCS differences, or it's within normal DN2-vs-KH variance).

## Implications for action
- **Per-group clipping / raising the clip is still net-positive** (removes the 3–7% Adam throttle at
  zero cost to the body — body is provably clip-invariant). Low-risk, do it.
- **Do NOT expect it to fully close the DN2 gap** — the measured Adam effect is too small to explain
  a 0.3–0.6 nats loss lag by itself.
- Momentum-staleness on the body is negligible (0.0008) — not a concern.
- Next: groupwise norm telemetry (Probe B) to see WHO sets the clip live, and a look at the
  non-clip causes (schedule/data/batch).

## Caveats (honest)
- Gradients are realistic-magnitude synthetic (random, ~5% unit-RMS field). The property under test
  (transform response to scaling c) is gradient-CONTENT-independent — polar(cG)=polar(G) — so this
  is valid for the body. For Adam, the m/√v cancellation depends on the warm-state correlation
  between exp_avg/exp_avg_sq and G; I used a plausibly-correlated warm state (exp_avg≈0.9·G). A
  real-gradient + real-optimizer-state version would tighten the Adam number, but the SIGN and rough
  magnitude are robust.
- Single checkpoint (step 3500). The asymmetry is structural so it shouldn't vary across steps, but
  the exact Adam % could shift with the v-buffer's maturity.

## Robustness (verified)
- **Seed-invariant:** seed 0 vs seed 7 match to 4 decimals (Adam @c0.5: 0.9656 vs 0.9657; body
  cosU@0.1 0.9998 both). Not a random-gradient artifact.
- **Step-invariant:** step 1500 gives an IDENTICAL Adam throttle curve (0.9846/0.9656/0.9431/0.9279)
  and identical body-invariance (warm shift 0.00077) to step 3500 — confirms the asymmetry is
  STRUCTURAL, independent of v-buffer maturity. This strengthens confidence that the body
  clip-invariance and the ~7%-max Adam throttle are real properties of the optimizer stack, not
  numerical luck.
