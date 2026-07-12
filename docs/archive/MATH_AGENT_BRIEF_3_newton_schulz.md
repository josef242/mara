# Body-Norm Ramp — Brief #3 for the Math Agent (2026-06-23): IT'S NEWTON-SCHULZ

**Major correction. The whole "structural FSDP2 sharding" conclusion from brief #2 is WRONG —
it was a measurement-labeling error. The −0.0129 lean is the Newton-Schulz UPDATE, not the raw
gradient. Your instinct (path non-equivalence / measurement audit) pointed the right way.**

## The error
The in-situ probe's `cos_grad_W` reads `p.grad` AFTER `optimizer.step()`. But Muon's step()
**scatters the Newton-Schulz-orthogonalized update back into `p.grad`** (muon_fsdp2.py:357). So
`cos_grad_W` measured the POST-NS update direction, never the raw CE gradient. (A fast probe
reading `.grad` PRE-step gave ~0, which surfaced the discrepancy.)

## The decisive measurement (mf, bf16, T=8192, SAME step, WD_PREPOST_PROBE)
| quantity | cos(·, W) median | negfrac |
|---|---|---|
| **RAW CE gradient** (pre-step `.grad`) | **−0.0000029** | 58% (sign-random) → **NULL** |
| **post-Newton-Schulz update** (post-step `.grad`) | **−0.01285** | **100%** |

Self-consistency check: the applied ΔW = −eff_lr·update, so applied `cos(ΔW,W) = +0.01285`,
matching the independently-measured `cos(ΔW,W)=+0.0129` and the log-parsed (probe-independent)
`‖W‖` ramp. So the growth is real; only its SOURCE was mislabeled.

## What this overturns / clarifies
- **Part B was correct all along.** Raw CE gradient is radial-null (+0.0003 sign-random).
  There was NEVER a "Part B vs in-situ contradiction" — they measured different quantities
  (raw gradient vs post-NS update). Months of reconciliation effort chased a labeling artifact.
- **Mechanism = Newton-Schulz orthogonalization**, NOT CE physics, NOT data, NOT FSDP sharding,
  NOT bf16/fp32 precision. Every prior leaning number (−0.0129 in-situ, −0.0075 replicated-data,
  −0.0159 fp32, "structural sharding") measured the post-NS update. They leaned because NS leans.
  (Single-card "null" baselines used an offline non-Muon probe → no NS → naturally null. That's
  why unsharded looked null — not sharding, just absence of the Muon step.)
- **The ~40% "data" component** (−0.0129→−0.0075 under replicated data) is now also suspect as a
  post-NS-of-different-data effect, not raw-CE physics. Needs re-examination with the raw grad.

## The harm is REAL and unchanged (this is why it's worth fixing)
NS produces an update with a systematic radial component; descent applies +radial → ‖W‖ grows.
Decoupled WD removes ηλ‖W‖ which grows with ‖W‖ → WD's share of the update reached **99.8%**,
starving real learning → runs have a finite useful lifespan. Established independently of the
probe (Part A log-parse).

## The proposed FIX
Tangent-project the Muon update after Newton-Schulz, before it's applied:
[ \Delta W \leftarrow \Delta W - W\,\frac{\langle \Delta W, W\rangle}{\lVert W\rVert^2} ]
Removes the radial component at the optimizer source. No WD, no precision change, no FSDP
rewrite. A few lines in muon_fsdp2. (This was the doc's "future surgical lever"; now it's THE fix.)

## QUESTIONS FOR THE MATH AGENT
1. **Why does Newton-Schulz orthogonalization inject a systematic ANTI-radial component into the
   update when the input gradient is radial-null?** NS5 (quintic Newton-Schulz, `zeropower_via_
   newtonschulz5`) drives singular values toward 1, i.e. `G → U V^T` (the matrix-sign / polar
   factor). For a gradient `G ⟂ W` (in the `⟨G,W⟩=0` sense), why would `polar(G)` have
   `⟨polar(G), W⟩ < 0` systematically? Is this a known property of the polar factor / matrix sign
   relative to a reference matrix `W`?
2. **Why is it 100% sign-consistent (always anti-radial in the update → outward in ‖W‖)?** What
   makes the sign uniform across all 490 body matrices and all depths? (This uniformity was the
   strongest clue throughout — now it must come from NS, not sharding.)
3. **Is tangent-projection (above) the right fix, and does it compose with NorMuon's neuron-wise
   normalization?** NorMuon rescales the update per-neuron after NS (`apply_normuon` via the warm
   2nd-moment). Projecting out the radial component — before or after apply_normuon? Does the
   projection need a paired WD reduction or it just shrinks the body?
4. **Does removing the radial component hurt the optimizer's intent?** NS orthogonalization is
   the whole point of Muon (condition the update). Is the radial component a benign side-effect
   safe to strip, or does it carry signal? (My read: it's loss-null in the raw gradient, so the
   radial part of the *update* is NOT loss-driven — safe to remove. Confirm?)
5. **Equilibrium / sizing:** with the radial component stripped, ‖W‖ should stop ramping (or ramp
   only from any residual). Does this change the WD we'd want (probably reduce it, since WD was
   compensating the NS radial push)? 

## Status / next
- Branch `body-ramp-fsdp-probe`. Probe `WD_PREPOST_PROBE`. Doc correction banner on
  WD_WASTE_ANALYSIS.md.
- NEXT (planned): prototype the tangent-projection in muon_fsdp2, re-run WD_PREPOST_PROBE, expect
  cos(projected update, W) → ~0 and body-norm slope → ~0. Then a short training branch to confirm
  ‖W‖ flattens without loss/throughput cost.
- POLICY: still no body-WD / renorm / head-WD until the fix is validated (and the fix likely
  REPLACES the WD lever entirely).
