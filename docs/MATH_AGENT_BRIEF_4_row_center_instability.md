# Brief for Math Agent — Head Row-Centering Instability (2026-06-23)

**SEPARATE from the body-ramp/Newton-Schulz work (that's SOLVED — tangent projection holds body
‖W‖ flat). NEW problem: head LM row-centering has killed TWO runs in two days. We'd like your
thoughts on the mechanism while a clean projection-only run trains.**

## What row-centering does (the intervention under suspicion)
Function-preserving gauge subtraction on the LM head: `W ← W − 1·μ^T` where μ = global vocab-row
mean of the head weight. The softmax is shift-invariant (subtracting a per-position scalar from all
logits leaves CE/softmax/sampling UNCHANGED), so on the MODEL OUTPUT it's exactly function-
preserving at any magnitude. It ALSO projects the Adam 1st moment (exp_avg) by ITS OWN row-mean each
step so the gauge "can't regrow." Main LM head only; assumes untied, bias-free head; z-loss OFF.
Telemetry: muW_pre (gauge magnitude before centering), muW_post (after), head output ‖W‖.

## Two corpses
1. **DN2** (2026-06-22): collapsed during the MID-TRAINING row-center anneal (engaged at step 18000,
   300-step cosine warmup). Artifacts on valhalla: rowcenter_dn2_18500.json, dn2_018250/018500.pt.
   (Full diagnostics.jsonl being recovered from backup; available if you want it.)
2. **Keel Haul v1** (2026-06-23): FROM-SCRATCH run, row-center enabled full-strength from step 0
   (no warmup). DIVERGED ~step 450. Clean complete trajectory below.

## Keel Haul v1 — the runaway, quantified (diagnostics.jsonl)
| step | head output ‖W‖ | muW_pre (gauge) | muW_post | train ls / ppl |
|---|---|---|---|---|
| 100 | 462    | 0.0037 | 0.0036  | 8.13 / 3404 |
| 200 | 471    | 0.0032 | 0.0012  | (healthy) |
| 300 | 902    | 0.0202 | 1.6e-8  | ~4.6 (good) |
| 400 | 3,587  | 0.0609 | 7.2e-8  | ~5.3 |
| 500 | 33,807 | 0.4422 | 6.7e-7  | ~7–12 (climbing) |
| 600 | 110,585| 0.6645 | 2.3e-6  | ppl → billions (diverged) |

**Body matrix ‖W‖ (tangent projection's job) stayed PERFECTLY FLAT throughout** (attn 167.27→167.32,
ffn 315.45), isolating row-centering as the sole culprit. grad-norm went 3.4 → 50 → 973 as it blew.

## The pattern we see
**Centering IS working** (muW_post → ~0 every step — the gauge IS being subtracted to near-zero).
**But the gauge REGROWS bigger each step** (muW_pre 0.0037 → 0.66, ~180×) and the **head norm
INFLATES unbounded** (462 → 110,585, 240×). So it's a RUNAWAY FEEDBACK LOOP: subtract gauge → head
compensates by growing → bigger gauge → bigger subtraction → divergence. Row-centering isn't failing
to center; it appears to be PROVOKING head inflation.

## QUESTIONS FOR THE MATH AGENT
1. **Why would gauge subtraction provoke head-norm growth?** On the output it's function-preserving
   (any μ removed = same logits). So the head norm should be free to do whatever — yet it runs away
   ONLY when centering is on. Is the centering creating a degenerate direction the optimizer then
   inflates into (since the loss is flat along the gauge direction, Muon/Adam can push arbitrarily
   far there with no CE penalty)? I.e. is the gauge direction a NULL direction the optimizer fills?
2. **The Adam-1st-moment projection** (exp_avg row-mean removed each step) — is THAT the destabilizer
   rather than the weight projection? It alters the optimizer trajectory, not just the output.
3. **Is it the same mechanism mid-train (DN2) and from-scratch (Keel Haul)?** DN2 had a trained head
   + 300-step anneal and still died; Keel Haul had random init + full-strength and died faster.
   Common cause, or two different failure modes that both involve centering?
4. **Does z-loss being OFF matter?** Row-centering was designed alongside z-loss (logZ control).
   Without z-loss constraining the head magnitude, is centering removing the only thing that WAS
   implicitly bounding it — leaving the head free to inflate?
5. **Is head row-centering salvageable, and how?** (norm-cap the head? keep z-loss on? center the
   weight but NOT the Adam moment? a much gentler/periodic centering? abandon it for a different head
   gauge fix?) Or is the whole approach unstable and we should drop it?

## Context / status
- The tangent-projection body-ramp fix is PROVEN and unaffected (body norms flat through both the
  good phase AND the divergence). Row-centering is an INDEPENDENT intervention we bundled in and it
  bit us.
- Keel Haul is being relaunched with `row_center_head: enabled: false` (projection-only, clean).
- Memory: row_center_head_instability. Implementation ref: the row_center_head feature.
