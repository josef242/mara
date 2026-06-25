# Centered z-loss — implementation plan (pending Rook greenlight)

**Motivation (from the row-center probe, Nexus #139→#141):** dn2's logZ ≈ 502 decomposes into `h·μ` ≈ 394 (a **CE-invisible common-mode gauge**, 78%) + `logZ_c` ≈ 108 (genuine centered margin, 22%). The current z-loss penalizes **raw** logZ, so ~78% of its gradient budget pushes a free gauge the model can't see in CE. **Centered z-loss** penalizes `logZ_c` instead, aiming 100% of the pressure at the part CE actually responds to.

## 1. The math (VERIFIED — see below)

For token `n` with hidden `h_n` and head `W` (rows `w_v`):
- `logits_v = h_n·w_v`, `logZ = logsumexp_v(logits_v)`.
- Row-mean (common mode): `μ = mean_v(w_v)` (shape `[D]`, a function of W).
- `logZ_c = logsumexp_v(h_n·(w_v − μ)) = logZ − h_n·μ` (exact, since `h_n·μ` is a per-token constant added to every vocab logit, and logsumexp shifts by it).

**Objective A (chosen):** `zloss_c = mean_n( logZ_c² )`, **with gradient flowing through `μ(W)`.**

**Verified property (independent derivation, exact/unconditional):** because `logZ_c` is gauge-invariant (invariant to adding any common vector `c` to all rows), `∂(zloss_c)/∂W` has **exactly zero component along the common-mode direction** — `Σ_v(softmax_{n,v} − 1/V) = 1 − 1 = 0` per token, for all `h`, all `c`. So A **cannot push the gauge, only the centered structure.** The detached-μ variant (Objective B) does NOT have this property (its common-mode gradient is `(2/N)Σ_n r_n h_n`, generically nonzero) → **reject B, use A.**

Scope notes: "zero gauge gradient" = the *sum over rows* vanishes (μ unchanged); individual rows still move (reshaping the zero-sum part of each logit pattern) — exactly the intent. The gauge is the across-vocab common mode (a D-dim subspace); A is orthogonal to precisely that.

## 2. How it drops into `_zloss_optionD` (no new [N,V])

Current code computes, per token (all without materializing logits):
- `ce_none = cce_loss(h, W, tgt, reduction='none')` (CCE fused)
- `logit_target = (h * W[tgt]).sum(-1)`
- `logZ = ce_none + logit_target`

Centered just subtracts the common-mode offset:
- `μ = W.mean(dim=0)` — `[D]`, one mean over the head rows (cheap; differentiable through W).
- `h·μ = h_flat @ μ` — `[N]`, a cheap matvec (NOT `[N,V]`).
- `logZ_c = logZ − h·μ = ce_none + logit_target − h·μ`.
- `zloss_c = mean(logZ_c[valid]²)`.

**Zero added [N,V] materialization** — `μ` is `[D]`, `h·μ` is `[N]`. The only new compute is one `[V,D]→[D]` mean and one `[N,D]·[D]→[N]` matvec. Trivial vs the existing head.

## 3. Q4 — gradient precision (the real engineering risk)

The raw-logZ path is already a bf16 catastrophic-cancellation case (`ce_none = logZ − logit_target`, both ~O(8)), which is why `backend='fp32_accum'` exists. Centering adds **another** subtraction at MUCH larger scale: `logZ_c = logZ − h·μ` with `logZ ≈ 502`, `h·μ ≈ 394`, result `≈ 108`. Subtracting two O(400-500) quantities to get O(100) loses ~2 bits of precision in bf16 — a NEW cancellation.

**Plan:**
- Compute `μ` and `h·μ` in **fp32** (`W.float().mean(0)`, `h_flat.float() @ μ`). The mean over 32k rows especially wants fp32. `μ` is `[D]` and `h·μ` is `[N]` so fp32 here is ~free memory-wise.
- Form `logZ_c` in fp32: `(ce_none.float() + logit_target.float()) − h_dot_mu_fp32`.
- The `fp32_accum` backend (CCE accum flags) still governs the `ce_none`/`logit_target` GRADIENT precision; the centering subtraction is in fp32 on top.
- **MUST re-validate gradient agreement** for the centered path against an fp32 analytic truth, same as we did for raw (the `zloss_variants_rig.py` methodology). The centered analytic truth: `gL_c = (2·logZ_c/Nk)·(softmax − 1/V)` chained to e,c (note the `−1/V` from the μ path — that's the gauge-projection term). Build a `zloss_centered_rig.py` mirroring the variants gate.

## 4. Q2 — config: NEW knob, not a replacement

Keep raw z-loss (live on dn2, comparability). Add:
```yaml
z_loss:
  target: centered     # raw (default, current behavior) | centered
```
- `target: raw` (default) → exactly today's `zloss = mean(logZ²)`. Byte-identical.
- `target: centered` → `zloss = mean(logZ_c²)`.
- Validate in Settings (raw|centered). Thread to the model via a flag like the backend bool, OR fold into the existing `_zloss_fp32_accum` mechanism (e.g. a small `_zloss_target` attr: None=off, else 'raw'/'centered').
- Annealing (`get_zloss_alpha`), `backend`, masking, scaffold selection — all unchanged; only the per-token penalized quantity changes.

## 5. Q3 — logging (keep BOTH raw and centered)

Raw logZ is the diagnostic (what the probe and dashboards track); centered is what's optimized when `target=centered`. Log both so we can watch the gauge drain:
- Keep existing: `zloss` (the OPTIMIZED quantity — becomes mean(logZ_c²) when centered), `logZ`, `logZ_rms`, `logZ_p95`.
- Add when `target=centered`: `logZ_c` (mean), `h_mu` (mean common-mode offset = the gauge magnitude). Watching `h_mu` flat + `logZ_c` falling = centered z-loss working as designed (drains real margin, leaves gauge). Watching `logZ` fall mostly via `h_mu` = the OLD failure mode.
- diagnostics.jsonl `z_loss` block: add `logZ_c`, `h_mu` fields.
- NOTE: when `target=centered`, the `zloss`/`logZ_rms` columns now reflect the CENTERED quantity — keep a raw `logZ` column too so the absolute gauge is still visible. Be explicit in the column semantics so the dashboard parser (Nexus #133/#134) knows.

## 6. Q5 — scaffold / aux heads

Under SCS scaffold, the live readout is the deepest aux tap, and z-loss applies there. Aux heads have ~no common mode (probe: `||W_c||/||W|| ≈ 0.996`), so centering is ~a no-op there — but the code path must still compute `μ` of the **aux head's** weight, not the main head's. `_zloss_optionD` already receives the right `weight` arg per call site (main `self.output.weight` / aux `self.linear.weight`), so centering uses the correct head's μ automatically. ✓ No special-casing.

## 7. Validation checklist (before any enabled centered run)

1. **Disabled / target=raw byte-identical** to current main (no drift). Centering code only runs under `target=centered`.
2. **Centered forward correctness:** `logZ_c` from the reconstruction == `logsumexp(h@(W−μ).T)` from a materialized reference (small dims), fp32, to tolerance. (Reuse the row-center probe's exactness check.)
3. **Centered gradient correctness (rig):** `zloss_centered_rig.py` — grad of `mean(logZ_c²)` w.r.t. e,c vs the fp32 analytic truth `gL_c=(2logZ_c/Nk)(softmax−1/V)` chained. Report cosine/norm-rel per backend (bf16 vs fp32_accum). Confirm the extra subtraction didn't wreck precision (may need the centering in fp32, per Q4).
4. **Gauge-invariance sanity:** confirm CE is unchanged by centering (it must be — `logZ_c` doesn't enter CE) and that the centered gradient's common-mode component is ~0 (numerically verify Σ_v ∂/∂w_v ≈ 0).
5. **Memory:** confirm no [N,V] and negligible delta vs raw z-loss at the head shape.
6. **Scaffold:** centered z-loss on the deepest aux tap computes that head's μ; runs clean.

## 8. Open questions for Rook (with the greenlight)

- **A vs B:** plan uses A (gauge-invariant, verified). Confirm. (B is simpler grad but pushes the gauge — defeats the purpose.)
- **Default for dn2:** if greenlit, switch dn2 to `target: centered` at the next ramp reset, OR run a fresh comparison? (Probably: let current raw run continue to get the dynamic data point, then switch.)
- **alpha re-tune:** centered penalizes logZ_c ≈ 108 not logZ ≈ 502, so `mean(logZ_c²) ≈ 11700` vs `mean(logZ²) ≈ 276000` — ~24x smaller. At fixed alpha the centered z-term is ~24x weaker. So **alpha must be re-scaled up ~24x** for the same fraction-of-CE bite (or re-derive: `alpha_centered = target_frac·CE / mean(logZ_c²)`). Flag this — it's the same logZ-scaling lesson, applied to the centered magnitude.
