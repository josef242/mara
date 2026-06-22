# Weight-Decay Gradient-Waste & Weight-Norm Ramp — Analysis

**Status:** complete (Part A + Part B, 4 checkpoints across 2 architectures incl. a dead run).
**Date:** 2026-06-22. **Nexus thread:** 139 (msg #167 hypothesis, #170 results).
**Tools (offline):** `tools/wd_waste_probe.py` (Part A, log-parse), `tools/wd_waste_partb.py` (Part B, cos test).
**Raw data:** `//valhalla/valhalla/code/ckpt/partb_{dn1_10000,dn1_14000,dn2_18000,mf_35000}.json`

---

## TL;DR / Verdict

Body weight matrices show a **real, ongoing, LR-coupled `||W||` RAMP** (not at equilibrium) — but the ramp is **LOSS-NULL in direction**: the loss-gradient is orthogonal to the body weights to ~0.1% (cos ≈ 0.0002–0.0008), confirmed across two architectures **including the checkpoint that died**. So it is **cosmetic scale inflation, not loss-coupled distortion** — but it is **NOT cost-free** (effective-LR decay + optimizer-dynamics, not clip-triggering).

- **Head ≠ body.** The head is loss-relevant (wd/loss 1.4–4.9× vs body 31–98×). Its growth is a *gauge* (common-mode), fixed by **row-centering** (live on dn2 @18000+). The body's high WD-share is a separate, benign-in-direction phenomenon.
- **Body correction, if any, = annealed RENORM** (function-safe, loss-null direction). **NOT more WD** (dn1 ramped head-WD to 0.1 to fight growth and died).

---

## Methodology

**WD is DECOUPLED** (confirmed `muon_fsdp2`: `p.mul_(1 - eff_lr*wd)`, `lambda=0.02`), applied inside `optimizer.step()`, **after** grad-clip. So:
- WD gradient magnitude is exactly `||lambda*W|| = lambda*||W||_F` (no backward needed) and is purely **radial** (∥ W).

**Part A** (nearly free, full-run): per-layer `w_norm` + `g_norm` are already in `diagnostics.jsonl`; `g_norm` is the LOSS gradient (captured post-backward, **pre-WD**). `wd_grad_share = lambda*||W|| / (lambda*||W|| + g_norm)`.

**Part B** (the cos test): `cos(g_loss, W) = <g_loss,W> / (||g_loss|| ||W||)` from **CE-only** gradients (plain `F.cross_entropy` on eval-branch logits; no z-loss, no aux weighting, no WD). Read-only fwd+bwd, no optimizer step, no mutation. `nbatch 2 seq 1024` (dn2-18000: `nbatch1 seq512` due to aux-head OOM). Sharded across rig-31. Also: `wd_over_loss = lambda*||W|| / ||g_loss||`; `wasted_wd_frac = sqrt(1-cos^2)` (the WD vector is ∥ W, so its wasted fraction == this).
Classes: `body_proj_prenorm`={wo,w2} (output → Post-LN directly); `body_in_prenorm`={wq,wk,wv,w1,w3} (reads normed input); `head`={output.weight}; `embedding`.

### The invariance (why cos≈0 is expected for body)
Pre-norm matrices feed RMSNorm: `L(cW) = L(W)` ⟹ `d/dc L(cW)|_{c=1} = <g_loss, W> = 0` ⟹ `g_loss ⟂ W` (the radial direction). WD's `-lambda*W` is entirely radial ⟹ **loss-null**.
The **head is different**: it feeds softmax (shift-invariance / common-mode), not a norm. Its null direction is the row-mean `mu`, **not** scale — so the cos-vs-W test does **not** capture the head gauge (that's the row-center probe's job).

---

## Part B raw — cos(g_loss, W) per class

Format: `cosmean | |cos| [p50 p90 p99 max] | wd/loss [min med max]`

### dn1-10k HEALTHY (step 10000)
| class | n | cosmean | \|cos\| p50 | p90 | p99 | max | wd/loss min/med/max |
|---|---|---|---|---|---|---|---|
| body_proj | 138 | +0.00038 | .00047 | .00162 | .00303 | .00569 | 4.7 / 95.9 / 763 |
| body_in | 345 | +0.00032 | .00007 | .00136 | .00863 | .01957 | 6.4 / 80.4 / 10527 |
| head | 1 | −0.00001 | .00001 | | | | 4.9 |
| embed | 1 | −0.00000 | | | | | 60.9 |

### dn1-14k DEAD (step 14000)
| class | n | cosmean | \|cos\| p50 | p90 | p99 | max | wd/loss min/med/max |
|---|---|---|---|---|---|---|---|
| body_proj | 138 | +0.00028 | .00038 | .00171 | .00329 | .00571 | 3.2 / 98.0 / 958 |
| body_in | 345 | +0.00020 | .00007 | .00104 | .00507 | .00983 | 11.8 / 79.3 / 14876 |
| head | 1 | −0.00008 | .00008 | | | | 4.4 |
| embed | 1 | −0.00001 | | | | | 90.5 |

### dn2-18000 LIVE (step 18000, nbatch1/seq512)
| class | n | cosmean | \|cos\| p50 | p90 | p99 | max | wd/loss min/med/max |
|---|---|---|---|---|---|---|---|
| body_proj | 138 | +0.00049 | .00051 | .00187 | .00502 | .00612 | 1.7 / 37.7 / 330 |
| body_in | 345 | +0.00045 | .00011 | .00125 | .01115 | .02503 | 0.5 / 31.0 / 4350 |
| head | 1 | −0.00011 | .00011 | | | | 1.4 |
| embed | 1 | −0.00007 | | | | | 13.1 |

### mf-35k CONTROL (step 35000) — dim 1344 (vs 2560), n_heads 14, inner 2240, rope_theta 100k — DIFFERENT ARCH, low-LR
| class | n | cosmean | \|cos\| p50 | p90 | p99 | max | wd/loss min/med/max |
|---|---|---|---|---|---|---|---|
| body_proj | 140 | +0.00043 | .00079 | .00216 | .00384 | .00534 | 1.3 / 49.6 / 123 |
| body_in | 350 | +0.00018 | .00015 | .00143 | .00362 | .00682 | 1.7 / 36.9 / 135 |
| head | 1 | −0.00007 | .00007 | | | | 3.8 |
| embed | 1 | −0.00034 | | | | | 39.5 |

**Read:** body `|cos| ≈ 0` to the 99th percentile (worst body matrix anywhere = 0.025, i.e. `wasted_wd_frac` 0.9997) across **all four** checkpoints incl. the dead one and a different architecture. HEALTHY == DEAD == LIVE == CONTROL. Most loss-coupled matrices are always early/late `attention.wv` — still cos < 0.025. ⟹ architectural (RMSNorm scale-invariance), not a dreadnought quirk, not a death signature.

**Head vs body (loss-relevance signature):** head `wd/loss` = 1.4–4.9× vs body 31–98× — a ~10–20× gap. The loss cares ~order-of-magnitude more (relative to WD) about the head than body scale. dn2-18000 head `wd/loss=1.4×` is the lowest — at that step the loss-grad on the head nearly equals the WD pull.

---

## Part A raw — ||W|| ramping + wd_share over the run

**Summed `wd_share`** (`lambda*||W||` as fraction of total weight motion): dn2 92% (s100) → **99.8%** (s18000); same shape on dn1. By mid-training summed loss-grad ~2–4 while summed `lambda*||W||` ~2200.

**`||W||` recent slope** (%/kstep of current `||W||`) — ALL body matrices still climbing (0 flat):
| run | body median | example | head |
|---|---|---|---|
| dn2 (live) | +2.24%/kstep | attn TRIPLED: L49 207.8→642.0 (+209%) | +4.63% |
| dn1 (dead) | +2.88%/kstep | | +5.14% |
| mf (low-LR) | +0.91%/kstep (~2.5× slower) | | +4.62% |

⟹ **Body ramp is LR-COUPLED** (mf's lower LR → 2.5× slower). **Head ramp is LR-INDEPENDENT** (mf head +4.62% ≈ dn2 +4.63% despite 2.5× slower body) ⟹ head growth is **gauge accumulation**, a distinct mechanism from body norm-growth.

---

## Clip mechanics (code-confirmed)

`_clip_grad_norm_mixed_mesh` computes the global norm over `p.grad` **only** (loss gradient). WD is applied later in `optimizer.step()` via `p.mul_(1-eff_lr*wd)`, so WD is **never** in the clipped vector. The clip sees loss-grad-only, which Part A shows is **flat ~2–4** regardless of `||W||`. ⟹ **the body ramp does NOT directly trigger norm clipping.**

---

## Nuance: loss-null ≠ harmless

The ramp is blind to the **loss** but costs **dynamics**:
1. **Effective-LR decay:** `||dW||/||W||` shrinks as `||W||` inflates → learning slows in the loss-relevant (tangential) direction over time.
2. **Decay term dominance:** in each body weight's Adam update `m/sqrt(v) - eta*lambda*W`, the decay term grows with `||W||`; at `wd/loss` 31–98× it dominates the learning term on body matrices.
3. **Numerical headroom:** bigger pre-norm activations.

NOT via clipping (decoupled, see above). So: **worth correcting**, but the harm channel is optimizer-dynamics / effective-LR, not clip-triggering.

---

## Implications

- **Body correction (if we act):** a **RENORM** (rescale pre-norm matrices toward a target `||W||`) lives in the **locally** loss-null scale direction. **IMPORTANT CAVEAT (Chatty review):** `cos(g_loss,W)≈0` proves the *infinitesimal radial* direction is loss-null; it does **NOT** prove `L(cW)=L(W)` for a large finite `c`. The head case is *exactly* function-preserving (the `z_i'=z_i-h·μ` identity is algebraic, holds at any magnitude); the body case is an *empirical local orthogonality* that finite rescale could break via RMSNorm epsilon, KEEL residual/highway ratios, gated-MLP (`w1`/`w3`) coupling, or QK-norm placement. So: **finite-rescale invariance is an OPEN EMPIRICAL QUESTION (see Finite-Rescale Probe below), not established.** If it holds for moderate `c`, renorm is a gentle-control tool with a known safe range; if not, it's off the table. Either way: function-safe ≠ optimizer-state-safe (the head lesson — the hard switch shocked Adam at 18500), so **ANNEAL** any renorm and handle optimizer state, and prefer a **next-run-from-step-0 gauge** over mid-run surgery.
- **Do NOT use more WD to fight it:** dn1 ramped head-WD to 0.1 to fight head growth and **died** (coherency collapse). WD is a blunt radial pull that interacts badly with the optimizer/gauge; it is not a clean null-space correction. Renorm is; WD is not.
- **Head:** the gauge fix is **row-centering** (live on dn2 @18000+), correct and sufficient for the head.

---

## Open questions for the math agents

1. Is periodic renorm the right correction, or does the loss-null property mean we should just **accept** the ramp (does it self-limit via effective-LR decay → a slow equilibrium)? I.e. is the ramp asymptotically bounded or truly unbounded?
2. If renorm: what **target `||W||`** per matrix-class, and what cadence/anneal schedule to stay optimizer-safe?
3. **Interaction with NorMuon:** Muon/NorMuon already normalizes the *update* (Newton-Schulz / RMS). How does an external weight renorm compose with that? Does the update-normalization already partially counteract the ramp, or is it orthogonal?
4. Why does the ramp **not self-arrest** given effective-LR decay should slow growth? (recent slope still ~peak at 18k — not decelerating.) Is `lambda=0.02` simply too weak at these `||W||` scales, or is something feeding the radial drift?
5. The **body_in vs body_proj split:** body_in (reads normed input) has slightly higher tail `|cos|` (p99 up to 0.011–0.025) than body_proj (output→Post-LN, p99 ~0.003–0.005). Is body_in's weaker scale-invariance (input side, epsilon / partial-dependence) the reason, and does it matter?

---

## Pending tests (Chatty math review, 2026-06-22)

The cos test measured the **gradient** `g_loss`; under NorMuon the **actual step** is the
orthogonalized/normalized update, not the gradient — so two follow-ups:

1. **Finite-rescale invariance probe (DECISION-MAKER, forward-only, cheap):** per class
   `{wo, w2, wq, wk, wv, w1, w3}`, rescale that class by `c ∈ {0.95,0.9,0.8,0.7}` and
   measure `ΔCE, Δlogp_y, Δlogits, Δhidden-RMS` on a fixed batch. If CE is allclose at
   `c=0.9` but not `c=0.7`, renorm is a **gentle-control tool with a known safe range**,
   not a hard gauge projection. If even `0.95×` moves CE, renorm is off the table.
   `body_proj`={wo,w2} expected safest (cleaner cos); `body_in` (esp. gated `w1`/`w3`)
   more cautious — may need PAIRED scaling (scaling `w1` alone ≠ scaling the branch).
2. **Actual-update decomposition (mechanism):** capture the NorMuon update `ΔW_Muon`
   (pre-decoupled-WD) and report `cos(ΔW_Muon, W)`, radial/tangential fractions, and
   `<ΔW_Muon, W>`. If the normalized update has a small radial component, it FEEDS the
   ramp even when raw CE grads are radial-null — the gradient-based cos can't see this.

Equilibrium note (Chatty): `||W_{t+1}||² ≈ (1-ηλ)²||W_t||² + ||U_t||²` with `U_t ⟂ W_t`;
equilibrium at `||U||² ≈ 2ηλ||W||²`. With `ηλ ≈ 2.96e-4·0.02 ≈ 5.9e-6` and body slope
~+2.24%/kstep, dn2 body may still be well below balance — "probably bounded, possibly at
an uncomfortably high scale (~2× current?)", NOT proven unbounded. But NorMuon's
fixed-magnitude tangential injection may push that equilibrium far out.

Optimizer-state under a scale transform `W←cW` (if renorm is ever applied): exact-scale-
invariance implies `g←g/c`, so a consistent Adam transform would be `m←m/c, v←v/c²`;
Muon momentum `M←M/c` but Muon normalizes the update so the exact consequence needs an
empirical test. Do NOT implement renorm as "just scale the weights."

Sequencing: this is a TELEMETRY + BRANCH item, NOT for the current live transition.
Live priority: finish row-centering → (optional) SCS cleanup → later head-LR dampener.
Body renorm, if green-lit, is best as a **next-run step-0 gauge** (start in the chosen
gauge) or a slow log-space-annealed norm ceiling on a branch, never bolted onto the
current run.

## Finite-rescale invariance — RESULT (2026-06-22): RENORM IS **NOT** FUNCTION-SAFE

The decision-maker ran (`tools/finite_rescale_probe.py`, forward-only). **Verdict:
finite body rescale is NOT loss-invariant — Chatty's caution was empirically
correct. `cos(g_loss,W)≈0` (infinitesimal radial loss-null) does NOT extend to
finite `c`.** Post-hoc renorm is OFF the table as a "free" gauge correction.

ΔCE from rescaling each class by `c` (per-class matrices ×c, re-forward, restore):

### mf-35k (baseline CE 2.609)
| class | c=0.95 | c=0.9 | c=0.8 | c=0.7 |
|---|---|---|---|---|
| wo | 0.039 | 0.047 | 0.018 | 0.112 |
| w2 | 0.0022 | 0.018 | 0.103 | 0.333 |
| wq | 0.0009 | 0.018 | 0.004 | 0.012 |
| wk | 0.0026 | 0.005 | 0.009 | 0.003 |
| wv | 0.015 | 0.043 | 0.013 | 0.112 |
| w1 | 0.009 | 0.032 | 0.179 | 0.507 |
| w3 | 0.014 | 0.015 | 0.120 | 0.323 |
| head | 0.008 | 0.022 | 0.107 | 0.057 |
| **w1w3 paired** | 0.020 | 0.141 | 0.600 | **2.21** |

### dn1-14k DEAD (baseline CE 2.915)
| class | c=0.95 | c=0.9 | c=0.8 | c=0.7 |
|---|---|---|---|---|
| wo | 0.025 | 0.029 | 0.008 | 0.061 |
| w2 | 0.029 | 0.029 | 0.169 | 0.463 |
| wq | 0.026 | 0.031 | 0.012 | 0.013 |
| wk | 0.016 | 0.008 | 0.028 | 0.012 |
| wv | 0.011 | 0.050 | 0.004 | 0.067 |
| w1 | 0.013 | 0.003 | 0.231 | 0.911 |
| w3 | 0.029 | 0.014 | 0.188 | 0.528 |
| head | 0.032 | 0.006 | 0.096 | 0.361 |
| **w1w3 paired** | 0.026 | 0.160 | 0.823 | **2.06** |

dn1-14k SAFE RANGE: **NOT safe even at c=0.95 for ANY class** (dCE_rel ≥ 1e-3 everywhere).
mf SAFE RANGE: only wq/wk/w2 squeak under the 1e-3 bar at c=0.95; all break by c=0.9.
(dn2-18000 pending; expected to confirm.)

### What this means
- **NOT the head analog.** Head row-centering is *algebraically exact* at any
  magnitude (`z_i'=z_i-h·μ`); body rescale changes CE measurably even at 5%.
- **ΔCE is non-monotonic / jagged** (e.g. wo 0.039→0.047→0.018→0.112) — not the
  smooth signature of a true gauge; finite scale genuinely couples into the output
  via RMSNorm epsilon, KEEL residual/highway ratios, gated-MLP structure.
- **Gated-MLP w1w3 paired is catastrophic** (c=0.7 ΔCE ~2.1, ~70-85% loss increase) —
  SwiGLU's two legs compound ~quadratically; scaling one leg ≠ scaling the branch.
  Confirmed across both architectures.
- **The WD-waste finding (infinitesimal loss-null) STANDS; the proposed FIX (renorm)
  is DEAD.** You cannot project the body scale out the way you can the head gauge.

### Revised implication for the body ramp
If the body ramp is worth addressing (the effective-LR cost is real), it must be a
**TRAINING-DYNAMICS** approach, NOT post-hoc rescale:
- The NorMuon **actual-update decomposition** (pending test #2): is the normalized
  update itself feeding radial drift? That's the mechanism to target.
- A next-run **gauge/WD-schedule choice from step 0** (start in a controlled regime).
- NOT: post-hoc renorm (breaks the function), and NOT: more WD (killed dn1).

## Taming battery (2026-06-22) — mechanism test + an OPEN PUZZLE

Goal: find how to TAME the body norm ramp (Josef; restart-from-scratch acceptable;
"leave it alone" rejected). Built Test 1 (NorMuon actual-update decomposition) +
Test 2 (WD-equilibrium) + a ground-truth cross-check. Result: ruled out the simple
mechanisms and **falsified the toy equilibrium model** — the real growth force is
currently unaccounted-for. This is the live question for the Math Agent.

### Test 1 — NorMuon actual-update decomposition (tools/normuon_update_decomp.py)
Part B measured `cos(g_loss,W)` (gradient). NorMuon steps along the
Newton-Schulz-orthogonalized + neuron-normalized update, not the gradient. Replicated
the EXACT transform (`apply_momentum → zeropower_via_newtonschulz5 → apply_scaling →
apply_normuon`, real muon_fsdp2 funcs) and measured `cos(update, W)`. **mf-35k, exact
single-card, COLD momentum/2nd-moment buffers:**
| class | cos(grad,W) | cos(UPDATE,W) |
|---|---|---|
| body_proj | 0.0010 | 0.0019 |
| body_in | 0.0005 | 0.0014 |
| head* | 0.0001 | 0.117* |

NS orthogonalization DOES inject a radial component (~2-3× the gradient's), but on
the body it stays **negligible** (cos~0.002, 99.9998% tangential). *head 0.117 is a
PROBE ARTIFACT — the head is the Adam group, not Muon; the NS transform doesn't apply
to it in production. Body numbers are valid.

### Ground-truth cross-check (no model) vs the toy equilibrium model — CONTRADICTION
Toy model (Math Agent): `||W_{t+1}||² ≈ (1-ηλ)²||W_t||² + ||U_t||²`, equilibrium
`||W_eq|| = ||U||/√(2ηλ)`. With mf measured `||update||≈28.7`, `η≈3.3e-4`, `λ=0.02`:
- per-step tangential `||U|| = η·||update|| ≈ 0.0095`
- WD removal/step `= ηλ||W|| ≈ 0.0036` (DOWN, at ||W||≈540 block)
- tangential add/step `= ||U||²/(2||W||) ≈ 8e-8` (UP) — **negligible**
- ⟹ model predicts net `d||W||/step ≈ −0.0036` (should SHRINK)

**ACTUAL (mf L35 ffn block, diagnostics.jsonl, steps 35200→35600, zero modeling):**
`d||W|| = +0.0031/step (+0.57%/kstep)` — **GROWING.**

⟹ The toy model has the **WRONG SIGN** and is ~2× off. There is an unaccounted
**+~0.0067/step radial (+W) growth force** that none of our measured sources explain:
NOT the gradient (cos≈0), NOT NS radial injection (cos 0.002 → ~1.9e-5/step, 350× too
small), NOT tangential quadrature (8e-8). And it OVERWHELMS a WD removal two orders
larger than any visible injection term.

### Leading hypotheses (for the Math Agent)
1. **Warm buffers.** The probe used COLD momentum (β=0.95, ~20-step memory) and 2nd-
   moment buffers. In real training these are warm — the momentum buffer may carry a
   persistent radial component, and `apply_normuon`'s norm-preserving rescale (it forces
   `||update|| = ||pre-norm||` via the warm 2nd-moment `second_momentum`) could make the
   real per-step `||U||` much larger / differently-directed than the cold 28.7.
2. **`||U||` underestimated.** If warm `||U||` is ~10-30× the cold estimate, the
   tangential quadratic term becomes significant and equilibrium moves far out (could
   reconcile both the growth AND the "still ramping at 18k" observation).
3. **A genuine small persistent radial bias** in the warm update that single-step cold
   analysis can't resolve.

### Implication for taming (provisional)
- The "just turn up WD" dial (Test 2) is **not validated** — the toy model it rests on
  is falsified, so its equilibrium predictions (eq@1x≈2.6 while actual=107) are wrong.
  DO NOT size a WD change off that model until the real force balance is known.
- The right next test is **warm-buffer**: measure `cos(update,W)` and `||U||` with the
  momentum + 2nd-moment buffers RESTORED from the checkpoint's optimizer state (not
  cold), and/or decompose the ACTUAL `ΔW` between two adjacent real training steps.
- Renorm remains OFF (finite-rescale unsafe, separate result above). The taming lever,
  if any, is in the optimizer dynamics (warm-update radial component) — once we know
  what's actually adding the +0.0067/step, the fix targets THAT.

## Reproduce
```
# Part A (offline, log-parse, no GPU):
python tools/wd_waste_probe.py --diag <run>/diagnostics.jsonl --wd 0.02

# Part B (GPU; rig-31 sharded — note CUDA_DEVICE_ORDER=PCI_BUS_ID so indices match nvidia-smi):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4,5,6 \
  python tools/wd_waste_partb.py --ckpt <model_step_*.pt> \
  --groups "<comma,sep,groups>" --nbatch 2 --seq 1024 --wd 0.02 --shard balanced \
  --out partb_<tag>.json
```
