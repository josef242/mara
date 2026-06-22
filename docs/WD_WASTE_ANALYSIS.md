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

- **Body correction (if we act):** a **RENORM** (rescale pre-norm matrices toward a target `||W||`) is **function-safe** — it lives in the loss-null scale direction, so model outputs are invariant (the body analog of head row-centering being function-preserving). It reclaims effective-LR + numerical headroom. BUT: function-safe ≠ optimizer-state-safe (the head lesson — the hard switch shocked Adam at 18500). So **ANNEAL** any renorm, don't apply cold.
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
