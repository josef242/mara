# Replicated-Data Test — RESULT (2026-06-23, autonomous run)

## ⚠️ SUPERSEDED — the "structural FSDP2 sharding" conclusion below is WRONG
**Root cause found later (see WD_WASTE_ANALYSIS.md top banner + newton_schulz_radial_finding):**
the `cos_grad_W` probe read `p.grad` AFTER `optimizer.step()`, where Muon had overwritten it with
the Newton-Schulz UPDATE. So every leaning number here (−0.0075 replicated, −0.0129 diverse, the
"structural sharding" finding) is the **post-NS update**, not the raw gradient. The raw CE gradient
is radial-null (Part B was right). The mechanism is **Newton-Schulz spectral flattening**, NOT
sharding/data/precision. The "single-card null" baselines were null because the offline probe runs
NO Muon step (no NS). The ~40% "data diversity" split is also post-NS-of-different-data, not raw-CE
physics. **Fix:** tangent-project the Muon update (validated, Probe 1). This doc is kept as
investigation history; read it knowing the lean it chases is the NS update.

---

**Math Agent test #3 (data-vs-machinery cut). Run autonomously overnight with Josef's
go-ahead on the data hypothesis. NOTHING committed — staged for review.**

## WHY THIS IS WORTH FIXING (motivation, Josef 2026-06-23)
The body-norm ramp is NOT cosmetic. It is **self-amplifying and starves learning**: decoupled
WD removes `ηλ‖W‖` per step, which GROWS as ‖W‖ grows, so WD's share of the update climbed to
**99.8%** by mid-training (summed λ‖W‖ ~2200 vs loss-grad ~2–4). The optimizer ends up spending
almost all its motion fighting its own weight decay; real learning gets crowded out. A run on
this trajectory has a **finite useful lifespan** — it doesn't crash, it suffocates. So the ramp
must be tamed. The decomposition below makes the fix TARGETABLE: ~40% is genuine CE/data
pressure (leave it — that's learning), ~60% is the FSDP sharded-path artifact (**free growth
that buys nothing — pure tax**). Goal: kill the ~60% artifact at its source WITHOUT over-decaying
the ~40% real signal. (The pending fp32-param test says WHAT the artifact is → how to remove it.)

## Setup
New env flag `WD_REPLICATED_DATA=1` (in `train_mara.py`, grad-accum loop): broadcasts
rank 0's microbatch to ALL 8 ranks via `dist.broadcast`, so every rank computes its
gradient over **identical data**. Combined with `WD_INSITU_PROBE=1` (the proven probe that
produced the canonical −0.0129). Memory-safe single-pass, T=8192, bf16, real 8-GPU FSDP2
path, warm mf-low-lr step 35500. Raw: `ckpt/wd_insitu_mf_REPLICATED_t8192.json`.

## Result — a TWO-COMPONENT split (both real, both 100% negative)

| condition | cos(g,W) body median | negfrac | data | path |
|---|---|---|---|---|
| single-card offline (Stage B) | **+0.0001** | ~40% (random) | stories/ao3 panel, T=1024 | 1 GPU, full model |
| **8-GPU REPLICATED** (all ranks=rank0) | **−0.00750** | **100%** | rank0 real stream, T=8192 | FSDP all-gather bf16 |
| 8-GPU DIVERSE (anchor) | **−0.01286** | **100%** | 8 different shards | FSDP all-gather bf16 |

Per-class (replicated): body_proj −0.00869, body_in −0.00741 (both ~halved vs diverse,
proportionally). Per-depth (replicated): early −0.0085 / mid −0.0068 / late −0.0081 —
**uniform across depth**, no structure.

## Interpretation — the −0.0129 lean has TWO additive contributions

1. **Cross-rank DATA DIVERSITY ≈ 40% of the effect** (−0.0129 → −0.0075 when data made
   identical). Averaging gradients over 8 *different* data streams adds ~0.005 of
   anti-radial lean vs 8 copies of one stream. **Math Agent's #1 suspect (data) is
   PARTIALLY CONFIRMED** — but it is NOT the whole story.

2. **A RESIDUAL ≈ −0.0075 (the bigger ~60%) that survives identical data**, 100% negative,
   uniform across class and depth. This is present even when every rank sees the same
   tokens — so it is NOT cross-rank data composition. It lives in the **FSDP machinery /
   single-rank bf16 forward path** OR in the **real-training-data-stream vs offline-panel
   difference** (see confound below). Its uniformity (every layer, every class, same sign)
   points to a GLOBAL systematic effect, consistent with the Math Agent's "branch-ratio
   distortion in the bf16 forward" idea rather than anything structural.

## The remaining CONFOUND (important — do not over-conclude)
The single-card +0.0001 and the replicated −0.0075 differ in MORE than the FSDP path:
- **Data:** offline used a fixed stories/ao3 panel; replicated used rank0's real training
  stream. BUT — the offline probe resolves groups from the mf config = stories/ao3, the
  SAME dominant groups. So the data DISTRIBUTION is ~matched (same groups), though not the
  exact tokens.
- **Sequence length:** offline T=1024 vs replicated T=8192. (Stage B showed the lean is
  T-insensitive on single-card — stayed +0.0001 at T=1024→12288 — so T is unlikely to
  explain a swing to −0.0075, but it is not perfectly controlled.)
- **Forward path:** single-card full-model vs FSDP sharded/all-gathered-bf16. THIS is the
  prime suspect for the residual.

## EXACT-TOKEN REPLAY — DONE (Math Agent #1). RESIDUAL = FSDP/bf16 PATH, **NOT** data.

Captured rank-0's exact tokens (`WD_DUMP_TOKENS=1`, raw binary, OPAQUE — never decoded) and
replayed them through the **single-card** offline train-CCE forward at the same T=8192
(`keel_radial_probe --mode stageb --tokens-file`). Result:

| condition | data | path | cos(g,W) |
|---|---|---|---|
| offline panel | stories/ao3 panel | single-card | +0.0001 |
| **EXACT rank-0 tokens** | **real rank-0 stream** | **single-card** | **+0.000003 (42% neg, NULL)** |
| 8-GPU replicated | rank-0 stream (all ranks) | FSDP all-gather bf16 | −0.0075 |
| 8-GPU diverse | 8 different shards | FSDP all-gather bf16 | −0.0129 |

**VERDICT: the −0.0075 residual is NOT the data.** The IDENTICAL tokens give +0.000003 (null)
on single-card but −0.0075 on the FSDP path. The only difference is the **forward execution
path** — FSDP's sharded, all-gathered-**bf16** matmuls. So the residual is the **FSDP/bf16
MACHINERY**, confirming the Math Agent's #2 suspect ("branch-ratio distortion in the bf16
forward"). His #1 (data) is real but only the ~40% cross-rank-diversity component.

**Caveat (honest):** the replay is single-microbatch (1×8192) vs the 46-mb replicated
accumulation, so an exact magnitude match isn't expected — but the SIGN is decisive
(+0.000003 vs −0.0075 is a clean null-vs-lean split, robust to microbatch count). Stage B
also showed single-card is T-insensitive, so T isn't confounding.

## FINAL DECOMPOSITION of the −0.0129 lean
1. **~40% cross-rank DATA diversity** (−0.0129 → −0.0075 when data made identical). REAL CE/data effect.
2. **~60% FSDP/bf16 forward-PATH** (−0.0075 → +0.000003 when same tokens run single-card). NUMERICAL/path effect — the all-gathered bf16 parameter forward, NOT the data, NOT the reduce-scatter (ruled out earlier), NOT the CCE kernel (ruled out Stage B).

## fp32-PARAM TEST — DONE (rig-30, 8×3090). RESIDUAL IS **STRUCTURAL SHARDING**, NOT bf16.
Ran the whole model in **full fp32** on the real 8-GPU FSDP2 path (`FSDP_param_dtype: fp32`,
fp32 forward, `WD_CCE_IMPL=torch_compile` to bypass the fused-CCE bf16-only kernel — Josef's
unblock), T=12288, in-situ probe. Result:

| condition | cos_grad_W body median | negfrac | dtype in FSDP forward |
|---|---|---|---|
| single-card (exact tokens) | +0.000003 | 42% (null) | — (unsharded) |
| FSDP bf16 | −0.0129 | 100% | bf16 |
| **FSDP fp32-param** | **−0.01592** | **98%** | **fp32 (NO bf16 anywhere)** |

Per-class: body_proj −0.0176, body_in −0.0143. `total_dW_radial` +0.00082, 96% pos — **‖W‖
still grows in full fp32.**

**VERDICT: removing bf16 ENTIRELY from the FSDP forward did NOT kill the lean — it persists at
full strength (−0.0159, if anything stronger than bf16's −0.0129).** So the ~60% "residual" is
**NOT a precision artifact** (not bf16 cast, not bf16 matmul, not bf16 values — all ruled out
across this + the single-card replay + the reduce-scatter test). It is **STRUCTURAL FSDP2
SHARDING**: the act of sharding/all-gathering/resharding the params produces the anti-radial
gradient lean, independent of dtype. (Single-card unsharded = null; sharded = lean, in BOTH
bf16 and fp32.) Josef's "sharded vs unsharded" instinct was correct as the WHOLE story for the
non-data component.

**Why fp32 is slightly STRONGER (−0.0159 vs −0.0129):** unconfirmed. Candidates: (a) noise
(single step); (b) T=12288 vs 8192; (c) fp32 removes rounding noise that was blurring the
systematic lean, so it reads sharper. Not over-read yet.

## REVISED DECOMPOSITION of the −0.0129 lean
1. ~40% cross-rank DATA diversity (real CE effect)
2. ~60% **STRUCTURAL FSDP2 sharding** (dtype-independent — all-gather reconstruction /
   reshard-recompute / sharded autograd). NOT precision, NOT data, NOT reduce-scatter, NOT CCE.

## NEXT — FSDP1 (now the real test, not hypothetical)
FSDP1 (FlatParameter: all-gather one flat buffer, unflatten/view slices) reconstructs params
DIFFERENTLY than FSDP2 (per-param DTensor all-gather). If the lean is "how the gathered tensor
is reconstructed," FSDP1 vs FSDP2 could differ → diagnostic. New run, larger surface (FSDP1
wrapping, ckpt format, optimizer sharding), but on the table (Josef). If FSDP1 ALSO leans →
the effect is intrinsic to sharded data-parallel reconstruction itself (deep water).
Cheaper intermediate probes to consider first: vary `reshard_after_forward`, or a 1-GPU FSDP2
run (world=1: full FSDP machinery, no real sharding) — isolates "FSDP wrapper" from "multi-shard".

## Status / policy (unchanged)
- The −0.0129 lean is REAL and now PARTIALLY DECOMPOSED: ~40% cross-rank data diversity,
  ~60% residual (machinery/bf16-path or real-stream data — confound not yet split).
- DO-NOTs hold: no body-WD, no renorm, no head-WD until the residual source is pinned.
  (Math Agent concurs.) The two-component nature matters: even if the data component is
  "real CE physics," the machinery component (if that's what the residual is) would be a
  numerical artifact — and the body-WD lever only makes sense against the physics part.
- Files (all UNCOMMITTED): `train_mara.py` (WD_REPLICATED_DATA flag + earlier WD_REDUCE_PROBE
  + Stage A in-situ additions), `docs/WD_WASTE_ANALYSIS.md`, `docs/MATH_AGENT_BRIEF_body_ramp.md`,
  this file. Probe configs `configs/_probe_mf_*`. Raw JSONs in `ckpt/wd_insitu_mf_*`.
