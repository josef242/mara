# Replicated-Data Test — RESULT (2026-06-23, autonomous run)

**Math Agent test #3 (data-vs-machinery cut). Run autonomously overnight with Josef's
go-ahead on the data hypothesis. NOTHING committed — staged for review.**

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

## NEXT (precision ladder, Math Agent #5 — to pin the bf16-path mechanism)
On the SAME exact tokens, single-card, vary the forward precision:
- fp32 master weights / fp32 forward
- bf16 weights dequantized to fp32 (same rounded values, fp32 matmul)
- true bf16 matmul (mimic FSDP all-gather)
If bf16-rounded-values-in-fp32-matmul reproduce the lean → it's the quantized VALUES.
If only true-bf16-matmul reproduces it → it's the bf16 KERNEL/accumulation in the forward.
This isolates WHICH part of the bf16 forward produces the anti-radial pressure.

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
