# Body-Norm Ramp — Brief #2 for the Math Agent (2026-06-23)

**Major update since brief #1: the residual is NOT precision. It is STRUCTURAL FSDP2 SHARDING.**
Compact, self-contained. A world=1 control is running now (result pending).

---

## The phenomenon (recap)
mf-low-lr (KEEL ultra-deep, 70L, NorMuon, FSDP2, 8×GPU). The real-training body gradient leans
anti-radial: `cos(g, W) = −0.0129` on ~100% of 490 body matrices (wq/wk/wv/wo, w1/w2/w3).
Descent (−g) flips it to a +radial applied ΔW → ‖W‖ grows. **Why it matters:** decoupled WD
removes `ηλ‖W‖`/step, which grows with ‖W‖; WD's share of the update climbed to **99.8%** by
mid-training (summed λ‖W‖ ~2200 vs loss-grad ~2–4) — real learning gets starved, runs have a
finite useful lifespan. So the ramp must be tamed.

## The decomposition (all on CLEAN, MEASURED in-situ data — cos = raw `.grad` vs W)

| condition | data | path / dtype | cos(g,W) median | negfrac |
|---|---|---|---|---|
| single-card offline | stories/ao3 panel | unsharded, bf16 | **+0.0001** | ~40% (random) |
| single-card, EXACT rank-0 tokens | rank-0 real stream | unsharded, bf16 | **+0.000003** | 42% (null) |
| 8-GPU FSDP, replicated data (all ranks=rank0) | rank-0 stream | sharded, bf16 | **−0.0075** | 100% |
| 8-GPU FSDP, diverse data (anchor) | 8 diff shards | sharded, bf16 | **−0.0129** | 100% |
| **8-GPU FSDP, FULL fp32** (params+forward+CCE) | 8 diff shards | sharded, **fp32** | **−0.0159** | 98% |

**Two additive components:**
1. **~40% cross-rank DATA diversity** (−0.0129 → −0.0075 when all ranks fed identical data).
   A real CE/data effect. (Replicated via `dist.broadcast` of rank0's batch to all ranks.)
2. **~60% the SHARDED PATH itself** (−0.0075 survives identical data; single-card on the SAME
   exact tokens = +0.000003 null). The only difference single-card↔FSDP is the execution path.

## What is RULED OUT (each by a clean control)
- **bf16 gradient reduce-scatter** — fp32 reduce gave identical −0.01288 (vs bf16 −0.01286).
- **fused-CCE kernel / bf16-CCE accumulation** — single-card train-CCE == eval-CE; forcing
  fp32 accum changed nothing.
- **long context** — single-card null at T=1024 AND T=12288; lean is T-insensitive.
- **activation-checkpoint recompute** — on, still null single-card.
- **the DATA (for the ~60% residual)** — exact rank-0 tokens, single-card = null.
- **bf16 PRECISION entirely (NEW, decisive)** — full fp32 params + fp32 forward + fp32-capable
  CCE (torch_compile impl) on the real 8-GPU path: lean PERSISTS at −0.0159, if anything
  STRONGER than bf16's −0.0129. So it is NOT the bf16 cast, NOT bf16 matmul, NOT bf16 values.

## THE FINDING: the ~60% residual is STRUCTURAL FSDP2 SHARDING, dtype-independent
Sharded (FSDP2, any dtype) → lean. Unsharded (single-card, same tokens) → null. The *act of
sharding* — all-gathering params per-layer for the forward and the FSDP-wrapped autograd —
produces a systematic anti-radial gradient component on ~100% of body matrices, in both bf16
and fp32.

**IMPORTANT CONFIG FACT: `reshard_after_forward: false`** (production AND all probe runs). So
FSDP2 all-gathers each layer's params ONCE, keeps them resident (full) through forward AND
backward — NO reshard between fwd/bwd, NO re-gather for the backward. This WEAKENS the
"reshard/re-gather re-rounds the param" class of explanation: the gathered param is reused, not
reconstructed twice. Remaining structural suspects: (a) the all-gather reconstruction itself
(is the gathered-from-shards full param subtly ≠ single-card's resident param, even gathered
once?); (b) the FSDP-wrapped AUTOGRAD GRAPH (backward flows through FSDP's gradient
machinery/hooks — even with reduce *dtype* innocent, the graph structure differs from
single-card); (c) activation-checkpoint recompute interacting with the gathered param. A
`reshard_after_forward: true` probe (forces reshard+re-gather between fwd/bwd) is planned to
test (a)/(reshard) directly.

## OPEN QUESTIONS FOR THE MATH AGENT
1. **What is a plausible MECHANISM?** How does sharding + per-layer all-gather/reshard +
   FSDP-wrapped autograd inject a *systematic* `⟨g,W⟩ < 0` (i.e. `dL/dlog‖W‖ < 0`) that is
   uniform in sign across all body matrices and depth, when the SAME forward unsharded is null?
   The all-gather *reconstructs the full param from shards before the matmul* — should be
   bit-exact (concatenation), so the forward output ought to match single-card. Where could a
   systematic gradient bias enter? (autograd through the all-gather/reshard graph? the
   recompute under activation checkpointing seeing a re-gathered param? gradient hooks?)
2. **Why is fp32 slightly STRONGER (−0.0159 vs −0.0129)?** Noise (single step), T (12288 vs
   8192), or does removing bf16 rounding *sharpen* an otherwise-present systematic lean?
3. **Does this look like a known FSDP/sharded-DP artifact** in your experience, or genuinely
   novel? If intrinsic to sharded data-parallel reconstruction, that's a broad claim.
4. **The 100% sign-consistency** keeps being the strongest clue. Reduce-scatter would have
   explained it (same op every tensor) but is ruled out. What sharding mechanism is similarly
   uniform across all matrices?
5. **Is the OLD branch-gain story (⟨g,W⟩<0 because CE wants more learned branch) compatible**
   with this being a sharding artifact, or do they exclude each other? I.e. is the −0.0129
   physics or sharding-numerics — and the ~40% data component muddies this (data IS physics).

## NEXT TESTS (in flight / planned)
- **RUNNING NOW: world=1 FSDP2** (full FSDP machinery, NO real sharding, bf16, matched config).
  NULL ⟹ needs real multi-shard (cross-rank all-gather) ⟹ FSDP1 is a sharp diagnostic.
  LEAN ⟹ it's the FSDP wrapper/recompute itself, not cross-rank gather.
- **FSDP1** (FlatParameter: all-gather one flat buffer, unflatten/view) reconstructs params
  differently than FSDP2 (per-param DTensor). If it leans differently → reconstruction is it.
- vary `reshard_after_forward` (currently false); gradient-hook audit of the FSDP path.

## STATUS / POLICY (Math Agent concurs)
No body-WD / no renorm / no head-WD until the source is pinned. The decomposition makes the fix
TARGETABLE: ~40% is learning (leave it, modest WD), ~60% is a structural sharding tax (free
growth — kill at the source once we know the mechanism). The fp32 "fix" is OFF the table (we
proved fp32 doesn't remove it). Canonical: docs/REPLICATED_DATA_RESULT.md, WD_WASTE_ANALYSIS.md.
