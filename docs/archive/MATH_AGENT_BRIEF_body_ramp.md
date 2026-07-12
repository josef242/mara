# Body-Norm Ramp — Brief for the Math Agent (2026-06-23)

**The phenomenon:** mf-low-lr (KEEL ultra-deep, 70L, NorMuon, FSDP2, 8×GPU) body weight
matrices (`wq/wk/wv/wo`, `w1/w2/w3` — all pre-RMSNorm) show a steady `‖W‖` RAMP. The
operative driver is a small, dead-consistent **anti-radial lean in the real training
gradient**: `cos(g, W) = −0.0129` on **100% of 490 body matrices**. Gradient descent (−g)
flips this to a **+radial** applied ΔW → ‖W‖ grows. The number is rock-solid: it reproduces
across T=1024→12288 and is the thing we've spent the day trying to mechanistically explain.

---

## What we have DEFINITIVELY RULED OUT (all on clean, measured runs)

We ran a faithful **offline single-card** probe and a series of **in-situ 8-GPU** controls.
The lean's value is `cos(g,W)`, body-matrix median, raw fresh `.grad` vs W (no WD subtraction):

| Test | Setup | cos(g,W) | Verdict |
|---|---|---|---|
| Offline single-card (Stage B) | 1 GPU, train-branch fused-CCE, **bf16**, T=1024→12288, act-ckpt ON, fp32-accum forced | **+0.0001 sign-random** | NO lean |
| In-situ anchor | 8-GPU FSDP2, **bf16** reduce-scatter, T=8192 | **−0.01286, 100% neg** | lean present |
| In-situ **fp32-reduce** | 8-GPU FSDP2, **fp32** reduce-scatter, T=2048 | **−0.01288, 100% neg** | lean UNCHANGED |

**Ruled out as the cause:**
1. **Fused-CCE loss kernel** — train-CCE == eval-CE on the same batch (single card).
2. **bf16 CCE accumulation** — forcing the fused kernel's internal accum to fp32 changed nothing.
3. **Long context** — T=1024 and T=12288 give the same single-card null.
4. **Activation-checkpoint recompute** — on, still null on single card.
5. **bf16 gradient reduce-scatter** — fp32-reduce gave an *identical* −0.01288. (This matters:
   there IS a documented FSDP2 bf16-reduce accumulation bug — PyTorch tutorial recommends
   `reduce_dtype=fp32`, torchtitan hardcodes it, issue #106395 — and the ">2-summand" rule
   superficially fit our single-vs-multi-card split. **Our fp32 control falsifies it as OUR
   cause.** The bug is real but it is not what grows our body norms.)

---

## THE TWO HARD FACTS

1. **Single card → NO drift** (+0.0001), even with bf16 CCE / long context / act-ckpt.
2. **Multiple cards → drift APPEARS** (−0.0129), even with **fp32** reduce-scatter.

So the lean is **PRE-reduction**: it exists in the gradient before/independent of the
cross-rank combination dtype. It lives somewhere in the **per-rank sharded computation**.

---

## WHAT WE ARE TESTING RIGHT NOW (WD_REDUCE_PROBE)

A one-shot in-situ probe on the real 8-GPU path that captures, on the same step:
- **`cos(g_reduced, W)`** — GLOBAL, the real reduce-scattered gradient (expect ≈ −0.013).
- **`cos(g_local, W)`** — PER-RANK, a local grad-accum run under
  `set_requires_gradient_sync(False)` so each rank's gradient is **NOT** reduced across ranks,
  and the cos is computed on **each rank's own shard with NO cross-rank aggregation**.

It reports every rank's local cos separately. **The fork:**

- **If every rank's LOCAL grad already leans −0.013** ⟹ the anomaly is intrinsic to a *single
  rank's* sharded gradient computation — NOT the cross-rank combine. This is the unsettling
  branch, because the **single-card offline probe does the same "one GPU, one backward" and
  got +0.0001.** That paradox would force the cause into the short list of things that differ
  between "an FSDP rank" and "a single-card model":
    1. **All-gathered bf16 params** (prime suspect): FSDP shards weights and all-gathers them
       to bf16 just-in-time for each forward; if that bf16-weight forward differs from the
       offline probe's forward, the gradient differs. This would make the lean a **numerical
       artifact of bf16 PARAMETER precision in the forward** — a far bigger fish than the
       reduce-scatter (it's not a one-line config fix; it implicates bf16 weights themselves).
    2. **Data shard** — each rank trains on a different slice of the real streaming data; the
       offline probe used a fixed stories/ao3 panel. If the lean is data-distribution-driven,
       it's not numerics at all.
    3. The FSDP-wrapped autograd / act-ckpt recompute path itself.

- **If LOCAL ≈ 0 but REDUCED ≈ −0.013** ⟹ the lean emerges ONLY from summing 8 partial
  gradients over 8 *different* data shards — i.e. it's a property of the cross-rank
  combination of different-data partials (even in fp32), not of any single rank.

**Next test either way:** make the offline single-card probe replicate a real rank EXACTLY
(same bf16 all-gathered weights, same autocast, the exact data batch that rank saw) and see if
it then reproduces −0.013. That isolates bf16-param-forward (#1) vs data (#2) vs FSDP-autograd.

---

## QUESTIONS FOR THE MATH AGENT

1. **Mechanism intuition:** does a small **systematic** anti-radial gradient component
   (`⟨g,W⟩ < 0`, i.e. `dL/d log‖W‖ < 0`, 100% of matrices) have a natural explanation in a
   bf16-parameter forward through RMSNorm/KEEL highways? E.g. does bf16 *rounding of the
   weights* (not the grads) bias the effective `⟨g,W⟩` negative in a scale-invariant net?
2. **Why 100% consistent?** Whatever the source, it hits every body matrix with the same sign.
   Reduce-scatter would have explained that (same op on every tensor) but it's ruled out. What
   per-rank mechanism is similarly uniform across all matrices?
3. **Does the single-vs-multi split survive your scrutiny?** Single card = exactly +0.0001;
   8-GPU = exactly −0.0129. If it's bf16 all-gathered params, single-card-in-bf16 *should*
   show it too — unless the offline probe's param/forward path differs from FSDP's all-gather
   in a way that matters. Is there a precision/accumulation asymmetry you'd predict there?
4. **Sanity on the prior physics:** the earlier "branch-gain knob" / "CE prefers more branch"
   story (⟨g,W⟩<0 because increasing body scale raises branch influence) was a *real-gradient*
   explanation. Is that compatible with the lean being a per-rank numerical artifact, or does
   one exclude the other? (I.e. is −0.0129 physics or numerics — and how would you tell?)

---

**Policy / status:** body-WD lever sizing (λ_pin ≈ 0.03–0.037) still rests on −0.0129 being a
real operative-gradient property — which is now solid (survives fp32-reduce, reproduces across
T). But if WD_REDUCE_PROBE shows it's a per-rank **numerical** artifact (bf16 params), the
right fix may be a precision change, not a WD dial. DO-NOTs hold (no head-WD, no renorm, no
body-WD change) until the source is pinned. Canonical record:
`mara_fsdp2/docs/WD_WASTE_ANALYSIS.md` "STAGE A" section.
