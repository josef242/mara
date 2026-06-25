# Stage A — bf16-reduce ANCHOR did not reproduce at T=8192

**Status:** SETBACK — anchor not re-established. fp32 comparison ON HOLD.
**Date:** 2026-06-23. **Run:** mf-low-lr (KEEL 70L, NorMuon, FSDP2, 8-GPU rig-31), resumed @35500.
**Probe:** `WD_REDUCE_PROBE=1` in `train_mara.py` (one-shot, dumps `wd_reduce.json`, exits pre-step).
**Configs:** anchor `configs/_probe_mf_bf16reduce_t2048.yaml` (bf16 reduce, **T=8192**), test `configs/_probe_mf_fp32reduce_t2048.yaml` (fp32 reduce, T=8192).

---

## What we expected vs what we got

The plan was a clean two-arm comparison at matched **T=8192** on the real 8-GPU sharded path:
- **bf16-reduce ANCHOR** — must reproduce the production lean (~ **-0.013** on the body matrices) to validate the regime.
- **fp32-reduce TEST** — `FSDP_reduce_dtype: fp32`, params still bf16; if the lean vanishes, the bf16 reduce-scatter is the mechanism.

**The anchor did NOT reproduce at T=8192.** `cos(g_REDUCED, W)` came back at **`<ANCHOR_COS>`** — not the ~ -0.0129 / 100%-negfrac signature seen in the production WD_INSITU_PROBE that opened this branch. (Attenuated / null, not the regime that produced the phenomenon.)

## What this does and does NOT mean

- It does **NOT** clear or implicate the bf16 reduce-scatter. The fp32 TEST arm is **MOOT** until the anchor is re-established — comparing two arms is meaningless when the control arm doesn't show the effect.
- The original -0.0129 was captured at the production regime **T=12288, GA~31** (the live ga_schedule). The T=8192 probe configs run a **different T and a different effective grad-accum count**. So the most likely culprits, in order:
  1. **Context / T dependence** — the lean is stronger (or only present) at the production T=12288 than at T=8192.
  2. **GA / effective-batch dependence** — the radial bias scales with the number of microbatches summed in the bf16 reduce accumulator; fewer microbatches at the probe's GA could attenuate it.
  3. **Resume-state dependence** — both arms resume @35500; if the -0.0129 was captured at a different step, optimizer/momentum/data-cursor state differs.

This is consistent with Stage B's offline finding that single-card T (1024 vs 12288) made **no** difference (~ +0.0001 both) — i.e. T-sensitivity, if real, is a property of the **distributed reduce path**, not the forward/CCE math. That would actually *strengthen* the reduce-scatter suspicion — but it is **unproven** until the anchor reads ~ -0.013 again.

## Next step (before ANY fp32 comparison)

Re-establish the anchor at the **original production regime** that gave -0.0129:
- **T=12288** (config already staged: `configs/_probe_mf_fp32reduce.yaml` is T=12288 — clone/flip its `FSDP_reduce_dtype` back to **bf16** for the anchor arm).
- Match the production **GA** at that T (live ga_schedule; ~31 at the current step band).
- Confirm `cos(g_REDUCED, W)` returns to ~ -0.013 with ~100% negfrac on the body matrices.

Only once the anchor is recovered does the bf16-vs-fp32 reduce comparison become valid. Until then: **regime/context dependence is the open question, not the reduce dtype.**

## Discipline note

We do not have a confirmed mechanism. The bf16 reduce-scatter remains the **prime suspect, unconfirmed**. This branch records a failed anchor, not a result. No mechanism claim ships off a probe whose own control did not fire.
