# z-loss on Cut Cross-Entropy: gradient-equivalence problem set

**Status:** z-loss feature implemented and merged (default OFF, disabled path byte-identical and safe). A rig-only validation uncovered that the *enabled* path does not work as written on the installed CCE version. The gradient discrepancy is now **fully diagnosed and resolved** — see **§11 (RESOLUTION)** at the bottom for the answer and the open decision. The body (§1–10) is the investigation that led there, preserved for context.

> **TL;DR of the resolution (read §11 for detail):** "Option D" (reconstruct logZ = CCE's `reduction='none'` CE + a plain-torch target logit) is **mathematically correct** — its gradient matches the fp32 analytic truth to ~1e-5. The ~12% gradient discrepancy we chased is **benign bf16 precision loss** from catastrophic cancellation in `logZ = ce_none + logit_target` (CCE computes `ce_none = logZ − logit_target` inside its kernel, where both terms are ~O(8) and the difference is small; the lost low-order bits can't be recovered). It is **NOT a bug** and **NOT the OLMo broken-kernel failure mode**. The damage is baked into CCE's `ce_none` inside the kernel, so it **cannot be fixed from the trainer side** without materializing `[N,V]` logits (which violates the no-VRAM constraint). The remaining decision is purely: *ship option D with a ~0.99-cosine gradient (fine for a small annealed regularizer), or invest in a custom autograd Function for an exact gradient?*

**Author:** Code (codebase specialist). **Date:** 2026-06-14. **Rig:** rig-30, RTX 3090, torch 2.9.1+cu128, `cut_cross_entropy` **25.4.3**. **All §11 numbers are from the real CCE kernel on rig-30.**

---

## 1. What z-loss is and why we want it

z-loss (log-partition regularization) adds a penalty on the softmax log-partition to the LM objective:

```
L_total = L_CE + alpha * mean(logZ^2)
```

where `logZ = logsumexp(logits)` over the vocab dimension (per token). It tethers the **absolute magnitude** of the logit vector without penalizing logit **differences** — counteracting the unbounded-softmax pressure that grows the output head's dominant singular value / grad norm over long training. Standard technique (ST-MoE, PaLM, Chinchilla/Gopher). We want it as an opt-in, config-gated, default-OFF feature with an optional resume-safe alpha-annealing schedule. Target use: inject into an in-progress run (e.g. `dreadnought_v2`, ~step 17.6k of 200k) without a transient.

The headline requirement: **pure-CE loss/ppl must stay unchanged** for historical comparability; z-loss is added to the optimized objective and logged separately.

---

## 2. The architectural constraint that drives everything

In `mara_fsdp2`, **cross-entropy is computed inside the model forward** (`common_fsdp2/model_v2.py`), not in the trainer, via **Cut Cross-Entropy (CCE)** — Apple's fused linear+CE Triton kernel (`cut_cross_entropy.linear_cross_entropy`, wrapped as `cce_loss`). On the Linux/CUDA rigs CCE **fuses the head projection and never materializes the `[N, V]` logits** (its entire purpose: ~24 GB → ~1 MB for the head). On Windows it falls back to `hidden @ w.T` + `F.cross_entropy` (CCE disabled via `os.name == 'nt'`).

So the logZ that z-loss needs is computed *inside* the fused kernel and thrown away — there is no `[N, V]` logits tensor on the training path to read it from.

**Hard constraint from Josef (CTO):** experiments are memory-bound — they press right against VRAM limits. **No added VRAM is acceptable.** This rules out materializing `[N, V]` logits just to compute logZ (~3.1 GB fp32 at dreadnought's `N=B·T=24576`, `V=32000`). Any solution must stay fused / no-materialization.

**Second constraint:** don't upgrade CCE. It's the CE kernel the in-flight baseline depends on; upgrading changes baseline numerics and needs fleet-wide coordination. (Available as a fallback only.)

---

## 3. What we discovered about the installed CCE (25.4.3)

The implementation was first written assuming CCE's public `return_lse=True` (returns `(loss, lse)` — exists in the current public Apple repo). **The installed 25.4.3 does NOT have it:**

```
linear_cross_entropy(e, c, targets, bias=None, ignore_index=-100, softcap=None,
    reduction='mean'|'sum'|'none', shift=0, filter_eps='auto',
    accum_e_fp32=False, accum_c_fp32=False, filter_e_grad=True, filter_c_grad=True,
    impl=CCE, vocab_parallel_options=None) -> torch.Tensor    # scalar loss ONLY
```

`linear_cross_entropy(..., return_lse=True)` → `TypeError: unexpected keyword argument 'return_lse'`.

Other relevant exports in 25.4.3:
- `indexed_dot` — a **module**, not a function. Its only callable is `indexed_neg_dot_forward_kernel` (raw Triton, **not autograd-wrapped**, returns the **negative** dot).
- `cce_lse_forward_kernel(e, c, ...) -> (loss, lse) | lse` — a raw forward-only kernel that computes per-token logZ, but **not differentiable** (no autograd Function around it).
- `cce.cce.LinearCrossEntropyFunction` — the real autograd Function for CE.
- `cce_backward_kernel(do, e, c, bias, lse, ...)` — the backward kernel; consumes `lse`.

So: **no public differentiable way to get logZ out of CCE** in this version.

---

## 4. The chosen workaround ("option D") and the identity it rests on

Cross-entropy per token = `logZ - logit_target`. Therefore:

```
logZ = CE_per_token + logit_target
```

Both pieces are obtainable **without materializing `[N, V]`**:

```python
ce_none      = linear_cross_entropy(e, c, targets, reduction='none', ignore_index=pad)  # CCE, differentiable, fused
logit_target = (e * c.index_select(0, targets)).sum(-1)   # c[targets] is [N, D] (like e), NOT [N, V]; plain torch
logZ_recon   = ce_none + logit_target
zloss        = mean(logZ_recon[non_pad]**2)
```

`c[targets]` is an `[N, D]` gather — the same footprint as the hidden states, **not** `[N, V]`. So this is **zero added VRAM**, plain differentiable torch for the target-logit term + CCE's own autograd for the CE term. No custom kernel, no upgrade. This satisfies all constraints **if its gradient is correct.**

---

## 5. The problem: the gradient diverges from a pure-torch reference

We compare option D's `d(zloss)/d(e)` and `d(zloss)/d(c)` against a pure-torch reference:

```python
logZ_ref = logsumexp((e @ c.t()).float(), -1);  zloss_ref = mean(logZ_ref[non_pad]**2)
```

All tests: bf16, CUDA, N=512, V=4096, D=512, pad_id=0, fixed seed, **shared leaf tensors** (so the only difference is the path under test). Agreement reported as **(norm-relative error, cosine similarity)** of the gradient tensors — `||gA - gB|| / ||gB||` and `cos(gA, gB)`. (Per-element max-relative was tried first and rejected: near-zero gradient elements make it explode under bf16, it's not a meaningful metric.)

**Result:**

| comparison | grad_c (norm-rel, cos) | grad_e (norm-rel, cos) |
|---|---|---|
| option D vs pure-torch ref | **0.117, 0.9933** | **0.140, 0.9901** |

A ~12–14% norm-relative gradient disagreement, cosine ~0.99. **Forward logZ matches to ~0.011** (bf16-level) — so the forward is fine; the disagreement is in the **backward**.

This is precisely the failure mode Rook flagged from the OLMo-2 paper ("2 OLMo 2 Furious", arXiv 2501.00656), which documented a stark discrepancy between Flash-Attention's fused-CE z-loss and a reference implementation. The whole reason we're gradient-checking is that paper.

---

## 6. Hypotheses tested and ruled OUT (all on-rig, same setup)

1. **CCE's backward is broken (OLMo case).** Tested plain MEAN CE grad, CCE vs torch: **(0.0034, 0.999994)** for grad_c, **(0.0071, 0.999975)** for grad_e. → **CCE's CE backward is essentially perfect. NOT a broken kernel.** RULED OUT.

2. **CCE's `reduction='none'` mishandles per-token upstream grad_output.** Tested `(ce_none * w).sum().backward()` with arbitrary per-token weights `w`, CCE vs torch: **(0.0033, 0.999995)** grad_c, **(0.007, 0.999976)** grad_e. → **`reduction='none'` correctly propagates arbitrary per-token grad_output.** RULED OUT. (This matters because `mean(logZ^2)` produces a per-token upstream of `2·logZ_i/N` into `ce_none`.)

3. **CCE's gradient-filter approximation.** `filter_eps=0.0`, `filter_e_grad=False`, `filter_c_grad=False`, and all combinations: **no change** (0.116–0.117). RULED OUT.

4. **Target-logit precision mismatch (bf16 vs CCE's fp32-accumulated logit).** Computing `logit_target` in fp32 accumulation: **0.1275 vs 0.1168 — slightly WORSE, not better.** RULED OUT.

5. **bf16 backward nondeterminism / noise floor.** Pure torch-vs-torch control (reference vs itself with a trivially different but identical-math op order): **exactly 0.0, cos 1.0.** → bf16 backward is deterministic here; **there is no noise floor to hide behind — the 0.12 is fully real.** RULED OUT as an explanation.

---

## 7. The leading hypothesis (not yet confirmed) — the reference may be the wrong baseline

Here is the reframe that fits all the evidence:

- CCE's CE gradient is correct (fact 1). `reduction='none'` propagates per-token weights correctly (fact 2). The forward logZ matches (§5).
- The pure-torch **reference** computes `logits = e @ c.t()` **in bf16** (e, c are bf16), then upcasts for `logsumexp`. CCE computes its CE with **fp32 accumulation over the D dimension inside the kernel**.
- So the two paths see **different logit values** (bf16-matmul logits vs fp32-accumulated logits), differing at ~bf16 level (~0.01).
- The z-loss gradient `d(zloss)/d(logits) = (2·logZ/N)·softmax(logits)` is **extremely sensitive to small logit perturbations** because softmax is sharp at V=4096 with random logits. A ~0.01 logit difference is plausibly amplified into a ~0.1 gradient difference.

**Conclusion if true:** option D is **more accurate** than the bf16 reference, not less. The 0.12 discrepancy is the reference being inaccurate, not option D being wrong. We've been chasing a phantom created by comparing against a bf16-matmul baseline.

**How to settle it definitively (in progress):** compute the **analytic truth** in fp32 — `gL_true = (2·logZ/N)·softmax(L)` from a high-precision logits leaf `L`, masked to non-pad — then push it through the chain rule to e and c:

```
ge_truth = gL_true @ c          # d(zloss)/d(e),  since logits = e @ c.T
gc_truth = gL_true.T @ e        # d(zloss)/d(c)
```

Then compare **both** option D and the bf16 torch-reference against `(ge_truth, gc_truth)`. Whichever matches the fp32 analytic truth is the correct one. **Expected (to be confirmed):** option D matches the analytic truth and the bf16 reference is the outlier → option D is correct, ship it, set test tolerances against the analytic truth rather than the bf16 reference.

(A diagnostic battery script that runs all of §6 + this analytic-truth master check in one paste is being finalized; the chain-rule transposes/shapes and the fp32-ness of the "truth" are being adversarially reviewed before it goes to the rig.)

---

## 8. If option D is genuinely wrong (the fallback)

If the analytic-truth check shows option D does **not** match `(ge_truth, gc_truth)`, the composition of two separately-fused gradients is the problem, and the fix is a **custom `torch.autograd.Function`** for the z-term:
- **Forward:** logZ as a detached value (via `cce_lse_forward_kernel`, no materialization).
- **Backward:** apply the closed-form `(2·logZ/N)·softmax(logits)` contracted against `e` and `c` — computed in **vocab-chunks** to honor the no-VRAM constraint (the softmax is `[N, V]` if formed all at once, so it must be tiled, the way CCE itself tiles). This is correct by construction and validatable against the fp32 analytic truth to high precision.

Cost: real kernel-aware code (chunked backward). It's the robust fallback if the cheap composition can't be made to match.

---

## 9. Open questions for fresh eyes

1. **Is the bf16-matmul reference actually the wrong baseline?** (§7) Does the fp32 analytic-truth comparison confirm option D is the accurate one? This is the crux.
2. **Does CCE's `reduction='none'` return exactly `logZ - logit_target` per token**, or does it apply any label-smoothing / softcap / shift / normalization we haven't accounted for? (Defaults say no: `softcap=None`, `shift=0` — but unverified per-token against a hand-checkable tiny case.)
3. **Pad interaction:** `ce_none` is 0 at ignore_index rows, but `logit_target` is nonzero at pad rows, so `logZ_recon` at pad rows is garbage. We mask zloss to non-pad, so pad rows don't enter the loss — but do they perturb the graph/grad via CCE's internal `valids` handling? (Re-running with no pad tokens isolates this.)
4. **Is ~0.99 cosine / ~0.12 norm-rel even a problem in practice?** z-loss is a small regularizer (alpha ~1e-4) with an annealed ramp. If the gradient direction is 99% aligned and only the magnitude is bf16-fuzzy, does it matter for training stability? (Leaning: we should still get it right, because the whole point of the feature is a controlled, trustworthy intervention — but worth stating the stakes.)
5. **Tolerance philosophy:** if option D matches the fp32 analytic truth, what's the right pass tolerance for the bf16 *training* path — vs the analytic truth, not vs a bf16 reference?

---

## 10. Current state of the code

- Feature is **implemented and merged to main** in both repos (model half in `common_fsdp2/model_v2.py`, trainer half in `mara_fsdp2/train_mara.py`, config in `scs_pilot_1.yaml`, commented example in `dreadnought_v2.yaml`).
- **Disabled path is byte-for-byte identical** to baseline (verified: same loss + all gradients, maxdiff 0.0) — so all current/baseline runs are safe and unaffected.
- The **enabled path is currently broken on the rig** (it still calls `return_lse`, which 25.4.3 rejects). It must be rewritten to option D (or the fallback) before any z-loss run. No enabled run has happened.
- alpha-annealing schedule (`get_zloss_alpha`, resume-safe via absolute global step) is implemented and unit-tested (21 asserts pass) — independent of this gradient issue.
- Validation scripts in `mara_fsdp2/`: `test_zloss_cce_equivalence.py` (original return_lse canary, now superseded), `test_zloss_optionD_rig.py` (option D forward+backward+memory; `--fast` mode for low-VRAM correctness-only). Both self-tested bug-free on Windows via stub.

---

## TL;DR for a reviewer

We need per-token logZ (for `alpha·mean(logZ²)`) out of a fused CE kernel that never materializes logits, with zero added VRAM and no kernel upgrade. CCE 25.4.3 won't hand back a differentiable logZ, so we reconstruct it as `CE(reduction='none') + target_logit`. Its gradient disagrees with a pure-torch reference by ~12% norm-rel / 0.99 cosine. **This is now resolved — see §11.** The resolution: option D is mathematically correct (matches the fp32 analytic truth to ~1e-5); the ~12% is benign bf16 catastrophic-cancellation precision loss, baked into CCE's `ce_none` inside the kernel and un-fixable from the trainer side without violating the no-VRAM constraint. The only open question is whether ~0.99-cosine is good enough for a small annealed regularizer, or whether to build a custom autograd Function for an exact gradient.

---

## 11. RESOLUTION (confirmed on rig with the real CCE kernel)

### 11.1 Root cause: catastrophic cancellation in the bf16 reconstruction

`logZ = ce_none + logit_target`. CCE computes `ce_none = logZ − logit_target` **inside its Triton kernel** (from bf16 `e`, `c`), where `logZ ~ O(8)` and `logit_target ~ O(8)` and their difference `CE ~ O(small)`. When we re-add `logit_target` to recover `logZ`, the low-order bits lost in that subtraction **cannot be recovered** — classic catastrophic cancellation. This perturbs the per-token upstream `2·logZ/N` that flows into the backward, producing a gradient that is **direction-correct (~0.99 cosine) but magnitude-noisy (~12% norm-rel)**.

The pure-torch reference only looked "better" because it computes `logsumexp` **directly** (no subtract-then-re-add round-trip). But it is the *less* representative baseline — training literally uses CCE's `ce_none`.

### 11.2 The proof: option D is mathematically correct (rig, real CCE 25.4.3)

Diagnostic `mara_fsdp2/zloss_diagnostic_rig.py` compares option-D(bf16), torch-ref(bf16), and **option-D-composition-in-fp32** against the **fp32 analytic truth** `gL = (2·logZ/Nk)·softmax(L)`, chained `ge = gL@c`, `gc = gL.T@e`. Rig output (N=512, V=4096, D=512, bf16):

| comparison vs fp32 analytic truth | ge (norm-rel, cos) | gc (norm-rel, cos) |
|---|---|---|
| **SANITY** (autograd vs closed-form `(2logZ/N)softmax`) | 3.9e-7, 1.000000 | — |
| **option D, bf16** | 0.139, 0.9904 | 0.117, 0.9933 |
| pure-torch ref, bf16 | 0.0027, 0.999996 | 0.0039, 0.999992 |
| **option D composition, fp32 (CONTROL)** | **1.0e-5, 1.000000** | **9.3e-6, 1.000000** |

The **fp32 control is the decisive line**: the *same* option-D algebra, run in fp32, matches the analytic truth to **~1e-5 / cos 1.0**. So the composition is **mathematically exact**; the bf16 gap is *only* precision. Supporting checks all passed:
- **(1)** CCE `reduction='none'` vs torch CE per-token: **2.9e-6** → it is raw CE, no smoothing/softcap/shift.
- **(2)** forward identity `ce_none + (e·c[tgt])` vs `logsumexp`: **0.011** (bf16 level) → identity holds.
- **(4)** pad isolation (no pad tokens): **identical** (0.141/0.990) → not a pad/ignore_index artifact.
- zloss scalar: optionD **77.74416** vs truth **77.74395** → forward value exact to 5 sig figs.

This also **clears the OLMo-2 concern** (arXiv 2501.00656): CCE's CE backward is correct (plain-CE grad vs torch ~0.003); there is no broken fused kernel.

### 11.3 The fp32 fix does NOT work (rig-confirmed) — damage is in the kernel

Natural idea: accumulate the reconstruction in fp32 (`.float()` is on `[N,D]`/`[N]` tensors → still zero VRAM). **Tested on rig — it does not help:**

| variant | ge (norm-rel, cos) | gc (norm-rel, cos) |
|---|---|---|
| A: all bf16 (current option D) | 0.136, 0.9907 | 0.116, 0.9933 |
| B: fp32 reconstruction (`ce_none.float()` + fp32 target logit) | 0.143, 0.9898 | 0.128, 0.9919 |

**Why:** `ce_none.dtype` is **already `torch.float32`** — CCE returns the loss in fp32. The cancellation happened *inside the kernel* (bf16 `e@c` → `logZ − logit_target`) **before** we receive it. Upcasting afterward recovers nothing. The fp32 *control* in §11.2 was only exact because it recomputed `ce_none` itself from fp32 logits (`F.cross_entropy(e.float()@c.float().T)`) — which requires materializing `[N,V]` logits, **violating the no-VRAM constraint**. So there is **no trainer-side fp32 fix.**

### 11.4 The decision (open — getting second opinions)

Option D's gradient is correct but **bf16-precision-limited at ~0.99 cosine / ~12% magnitude**, and that's the floor for any no-materialization trainer-side approach. The choice:

**(A) Adopt option D as-is.** Zero added VRAM, ~20 lines, no kernel code, ready now. Accept a gradient that is 99% direction-correct with ~12% magnitude noise. For a **small annealed regularizer** (`alpha ~ 1e-4`, ramped from 0) this is almost certainly immaterial — it's a gentle nudge on logit magnitude, not a precision-critical loss term, and the ramp itself dwarfs the 12%. *Lowest risk.*

**(B) Custom autograd Function (exact gradient).** Forward `logZ` detached via `cce_lse_forward_kernel` (no materialization); backward = closed-form `(2·logZ/Nk)·softmax` contracted against `e`, `c`, computed in **vocab chunks** to stay within VRAM. Matches the fp32 truth. Cost: ~100+ lines of kernel-aware tiled code, careful validation, and it **reintroduces the "did we get the fused backward right" risk** (the very OLMo-style risk we just cleared for CCE) that we'd have to test exhaustively. *Higher effort + risk.*

**(C) Ship (A) now, build (B) later only if needed.** Adopt option D to unblock the experiment, clearly noted as precision-limited. If a real z-loss run shows the regularizer isn't biting (logZ not declining as `alpha` ramps), *then* invest in (B). Validates the whole feature cheaply before committing to kernel work. *Pragmatic.*

**Code's recommendation: (C)** — adopt option D now, because (i) the gradient is provably correct in direction and exact in forward value, (ii) the magnitude noise is tiny relative to a ramped `alpha~1e-4`, and (iii) we can *measure* whether it matters from the first run's `logZ` trajectory before spending effort on a custom kernel. If `logZ_mean` declines smoothly as `z_a` ramps, (A) was sufficient and we're done.

### 11.5 Questions for second opinions

1. **Is ~0.99 cosine / ~12% magnitude gradient noise acceptable for a log-partition regularizer at `alpha~1e-4`?** Intuition says yes (it's a soft tether, not a hard objective), but is there a reason the magnitude error specifically would matter — e.g. interaction with the `alpha` sweep, or with the annealing ramp where the effective coefficient is even smaller?
2. **Does anyone know a CCE 25.4.3 path to a differentiable per-token logZ that avoids the cancellation** (i.e. computes `logsumexp` directly, not as `CE + target_logit`)? `cce_lse_forward_kernel` returns logZ but is forward-only/non-autograd. Is there a supported way to get it *with* a backward, short of writing the Function?
3. **Is the custom-Function (B) worth the OLMo-style risk it reintroduces?** We just spent significant effort *proving* CCE's backward is correct so we don't have to trust a hand-written fused backward; (B) puts us back in the business of hand-writing one. Does the exactness gain justify that?
4. **Upgrade angle:** newer public Apple `cut_cross_entropy` has `return_lse=True` (returns logZ directly, computed in-kernel — no cancellation). We rejected upgrading because it changes the CE kernel the in-flight baseline depends on and needs fleet coordination. Is that the right call, or is a controlled upgrade (pin version, diff baseline CE numerics before/after) actually cleaner than either (A) or (B)?

---

## 12. FINAL RESOLUTION — backend gate + implementation (rig-measured, IMPLEMENTED)

Rook directed a **4-variant precision gate** (msg #125, superseding his earlier "fp32 by default") instead of assuming any python-side fp32 fix works. The untested lever was **CCE's own `accum_e_fp32`/`accum_c_fp32` backward flags** — which force fp32 accumulation *inside* the CCE backward, exactly where the cancelling target-class term lives (distinct from the python-side reconstruction, which §11.3 already showed can't help).

### 12.1 The gate result (rig, real CCE 25.4.3, vs fp32 analytic truth)

| variant | grad_e (cos) | grad_c (cos) | gradient verdict | real-shape peak vs baseline |
|---|---|---|---|---|
| **A** current bf16 | 0.9903 | 0.9933 | the floor | **+0.29 GB** |
| B python fp32 recon | 0.9899 | 0.9919 | no better than A | +0.72 GB |
| **C** CCE `accum_*_fp32` | **0.9987** | **0.9962** | **~halves the error** | **+0.74 GB** (≈+0.45 vs A) |
| D accum + fp32 dot | 0.9983 | 0.9949 | worse than C | OOM (then larger; see note) |

**C wins on gradient** (cosine 0.990→0.999, norm-rel ~0.12→~0.05). The cost is **~+0.45 GB** over A at dreadnought's head shape (N=24576, V=32000, D=2560). B doesn't help (confirms §11.3); D is worse than C *and* heavier. (D's "OOM" in the single-process run was allocator fragmentation from running 4th; an isolated-subprocess re-measure is available in `zloss_mem_isolated.py`, but D is dominated by C regardless so it doesn't matter.)

### 12.2 The decision: expose it as a config knob (no single hardcoded choice)

The right backend is **run-dependent** (memory-bound runs want A's lightness; runs with headroom want C's trustworthy gradient). So instead of baking one in, `z_loss.backend` selects:
- **`fp32_accum`** (default) = variant C: grad cosine ~0.999, ~+0.45 GB. The trustworthy default for a feature we read `logZ` from on a model we care about.
- **`bf16`** = variant A: lightest memory, grad cosine ~0.990 (fine for a small annealed regularizer).

Custom autograd Function (§11 option B) stays **deferred** — the gate gives us a near-exact gradient (C) without re-entering hand-written-fused-backward / OLMo-risk territory.

### 12.3 Implemented (option D rewrite, replacing the broken `return_lse` path)

- `common_fsdp2/model_v2.py`: new `_zloss_optionD(h, W, tgt, pad_id, fp32_accum)` — logZ = `CE(reduction='none')` + `(h * W[safe_targets]).sum(-1)`, no `[N,V]`. `safe_targets` clamps masked rows into `[0, vocab)` (guards a future non-vocab `ignore_index`); vocab-parallel caveat noted. `_masked_zloss`/`return_lse` removed. AuxHead + main-head + the model flag now thread `_zloss_fp32_accum` (None=off | False=bf16 | True=fp32_accum).
- `mara_fsdp2/train_mara.py`: `z_loss.backend` parsed/validated (`bf16`|`fp32_accum`, default `fp32_accum`); post-build sets `_zloss_fp32_accum` accordingly.
- Configs: `backend` field + tradeoff comment in `scs_pilot_1.yaml` and the `dreadnought_v2.yaml` example.

**Verified (Windows, real model code):** disabled path **bit-identical** (loss + all grads, maxdiff 0.0); both backends leave the main CE loss unchanged and stash a **differentiable** zloss (grad flows to the head); `eval()` skips it; scaffold deepest-tap wired; `safe_targets` clamp prevents OOB gathers; config backend validation (valid/default/fatal) passes. The accum-flag *numerical* effect (C's 0.999) is only exercisable with the real CCE kernel — that's what the rig gate above already measured.

### 12.4 Status

Feature is now **correctly implemented on the installed CCE** (the `return_lse` breakage is fixed). Remaining before an enabled run: the head-metric logging for run #1 (Rook #124 item 6 — head weight_norm / grad_norm / grad_norm-ratio as global FSDP-reduced norms, logZ rms/p95), and a rig smoke of an actual enabled step. Disabled runs remain byte-identical and safe throughout.
