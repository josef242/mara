# dn4 head-hygiene — implementation spec

Status: **IMPLEMENTED + VALIDATED** (both levers). Math-APPROVED design; passed an internal 3-agent
code-review (✎) AND a blind 4-agent review (one HIGH precision defect found + fixed; no blockers). Both
levers SHIP OFF / inert by default. The design below matches the shipped code.

**Implementation & validation:**
- **Lever 1 — head gauge projection** (`head_gauge_projection: {enabled, init_row_center}`):
  `common_fsdp2/muon_fsdp2.py` (`_project_head_update_gauge_` + the Adam-path hook) + `train_mara.py`
  (config, guards, param-ID startup assertion, init, banner, telemetry). Validated: fp32 unit test
  (theorem; gauge accumulation prevented), config (dn3 inert + guards fire), **live 2-GPU FSDP**.
  **MERGED to main.**
- **Lever 2 — deadband centered z-loss** (`z_loss: {target: centered, tau, alpha, backend}`):
  `common_fsdp2/model_v2.py` (`_centered_zloss_deadband` + forward branch) + `train_mara.py` (target/tau
  validation incl. an untied-head guard, config setter, banner). Validated: math unit test
  (gauge-invariance; zero common-mode gradient `6.25e-7`; deadband), **live 2-GPU FSDP** (`zloss ==
  (logZ_c−τ)²` exact). **Blind-review fix:** the centered target logit is now formed DIRECTLY in fp32
  (`h·(W_target−μ)`), dodging a bf16 cancellation at the O(400) gauge scale that corrupted the deadband
  near τ — real-vs-reference tightened from `1.8e-4` to `1.2e-7`.

Deferred polish (non-blocking): clean centered diag labels + surface `h_mu` telemetry; an automated
multi-rank bf16/SR test (the live smoke covers it); τ/α tuning post-dn3.

The two earlier Math edits are folded in (⚑): the validation correction (CE is gauge-invariant — must
MATCH, not drift) and the param-ID startup assertion.

Goal, per Math:

> **Let the centered head mature; prevent useless gauge buildup and pathological centered collapse.**
> Separation of concerns: **gauge hygiene ≠ centered-head control ≠ head-weight shrinkage.**

dn4 is a **fresh run** (head control from step 0, not a retrofit). Untied head (`tie_word_embeddings:
false`), `optimizer_type: normuon_fsdp2`, `muon_adam_state_dtype: fp32` (selects the non-fused Adam code
path; this does **not** make the buffers fp32 — see ✎dtype).

> **What this buys (and doesn't), per Math.** This is **training-time gauge hygiene**, not CE-visible
> head regularization — in exact arithmetic it does NOT change the CE-visible centered trajectory vs
> training uncentered + centering offline. *Buys:* cleaner raw logZ / raw head norm, less bf16
> common-mode precision risk, gauge-clean checkpoints throughout, better diagnostics. *Does NOT buy:*
> better loss, lower `logZ_c`, lower `‖W_c‖`, or any centered-head brake — those are Lever 2. Three
> separate problems: body radial-update projection / head applied-update gauge projection / centered
> logit ceiling.

---

## Lever 1 — Head applied-update gauge projection  **(BUILD NOW; body-independent)**

### The math
The head `W = output.weight` has shape `[V, D]` (vocab × dim). The CE-invisible gauge is the common
vocab-row direction: adding `1·c^T` to all rows shifts every logit by `h·c`, which logsumexp/CE/softmax
cancel. Let `U = m̂/√v̂` be the Adam **applied update** (post bias-correction, pre weight-step). Each step
we remove the gauge component of the *applied update*:

```
Ū = (1/V) Σ_v U_v              # mean over vocab rows, shape [D]
U ← U − 1·Ū^T                  # subtract Ū from every row
```

`P(W − ηU) = P(W) − η·P(U)` (projection is linear/idempotent, and scalar-LR commutes with it): the
**centered** head `P(W)` evolves only by the CE-visible `P(U)`, while the **gauge** `(I−P)(W)` never
accumulates. We do **NOT** project `exp_avg`/`exp_avg_sq` — the failed `row_center_head` path; a row-mean
in `m` is not pure gauge after `/√v` (coordinate-preconditioned), so projecting the first moment distorts
the real centered update. Projecting the applied `U` is the clean analogue of the body's Muon tangent
projection (null direction: radial `W` there; common vocab-row mode here).

> **⚑ `V` = the exact logit dimension the softmax sees.** `Ū` averages over the rows CE actually uses
> (all 32768 for dn4). If any rows are masked out of the softmax, exclude them — the gauge is the common
> mode over *participating* logits only.

**✎ WD does not break it (review-confirmed).** With the real code (`p.mul_(1−eff_lr·wd)` at 734 on the
full weight, then `p.add_(update, −eff_lr)` at 736): if the arms start a gauge `1·c^T` apart, after one
step the gap is `1·[(1−eff_lr·wd)c + eff_lr·Ū]^T` — still a pure rank-1 gauge. WD maps gauge→gauge
(`1·c^T ↦ (1−ηλ)·1·c^T`), so it decays any residual gauge toward zero (benign). We project **only U**.

### Insertion point (exact)
`common_fsdp2/muon_fsdp2.py`, NorMuon **non-fused** Adam path in `MuonFSDP2.step()`: `update =
adam_update(...)` at **line 717**; decoupled WD 719-734; weight step at **line 736**. Insert immediately
after 717, gated to the head; WD + weight step unchanged:
```python
update = adam_update(...)
if id(p) in self.head_gauge_ids:
    _project_head_update_gauge_(update)      # in-place: U ← U − 1·mean_v(U)^T
p.mul_(1 - effective_lr * weight_decay)
p.add_(update, alpha=-effective_lr)
```

### ✎ Reduction — reuse the tested helpers (do NOT hand-roll)
`output.weight` is FSDP2-sharded **Shard(0) on vocab** (root `fully_shard(model, dp_mesh)`, `[V,D]` head
not separately wrapped). The vocab-row-mean is a **global** reduction over the sharded dim, not per-shard.
Implement `_project_head_update_gauge_` as a thin wrapper over `row_center.py`'s internals:
```python
mu = _global_row_mean(update, vocab_dim=0)       # fp32 local row-sum + all_reduce(SUM)/count over the
_subtract_row_mean_(update, mu, vocab_dim=0)      # head's own device_mesh group; broadcast-subtract
```
Mandatory, not stylistic:
- **✎dtype: the update is bf16** (inherits `p`'s dtype = `FSDP_param_dtype: bf16`; `muon_adam_state_dtype:
  fp32` only picks the code path). A naive bf16 `sub_` leaves a **biased residual common-mode** that
  accumulates. `_subtract_row_mean_` does fp32 compute + **stochastic-rounding** bf16 write-back; fp32 →
  plain copy. Correct either way.
- **✎pg not in scope** in the Adam loop; `_global_row_mean` derives the group from
  `update.device_mesh.get_group()` and DTensor-guards (world=1 / non-dist safe).

`_project_head_update_gauge_` is **new code** calling those two helpers — not `row_center_head_` (which
projects the weight + optional exp_avg).

### Gating + ⚑ param-ID startup assertion
`head_gauge_ids: set[int]` built in `train_mara` and attached (`optimizer.head_gauge_ids =
{id(output.weight)}`), read as `id(p) in self.head_gauge_ids` (same pattern as `wd_overrides`). **✎ Add
`self.head_gauge_ids = set()` in `MuonFSDP2.__init__`** so the disabled case is an empty-set test.

**⚑ Build `head_gauge_ids` from the post-FSDP-wrap optimizer params** — `id(output.weight)` must be the
SAME object the optimizer loop iterates (DTensor wrapping / param replacement can change identity if
captured too early). Assert at startup, when `head_gauge_projection.enabled`:
- **exactly one** optimizer param matches `head_gauge_ids` (count 0 ⇒ silent no-op ⇒ fatal; count >1 ⇒
  fatal),
- matched param's logical shape is `[V, D]`,
- matched param is in the **non-fused Adam path**.

### What it deliberately does NOT do
No `exp_avg`/`exp_avg_sq` projection. No post-step weight surgery. No optimizer-state mutation; **no
checkpointed state** (mutates only the transient update).

### Init (worth doing for dn4) — ✎ needs its own gate
One-time **weight-only** row-center at init via `_row_center_head_step(model, optimizer,
want_exp_avg=False)` (before any optimizer state exists), so the stored head starts gauge-free. **✎ NEW
call site under a new `head_gauge_projection.init_row_center` flag — NOT the legacy `row_center_enabled`
branch** (which carries the failed per-step weight/state surgery).

### ✎/⚑ Guards / validation (key all to `head_gauge_projection.enabled`)
In the Settings validation block (near the existing ~line 5010 `fatal_error`s):
- `fatal_error` if `tie_word_embeddings` (tied ⇒ projecting corrupts the embedding).
- `fatal_error` if `muon_adam_state_dtype` selects the **16-bit fused** Adam path (`U` never exposed at
  717 ⇒ silent no-op).
- **✎ Mutual exclusion:** `fatal_error` if `head_gauge_projection.enabled` AND legacy `row_center_head`
  enabled (competing implementations; both ⇒ double-projection + rejected `exp_avg` surgery).
- **⚑ Bias guard:** `fatal_error` (or warn + handle) if the head has an **output bias** — the bias mean
  is also a gauge. dn4 is bias-free (`nn.Linear(D, V, bias=False)`); future-proofing.

### Config schema (mirrors `tangent_project`)
```yaml
head_gauge_projection:
  enabled: true          # project the applied Adam update's vocab-row-mean out of the LM head each step
  init_row_center: true  # one-time weight-only row-center at init (new gate; not the legacy flag)
```

### ⚑ Validation (CORRECTED per Math — CE is gauge-INVARIANT, it does NOT drift)
The earlier draft was **wrong** to say "raw loss drifts; regress the inter-arm loss difference on h·c."
For a pure gauge gap `W_B = W_A + 1·c^T`: logits shift `z_B = z_A + h·c`, so **both** `logZ` and the
target logit `z_y` shift by `h·c` and **cancel** in `CE = logZ − z_y`. So CE/log-probs **match**; only
raw `logZ` (and `z_y`) drift.

**Step 1 — deterministic fp32 unit test (do first; validates the theorem directly):** clone identical
model+optimizer state into arms A/B, feed the same gradient, apply ordinary Adam in A vs gauge-projected
Adam in B, and assert centered heads match: `‖P(W_A) − P(W_B)‖ / ‖P(W_A)‖ ≈ 0` (fp32 tol). Direct test of
`P(W − ηU) = P(W) − η·P(U)`.

**Step 2 — bf16/FSDP/SR equivalence run (looser tol, "no growing centered drift"):**
1. **Centered heads match** (production assertion): `‖P(W_proj) − P(W_unproj)‖_F / ‖P(W_proj)‖_F` stays at
   tolerance and does **not** systematically grow beyond bf16/SR noise.
2. **CE / log-probs MATCH:** `CE_proj ≈ CE_unproj`, `log p_y,proj ≈ log p_y,unproj` (gauge-invariant).
3. **Raw logZ and target logit each shift by the same scalar `h·c`:** `ΔlogZ ≈ h·c`, `Δz_y ≈ h·c`, so
   `ΔCE ≈ ΔlogZ − Δz_y ≈ 0`. (logZ drifts; CE does not.)
4. **`logZ_c` matches** between arms.

Caveat: over long horizons a projected run can still diverge from an unprojected one for finite-precision
reasons — because the *unprojected* head carries a growing bf16 common mode. Not a bug; it's a reason for
in-loop hygiene. The right signal is **no growing centered drift**, never bit-equality.

---

## Lever 2 — Deadband centered z-loss  **(BUILD, SHIP OFF; tune τ/α after dn3)**

A **ceiling**, not constant pressure: `L_zc = α_c · max(0, logZ_c − τ)²`, with `logZ_c = logZ − h·μ`.
Gauge-invariant: `∂logZ_c/∂z_i = p_i − 1/V`, which **sums to zero** over vocab → zero common-mode
gradient (exactly the intent). Implements `docs/ZLOSS_CENTERED_PLAN.md` Objective A inside
`_zloss_optionD` (no `[N,V]` materialization) plus the deadband. **Defer** `τ`, `α_c` to post-dn3.
Reference branch: `τ ≈ 120–128`, `α_c ≈ 5e-6–1e-5`, fp32 accum. Ship `enabled: false`; if `logZ_c` lives
in the KEEL-normal band the loss is identically inactive.

Raw z-loss stays **off**. (✎ wording: the 78% gauge fraction is a *decomposition fact* — the gauge was
the path of least resistance, not literally "78% of gradient budget". The centered deadband is justified
as **gauge-invariant + thresholded**, not as budget efficiency.)

---

## Telemetry  (centered geometry already built; small adds)
Already at val cadence in `centered_geom`: `logZ_c`, `Wc_fro` (=‖P(W)‖), `s1_c`, `eff_rank_c`,
`spec_conc_c` — reuse as-is. Add:
- **⚑ `‖Ū‖` before AND after the SR write-back** (per step, ~free): "after" ≈ 0 confirms the projection
  actually landed; a nonzero "after" flags a biased bf16 residual. Stash `optimizer._last_head_ubar_norm`
  for the train loop to read after `step()`, gated on the new flag (not `row_center_enabled`).
- **`pdr_head,c = ‖P(ΔW_head)‖/‖P(W_head)‖`:** at val cadence, folding `‖P(ΔW)‖` over the existing
  `Wc_fro` into the `centered_geom` record.
- Also log `‖μ(W)‖`, `h·μ`, `logZ`, `‖W_c‖`, `eff_rank_c`, `spec_conc_c`, rare-token NLL by frequency
  bucket. **Raw head ‖W‖ alone is demoted** — DN2's raw head story was too gauge-contaminated to be a
  health metric.

## Offline row-center  (trivial; do regardless)
Before SFT/RLHF/DPO/quant/export: `W ← W − 1·μ^T` on the untied bias-free head — function-preserving
(reuse `row_center.py` as a checkpoint script). Independent of the in-loop lever.

---

## dn4 v1 recipe (fresh run)
```yaml
head_gauge_projection: { enabled: true, init_row_center: true }   # Lever 1
row_center_head:       { enabled: false }                         # legacy; mutually exclusive
z_loss:                { enabled: false }                         # raw z-loss off
centered_z_loss:       { enabled: false, form: deadband }         # Lever 2 built, staged off
# output-head WD gentle/default (no cranking); output_lr_batch_adjust kept (pacing, not a brake);
# aux heads per representation goals only (NOT head control); centered telemetry + pdr_head,c + ‖Ū‖ on.
```
Escalate only if `logZ_c` / `eff_rank_c` / rare-token behavior leave the KEEL band → enable Lever 2.
Logit soft-cap / head weight-norm remain research branches, only if `W_c` proves pathological *despite*
gauge hygiene + deadband.

---

## Math's answers to the six questions (for the record)
1. **Mechanism:** yes — project the bf16 applied update `U` (fp32 global row-mean + SR write-back, never
   `exp_avg`); much safer than projecting `exp_avg`.
2. **Validation:** centered-weight-gap + `logZ_c` tracking are right; **CE must MATCH** (raw logZ + target
   logit drift by `h·c` and cancel) — corrected above.
3. **Init:** yes, one-time weight-only init row-center under a new gate.
4. **WD:** leave WD unprojected; it decays residual gauge naturally. No post-step surgery.
5. **Centered guard:** yes — deadband centered z-loss, shipped off, τ/α from the observed KEEL band.
6. **Sequencing:** agreed — land Lever 1 + telemetry now; build Lever 2 disabled; tune after dn3/dn4.
