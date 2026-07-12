# Math Agent Brief #5 — Is KeelHaul's high `nrm` actually a problem, given Muon discards gradient magnitude? (+ tangent-projection norm audit)

## TL;DR / what we want reviewed
KeelHaul (KEEL ultra-deep + tangent-projection body-ramp fix) is showing a **high pre-clip
gradient norm** (`nrm` ~1.2–2.2, spiky to ~5, against a `clip=1.0`), and on the Dashboard its
loss has **fallen behind DN2 at matched token counts** in the ~480–640M-token region. Josef asked
us to investigate whether our own **tangent-projection `preserve_norm` flag** could be inflating the
norm. It is NOT (audit below). But the investigation surfaced a **deeper question we want you to
pressure-test**:

> For a Muon-family optimizer (Newton-Schulz orthogonalization + NorMuon neuron-renorm), the
> logged `nrm` is the **raw pre-clip gradient** norm — but the optimizer **discards gradient
> magnitude** (NS keeps only direction; NorMuon renormalizes back to pre-norm magnitude). So is the
> high `nrm` — and the `clip=1.0` fighting it — **largely decoupled from the actual update the body
> weights receive**? If so, "reduce the norm" may be the wrong goal, and the clip may be doing
> something subtler (and possibly harmful) than we assumed.

We want your read before we change a knob or restart.

---

## Part 1 — Tangent-projection `preserve_norm` audit (the thing we were asked to check)

### What the flag does (source: `common_fsdp2/muon_fsdp2.py`, projection block ~L377–399)
Applied AFTER NS + apply_scaling + NorMuon, body matrices only, global all-reduced coefficient:
```
c = <U,W> / ||W||²            # global, all-reduced over FSDP shards
U ← U − c·W                   # strip radial component
if preserve_norm:             # OPTIONAL rescale
    U ← U · (||U||_before / ||U||_after)
```
- By Pythagoras (the removed `c·W` is orthogonal to the result), the projection **always reduces**
  ‖U‖: `‖U‖_after = ‖U‖_before · √(1 − cos²(U,W))`.
- Measured `cos(U,W) ≈ −0.013` (the anti-radial lean we projected out). So the magnitude reduction
  is `√(1 − 0.013²) ≈ 0.99992` — i.e. **~0.008%**.

### KeelHaul's actual setting (confirmed in the live run's saved `config_*.yaml`)
```
tangent_project: true
tangent_project_preserve_norm: false      # comment: "removed component ~1.3% of norm; rescale unnecessary"
```
### Conclusion on the audit
**`preserve_norm` is OFF, and even if ON it would restore only ~0.008% of magnitude.** Tangent
projection is, if anything, **trivially norm-REDUCING**, never inflating. **It is not the source of
the high `nrm`. Our fix is exonerated.** (Please sanity-check the Pythagoras argument and the claim
that projecting AFTER NorMuon's non-orthogonal per-neuron rescale is still the right order — that's
your prior recommendation, we just want it re-confirmed in light of the magnitude question below.)

---

## Part 2 — the real question: `nrm` vs. the magnitude Muon actually applies

### What `nrm` measures (source: `train_mara.py` `_clip_grad_norm_mixed_mesh` L1041–1075)
`nrm` = global **pre-clip** L2 norm of the raw gradient over ALL params, all-reduced. The clip then
does `g ← g · min(1, max_norm/nrm)` — **before** the gradient enters Muon. So `nrm` is a property
of the **raw gradient/loss landscape**, captured upstream of all of Muon's processing.

### What Muon does to magnitude downstream (source: `muon_fsdp2.py`)
1. **Newton-Schulz** (`zeropower_via_newtonschulz5`): orthogonalizes — drives singular values → 1.
   **Discards gradient magnitude**; output magnitude depends on shape/rank, not on ‖G‖.
2. **apply_scaling** (rms_scale=True, L119–127): `update *= 0.2·√(max(d_out,d_in))` — a **fixed
   per-matrix constant**, no dependence on training state or ‖G‖.
3. **apply_normuon** (L188–203): neuron-wise 2nd-moment normalization, then **explicitly
   renormalizes back to the pre-normalization norm** (L202: `update.mul_(vnorm / vnorm_new)`) —
   so it **preserves** update magnitude, doesn't grow it.
4. **tangent projection**: −0.008% (Part 1).
5. Final: `W ← W − lr · lr_scale · update`.

### The implication we want checked
Putting (1)–(4) together: **the per-matrix update magnitude Muon applies is essentially
state-independent** — a fixed constant (`0.2·√(maxdim)`) × `lr` × `lr_scale`, *regardless of how
large the raw gradient norm `nrm` was.* NS threw the magnitude away in step (1).

If that's right, then:
- **(a)** The high `nrm` does **not** mean the body weights are taking big steps. The step size is
  governed by lr × the fixed scale, not by ‖G‖.
- **(b)** The `clip=1.0` is rescaling the raw gradient **before** NS — but NS is **scale-invariant**
  (orthogonalizing `k·G` gives the same result as orthogonalizing `G` for k>0). So **uniform
  clipping of an all-Muon gradient is very nearly a no-op on the update direction**, and a true
  no-op on the magnitude.
- **(c)** Where the clip *does* bite: it's a **global** norm over ALL params, and the rescale
  `clip_coef` is applied to **every** grad including the **Adam-group** params (embeddings, router,
  GDN small params, norms). For those, magnitude is NOT discarded (Adam uses it via m/√v at
  non-steady-state, and clipping changes the effective LR for that step). And it's not perfectly
  uniform across the NS step either if `nrm` interacts with NS's finite-iteration conditioning.

### So our hypotheses for KeelHaul's loss lag (for you to rank / refute)
1. **The high `nrm` is a red herring for the body.** It reflects raw-gradient scale, which Muon
   discards; the body update is fine. The loss lag has a *different* cause (data/LR schedule/batch),
   and "reduce the norm" is treating a symptom that isn't coupled to the body learning.
2. **The clip is silently throttling the Adam-group params** (embeddings/head/router), because the
   global clip fires ~77–100% of steps and uniformly shrinks *their* gradients — and those DO use
   magnitude. If so, the harm is to the embedding/head learning rate, not the Muon body, and the
   fix is to **exclude Muon params from the norm/clip** (or clip per-group), not to lower body norm.
3. **NS finite-iteration conditioning is `nrm`-sensitive**: with only `ns_steps` iterations,
   orthogonalization of a poorly-scaled gradient may be imperfect, so `nrm` *does* leak into the
   update direction. If so, pre-NS normalization (not clipping) would help.
4. **It's genuinely a magnitude problem** and we're wrong about (1) — in which case we need to know
   which stage reintroduces magnitude sensitivity.

---

## Part 3 — what we're considering doing (rank these too)
- **Raise / remove the clip** (GPM says the gradients are productive; if the clip is a near-no-op on
  the Muon body but throttles Adam params, raising it can only help). LIVE, cheap.
- **Per-group clip** (clip Adam params, leave Muon unclipped since NS handles it). Restart.
- **Bigger batch** (lowers gradient variance → fewer spikes → less clip firing → also helps Adam
  params). In flight already.
- **Body-only WD** (the original taming dial) — but only if the norm is actually coupled to learning.

## The questions, concretely
1. Confirm/refute: **for the all-Muon body, uniform pre-NS gradient clipping is ~a no-op** (NS
   scale-invariance), so `nrm`-vs-clip dynamics don't govern body step size. Is that right?
2. Is hypothesis **#2 (clip throttling the Adam-group params via the GLOBAL norm)** the most likely
   real harm? Should we **split the clip** (Muon excluded / per-group)?
3. Does NorMuon's renorm-to-pre-norm (L202) plus fixed rms_scale really make the body update
   **magnitude state-independent**, or is there a path where `nrm` leaks into the body step?
4. Given all this — **is "reduce the norm" even the right objective**, or should we reframe to
   "stop the global clip from throttling the magnitude-sensitive (Adam) params"?

## Reference (source-grounded, all in this repo)
- `common_fsdp2/muon_fsdp2.py`: projection block L370–399; `apply_scaling` L119–127; `apply_normuon`
  L188–203; group defaults L569–580.
- `mara_fsdp2/train_mara.py`: `_clip_grad_norm_mixed_mesh` L1041–1075 (what `nrm` is); clip call
  L1792.
- Config: `configs/keelhaul.yaml` (clip_standard 1.0, tangent_project true, preserve_norm **false**,
  weight_decay 0.002).
- Background: `docs/WD_WASTE_ANALYSIS.md`, `docs/MATH_AGENT_BRIEF_3_newton_schulz.md`, `docs/GPM.md`
  (KeelHaul GPM ~+0.27, productive gradients → high-quality norm).

---

# MATH AGENT RESPONSE (2026-06-24) — resolved

**Verdict:** Tangent-projection audit clean (confirmed ‖U⊥‖=‖U‖√(1−cos²)≈0.999915·‖U‖, ~0.0085%
removed; preserve_norm off → microscopically norm-REDUCING; body fix exonerated; projecting AFTER
NS+scaling+NorMuon is correct because the radial component is created by the optimizer-transformed
update, not the raw CE gradient).

**Core correction to our brief:** "High nrm is the wrong target" is right, BUT not "clipping is
irrelevant to the body." Precise statement:
> Pre-NS clipping does NOT control body update **magnitude** (NS scale-invariant: polar(cG)=polar(G);
> fixed apply_scaling; NorMuon renorms to pre-norm), BUT it can alter body update **direction through
> WARM MOMENTUM**: M_t = β·M_{t−1} + (1−β)·c_t·G_t. Spiky c_t (which we have) changes the
> current-grad-vs-stale-buffer mix step to step → can make Muon more stale. Constant c forever ≈
> harmless; VARIABLE c on hard batches is the issue.

**Ranked hypotheses (his):**
1. High nrm is a red herring for the body — **mostly true** (raw nrm is upstream of the Muon
   transform; right body metrics are ‖ΔW_post-NorMuon‖, ‖ΔW‖/‖W‖, cos(ΔW,W), cos(ΔW_clip,ΔW_unclip)).
2. **Global clip throttling Adam groups — LEADING / most likely real harm.** Failure mode:
   large Muon raw grad ⇒ small global clip coef ⇒ Adam head/emb/norm grads scaled down ⇒
   magnitude-sensitive params learn too slowly on hard/productive batches. (Adam's steady-state
   scale-invariance does NOT save it: variable clipping perturbs m, v, ε regime, bias-correction
   transient, rare-row learning, update timing.)
3. NS finite-iteration nrm-sensitivity — possible, lower probability (easy to test via scalar sweep).
4. Genuine body magnitude problem — **low probability** (projection not inflating; Muon discards
   magnitude; body WD is NOT the right knob).

**The objective (reframed):** NOT "reduce nrm." Instead → "prevent global raw-norm clipping from
distorting the parameters that actually use magnitude," and secondarily "verify whether clipping
changes Muon direction through momentum."

## Action plan he handed us (read-only diagnosis → then act)

**Probe A — clip-replay (one real step, before optimizer.step):** for c ∈ {1.0, 0.75, 0.5, 0.25,
0.1}, push c·G through the FULL transform (momentum → NS → apply_scaling → NorMuon → tangent proj).
- Body matrices: measure cos(U_c, U_1), ‖U_c‖/‖U_1‖, cos(U_c, W).
- Adam groups: ‖Δ_c‖/‖Δ_1‖ and cosine to unclipped.
- **Run BOTH cold and with warm momentum** (the cold/warm gap IS the momentum-staleness test).
- Closes the diagnosis: if Muon update barely changes but Adam update shrinks materially → confirmed.

**Probe B — groupwise norm telemetry (every step):** log ‖g_Muon_body‖, ‖g_Adam_head‖, ‖g_emb‖,
‖g_norm/router/small‖, and the global clip coefficient. Find WHO sets the clip and WHO is harmed.
Smoking gun: ‖g_body‖ ≫ ‖g_Adam‖ while Adam update RMS drops whenever clip coef drops.

**Then (decision tree):**
- Muon ≈ invariant but Adam clipped → **raise clip live 1.0→2.0 (or 3.0)** (cheap test of H2; barely
  moves body, restores Adam signal on hard batches) → then implement **per-group clipping** (Muon
  body unclipped or high safety clip for NaN/Inf only; Adam groups separate clip ~1.0).
- Muon direction ALSO changes under clipping → global clip is making body momentum stale too →
  per-group / no-Muon-clip even more compelling.
- Neither changes much → high nrm is NOT the loss-lag cause → look at LR schedule, data mix, batch,
  head-LR dampener, SCS/no-SCS, or normal dashboard variance.
- **Do NOT:** touch tangent projection, add body WD, or chase raw nrm down. Bigger batch is
  directionally aligned (lowers c_t variance → helps both Adam throttling and momentum staleness).

**Status:** awaiting Josef's return to (1) wire Probe A + Probe B, (2) likely raise clip live as the
H2 test. Both probes are read-only / low-risk; the clip raise is a one-line config change.
