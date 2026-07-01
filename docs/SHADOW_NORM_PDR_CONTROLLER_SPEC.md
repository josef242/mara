# Shadow-Norm Progressive PDR Controller — Implementation Spec
### (`reference.mode: auto_shadow_growth` — encodes Math's Q12 ruling)

**Status:** engineering spec, **review-hardened + MATH-SIGNED-OFF (v3)**. Cleared for implementation. Encodes Math's Q12 ruling (shadow-norm reference, progressive engagement) and Josef's goal: **replace fitted magic numbers with a constructed online reference.** 2026-06-29.

> **v3 changelog (Math sign-off folded in).** Math approved `auto_shadow_growth` with: aggregation = **median in log form** `m = exp(median_i[log(R_i/S_i)])`, no per-matrix multipliers (§2.3); **terminal f<1 = FATAL by default**, partial-f only via a separate explicit `auto_shadow_partial` mode (§8); `r_freeze` captures the **post-guardrail commanded m** (§2.6); **WD shrink included in S** `S_i ← S_i + ΔR_free,i − η·λ_body·S_i` (§2.3 — *material for our recipe: the WD taper overlaps the ramp at wd≈0.02 ⇒ ~10–20% of radial growth*); add **subgroup telemetry** w1/w3 vs w2 (§6); negative-ΔR_free outliers clipped/smoothed but not zero-clamped (§5); "unscaled η" = non-controller effective LR (excludes m, includes any other fixed lr scales) (§2.2). All prior v2 fixes retained below.
>
> **v4 (Math — WD radial-budget law + partial-f).** Body WD retires its hand-taper for a **measured radial-budget law** `λ_body = clamp(λ_min, λ_max, ρ(1−f)γ_EMA)` (§2.7) — `γ=−_dot/_wsq` is already computed; folds into the controller as a 2nd output (`wd_overrides`), `m` cancels so no `1/m` comp. **`auto_shadow_partial`** mode added for first-class terminal f<1 (stays in ramp-law, no handoff; §3/§8). `radial_stats` simplified to `(‖W‖, γ)`. Shadow `S` uses `λ_S=ρ·γ` (f=0); optimizer uses `λ_body=ρ(1−f)γ` (Math-confirmed — the gap is 1–3% of m, not cosmetic). **All design questions closed — spec is implementation-final.**

**Provenance:** Q11 (`docs/MATH_AGENT_Q11_self_anchoring_pdr_controller.md`, BLESSED) → Q12 problem (`docs/MATH_AGENT_Q12_partial_f_progressive_engagement.md`) → Math's ruling (shadow-norm) → this spec.

> **v2 changelog (adversarial review wf wmd42df0x, 3 lenses vs live code).** Math verdict **sound** (all flags doc-only). Implementation verdict was `revise_major`; these are now folded in:
> - **B1** `radial_stats` must be *threaded into the Work object* (it's a no-op on the optimizer) — §7.1.
> - **B2** ΔR_free must be *accumulated every step* (the reader runs at cadence) — §7.2.
> - **B3** persistent `S` must be *keyed by param NAME*, not `id()` (id changes on resume) — §8.
> - Gate to a **DTensor body** (single-device path has no tangent block) — §7.1.
> - `.item()` **`group['lr']`** (0-dim tensor) — §7.1.
> - Control-law seam, level-triggered handoff, unit-correct v2→v3 migration, post-freeze missing-state FATAL — §3/§7.2/§8.
> Open questions Q1/Q3/Q4 were **resolved by the reviewers** (§10).

---

## 1. The idea in one paragraph
Tangent projection at strength `f` removes the f-fraction of the body's radial (norm-growing) update, which is what provides the body's self-anneal (pdr = ‖ΔW‖/‖W‖ shrinks as ‖W‖ grows). The controller's job is to **replace the removed anneal with an explicit LR cut, continuously, as it is removed.** Maintain a **shadow norm** `S` = what ‖W‖ *would* be under free growth (no projection); the actual norm is `R = ‖W‖`. The multiplier that makes the actual pdr match the free-growth pdr is, by pdr geometry, simply `m = R/S` — a **constructed** reference needing no `anchor_step`, `anchor_samples`, `warmup_step`, or fitted glide.

---

## 2. The math

### 2.1 Per-step quantities (already computed in `muon_fsdp2.py:411-426`)
The tangent block already computes, globally all-reduced over FSDP shards: `_dot = ⟨U,W⟩`, `_wsq = ‖W‖²`, and the raw radial coefficient `c = _dot/_wsq` (it forms `_c = c·f`). So `‖W‖ = √_wsq`, `f = _strength`, `η = group['lr']` are all in hand. **No new reduction.**

### 2.2 Free-growth (counterfactual) norm increment
The change in ‖W‖ from the radial part of the **free** (un-projected, m=1) update is, to first order:
$$ \Delta R_{\text{free}} = -\,\eta\,\frac{\langle U,W\rangle}{\lVert W\rVert} = -\,\eta\,\frac{\_dot}{\sqrt{\_wsq}}. $$
`_dot < 0` (NS leaves a consistent anti-radial component) ⇒ `ΔR_free > 0`. Uses the **raw `_dot`** and the **non-controller effective LR `η`** — the m=1, f=0 counterfactual rate. *(Math: "unscaled η" = the effective LR **excluding the controller multiplier m**, but **including** any other fixed per-param LR scales. Today `group['lr']` is the only non-controller scale, so `η = group['lr']` is correct; if future code adds fixed per-param scales, include them but never m.)* *(Reviewer note: `−η·_dot/‖W‖ = −η·cos·‖U‖`, so the increment is essentially independent of which body norm it is evaluated at — see §10 Q1.)*

### 2.3 Shadow norm + commanded multiplier (per matrix `i`)
- While `f = 0`: `S_i = R_i` (no correction, m=1).
- **At the first observe with `f > 0`:** snapshot `S_i = R_i` for **all** actuated matrices (eager, not lazy — §7.2).
- Each step thereafter (**accumulated every step**, §7.2/B2): `R_i = ‖W_i‖` (latest), and
  $$ S_i \leftarrow S_i + \Delta R_{\text{free},i} - \eta\,\lambda_S\,S_i. $$
  The **`−η·λ_S·S` term is the counterfactual WD shrink** (Math): S is the body under *no projection, m=1*, so it decays at the **m=1** rate (NOT `η·m·λ`). **Two WD values (Math-confirmed):** `λ_body_actual = clamp(λ_min,λ_max, ρ(1−f)γ_EMA)` drives the **optimizer**; `λ_S = clamp(λ_min,λ_max, ρ·γ_EMA)` (the **f=0** value, ≈λ_max early) drives **S** — because S is the f=0 counterfactual. Using the tapering `λ_body` in S would make S grow too fast → R/S too small → **over-cool**; the gap compounds to **1–3% of S/m during a high-WD ramp** (not cosmetic). *Always include the S term; it self-zeroes when λ sits at the floor.*
- **Command (Math — log-median):** `m = exp( median_{i: S_i>0}[ log(R_i/S_i) ] )` (the multiplicatively-natural form; ≈ plain median of ratios near 1). If no qualifying matrix this cadence, **hold m (stale)**. Then rate-limit + f-aware floor + clamp (§5). Median (not mean): robust to one bad/missing matrix, keeps a single FFN multiplier, avoids a hidden per-matrix LR optimizer.

### 2.4 Why `m = R/S` is the right correction
`pdr_actual = η·m·N / R` and `pdr_free = η·N / S` (N = ‖U_proj‖). Setting them equal ⇒ `m = R/S` — the missing-denominator correction. *(Math caveat — the one elevated: the WD step `−η·m·wd·W` is radial; the right yardstick is the **radial norm-growth** `|cos|·pdr ≈ 0.013·pdr`, NOT total pdr. At `wd=0.002`, `η·wd ≈ 6e-7` is ~1–2% of radial growth → ignore in S. At `wd=0.02`, `η·wd ≈ 6e-6` is ~10–20% → NOT negligible. Handled by the WD term in the §2.3 shadow update; see the recipe-overlap note there.)*

### 2.5 Stability (confirm no runaway — reviewer re-derived & confirmed)
With `ρ ≡ R/S = m`, free fractional rate `γ ≡ -η·_dot/_wsq > 0`, and `ΔR_free = γ·R`:
- `dR = γ(1−f)·m·R`,  `dS = γ·R`  ⇒  **`dρ = −γ f ρ²`.**

So `m` **decreases monotonically and *deceleratingly*** during the ramp (∝ −ρ², self-limiting), faster as `f` grows. Cannot run away; the f-aware floor (§5) bounds it. *(Caveat: this is first-order; at f=1 a second-order tangential term still grows ‖W‖ at ~O(pdr²) — reviewer measured +1.3e-5/step — negligible per-step and harmless because `r_freeze` latches once, §2.6.)*

### 2.6 Handoff at f=1 (the automatic, magic-number-free "anchor")
**Level-triggered & idempotent** (mirrors v2's `anchor_set` latch; `interpolate_lr_mod` flat-extrapolates so `f` pins at 1.0 forever past the last knot → the level test is resume-safe and cannot double-capture):
```
if f_now >= 1 - 1e-6 and not self.frozen:
    r_freeze  = K_ema * m_cmd          # m_cmd = the POST-guardrail commanded m (after rate-limit/floor/clamp)
    lr_freeze = scheduled_lr           #         NOT raw median(R/S) — see note. r_freeze is a TARGET PDR.
    self.frozen = True                 # checkpointed (§8)
```
"reaches f=1" means **"first observe at which `f_now ≥ 1−1e-6`"**, NOT a per-step edge. Then ride the validated Q11 frozen-body tail:
$$ r(t) = r_{\text{freeze}}\cdot \frac{\eta(t)}{\eta_{\text{freeze}}},\qquad m = r(t)/K_{\text{ema}}. $$
**Continuity requirement (reviewer + Math):** `K_ema` must be the *single running EMA used on both sides of the seam* and **not reset/re-windowed at f=1**. Because the ramp is pure feedforward with `r ≡ K_ema·m_cmd` (so `m = r/K_ema = m_cmd` exactly, K_ema cancels), capturing `r_freeze` with that same K_ema gives exact m-continuity at the seam. **Math refinement:** capture `r_freeze` from the **post-guardrail commanded `m_cmd`** (what the optimizer actually applied), not raw `median(R/S)` — if a safety rail (floor/slew/clamp) ever binds at the seam, the raw value would create a discontinuity; the commanded value never does. **No `anchor_step`, `anchor_samples`, geomean window, or sanity bands.**

---

### 2.7 Body WD — radial-budget law (Math; replaces the hand-timed taper)
**WD is no longer a scheduled knot — it is a measured radial-budget term.** Keep WD only to the extent there is live radial growth to spend it against; as projection removes the radial channel, WD auto-tapers to a floor.
$$ \lambda_{\text{body}}(t) = \mathrm{clamp}\big(\lambda_{\min},\ \lambda_{\max},\ \rho\,(1-f_t)\,\gamma_{\text{EMA}}\big),\quad \gamma_i=\max\!\Big(0,\tfrac{-\langle U_i,W_i\rangle}{\lVert W_i\rVert^2}\Big),\quad \gamma_{\text{EMA}}=\mathrm{EMA}\big(\mathrm{median}_i\,\gamma_i\big). $$
`γ` = measured free radial growth rate per unit LR = **`−c_raw = −_dot/_wsq`, already computed at `muon_fsdp2.py:426`.** Defaults: **`λ_max=0.02, λ_min=0.002, ρ=0.20`** ("WD may consume ≤20% of remaining free radial growth" — a dimensionless budget, not a timing knot).

- **Why clean (Math):** residual growth `η·m·(1−f)·γ·R` vs WD shrink `η·m·λ·R` → ratio `λ/((1−f)γ)`; **η, m, R all cancel.** So `λ=ρ(1−f)γ` removes a controlled fraction ρ of the *remaining* radial growth in both full-clamp and partial-f. The `m`-cancellation means Muon's existing coupling `W ← W·(1−η·m·λ)` (`:470`) needs **no `1/m` compensation**.
- **Emergent taper (no knots):** at our scale (`γ≈0.11`), `ρ·γ≈0.022` clips to `λ_max` early; as f rises → `0.011 @f=.5 → 0.0022 @f=.9 → λ_min @f=1`. The old hand-taper *emerges from the measured budget*; the `λ_max` ceiling encodes the "high early WD is real regularization" prior, used when the budget supports it.
- **WD is NOT a pdr actuator (Math):** the shadow controller (`m`) owns pdr/denominator coordination. WD changes R and could even *raise* `‖ΔW‖/‖W‖`; two actuators on R/S create hidden feedback. WD = bounded radial regularization only.
- **Partial-f-safe by construction:** WD stays ≤ρ of residual growth, so it never balances away the `(1−f)` growth a partial-f run deliberately preserves.
- **Shadow WD (Math-confirmed):** the optimizer uses `λ_body_actual=ρ(1−f)γ`; the shadow `S` uses `λ_S=ρ·γ` (f=0) — see §2.3 (1–3% bias if confused, not cosmetic).
- **Telemetry:** `γ_EMA`, `λ_raw=ρ(1−f)γ`, `λ_body`, `λ_S`, the key ratio `λ/((1−f)γ)` (≈ρ when active; floor-dominated ⇒ WD is now a small residual regularizer), the cumulative shadow-drift `Σ η(λ_S−λ_body)` (how much S would have drifted under the wrong WD), floor/ceiling flags.
- **Plumbing:** `radial_stats` carries **`(‖W‖, γ)`** per matrix (simplifies the earlier 3-tuple — the controller derives `ΔR_free=η·γ·R` and owns `λ`). The controller computes `λ_body` and writes the **existing `wd_overrides` side-dict** (Muon reads it at `:457`) — one-step-delayed, exactly like `m`. → **unified body controller:** `m`→`lr_scale_overrides`, `λ`→`wd_overrides`.
- **Fallback** (if too much for v1): `λ(f)=λ_min+(λ_max−λ_min)(1−f)^p`, `p∈[1,2]` — a pure f-schedule (Math's "simpler approximation"). The measured law is the cleaner automatic version.

## 3. The two phases (state machine)
| phase | condition | command | shadow norm |
|-------|-----------|---------|-------------|
| **idle** | `f = 0` | `m = 1` | `S_i = R_i` (tracks) |
| **ramp** | `0 < f < 1` | `m_raw = median(R_i/S_i)` → rate-limit → f-floor → clamp | `S_i += ΔR_free,i` every step |
| **frozen** | `f ≥ 1` (after latch) | `m = r(t)/K_ema`, `r(t)=r_freeze·η(t)/η_freeze` | frozen for control (telemetry may continue) |
| **partial-terminal** | `f` plateaus `<1` *(only in `auto_shadow_partial`)* | **stays in ramp law** `m = median(R_i/S_i)` forever — **no `r_freeze`, no LR-track tail** | `S_i += ΔR_free,i` continues |

In **`auto_shadow_growth`** (default) the schedule must reach `f=1` → the `frozen` row. The **`partial-terminal`** row is the separate **`auto_shadow_partial`** mode (Math): the controller never hands off — it stays in the shadow-ramp law as f sits below 1, deliberately leaving `(1−f)` of the radial growth alive (the radial-budget WD law §2.7 keeps WD from cancelling it). Dynamics: `dρ=−γfρ²` at constant f ⇒ `m` self-limits as `~1/(1+γf·t)`.

**Control-law seam (reviewer + Math — do NOT double-apply K_ema):** in the ramp branch compute `m_raw = exp(median_i[log(R_i/S_i)])` **directly**, feed it through the *existing* rate-limit (`body_lr_controller.py:375`) + f-aware floor + clamp (`:377`) to get `m_cmd`, and **skip the `m_ff = r/K_ema` inversion** (`:370`). `K_ema` is computed throughout for **telemetry only** (logged `r = K_ema·m_cmd`) and for the frozen-phase inversion + the `r_freeze` capture (§2.6). PI trim stays **off** (Math Q11) — pure feedforward in both phases.

---

## 4. The magic-number ledger (Josef's win, concretely)
**Killed:** `reference.anchor_step` (+ the `anchor_step: auto` band-aid), `reference.anchor_samples`, `warmup_step`, `anchor_warn_band`/`anchor_fatal_band`/`anchor_abs_warn`, DN2 `knots`.
**Survives as interpretable safety/smoothing:** `m_max: 1.0` (never amplify, fatal-guarded); `m_min_full: 0.20` with a formulaic floor `m_min(f) = 1 − f·(1−m_min_full)`; `rate_down`/`rate_up`; one `k_ema_alpha`.
**Single source of truth:** the only schedule the controller reads is `tangent_project_strength` (the body recipe).

---

## 5. f-aware guardrails (control law stays simple; safety is f-shaped)
Do **not** multiply the core command by `f` (under-corrects, systematically late — Math). Instead:
- **f-aware floor:** `m_min(f) = 1 − f·(1−m_min_full)`.
- **Asymmetric slew:** keep `m·(1−rate_down) ≤ m_cmd ≤ m·(1+rate_up)`. *(Reviewer: at realistic γ the per-cadence target drop is 0.06–0.65% ≪ a 5–8% `rate_down`, and the floor target stays above `m_min(f)` at all f — so **neither binds**; both are pure safety rails at current pdr/ramp speeds. Revisit only if body pdr or ramp speed is ~10× larger.)*
- **Robust ΔR_free (Math):** cadence-accumulate (§7.2). Do **not** zero-clamp all negative increments — genuine local inward radial updates exist — but **robustly clip/smooth large negative outliers** (a single spike would corrupt the S integral).

---

## 6. Telemetry & alarms
- Log per cadence: `f`, `m_cmd`, `R`, `S`, `R/S`, `m_min(f)`, `K_ema`, `r = K_ema·m_cmd`, `pdr_ffn`.
- **Subgroup telemetry (Math):** log `median(R/S)` **separately for w1/w3 vs w2**. If they diverge persistently, the next escalation is a **subgroup controller** (one m for w1/w3, one for w2) — **not** per-matrix multipliers. (Default stays a single FFN m until telemetry forces it.)
- **Cumulative angle as a DIAGNOSTIC, not an actuator** (Math — elevated): track `Θ_actual = Σ pdr_actual`, `Θ_ref = Σ pdr_ref`. This is the key monitor for the one unavoidable approximation — `U` and `cos(U,W)` come from the *controlled actual* trajectory, not the exact unprojected counterfactual (no twin run). If `Θ_actual − Θ_ref` grows during the ramp, the shadow estimate or guardrails are too weak — **alarm only**. No cumulative-payback term (that over-cools after the instantaneous trajectory is healthy — the late-hammer pathology Q12 demonstrated).
- **Lower rail** (real hot-body): `m` at `m_min(f)` yet `pdr > 1.1·r` → keep Q11's consecutive-sample alarm.
- **Hidden bad base LR:** `m ≪ 0.5` in the first half of the ramp without excellent pdr/loss → flag.

---

## 7. Plumbing (review-hardened)

### 7.1 Produce `radial_stats` — thread it into the Work (fixes B1)
The tangent block runs inside `Fsdp1dWork.finish`, where `self` is the **Work**, not the optimizer. So thread a `radial_stats` dict the *same way* as `wd_overrides`/`lr_scale_overrides`:
1. `Muon.__init__` (next to `wd_overrides`, ~`muon_fsdp2.py:638`): `self.radial_stats = {}`.
2. `Fsdp1dWork.__init__` (~`:280`): add `radial_stats` param, `self.radial_stats = radial_stats`.
3. Work construction (~`:696`): pass `self.radial_stats` in the `class_work(...)` call.
4. In the tangent block (~`:426`, where `_dot`/`_wsq` are floats from `.item()`):
   ```python
   rs = getattr(self, "radial_stats", None)
   if rs is not None and _wsq > 0:
       _wn  = _wsq ** 0.5
       _lr  = self.group["lr"]; _lr = _lr.item() if hasattr(_lr, "item") else float(_lr)  # 0-dim tensor!
       _wd  = self.wd_overrides.get(id(self.param), self.group["weight_decay"])           # non-controller body WD λ
       rs[id(self.param)] = (_wn, -_lr * _dot / _wn, _lr * _wd)   # (‖W‖, ΔR_free, η·λ_body) — all at m=1, f=0
   ```
   - `group['lr']` is a **0-dim torch.Tensor** (`train_mara.py:2010` sets `param_group['lr']=torch.tensor(...)`); `.item()` it or it contaminates the float median + serialization.
   - **DTensor-only gate:** `SingelDeviceWork` has **no tangent block** (and calls an undefined `muon_update()` — broken/unused). So `radial_stats` is produced **only** on the `Fsdp1dWork` path — which is the path **all** FFN body matrices take under FSDP2. The controller MUST therefore **assert a DTensor/FSDP body at wire-time** and refuse single-device (`isinstance(p, DTensor)`); do not claim single-device support.
   - `radial_stats` is **transient** (rebuilt each step, never checkpointed), so it may stay **id-keyed**.
   - **Producer gaps to tolerate:** a matrix frozen via `lr_scale==0` short-circuits the whole pipeline (`:684`, no tangent block) and `_wsq==0` is guarded (`:419`) — either yields **no entry** that step. The consumer must use `rs.get(id)` and skip/hold, never KeyError.

### 7.2 Accumulate every step + own the shadow norm (fixes B2)
The reader `observe()` runs only at val_step cadence (`train_mara.py:2994`), but ΔR_free must be summed **every step**. Add a per-step accumulation site in the train loop **after `optimizer.step()`** (~`:2450`), before the next step overwrites `radial_stats`:
```python
for pid in _ffn_param_ids_for_ctrl:
    e = optimizer.radial_stats.get(pid)
    if e is not None:                                  # tolerate missing (frozen / _wsq==0) — §7.1
        R_latest[pid] = e[0]
        dR_accum[pid] = dR_accum.get(pid, 0.0) + e[1]  # Σ ΔR_free over the window
        wd_accum[pid] = wd_accum.get(pid, 0.0) + e[2]  # Σ η·λ_body over the window
```
At cadence, pass `R_latest` + `dR_accum` + `wd_accum` into `observe()`; the controller does `S_name ← S_name + dR_accum − S_name·wd_accum` (first-order WD shrink, §2.3) and `R_name = R_latest`, then resets `dR_accum`/`wd_accum`. Keep **all persistent controller state inside the controller** (option (b)) so checkpoint locality holds. New `observe()` branch `ref_mode == "auto_shadow_growth"` implements idle/ramp/frozen (§3); it reuses the existing rate-limit/clamp/alarm tail (`:374-409`) but **injects `m_raw = median(R/S)`** rather than the K_ema inversion.

### 7.3 Wiring that already exists (reuse)
- Per-step `f`: `train_mara.py:2023` (`_pg['tangent_project_strength']=_f_now`); pass the same `_f_now` to `observe(... f_now=_f_now)` (already plumbed).
- Actuation: `lr_scale_overrides[id]=current_multiplier()` over `_ffn_param_ids_for_ctrl` (`:2133`). Unchanged — rides the validated lr_scale path (`muon_fsdp2.py:455-472`).
- Median pdr + observe call: `:2961/2994`.

---

## 8. Checkpoint / resume (review-hardened)
- **`state_dict v3`** adds: `S` (**dict keyed by param NAME**, not `id()` — see below), `shadow_active` (f-onset latched), `r_freeze`, `lr_freeze`, `frozen` (bool). Removes the anchor buffer + sanity-band baseline. **Keeps `logK` (K_ema)** — the v3 tail inversion needs it on the first post-resume step.
- **B3 — stable keying.** `id(param)` is a process-local address (the actuated set is itself rebuilt each run at `:1508-1513`), so an id-keyed S is empty after resume. **Key persistent `S` by param NAME** (`'layers.{i}.feed_forward.w{1,2,3}.weight'`, `_orig_mod.`-stripped as at `:1512`). Build an `id↔name` bridge once at wire-time from `named_parameters()`; translate before storing into S; serialize name-keyed; re-resolve `name→current id()` on load. (Transient `radial_stats` stays id-keyed.)
- **Global, not rank-0.** `S` is a **GLOBAL (all-reduced) scalar per matrix** — identical on every rank because `_dot/_wsq` are all-reduced (`:417`); saved in the `bodylr_state_*.pt` file and loaded on all ranks (`:3726`). Implementations MUST derive S **only** from the all-reduced `radial_stats`, never a local norm (preserves the cross-rank bit-identity `observe()` requires, `:2976-2978`).
- **v2→v3 migration (unit-correct).** v2 `K_anchor` is a **plant gain** (pdr/m); v3 `r_freeze` is a **target pdr**. They coincide *only* because v2 holds `m=1` during anchoring: `r_freeze = K_anchor · m_at_freeze = K_anchor` iff `m==1`. So: **assert `m_at_freeze==1.0`** (refuse the silent map otherwise), set `lr_freeze=lr_anchor`, and **carry `logK` over** unchanged. Pre-freeze v2 resume: re-init `S=R` at the resume step (re-converges).
- **Resume guards (`auto_shadow_growth`).** Pre-freeze: missing state is a benign **warn** (reconstructed from live `R/S`). **Post-freeze: missing state is FATAL** (analogous to `:3748-3760`) — otherwise the pre-freeze fallback would silently re-enter the ramp and **re-capture `r_freeze` on an already-frozen body**, rebasing the run. Gate the fatal on `resume_step ≥ the f=1 crossing step`. The only other guard is the existing frozen-phase **lr-fingerprint** check (the tail rides `η(t)`).
- **Resume guard (`auto_shadow_partial`, Math).** There may be **no** frozen phase, but once `f>0` the shadow `S` carries the entire accumulated history. So in partial mode, **missing `S` when `f>0` is FATAL** (a silent `S←R` reset would zero the integral → `m→1`, erasing all controller history). Pre-onset (`f=0`) missing state stays a benign warn.
- **Terminal `f < 1` policy — RESOLVED (Math): FATAL by default.** `auto_shadow_growth` **requires `f→1` + the frozen handoff** (the grow-then-clamp recipe). Wire an explicit f==1-terminal fatal for this mode (extend/reuse `:5028-5033`). Indefinite partial-f cooling is a *different regime* (partial projection + partial shadow cooling forever; never enters the LR-track tail; S keeps drifting while m crawls toward the floor) — exposed **only** via a separate explicit mode `reference.mode: auto_shadow_partial` + `allow_terminal_partial_f: true`, treated as its own experiment with its own alarms. Do not silently permit it in `auto_shadow_growth`.

---

## 9. Implementation plan (ordered, each independently testable)
1. **Produce `radial_stats`** — thread into `Fsdp1dWork`/`Muon.__init__`/`:696`, write with `.item()`'d lr + the `η·λ` WD factor, DTensor-gate the mode (§7.1, fixes B1 + lr-tensor + single-device). **Key test (Math — *the* test for sign / LR-scale / `_dot` timing):** at f=0, `ΔR_free` must match the measured `Δ‖W‖` per step **up to the WD and second-order tangential terms** (set the tolerance to cover `η·λ·‖W‖` + `O(pdr²)·‖W‖`). Also assert single-device wiring raises the gate error.
2. **Per-step accumulator** in the train loop after `optimizer.step()` (§7.2, fixes B2). Test: over N steps, `Σ dR_accum` matches the sum of per-step increments (no 1/cadence loss).
3. **Controller `auto_shadow_growth` mode** — idle/ramp/frozen, `m=median(R/S)` direct, f-aware floor, **name-keyed S**, level-triggered handoff, v3 state. Torch-free unit tests (like `test_body_lr_controller_auto.py`): synthetic R/S streams → m tracks R/S; floor schedule; handoff m-continuity; stability (m monotone-decreasing); **v3 round-trip + name-keyed resome survives a simulated id change**; v2→v3 migration (assert m==1, carries logK); missing-matrix tolerance.
4. **Settings validation** — require `tangent_project` on + DTensor body; reject `anchor_step`/`anchor_samples`/`warmup_step` as stale (helpful message); wire the terminal-f<1 decision.
5. **Wire into train loop** (accumulator + observe branch + name bridge).
6. **Rig smoke** (resume, short): confirm `R/S`, `m` track and the [body-pdr] ffn line cools onto the **extrapolated free-growth glide** (compare vs an uncontrolled f-ramp twin — should sit on the glide, not run hot).
7. **dn4:** swap the controller block to `auto_shadow_growth`; drop `anchor_step`, `warmup_step`.

---

## 10. Questions — ALL RESOLVED (review + Math sign-off)
- **Q1 (shadow grows on R vs S):** *either acceptable* — `−η·_dot/‖W‖ = −η·cos·‖U‖`, so ‖W‖ cancels; second-order. Evaluate on live R.
- **Q2 (aggregation):** **Math: `median` in log form** `m = exp(median_i[log(R_i/S_i)])`; no per-matrix multipliers; add w1/w3-vs-w2 subgroup telemetry, escalate to *subgroup* (not per-matrix) control only if it forces it (§2.3/§6).
- **Q3 (handoff continuity / K_ema step):** *no extra smoothing* — single running K_ema, never reset; ramp `r ≡ K_ema·m_cmd` so K_ema cancels; capture `r_freeze` from the post-guardrail `m_cmd` (§2.6).
- **Q4 (faster ramp `rate_down`?):** *unwarranted at current γ* — slew never binds; the asymmetric rail is still kept as safety. Revisit only if pdr/ramp ~10× faster.
- **Terminal f<1:** **Math: FATAL by default** in `auto_shadow_growth`; partial-f is first-class via a separate **`auto_shadow_partial`** mode — stays in ramp-law forever, no frozen handoff, missing-S-after-f>0 FATAL (§3/§8).
- **Body WD (Math — radial-budget law, §2.7):** retire the hand-taper; `λ_body = clamp(λ_min, λ_max, ρ(1−f)γ_EMA)` from the measured `γ=−_dot/_wsq`. `m` cancels (no `1/m` comp); partial-f-safe; folds into the controller as a second output (`wd_overrides`). **Two WD values (Math-confirmed):** optimizer uses `λ_body=ρ(1−f)γ`, shadow `S` uses `λ_S=ρ·γ` at f=0 — the gap is **1–3% of m during a high-WD ramp**, not cosmetic (§2.3). Monitor `λ/((1−f)γ)≈ρ`, `Σ η(λ_S−λ_body)`, `Θ_actual−Θ_ref` (§2.7/§6). **All design questions now closed.**

---

## 11. References
- Q11 `docs/MATH_AGENT_Q11_self_anchoring_pdr_controller.md`; Q12 `docs/MATH_AGENT_Q12_partial_f_progressive_engagement.md`; brief `docs/PDR_CONTROLLER_STATE_BRIEF.md`; review `tasks/wmd42df0x.output`.
- Code: tangent block `muon_fsdp2.py:411-426`; Work ctor `:280`/construction `:696`/freeze short-circuit `:684`; actuation `:455-472`; controller `body_lr_controller.py` (observe 290-410, common loop 358-409, state 453-499); train-loop f-push `train_mara.py:2023`, lr-tensor `:2010`, post-step `:~2450`, actuator `:2133`, observe `:2994`, side-dicts `:5975-5977`, resume `:3726`, post-freeze guard `:3748-3760`, terminal-f validator `:5028-5033`.
