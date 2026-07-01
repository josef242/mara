# MATH_AGENT_Q11 — Self-Anchoring PDR Controller (retire the hand-fit knots)

**Status:** ✅ BLESSED by Math (2026-06-28) → ✅ **IMPLEMENTED** (2026-06-28): self-anchored **LR-track** `auto` mode built in `body_lr_controller.py` + `train_mara.py`, 26/26 unit checks (`common_fsdp2/test_body_lr_controller_auto.py`), adversarially reviewed (dn3 knots path byte-identical; cross-rank determinism clean; 2 HIGH + 1 MED review findings fixed). Learned-glide remains a gated research arm (not built). Not yet committed / not yet run live.
**Thread:** body-ramp / tangent-projection / pdr-controller (BRIEF_6 "projection removes annealing", Q7 anneal-timing, BRIEF_7 pdr-PID, Q8 sim, Q9 gradnorm→1). Companion to the dn4 head-hygiene levers (Q10).
**Date:** 2026-06-28.

---

## ✅ Math ruling (2026-06-28) — BLESSED; build LR-track first

Math approved the design. Build the conservative self-anchored **LR-track** `auto` mode now. Rulings + refinements to fold into the build (these SUPERSEDE the body below where they differ):

1. **LR-track is the default.** `r(t) = K_anchor · lr(t)/lr_anchor`; the loop's only job is to cancel the slow K-drift (`m ≈ K_anchor/K(t)`), NOT to invent a second anneal. Math's pivot answer: `pdr_natural ≈ lr-track × slow K-drift` — option **(a)**. Learned-glide only "after LR-track proves too warm late."
2. **Feedforward-only first; PI present but disabled** (gains = 0).
3. **⚠️ CORRECTION — upper-rail semantics (our brief had this backwards).** `m_raw > 1` ⇒ `K(t) < r(t)` ⇒ the body at m=1 is **BELOW** the reference (cooler than target), **not hot**. Upper rail = **"reference unreachable / no upward authority"** (anchor too high, reference too gentle/high, base LR too low, negative K-drift, or the body simply cooling faster than asked). **Informational / warn — not fatal** unless the run also underfits. The **lower rail** (`m=m_floor` AND `pdr>1.1·r`) is the real **hot-body / insufficient-cooling-authority** alarm.
4. **Both rails, regime-aware.** Lower-rail: warn only if `pdr>1.1·r` at floor (NOT merely because m hit the floor). Upper-rail: warn if `m_ff_raw>1+margin` for N samples. Always log `m_ff_raw` (pre-clamp).
5. **Store `K_anchor = pdr_anchor / m_anchor`** (= pdr_anchor since m=1 at capture) so the form is robust if a future mode anchors at m≠1. Then `r(t)=K_anchor·lr(t)/lr_anchor`, `m_ff = r(t)/K_ema(t)` (clean plant inversion).
6. **m_floor → 0.20–0.25** (from 0.30) for long runs — under LR-track `m≈K_anchor/K`, so a 2–3× K-rise needs m down to ~0.33–0.5; 0.30 leaves little margin. FFN-only + cuts-only makes 0.20 safe.
7. **Learned-glide guard = bounded EXPONENT, not just the LR-track ceiling.** With `g=(lr/lr_anchor)^p`, a steeper glide already sits *below* the ceiling — so the ceiling does NOT bound it. Bound `p ∈ [1.0, 1.3]` (LR-track is p=1); fall back to LR-track on poor fit; no extrapolation past the fit window unless it reverts to LR-track.
8. **Anchor sanity bands: loose** (catch capture bugs, not enforce theory): warn outside `[0.6, 1.4]×`, fatal outside `[0.35, 2.0]×` the trailing pre-freeze pdr EMA; plus absolute warn if `pdr_anchor < 1e-3` or `> 5e-3`. Do NOT require `pdr_anchor ≤ last growing-body pdr` as a hard invariant.
9. **Confirmed:** geometric mean over 5–10 post-freeze samples; anchor only when `f=1` AND body WD-taper complete AND `m=1` AND ≥N post-freeze samples seen (decoupled from `warmup_step`); checkpoint anchor + lr-schedule fingerprint, fatal on post-anchor resume with missing anchor state; `m_max=1.0` hard (never amplify).

Math's recommended shipped `auto` config is in §7 (annotated with these refinements).

---

## 0. TL;DR — the one question that decides everything

We want to replace the controller's **hand-fit knots** (DN2's measured pdr glide + an offline-computed lr-track tail) with a reference the controller **discovers from the run itself**, so a fresh-from-scratch dn4 needs no twin run. The whole design pivots on a single physics question:

> **After the body is frozen by tangent projection (f→1: ‖W‖ fixed, radial update component removed), what does the frozen FFN body's NATURAL per-step pdr do as the cosine LR decays?**
>
> - **(a) It declines at exactly the LR rate** (pdr ∝ lr) → an **LR-track** reference `r(t) = pdr_anchor · lr(t)/lr_anchor` *is* the natural trajectory; commanded **m ≈ 1**, and the controller's only real job is cancelling the measured **K-drift** (+10–15%/300M tok).
> - **(b) It keeps cooling structurally** (steeper than lr) → we need a **learned-glide** reference (slope estimated from the run's own growth phase).
> - **(c) It cools more slowly than lr** → LR-track *over*-cools; we'd need a gentler-than-lr reference.

**Our lean: (a) LR-track as the shipped default**, learned-glide a gated opt-in research arm. We want Math to rule on (a)/(b)/(c) — it decides the default reference shape, **which rail (m_floor vs m_max) is the operative failure mode**, whether m_floor can bind, and whether the closed loop's job is *only* de-drift or also shape.

---

## 1. Why retire the knots

Josef's word for the knots is "a hack," and it's the right call. They require (1) a **twin run** to measure the target glide, (2) a **human** to fit them, and (3) they go **silently stale** if the LR schedule or model scale changes. And note we are *already* doing lr-track by hand: dn3's reference tail past DN2's data is the offline-computed `1.19e-3 · lr/lr_handoff`, i.e. lr-track for ~60% of the run. A fresh dn4 has **no twin**. Self-anchoring also **retires the dn4 fork-vs-fresh question** — with it, dn4 need not fork DN2 just to borrow its glide.

---

## 2. The plant (verified, code-grounded)

`pdr = ‖ΔW‖/‖W‖ = K·m`, with `K = lr·C/‖W‖`, where `C = ‖U‖_F` is the Newton-Schulz update norm — a **shape-only constant**, gradient-magnitude-independent (NS normalizes singular values to ≈1). On a **frozen** body ‖W‖ is constant, so **K ∝ lr** to first order, with a measured **second-order K-drift +10–15%/300M tok** (kv2, BRIEF_7) from evolving gradient structure. The controller drives measured pdr → reference `r(t)` via feedforward `m = r/K_ema`, asymmetric rate limits, clamp to `[m_floor=0.30, m_max=1.0]`. **m_max=1.0 is hard (fatal_error if >1): the controller can only CUT pdr, never amplify.**

---

## 3. The mechanism (self-anchoring)

- Controller **gated** (m=1) through the body's growth phase — body grows + self-anneals, no actuation against a moving plant.
- At the **freeze point** (f reaches 1.0) the controller **captures the body's own measured pdr** as `pdr_anchor` and `lr_anchor = lr` at that step.
- Thereafter `r(t) = pdr_anchor · g(t)`, and the existing closed loop drives measured pdr → r(t).
- `g(t) = lr(t)/lr_anchor` (LR-track, default) or a steeper **learned glide** (opt-in).

**Key algebra (the reassuring result):** under LR-track the reference equals the body's natural pdr (because natural pdr ∝ lr), so **commanded m ≈ 1** — the controller does *not* push against the LR; it only trims the slow K-drift. **This is the structural reason LR-track cannot fight the LR curve.**

---

## 4. "Must not fight the LR curve" — the analysis (Josef's hard requirement)

We adversarially enumerated every way the controller could beat against the cosine. **Two axes are structurally closed; three are residual risks** we're designing explicit guards for.

### Structurally closed (good)
1. **Amplification / upward fighting.** `m_max=1.0` + the fatal_error on `m_max>1` make it *physically impossible* to push body-LR above the scheduled cosine. The controller can only cut. ✅
2. **Beat / timing.** With an **lr-relative** reference, *both* the plant (K∝lr) and the reference (r∝lr) move with lr in lock-step, so the lr motion **cancels** in the log-error `e = log(r/pdr)` — the laggy ~100-step controller only ever sees the slow K-drift residual (300M-tok timescale). Measured cosine drop ≤0.22%/cadence sits far inside the 2%/5% rate-limit budget. Beat is **eliminated by construction**, not merely damped. ✅ *(Requires: compute the reference's lr factor from the SAME `get_lr(step)` the optimizer uses — single source of truth.)*

### Residual risks (need explicit guards)
3. **PEG-AT-m=1 "silently off" — the sharpest hole.** The *existing* alarm only watches the **m_floor** rail. If the natural ceiling `K·1` falls **below** the reference (negative K-drift, a too-high/too-gentle reference, or a bad anchor), the controller wants `m>1`, clamps to 1, and the body runs **below target — cooler than the reference — with ZERO alarm** (m=1 reads like "controller chose unity"). *(Per Math's correction this is "no upward authority / under-target," NOT a hot body — see the ruling box. It is informational unless the run also underfits.)* Simulated −12%/300M drift pegs m=1 for 100% of samples, silent. **`m_max=1.0` prevents the upward *mechanism* but MASKS a mis-specified reference.**
   → **MUST-HAVE guard:** a **symmetric upper-rail alarm** — warn if m is pegged at `m_max` while the *unclamped* demand `m_ff_raw > 1+margin` (equivalently measured `pdr < r/1.1`) for N samples; and **log `m_ff_raw` (pre-clamp) every cadence** so the controller's true intent is always visible even when clamped.
4. **Double-anneal (learned-glide only).** A `g(t)` steeper than lr-track forces persistent `m<1` — cutting *on top of* the cosine. This is the *intended* mechanism of learned-glide, so the line between "feature" and "fight" is **entirely the slope estimate**; **cuts-only does NOT protect** (the cut *is* the failure). Sim: `g=lr^1.3 → m~0.82 mid/0.50 late`; `g=lr^2` floors it.
   → **Guards:** clamp `g(t) ≤ lr-track ceiling` at all t (steeper-than-lr becomes bounded/opt-in, never a runaway); **bound the fitted exponent** (e.g. p∈[1.0,1.3]) + fit-quality gate + **auto-fallback to lr-track**; revert to lr-track outside the fit window (no flat-extrapolating a steep glide into the cosine tail); a **double-anneal detector** (alarm if `m < ~0.6` for N samples *while* pdr already tracks r).
5. **Anchor semantics (timing + noise).** "Anchor at warmup_step" **conflates engage-time with freeze-time** — in dn3, `warmup_step=7000` (f=0, body *still growing*) is **5000 steps before** f→1 at step 12000; anchoring there captures a *growing-body* pdr for a *frozen-body* loop (the physically wrong quantity, and it is the entire premise of the experiment that those differ). Also `param_delta_ratio` is a **single instantaneous 1-step** measurement (~10% noise); a single-draw anchor **linearly biases the whole run**.
   → **Guards:** **decouple `anchor_step` from `warmup_step`** — capture at the step where f first reaches 1.0, *after* the WD-taper also completes (both end at step 12000 in dn3); **capture as a geometric mean over ~5–10 post-freeze samples**; latch `lr_anchor` at the same step via the same `get_lr`.

---

## 5. Other design risks (with mitigations) — abbreviated

- **Wrong-anchor asymmetry (corrected per Math):** anchor too HIGH → reference unreachable, m pegged at 1, body runs at its natural *cooler-than-target* pace (upper rail = informational, no upward authority). Anchor too LOW → over-cutting toward m_floor; fires the lower-rail (real hot / insufficient-authority) alarm only if `pdr>1.1·r` at the floor. Add *loose* sanity bands on the captured anchor (warn `[0.6,1.4]×`, fatal `[0.35,2.0]×` trailing pre-freeze EMA).
- **Checkpoint / resume:** the anchor is NEW state. Bump `_STATE_VERSION` 1→2; checkpoint `(pdr_anchor, lr_anchor, anchor_step, anchor_set)`; **idempotent one-shot capture**; on a post-anchor resume with missing anchor state → **fatal_error, never silent re-capture** (re-capturing on a now-frozen/cooler body permanently re-baselines the reference). Mirrors the row-center μ₀ resume pattern.
- **LR-schedule change on resume** breaks the lr-track cancellation → checkpoint the lr-defining settings (max_lr, min_lr, max_steps, warmup_steps, schedule_type); fatal/loud-warn if they change while an anchor exists ("controller will fight the new curve").
- **K-drift makes open-loop `m=1` unsafe:** +10–15%/300M compounds to ~+48% over 1000M and ~+225% over 3000M post-freeze tok — the non-decaying-pdr (KH-v1) failure. So **the closed loop is required even under lr-track** (it is *not* a glorified m=1). Worth one **`m=1` control arm** to empirically pin the drift and prove the loop earns its keep.
- **m_floor may bind legitimately** under a self-captured anchor (a frozen body genuinely wanting to cool below 0.30 is plausibly the desired "carry the anneal forward") → consider lowering m_floor to ~0.15–0.20 and/or making the floor-alarm **regime-aware** (`pdr ≫ r` at floor = base-LR-too-high vs `pdr ≈ r` at floor = benign deep-cool).
- **lr_scale↔WD coupling** (Muon `effective_lr` scales both update and WD, so `m<1` also cools WD) → capture the anchor *after* both the f-ramp and the WD-taper complete (aligned at step 12000 in dn3).
- **Head-hygiene orthogonality — confirmed:** the controller actuates only `feed_forward.w1/w2/w3` (Muon); the head levers act on `output.weight` (Adam) — disjoint param sets. Any loss-path coupling is absorbed by the measured pdr the loop already closes on.

---

## 6. Questions for Math (ranked)

1. **THE PIVOT — frozen-body natural pdr anneal:** after f→1, does the frozen FFN's natural pdr track lr exactly (**a**/LR-track, m≈1), cool faster (**b**/learned-glide), or slower (**c**)? This decides the default reference shape, which rail is operative, and whether m_floor binds.
2. **Is LR-track the right run-1 default**, with learned-glide a gated opt-in clamped to the lr-track ceiling — or do you want learned-glide from the start?
3. **m_floor for the self-anchoring regime:** keep 0.30, lower to ~0.15–0.20, or make the floor-alarm regime-aware? (A frozen body cooling below 0.30 may be exactly the goal.)
4. **Anchor-capture window:** geometric mean over ~5–10 post-freeze samples acceptable? Any preference on requiring f=1 AND WD-taper-complete AND lr locally-flat before latching?
5. **Sanity-bound on the captured anchor** (e.g. `pdr_anchor ≤` last growing-body pdr, within ~0.7–1.0×) — reasonable, or does the frozen/growing pdr ratio make that wrong?

---

## 7. Build plan (verified; backward-compatible)

New `reference: {mode: auto}` path **alongside** the existing knots path (knots untouched → **live dn3 unaffected**). Edits, all verified against the code:

- `body_lr_controller.py`: parse `reference.mode` (default `'knots'`); on the first observe at/after `anchor_step`, capture+latch `(pdr_anchor, lr_anchor)`; branch `reference()` on mode; accept live `scheduled_lr` in `observe()`; add anchor fields to `state_dict`/`load_state_dict`; `_STATE_VERSION` 1→2; add the **upper-rail alarm** + **`m_ff_raw` logging**.
- `train_mara.py`: pass `scheduled_lr` (= `get_lr(step, settings)`, line 219-234) into `observe()` (line 2963); checkpoint the lr-defining settings for the resume guard.
- Validated against the controller's existing state/restore (`body_lr_controller.py:298-320`), the LR source (`train_mara.py:219-234`), and the m_max guard (`train_mara.py:4827-4829`).
- **Offline validation in `tools/pdr_controller_sim.py`** (negative-drift peg-at-1 alarm; steeper-than-lr clamp; anchor-noise sensitivity) **before any live run.**

*Risk-tier note: the SHIPPED dn3 tail is already an lr-track, so dn3 sits on the safe side — all residual risks above attach to the proposed self-anchoring/learned-glide UPGRADE, which is not yet in the code.*
