# Math Agent — Q12: Partial-f Progressive Engagement
### (the PDR controller MUST engage *during* the ramp, not at f=1)

**Status:** design problem + NEW empirical evidence, for Math to rule on. Direct follow-up to Q11 (self-anchoring PDR controller — BLESSED, BUILT, and actuator now confirmed). Relayed via Josef 2026-06-29.

**Josef's call after seeing the probe:** "It's now obvious that this is absolutely required to have a functional pdr controller." This is no longer a nice-to-have refinement.

---

## TL;DR / the ask
The Q11 controller engages only at the freeze point (f=1 / anchor latch). The open-loop `force_m` probe just demonstrated that **the body's pdr has already diverged from the healthy trajectory by ~150–170M tokens — mid-ramp, f ≈ 0.5–0.7 — long before any correction arrives, and a maximal late cut (m=0.2) cannot catch back up.** We need Math's ruling on the **reference + engagement law during the ramp** (f ∈ (0,1)).

---

## Recap: what Q11 does, and its blind spot
- **Grow-then-clamp body.** Tangent-projection strength `f` ramps 0→1 over a window; at strength `f` the f-fraction of the *radial* (norm-growing) update component is removed → the body's self-anneal (pdr = ‖ΔW‖/‖W‖ shrinking as ‖W‖ grows) is **progressively removed**.
- **Q11 controller.** At f=1, capture `K_anchor = pdr_anchor / m_anchor` and `lr_anchor`, then ride `r(t) = K_anchor · lr(t)/lr_anchor`; FF-only, m ∈ [m_floor, 1], cuts-only. **m = 1 for the entire ramp.**
- **Blind spot.** During the ramp the body is *partially frozen but uncontrolled*. The removed anneal is **continuous in f**, but the correction is a **step function** that switches on only at f=1.

---

## The empirical evidence (force_m open-loop probe — skiff, 2026-06-29)
Config: T2048, **65,536 tok/step**. f-ramp (`tangent_project_strength`) = steps **1500 → 3000 = 98M → 197M tokens**. WD taper same window. Anchor latches ~step 3450 (~226M). `m` only *visibly* bites ~470M.

1. **Actuator is confirmed linear** (3-run cross-check + code trace, adversarially verified): `ffn_pdr = 1.08 · m · attn_pdr`. m=0.2 → ffn pdr cut to 0.2× (1.0e-3 → 2.06e-4 at 491M); attn (m=1 control) untouched. **pdr ∝ m, clean.**
2. **Divergence begins mid-ramp.** The clamped body's pdr departs the healthy (growing-body / picket) trajectory at **~150–170M tokens ≈ step 2290–2594 ≈ f ≈ 0.53–0.73** — when the body is only ~½–¾ clamped. That is **50–75M tokens before f=1**, and **~75–100M before the anchor latch.**
3. **A late hammer cannot catch up.** Resuming at 491M and slamming m to 0.2 (a 5× cut — the *maximum* the controller could ever command) does **not** restore the trajectory toward base. Reason: **pdr is a rate; the accumulated displacement gap from the ramp is an integrated state.** You cannot un-integrate the excess ∫pdr after the fact by lowering the rate later.

---

## Why this is structural, not a tuning issue
The harm = **∫(excess pdr)** accumulated over the ramp window where f>0 and m=1. Engaging at f=1 leaves this integral uncorrected *by construction* — the actuator only affects future steps. No choice of post-freeze gains/floors recovers it. The only fix is to **start cutting while f is rising.**

---

## Proposed direction (for Math to rule on)
Make engagement and the reference **f-aware**, so the controller restores the f-fraction of the removed anneal *as it is removed*:

- **Engage from f-onset** (~step 1500), not f=1.
- **Reference = the body's own natural-growth anneal, extrapolated.** Anchor the anneal slope **early** — during the free-growth phase (f=0, steps 0–1500) — and ride that glide through the ramp and beyond. The clamped body runs *hot* relative to this glide (its ‖W‖ stops growing), and the loop cuts m to hold *measured* pdr on the glide. The Q11 frozen lr-track becomes just the **f=1 tail** of this same glide.
- **Handoff intuition.** During the ramp the realized anneal = (1−f)·(residual ‖W‖ growth) + f·(m-cut). Closing the loop on *measured* pdr should absorb the (1−f) residual automatically; m supplies the missing f-fraction.

### Open questions for Math
1. **Reference during the ramp.** Does Q11's lr-track suffice if we merely engage it earlier, or does the ramp **require the learned-glide** (natural anneal slope) that Q11 deferred to a gated opt-in? During the ramp ‖W‖ is still partly growing, so a flat-ish lr-track reference looks wrong — the body is genuinely still annealing.
2. **Where to anchor the glide slope.** Is the f=0 free-growth window (steps 0–1500) clean enough to fit the anneal slope, or too short/noisy? Does it need the early ramp included?
3. **Engagement law.** Should m track measured-pdr-to-glide via the same FF loop (just turned on at f-onset), or should the *authority* of the cut be tied explicitly to f (e.g., correction strength scales with f)?
4. **New failure modes.** Q11's cuts-only / no-LR-fight guarantees carry over — but does early engagement during the noisy growth phase introduce any new way to over-cut or fight the LR?

---

## Constraints carried from Q11 (unchanged)
- **Cuts-only:** m ≤ 1, m_max=1.0 fatal-guarded (never amplify).
- **Must not fight the LR curve** by construction.
- **Self-anchoring** (no twin run).
- **FF-only** default (PI off) unless Math rules otherwise.

---

## References
- Q11 spec: `docs/MATH_AGENT_Q11_self_anchoring_pdr_controller.md` (BLESSED).
- Actuator confirmation + dashboard-median gotcha: `docs/PDR_CONTROLLER_STATE_BRIEF.md`.
- Controller code: `common_fsdp2/body_lr_controller.py`; actuation `common_fsdp2/muon_fsdp2.py:455-472`; pdr measurement `common_fsdp2/diagnostics.py:604-650`.
