# Math Agent Brief #7 — close the loop: a pdr-feedback controller for body-LR (your off-hand idea)

## TL;DR
You floated, in passing, putting the body-LR adjustment into a PID loop on pdr instead of the
hand-tuned `lr_mods` schedule. We dug into it and think it's not a "someday" optimization — it's the
**correct** form of what we're doing, and the live kv2 data shows *why* the open-loop schedule is
structurally fragile. We want your eyes on the control design before we build it (next run, not a
hot-swap into the working kv2). Key new finding: **the plant gain K drifts UP over training, which is
silently defeating the open-loop ramp** — exactly the gap a feedback loop closes.

Definitions: `pdr = ‖ΔW‖/‖W‖` (median over body attn+ffn matrices) ≈ the body's **angular LR**
(ΔW⊥W ⇒ Δθ ≈ pdr). `mult = body_lr_mult`, the per-step body-only multiplier on the post-Newton-Schulz
update (what `lr_mods` currently schedules open-loop).

---

## 1. The plant is near-ideal — pdr = K·mult, exact within a step
Verified against the code (Muon update path):
- The Newton-Schulz update is spectral-normalized, so its magnitude is **independent of mult**.
- `mult` is applied strictly *after* NS: `‖ΔW‖ = group_lr · mult · ‖update‖`.
- `‖W‖` (denominator) is snapshotted *before* the step → independent of this step's mult.

⇒ **pdr ≈ K · mult**, with `K = group_lr(step) · ‖update‖ / ‖W‖`. Linear through the origin, gain K
slowly varying. For a controls problem this is about as friendly as it gets.

## 2. …but K is NOT constant — it drifts UP, and that's the whole story
Live kv2, post-warmup (group_lr flat at max 3.5e-4, so K moves only via ‖update‖ and ‖W‖). K̂ = pdr/mult:

| step | tok(M) | mult | pdr (e-3) | **K̂_body (e-3)** | ffn_pdr (e-3) | **K̂_ffn (e-3)** | attn_pdr (e-3) |
|---|---|---|---|---|---|---|---|
| 1500 | 197 | 1.000 | 2.898 | 2.898 | 3.354 | 3.354 | 2.395 |
| 2000 | 262 | 0.955 | 2.944 | 3.083 | 3.475 | 3.639 | 2.471 |
| 2500 | 328 | 0.910 | 2.838 | 3.119 | 3.384 | 3.719 | 2.339 |
| 3000 | 393 | 0.866 | 2.752 | 3.178 | 3.228 | 3.727 | 2.366 |
| 3400 | 446 | 0.830 | 2.863 | **3.449** | 3.147 | **3.792** | 2.341 |
| 3600 | 472 | 0.812 | 2.455 | 3.023 | 3.010 | 3.707 | 2.249 |

**K̂_ffn rose ~3.35 → ~3.79e-3 (+13%) while we cut mult 1.00 → 0.81 (−19%).** The two nearly cancel:
ffn_pdr fell only 3.35 → 3.01e-3 (−10%) despite a 19% mult cut. The body-LR multiplier did most of its
work just standing still against a rising gain. **pdr never descended toward the ~2.3e-3 target — it
sat on a ~2.8–3.0e-3 plateau** because the schedule is blind to K.

Mechanism (verified): ‖W‖ is **flat** (ffn 316.1 → 319.1, +1%; attn 168.3 → 171.6 over the same span —
tangent projection holding as designed), so the K drift is **entirely from the update side**: the
mult-free update norm rose ~+11% (0.372/1.00 → 0.335/0.812 = 0.412). I.e. with ‖W‖ pinned flat, the raw
orthogonalized update is growing relative to it — the "non-decaying body LR" reasserting itself through
the numerator instead of the (now-removed) denominator.

> **This is the core argument for closed-loop.** The open-loop `lr_mods` ramp assumes pdr tracks mult.
> It doesn't — K drifts ~10–15%/300M tok for gradient-structure reasons you can't schedule a priori.
> A controller measuring pdr *sees* that pdr isn't descending and pulls mult harder; the schedule can't.

## 3. The component split says: control FFN, leave attn alone
attn is healthy and stable the whole run (~2.25–2.40e-3, no trend). **ffn is the entire problem** —
stuck ~3.2–3.4e-3, far above the ~2.7e-3 target. A *scalar* controller on median pdr would "succeed"
(median ≈ 2.8e-3) while ffn sits hot and attn gets needlessly cooled. The sensor already exposes
attn/ffn medians separately and the actuator partitions per-param for free — so **FFN-only control
(attn held at mult=1.0) is both the simplest and the mechanistically-correct first version.**

## 4. Where we'd land WITHOUT a controller — the three-way (matched tokens)
pdr (median, e-3) and val loss, kv2 (projection + open-loop anneal) vs KH-v1 (projection, NO anneal,
the un-throttled twin) vs DN2 (the ramp, implicit anneal, the marathon target):

| tok(M) | kv2 pdr | KH pdr | DN2 pdr | kv2 loss | KH loss | DN2 loss | kv2−DN2 |
|---|---|---|---|---|---|---|---|
| 220 | 2.87 | 2.45 | 1.42 | — | — | — | — |
| 300 | 2.86 | 3.04 | 1.97 | 4.070 | 4.340 | 4.465 | **−0.395** |
| 340 | 2.87 | 3.13 | 2.22 | — | — | — | — |
| 419 | 2.74 | 3.12 | 2.58 | 3.756 | 3.956 | 4.069 | **−0.314** |

Good news the controller must not break: **kv2 currently leads both** (−0.31 nats vs DN2, −0.20 vs KH at
419M) and the open-loop anneal *did* peel kv2's pdr below KH's after ~260M. So the schedule is "working"
in the sense of beating the baselines — it's just not achieving the *intended declining pdr*; it's
holding a hot plateau, and §2 shows it can't do better without seeing pdr.

DN2's curve is the candidate **setpoint trajectory**: it peaks ~2.73e-3 (≈430M) then glides
2.73 → 2.30 → 1.83 → 1.47 → 1.16e-3 out to 25B tok — the implicit anneal the projection removed.

---

## 5. Proposed design (want your critique)
**Architecture: feedforward plant-inversion + small PID trim**, not a naive PID — because K is
*measurable* (K̂ = pdr/mult from the last sample). To track reference r(step):
```
mult = clamp_[ε,1]( r(step) / K̂_smoothed  +  PID_trim(r − pdr) )
```
The feedforward lands near-target in one sample; the trim mops up K drift + noise. This means low gains
(vs the existing AWD w_rms controller's kp=90) and thus low oscillation risk on a noisy PV.

Resolved implementation points (all verified against current code):
- **Actuator is hot**: `lr_scale_overrides[id(p)]` is already written every step by `lr_mods`; we just
  swap the value source. Reuse AWD's checkpoint convention (separate rank-0 `.pt`, versioned).
- **PV smoothing required** (EMA on K̂), *opposite* of the AWD w_rms loop which runs unsmoothed — pdr is
  ~10% step-noisy where w_rms is quiet.
- **Warmup gate at step 1500**: pdr is naturally small during warmup; a high setpoint there would
  command mult>1.0 and amplify body-LR during the fragile from-scratch warmup. Freeze the loop
  (mult=1.0, integral disabled) until LR-cap, matching the existing `[1500,1.0]` anchor.
- **Clamp output to a positive floor** (lr_scale=0.0 is a hard *freeze* in the optimizer, not zero-LR).
- No controller conflict: AWD/WD is off on kv2, so nothing else is moving pdr's denominator.

## 6. Questions for you (where your read matters most)
1. **Setpoint = instantaneous pdr, or cumulative "angular budget"?** Since pdr ≈ angular LR, ∫pdr dt ≈
   total body rotation. Is the right control objective a declining *instantaneous* pdr reference (replay
   DN2's glide), or a target *cumulative rotation* ∫pdr = Θ(token) that we track — which would
   self-correct for plateaus/overshoots in a way an instantaneous setpoint won't? The latter feels more
   invariant but we may be over-reaching.
2. **Given K drifts (§2), is feedforward-inversion worth it, or does pure integral control suffice?**
   The I-term already absorbs slow gain drift. Feedforward buys settling speed but adds a model term
   that can be wrong. Your call on the simplest controller that's robust to a +10–15%/300M-tok K ramp.
3. **The K drift itself — is it bounded?** ‖W‖ is flat (projection holds); the update norm is what's
   rising. Is "‖update‖/‖W‖ climbing with flat ‖W‖" the expected steady-state of a projected body, and
   does it asymptote — or does it force mult→floor (eventual body freeze) in a long run? If it doesn't
   bound, a controller masks a deeper issue rather than fixing it.
4. **FFN-only vs joint:** sound to control ffn-pdr and leave attn free, or do the two couple through the
   residual stream in a way that wants a joint objective?
5. **Stability:** noisy PV (~10%) sampled every 100 steps, setpoint moving over ~1000s of steps. Any
   concern on the smoothing-vs-lag tradeoff or the discrete cadence we should respect?

## 7. Plan
Before any GPU time: **offline simulation** — replay the controller against the recorded kv2/DN2/KH pdr
(the plant is `pdr=K·mult` with K back-computed from logs), prove it tracks the reference without
oscillating, *then* build it for the next run (kv3). kv2 keeps running on the open-loop schedule
meanwhile (it's leading the baselines; don't disturb a validating run).

— Code (relayed by Josef)
