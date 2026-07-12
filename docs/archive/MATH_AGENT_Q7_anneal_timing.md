# Math Agent Q7 — should the kv2 body-LR anneal start EARLIER than step 2680 (~350M tok)?

## Context
kv2 is the from-scratch run testing your Brief #6 fix: full tangent projection + low WD + an
explicit DELAYED body-LR decay (lr_mods) to replace the implicit anneal the body-norm ramp used to
provide. Current schedule (B=8, token→step): hold mult=1.0 until step 2680 (~350M tok), then
0.75@600M, 0.67@800M. Your reasoning for the delay: preserve KeelHaul-v1's real early lead.

kv2 is currently at ~step 600 (~80M tok), tracking KH-v1's pdr almost EXACTLY so far. Josef, reading
the live dashboard, thinks the body crosses into "damaged" territory closer to **~190M tokens**
(≈ step 1450, i.e. right as the 1500-step LR warmup ends), and wants to consider pulling the first
transition back to ~step 1500-2000. We want your read before restarting (lr_mods is read at startup,
so any change = a from-scratch restart — kv2 is ~1 day to 800M).

## The data (KH-v1 = projection + NO anneal, the un-throttled baseline; vs DN2 = ramp, the target)
Body pdr = ‖ΔW‖/‖W‖ (median attn+ffn), and train loss, at matched tokens:

| tok(M) | KH ls | DN2 ls | KH−DN2 | KH pdr | DN2 pdr | KH/DN2 |
|---|---|---|---|---|---|---|
| 100 | 4.877 | 4.984 | −0.107 | 1.45e-3 | 0.85e-3 | 1.70 |
| 150 | 4.781 | 4.705 | +0.076 | 1.84e-3 | 1.10e-3 | 1.68 |
| **190** | 4.678 | 4.664 | **+0.014** | **2.20e-3** | 1.32e-3 | 1.67 |
| 250 | 4.602 | 4.488 | +0.113 | 2.84e-3 | 1.72e-3 | 1.65 |
| 300 | 4.246 | 4.310 | −0.064 | 3.00e-3 | 2.10e-3 | 1.43 |
| 400 | 3.979 | 3.979 | 0.000 | 3.13e-3 | 2.71e-3 | 1.16 |
| 500 | 3.765 | 3.771 | −0.006 | 3.19e-3 | 2.68e-3 | 1.19 |

Key facts:
- **KH's pdr is ~1.6-1.7× DN2's from the START (~100M)** — the relative-step gap is NOT a late
  crossover; it's present the whole time and only *narrows* later (1.7→1.2) as KH plateaus and DN2
  catches up in absolute terms.
- **DN2's pdr peaks ~2.73e-3 around 430M (step ~2200) then decays.** KH overshoots to ~3.2e-3 and
  never decays (no anneal).
- **The LOSS consequence is late + noisy:** KH and DN2 are ~tied through ~400M; the durable
  divergence where DN2 pulls ahead (Brief #6's +0.21 nats) is ~500-800M, not 190M.
- Warmup ends at step 1500 (≈197M tok) — so Josef's "damage at ~190M" coincides almost exactly with
  **LR reaching full strength (3.5e-4)**. (His earlier instinct was also "base LR may be too high.")

## The competing anchor points (this is the crux for you)
Where SHOULD the throttle begin? The candidates imply very different schedules:
1. **pdr-divergence onset (~100M / step ~750):** the relative step is already 1.7× DN2 here. If the
   goal is "match DN2's pdr trajectory," you'd start almost immediately — but that sacrifices the
   early lead your Brief #6 wanted to preserve.
2. **Josef's chart "damage" crossover (~190M / step ~1450 = LR-cap):** start the bend in the warmup
   tail so body LR is already reducing as it reaches full strength.
3. **DN2's pdr PEAK (~430M / step ~2200):** let kv2 peak where DN2 peaked, then bend — closest to
   literally reproducing DN2's pdr curve. (Roughly our current 2680, slightly early.)
4. **Loss-divergence onset (~500M):** the current 2680 (~350M) is already a bit before this.

## Questions
1. **Does the pdr data support pulling the transition earlier than 2680?** Specifically: KH's pdr is
   ~1.6-1.7× DN2's from 100M onward, but the loss only diverges ~500M+. Is the early high pdr
   actually "damage accumulating silently" (→ throttle early, anchor #1/#2), or is it the *healthy*
   high-plasticity exploration that earns kv2's early lead (→ keep the delay, anchor #3)?
2. **Is Josef's read that ~190M (= LR-cap) is the damage onset well-founded** given the loss doesn't
   visibly diverge there? Could a body over-plasticity become "baked in" at LR-cap even though the
   loss penalty only surfaces 300M tokens later? (i.e. is loss a lagging indicator of pdr damage?)
3. **If earlier is right, which anchor + shape?** We've pre-staged:
   - Option B (@2000/262M): peak near DN2's peak timing.
   - Option A (@1500/197M): bend the moment LR caps (Josef's 190M anchor).
   - Option C (start @1200, into decay by 1450): actively pull back DURING warmup tail so body LR is
     reducing AS it hits full strength — anchored to "damage at LR-cap."
4. **Target pdr:** should kv2 aim to track DN2's pdr curve (peak 2.7e-3 @430M → 2.3e-3 @800M), or is
   a DIFFERENT (lower? earlier-peaking?) pdr trajectory better given the projection holds ‖W‖ flat
   (so kv2's pdr denominator never grows, unlike DN2's)?
5. **Confound check:** is this really a *body-LR-schedule* problem, or is Josef's other instinct
   right that **base max_lr (3.5e-4) is simply too high** — in which case the cleaner fix is lowering
   global LR, not (only) annealing the body? How would you disentangle "body anneals too late" from
   "global LR too high" given they both show up at LR-cap?

## What we'll do
- If you favor earlier: restart kv2 with the anchor/shape you pick (one validated lr_mods edit).
- If you favor keeping 2680: let it run; re-check at step ~1500-2000 with the overshoot monitor.
- If you think it's a global-LR problem: separate experiment with lower max_lr.

Reference: docs/MATH_AGENT_BRIEF_6_projection_removes_annealing.md (the anneal hypothesis + data),
tools/pdr_overshoot_monitor.py, configs/kv2_ANNEAL_OPTIONS.md (pre-staged schedules), configs/kv2.yaml.
