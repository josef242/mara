# Math Agent Q10 — head-growth control for a future dn4

## Why now
dn3 (grow-then-clamp) is in flight to answer the **BODY** question: is body-growth's benefit in the
grown STATE or the ongoing PROCESS? Once that lands, the next front is the **HEAD** (`output.weight`,
the LM unembedding), which in DN2 grew faster than anything else and is **not** addressable by the
body's lever (tangent projection is Muon-only; the head is an Adam param). This brief assembles (1)
the DN2 data, (2) the gauge subtlety that decides which lever is right, and (3) the inventory of
BUILT vs designed interventions — so we can pick dn4's head intervention deliberately.

## 1. DN2 data — the head outgrows everything, and accelerates

Mean-per-layer / component weight norms from `dreadnought_v2/diagnostics.jsonl` (rank-0, snapshot
cadence). Clean organic window is steps 100→17800; see the §1a caveat for 18200.

| step | tok(M) | **head ‖W‖** | head pdr | emb ‖W‖ | ffn ‖W‖ (mean/layer) | aux heads |
|------|--------|--------------|----------|---------|----------------------|-----------|
| 100   | 20    | 517.3  | 4.0e-4 | 518.0 | 355.5 | off |
| 2100  | 413   | 546.7  | 1.7e-3 | 563.1 | 412.3 | off |
| 6700  | 2379  | 630.0  | 1.3e-3 | 754.4 | 691.7 | off |
| 8600  | 4130  | 721.0  | 1.1e-3 | 823.1 | 764.9 | ramping (11k start)\* |
| 11800 | 8554  | 982.3  | 5.9e-4 | 928.4 | 856.9 | ramping |
| 14800 | 14748 | 1308.0 | 4.2e-4 | 1020.9| 924.8 | full (≥15k) |
| 17800 | 24383 | 1660.2 | 2.9e-4 | 1108.1| 983.0 | full |

\*aux-head weights ramp on the **11000→15000** schedule (layers 50/55/60), so the whole back half
runs *with* aux heads active.

**The headline numbers:**
- Head ‖W‖: **517 → 1660 (+221%)** over the run, and it **accelerates**: +22% in the first 2.4B tok,
  then **+163%** from step 6700→17800 while the body FFN grew only **+42%** over the same window.
- Head/FFN ratio: 1.46 → 0.91 → **1.69** — the head ends ~1.7× the body's per-layer norm.
- Head vs emb: started equal (517≈518), head ends **+50%** above emb (1660 vs 1108).
- **Aux heads did NOT tame it.** The head's steepest growth (878→1660) is entirely *inside* the
  aux-head-active window. Distributing readout pressure across the body did not stop the main head's
  norm from running.
- head pdr declines (1.7e-3 → 2.9e-4) as ‖W‖ grows — the usual 1/‖W‖ shadow — i.e. the head keeps
  taking sizeable *absolute* steps; the relative step shrinks only because the denominator inflates.

### 1a. The 18200 reading is an intervention artifact — and DN2's head levers were all LATE retrofits
The final diagnostics entry (step 18200) reads head ‖W‖ = **52227**, `effective_rank_c ≈ 1.1`, a near
rank-1 spike — NOT organic growth. It is the **row-center onset** (Brief #4 instability), and it caps
a short sequence of head-control *retrofits* DN2 ran at the very end, all at step ~18000
(reconstructed from 32 saved runtime configs):

| step / date | what was toggled |
|------|------------------|
| ~11000 (May) | **aux heads** on (layers 50/55/60, ramp 11k→15k) — head kept accelerating anyway |
| **18000 (Jun 16)** | **z-loss ON, α=1e-4** → re-tuned to **α=3e-7** within hours (1e-4 too hot) |
| 18500 (Jun 21) | z-loss **OFF** |
| 18000 (Jun 21) | **rewound to 18000, row-center ON** instead → the 52227/rank-1 instability (Brief #4) |

**Both built levers (z-loss AND row-center) were tried at step 18000 and neither was kept.** But the
decisive caveat: these were **retrofits onto an already-grown, gauge-laden head** (‖W‖ already ~1660
and climbing, ~78% gauge by then). A late z-loss needing a 330× α cut, and row-center destabilizing,
may say more about *retrofitting a sick head at step 18000* than about the tools themselves. **The
open question is whether the same levers, applied PREVENTIVELY from the start of a fresh run, prevent
the pathology rather than failing to reverse it** — see §4 Q1.

## 2. The gauge subtlety — most of the "head norm/logZ" is CE-invisible
From the row-center probe (`ZLOSS_CENTERED_PLAN.md`, Nexus #139→#141), DN2's head log-partition
decomposes as:

> **logZ ≈ 502  =  h·μ ≈ 394 (a CE-invisible common-mode gauge, ~78%)  +  logZ_c ≈ 108 (genuine
> centered margin, ~22%).**

`μ = mean_v(w_v)` is the head's common row-mode; adding any common vector to all output rows shifts
every logit equally and is **invisible to CE** (softmax gauge freedom). So a large fraction of the
"alarming head growth" may be a *free gauge the model cannot see* — which means **a naive norm- or
raw-logZ penalty spends ~78% of its budget pushing the gauge**, not the part CE responds to. The
real target is the **centered** head geometry (`logZ_c`, `‖W_c‖`). dn3 currently logs healthy
centered head geometry at the fork (`logZ_c≈24`, `‖W_c‖≈218`, eff_rank≈379) — full-rank, small
margin — so we have a clean baseline to watch.

**Open question for you:** is the *centered* head growth (the 22% real part) actually pathological,
or is the head, like the body's radial growth, mostly accumulating a harmless gauge that only *looks*
dire in the raw norm? This determines whether dn4 needs to fight the head at all, or only its gauge.

## 3. Inventory of head levers — built vs designed

| Lever | Mechanism | Built? | Observed / status |
|-------|-----------|--------|-------------------|
| **z-loss (raw logZ)** | soft penalty on `logsumexp(logits)²`; option-D reconstruction (CE_none + target-logit), fp32_accum | **BUILT, merged to main** | **RAN on DN2 @step 18000** (α=1e-4, re-tuned to 3e-7 within hours, OFF after ~500 steps) — a late retrofit, not kept. ~78% of its budget pushes the CE-invisible gauge → motivated the switch to a *centered* target. Never tried from a fresh start. |
| **row_center_head** | in-loop `W ← W − 1·μᵀ` (subtract common row-mode), function-preserving | **BUILT** | **RAN on DN2** (rewound to 18000 after z-loss) → **instability** (Brief #4; the 52227/rank-1 reading). Theoretically gauge-only, but the hard in-loop subtraction destabilized — again, a retrofit on a grown head. |
| **auxiliary_heads** | RMSNorm+Linear NTP taps at depths 50/55/60, distribute readout-shaping pressure across the body | **BUILT, ran in DN2** | Did **not** tame head-norm growth (head accelerated through the aux window). May aid readout quality / body differentiation independently. |
| **centered z-loss (logZ_c)** | soft penalty on `logZ_c² = (logZ − h·μ)²`, gradient through μ(W) (Objective A) | **DESIGNED, math-verified, NOT built** ("pending greenlight") | Provably **zero common-mode gradient** → 100% of pressure on the CE-visible centered structure. The cleanest candidate. |
| **head WD** | weight decay on `output.weight` (currently 0.02) | present | Cranking it as a growth control is a known **DO-NOT** (rejected in the WD-waste work). |

(Body-side context: the body's analogous problem — radial ‖W‖ growth — is handled by **tangent
projection** + the **FFN pdr controller**, the dn3 machinery. Neither applies to the head: tangent
projection is a Muon-update operation, the head is Adam; and the head's "gauge" is the across-vocab
common mode, a different object than the body's radial direction.)

## 4. The dn4 questions
Assume dn3 answers the body question and dn4 turns to the head — possibly as a **fresh run** with head
control from step 0, not another retrofit. We'd like your read on:

1. **Retrofit vs preventive (the framing question).** DN2's z-loss and row-center both *failed as
   late retrofits* at step 18000, on a head already grown to ~1660 with ~78% gauge. Would the *same*
   tools applied **from the start of a fresh run** behave fundamentally differently — holding the head
   near its healthy birth geometry (`logZ_c≈24`, `‖W_c‖≈218`, full rank) and preventing the gauge /
   rank-1 accumulation, rather than trying to reverse a pathology already baked in? I.e. is dn4 best
   cast as a **fresh run with head control on from step 0**, where these levers may finally be in
   their natural regime? (We no longer have the dn2 retrofit metrics — that's fine; the question is
   forward-looking.)
2. **Does the head need taming at all?** Is the *centered* head growth genuinely harmful to loss /
   generalization, or is the alarming raw growth ~78% harmless gauge (the §2 question)?
3. **Primary lever.** If we do act, is **centered z-loss (logZ_c, Objective A)** the right primary —
   the only built-or-near-built lever that targets the CE-visible part with provably zero gauge
   waste? What α and schedule, given dn2's logZ_c≈108 baseline?
4. **Stability.** Is a *soft* centered-logZ penalty inherently safer than the *hard* in-loop
   row-center subtraction that destabilized at 18000 (Brief #4)? Or do they share a failure mode we
   should expect?
5. **Aux-heads' role.** Aux heads didn't reduce head norm — should dn4 keep them (for readout
   quality / body pressure) while a separate lever handles the head norm, or are they orthogonal?
6. **Body↔head coupling.** The head reads a final-RMSNorm'd hidden, so it's magnitude-decoupled from
   the body. But if dn3 shows the body benefit is in the grown STATE, does a *frozen* body change the
   readout problem enough to alter what head control is needed (e.g. less/more sharpening pressure)?
   I.e. should dn4's head plan wait on dn3's body verdict, or is it independent?

— Code (relayed by Josef)
