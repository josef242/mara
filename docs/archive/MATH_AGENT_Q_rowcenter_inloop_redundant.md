# Follow-up Q for Math Agent — does your theorem make IN-LOOP weight-only row-centering redundant?

Your row-centering analysis (exp_avg projection is the prime suspect; weight-only should be
trajectory-preserving) prompted a question we want your read on before we build the B-vs-C
experiment, because it may change whether we run it at all.

## The crux
Your theorem: with CE shift-invariant and the gradient already row-centered (Σᵢ∇wᵢL=0),
weight-only post-step centering gives
  P(W_{t+1}) = (1−ηλ)P(W_t) − η·P(U_t)
— i.e. the centered component evolves EXACTLY as the unprojected run's would. The removed part
is pure gauge.

If that's right, then for the **centered quotient** (which is all CE/softmax/sampling sees),
in-loop weight-only centering is **trajectory-identical to NO in-loop centering**. The only
difference is cosmetic: W "looks centered" in the saved checkpoint. But:
  P(W_t) [centering every step]  ==  P(W_t) [centering once, offline, at the end]
up to float/distributed-order noise — because projection commutes with the centered evolution.

## The question
**Does in-loop weight-only row-centering buy anything over "train uncentered + center offline
at checkpoint/export"?** Our reading of your own theorem says: no — they produce the same
centered model. If so, the simplest correct policy isn't "salvage weight-only in-loop," it's:

  → retire in-loop row-centering ENTIRELY (it's at best cosmetic, at worst — via exp_avg — fatal)
  → row-center OFFLINE only, at export/quantization/post-training boundaries

i.e. your B branch (weight-only in-loop) might be **provably equal to A (no centering) in every
way that matters**, making the whole in-loop feature redundant rather than salvageable.

## Where we might be wrong (please pressure-test)
1. **Is there a real reason to center in-loop that the theorem misses?** E.g.:
   - raw-logit numerical hygiene DURING training (your "forward-only centering" note) — but that
     doesn't need to mutate W either.
   - keeping the gauge from growing so large that bf16 storage of W loses precision in the
     centered part (a finite-precision argument the exact-arithmetic theorem doesn't cover)?
   - any interaction with z-loss, diagnostics, or downstream resumes that assumes centered W?
2. **Does the theorem actually hold under our real conditions** — bf16 weights, fp32 Adam state,
   FSDP-sharded all-reduce of the gauge, grad-accum? Or does finite precision / sharding make
   in-loop centering meaningfully different from offline (e.g. the gauge growing to O(100s) in
   bf16 W could lose low-order bits of the centered part, which periodic in-loop centering would
   prevent but offline-once would not)?
3. If the precision argument in (2) holds, the right answer might be "in-loop weight-only IS
   worth it, but ONLY for precision hygiene, and the exp_avg projection must go."

## What we'll do based on your answer
- If in-loop weight-only is **redundant** with offline → retire the in-loop feature entirely,
  keep offline-only. (Simplest, and we skip building the B-vs-C branches.)
- If there's a **real precision/numerical reason** to center in-loop → keep weight-only in-loop,
  retire exp_avg projection, and we run B-vs-C to confirm weight-only is stable + matches A in the
  centered quotient.
- Either way: exp_avg projection is retired, no head-WD, z-loss not used as rescue (per your prior).

Context: Keel Haul (body tangent-projection run, row-centering OFF) is training fine — so this is
a "clean up the head approach properly," not a fire. We just want to not build an experiment whose
result your theorem may already determine.
