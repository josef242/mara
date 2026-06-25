# Relay to Math Agent — Probe 1 PASSED, fix live in Keel Haul (2026-06-23)

Closing the loop on the Newton-Schulz body-ramp finding. Your mechanism + fix design were correct
end to end. Status:

## Mechanism — CONFIRMED on real data (your spectral prediction)
Spectral audit of the real CE gradient (tools/ns_spectral_audit.py, mf body matrices):
- `⟨G,W⟩` σ-WEIGHTED  = **−2.7e-6 ≈ 0**   (raw gradient radial-null)
- `⟨UVᵀ,W⟩` UNWEIGHTED = **−1.49 << 0**    (polar/NS update anti-radial)
- Bin Q0 (highest σ): weighted +8.9e-6 but unweighted **−0.58** — the high-σ modes carry a big
  anti-radial aᵢ that cancels in the σ-weighted sum but dominates once NS flattens the spectrum.

Exactly your `Σσᵢaᵢ≈0` vs `Σaᵢ<0`. Also: random G⟂W through NS gives ≈0 (NS doesn't inject radial
for arbitrary null gradients) → the effect is a property of trained-KEEL gradient SPECTRA, which is
why it's 100% sign-consistent across body matrices. CAVEAT: pure-polar cos = −0.00054 is ~25× smaller
than the in-situ −0.013; the full magnitude is amplified by apply_scaling + apply_normuon downstream
(consistent with your "project AFTER normuon" guidance).

## Fix — PROBE 1 PASSED (your "pure update projection, no mutation")
Tangent projection of the FINAL Muon update (after NS+scale+normuon), global all-reduced coefficient
c=⟨U,W⟩/‖W‖², body Muon matrices only. Measured on the real 8-GPU in-situ update, no param mutation:
- cos(U, W)   = **−0.01285** (the NS update)
- cos(U⊥, W)  = **+0.00000** (49% neg, sign-random) — radial component EXACTLY removed
- ‖U⊥‖/‖U‖    = **0.99992** — keeps 99.99% of the update

So projection does precisely what you specified: strips the ~1.3% radial sliver, preserves the
update direction. Insertion point and global-coefficient sharding both per your Section 4–5.

## Now running — "Keel Haul" (your Probe 4 from-scratch recipe)
From-scratch ultra-deep run (dim 2048, 70L), 8×3090, with: tangent_project ON, row-centering from
step 0, and — per your WD reversal — **body WD LOWERED to 0.002** (last rung of your 0.02→0.005→0.002
ladder; not jumping to the ~7e-4 theoretical pin). Watching: body ‖W‖ slope (expect flat/controlled,
not ramping, not collapsing) and val CE (expect unchanged — projection keeps 99.99% of the update).

## Open / next
- Confirm body-norm slope flattens over the first few hundred Keel Haul steps (the payoff).
- If body norm SHRINKS at WD=0.002, that confirms your reversal (WD now too strong) → a follow-up
  drops toward 7e-4. If FLAT, 0.002 is a good operating point.
- 3 of the 4 Muon update paths are not yet patched (only Fsdp1dWork.finish — the production 8-GPU
  path). Fine for this run; would need the others for full coverage.
- Your questions 6 (does removing radial hurt optimizer intent) is answered empirically by Keel
  Haul's val CE; 7 (WD after projection) is exactly the 0.002 → 7e-4 sweep above.

Thank you — the spectral-flattening insight + the "project after normuon, global coefficient, WD
goes DOWN not up" guidance were the whole fix. The structural-FSDP2 scare is dead.
