# PDR Controller — Session State Brief (for post-compaction continuity)
**Written 2026-06-29 mid-investigation. Read this first if context was compacted.**

## DASHBOARD READOUT GOTCHA (why the controller looks like a no-op on the chart)
The dashboard's headline `pdr=` is a **POOLED MEDIAN over all per-layer attn+ffn pdrs** (train_mara.py:2944-2946: `_all=sorted(_att+_ffn); _med=_all[len//2]`), NOT a norm-ratio. Once FFN is cut below attn, the median rides the **attn** cluster (which the FFN controller never touches, m=1), so a real **5× FFN cut shows as only ~−10% on the headline** (m=1→~3.55e-4 vs m=0.2→~3.19e-4). Plus large token buckets smooth it further. **To SEE the controller's effect, plot the FFN component** (`[body-pdr]` `ffn=` field, or `[subgroup pdr]` ffn w1w3/w2), or `ffn/attn` — NOT the headline `pdr=`. This applies to ALL controller runs (dn3/dn4 included): a working FFN controller is nearly invisible on the aggregate by construction. The measurement itself is real: `diagnostics.py` `snapshot_weights()` L604-614 (clone pre-step) + `capture_updates()` L638-650 (`‖W_after−W_before‖_F/‖W‖_F`, no m in the formula).

## WHERE WE ARE: RESOLVED ✅ (2026-06-29)
Is the FFN pdr controller's **actuation** (m → FFN body LR → ‖dW‖ → pdr) actually working live? **YES — CONFIRMED.** The open-loop `force_m=0.2` probe (`skiff-probe`, resume skiff-auto ckpt @ step 10000, world4, feedback BYPASSED) gave: attn pdr (control m=1) = 3.85e-4/3.74e-4, ffn pdr (forced m=0.2) = 8.29e-5/8.19e-5 at steps 10050/10100 → **ffn/attn = 0.215/0.219 ≈ commanded 0.2×** (natural ffn/attn≈1). Ratio STABLE over 100 steps → actuator ALIVE, plant **LINEAR** for the frozen body (no creep-back; kv2's sub-linear hint was a growing-body artifact). `[ffn-ctrl]` also showed feedback wanting m_raw=1.437 but `force_m` overrode to 0.200 → override path validated. **The remaining open work is the controller DESIGN (§THE THREE OBJECTIVES #2 partial-f / #3 tuning, Math-worthy) — NOT the wire.** The §NEXT STEP probe plan below is now DONE; kept for the record.

## THE BIG CORRECTION (do not repeat my mistake)
I initially declared "CONFIRMED broken actuator." **That was wrong — I jumped off a confounded metric.** Investigation (workflow `wb1x6gm1v`, 2 of 3 agents landed) found:
- **The wire WORKS.** `muon_fsdp2.py:455-472` (standard `Fsdp1dWork.finish`, the path dense FFN w1/w2/w3 take) reads `lr_scale_overrides.get(id(self.param),1.0)` → `effective_lr = group['lr']*lr_scale` → `param.add_(update, alpha=-effective_lr)`. So m scales the post-NS/post-tangent-projection ‖dW‖. An agent verified on the **real MuonFSDP2**: m=0.5 → ‖dW‖ halves.
- **Param ids ALIGN** across (a) controller write-set `_ffn_param_ids_for_ctrl`, (b) optimizer's stepped params, (c) diagnostics measured-set — all the same `model.named_parameters()` objects (the SAME traversal the *validated* `lr_mods` uses; per-submodule compile + FSDP2 preserve `id()`). No id-mismatch.
- **My A/B "proof" was CONFOUNDED.** The auto reference `r = K_anchor·lr/lr_anchor` self-anchors to the body's OWN frozen pdr, and the frozen base arm rides the same cosine LR, so `base_pdr(t) ≈ r(t)`. Therefore `auto/base ≈ 1` **whether the controller works or not** (working → regulates pdr *to* r≈base; dead → leaves pdr *at* base). The A/B literally can't discriminate.
- **The data is genuinely ambiguous:** over 118 matched post-latch points, `auto_ffn/base_ffn = 1.0007`, mean `m = 0.9934`. Working → ratio should = 0.993; dead → ratio = 1. Observed 1.0007 "leans dead" but it's 0.7% inside ~0.9% noise. The recent window (m≈0.975) gave ratio≈0.981 → "leans alive." Inconclusive.

## NEXT STEP: the force_m open-loop probe (definitive)
The only clean test (the closed-loop A/B can't do it):
1. **Build `force_m`** debug override in `common_fsdp2/body_lr_controller.py`: `current_multiplier()` returns `force_m` when set (bypass the feedback loop → true open-loop); add a loud "DEBUG: m pinned, feedback bypassed" banner. ~3 lines. (NOT YET BUILT as of this brief.)
2. **Test config:** copy `configs/skiff-auto.yaml` → `resume_training: true` from skiff-auto's latest checkpoint (~step 9000, post-freeze, post-anchor) + `force_m: 0.2` (a 5× cut; m can ONLY go down — m_max=1.0 is fatal-guarded, cuts-only).
3. **Run** ~200 steps on **free GPUs 0,2,3,7** (skiff-base FINISHED; running skiff-auto holds 1,4,5,6). world_size 4.
4. **Read [body-pdr] ffn pdr:**
   - drops ~5× → **actuator ALIVE**; doesn't move → **dead**.
   - drops-and-STAYS → linear plant (controller can work); drops-then-CREEPS-BACK → ‖W‖ re-equilibration (an agent's steady-state concern; would mean a constant m has reduced steady-state authority — kv2 hinted sub-linear: 19% m cut → ~10% pdr, partly offset by +13% K-drift).

## THE THREE OBJECTIVES (user's framing)
1. **Does m work at all?** → the force_m probe above.
2. **Build a controller that calibrates m alongside a GROWING f (partial-f engagement).** User's insight: the controller engages only at full freeze (f=1); the grow-then-clamp ramp — *especially the late ramp* where the body is nearly frozen (anneal mostly gone) but the controller still sits at m=1 — is uncontrolled = a mini-KH-v1. Lost-anneal is *progressive in f*, so the fix is engagement that ramps in *with f* (restore the ~f-fraction of removed anneal). Needs a reference-shape change (lr-track assumes frozen ‖W‖). **Math-worthy.**
3. **If working but too weak, tune it** (alphas/rates; possibly the sub-linear plant).

## USER'S LOSS OBSERVATION (new corroborating evidence — valuable)
Tangent projection (skiff, growth-clamped body) made the loss **much jaggier** than picket-hyg (growing body), **most on AO3** (noisiest domain), other domains barely. Theory (correct): clamping body growth removes its self-anneal → relative step (pdr) stays effectively higher → behaves like a **raised LR** → noisy batches whip the loss, worst where data is noisiest. = **visible confirmation the frozen body "runs hot"** — the exact problem the controller targets. Gives a SECOND validation signal: a working controller should *reduce* the jaggedness vs base. Not chasing ghosts.

## KEY METHODOLOGICAL TAKEAWAY for dn4
The picket/skiff-style A/B (self-anchored controller vs uncontrolled base) **CANNOT validate the controller** — they look identical by construction. Controller validation requires **open-loop** (force m, watch pdr). This reframes what dn4 can prove about the controller. Flag for Math.

## RUNNING / FINISHED EXPERIMENTS
- **skiff-auto**: RUNNING, rig-31 tmux `skiff-auto`, GPUs 1,4,5,6, ~step 9335/12000, closed-loop (inconclusive). Log `/tmp/skiff-auto.log`. Latch was clean @3450 (K_anchor=2.27e-3, lr_anchor=3.13e-4, no warn/fatal).
- **skiff-base**: FINISHED @12000. `/tmp/skiff-base.log`.
- **dn3**: KILLED 2026-06-29 (Josef). Knots-mode controller had real authority (the force_m probe confirmed the wire), but its hand-fit reference is superseded by shadow-norm; enough learned (actuator confirmed, partial-f gap, shadow-norm direction) that the marginal signal didn't justify the rig. BIG RIG NOW FREE.
- **picket-ctrl/hyg**: FINISHED — head-hygiene A/B, conclusive: control gauge ballooned to ~60% of head, treatment held ~0, identical loss (gauge CE-invisible). Head-gauge projection (Lever 1) validated free + safe.

## KEY CODE LOCATIONS
- `common_fsdp2/body_lr_controller.py`: controller. `observe()` ~280-380; `_latch_anchor` ~232; `current_multiplier()` ~278 (← add force_m here).
- `mara_fsdp2/train_mara.py`: actuator write ~2131-2134 (`lr_scale_overrides[ffn_id]=current_multiplier()`); `_ffn_param_ids_for_ctrl` ~1508; observe call ~2963; Settings validation for `ffn_pdr_controller` ~4860-5030; `anchor_step: auto` resolution ~4983; banner ~5915; `_lr_schedule_fingerprint` ~219.
- `common_fsdp2/muon_fsdp2.py`: lr_scale read+apply ~455-472 (FFN body path).
- Configs: `skiff-auto.yaml`, `skiff-base.yaml`, `dn4.yaml` (golden DRAFT — grow-then-clamp+auto-controller+head-hygiene; freeze timing 25k/45k is a PLACEHOLDER for dn3+Math), `dn3.yaml` (running).
- Tests: `common_fsdp2/test_body_lr_controller_auto.py` (31/31). Run via miniconda python (torch-free).
- Doc: `docs/MATH_AGENT_Q11_self_anchoring_pdr_controller.md`.

## SESSION FOOTGUN REMOVALS (all done + verified)
- `output_lr_batch_adjust` removed (skiff + dn4) — outmoded head-LR brake, superseded by head hygiene.
- `anchor_step: auto` — resolves the freeze step from the tangent_project_strength schedule (earliest step f hits terminal). Verified rig-side (dn4 auto→45000). dn4 dogfoods it.
- `cadence` field removed — dead config (observe fires at val_step). Replaced with a val_step∈[25,250] sanity warning + deprecation note. dn3 (running, keeps cadence) is resume-safe (ignored).

## UNCOMMITTED: everything above (auto-mode controller, trainer wiring, anchor_step:auto, cadence removal, configs, Q11 doc, tests) is UNCOMMITTED across both repos. User was offered a commit; hasn't said go.

## RIG ACCESS REMINDERS
- `ssh rig-31` works. trainenv: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate trainenv`. Repo at `~/valhalla/code/mara_fsdp2` (== V: NAS, edits live instantly).
- Standalone Settings test pattern: `python -c "import sys; sys.path.insert(0,'.'); import train_mara as T; s=T.Settings.from_yaml('configs/X.yaml')"` (run from the repo dir; fatal_error hard-exits so catch SystemExit).
- Local miniconda python `C:\Users\josef\miniconda3\python.exe` runs the torch-free controller + tests; the LOCAL trainenv torch is too old for FSDP.
