# kv2 body-LR anneal — alternative schedules (pre-staged for a fast restart)

If the pdr-overshoot monitor (`tools/pdr_overshoot_monitor.py`) shows kv2's pdr flattening ABOVE
DN2's peak (~2.73e-3) and not decaying by step ~1500-2000, pull the first transition back so pdr
tops out near DN2's ceiling instead of overshooting it ~20%. These are drop-in `lr_mods` replacements
for the line in `kv2.yaml`. Restarting is required (lr_mods is read at startup, not hot-editable).

Context (from the monitor, 2026-06-25):
- DN2 (target) pdr PEAKED at 2.73e-3 at step 2200 (433M tok), then decayed on its own.
- KH-v1 (unthrottled, kv2's twin so far) reached 3.29e-3 by step 2680 — overshot DN2's peak ~20%.
- kv2 is tracking KH-v1 almost exactly → expect a similar ~20% overshoot before the current 2680
  transition engages. (Mid-warmup projections to 7e-3 are an LR-ramp artifact — ignore those.)

## CURRENT (transition @ 2680 ≈ 350M tok) — what's running now
```yaml
lr_mods:
  - [all, all, [[0, 1.0], [2680, 1.0], [4290, 0.75], [5060, 0.67], [6580, 0.55]]]
```

## OPTION A — transition @ 1500 (warmup end, ≈197M tok) — most aggressive
Starts the bend the moment LR caps. pdr never gets a flat-at-full-LR plateau to climb.
```yaml
lr_mods:
  - [all, all, [[0, 1.0], [1500, 1.0], [3110, 0.75], [3880, 0.67], [5400, 0.55]]]
```
(0.75 @ 408M tok, 0.67 @ 509M tok)

## OPTION B — transition @ 2000 (≈262M tok) — aligned to DN2's peak timing  ← likely pick
Lets pdr peak around where DN2 peaked (step ~2200), then bends down — closest to reproducing DN2's
actual pdr trajectory. Preserves a bit more of kv2's early lead than Option A.
```yaml
lr_mods:
  - [all, all, [[0, 1.0], [2000, 1.0], [3610, 0.75], [4380, 0.67], [5900, 0.55]]]
```
(0.75 @ 473M tok, 0.67 @ 624M tok)

## Decision rule
Re-run `python tools/pdr_overshoot_monitor.py` at step ~1500-2000:
- pdr flattening AT/BELOW ~2.7e-3 and stable/decaying → 2680 is fine, **do nothing**.
- pdr flat ABOVE ~2.7e-3 and still rising → restart with **Option B** (or A if it's well above 3e-3).
- All targets/keep-everything-else (projection on, WD 0.002, clip 3.0/2.0) unchanged — the schedule
  is the only edit.
