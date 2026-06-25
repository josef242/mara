# Gradient Productivity Metric (GPM)

**A live, per-step measure of whether your gradients are actually *productive* — i.e. whether
big gradient steps are buying you real loss reduction, or just noise/churn.** Shown on the
training status line as `gpm: +0.31/+0.25` (GPM-S/GPM-L) and retrofittable into historical logs
for the Dashboard.

## The question it answers
You've probably noticed it watching training: *"Whoa, that was a big norm spike... and the next
step the loss dropped a ton."* GPM quantifies exactly that intuition:

> When grad-norm spikes **above its local trend**, does loss drop **more than its local trend**
> on the **next** step?

- **GPM > 0** → big gradients here are *productive* (norm spikes predict loss drops). Strong
  learning signal. A high grad-norm floor with high GPM is *high-quality* norm, not a problem.
- **GPM ≈ 0** → spikes are *noise* (uncorrelated with improvement).
- **GPM < 0** → spikes are *anti-productive* (norm spikes precede loss *increases*). A struggling
  regime. (The older `dreadnought` run was negative for its first ~5k steps before it found
  productive directions; DN2/`dreadnought_v2` only dips briefly negative mid-run, ~st 3k–6k.)

## Why it's not a trivial "norm tracks learning" tautology
Both grad-norm and loss-drops are large early and small late — they ride the same training
envelope. A naive correlation of raw nrm vs raw loss-drop would be positive **for any run**, just
because both shrink over training. That tells you nothing.

GPM avoids this by **detrending**: it subtracts a rolling median from BOTH nrm and the loss-drop
(over a window W), then correlates the **residuals**. So it measures whether *local fluctuations*
in norm predict *local* improvement — above and beyond the shared envelope. It uses a **Spearman
(rank) correlation**, which is robust to the heavy-tailed norm spikes that are the whole point.

## The honest caveat (what GPM is NOT)
Step N's loss and step N+1's loss are computed on **different batches**. So GPM is *not* same-batch
update effectiveness (which would require re-running the same batch before/after the step). What it
**is**: a measure of batch-to-batch **transfer** — do high-norm updates help the *next, unseen*
batch? That's arguably a *generalization* signal, and the window smooths out per-batch luck.
(If we ever want the true same-batch number to calibrate this, a one-step before/after probe on a
fixed batch would do it.)

## The two windows: GPM-S and GPM-L
GPM is reported as a **short/long pair** because the window size is itself meaningful:
- **W too small** → batch-difficulty noise dominates → GPM ≈ 0 (no signal)
- **W too large** → just re-confirms the envelope → GPM → trivially high (tautology)
- the real signal lives in between (on our runs GPM is flat ~+0.25 from W=15 to W=401).

So:
- **GPM-S** (short, default W=15) — responsive; reacts within ~15 steps. Jumpy by design.
- **GPM-L** (long, default W=101) — stable baseline of the regime's productivity.

**The GAP between them is its own signal** (like a fast/slow moving-average crossover):
- `gpm: +0.31/+0.25` — **S > L**: productivity *increasing right now* (breakthrough/grok igniting)
- `gpm: +0.18/+0.25` — **S < L**: productivity dipping (plateau/saturation starting) — early warning
- `gpm: +0.25/+0.25` — steady

(There is no trend arrow on the status line — trending is read on the Dashboard, which plots both
the GPM-S and GPM-L curves, so the S-vs-L crossover is visible there. The status line stays compact:
just the two numbers. This also keeps the live tag byte-identical to the retrofit injector's output.)

## What we've learned with it (validated result)
At W=51 (centered, full-run), across the from-scratch ultra-deep runs:

| run | regime | GPM (centered W=51) | early phase |
|---|---|---|---|
| **KeelHaul** | tangent-projection + low WD | **+0.25** (rising to +0.40) | productive from step 1 |
| mf-low-lr | normal NorMuon | +0.13 | gently positive, decays late |
| **DN2** (`dreadnought_v2`) | normal NorMuon | **+0.18** | positive +0.32 early, brief dip to −0.05 (~st 3k–6k), recovers |
| dreadnought (older run) | normal NorMuon | +0.07 | **NEGATIVE −0.17 for the first ~5k steps**, crosses positive ~st 5k |

DN2 = `dreadnought_v2` is the canonical comparison run (longer, more recent; ~18k steps). The
older `dreadnought` is the run that had the persistent early-negative phase — a useful contrast,
but **not** the one we benchmark against. Both are retrofitted with the live `gpm:` field.

**The tangent-projected regime has ~2-3.5× more productive gradients**, and — unlike the
unprojected runs — **zero anti-productive phase**. Mechanism: tangent projection removes the
loss-null radial component of the Muon update (the "shock absorber"), so 100% of the update is
loss-relevant. A norm spike is therefore necessarily a spike in *useful* motion → it reliably
produces a loss drop → higher GPM. The projection didn't just flatten body norms; **it made the
gradients more honest.** (This also resolved a worry that KeelHaul's high grad-norm floor was bad:
it's not — it's high-*quality* norm, and the right lever is to RAISE the clip threshold, not lower
LR, since the gradients being clipped are productive.)

## Reading it live
On the status line: `... | nrm: 1.49 | gpm: +0.31/+0.25 | ...` (GPM-S/GPM-L)
- Both positive and healthy (>+0.1) → gradients productive
- S pulling above L → learning accelerating
- S dropping below L → plateau/saturation forming
- Either going negative → trouble (gradients fighting the loss); investigate

(Trend is read on the Dashboard, which plots both curves; the status line shows just the two numbers.)

## Enabling it (live, in training)
In the run config YAML:
```yaml
track_gpm: true
gpm_window_short: 15    # optional (default 15)
gpm_window_long: 101    # optional (default 101)
```
Implementation: `GPMTracker` in `train_mara.py` (rolling deque + detrended lagged Spearman).
Opt-in, negligible compute (a Spearman over ≤101 points once per logged step), rank0 only.

**Resume / restart behavior (no seam).** The rolling buffer is NOT checkpointed, but on resume
`GPMTracker.seed_from_log()` warm-starts it from the gen_log: it loads the `(nrm, ls)` of training
lines with `step < start_step` (the last `w_long+2` of them) — exactly what the buffer would have
held had the run never paused. So GPM-L stays continuous across a checkpoint restart instead of
re-warming from a cold short-memory window (which would show as `S == L` for ~`w_long` steps right
after the resume — the "seam"). It no-ops cleanly on a fresh run (no log / `start_step ≤ 1`) and is
best-effort (a parse failure never breaks the resume; you just get the cold warm-up). A log line
`GPM: warm-started buffer with N pre-resume points` confirms it fired. (Runs resumed *before* this
fix existed carry a one-time cosmetic seam in their gen_log at the resume point; it self-heals in
~`w_long` steps and the Dashboard barely shows it.)

## The tool: `tools/gpm.py` (offline analysis + retrofit)
```bash
# Window sweep + over-time for one run (find the meaningful window scale):
python gpm.py --log <run>/gen_log.txt

# Compare runs:
python gpm.py --compare keelhaul,dreadnought,mf-low-lr

# RETROFIT -> per-step CSV (ad-hoc analysis, plotting outside the Dashboard):
python gpm.py --retrofit <run>/gen_log.txt --out <run>_gpm.csv
#   emits: step,ls,nrm,gpm_s,gpm_l   (gpm blank until the trailing window fills)

# Machine-readable summary:
python gpm.py --log <run>/gen_log.txt --json
```

### Retrofitting the field INTO a historical gen_log (for the Dashboard): `tools/gpm_retrofit_inject.py`
The Dashboard parses `gpm:` straight out of each gen_log step line (same as `nrm:`, `ls:`). To make
a run that finished *before* the metric existed (DN2, mf-low-lr) show GPM, splice the live-format
field directly into its gen_log:
```bash
python gpm_retrofit_inject.py <run>/gen_log.txt --dry-run   # preview counts + a sample line
python gpm_retrofit_inject.py <run>/gen_log.txt             # in place (writes a timestamped .bak)
```
- Appends ` | gpm: +0.31/+0.25` to each **training** line (the ones with `nrm:` + `t_tk:`),
  **byte-identical** to what the live `GPMTracker.status_tag` writes going forward. Eval/AVG lines,
  headers, and the first <5 steps (where GPM isn't defined yet) are untouched — exactly as live.
- Replays `GPMReplay` from gpm.py (shared `gpm_window` math), so retrofitted == live.
- **Safe**: timestamped `.bak` first, atomic replace, idempotent (re-running won't double-inject).
- Applied 2026-06-24 to **`dreadnought_v2` (DN2, 19,879 lines)**, `dreadnought` (older run, 16,748
  lines), and `mf-low-lr` (36,467 lines). The older `dreadnought` reproduces the documented
  early-negative signature (NEGATIVE for the first ~5k steps, crossing positive ~st 5000); DN2
  (`dreadnought_v2`) starts positive (+0.32), dips briefly to ~−0.05 around st 3k–6k, then recovers.

### Retrofit == Live (important for the Dashboard)
`--retrofit` replays the log through the **same trailing-window algorithm** the in-trainer
`GPMTracker` uses (`GPMReplay` in gpm.py shares the core `gpm_window` math). So a retrofitted
historical run (DN2, mf) and a future live run (KeelHaul-v2) are computed **identically** and are
directly comparable on the Dashboard. Use the **same `--short`/`--long`** as the live config
(defaults 15/101 match the trainer defaults).

Note: the offline `--log`/`--compare` sweep uses a **centered** rolling-median detrend (symmetric
window per point — best for *analysis*, "what's the true productivity here"); `--retrofit` and the
live tracker use a **trailing** window (only past data — required for a real-time/streaming metric).
They're both valid and close; just don't mix a centered summary number with a trailing Dashboard
series and call them the same thing.
```
