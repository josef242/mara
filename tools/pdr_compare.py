#!/usr/bin/env python
"""
pdr_compare.py — overlay body-pdr (and val loss) across runs on matched tokens.

Reconstructs the [body-pdr] metric from diagnostics.jsonl so it works on runs
that predate the live log line (DN2, KH-v1). The metric is the MEAN of per-layer
param_delta_ratio (= ||dW||/||W|| per matrix), split attn / ffn / body(=both).
Validated against kv2's logged [body-pdr] lines (matches to ~3 sig figs).

Usage:
  python tools/pdr_compare.py                      # default kv2 vs keelhaul vs dreadnought_v2
  python tools/pdr_compare.py kv2 keelhaul dreadnought_v2 dreadnought
  python tools/pdr_compare.py --root B:/checkpoints/current --comp ffn
  python tools/pdr_compare.py --loss               # also overlay val-loss AVG vs tokens
"""
import argparse, json, os, sys

DEFAULT_ROOT = "B:/checkpoints/current"
DEFAULT_RUNS = ["kv2", "keelhaul", "dreadnought_v2"]


def parse_jsonl_tolerant(path):
    """Yield JSON objects from a file that may have blank lines or concatenated objects."""
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            idx = 0
            n = len(s)
            while idx < n:
                # skip whitespace between concatenated objects
                while idx < n and s[idx] in " \t\r\n":
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = dec.raw_decode(s, idx)
                except json.JSONDecodeError:
                    break  # give up on the remainder of this physical line
                yield obj
                idx = end


def body_pdr(rec, comp="body"):
    """Mean per-layer param_delta_ratio for attn / ffn / body(both)."""
    layers = rec.get("layers", [])
    if not layers:
        return None
    vals = []
    for L in layers:
        if comp in ("attn", "body") and "attn" in L:
            r = L["attn"].get("param_delta_ratio")
            if r is not None:
                vals.append(r)
        if comp in ("ffn", "body") and "ffn" in L:
            r = L["ffn"].get("param_delta_ratio")
            if r is not None:
                vals.append(r)
    return sum(vals) / len(vals) if vals else None


def load_run_pdr(root, run, comp):
    """Return list of (tokens_M, step, pdr) for a run."""
    path = os.path.join(root, run, "diagnostics.jsonl")
    if not os.path.exists(path):
        return None
    out = []
    for rec in parse_jsonl_tolerant(path):
        if "step" not in rec:
            continue
        p = body_pdr(rec, comp)
        if p is None:
            continue
        out.append((rec.get("total_tokens", 0) / 1e6, rec["step"], p))
    return out


def load_run_loss(root, run):
    """Return list of (tokens_M, step, avg_loss) parsed from val_log.txt."""
    path = os.path.join(root, run, "val_log.txt")
    if not os.path.exists(path):
        return None
    out = []
    with open(path, "r") as f:
        for line in f:
            if "AVG:" not in line or "tok:" not in line:
                continue
            try:
                tok = float(line.split("tok:")[1].split("|")[0].strip()) / 1e6
                st = int(line.split("st:")[1].split("|")[0].strip())
                avg = float(line.split("AVG:")[1].split("[")[0].strip())
                out.append((tok, st, avg))
            except (IndexError, ValueError):
                continue
    return out


def interp_at(series, tok_targets):
    """Linear-interpolate value vs tokens at each target token (M). series=[(tok,_,val)]."""
    pts = sorted((t, v) for t, _, v in series)
    res = []
    for tt in tok_targets:
        if tt < pts[0][0] or tt > pts[-1][0]:
            res.append(None)
            continue
        # find bracketing
        lo = max(p for p in pts if p[0] <= tt)
        hi = min(p for p in pts if p[0] >= tt)
        if lo[0] == hi[0]:
            res.append(lo[1])
        else:
            f = (tt - lo[0]) / (hi[0] - lo[0])
            res.append(lo[1] + f * (hi[1] - lo[1]))
    return res


def ascii_chart(runs_data, tok_grid, fmt="{:.2f}", height=18, label="pdr (e-3)"):
    """runs_data: dict run -> list of values (aligned to tok_grid), in display units."""
    allv = [v for vals in runs_data.values() for v in vals if v is not None]
    if not allv:
        print("  (no data to chart)")
        return
    vmin, vmax = min(allv), max(allv)
    if vmax == vmin:
        vmax = vmin + 1
    span = vmax - vmin
    marks = {run: chr(ord("A") + i) for i, run in enumerate(runs_data)}
    grid = [[" "] * len(tok_grid) for _ in range(height)]
    for run, vals in runs_data.items():
        m = marks[run]
        for x, v in enumerate(vals):
            if v is None:
                continue
            y = int((v - vmin) / span * (height - 1))
            y = max(0, min(height - 1, y))
            grid[height - 1 - y][x] = m
    print(f"  {label}   [{vmin:.2f} .. {vmax:.2f}]   legend: " +
          "  ".join(f"{m}={run}" for run, m in marks.items()))
    for row in grid:
        print("   |" + "".join(row))
    print("   +" + "-" * len(tok_grid))
    # token axis labels (sparse)
    axis = [" "] * (len(tok_grid) + 1)
    for x in range(0, len(tok_grid), max(1, len(tok_grid) // 8)):
        lab = f"{int(tok_grid[x])}"
        for i, ch in enumerate(lab):
            if x + i < len(axis):
                axis[x + i] = ch
    print("    " + "".join(axis) + "  (tokens, M)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=None)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--comp", default="body", choices=["body", "attn", "ffn"])
    ap.add_argument("--loss", action="store_true", help="also overlay val-loss")
    ap.add_argument("--max-tok", type=float, default=None, help="clip token axis (M)")
    ap.add_argument("--ncols", type=int, default=60)
    args = ap.parse_args()
    runs = args.runs if args.runs else DEFAULT_RUNS

    series = {}
    for run in runs:
        s = load_run_pdr(args.root, run, args.comp)
        if not s:
            print(f"!! {run}: no pdr data (missing diagnostics?)", file=sys.stderr)
            continue
        series[run] = s

    if not series:
        print("No runs loaded.")
        return

    # token grid: from max-of-mins to min-of-maxes (common overlap), unless clipped
    lo = max(min(t for t, _, _ in s) for s in series.values())
    hi = min(max(t for t, _, _ in s) for s in series.values())
    if args.max_tok:
        hi = min(hi, args.max_tok)
    tok_grid = [lo + (hi - lo) * i / (args.ncols - 1) for i in range(args.ncols)]

    print(f"\n=== body-pdr overlay [comp={args.comp}] — runs: {', '.join(series)} ===")
    print(f"    common token overlap: {lo:.0f}M .. {hi:.0f}M  ({args.comp} pdr in units of e-3)\n")

    # table at a sparse set of token checkpoints
    checkpoints = [lo + (hi - lo) * f for f in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)]
    hdr = "  tok(M) | " + " | ".join(f"{r:>14s}" for r in series)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    interp = {run: interp_at(s, checkpoints) for run, s in series.items()}
    for i, tt in enumerate(checkpoints):
        row = f"  {tt:6.0f} | "
        cells = []
        for run in series:
            v = interp[run][i]
            cells.append(f"{v*1e3:14.3f}" if v is not None else f"{'—':>14s}")
        print(row + " | ".join(cells))

    print()
    chart = {run: [v * 1e3 if v is not None else None for v in interp_at(s, tok_grid)]
             for run, s in series.items()}
    ascii_chart(chart, tok_grid, label=f"{args.comp} pdr (e-3)")

    if args.loss:
        lseries = {}
        for run in runs:
            s = load_run_loss(args.root, run)
            if s:
                lseries[run] = s
        if lseries:
            llo = max(min(t for t, _, _ in s) for s in lseries.values())
            lhi = min(max(t for t, _, _ in s) for s in lseries.values())
            if args.max_tok:
                lhi = min(lhi, args.max_tok)
            ltok = [llo + (lhi - llo) * i / (args.ncols - 1) for i in range(args.ncols)]
            print(f"\n=== val-loss AVG overlay — common overlap {llo:.0f}M .. {lhi:.0f}M ===\n")
            lchart = {run: interp_at(s, ltok) for run, s in lseries.items()}
            ascii_chart(lchart, ltok, label="val AVG loss")


if __name__ == "__main__":
    main()
