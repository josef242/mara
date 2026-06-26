#!/usr/bin/env python
"""
reformat_body_pdr_mult.py — convert [body-pdr] lines from the old single-mult format
  '| body_lr_mult=X'  ->  '| lr_mult attn=X ffn=X'

For runs with NO attn/ffn split (uniform body multiplier — e.g. the backfilled
dreadnought_v2 / mf-low-lr / keelhaul lines, all at 1.000, or any pre-split run where
attn==ffn), this is a faithful 1:1 reformat that lets the Dashboard parse a single
(new) format instead of detecting two.

DO NOT run this on a run whose attn and ffn were on DIFFERENT schedules (the old single
value only recorded one of them — reformatting to attn=X ffn=X would misrepresent the
other). It is only valid where the body multiplier was uniform.

Backs up to gen_log.txt.pre-reformat.bak; idempotent (skips lines already in new format).

Usage:
  python tools/reformat_body_pdr_mult.py keelhaul dreadnought_v2 mf-low-lr [--dry-run]
"""
import argparse, os, re

ROOT = "B:/checkpoints/current"
OLD = re.compile(r"\| body_lr_mult=([\d.]+)\s*$")


def reformat(run, dry):
    path = os.path.join(ROOT, run, "gen_log.txt")
    if not os.path.exists(path):
        print(f"!! {run}: no gen_log")
        return
    lines = open(path, "r", encoding="utf-8", errors="replace").readlines()
    out, n, already = [], 0, 0
    sample = None
    for ln in lines:
        raw = ln.rstrip("\n")
        if "[body-pdr]" not in raw:
            out.append(ln)
            continue
        if "lr_mult attn=" in raw:          # already new format
            already += 1
            out.append(ln)
            continue
        m = OLD.search(raw)
        if not m:
            out.append(ln)
            continue
        x = m.group(1)
        raw2 = OLD.sub(f"| lr_mult attn={x} ffn={x}", raw)
        out.append(raw2 + "\n")
        n += 1
        if sample is None:
            sample = raw2.strip()
    print(f"== {run}: reformatted {n} | already-new {already}")
    if sample:
        print(f"     e.g. {sample}")
    if dry:
        print(f"   (dry-run — {path} NOT modified)")
        return
    if n == 0:
        print("   nothing to do")
        return
    bak = path + ".pre-reformat.bak"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").writelines(lines)
        print(f"   backup -> {bak}")
    else:
        print(f"   backup already exists ({bak}) — left as-is")
    open(path, "w", encoding="utf-8").writelines(out)
    print(f"   wrote {path}")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["keelhaul", "dreadnought_v2", "mf-low-lr"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    ROOT = args.root
    runs = args.runs if args.runs else ["keelhaul", "dreadnought_v2", "mf-low-lr"]
    for run in runs:
        reformat(run, args.dry_run)


if __name__ == "__main__":
    main()
