#!/usr/bin/env python
"""
backfill_body_pdr.py — inject [body-pdr] lines into the gen_log of runs that
predate the metric, so the Dashboard (which reads gen_log) can plot them.

The live metric is emitted in train_mara.py as:
    [body-pdr] pdr={_med:.3e} (attn={_amed:.3e} ffn={_fmed:.3e}) | body_lr_mult={_bmult:.3f}
where _med  = MEDIAN of (all attn + all ffn per-layer param_delta_ratio),
      _amed = MEDIAN of attn param_delta_ratio,
      _fmed = MEDIAN of ffn  param_delta_ratio.
We replicate that EXACTLY (median, not mean) so injected lines match kv2's real ones.

body_lr_mult is hardcoded 1.000 for these runs: none of them ran a body-LR anneal
(that schedule is the kv2 experiment). 1.000 is the truthful value — full body LR.

Injection point: inside each "=== Diagnostics @ step N ===" block, immediately
after that block's timestamped "ffn update:" line (mirrors the live emit order).
We key blocks by step number (robust to partial lines), match each to its
diagnostics.jsonl record by step, compute the medians, and insert.

Safety: writes gen_log.txt.pre-pdr-backfill.bak first; idempotent (skips a block
that already has a [body-pdr] line); --dry-run shows samples without writing.
"""
import argparse, json, os, re, sys

ROOT = "B:/checkpoints/current"
STEP_RE = re.compile(r"=== Diagnostics @ step (\d+) ===")
FFN_UPDATE_RE = re.compile(r"^\s*\d[\d\-: ]*\|\s+ffn update:")
TS_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|)")


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")  # matches train_mara's [len//2]


def parse_jsonl_tolerant(path):
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            idx, n = 0, len(s)
            while idx < n:
                while idx < n and s[idx] in " \t\r\n":
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = dec.raw_decode(s, idx)
                except json.JSONDecodeError:
                    break
                yield obj
                idx = end


def pdr_by_step(diag_path):
    """step -> (med, amed, fmed) using the SAME median logic as train_mara.py."""
    out = {}
    for rec in parse_jsonl_tolerant(diag_path):
        step = rec.get("step")
        layers = rec.get("layers")
        if step is None or not layers:
            continue
        att = [L["attn"]["param_delta_ratio"] for L in layers
               if L.get("attn", {}).get("param_delta_ratio") is not None]
        ffn = [L["ffn"]["param_delta_ratio"] for L in layers
               if L.get("ffn", {}).get("param_delta_ratio") is not None]
        allr = att + ffn
        if not allr:
            continue  # this step predates the field — no pdr to backfill
        out[step] = (median(allr), median(att), median(ffn))
    return out


def make_line(ts_prefix, med, amed, fmed, mult=1.000):
    # EXACT format from train_mara.py (logger prepends the "TS |" itself; here the
    # block lines already carry the prefix, so we attach it to match neighbors).
    body = (f"  [body-pdr] pdr={med:.3e} (attn={amed:.3e} ffn={fmed:.3e}) "
            f"| body_lr_mult={mult:.3f}")
    return f"{ts_prefix} {body}\n"


def backfill(run, dry_run=False):
    rundir = os.path.join(ROOT, run)
    gen_path = os.path.join(rundir, "gen_log.txt")
    diag_path = os.path.join(rundir, "diagnostics.jsonl")
    if not (os.path.exists(gen_path) and os.path.exists(diag_path)):
        print(f"!! {run}: missing gen_log or diagnostics — skipped")
        return

    pdr = pdr_by_step(diag_path)
    if not pdr:
        print(f"!! {run}: no per-layer param_delta_ratio in diagnostics — nothing to backfill")
        return

    with open(gen_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    out = []
    i = 0
    n = len(lines)
    injected, skipped_exists, skipped_nodata, blocks = 0, 0, 0, 0
    samples = []
    while i < n:
        line = lines[i]
        m = STEP_RE.search(line)
        if not m:
            out.append(line)
            i += 1
            continue
        # entered a diagnostics block for this step
        step = int(m.group(1))
        blocks += 1
        out.append(line)
        i += 1
        # scan forward within the block to the 'ffn update:' anchor; stop if we hit
        # the next block or an existing [body-pdr] (idempotency).
        already = False
        inserted = False
        while i < n and not STEP_RE.search(lines[i]):
            cur = lines[i]
            if "[body-pdr]" in cur:
                already = True
            out.append(cur)
            i += 1
            if (not inserted and not already and FFN_UPDATE_RE.match(cur)
                    and step in pdr):
                tsm = TS_PREFIX_RE.match(cur)
                ts_prefix = tsm.group(1) if tsm else cur.split("|")[0].strip() + " |"
                med, amed, fmed = pdr[step]
                newline = make_line(ts_prefix, med, amed, fmed)
                out.append(newline)
                inserted = True
                injected += 1
                if len(samples) < 3:
                    samples.append((step, newline.rstrip("\n")))
        if already:
            skipped_exists += 1
        elif not inserted:
            skipped_nodata += 1  # block had no matching pdr record (early-schema step)

    print(f"== {run}: {blocks} diagnostics blocks | injected {injected} | "
          f"already-had {skipped_exists} | no-pdr-data {skipped_nodata}")
    for st, s in samples:
        print(f"     e.g. step {st}: {s.strip()}")

    if dry_run:
        print(f"   (dry-run — {gen_path} NOT modified)")
        return

    bak = gen_path + ".pre-pdr-backfill.bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"   backup -> {bak}")
    else:
        print(f"   backup already exists ({bak}) — left as-is")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"   wrote {gen_path} (+{injected} lines)")


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
        backfill(run, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
