#!/usr/bin/env python
"""Rebuild token-river shards into DOCUMENT-ALIGNED shards.

The existing tokenized shards are a continuous token river chopped every
~100M tokens: documents straddle shard files, so every shard TRANSITION in
the dataloader splices two mid-documents together with no BOS between them
(the doc-attention mask then treats the splice as one document), and the
loader's abandon-leftover shortcut punches a small hole at the same seam.

This tool re-chunks the river so that every output shard starts at a BOS
and contains only whole documents. It also nudges the shard COUNT of each
(group, split) to be coprime with --coprime (default 6), so the loader's
stride-by-world_size shard walk (dataloader.py: shard_idx = (rank + pos*W)
% S) gives every rank the FULL shard set for any world size of the form
2^a * 3^b (2, 4, 6, 8, 12, ...) — no more rank pairs sharing partial
orbits (the W=8 x S=60 launch-ledger issue).

The input tree is never modified. Output is a parallel tree:

    <out>/<group>/<group>_<split>_NNNNNN.npy     (+ manifest_<split>.json)

Content parity: concat(output shards) == input river from its first BOS.
Tokens before the river's first BOS (a headless fragment, at most one
partial doc per group) are dropped and logged. Docs are never split; a
single doc larger than --cap becomes its own oversized shard (warned).
Each shard is read back and byte-compared after writing (--no-readback to
skip). Loader note: shards below B*T+1 tokens are skipped by the loader;
--min-shard (default 5M) keeps the final shard far above that.

Usage (rig, CPU-only; safe to nohup):
    python tools/rechunk_doc_aligned.py \
        --root /home/josef/valhalla/notebooks/datasets/tokenized/llama \
        --out  /home/josef/valhalla/notebooks/datasets/tokenized/llama_docaligned \
        --bos 1 \
        --groups stories ao3 books code_python code_c preselect edufineweb_1.5TT

    # afterwards, or on a rebuilt tree at any time:
    python tools/rechunk_doc_aligned.py --out <out> --groups ... --verify-only

Disk: output ~= input size per group (this is a duplicate). Check NAS free
space first — edufineweb-scale groups are multi-TB.
"""

import argparse
import bisect
import glob
import hashlib
import json
import math
import os
import time

import numpy as np

CHUNK = 1 << 26  # 64M tokens per streaming read


def log(msg):
    print(f"[rechunk] {msg}", flush=True)


class River:
    """Read-only view over an ordered list of 1-D .npy token shards as one stream."""

    def __init__(self, paths):
        self.paths = paths
        self.arrays = [np.load(p, mmap_mode="r") for p in paths]
        for p, a in zip(paths, self.arrays):
            if a.ndim != 1:
                raise ValueError(f"{p}: expected 1-D token array, got shape {a.shape}")
            if a.dtype != self.arrays[0].dtype:
                raise ValueError(f"{p}: dtype {a.dtype} != {self.arrays[0].dtype}")
        self.dtype = self.arrays[0].dtype
        self.lengths = [int(a.shape[0]) for a in self.arrays]
        self.offsets = [0]
        for n in self.lengths:
            self.offsets.append(self.offsets[-1] + n)
        self.total = self.offsets[-1]

    def slice(self, start, end):
        """tokens[start:end) across file boundaries, as one in-memory array."""
        assert 0 <= start <= end <= self.total, (start, end, self.total)
        parts = []
        i = bisect.bisect_right(self.offsets, start) - 1
        pos = start
        while pos < end:
            lo = pos - self.offsets[i]
            hi = min(end - self.offsets[i], self.lengths[i])
            parts.append(np.asarray(self.arrays[i][lo:hi]))
            pos = self.offsets[i] + hi
            i += 1
        if not parts:
            return np.empty(0, dtype=self.dtype)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


class BosStream:
    """Monotone access to a river's BOS positions with O(window) memory.

    Queries must be non-decreasing (the planner walks left to right); positions
    already passed are dropped via drop_below()."""

    def __init__(self, river, bos):
        self._gen = self._chunks(river, bos)
        self.buf = np.empty(0, dtype=np.int64)
        self.exhausted = False
        self.count = 0  # total BOS positions seen so far

    @staticmethod
    def _chunks(river, bos):
        for i, a in enumerate(river.arrays):
            base = river.offsets[i]
            for s in range(0, river.lengths[i], CHUNK):
                block = np.asarray(a[s : s + CHUNK])
                idx = np.flatnonzero(block == bos).astype(np.int64)
                if idx.size:
                    yield idx + (base + s)

    def _extend_past(self, x):
        while not self.exhausted and (self.buf.size == 0 or self.buf[-1] <= x):
            try:
                nxt = next(self._gen)
            except StopIteration:
                self.exhausted = True
                return
            self.count += int(nxt.size)
            self.buf = np.concatenate([self.buf, nxt]) if self.buf.size else nxt

    def drop_below(self, x):
        i = np.searchsorted(self.buf, x, "left")
        if i:
            self.buf = self.buf[i:]

    def last_le(self, x):
        """Largest not-yet-dropped BOS position <= x, or None."""
        self._extend_past(x)
        i = np.searchsorted(self.buf, x, "right") - 1
        return int(self.buf[i]) if i >= 0 else None

    def first_gt(self, x):
        """Smallest not-yet-dropped BOS position > x, or None."""
        self._extend_past(x)
        i = np.searchsorted(self.buf, x, "right")
        return int(self.buf[i]) if i < self.buf.size else None

    def drain(self, total):
        self._extend_past(total)


def plan_shards(river, bos, cap, min_shard):
    """Greedy doc-aligned packing. Returns (bounds [(start,end)...], head_tokens,
    total_docs, n_oversized). Never splits a document."""
    bs = BosStream(river, bos)
    first = bs.first_gt(-1)
    if first is None:
        raise ValueError("no BOS token found in river -- wrong --bos id?")
    head = first  # tokens before the first document start are dropped
    starts = [first]
    oversized = 0
    cur = first
    while cur + cap < river.total:
        bs.drop_below(cur + 1)
        cut = bs.last_le(cur + cap)
        if cut is None or cut <= cur:
            # the doc starting at cur is longer than cap: it becomes its own shard
            cut = bs.first_gt(cur)
            if cut is None or cut >= river.total:
                break
            oversized += 1
        starts.append(cut)
        cur = cut
    bs.drain(river.total)
    # merge an undersized tail into the previous shard (loader skips tiny shards)
    if len(starts) > 1 and river.total - starts[-1] < min_shard:
        log(f"  merging {river.total - starts[-1]:,}-token tail into previous shard")
        starts.pop()
    bounds = [(s, e) for s, e in zip(starts, starts[1:] + [river.total])]
    return bounds, head, bs.count, oversized


def find_split(river, bos, start, end):
    """BOS position strictly inside (start, end) nearest the midpoint, or None."""
    mid = (start + end) // 2
    best = None
    for s in range(start, end, CHUNK):
        block = river.slice(s, min(s + CHUNK, end))
        idx = np.flatnonzero(block == bos)
        for p in (idx + s):
            p = int(p)
            if p <= start:
                continue
            if best is None or abs(p - mid) < abs(best - mid):
                best = p
    return best


def adjust_coprime(river, bos, bounds, coprime):
    """Split largest shards at doc boundaries until gcd(len(bounds), coprime) == 1."""
    guard = 0
    while math.gcd(len(bounds), coprime) != 1 and guard < 8:
        guard += 1
        for idx in sorted(range(len(bounds)), key=lambda i: bounds[i][0] - bounds[i][1]):
            s, e = bounds[idx]
            sp = find_split(river, bos, s, e)
            if sp is not None:
                bounds[idx : idx + 1] = [(s, sp), (sp, e)]
                log(f"  orbit fix: split shard {idx + 1} at doc boundary -> S={len(bounds)}")
                break
        else:
            log("  WARNING: no splittable shard found; shard count left non-coprime")
            return bounds
    return bounds


def write_shards(river, bos, bounds, out_dir, group, split, readback):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i, (s, e) in enumerate(bounds, start=1):
        buf = river.slice(s, e)
        if int(buf[0]) != bos:
            raise AssertionError(f"shard {i} does not start with BOS — planner bug")
        docs = int((buf == bos).sum())
        digest = hashlib.blake2b(buf.tobytes(), digest_size=16).hexdigest()
        name = f"{group}_{split}_{i:06d}.npy"
        final = os.path.join(out_dir, name)
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, buf)
        os.replace(tmp, final)
        if readback:
            back = np.load(final, mmap_mode="r")
            if back.shape[0] != buf.shape[0]:
                raise AssertionError(f"{name}: read-back length mismatch")
            for c in range(0, buf.shape[0], CHUNK):
                if not np.array_equal(np.asarray(back[c : c + CHUNK]), buf[c : c + CHUNK]):
                    raise AssertionError(f"{name}: read-back byte mismatch at {c}")
        entries.append({
            "file": name, "start": s, "end": e,
            "tokens": int(e - s), "docs": docs,
            "blake2b16": digest,
        })
        log(f"  wrote {name}: {e - s:,} tokens, {docs:,} docs")
    return entries


def process_group(root, out_root, group, split, args):
    in_dir = os.path.join(root, group)
    out_dir = os.path.join(out_root, group)
    manifest_path = os.path.join(out_dir, f"manifest_{split}.json")
    if os.path.exists(manifest_path) and not args.force:
        log(f"{group}/{split}: manifest exists, skipping (--force to redo)")
        return
    paths = sorted(glob.glob(os.path.join(in_dir, f"*_{split}_*.npy")))
    if not paths:
        log(f"{group}/{split}: no input shards, skipping")
        return
    t0 = time.time()
    river = River(paths)
    log(f"{group}/{split}: {len(paths)} input shards, {river.total:,} tokens ({river.dtype})")

    bounds, head, total_docs, oversized = plan_shards(river, args.bos, args.cap, args.min_shard)
    # Sanity guards: a legitimate river-chop head fragment is at most one partial
    # document, and no pretraining group has multi-million-token mean docs. Either
    # tripping means --bos is not this group's document delimiter -- abort BEFORE
    # writing a plausible-looking but semantically wrong tree.
    if head > 10_000_000 or head > 0.02 * river.total:
        raise ValueError(
            f"headless fragment is {head:,} tokens ({head / river.total:.1%} of river) "
            f"-- token id {args.bos} does not look like this group's document delimiter")
    mean_doc = (river.total - head) / max(total_docs, 1)
    if mean_doc > 5_000_000:
        raise ValueError(
            f"mean document length {mean_doc:,.0f} tokens "
            f"-- token id {args.bos} does not look like this group's document delimiter")
    if head:
        log(f"  dropping {head:,}-token headless fragment (river starts mid-document)")
    if oversized:
        log(f"  WARNING: {oversized} single documents exceed cap; kept as oversized shards")
    bounds = adjust_coprime(river, args.bos, bounds, args.coprime)
    S = len(bounds)
    log(f"  plan: {S} shards (gcd(S,{args.coprime})={math.gcd(S, args.coprime)}), "
        f"{total_docs:,} docs, mean doc {(river.total - head) / max(total_docs, 1):,.0f} tok")

    entries = write_shards(river, args.bos, bounds, out_dir, group, split, not args.no_readback)

    out_tokens = sum(x["tokens"] for x in entries)
    out_docs = sum(x["docs"] for x in entries)
    if out_tokens != river.total - head:
        raise AssertionError(f"token parity failed: {out_tokens} != {river.total - head}")
    if out_docs != total_docs:
        raise AssertionError(f"doc-count parity failed: {out_docs} != {total_docs}")

    manifest = {
        "group": group, "split": split, "bos": args.bos, "cap": args.cap,
        "coprime_base": args.coprime, "shard_count": S,
        "coprime_ok": math.gcd(S, args.coprime) == 1,
        "input_files": [os.path.basename(p) for p in paths],
        "input_tokens": river.total, "dropped_head_tokens": head,
        "tokens": out_tokens, "docs": out_docs, "dtype": str(river.dtype),
        "shards": entries,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, manifest_path)
    log(f"{group}/{split}: DONE in {time.time() - t0:,.0f}s -- "
        f"{S} shards, {out_tokens:,} tokens, {out_docs:,} docs, parity OK")


def verify_group(out_root, group, split, args):
    out_dir = os.path.join(out_root, group)
    manifest_path = os.path.join(out_dir, f"manifest_{split}.json")
    if not os.path.exists(manifest_path):
        log(f"{group}/{split}: no manifest, skipping")
        return True
    with open(manifest_path) as f:
        man = json.load(f)
    ok = True
    for ent in man["shards"]:
        path = os.path.join(out_dir, ent["file"])
        try:
            a = np.load(path, mmap_mode="r")
        except Exception as e:
            log(f"  FAIL {ent['file']}: unreadable ({e})"); ok = False; continue
        if a.shape[0] != ent["tokens"]:
            log(f"  FAIL {ent['file']}: length {a.shape[0]} != {ent['tokens']}"); ok = False
        if int(a[0]) != man["bos"]:
            log(f"  FAIL {ent['file']}: first token {int(a[0])} != BOS"); ok = False
        docs = 0
        h = hashlib.blake2b(digest_size=16) if args.deep else None
        for s in range(0, a.shape[0], CHUNK):
            block = np.asarray(a[s : s + CHUNK])
            docs += int((block == man["bos"]).sum())
            if h is not None:
                h.update(block.tobytes())
        if docs != ent["docs"]:
            log(f"  FAIL {ent['file']}: docs {docs} != {ent['docs']}"); ok = False
        if h is not None and h.hexdigest() != ent["blake2b16"]:
            log(f"  FAIL {ent['file']}: checksum mismatch"); ok = False
    log(f"{group}/{split}: verify {'OK' if ok else 'FAILED'} "
        f"({man['shard_count']} shards, {man['tokens']:,} tokens, "
        f"coprime_ok={man['coprime_ok']})")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="input data root (dir containing group subdirs)")
    ap.add_argument("--out", required=True, help="output data root (created; sibling of --root)")
    ap.add_argument("--groups", nargs="+", required=True, help="group subdir names to process")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--bos", type=int, default=None,
                    help="document-separator token id baked into the shards by pre_tokenize "
                         "(= that tree's tokenizer.bos_id: llama=1; verify by id-histogram "
                         "for other trees). Required unless --verify-only.")
    ap.add_argument("--cap", type=int, default=100_000_000, help="target max tokens per shard")
    ap.add_argument("--coprime", type=int, default=6,
                    help="force gcd(shard_count, this)=1; 6 covers world sizes 2^a*3^b")
    ap.add_argument("--min-shard", type=int, default=5_000_000,
                    help="merge a final shard smaller than this into its neighbor")
    ap.add_argument("--force", action="store_true", help="rebuild groups that have a manifest")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify an existing output tree against its manifests")
    ap.add_argument("--deep", action="store_true", help="verify-only: also recompute checksums")
    ap.add_argument("--no-readback", action="store_true",
                    help="skip byte-level read-back verification after each shard write")
    args = ap.parse_args()

    if args.verify_only:
        all_ok = all(verify_group(args.out, g, sp, args)
                     for g in args.groups for sp in args.splits)
        raise SystemExit(0 if all_ok else 1)

    if not args.root:
        ap.error("--root is required (unless --verify-only)")
    if args.bos is None:
        ap.error("--bos is required (unless --verify-only): pass the tree's "
                 "tokenizer.bos_id explicitly — llama=1. No default, by design.")
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    if out == root or out.startswith(root + os.sep):
        ap.error("--out must not be the input root or inside it")

    failures = []
    for g in args.groups:
        for sp in args.splits:
            try:
                process_group(root, out, g, sp, args)
            except Exception as e:
                log(f"ERROR {g}/{sp}: {e}")
                failures.append(f"{g}/{sp}")
    if failures:
        log(f"FAILED: {', '.join(failures)} (successful groups are unaffected)")
        raise SystemExit(1)
    log("all groups done")


if __name__ == "__main__":
    main()
