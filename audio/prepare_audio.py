#!/usr/bin/env python
r"""
prepare_audio.py -- corpus-agnostic audio -> mu-law token bins/shards.

Turns an arbitrary speech corpus (VCTK, LibriTTS-R, LibriSpeech, Common Voice,
...) into next-sample-prediction training data, in either:

  * standalone  -> train.bin / val.bin (uint8 memmaps) + meta.json
                   drops straight into  mulaw_lm.py train --data <out>
  * keel        -> {group}_{split}_{NNNNNN}.npy shards (uint16 by default)
                   drops straight into the KEEL trainer as a dataset group

Handles nested speaker/chapter directory trees, any format torchaudio can read
(FLAC/MP3/OGG/... with a soundfile/ffmpeg backend; WAV works with no backend via
the stdlib fallback in mulaw_lm.py), optional per-speaker balancing, and a
speaker-disjoint validation split (hold out whole voices to measure whether the
model generalizes to speakers it never heard).

Dtype: mu-law tokens are natively 8-bit, so KEEL shards default to uint8 (half
the NAS storage + read bandwidth of uint16 per audio sample). The dataloader has
first-class uint8 support (see common_fsdp2/test_dataloader_dtype.py); --dtype
uint16 remains for compatibility with mixed text-vocab trees.

Shard count is made coprime with 6 so the loader's orbit stride
  shard_idx = (rank + pos*world_size) % num_shards
covers every shard for any world size of the form 2^a * 3^b (8 GPUs included).
Audio has no document boundaries, so shards may split anywhere.

Usage:
  # VCTK -> KEEL shards at 16 kHz, hold out 6 whole speakers for val
  python prepare_audio.py --root W:\nn_data\VCTK-Corpus-0.92 --out V:\...\vctk16k \
      --format keel --group vctk --sr 16000 --speaker-level 1 --val-speakers 6

  # LibriSpeech (spk/chapter/utt.flac): speaker is 2 dirs up
  python prepare_audio.py --root .../LibriSpeech/train-clean-360 --out .../libri16k \
      --format keel --group libri --sr 16000 --speaker-level 2 --val-speakers 20

  # Common Voice (flat clips/, one voice dominates): cap per speaker via regex
  python prepare_audio.py --root .../cv/en --out .../cv16k --format keel \
      --sr 16000 --per-speaker-max-clips 50 --val-frac 0.01

  # quick standalone bins for the mulaw_lm.py control (KEEP --sr 8000 to match it)
  python prepare_audio.py --root .../VCTK --out .../vctk8k --format standalone --sr 8000

  # HuggingFace dataset -> BOS-256 uint16 KEEL shards (dirty-paws). Audio decodes
  # via soundfile (pip install soundfile; datasets 4.x's own decoder wants torchcodec
  # which we avoid). vocab 257 (BOS=256) can't fit uint8, so --bos-token 256 forces uint16.
  python prepare_audio.py --hf-dataset openslr/librispeech_asr --hf-config clean \
      --hf-split train.360 --out .../libri_dp --group libri --sr 8000 \
      --bos-token 256 --speaker-column speaker_id --val-speakers 20
  # Common Voice (mp3): client_id is the speaker column (recovers speaker-disjoint val)
  python prepare_audio.py --hf-dataset mozilla-foundation/common_voice_17_0 --hf-config en \
      --out .../cv_dp --group commonvoice --sr 8000 --bos-token 256 \
      --speaker-column client_id --val-frac 0.01

  # verify an existing keel tree
  python prepare_audio.py --out .../vctk16k --group vctk --verify-only [--deep]

Deps: torch, torchaudio, numpy (file mode); + datasets, soundfile (HF mode).
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import wave as wave_mod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import torch

# Reuse the audio loaders from the sibling standalone script (backend-aware,
# with the stdlib WAV fallback for backend-less boxes).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mulaw_lm import _read_wav_stdlib, _load_audio  # noqa: E402

MU = 256                                   # mu-law levels == vocab_size
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm")


# ---------------------------------------------------------------------------
# discovery / speaker inference / splitting
# ---------------------------------------------------------------------------

def discover_files(root, exts, limit=None):
    files = []
    exts = tuple(e.lower() for e in exts)
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(dirpath, n))
    files.sort()
    if limit:
        files = files[:limit]
    return files


def speaker_of(path, root, level, regex):
    """Infer a speaker id from the path. `level` = directories up from the file
    (1 = immediate parent, 2 = grandparent). `regex` (capture group 1) overrides."""
    rel = os.path.relpath(path, root).replace("\\", "/")
    if regex is not None:
        m = re.search(regex, rel)
        return m.group(1) if m else "_unknown"
    parts = rel.split("/")
    idx = len(parts) - 1 - level
    return parts[idx] if idx >= 0 else "_root"


def apply_speaker_cap(files, speakers, max_clips, seed):
    """Keep at most `max_clips` files per speaker (deterministic sample)."""
    if not max_clips:
        return files
    by_spk = {}
    for f, s in zip(files, speakers):
        by_spk.setdefault(s, []).append(f)
    rng = np.random.default_rng(seed)
    kept = set()
    for s, fs in by_spk.items():
        if len(fs) <= max_clips:
            kept.update(fs)
        else:
            idx = rng.choice(len(fs), size=max_clips, replace=False)
            kept.update(fs[i] for i in idx)
    return [f for f in files if f in kept]


def split_train_val(files, speakers, val_speakers, val_frac, seed):
    """Return (is_val list[bool]). Speaker-disjoint when val_speakers>0."""
    rng = np.random.default_rng(seed)
    if val_speakers and val_speakers > 0:
        uniq = sorted(set(speakers))
        if val_speakers >= len(uniq):
            raise ValueError(f"--val-speakers {val_speakers} >= total speakers {len(uniq)}")
        perm = rng.permutation(len(uniq))
        val_set = {uniq[i] for i in perm[:val_speakers]}
        return [s in val_set for s in speakers]
    # fractional: hash each path deterministically to [0,1)
    out = []
    for f in files:
        h = int(hashlib.blake2b(str(f).encode("utf-8"), digest_size=8).hexdigest(), 16)
        out.append((h % 10_000_000) / 10_000_000.0 < val_frac)
    return out


# ---------------------------------------------------------------------------
# audio info (header-only, for planning the shard count)
# ---------------------------------------------------------------------------

def audio_info(path):
    """(num_frames, sample_rate) without decoding. Header read only."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        with wave_mod.open(path, "rb") as w:
            return w.getnframes(), w.getframerate()
    import torchaudio
    if torchaudio.list_audio_backends():
        info = torchaudio.info(path)
        return info.num_frames, info.sample_rate
    raise RuntimeError(f"no torchaudio backend to read header of {path}")


def _info_one(path):
    try:
        nf, sr = audio_info(path)
        return path, nf, sr, None
    except Exception as e:  # noqa: BLE001
        return path, 0, 0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# decode worker (CPU-heavy: load -> mono -> resample -> peak-norm -> mu-law)
# ---------------------------------------------------------------------------

def _encode_file(job):
    path, sr_target, peak_norm = job
    try:
        import torchaudio.functional as AF
        torch.set_num_threads(1)
        wav, sr = _load_audio(path)
        wav = wav.float()
        if wav.dim() == 2:
            wav = wav.mean(0)                       # -> mono (time,)
        if sr != sr_target:
            wav = AF.resample(wav, sr, sr_target)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * peak_norm
        tok = AF.mu_law_encoding(wav, MU).to(torch.uint8).numpy()
        return tok
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# coprime shard count
# ---------------------------------------------------------------------------

def coprime_ceil(n_min):
    """Smallest S >= max(1, n_min) with gcd(S, 6) == 1."""
    s = max(1, int(n_min))
    while math.gcd(s, 6) != 1:
        s += 1
    return s


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

class BinWriter:
    """Standalone uint8 stream -> a single <split>.bin (mulaw_lm.py format)."""
    def __init__(self, out_dir, split, gap_tokens):
        self.path = os.path.join(out_dir, f"{split}.bin")
        self.f = open(self.path, "wb")
        self.gap = gap_tokens
        self.total = 0

    def add(self, tokens):
        self.f.write(tokens.tobytes())
        self.total += len(tokens)
        if self.gap is not None and len(self.gap):
            self.f.write(self.gap.tobytes())
            self.total += len(self.gap)

    def close(self):
        self.f.close()
        return {"tokens": self.total}


class ShardWriter:
    """KEEL shard stream -> {group}_{split}_{NNNNNN}.npy files, coprime-with-6 count.

    est mode (total_est given — file path with a header pre-scan): flush by
    estimate-based thresholds so shards are ~even; close() fills exactly n_shards.
    fixed mode (total_est=None — HF path with no cheap pre-scan): flush at
    shard_tokens; close() splits the tail to reach a coprime-with-6 count."""
    def __init__(self, out_dir, group, split, dtype, n_shards, total_est, gap_tokens,
                 shard_tokens=100_000_000, min_shard=1_000_000):
        self.out_dir = out_dir
        self.group = group
        self.split = split
        self.dtype = dtype
        self.gap = gap_tokens
        self.shard_tokens = shard_tokens
        self.min_shard = min_shard
        self.fixed = total_est is None
        self.n_shards = n_shards
        self.total_est = max(1, total_est) if total_est is not None else None
        self.buf = []
        self.buf_len = 0
        self.cum = 0
        self.written = 0
        self.shard_sizes = []

    def _write(self, arr):
        name = f"{self.group}_{self.split}_{self.written:06d}.npy"
        final = os.path.join(self.out_dir, name)
        tmp = final + ".tmp"
        arr = arr.astype(self.dtype, copy=False)
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, final)
        self.written += 1
        self.shard_sizes.append(int(arr.shape[0]))

    def add(self, tokens):
        self.buf.append(tokens)
        self.buf_len += len(tokens)
        self.cum += len(tokens)
        if self.gap is not None and len(self.gap):
            self.buf.append(self.gap)
            self.buf_len += len(self.gap)
            self.cum += len(self.gap)
        if self.fixed:
            if self.buf_len >= self.shard_tokens:
                self._write(np.concatenate(self.buf))
                self.buf, self.buf_len = [], 0
        else:
            while (self.written < self.n_shards - 1 and
                   self.cum >= (self.written + 1) * self.total_est / self.n_shards):
                self._write(np.concatenate(self.buf))
                self.buf, self.buf_len = [], 0

    def close(self):
        if self.fixed:
            if self.buf:
                self._write(np.concatenate(self.buf))
                self.buf, self.buf_len = [], 0
            self._finalize_coprime()
        else:
            remaining = self.n_shards - self.written
            buf = np.concatenate(self.buf) if self.buf else np.empty(0, dtype=self.dtype)
            if remaining <= 1:
                self._write(buf)
            else:
                for p in np.array_split(buf, remaining):
                    self._write(p)
        return {"shards": self.written, "shard_sizes": self.shard_sizes,
                "tokens": int(sum(self.shard_sizes))}

    def _finalize_coprime(self):
        """Split trailing shards until the count is coprime with 6. Only the tail
        (highest indices) is rewritten, so earlier shards stay byte-stable."""
        S = self.written
        target = coprime_ceil(S)
        if target == S or S == 0:
            return
        need = target - S
        files = sorted(glob.glob(os.path.join(
            self.out_dir, f"{self.group}_{self.split}_*.npy")))
        pull = min(S, need + 1)
        tail = files[-pull:]
        combined = np.concatenate([np.load(f) for f in tail])
        for f in tail:
            os.remove(f)
        self.written = S - pull
        self.shard_sizes = self.shard_sizes[:self.written]
        for p in np.array_split(combined, pull + need):
            self._write(p)


# ---------------------------------------------------------------------------
# main prepare
# ---------------------------------------------------------------------------

def make_gap(sr, gap_seconds):
    if gap_seconds <= 0:
        return None
    import torchaudio.functional as AF
    n = int(gap_seconds * sr)
    return AF.mu_law_encoding(torch.zeros(n), MU).to(torch.uint8).numpy()


def token_dtype_of(args):
    """uint16 whenever a >255 BOS is emitted (doesn't fit in uint8), else the
    requested keel dtype (standalone bins are always uint8)."""
    if args.format == "keel" and args.dtype == "uint16":
        return np.uint16
    if args.bos_token is not None and args.bos_token > 255:
        return np.uint16
    return np.uint8


def apply_bos(tok, bos, dtype):
    """Prepend the clip-boundary BOS token so doc-mask/pos-reset see a boundary."""
    if bos is None:
        return tok.astype(dtype, copy=False)
    return np.concatenate((np.array([bos], dtype=dtype), tok.astype(dtype, copy=False)))


def iter_decode(files, sr, peak_norm, workers):
    """Yield (path, tokens_or_None, err) in input order."""
    jobs = [(f, sr, peak_norm) for f in files]
    if workers <= 1:
        for f in files:
            r = _encode_file((f, sr, peak_norm))
            yield (f, None, r[4:]) if isinstance(r, str) else (f, r, None)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f, r in zip(files, ex.map(_encode_file, jobs, chunksize=8)):
            yield (f, None, r[4:]) if isinstance(r, str) else (f, r, None)


def prepare(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    print(f"[prep] scanning {args.root}")
    files = discover_files(args.root, args.exts, args.limit)
    if not files:
        raise SystemExit(f"no audio files ({args.exts}) under {args.root}")
    speakers = [speaker_of(f, args.root, args.speaker_level, args.speaker_regex) for f in files]
    n_spk0 = len(set(speakers))
    print(f"[prep] {len(files):,} files, {n_spk0:,} speakers (level={args.speaker_level}"
          f"{', regex' if args.speaker_regex else ''})")

    files = apply_speaker_cap(files, speakers, args.per_speaker_max_clips, args.seed)
    speakers = [speaker_of(f, args.root, args.speaker_level, args.speaker_regex) for f in files]
    if args.per_speaker_max_clips:
        print(f"[prep] after per-speaker cap {args.per_speaker_max_clips}: {len(files):,} files")

    is_val = split_train_val(files, speakers, args.val_speakers, args.val_frac, args.seed)
    train_files = [f for f, v in zip(files, is_val) if not v]
    val_files = [f for f, v in zip(files, is_val) if v]
    tr_spk = {s for s, v in zip(speakers, is_val) if not v}
    va_spk = {s for s, v in zip(speakers, is_val) if v}
    print(f"[prep] split: train {len(train_files):,} files / {len(tr_spk):,} spk, "
          f"val {len(val_files):,} files / {len(va_spk):,} spk"
          + (" (speaker-disjoint)" if args.val_speakers else f" (frac {args.val_frac})"))
    if args.val_speakers and (tr_spk & va_spk):
        raise SystemExit("BUG: val/train speaker overlap")
    if not train_files or not val_files:
        raise SystemExit("empty train or val split -- adjust --val-speakers/--val-frac")

    gap = make_gap(args.sr, args.gap_seconds)
    token_dtype = token_dtype_of(args)
    splits = {"train": train_files, "val": val_files}
    report = {"root": args.root, "format": args.format, "sr": args.sr,
              "vocab": MU if args.bos_token is None else max(MU, args.bos_token + 1),
              "bos_token": args.bos_token,
              "dtype": str(np.dtype(token_dtype)),
              "gap_seconds": args.gap_seconds, "speaker_level": args.speaker_level,
              "splits": {}}

    for split, split_files in splits.items():
        # plan shard count from header-estimated resampled token totals (keel only)
        est = 0
        gap_len = 0 if gap is None else len(gap)
        if args.format == "keel":
            print(f"[prep] [{split}] header pre-scan ({len(split_files):,} files)...")
            with ThreadPoolExecutor(max_workers=max(4, args.workers)) as ex:
                for path, nf, sr, err in ex.map(_info_one, split_files):
                    if err is None and sr > 0:
                        est += int(nf * args.sr / sr) + gap_len
            n_shards = coprime_ceil(math.ceil(est / args.shard_tokens))
            approx = est / max(1, n_shards)
            if approx < args.min_shard_tokens:
                n_shards = coprime_ceil(max(1, est // args.min_shard_tokens))
            print(f"[prep] [{split}] est {est/1e6:.1f}M tokens ({est/args.sr/3600:.2f} h) "
                  f"-> {n_shards} shards (~{est/max(1,n_shards)/1e6:.1f}M each, coprime6)")
            writer = ShardWriter(args.out, args.group, split, args.dtype,
                                 n_shards, est, gap)
        else:
            writer = BinWriter(args.out, split, gap)

        # decode pass
        n_ok = n_bad = 0
        actual = 0
        bad_examples = []
        tlast = time.time()
        for i, (path, tok, err) in enumerate(iter_decode(split_files, args.sr,
                                                          args.peak_norm, args.workers)):
            if tok is None:
                n_bad += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"{os.path.basename(path)}: {err}")
                continue
            tok = apply_bos(tok, args.bos_token, token_dtype)
            writer.add(tok)
            actual += len(tok)
            n_ok += 1
            if (i + 1) % 2000 == 0 or time.time() - tlast > 30:
                tlast = time.time()
                print(f"[prep] [{split}] {i+1:,}/{len(split_files):,} files, "
                      f"{actual/1e6:.1f}M tokens, {n_bad} skipped, {time.time()-t0:.0f}s")
        info = writer.close()
        info.update(dict(files_ok=n_ok, files_skipped=n_bad,
                         hours=round(actual / args.sr / 3600, 3),
                         bad_examples=bad_examples))
        report["splits"][split] = info
        print(f"[prep] [{split}] DONE: {n_ok:,} files, {actual:,} tokens "
              f"({info['hours']:.2f} h)"
              + (f", {info['shards']} shards" if args.format == "keel" else "")
              + (f", {n_bad} skipped" if n_bad else ""))
        if n_bad:
            for b in bad_examples:
                print(f"           skip: {b}")

    if args.format == "standalone":
        meta = {"sample_rate": args.sr, "vocab_size": MU, "encoding": "mu-law-256"}
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if args.sr != 8000:
            print(f"[prep] NOTE: standalone bins at {args.sr} Hz -- mulaw_lm.py assumes "
                  f"8000 Hz (SAMPLE_RATE). Match its constant or generated wavs mis-time.")

    with open(os.path.join(args.out, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[prep] all done in {time.time()-t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(args):
    ok = True
    hi = MU if getattr(args, "bos_token", None) is None else max(MU, args.bos_token + 1)
    for split in ("train", "val"):
        shards = sorted(glob.glob(os.path.join(args.out, f"{args.group}_{split}_*.npy")))
        if not shards:
            print(f"[verify] {split}: no shards"); ok = False; continue
        S = len(shards)
        cop = math.gcd(S, 6) == 1
        total = 0
        bad = 0
        for p in shards:
            try:
                a = np.load(p, mmap_mode="r")
                if a.dtype not in (np.uint16, np.uint32, np.uint8):
                    print(f"[verify]   {os.path.basename(p)}: dtype {a.dtype}"); bad += 1
                if a.min() < 0 or a.max() >= hi:
                    print(f"[verify]   {os.path.basename(p)}: values out of [0,{hi})"); bad += 1
                total += a.shape[0]
                if args.deep:
                    _ = np.asarray(a).sum()      # force full read
            except Exception as e:  # noqa: BLE001
                print(f"[verify]   {os.path.basename(p)}: {type(e).__name__}: {e}"); bad += 1
        status = "OK" if (cop and bad == 0) else "PROBLEM"
        print(f"[verify] {split}: {S} shards, coprime6={cop}, {total:,} tokens, "
              f"{bad} bad -> {status}")
        ok = ok and cop and bad == 0
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# HuggingFace datasets ingestion (arrow/parquet -> mu-law shards)
# ---------------------------------------------------------------------------

def detect_speaker_column(ds, override):
    if override:
        return override if override in ds.column_names else None
    for c in ("speaker_id", "client_id", "speaker", "speakerid", "speaker_name", "spk_id"):
        if c in ds.column_names:
            return c
    return None


def load_hf(args):
    """Load a HF dataset split; cast the audio column to decode-on-access (native
    SR — we resample ourselves to avoid a librosa dependency)."""
    from datasets import load_dataset, load_from_disk, Audio, Dataset
    if os.path.isdir(args.hf_dataset):
        d = load_from_disk(args.hf_dataset)
        ds = d if isinstance(d, Dataset) else d[args.hf_split]
    else:
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    audio_col = args.hf_audio_column
    if audio_col not in ds.column_names:
        cand = [c for c in ds.column_names if c.lower() in ("audio", "wav", "speech")]
        if not cand:
            raise SystemExit(f"no audio column '{audio_col}' in {ds.column_names}")
        audio_col = cand[0]
    # decode=False -> raw bytes/path; we decode with soundfile ourselves (datasets
    # 4.x wants torchcodec for its own decoder; soundfile handles flac/mp3/wav/ogg).
    ds = ds.cast_column(audio_col, Audio(decode=False))
    spk_col = detect_speaker_column(ds, args.speaker_column)
    return ds, audio_col, spk_col


def decode_hf_example(ex, audio_col, sr_target, peak_norm):
    import io
    import soundfile as sf
    import torchaudio.functional as AF
    a = ex[audio_col]
    if a.get("bytes") is not None:
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
    else:
        wav, sr = sf.read(a["path"], dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:                       # (frames, channels) -> mono
        wav = wav.mean(axis=1)
    wt = torch.from_numpy(np.ascontiguousarray(wav))
    if sr != sr_target:
        wt = AF.resample(wt, sr, sr_target)
    peak = wt.abs().max()
    if peak > 0:
        wt = wt / peak * peak_norm
    return AF.mu_law_encoding(wt, MU).to(torch.uint8).numpy()


def prepare_hf(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ds, audio_col, spk_col = load_hf(args)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"[prep-hf] {args.hf_dataset} [{args.hf_split}] {n:,} examples | "
          f"audio='{audio_col}' speaker='{spk_col or 'NONE'}'")
    speakers = [str(x) for x in ds[spk_col]] if spk_col else [str(i) for i in range(n)]
    n_spk0 = len(set(speakers))

    idxs = list(range(n))
    idxs = apply_speaker_cap(idxs, speakers, args.per_speaker_max_clips, args.seed)
    speakers = [speakers[i] for i in idxs]
    if args.per_speaker_max_clips:
        print(f"[prep-hf] after per-speaker cap {args.per_speaker_max_clips}: {len(idxs):,} clips")

    if args.val_speakers and not spk_col:
        print("[prep-hf] WARNING: --val-speakers requested but no speaker column found; "
              "falling back to --val-frac (val is NOT speaker-disjoint).")
        vs = 0
    else:
        vs = args.val_speakers
    is_val = split_train_val(idxs, speakers, vs, args.val_frac, args.seed)
    tr = [i for i, v in zip(idxs, is_val) if not v]
    va = [i for i, v in zip(idxs, is_val) if v]
    tr_spk = {s for s, v in zip(speakers, is_val) if not v}
    va_spk = {s for s, v in zip(speakers, is_val) if v}
    print(f"[prep-hf] {n_spk0:,} speakers | split: train {len(tr):,} / {len(tr_spk):,} spk, "
          f"val {len(va):,} / {len(va_spk):,} spk"
          + (" (speaker-disjoint)" if vs else f" (frac {args.val_frac})"))
    if vs and (tr_spk & va_spk):
        raise SystemExit("BUG: val/train speaker overlap")
    if not tr or not va:
        raise SystemExit("empty train or val split -- adjust --val-speakers/--val-frac")

    gap = make_gap(args.sr, args.gap_seconds)
    token_dtype = token_dtype_of(args)
    report = {"hf_dataset": args.hf_dataset, "hf_config": args.hf_config,
              "hf_split": args.hf_split, "sr": args.sr,
              "vocab": MU if args.bos_token is None else max(MU, args.bos_token + 1),
              "bos_token": args.bos_token, "dtype": str(np.dtype(token_dtype)),
              "gap_seconds": args.gap_seconds, "splits": {}}

    for split, split_idx in (("train", tr), ("val", va)):
        if args.format == "keel":
            writer = ShardWriter(args.out, args.group, split, args.dtype,
                                 n_shards=None, total_est=None, gap_tokens=gap,
                                 shard_tokens=args.shard_tokens,
                                 min_shard=args.min_shard_tokens)
        else:
            writer = BinWriter(args.out, split, gap)
        n_ok = n_bad = actual = 0
        bad_examples = []
        tlast = time.time()
        for j, idx in enumerate(split_idx):
            try:
                tok = decode_hf_example(ds[int(idx)], audio_col, args.sr, args.peak_norm)
            except Exception as e:  # noqa: BLE001
                n_bad += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"idx {idx}: {type(e).__name__}: {e}")
                continue
            tok = apply_bos(tok, args.bos_token, token_dtype)
            writer.add(tok)
            actual += len(tok)
            n_ok += 1
            if (j + 1) % 2000 == 0 or time.time() - tlast > 30:
                tlast = time.time()
                print(f"[prep-hf] [{split}] {j+1:,}/{len(split_idx):,}, "
                      f"{actual/1e6:.1f}M tokens, {n_bad} skipped, {time.time()-t0:.0f}s")
        info = writer.close()
        info.update(dict(clips_ok=n_ok, clips_skipped=n_bad,
                         hours=round(actual / args.sr / 3600, 3), bad_examples=bad_examples))
        report["splits"][split] = info
        print(f"[prep-hf] [{split}] DONE: {n_ok:,} clips, {actual:,} tokens "
              f"({info['hours']:.2f} h)"
              + (f", {info['shards']} shards" if args.format == "keel" else "")
              + (f", {n_bad} skipped" if n_bad else ""))
        for b in bad_examples:
            print(f"           skip: {b}")

    with open(os.path.join(args.out, "prepare_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[prep-hf] all done in {time.time()-t0:.0f}s -> {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="corpus root (recursively scanned) — file mode")
    # HuggingFace datasets mode (mutually exclusive with --root)
    p.add_argument("--hf-dataset", default=None,
                   help="HF dataset id (e.g. openslr/librispeech_asr) or a load_from_disk path")
    p.add_argument("--hf-config", default=None, help="HF dataset config (e.g. 'clean', 'en')")
    p.add_argument("--hf-split", default="train", help="HF split to ingest (default train)")
    p.add_argument("--hf-audio-column", default="audio")
    p.add_argument("--speaker-column", default=None,
                   help="HF column for speaker id (auto: speaker_id/client_id/...)")
    p.add_argument("--bos-token", type=int, default=None,
                   help="emit this token at every clip boundary (dirty-paws: 256). "
                        ">255 forces uint16 shards.")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--format", choices=["keel", "standalone"], default="keel")
    p.add_argument("--group", default="audio", help="group name for keel shard filenames")
    p.add_argument("--sr", type=int, default=16000, help="target sample rate")
    p.add_argument("--dtype", choices=["uint8", "uint16"], default="uint8",
                   help="keel shard dtype (uint8 = native mu-law, half disk+bandwidth; "
                        "uint16 for mixed text-vocab trees)")
    p.add_argument("--exts", nargs="+", default=list(AUDIO_EXTS))
    p.add_argument("--speaker-level", type=int, default=1,
                   help="dirs up from file for speaker id (1=parent, 2=grandparent)")
    p.add_argument("--speaker-regex", default=None,
                   help="regex on rel path, capture group 1 = speaker (overrides level)")
    p.add_argument("--per-speaker-max-clips", type=int, default=0,
                   help="cap files per speaker (0 = no cap)")
    p.add_argument("--val-speakers", type=int, default=0,
                   help="hold out this many WHOLE speakers for val (speaker-disjoint)")
    p.add_argument("--val-frac", type=float, default=0.01,
                   help="val fraction of clips (used when --val-speakers 0)")
    p.add_argument("--gap-seconds", type=float, default=0.1,
                   help="silence stitched between clips (0 to disable)")
    p.add_argument("--peak-norm", type=float, default=0.95)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--min-shard-tokens", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--limit", type=int, default=0, help="debug: only first N files")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--deep", action="store_true", help="verify: force full read of every shard")
    args = p.parse_args()
    args.limit = args.limit or None

    # BOS > 255 can't fit in uint8: force uint16 (keel) or reject (standalone bins)
    if args.bos_token is not None and args.bos_token > 255:
        if args.format == "standalone":
            p.error("--bos-token >255 needs uint16, but standalone bins are uint8. "
                    "Use --format keel.")
        if args.dtype == "uint8":
            args.dtype = "uint16"
            print(f"[prep] NOTE: --bos-token {args.bos_token} >255 -> forcing --dtype uint16.")

    if args.verify_only:
        sys.exit(verify(args))
    if args.hf_dataset:
        prepare_hf(args)
    elif args.root:
        prepare(args)
    else:
        p.error("provide --root (file mode), --hf-dataset (HF mode), or --verify-only")


if __name__ == "__main__":
    main()
