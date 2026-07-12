"""
mulaw_lm.py -- next-token prediction on raw audio, the WaveNet-classic way.

The whole idea in three lines:
  1. Audio -> resample to 8 kHz mono -> mu-law quantize -> uint8 token stream (vocab=256)
  2. Train a plain causal transformer with cross-entropy on next-sample prediction
  3. Sample autoregressively, mu-law expand, write a .wav, press play

Usage:
  python mulaw_lm.py prepare --lj /path/to/LJSpeech-1.1 --out data/lj8k
  python mulaw_lm.py train   --data data/lj8k --out runs/mulaw25m
  python mulaw_lm.py sample  --ckpt runs/mulaw25m/ckpt.pt --data data/lj8k \
                             --seconds 0.8 --primer-seconds 0.2 --temperature 0.95

Numbers to hold in your head:
  - Random baseline: 8.000 bits/sample (uniform over 256).  loss = ln(256) = 5.545 nats
  - A trained ~25M model on single-speaker 8 kHz speech: ~2.0-2.5 bits/sample
  - LJ Speech @ 8 kHz is ~690M tokens. One "epoch" at 128k tok/step is ~5,400 steps.
  - Temperature < 1.0 biases toward silence (quiet, muffled). > ~1.1 turns to static.
    The sweet spot for babble is usually 0.9-1.0.

Deps: torch, torchaudio, numpy. Everything else is stdlib.
"""

import argparse
import json
import math
import os
import struct
import time
import wave as wave_mod
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 256        # mu-law levels
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    block_size: int = 8192       # 1.024 s of context at 8 kHz
    dropout: float = 0.0
    ffn_mult: float = 8 / 3      # SwiGLU sizing

SAMPLE_RATE = 8000               # start here; 16k is the obvious second experiment
GAP_SECONDS = 0.1                # silence stitched between clips in the token stream


# ----------------------------------------------------------------------------
# Model: a small, boring, correct GPT with RoPE. Boring is the point --
# the only novel thing in this experiment should be the data.
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # persistent=True. dn1-dn4 taught us this the hard way.
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        t = torch.arange(max_pos).float()
        freqs = torch.outer(t, inv_freq)                      # (max_pos, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q, k, pos_offset: int):
        # q, k: (B, H, T, hd)
        T = q.shape[2]
        cos = self.cos[pos_offset:pos_offset + T].to(q.dtype)  # (T, hd/2)
        sin = self.sin[pos_offset:pos_offset + T].to(q.dtype)

        def rot(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            xr1 = x1 * cos - x2 * sin
            xr2 = x1 * sin + x2 * cos
            out = torch.stack((xr1, xr2), dim=-1)
            return out.flatten(-2)

        return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim, cfg.block_size)
        self.dropout = cfg.dropout

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        pos_offset = 0 if past_kv is None else past_kv[0].shape[2]
        q, k = self.rotary(q, k, pos_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Training / prefill: square causal. Incremental decode (T==1): attend to all past.
        is_causal = past_kv is None and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        hidden = (hidden + 63) // 64 * 64  # round up to multiple of 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class AudioGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init)
        # scale residual-out projections by 1/sqrt(2L)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        x = self.tok_emb(idx)
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                new_kvs.append(kv)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, new_kvs

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# prepare: LJSpeech wavs -> one big uint8 token stream (train.bin / val.bin)
# ----------------------------------------------------------------------------

def _read_wav_stdlib(path):
    """Backend-free WAV reader (stdlib `wave` + numpy). Returns (channels, time)
    float32 in [-1, 1] to match torchaudio.load's contract. Handles 8/16/32-bit PCM."""
    with wave_mod.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sw*8}-bit: {path}")
    a = a.reshape(-1, nch).T                       # (channels, time)
    return torch.from_numpy(np.ascontiguousarray(a)), sr


def _load_audio(path):
    """Prefer torchaudio (any format) when a backend is present; otherwise fall
    back to the stdlib WAV reader. torchaudio's I/O needs a soundfile/ffmpeg
    backend that isn't installed on every box -- but resample + mu-law don't."""
    import torchaudio
    if torchaudio.list_audio_backends():
        return torchaudio.load(str(path))
    return _read_wav_stdlib(path)


def cmd_prepare(args):
    import torchaudio.functional as AF

    lj = Path(args.lj)
    wav_dir = lj / "wavs"
    assert wav_dir.is_dir(), f"expected {wav_dir} to exist (LJSpeech-1.1 layout)"
    files = sorted(wav_dir.glob("*.wav"))
    assert files, "no wavs found"
    print(f"found {len(files)} clips")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_val = args.val_clips
    splits = {"train": files[:-n_val], "val": files[-n_val:]}

    gap = torch.zeros(int(GAP_SECONDS * SAMPLE_RATE))
    gap_tokens = AF.mu_law_encoding(gap, 256).to(torch.uint8).numpy()

    for split, split_files in splits.items():
        path = out / f"{split}.bin"
        total = 0
        t0 = time.time()
        with open(path, "wb") as f:
            for i, fp in enumerate(split_files):
                wav, sr = _load_audio(fp)
                wav = wav.mean(0)                                  # mono
                if sr != SAMPLE_RATE:
                    wav = AF.resample(wav, sr, SAMPLE_RATE)
                peak = wav.abs().max()
                if peak > 0:
                    wav = wav / peak * 0.95                        # normalize per clip
                tokens = AF.mu_law_encoding(wav, 256).to(torch.uint8).numpy()
                f.write(tokens.tobytes())
                f.write(gap_tokens.tobytes())
                total += len(tokens) + len(gap_tokens)
                if (i + 1) % 1000 == 0:
                    print(f"  [{split}] {i+1}/{len(split_files)} clips, "
                          f"{total/1e6:.1f}M tokens, {time.time()-t0:.0f}s")
        hours = total / SAMPLE_RATE / 3600
        print(f"{split}: {total:,} tokens ({hours:.2f} h) -> {path}")

    meta = {"sample_rate": SAMPLE_RATE, "vocab_size": 256, "encoding": "mu-law-256"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done. tokens are literal audio samples. let that sink in.")


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------

def get_batch(data: np.memmap, block: int, bs: int, device):
    ix = np.random.randint(0, len(data) - block - 1, size=bs)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(step, base_lr, warmup, max_steps, min_frac=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def write_wav(path: Path, samples: torch.Tensor, sr: int):
    """samples: float tensor in [-1, 1], 1-D. Stdlib only -- no backend surprises."""
    pcm = (samples.clamp(-1, 1) * 32767.0).to(torch.int16).numpy()
    with wave_mod.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@torch.no_grad()
def generate(model, primer: torch.Tensor, n_new: int, temperature=1.0, top_k=None):
    """primer: (T,) int64 on device. Returns (T + n_new,) int64 on cpu."""
    model.eval()
    device = primer.device
    total = primer.numel() + n_new
    assert total <= model.cfg.block_size, (
        f"primer+generation = {total} exceeds block_size {model.cfg.block_size}; "
        f"train a bigger block or shorten the ask")

    # Prefill: process the primer once, build the KV cache.
    idx = primer.unsqueeze(0)
    logits, _, kvs = model(idx, use_cache=True)
    out = [primer.cpu()]

    tok = None
    t0 = time.time()
    for i in range(n_new):
        logits_last = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits_last.float(), dim=-1)
        tok = torch.multinomial(probs, num_samples=1)          # (1, 1)
        out.append(tok.view(1).cpu())
        logits, _, kvs = model(tok, past_kvs=kvs, use_cache=True)
    dt = time.time() - t0
    print(f"  generated {n_new} samples in {dt:.1f}s ({n_new/dt:.0f} tok/s, "
          f"{n_new/SAMPLE_RATE:.2f}s of audio)")
    model.train()
    return torch.cat(out)


def sample_to_wav(model, val_data, out_path: Path, seconds: float,
                  primer_seconds: float, temperature: float, top_k, device):
    import torchaudio.functional as AF
    n_prime = int(primer_seconds * SAMPLE_RATE)
    n_new = int(seconds * SAMPLE_RATE)
    i = np.random.randint(0, len(val_data) - n_prime - 1)
    primer = torch.from_numpy(val_data[i:i + n_prime].astype(np.int64)).to(device)
    tokens = generate(model, primer, n_new, temperature=temperature, top_k=top_k)
    audio = AF.mu_law_decoding(tokens, 256)
    write_wav(out_path, audio, SAMPLE_RATE)
    print(f"  wrote {out_path}  (first {primer_seconds}s is real; the rest is the model)")


def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint8, mode="r")
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    cfg = ModelConfig(d_model=args.d_model, n_layer=args.n_layer,
                      n_head=args.n_head, block_size=args.block_size)
    model = AudioGPT(cfg).to(device)
    print(f"model: {model.num_params()/1e6:.1f}M params, "
          f"context = {cfg.block_size} samples = {cfg.block_size/SAMPLE_RATE:.2f}s")

    if args.compile:
        model = torch.compile(model)

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        print(f"resumed from step {start_step}")

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=(device.type == "cuda"))
    LN2 = math.log(2)
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args.lr, args.warmup, args.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with autocast:
                _, loss, _ = model(x, targets=y)
            (loss / args.grad_accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % args.log_interval == 0:
            toks = args.batch_size * args.grad_accum * cfg.block_size
            dt = time.time() - t0
            t0 = time.time()
            tps = toks * args.log_interval / dt if step > start_step else 0
            print(f"step {step:6d} | loss {loss.item():.4f} | "
                  f"{loss.item()/LN2:.3f} bits/sample | lr {lr:.2e} | "
                  f"gnorm {gnorm:.2f} | {tps/1e3:.0f}k tok/s")

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(args.eval_iters):
                    x, y = get_batch(val_data, cfg.block_size, args.batch_size, device)
                    with autocast:
                        _, vloss, _ = model(x, targets=y)
                    losses.append(vloss.item())
            model.train()
            vl = sum(losses) / len(losses)
            print(f"  == val loss {vl:.4f} | {vl/LN2:.3f} bits/sample ==")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "config": asdict(cfg)}, out / "ckpt.pt")

        if step > 0 and step % args.sample_interval == 0:
            # The ritual. Listen to every one of these.
            sample_to_wav(model, val_data,
                          out / "samples" / f"step{step:06d}.wav",
                          seconds=args.sample_seconds, primer_seconds=0.2,
                          temperature=0.95, top_k=None, device=device)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": args.max_steps - 1, "config": asdict(cfg)}, out / "ckpt.pt")
    print("training complete. go listen.")


# ----------------------------------------------------------------------------
# sample
# ----------------------------------------------------------------------------

def cmd_sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = AudioGPT(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    print(f"loaded step {ck['step']} ({model.num_params()/1e6:.1f}M params)")

    val_data = np.memmap(Path(args.data) / "val.bin", dtype=np.uint8, mode="r")
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        path = out / f"sample_{ck['step']}_{i}_t{args.temperature}.wav"
        sample_to_wav(model, val_data, path, seconds=args.seconds,
                      primer_seconds=args.primer_seconds,
                      temperature=args.temperature, top_k=args.top_k, device=device)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="LJSpeech -> mu-law token bins")
    pp.add_argument("--lj", required=True, help="path to LJSpeech-1.1 root")
    pp.add_argument("--out", required=True, help="output dir for train.bin/val.bin")
    pp.add_argument("--val-clips", type=int, default=500)

    pt = sub.add_parser("train", help="train the audio LM")
    pt.add_argument("--data", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--d-model", type=int, default=512)
    pt.add_argument("--n-layer", type=int, default=8)
    pt.add_argument("--n-head", type=int, default=8)
    pt.add_argument("--block-size", type=int, default=8192)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--grad-accum", type=int, default=1)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--warmup", type=int, default=500)
    pt.add_argument("--max-steps", type=int, default=30000)
    pt.add_argument("--weight-decay", type=float, default=0.1)
    pt.add_argument("--clip", type=float, default=1.0)
    pt.add_argument("--log-interval", type=int, default=20)
    pt.add_argument("--eval-interval", type=int, default=500)
    pt.add_argument("--eval-iters", type=int, default=50)
    pt.add_argument("--sample-interval", type=int, default=1000)
    pt.add_argument("--sample-seconds", type=float, default=0.8)
    pt.add_argument("--compile", action="store_true")
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--seed", type=int, default=1337)

    ps = sub.add_parser("sample", help="generate audio from a checkpoint")
    ps.add_argument("--ckpt", required=True)
    ps.add_argument("--data", required=True, help="dir containing val.bin (for primers)")
    ps.add_argument("--out", default="samples")
    ps.add_argument("--n", type=int, default=4)
    ps.add_argument("--seconds", type=float, default=0.8)
    ps.add_argument("--primer-seconds", type=float, default=0.2)
    ps.add_argument("--temperature", type=float, default=0.95)
    ps.add_argument("--top-k", type=int, default=None)

    args = p.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    main()
