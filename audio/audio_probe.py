#!/usr/bin/env python
r"""
audio_probe.py -- out-of-band diagnostics for a mu-law audio LM checkpoint.

Runs WITHOUT touching a live training run (defaults to CPU so it can't contend
for the training GPU's VRAM). Implements Rook's "act now" cluster:

  1. Temperature triplet: generate from the SAME primer at temps 0.90/0.95/1.00,
     write wav + spectrogram for each, report generated RMS / silence / hf-rms.
  2. Real-val reference: the same stats on RAW val audio -- the target line the
     generated stats should converge toward (not zero).
  3. Silence vs speech bits/sample: val CE split by |code-128| threshold. Once
     silence saturates, the speech-token bits are the honest progress signal.
  4. Spectrogram PNGs (torch.stft + PIL) next to every wav for fast eyeballing.

Usage:
  python audio_probe.py --ckpt runs/mulaw25m/ckpt.pt --data data/lj8k \
      --out runs/mulaw25m/probe --temps 0.9 0.95 1.0 --n-primers 2
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mulaw_lm import AudioGPT, ModelConfig, generate, write_wav, get_batch  # noqa: E402

LN2 = math.log(2)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = AudioGPT(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg, int(ck.get("step", -1))


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def audio_stats(x):
    """x: float np array in [-1,1]. Returns rms, peak, silence-frac, hf, hf/rms."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    d = np.diff(x)
    hf = float(np.sqrt((d ** 2).mean())) if d.size else 0.0
    return dict(rms=rms, peak=float(np.abs(x).max()) if x.size else 0.0,
                silence=float(np.mean(np.abs(x) < 0.01)) if x.size else 0.0,
                hf=hf, hf_rms=hf / rms if rms > 1e-9 else float("nan"))


def mu_decode(codes):
    import torchaudio.functional as AF
    return AF.mu_law_decoding(torch.as_tensor(np.asarray(codes), dtype=torch.int64), 256).numpy()


# --------------------------------------------------------------------------- #
# spectrogram PNG (torch.stft + PIL, no matplotlib)
# --------------------------------------------------------------------------- #

def _magma_lut():
    anchors = [(0.0, (0, 0, 4)), (0.25, (84, 15, 109)), (0.5, (187, 55, 84)),
               (0.75, (249, 142, 8)), (1.0, (252, 253, 191))]
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(anchors) - 1):
            p0, c0 = anchors[k]; p1, c1 = anchors[k + 1]
            if p0 <= t <= p1:
                f = (t - p0) / (p1 - p0)
                lut[i] = [int(round(c0[j] + f * (c1[j] - c0[j]))) for j in range(3)]
                break
    return lut


_LUT = _magma_lut()


def spectrogram_png(x, path, sr, n_fft=256, hop=64, height_scale=2, width_scale=1):
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    win = torch.hann_window(n_fft)
    S = torch.stft(xt, n_fft=n_fft, hop_length=hop, window=win,
                   return_complex=True).abs()                       # (F, T)
    S = torch.log1p(S).numpy()
    lo, hi = np.percentile(S, 2), np.percentile(S, 99.5)
    S = np.clip((S - lo) / (hi - lo + 1e-9), 0, 1)
    img = _LUT[(S * 255).astype(np.uint8)]                          # (F, T, 3)
    img = img[::-1]                                                 # low freq at bottom
    from PIL import Image
    im = Image.fromarray(img, "RGB")
    im = im.resize((im.width * width_scale, im.height * height_scale), Image.NEAREST)
    im.save(path)


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #

def real_val_reference(val_data, sr, out, n_chunks=24, chunk_sec=1.0):
    n = int(chunk_sec * sr)
    accs = []
    rng = np.random.default_rng(0)
    for _ in range(n_chunks):
        i = int(rng.integers(0, len(val_data) - n - 1))
        audio = mu_decode(val_data[i:i + n].astype(np.int64))
        accs.append(audio_stats(audio))
    # one reference spectrogram from a chunk that is not mostly silent
    for _ in range(50):
        i = int(rng.integers(0, len(val_data) - n - 1))
        audio = mu_decode(val_data[i:i + n].astype(np.int64))
        if audio_stats(audio)["silence"] < 0.5:
            spectrogram_png(audio, os.path.join(out, "REAL_val_reference.png"), sr)
            break
    keys = ("rms", "silence", "hf", "hf_rms")
    return {k: float(np.mean([a[k] for a in accs])) for k in keys}


def silence_speech_bits(model, val_data, cfg, device, batches=8, bs=2, sil_thr=8):
    sil_sum = sil_n = sp_sum = sp_n = 0.0
    tot_sum = tot_n = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(val_data, cfg.block_size, bs, device)
            logits, _, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 y.reshape(-1), reduction="none")
            yt = y.reshape(-1)
            is_sil = (yt - 128).abs() < sil_thr
            sil_sum += float(ce[is_sil].sum()); sil_n += int(is_sil.sum())
            sp_sum += float(ce[~is_sil].sum()); sp_n += int((~is_sil).sum())
            tot_sum += float(ce.sum()); tot_n += ce.numel()
    b = lambda s, n: (s / n / LN2) if n else float("nan")
    return dict(overall=b(tot_sum, tot_n), silence=b(sil_sum, sil_n),
                speech=b(sp_sum, sp_n),
                silence_frac=sil_n / tot_n if tot_n else float("nan"),
                sil_thr=sil_thr)


def temp_triplet(model, val_data, cfg, temps, n_primers, primer_sec, gen_sec,
                 out, sr, device):
    n_prime = int(primer_sec * sr)
    n_new = int(gen_sec * sr)
    rng = np.random.default_rng(123)
    rows = []
    for pi in range(n_primers):
        i = int(rng.integers(0, len(val_data) - n_prime - 1))
        primer = torch.from_numpy(val_data[i:i + n_prime].astype(np.int64)).to(device)
        for temp in temps:
            toks = generate(model, primer, n_new, temperature=temp)
            gen = mu_decode(toks[n_prime:].numpy())          # model portion only
            st = audio_stats(gen)
            base = f"p{pi}_t{temp:.2f}"
            audio_full = mu_decode(toks.numpy())
            write_wav(os.path.join(out, base + ".wav"),
                      torch.as_tensor(audio_full, dtype=torch.float32), sr)
            spectrogram_png(audio_full, os.path.join(out, base + ".png"), sr)
            rows.append((pi, temp, st))
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dir with val.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.90, 0.95, 1.00])
    ap.add_argument("--n-primers", type=int, default=2)
    ap.add_argument("--primer-sec", type=float, default=0.2)
    ap.add_argument("--gen-sec", type=float, default=0.8)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--val-bs", type=int, default=2)
    ap.add_argument("--sil-thr", type=int, default=8, help="|code-128| < thr counts as silence")
    ap.add_argument("--sr", type=int, default=8000, help="must match how the bins were prepared")
    ap.add_argument("--device", default="cpu", help="cpu (default; safe vs a live GPU run) or cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    model, cfg, step = load_model(args.ckpt, device)
    val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint8, mode="r")
    print(f"[probe] ckpt step {step}, {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"params, device={device}, sr={args.sr}")

    print("[probe] real-val reference...")
    ref = real_val_reference(val_data, args.sr, args.out)
    print("[probe] silence/speech bits...")
    bits = silence_speech_bits(model, val_data, cfg, device,
                               batches=args.val_batches, bs=args.val_bs, sil_thr=args.sil_thr)
    print("[probe] temperature triplet...")
    rows = temp_triplet(model, val_data, cfg, args.temps, args.n_primers,
                        args.primer_sec, args.gen_sec, args.out, args.sr, device)

    print(f"\n=== audio probe @ step {step} ({time.time()-t0:.0f}s) ===")
    print(f"REAL val reference:  rms {ref['rms']:.4f}  silence {ref['silence']*100:4.1f}%  "
          f"hf {ref['hf']:.4f}  hf/rms {ref['hf_rms']:.3f}")
    print(f"val bits/sample:  overall {bits['overall']:.3f}  "
          f"silence {bits['silence']:.3f}  speech {bits['speech']:.3f}  "
          f"(silence={bits['silence_frac']*100:.1f}% of tokens, |code-128|<{bits['sil_thr']})")
    print("generated (model portion only):")
    print("  primer  temp | rms     silence  hf/rms   (target hf/rms %.3f)" % ref["hf_rms"])
    for pi, temp, st in rows:
        print("  p%-5d %.2f | %.4f  %4.1f%%   %.3f" %
              (pi, temp, st["rms"], st["silence"] * 100, st["hf_rms"]))

    report = dict(step=step, real_val=ref, bits=bits,
                  generated=[dict(primer=pi, temp=t, **st) for pi, t, st in rows])
    with open(os.path.join(args.out, f"probe_step{step:06d}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] wavs + spectrograms + probe_step{step:06d}.json -> {args.out}")


if __name__ == "__main__":
    main()
