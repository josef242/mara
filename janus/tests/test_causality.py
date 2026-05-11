"""Causality verification suite for JanusModel.

REQUIRED to pass before any training run (per Nexus #89 §4 and #91 Q10).

Three tests from the v1 design doc §4:

  1. Scramble token N — verify that for every head (NTP, MTP h=2..5), only
     loss contributions at positions affected by the change are different.
     The "earliest affected" position is well-defined per head:
       - NTP loss at p depends on h[p] (sees tokens[0..p]) and target tokens[p+1].
         Changing tokens[N] affects h[p] for p>=N and target tokens[p+1] for p=N-1.
         → NTP loss[p] for p < N-1 MUST be unchanged.
       - MTP[h] loss at p depends on h[p] and target tokens[p+h].
         → MTP[h] loss[p] for p < (N-h) MUST be unchanged.

  2. Zero bridges — model still produces finite output and gradients flow to
     both stacks' parameters. Bridges are not required for the forward pass.

  3. Zero one stack at merge — model still produces finite output with the
     other stack alone. The trunk must remain functional with degraded input.

Run as a script:
    python -m janus.tests.test_causality
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
_COMMON = (_REPO.parent / "common_fsdp2").resolve()
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from janus.config import JanusConfig
from janus.model import JanusModel


# ----------------------------------------------------------------------------
# Tiny config for fast CPU-side verification
# ----------------------------------------------------------------------------

def _tiny_config(vocab_size: int = 64, seq_len: int = 32) -> JanusConfig:
    return JanusConfig(
        L_bilateral=6,
        d_stack=64,
        n_heads_stack=4,
        n_kv_heads_stack=4,
        inner_dim_stack=128,
        window_size_local=8,
        rope_theta_local=10_000.0,
        rope_theta_global=100_000.0,
        bridge_frequency=3,
        merge_op="cross_attn_fuse",
        d_synth=64,
        L_synthesis=2,
        n_heads_trunk=4,
        n_kv_heads_trunk=4,
        inner_dim_trunk=128,
        rope_theta_trunk=100_000.0,
        use_keel=True,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        mtp_horizons=(2, 3, 4, 5),
        use_activation_checkpointing=False,
        dropout=0.0,
    )


# ----------------------------------------------------------------------------
# Helper — compute per-position per-head loss without reduction.
# Bypasses cce_loss (which only supports reduction="mean") and uses manual CE.
# ----------------------------------------------------------------------------

def _per_position_losses(model: JanusModel, tokens: torch.Tensor, targets: torch.Tensor):
    """Return dict[head_name] -> [B, S-effective] per-position CE losses.

    For NTP at position p, target is tokens[p+1] (valid for p in [0, S-1)).
    For MTP[h] at position p, target is tokens[p+h] (valid for p in [0, S-h)).
    """
    with torch.no_grad():
        logits, _ = model(tokens, targets=None)
    bsz, seqlen, vocab = logits["ntp"].shape

    out = {}
    # NTP
    ntp_logits = logits["ntp"][:, :-1].contiguous()   # [B, S-1, V]
    ntp_targets = targets[:, 1:].contiguous()         # [B, S-1]
    out["ntp"] = F.cross_entropy(
        ntp_logits.view(-1, vocab),
        ntp_targets.view(-1),
        reduction="none",
    ).view(bsz, seqlen - 1)

    for h in model.mtp_horizons:
        valid = seqlen - h
        if valid <= 0:
            continue
        mtp_logits = logits[f"h{h}"][:, :valid].contiguous()
        mtp_targets = targets[:, h:h + valid].contiguous()
        out[f"h{h}"] = F.cross_entropy(
            mtp_logits.view(-1, vocab),
            mtp_targets.view(-1),
            reduction="none",
        ).view(bsz, valid)
    return out


# ----------------------------------------------------------------------------
# Test 1: scramble token N — causality leak detector
# ----------------------------------------------------------------------------

def test_scramble_token_N() -> None:
    """The most important test. Catches the worst class of causality bugs.

    For each head, after changing tokens[N], loss at positions strictly earlier
    than the head's "earliest reachable" position must be IDENTICAL (within
    a small tolerance for floating-point noise).
    """
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = JanusModel(cfg).eval()
    B, S = 1, 24
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = tokens.clone()  # arbitrary targets; we only compare loss differences

    # Reference forward
    ref_losses = _per_position_losses(model, tokens, targets)

    # Scramble at position N (middle of sequence)
    N = 12
    scrambled = tokens.clone()
    # Pick a token guaranteed to differ from the original at position N
    scrambled[0, N] = (tokens[0, N].item() + 17) % cfg.vocab_size
    # Targets must follow the change too (target at p=N-1 is tokens[N], etc.)
    scrambled_targets = scrambled.clone()
    new_losses = _per_position_losses(model, scrambled, scrambled_targets)

    tol = 1e-5

    # NTP: positions p < N-1 must be unchanged.
    # Why N-1: at p=N-1, target = tokens[N], which we just changed.
    # The hidden state h[p] is unchanged for p < N (didn't see the new token yet).
    ntp_diff = (ref_losses["ntp"] - new_losses["ntp"]).abs()
    assert ntp_diff[0, :N - 1].max().item() < tol, (
        f"NTP causality leak: position {ntp_diff[0, :N-1].argmax().item()} "
        f"changed (max diff {ntp_diff[0, :N-1].max().item():.2e}) when token "
        f"at N={N} was scrambled. Earliest expected change at p={N-1}."
    )
    # And it MUST change at p=N-1 (target changed) — sanity that the test runs
    assert ntp_diff[0, N - 1].item() > tol, (
        f"NTP test inert: loss at p={N-1} did not change after scrambling target. "
        f"Test is not actually exercising the model."
    )

    # MTP[h]: positions p < N-h must be unchanged.
    for h in cfg.mtp_horizons:
        if f"h{h}" not in ref_losses:
            continue
        diff = (ref_losses[f"h{h}"] - new_losses[f"h{h}"]).abs()
        valid_up_to = N - h
        if valid_up_to > 0:
            assert diff[0, :valid_up_to].max().item() < tol, (
                f"MTP h{h} causality leak: position "
                f"{diff[0, :valid_up_to].argmax().item()} changed when token at "
                f"N={N} was scrambled. Earliest expected change at p={N-h}."
            )

    print("[ok] scramble_token_N (NTP up to p<%d, MTP per-horizon up to p<N-h)" % (N - 1))


# ----------------------------------------------------------------------------
# Test 2: zero bridges — model still trains
# ----------------------------------------------------------------------------

def test_zero_bridges_forward_and_backward() -> None:
    cfg = _tiny_config()
    torch.manual_seed(1)
    model = JanusModel(cfg).train()
    B, S = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))

    _, loss_dict = model(tokens, targets=targets, zero_bridges=True)
    assert torch.isfinite(loss_dict["loss"]), \
        f"loss not finite with zero_bridges=True: {loss_dict['loss']}"
    loss_dict["loss"].backward()

    # Sanity: bridge params should have NO gradient (they were skipped). But
    # both stacks' non-bridge params should still have gradients.
    bridge_idx = cfg.bridge_layer_indices[0]
    bridge_kv_grad = model.local_blocks[bridge_idx].bridge.wk.weight.grad
    assert bridge_kv_grad is None or bridge_kv_grad.abs().sum() == 0, \
        "bridge K projection got a gradient despite zero_bridges=True"

    # A non-bridge sublayer in the local stack must have gradient flow
    local_attn_q = model.local_blocks[0].self_attn.wq.weight.grad
    global_attn_q = model.global_blocks[0].self_attn.wq.weight.grad
    assert local_attn_q is not None and local_attn_q.abs().sum() > 0, \
        "local stack received no gradient with zero_bridges=True"
    assert global_attn_q is not None and global_attn_q.abs().sum() > 0, \
        "global stack received no gradient with zero_bridges=True"
    print("[ok] zero_bridges_forward_and_backward")


# ----------------------------------------------------------------------------
# Test 3: zero one stack at merge — degraded but functional
# ----------------------------------------------------------------------------

def test_zero_local_stack_at_merge() -> None:
    cfg = _tiny_config()
    torch.manual_seed(2)
    model = JanusModel(cfg).eval()
    B, S = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        _, loss_dict = model(tokens, targets=targets, zero_local_at_merge=True)
    assert torch.isfinite(loss_dict["loss"]), \
        f"loss not finite with zero_local_at_merge=True: {loss_dict['loss']}"
    print("[ok] zero_local_stack_at_merge")


def test_zero_global_stack_at_merge() -> None:
    cfg = _tiny_config()
    torch.manual_seed(3)
    model = JanusModel(cfg).eval()
    B, S = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        _, loss_dict = model(tokens, targets=targets, zero_global_at_merge=True)
    assert torch.isfinite(loss_dict["loss"]), \
        f"loss not finite with zero_global_at_merge=True: {loss_dict['loss']}"
    print("[ok] zero_global_stack_at_merge")


def main() -> int:
    torch.manual_seed(0)
    tests = [
        test_scramble_token_N,
        test_zero_bridges_forward_and_backward,
        test_zero_local_stack_at_merge,
        test_zero_global_stack_at_merge,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {e!r}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} causality test(s) failed: {failed}")
        return 1
    print(f"\nAll {len(tests)} causality tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
