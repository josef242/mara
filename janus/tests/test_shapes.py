"""Shape sanity checks at every interface of JanusModel.

Run as a script:
    python -m janus.tests.test_shapes

These tests use a tiny config (small d, few layers, short T) so they run
quickly on CPU and exercise every interface: embedding, both stacks, bridge
sublayers, merge, trunk, output heads, and the loss path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add common_fsdp2 to sys.path so `from model_v2 import ...` works the same way
# the train script does.
_REPO = Path(__file__).resolve().parents[2]
_COMMON = (_REPO.parent / "common_fsdp2").resolve()
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from janus.config import JanusConfig
from janus.model import JanusModel


def _tiny_config(vocab_size: int = 256, seq_len: int = 64) -> JanusConfig:
    """Minimal config that exercises the full architecture on CPU.

    d=64 with 4 heads (head_dim=16). 6 bilateral layers with k=3 → bridges at
    layers 3 and 6 (1-indexed) i.e. indices {2, 5}. 2 trunk layers. Window=8.
    """
    return JanusConfig(
        L_bilateral=6,
        d_stack=64,
        n_heads_stack=4,
        n_kv_heads_stack=4,
        inner_dim_stack=128,
        window_size_local=8,
        rope_theta_local=10_000.0,
        rope_theta_global=100_000.0,
        bridge_frequency=3,             # bridges at indices {2, 5}
        merge_op="cross_attn_fuse",
        d_synth=64,                     # match stack for simplicity
        L_synthesis=2,
        n_heads_trunk=4,
        n_kv_heads_trunk=4,
        inner_dim_trunk=128,
        rope_theta_trunk=100_000.0,
        use_keel=True,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        mtp_horizons=(2, 3, 4, 5),
        use_activation_checkpointing=False,  # disable for cleaner test traces
    )


def test_config_derived_fields() -> None:
    cfg = _tiny_config()
    assert cfg.head_dim_stack == 16
    assert cfg.head_dim_trunk == 16
    assert cfg.bridge_layer_indices == (2, 5), \
        f"expected bridge indices (2, 5), got {cfg.bridge_layer_indices}"
    assert cfg.bridge_layer_count == 2
    # KEEL alphas: stack = 2*6 + 2 = 14, trunk = 2*2 = 4
    assert cfg.keel_alpha_stack == 14.0
    assert cfg.keel_alpha_trunk == 4.0
    print("[ok] config_derived_fields")


def test_logits_shapes() -> None:
    cfg = _tiny_config()
    model = JanusModel(cfg).eval()
    B, S = 2, 32
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        logits, loss = model(tokens, targets=None)
    assert loss is None
    assert "ntp" in logits
    assert logits["ntp"].shape == (B, S, cfg.vocab_size), \
        f"ntp logits {logits['ntp'].shape}"
    for h in cfg.mtp_horizons:
        assert f"h{h}" in logits
        assert logits[f"h{h}"].shape == (B, S, cfg.vocab_size), \
            f"h{h} logits {logits[f'h{h}'].shape}"
    print("[ok] logits_shapes")


def test_loss_shapes_and_finite() -> None:
    cfg = _tiny_config()
    model = JanusModel(cfg).eval()
    B, S = 2, 32
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        logits, loss_dict = model(tokens, targets=targets)
    assert logits is None
    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"]), f"total loss not finite: {loss_dict['loss']}"
    assert "loss_ntp" in loss_dict
    assert torch.isfinite(loss_dict["loss_ntp"])
    for h in cfg.mtp_horizons:
        key = f"loss_mtp_h{h}"
        assert key in loss_dict
        assert torch.isfinite(loss_dict[key]), f"{key} not finite: {loss_dict[key]}"
    print("[ok] loss_shapes_and_finite")


def test_backward_through_bridges() -> None:
    """Gradient must flow through bridge K/V projections — no detach."""
    cfg = _tiny_config()
    model = JanusModel(cfg).train()
    B, S = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    _, loss_dict = model(tokens, targets=targets)
    loss_dict["loss"].backward()

    # Pick a bridge layer; its K/V projection weights must have non-zero grads.
    bridge_idx = cfg.bridge_layer_indices[0]
    local_bridge_kv = model.local_blocks[bridge_idx].bridge.wk.weight.grad
    global_bridge_kv = model.global_blocks[bridge_idx].bridge.wk.weight.grad
    assert local_bridge_kv is not None and local_bridge_kv.abs().sum() > 0, \
        "local bridge K projection has no gradient"
    assert global_bridge_kv is not None and global_bridge_kv.abs().sum() > 0, \
        "global bridge K projection has no gradient"
    print("[ok] backward_through_bridges")


def test_window_seq_shorter_than_window() -> None:
    """Window > seqlen should still work (window just becomes full causal)."""
    cfg = _tiny_config(seq_len=32)
    # Set window larger than test seqlen
    cfg.window_size_local = 64
    model = JanusModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    targets = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        _, loss_dict = model(tokens, targets=targets)
    assert torch.isfinite(loss_dict["loss"])
    print("[ok] window_seq_shorter_than_window")


def test_split_brain_probes_runtime() -> None:
    """Probes must run without errors and produce finite loss."""
    cfg = _tiny_config()
    model = JanusModel(cfg).eval()
    B, S = 2, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        _, l_zero_local = model(tokens, targets=targets, zero_local_at_merge=True)
        _, l_zero_global = model(tokens, targets=targets, zero_global_at_merge=True)
        _, l_zero_bridges = model(tokens, targets=targets, zero_bridges=True)
    for name, ld in [("zero_local", l_zero_local), ("zero_global", l_zero_global), ("zero_bridges", l_zero_bridges)]:
        assert torch.isfinite(ld["loss"]), f"{name} probe loss not finite: {ld['loss']}"
    print("[ok] split_brain_probes_runtime")


def test_param_count_sanity() -> None:
    cfg = _tiny_config()
    model = JanusModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    # For this tiny config: embeddings + 6 bilateral × 2 stacks + 2 trunk +
    # merge + 5 heads. Just sanity: count should be in a reasonable range.
    assert 100_000 < n_params < 50_000_000, f"unreasonable param count: {n_params}"
    print(f"[ok] param_count_sanity ({n_params:,} params)")


def main() -> int:
    torch.manual_seed(0)
    tests = [
        test_config_derived_fields,
        test_logits_shapes,
        test_loss_shapes_and_finite,
        test_backward_through_bridges,
        test_window_seq_shorter_than_window,
        test_split_brain_probes_runtime,
        test_param_count_sanity,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {t.__name__}: {e!r}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        return 1
    print(f"\nAll {len(tests)} shape tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
