"""Meta-device init completeness — the RoPE-corruption regression net.

History: every FSDP2 meta-init run before 2026-07-02 trained without
functional RoPE because to_empty() clobbers the persistent=False freqs
buffers and nothing recomputed them. The fix recomputes them in
init_weights(). This file pins that fix and sweeps for siblings: ANY
param/buffer that init_weights leaves as uninitialized to_empty() garbage.

The meta flow here mirrors the trainer's (meta construct -> to_empty ->
init_weights) minus fully_shard, which does not change which tensors get
reinitialized.
"""
import pytest
import torch

from model_v2 import ModelArgs, Transformer, precompute_freqs_cis

from helpers import tiny_args


def _meta_init_model(**over):
    args = tiny_args(**over)
    with torch.device("meta"):
        model = Transformer(args)
    model = model.to_empty(device="cpu")
    # Poison the storage so "happened to be zero pages" can't mask a miss:
    # anything init_weights does not touch stays exactly 12345.0.
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point():
                t.fill_(12345.0)
    model.init_weights()
    return args, model


POISON = 12345.0


def _untouched(model):
    bad = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t.is_floating_point() and t.numel() and (t == POISON).float().mean() > 0.5:
            bad.append(name)
    return bad


def test_freqs_recomputed_after_meta_init():
    """THE regression test for the 2026-07-02 RoPE corruption."""
    args, model = _meta_init_model()
    fc, fs = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len,
                                  args.rope_theta)
    assert torch.equal(model.freqs_cos, fc)
    assert torch.equal(model.freqs_sin, fs)
    assert model.freqs_sin.abs().sum() > 0


def test_meta_init_touches_every_tensor_dense():
    _, model = _meta_init_model()
    assert _untouched(model) == []


def test_meta_init_touches_every_tensor_moe():
    _, model = _meta_init_model(
        moe_enabled=True, moe_num_experts=4, moe_top_k=2,
        moe_num_shared_experts=1, moe_load_balance_coeff=1e-3)
    assert _untouched(model) == []
    moe = model.layers[0].moe if model.layers[0].moe_enabled else model.layers[1].moe
    assert moe.expert_bias.abs().sum().item() == 0
    assert moe.tokens_per_expert.abs().sum().item() == 0


def test_meta_init_touches_every_tensor_mtp():
    _, model = _meta_init_model(mtp_enabled=True)
    assert _untouched(model) == []


def test_meta_init_attn_res_queries_are_zero():
    """AUDIT 2026-07-11 finding #3, fixed same day: attn_res_queries matched
    no init_weights branch and stayed to_empty() garbage."""
    _, model = _meta_init_model(attn_res_enabled=True)
    assert _untouched(model) == []
    for q in model.attn_res_queries:
        assert q.abs().sum().item() == 0.0


def _has_fla():
    try:
        import fla  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_fla(), reason="fla not installed (rig-only)")
def test_meta_init_touches_every_tensor_gdn():
    """AUDIT 2026-07-11 finding #1 (CRITICAL latent), fixed same day: GDN's
    A_log/dt_bias/conv1d/o_norm matched no init_weights branch (RoPE-bug
    sibling). Beyond 'touched', pin the FLA recipe itself so a drive-by
    simplification can't replace it with generic garbage-but-finite init."""
    import math
    _, model = _meta_init_model(gdn_enabled=True, gdn_interleave_step=2,
                                n_gdn_heads=2, gdn_head_dim=16)
    assert _untouched(model) == []
    checked = 0
    for layer in model.layers:
        if not layer.use_gdn:
            continue
        gdn = layer.gdn_attn
        # A_log = log(U(0,16)): all <= log 16, and spread out (not constant)
        assert gdn.A_log.max().item() <= math.log(16.0) + 1e-6
        assert gdn.A_log.std().item() > 0
        # dt_bias = inv_softplus(dt), dt in [1e-4, 0.1] -> softplus(dt_bias) in range
        dt = torch.nn.functional.softplus(gdn.dt_bias)
        assert dt.min().item() >= 1e-4 * (1 - 1e-4)
        assert dt.max().item() <= 0.1 * (1 + 1e-4)
        # o_norm is ones-init
        assert torch.equal(gdn.o_norm.weight, torch.ones_like(gdn.o_norm.weight))
        checked += 1
    assert checked > 0, "config produced no GDN layers — test is vacuous"


def test_rmsnorm_weights_are_ones_after_meta_init():
    _, model = _meta_init_model()
    for name, mod in model.named_modules():
        if type(mod).__name__ == "RMSNorm" and getattr(mod, "weight", None) is not None:
            assert torch.equal(mod.weight, torch.ones_like(mod.weight)), name


def test_paired_arm_trunk_identical_with_and_without_mtp():
    """MTP inits strictly after every trunk draw, so the trunk is bit-identical
    with mtp on/off at the same seed (the paired-arm A/B property)."""
    torch.manual_seed(99)
    _, m_off = _meta_init_model(mtp_enabled=False)
    torch.manual_seed(99)
    _, m_on = _meta_init_model(mtp_enabled=True)
    off_params = dict(m_off.named_parameters())
    for name, p in m_on.named_parameters():
        if name.startswith("mtp."):
            continue
        assert torch.equal(p, off_params[name]), f"trunk param {name} diverged"
