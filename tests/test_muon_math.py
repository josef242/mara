"""Muon optimizer math primitives — the pieces testable without a mesh.

The full Fsdp1dWork gather/NS/scatter path needs a process group (see
test_muon_dist.py, dist marker); these pin the pure math it composes.

MUON_NS_COMPILE=0 is forced before import: torch.compile autotuning is slow
and unavailable on the local Windows tier (no triton).
"""
import os

os.environ.setdefault("MUON_NS_COMPILE", "0")

import pytest
import torch

from muon_fsdp2 import (apply_momentum, apply_scaling, adam_update,
                        apply_normuon, zeropower_via_newtonschulz5)


def test_apply_momentum_never_returns_buffer():
    """The aliasing contract the code comments promise: callers scatter into
    the returned tensor's storage in place, so returning the momentum buffer
    would destroy accumulation (past real-bug class)."""
    for nesterov in (False, True):
        g = torch.randn(8, 8)
        m = torch.zeros(8, 8)
        out = apply_momentum(g, m, 0.9, nesterov)
        assert out is not m, f"nesterov={nesterov}: returned the momentum buffer"


def test_apply_momentum_ema_closed_form():
    beta = 0.9
    g1, g2 = torch.randn(4, 4), torch.randn(4, 4)
    m = torch.zeros(4, 4)
    apply_momentum(g1.clone(), m, beta, nesterov=False)
    out = apply_momentum(g2.clone(), m, beta, nesterov=False)
    # m after two lerps from zero: (1-beta)*(beta*g1 + g2)
    expect_m = (1 - beta) * (beta * g1 + g2)
    assert torch.allclose(m, expect_m, atol=1e-6)
    assert torch.allclose(out, expect_m, atol=1e-6)  # nesterov=False returns m's value


def test_newton_schulz_orthogonalizes():
    """NS(G) should have singular values ~1 (the quintic converges to a band
    around 1, per the docstring ~[0.3, 1.7] loosely — we pin a generous band
    and the shape/transpose contracts)."""
    torch.manual_seed(0)
    for shape in [(16, 48), (48, 16), (32, 32)]:
        G = torch.randn(*shape)
        X = zeropower_via_newtonschulz5(G.bfloat16(), 5).float()
        assert X.shape == G.shape
        s = torch.linalg.svdvals(X)
        assert s.max().item() < 1.8, f"{shape}: max sv {s.max().item()}"
        assert s.min().item() > 0.2, f"{shape}: min sv {s.min().item()}"
        # sign preservation: NS(G) stays positively aligned with G
        assert (X * G).sum().item() > 0


def test_newton_schulz_zero_input_is_finite():
    """A dead layer (zero grad) must not produce NaNs (0/0 in the norm)."""
    X = zeropower_via_newtonschulz5(torch.zeros(8, 16).bfloat16(), 5)
    assert torch.isfinite(X.float()).all()


def test_apply_scaling_rms_mode():
    g = torch.randn(64, 32)
    expect = g * 0.2 * (64 ** 0.5)
    out = apply_scaling(g.clone(), rms_scale=True)
    assert torch.allclose(out, expect)


def test_adam_update_matches_reference():
    """adam_update == textbook bias-corrected Adam direction (no lr, no wd)."""
    torch.manual_seed(1)
    betas, eps = (0.9, 0.95), 1e-8
    g = torch.randn(6, 6)
    buf1, buf2 = torch.zeros(6, 6), torch.zeros(6, 6)
    out = adam_update(g.clone(), buf1, buf2, step=1, betas=betas, eps=eps)
    m_hat = (1 - betas[0]) * g / (1 - betas[0])
    v_hat = (1 - betas[1]) * g.square() / (1 - betas[1])
    assert torch.allclose(out, m_hat / (v_hat.sqrt() + eps), atol=1e-6)


def test_apply_normuon_preserves_frobenius_norm():
    torch.manual_seed(2)
    u = torch.randn(16, 8)
    pre = u.norm().item()
    sm = torch.zeros(16, 1)
    out = apply_normuon(u, sm, 0.95)
    assert abs(out.norm().item() - pre) / pre < 1e-4
    assert (sm > 0).all()  # second moment engaged


def test_single_device_muon_path_fails_loudly():
    """AUDIT 2026-07-11 finding #4 (HIGH), fixed same day: the single-device
    Muon path used to be a half-implemented NameError; it now refuses loudly
    with guidance (1-rank fully_shard). If a real plain-tensor path ever
    lands, it must arrive with a parity test against a 1-rank Fsdp1dWork —
    then this test can be replaced by that one."""
    import muon_fsdp2
    work_cls = muon_fsdp2.SingelDeviceWork
    p = torch.nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)
    group = {"momentum": 0.95, "nesterov": True, "ns_steps": 5, "rms_scale": True,
             "lr": 0.01, "weight_decay": 0.0}
    state = {"momentum_buffer": torch.zeros(8, 8)}
    w = work_cls(p, state, group, 0, {}, {})
    before = p.detach().clone()
    with pytest.raises(NotImplementedError, match="1-rank"):
        w.start()
    assert torch.equal(p.detach(), before), "param mutated before the refusal"
