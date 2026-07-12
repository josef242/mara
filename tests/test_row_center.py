"""Tests for output-head row-centering (gauge subtraction).

Run:  python test_row_center.py
Single-device (no torchrun) covers the correctness math; the distributed
global-vs-per-shard test lives in test_row_center_dist.py (needs torchrun).
"""
import sys
sys.path.insert(0, "../common_fsdp2")

import torch
import torch.nn.functional as F
from row_center import row_center_head_, row_norm_of_mean, _global_row_mean

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_function_preserving():
    """Centering W (and exp_avg) leaves CE / softmax / logprobs unchanged."""
    print("test_function_preserving")
    torch.manual_seed(0)
    V, D, N = 4000, 256, 128
    W = torch.randn(V, D) * 0.5 + 0.4          # real common-mode offset baked in
    h = torch.randn(N, D)
    tgt = torch.randint(0, V, (N,))

    z0 = h @ W.t()
    ce0 = F.cross_entropy(z0, tgt)
    lp0 = F.log_softmax(z0, dim=-1)
    mu_true = W.float().mean(0)

    exp_avg = torch.randn(V, D) * 0.1
    tel = row_center_head_(W, exp_avg=exp_avg)

    z1 = h @ W.t()
    ce1 = F.cross_entropy(z1, tgt)
    lp1 = F.log_softmax(z1, dim=-1)

    check("CE unchanged", abs(ce0 - ce1).item() < 1e-5, f"|d|={abs(ce0-ce1).item():.2e}")
    check("logprobs allclose", torch.allclose(lp0, lp1, atol=1e-5),
          f"max|d|={(lp0-lp1).abs().max().item():.2e}")
    # the logit delta is the uniform per-token shift h.mu
    delta = z0 - z1
    expected = (h @ mu_true).unsqueeze(1).expand_as(delta)
    check("logit delta == h.mu (uniform shift)", torch.allclose(delta, expected, atol=1e-4),
          f"max|d|={(delta-expected).abs().max().item():.2e}")
    check("delta constant across vocab", delta.std(dim=1).max().item() < 1e-4,
          f"max std={delta.std(dim=1).max().item():.2e}")


def test_gauge_removed():
    """After projection ||mu(W)||~0 and ||m_bar(exp_avg)||~0."""
    print("test_gauge_removed")
    torch.manual_seed(1)
    V, D = 3000, 128
    W = torch.randn(V, D) + 0.7
    exp_avg = torch.randn(V, D) * 0.2 + 0.05    # its own nonzero gauge
    mu_before = row_norm_of_mean(W)
    m_before = row_norm_of_mean(exp_avg)
    tel = row_center_head_(W, exp_avg=exp_avg)
    check("||mu(W)|| -> 0", row_norm_of_mean(W) < 1e-4,
          f"{mu_before:.4f} -> {row_norm_of_mean(W):.2e}")
    check("||m_bar|| -> 0", row_norm_of_mean(exp_avg) < 1e-4,
          f"{m_before:.4f} -> {row_norm_of_mean(exp_avg):.2e}")
    check("telemetry mu_w_pre matches", abs(tel['mu_w_pre'] - mu_before) < 1e-4)
    check("telemetry mu_w_post ~0", tel['mu_w_post'] < 1e-4)
    check("telemetry m_bar matches", abs(tel['m_bar'] - m_before) < 1e-4)


def test_exp_avg_own_mean():
    """The first moment must be centered by ITS OWN row-mean, not the weight mu.
    If we (wrongly) reused W's mu, exp_avg's gauge would NOT vanish."""
    print("test_exp_avg_own_mean")
    torch.manual_seed(2)
    V, D = 2000, 64
    W = torch.randn(V, D) + 1.0                  # big weight gauge
    exp_avg = torch.randn(V, D) * 0.1 - 0.3      # different-sign small gauge
    row_center_head_(W, exp_avg=exp_avg)
    # exp_avg's own gauge gone:
    check("exp_avg gauge removed by own mean", row_norm_of_mean(exp_avg) < 1e-4,
          f"->{row_norm_of_mean(exp_avg):.2e}")


def test_bf16_stochastic_writeback():
    """bf16 exp_avg: projection still drives the (bf16) row-mean toward ~0.
    Stochastic rounding is unbiased, so over the full tensor the mean residual
    is tiny relative to the original gauge (not bit-zero like fp32)."""
    print("test_bf16_stochastic_writeback")
    torch.manual_seed(3)
    V, D = 8000, 256
    W = (torch.randn(V, D) + 0.5)
    exp_avg = (torch.randn(V, D) * 0.1 + 0.2).to(torch.bfloat16)
    m_before = row_norm_of_mean(exp_avg)
    row_center_head_(W, exp_avg=exp_avg)
    m_after = row_norm_of_mean(exp_avg)
    check("bf16 exp_avg stays bf16", exp_avg.dtype is torch.bfloat16)
    # SR is unbiased: residual should be a tiny fraction of the original gauge.
    check("bf16 m_bar reduced >50x", m_after < m_before / 50.0,
          f"{m_before:.4f} -> {m_after:.5f}")


def test_telemetry_ratio():
    """proj_fro = sqrt(V)*||mu||; proj_ratio = proj_fro/||W||_F in (0,1)."""
    print("test_telemetry_ratio")
    torch.manual_seed(4)
    V, D = 5000, 128
    W = torch.randn(V, D) * 0.3 + 0.6
    w_fro = W.norm().item()
    mu = W.float().mean(0)
    proj_fro_expected = (V ** 0.5) * mu.norm().item()
    tel = row_center_head_(W.clone())  # clone so we keep W for the ratio check
    check("proj_fro = sqrt(V)*||mu||", abs(tel['proj_fro'] - proj_fro_expected) < 1e-2,
          f"{tel['proj_fro']:.3f} vs {proj_fro_expected:.3f}")
    check("proj_ratio in (0,1)", 0.0 < tel['proj_ratio'] < 1.0, f"{tel['proj_ratio']:.4f}")
    check("proj_ratio = proj_fro/||W||", abs(tel['proj_ratio'] - proj_fro_expected / w_fro) < 1e-3)


if __name__ == "__main__":
    test_function_preserving()
    test_gauge_removed()
    test_exp_avg_own_mean()
    test_bf16_stochastic_writeback()
    test_telemetry_ratio()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL PASS")
