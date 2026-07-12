"""Progressive Tail Truncation: gradient isolation + schedule sanity.

Ported from the old mara_fsdp2/test_ptt.py, which printed PASS/FAIL but never
asserted (no failure signal). This version pins the same claims for real.
"""
import torch
import pytest

from helpers import tiny_model


def _ffn_grad_norms(model):
    out = {}
    for i, layer in enumerate(model.layers):
        g = layer.feed_forward.w1.weight.grad
        out[i] = g.norm().item() if g is not None else 0.0
    return out


def test_truncated_forward_gradient_isolation():
    """Skipped tail layers must receive exactly zero gradient; active layers,
    the final norm, output head, and embeddings must all receive gradient."""
    n_layers, active = 8, 5
    model = tiny_model(n_layers=n_layers)
    model.train()
    tokens = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    # Baseline: full forward reaches every layer.
    model.zero_grad()
    _, loss_full = model(tokens, targets)
    loss_full.backward()
    full = _ffn_grad_norms(model)
    assert all(full[i] > 0 for i in range(n_layers)), f"full forward left dead layers: {full}"

    # Truncated forward: layers >= active must be untouched.
    model.zero_grad()
    _, loss_trunc = model(tokens, targets, active_layers=active)
    loss_trunc.backward()
    trunc = _ffn_grad_norms(model)
    for i in range(active):
        assert trunc[i] > 0, f"active layer {i} got no gradient"
    for i in range(active, n_layers):
        assert trunc[i] == 0.0, f"skipped layer {i} got gradient {trunc[i]}"

    for name, p in [("norm", model.norm.weight),
                    ("output", model.output.weight),
                    ("embeddings", model.tok_embeddings.weight)]:
        assert p.grad is not None and p.grad.norm().item() > 0, \
            f"{name} got no gradient under truncation"


def test_ptt_schedule_respects_safe_fraction():
    """Cut points never dip below safe_fraction * n_layers, and the scheduled
    form produces no truncation before its ramp starts."""
    from tail_truncation import ProgressiveTailTruncation

    n_layers = 70
    ptt = ProgressiveTailTruncation(n_layers=n_layers, config={
        'enabled': True, 'safe_fraction': 0.60,
        'truncation_prob': 0.25, 'depth_power': 2.0,
    })
    safe_layer = int(n_layers * 0.60)
    cuts = [ptt.get_truncation_point(s) for s in range(5000)]
    truncated = [c for c in cuts if c < n_layers]
    assert truncated, "no truncation events in 5000 steps at prob=0.25"
    assert min(truncated) >= safe_layer, \
        f"cut {min(truncated)} below safe layer {safe_layer}"
    frac = len(truncated) / len(cuts)
    assert 0.15 < frac < 0.35, f"truncation rate {frac:.2f} far from prob=0.25"

    sched = ProgressiveTailTruncation(n_layers=n_layers, config={
        'enabled': True,
        'safe_fraction': [[0, 1.0], [1000, 0.60]],
        'truncation_prob': [[0, 0.0], [1000, 0.25]],
        'depth_power': 2.0,
    })
    early = [sched.get_truncation_point(s) for s in range(0, 50)]
    assert all(c == n_layers for c in early), "truncation fired before schedule ramp"
