"""KV-cache decode vs full forward — the inference-path differential tests.

Lesson baked in from the spec-decode work: differential tests on an untrained
model can pass with a broken component when distributions are too symmetric,
so these run in fp32 with a seeded, randomly-initialized model and compare
LOGITS (not samples) to tight tolerance.

Global attention only (SWA/flex parity is rig-tier: needs triton).
"""
import pytest
import torch

from helpers import tiny_model

B, PREFILL, TOTAL = 2, 5, 12
ATOL = 1e-5


@pytest.fixture()
def model():
    m = tiny_model(max_seq_len=32)
    m.eval()
    return m


def _full_logits(model, toks):
    with torch.no_grad():
        logits, _ = model(toks)  # inference branch: ([B, S, vocab], None)
    return logits


def test_prefill_plus_decode_equals_full_forward(model):
    torch.manual_seed(7)
    toks = torch.randint(0, 256, (B, TOTAL))
    full = _full_logits(model, toks)

    model.setup_caches(B, 32)
    with torch.no_grad():
        out = [model.generate_forward(toks[:, :PREFILL], start_pos=0)]
        for p in range(PREFILL, TOTAL):
            out.append(model.generate_forward(toks[:, p:p + 1], start_pos=p))
    cached = torch.cat(out, dim=1)
    model.clear_caches()

    diff = (cached - full).abs().max().item()
    assert diff < ATOL, f"decode/full divergence {diff}"


def test_suffix_prefill_equals_sequential_decode(model):
    """Multi-token chunk at start_pos>0 (the bottom-right-aligned mask branch)
    must equal one-at-a-time decode."""
    torch.manual_seed(8)
    toks = torch.randint(0, 256, (B, TOTAL))

    model.setup_caches(B, 32)
    with torch.no_grad():
        model.generate_forward(toks[:, :PREFILL], start_pos=0)
        seq = [model.generate_forward(toks[:, p:p + 1], start_pos=p)
               for p in range(PREFILL, TOTAL)]
    seq = torch.cat(seq, dim=1)

    model.setup_caches(B, 32, force=True)
    with torch.no_grad():
        model.generate_forward(toks[:, :PREFILL], start_pos=0)
        chunk = model.generate_forward(toks[:, PREFILL:TOTAL], start_pos=PREFILL)
    model.clear_caches()

    diff = (chunk - seq).abs().max().item()
    assert diff < ATOL, f"suffix-prefill/sequential divergence {diff}"


def test_training_forward_unaffected_by_cache_cycle(model):
    """Cache machinery must never leak into the training path."""
    torch.manual_seed(9)
    toks = torch.randint(0, 256, (B, 8))
    tgts = torch.randint(0, 256, (B, 8))
    model.train()
    _, loss_before = model(toks, tgts)

    model.eval()
    model.setup_caches(B, 16)
    with torch.no_grad():
        model.generate_forward(toks, start_pos=0)
    model.clear_caches()
    assert not model.has_caches()

    model.train()
    _, loss_after = model(toks, tgts)
    assert torch.equal(loss_before, loss_after)


def test_setup_caches_rejects_beyond_trained_length(model):
    with pytest.raises(ValueError):
        model.setup_caches(B, model.params.max_seq_len + 1)


def test_rope_relative_invariance():
    """RoPE's defining property, pinned exactly: the q·k score between a fixed
    query vector at position p+d and a fixed key vector at position p depends
    only on d, never on p. The 2026-07-02 corruption (cos table zeroed) turns
    this into an absolute-position envelope and fails this by orders of
    magnitude — no tuned threshold needed, it's exact math up to fp error."""
    from model_v2 import precompute_freqs_cis, apply_rotary_emb

    torch.manual_seed(10)
    S, H, HD = 24, 1, 16
    fc, fs = precompute_freqs_cis(HD, S, 500000.0)
    vq = torch.randn(HD)
    vk = torch.randn(HD)
    # The same vector at every position -> any score variation is pure RoPE.
    xq = vq.expand(1, S, H, HD).contiguous()
    xk = vk.expand(1, S, H, HD).contiguous()
    q_rot, k_rot = apply_rotary_emb(xq, xk, fc, fs)
    scores = torch.einsum("shd,thd->st", q_rot[0].float(), k_rot[0].float())

    for d in (1, 3, 7):
        diag = torch.tensor([scores[p + d, p] for p in range(S - d)])
        assert (diag - diag[0]).abs().max().item() < 1e-4, \
            f"score(p+{d}, p) varies with p: {diag.tolist()[:5]}..."

    # And the corrupted tables (cos<-0, sin<-cos) must NOT pass, so the test
    # can never rot into vacuity.
    q_bad, k_bad = apply_rotary_emb(xq, xk, torch.zeros_like(fc), fc)
    bad = torch.einsum("shd,thd->st", q_bad[0].float(), k_bad[0].float())
    d = 3
    bad_diag = torch.tensor([bad[p + d, p] for p in range(S - d)])
    assert (bad_diag - bad_diag[0]).abs().max().item() > 1e-2, \
        "corruption oracle no longer discriminates — test is vacuous"
