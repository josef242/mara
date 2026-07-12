"""MTP (multi-token prediction) invariants, incl. the doc-boundary target
mask added 2026-07-13 (audit #18): under packed documents, the t+2 objective
at a boundary row conditions doc-A state + doc-B's BOS embedding to predict
doc-B's first content token — cross-document supervision. The optional
mtp.doc_boundary_mask ignores exactly those rows.
"""
import pytest
import torch

from model_v2 import ModelArgs, Transformer

BOS = 1


def _mtp_model(seed=42, **over):
    args = dict(dim=64, n_layers=2, n_heads=4, vocab_size=256, max_seq_len=32,
                dropout=0.0, use_activation_checkpointing=False, pad_id=0,
                use_keel=False, mtp_enabled=True, bos_token_id=BOS)
    args.update(over)
    torch.manual_seed(seed)
    m = Transformer(ModelArgs(**args))
    m.train()
    return m


def _mtp_loss(model, toks, tgts):
    model.zero_grad()
    _, _ = model(toks, tgts)
    loss = model._last_mtp_loss
    assert loss is not None
    return loss.item()


def test_boundary_mask_inert_without_bos_in_batch():
    """Flag on vs off must be bit-identical when the window contains no BOS
    (same seed => same weights; the flag consumes no RNG)."""
    toks = torch.randint(2, 256, (2, 16))   # no BOS anywhere
    tgts = torch.randint(2, 256, (2, 16))
    on = _mtp_model(mtp_doc_boundary_mask=True)
    off = _mtp_model(mtp_doc_boundary_mask=False)
    assert _mtp_loss(on, toks, tgts) == _mtp_loss(off, toks, tgts)


def test_boundary_mask_changes_loss_at_boundaries():
    torch.manual_seed(3)
    toks = torch.randint(2, 256, (2, 16))
    tgts = torch.randint(2, 256, (2, 16))
    tgts[:, 5] = BOS  # a document boundary mid-window
    on = _mtp_model(mtp_doc_boundary_mask=True)
    off = _mtp_model(mtp_doc_boundary_mask=False)
    l_on, l_off = _mtp_loss(on, toks, tgts), _mtp_loss(off, toks, tgts)
    assert l_on != l_off, "boundary row was not excluded from the MTP loss"


def test_boundary_mask_requires_bos_id():
    with pytest.raises(ValueError, match="bos_token_id"):
        _mtp_model(mtp_doc_boundary_mask=True, bos_token_id=-1)


def test_mtp_loss_none_in_eval():
    m = _mtp_model()
    m.eval()
    with torch.no_grad():
        m(torch.randint(2, 256, (1, 8)))
    assert m._last_mtp_loss is None
