"""Doc-mask helper pure functions: doc_ids_from_tokens / doc_position_ids.

These two functions define what "document" means for the doc-attention-mask
and position-reset features — an off-by-one here silently changes what every
doc-mask run trains. Semantics pinned (from the docstrings, verified against
the flex mask_mod usage):
  - BOS starts a new document and belongs to the doc it opens.
  - Tokens before a window's first BOS are the tail of a cut document, doc 0,
    and keep window-relative positions.
"""
import torch

from model_v2 import doc_ids_from_tokens, doc_position_ids

BOS = 1


def test_doc_ids_basic():
    #        cut-tail | doc1        | doc2
    toks = torch.tensor([[5, 6, BOS, 7, 8, BOS, 9]])
    ids = doc_ids_from_tokens(toks, BOS)
    assert ids.tolist() == [[0, 0, 1, 1, 1, 2, 2]]
    assert ids.dtype == torch.int32  # int => immune to FSDP2 mp_policy bf16 cast


def test_doc_position_ids_reset_at_bos():
    toks = torch.tensor([[5, 6, BOS, 7, 8, BOS, 9]])
    pos = doc_position_ids(toks, BOS)
    assert pos.tolist() == [[0, 1, 0, 1, 2, 0, 1]]
    assert pos.dtype == torch.int64


def test_no_bos_window_is_identity_positions():
    """A window with no BOS must reproduce plain window-relative positions —
    the invariant that makes the no-BOS SDPA fast path exact."""
    toks = torch.randint(2, 100, (3, 16))  # no 1s
    pos = doc_position_ids(toks, BOS)
    expect = torch.arange(16).expand(3, 16)
    assert torch.equal(pos, expect)
    assert torch.equal(doc_ids_from_tokens(toks, BOS), torch.zeros(3, 16, dtype=torch.int32))


def test_bos_at_window_start_and_adjacent_bos():
    toks = torch.tensor([[BOS, 5, BOS, BOS, 6]])
    ids = doc_ids_from_tokens(toks, BOS)
    pos = doc_position_ids(toks, BOS)
    assert ids.tolist() == [[1, 1, 2, 3, 3]]
    assert pos.tolist() == [[0, 1, 0, 0, 1]]


def test_batch_rows_independent():
    a = torch.tensor([5, BOS, 6, 7])
    b = torch.tensor([8, 9, 10, BOS])
    batched = torch.stack([a, b])
    for row, single in [(0, a), (1, b)]:
        assert torch.equal(doc_position_ids(batched, BOS)[row],
                           doc_position_ids(single.unsqueeze(0), BOS)[0])
