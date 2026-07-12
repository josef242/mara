"""MoE invariants: padded-BMM vs loop parity, scatter/gather, router contract,
capacity-drop bookkeeping, and the counter-hygiene bugs found in the 2026-07
audit (xfail-strict until fixed — they flip loudly when the fix lands).
"""
import math

import pytest
import torch

from model_v2 import (GroupedExperts, TokenChoiceTopKRouter, MoE, ModelArgs,
                      Transformer, _scatter_to_padded, _gather_from_padded)

E, D, H, TOPK = 4, 16, 32, 2


def _moe_args(**over):
    base = dict(dim=D, n_layers=2, n_heads=2, vocab_size=64, inner_dim=H,
                moe_enabled=True, moe_num_experts=E, moe_top_k=TOPK,
                moe_num_shared_experts=0, moe_score_func="sigmoid",
                moe_route_norm=True, moe_load_balance_coeff=None,
                moe_capacity_factor=0.0)
    base.update(over)
    return ModelArgs(**base)


def _fresh_moe(**over):
    """A directly-constructed MoE has RAW torch.empty params — weight init
    lives in Transformer._init_weights, not in the module. Uninitialized pool
    memory is finite on some platforms and NaN on others (bit us on rig-31:
    torch 2.9 handed back real NaNs), so init explicitly."""
    moe = MoE(_moe_args(**over))
    with torch.no_grad():
        for p in moe.parameters():
            torch.nn.init.normal_(p, std=0.05)
    return moe


def test_bmm_equals_loop_path():
    """The compiled padded-BMM training path and the forward_loop eval path
    must be the same function, including zero-count and full-capacity experts."""
    torch.manual_seed(0)
    ge = GroupedExperts(D, H, E).double()
    for p in ge.parameters():
        torch.nn.init.normal_(p, std=0.05)
    counts, cap = [3, 0, 2, 1], 3
    x = torch.randn(sum(counts), D, dtype=torch.float64)
    out_loop = ge(x, torch.tensor(counts))
    out_bmm = _gather_from_padded(ge(_scatter_to_padded(x, counts, E, cap)), counts)
    assert (out_loop[:sum(counts)] - out_bmm).abs().max().item() < 1e-12


def test_scatter_gather_roundtrip_values_and_grads():
    torch.manual_seed(1)
    counts, cap = [3, 0, 2, 1], 3
    x = torch.randn(sum(counts), D, requires_grad=True)
    rt = _gather_from_padded(_scatter_to_padded(x, counts, E, cap), counts)
    assert torch.equal(rt, x.detach())
    g = torch.randn_like(rt)
    rt.backward(g)
    assert torch.equal(x.grad, g)


def test_router_combine_weights_ignore_expert_bias():
    """Aux-loss-free contract (DeepSeek): expert_bias steers SELECTION only;
    the combine weights must be the raw unbiased scores."""
    torch.manual_seed(2)
    r = TokenChoiceTopKRouter(D, E, TOPK, "sigmoid", route_norm=False, route_scale=1.0)
    x = torch.randn(6, D)
    bias = torch.tensor([0.0, 5.0, 0.0, 0.0])
    scores, sel, _, _ = r(x, bias)
    raw = torch.sigmoid(r.gate(x).float())
    assert torch.allclose(scores, raw.gather(1, sel))
    assert (sel == 1).sum().item() == 6  # the +5 bias forces expert 1 into everyone's top-k
    _, sel_nobias, _, _ = r(x, None)
    assert not torch.equal(sel, sel_nobias)  # ...and selection did respond to the bias


def test_router_counts_match_bincount():
    torch.manual_seed(3)
    r = TokenChoiceTopKRouter(D, E, TOPK, "sigmoid", route_norm=False, route_scale=1.0)
    _, sel, num_tpe, _ = r(torch.randn(37, D), None)
    expect = torch.bincount(sel.reshape(-1), minlength=E)
    assert torch.equal(num_tpe.long(), expect)


@pytest.mark.parametrize("score_before", [True, False],
                         ids=["score_before_experts", "score_after_experts"])
def test_moe_matches_per_token_reference(score_before):
    """Brute-force per-token reference, both score placements:
      before: MoE(x)_i == sum_k expert_{sel_ik}(score_ik * x_i)  (default)
      after:  MoE(x)_i == sum_k score_ik * expert_{sel_ik}(x_i)
    Score-before matters: the expert is nonlinear, so scaling the input is a
    different function than scaling the output."""
    torch.manual_seed(4)
    moe = _fresh_moe(moe_score_before_experts=score_before).double()
    moe.eval()  # loop path; parity with bmm pinned separately above
    x = torch.randn(1, 6, D, dtype=torch.float64)
    out = moe(x)

    scores, sel, _, _ = moe.router(x.view(-1, D), moe.expert_bias)
    scores = scores.double()
    w1, w2, w3 = moe.experts.w1, moe.experts.w2, moe.experts.w3

    def expert_fn(e, t):
        # 3D expert weights are stored (out_features, in_features) per expert,
        # i.e. F.linear layout.
        lin = torch.nn.functional.linear
        h = torch.nn.functional.silu(lin(t, w1[e])) * lin(t, w3[e])
        return lin(h, w2[e])

    ref = torch.zeros(6, D, dtype=torch.float64)
    for i in range(6):
        xi = x.view(-1, D)[i]
        for k in range(TOPK):
            e = sel[i, k].item()
            if score_before:
                ref[i] += expert_fn(e, scores[i, k] * xi)
            else:
                ref[i] += scores[i, k] * expert_fn(e, xi)
    # BOTH placements round-trip through .float() inside MoE (scaled input at
    # dispatch / bmm combine), so fp64 agreement is bounded by fp32 epsilon.
    assert (out.view(-1, D) - ref).abs().max().item() < 1e-6


def test_capacity_drop_conservation():
    """With a skewed router and tight capacity: kept slots per expert <= capacity
    and the layer still produces finite output."""
    torch.manual_seed(5)
    moe = _fresh_moe(moe_capacity_factor=0.5)
    moe.train()
    with torch.no_grad():
        moe.router.gate.weight.zero_()
        moe.router.gate.weight[0].fill_(1.0)
        moe.router.gate.weight[1].fill_(0.5)
    x = torch.randn(1, 6, D)
    out = moe(x)
    assert torch.isfinite(out).all()
    n, cap = 6 * TOPK, max(1, math.ceil(0.5 * 6 * TOPK / E))
    assert moe._tokens_dropped_accum > 0
    assert moe._tokens_dropped_accum <= n - cap  # can't drop more than overflow


def test_capacity_drop_leaves_untouched_tokens_untouched():
    """RULED 2026-07-13 (reference research: Switch/GShard/DeepSeek-V2 never
    rescale survivors; V3 drops nothing): a token whose slots all survive must
    produce output IDENTICAL to a dropless run — the old batch-global rescale
    inflated its combine weights because OTHER tokens overflowed."""
    torch.manual_seed(11)
    # Router steered by input direction: tokens 0-4 pile onto experts 0/1
    # (over capacity -> drops), token 5 routes to uncontested experts 2/3.
    moe = _fresh_moe(moe_capacity_factor=0.5)
    dirs = torch.eye(E, D)  # expert e keyed to input direction e
    with torch.no_grad():
        moe.router.gate.weight.copy_(dirs * 4.0)
    x = torch.zeros(1, 6, D)
    x[0, :5, 0] = 1.0
    x[0, :5, 1] = 0.8
    x[0, :5, 2:] = 0.05 * torch.randn(5, D - 2)
    x[0, 5, 2] = 1.0
    x[0, 5, 3] = 0.8

    moe.train()
    out_dropping = moe(x)
    assert moe._tokens_dropped_accum > 0, "setup produced no drops — test vacuous"

    moe_free = _fresh_moe(moe_capacity_factor=0.0)  # dropless
    with torch.no_grad():
        for pf, pd in zip(moe_free.parameters(), moe.parameters()):
            pf.copy_(pd)
        moe_free.router.gate.weight.copy_(moe.router.gate.weight)
    moe_free.train()
    out_free = moe_free(x)

    diff = (out_dropping[0, 5] - out_free[0, 5]).abs().max().item()
    assert diff < 1e-5, f"undropped token's output changed by {diff} under others' drops"
    # ...while the overflowing tokens genuinely lost contributions
    assert (out_dropping[0, :5] - out_free[0, :5]).abs().max().item() > 1e-4


# ── Counter hygiene (AUDIT 2026-07-11 #13, fixed 2026-07-13: side effects are
#    training-gated AND recompute-gated via _in_backward_recompute) ──

def test_tokens_per_expert_not_accumulated_in_eval():
    torch.manual_seed(6)
    moe = _fresh_moe(moe_load_balance_coeff=1e-3)
    moe.eval()
    moe.tokens_per_expert.zero_()
    moe(torch.randn(1, 6, D))
    assert moe.tokens_per_expert.sum().item() == 0.0


def test_tokens_per_expert_counts_once_under_activation_checkpointing():
    """The AC recompute re-executes MoE.forward inside loss.backward(); the
    balance counter must count the step exactly once (used to double)."""
    torch.manual_seed(7)
    args = _moe_args(max_seq_len=16, moe_load_balance_coeff=1e-3,
                     use_activation_checkpointing=True, tie_word_embeddings=True)
    tm = Transformer(args)
    tm.train()
    moe_layer = tm.layers[0].moe
    moe_layer.tokens_per_expert.zero_()
    toks = torch.randint(0, 64, (2, 8))
    _, loss = tm(toks, targets=torch.randint(1, 64, (2, 8)))
    loss.backward()
    assert moe_layer.tokens_per_expert.sum().item() == 2 * 8 * TOPK


def test_tokens_per_expert_counts_once_without_activation_checkpointing():
    """Companion boundary: the recompute-gate must NOT suppress counting on a
    plain (non-AC) training forward+backward."""
    torch.manual_seed(8)
    args = _moe_args(max_seq_len=16, moe_load_balance_coeff=1e-3,
                     use_activation_checkpointing=False, tie_word_embeddings=True)
    tm = Transformer(args)
    tm.train()
    moe_layer = tm.layers[0].moe
    moe_layer.tokens_per_expert.zero_()
    toks = torch.randint(0, 64, (2, 8))
    _, loss = tm(toks, targets=torch.randint(1, 64, (2, 8)))
    loss.backward()
    assert moe_layer.tokens_per_expert.sum().item() == 2 * 8 * TOPK


def test_expert_init_scale_matches_dense_convention():
    """AUDIT 2026-07-11 #14, ruled unintentional + fixed 2026-07-13: routed
    experts had BOTH w2 and w3 depth-scaled (torchtitan convention) while this
    codebase's dense FFN scales only w3 — leaving experts ~sqrt(2*n_layers)x
    weaker than the shared expert in the same layer. Pin: experts follow the
    DENSE convention (w1,w2 @ 0.02; w3 @ 0.02/sqrt(2*n_layers))."""
    import math
    torch.manual_seed(9)
    args = _moe_args(n_layers=8, moe_num_shared_experts=1, tie_word_embeddings=True)
    tm = Transformer(args)
    moe = next(l.moe for l in tm.layers if l.moe_enabled)
    output_std = 0.02 / math.sqrt(2 * 8)
    assert abs(moe.experts.w1.std().item() - 0.02) < 0.004
    assert abs(moe.experts.w2.std().item() - 0.02) < 0.004
    assert abs(moe.experts.w3.std().item() - output_std) < 0.002
    # shared expert (dense FeedForward) agrees
    assert abs(moe.shared_experts.w2.weight.std().item() - 0.02) < 0.004
    assert abs(moe.shared_experts.w3.weight.std().item() - output_std) < 0.002
