"""AdamC / AdamW16bit contracts, incl. the 2026-07-11 audit findings on the
16-bit resume path (pinned xfail-strict).

adamw_16bit imports torchao at module level -> those tests skip on the local
box and run on the rigs.
"""
import pytest
import torch

from adamc_optimizer import AdamC


def _toy(seed=0):
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(8, 4))
    return p


def _step(opt, p, grad):
    p.grad = grad.clone()
    opt.step()
    opt.zero_grad()


def test_adamc_decay_scales_linearly_with_lr_ratio():
    """AdamC (Defazio): per-step decay factor = wd * lr_t^2 / max_lr, i.e.
    AdamW's lr*wd scaled by (lr_t/max_lr). At lr==max_lr it equals AdamW; at
    half lr the decay is quarter (quadratic in lr_t at fixed max_lr)."""
    wd, max_lr = 0.1, 1e-2
    decays = {}
    for lr_frac in (1.0, 0.5):
        p = _toy()
        before = p.detach().clone()
        opt = AdamC([{"params": [p], "weight_decay": wd, "is_normalized": True}],
                    lr=max_lr, betas=(0.9, 0.95))
        for g in opt.param_groups:
            g["lr"] = max_lr * lr_frac
        _step(opt, p, torch.zeros_like(p))  # zero grad isolates the decay term
        decays[lr_frac] = ((before - p.detach()) / before).mean().item()
    ratio = decays[1.0] / decays[0.5]
    assert abs(ratio - 4.0) < 0.1, f"decay ratio {ratio}, expected 4 (quadratic)"
    assert abs(decays[1.0] - wd * max_lr) / (wd * max_lr) < 1e-3  # == AdamW at peak


def test_adamc_max_lr_is_constructor_peak_not_current():
    p = _toy()
    opt = AdamC([{"params": [p], "weight_decay": 0.1, "is_normalized": True}],
                lr=1e-2, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["lr"] = 1e-3  # warmup-style current lr
    assert opt.max_lr == 1e-2  # denominator must stay the schedule peak


def test_adamc_honors_wd_overrides():
    grads = torch.randn(8, 4)
    results = {}
    for override in (None, 0.0):
        p = _toy(seed=3)
        opt = AdamC([{"params": [p], "weight_decay": 0.1, "is_normalized": True}],
                    lr=1e-2, betas=(0.9, 0.95))
        opt.wd_overrides = {} if override is None else {id(p): override}
        _step(opt, p, grads)
        results[override] = p.detach().clone()
    assert not torch.equal(results[None], results[0.0]), \
        "wd_overrides={id: 0.0} did not change the step -> side-dict not read"


# ── 16-bit family (rig tier: needs torchao importable for the module) ──

adamw_16bit = pytest.importorskip("adamw_16bit", reason="module imports torchao at top level",
                                  exc_type=ImportError)


def test_adamw16bit_tracks_torch_adamw():
    """Trajectory parity with torch.optim.AdamW on a toy problem — the 16-bit
    state quantization should cost noise, not bias."""
    torch.manual_seed(4)
    g_seq = [torch.randn(16, 8) for _ in range(50)]
    p_ref = torch.nn.Parameter(torch.randn(16, 8))
    p_16 = torch.nn.Parameter(p_ref.detach().clone())
    ref = torch.optim.AdamW([p_ref], lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    o16 = adamw_16bit.AdamW16bit([p_16], lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    for g in g_seq:
        for p, o in ((p_ref, ref), (p_16, o16)):
            p.grad = g.clone()
            o.step()
            o.zero_grad()
    rel = (p_ref - p_16).norm().item() / p_ref.norm().item()
    assert rel < 2e-2, f"16-bit trajectory diverged {rel:.4f} from AdamW"


def test_sr_bf16_unbiased_and_exact_on_representable():
    sr = adamw_16bit._fp32_to_bf16_sr
    torch.manual_seed(5)
    # exact on representable
    x = torch.randn(1000).bfloat16().float()
    assert torch.equal(sr(x).float(), x)
    # unbiased strictly between neighbors
    lo = torch.tensor(1.0)
    hi = lo.bfloat16().float() * (1 + 2 ** -8)  # strictly between bf16 grid points
    draws = torch.stack([sr(hi.expand(4096).contiguous()).float().mean()
                         for _ in range(20)])
    err = abs(draws.mean().item() - hi.item()) / (hi.item() - 1.0)
    assert err < 0.15, f"SR bias {err:.3f} of the ULP gap"


def test_16bit_state_dtypes_survive_state_dict_roundtrip():
    """AUDIT 2026-07-11 finding #7 (HIGH), fixed same day: the stock
    load_state_dict promoted 16-bit states to fp32 (param dtype) on every
    resume; AdamW16bit now re-casts in a load_state_dict override."""
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = adamw_16bit.AdamW16bit([p], lr=1e-3, state_dtype="mixed")
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()
    opt2 = adamw_16bit.AdamW16bit([p], lr=1e-3, state_dtype="mixed")
    opt2.load_state_dict(sd)
    st = opt2.state[p]
    assert st["exp_avg"].dtype == torch.float16
    assert st["exp_avg_sq"].dtype == torch.bfloat16


def test_muon_embedded_16bit_state_dtypes_survive_roundtrip():
    """AUDIT 2026-07-11 #7-sibling, fixed 2026-07-13: MuonFSDP2's embedded
    Adam states (adam_state_dtype='mixed') suffered the same load-time fp32
    promotion as AdamW16bit; Muon now re-casts in a load_state_dict override.
    Only matters for muon_adam_state_dtype != fp32 configs (keel-moe-mega)."""
    import muon_fsdp2
    torch.manual_seed(6)

    def make():
        p = torch.nn.Parameter(torch.randn(16, 8))
        opt = muon_fsdp2.Muon([dict(params=[p], use_muon=False, lr=1e-3)],
                              adam_state_dtype="mixed")
        return p, opt

    p1, opt1 = make()
    p1.grad = torch.randn_like(p1)
    opt1.step()
    st = opt1.state[p1]
    assert st["exp_avg"].dtype == torch.float16          # fresh-run baseline
    assert st["exp_avg_sq"].dtype == torch.bfloat16

    p2, opt2 = make()
    opt2.load_state_dict(opt1.state_dict())
    st2 = opt2.state[p2]
    assert st2["exp_avg"].dtype == torch.float16
    assert st2["exp_avg_sq"].dtype == torch.bfloat16
    # values must round-trip exactly (they were produced in these dtypes)
    assert torch.equal(st2["exp_avg"].float(), st["exp_avg"].float())
    assert torch.equal(st2["exp_avg_sq"].float(), st["exp_avg_sq"].float())


def test_muon_16bit_refuses_head_gauge_and_cwd():
    """POLISH 2026-07-13: the 16-bit embedded Adam path implements neither
    head-gauge projection nor cautious WD; the fence used to live only in
    train_mara's boot fatals (consolidate_optimizer bypassed it). Now the
    optimizer itself refuses."""
    import muon_fsdp2
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = muon_fsdp2.Muon([dict(params=[p], use_muon=False, lr=1e-3)],
                          adam_state_dtype="mixed")
    p.grad = torch.randn_like(p)
    opt.head_gauge_ids = {id(p)}
    with pytest.raises(RuntimeError, match="head_gauge"):
        opt.step()
    opt.head_gauge_ids = set()
    opt.param_groups[0]["cautious_weight_decay"] = True
    with pytest.raises(RuntimeError, match="cautious_weight_decay"):
        opt.step()
    opt.param_groups[0]["cautious_weight_decay"] = False
    opt.step()  # clean config steps fine


def test_16bit_survives_dcp_style_fake_init_step():
    """AUDIT 2026-07-11 finding #6 — DOWNGRADED after Josef's field report:
    torch >= 2.9's _init_optim_state preserves tensor lr, so real resumes on
    the fleet never hit the old raise (the crash chain was torch <= 2.5 DCP,
    which set a bare float 0.0). This pins the general robustness contract
    instead: a float lr in the group — from old-torch DCP, tooling, or a
    stray manual write — must not crash and must not move params at lr 0."""
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = adamw_16bit.AdamW16bit([p], lr=1e-3)
    before = p.detach().clone()
    # emulate torch.distributed.checkpoint._init_optim_state
    for group in opt.param_groups:
        group["lr"] = 0.0  # float, not tensor — this is what DCP does
    p.grad = torch.zeros_like(p)
    opt.step()  # must not raise, must not move params
    assert torch.equal(p.detach(), before)
