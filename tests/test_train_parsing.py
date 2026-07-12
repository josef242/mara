"""Pure-function guts of train_mara.py: lr_mods/wd-rules parsing, schedule
interpolation, MTP lambda, resume guards. Rig tier — train_mara needs
torch>=2.6 to import (skips on the local box).

Bug pins from the 2026-07-11 audit ride along as xfail(strict).
"""
import pytest
import torch

train_mara = pytest.importorskip("train_mara", reason="needs torch>=2.6 (fully_shard export)",
                                 exc_type=ImportError)


class StubModel:
    """Just enough model for the name-driven parsers."""
    NAMES = [
        "tok_embeddings.weight",
        "output.weight",
        "norm.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention_norm.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.1.attention.wq.weight",
        "layers.1.feed_forward.w1.weight",
        "layers.2.gdn_attn.q_proj.weight",   # GDN hybrid names
        "layers.2.gdn_attn.o_norm.weight",   # GDN norm — must stay excluded
        "layers.2.moe.experts.w1",           # MoE 3D expert (no .weight suffix)
        "layers.2.moe.shared_experts.w1.weight",
        "layers.2.moe.router.gate.weight",
        "mtp.proj.weight",
    ]

    def __init__(self, names=None):
        self._params = [(n, torch.nn.Parameter(torch.zeros(2, 2)))
                        for n in (names or self.NAMES)]

    def named_parameters(self):
        return iter(self._params)

    def by_name(self, n):
        return dict(self._params)[n]


# ── interpolate_lr_mod ──

def test_interp_clamps_and_midpoint():
    f = train_mara.interpolate_lr_mod
    sched = [[100, 1.0], [200, 3.0]]
    assert f(sched, 0) == 1.0
    assert f(sched, 100) == 1.0
    assert f(sched, 150) == 2.0
    assert f(sched, 200) == 3.0
    assert f(sched, 9999) == 3.0


def test_unsorted_and_duplicate_schedules_rejected_at_parse(monkeypatch):
    """AUDIT 2026-07-11 finding #11, fixed same day: schedules were never
    validated — unsorted input silently clamped to the wrong knot, duplicate
    steps ZeroDivisionError'd mid-run. Now fatal at parse time."""
    monkeypatch.setattr(train_mara, "fatal_error",
                        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)))
    m = StubModel()
    with pytest.raises(RuntimeError, match="strictly ascending"):
        train_mara.parse_lr_mods([["all", "all", [[1000, 0.1], [0, 1.0]]]], m)
    with pytest.raises(RuntimeError, match="strictly ascending"):
        train_mara.parse_wd_rules(
            [["all", [[500, 0.1], [500, 0.2]]], ["emb", 0.0], ["out", 0.0]], m)
    # scalar wd values and well-formed schedules still parse
    train_mara.parse_wd_rules(
        [["all", [[0, 0.1], [500, 0.2]]], ["emb", 0.0], ["out", 0.0]], m)


# ── get_mtp_lambda ──

def _mtp_settings(**over):
    d = {"mtp_enabled": True, "mtp_loss_weight": 0.3,
         "mtp_loss_weight_schedule": None}
    d.update(over)
    return type("S", (), d)()


def test_mtp_lambda_disabled_and_constant():
    f = train_mara.get_mtp_lambda
    assert f(100, 0, _mtp_settings(mtp_enabled=False)) == 0.0
    assert f(100, 0, _mtp_settings()) == pytest.approx(0.3)


def test_mtp_lambda_step_schedule_boundary_exact():
    """Step-hold semantics: the NEW value applies exactly AT the breakpoint —
    the DeepSeek-V3 0.3 -> 0.1 token-keyed step-down."""
    f = train_mara.get_mtp_lambda
    s = _mtp_settings(mtp_loss_weight_schedule={
        "shape": "step", "key": "tokens",
        "points": [[0, 0.3], [10_000_000_000, 0.1]]})
    assert f(0, 9_999_999_999, s) == pytest.approx(0.3)
    assert f(0, 10_000_000_000, s) == pytest.approx(0.1)
    assert f(0, 20_000_000_000, s) == pytest.approx(0.1)


def test_mtp_lambda_linear_midpoint_and_unsorted_points():
    f = train_mara.get_mtp_lambda
    s = _mtp_settings(mtp_loss_weight_schedule={
        "shape": "linear", "key": "step",
        "points": [[200, 0.1], [100, 0.3]]})  # deliberately unsorted: it sorts
    assert f(150, 0, s) == pytest.approx(0.2)


# ── parse_wd_rules ──

def test_wd_rules_full_mapping_last_match_wins(monkeypatch):
    monkeypatch.setattr(train_mara, "fatal_error",
                        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)))
    m = StubModel()
    rules = [["all", 0.1], ["emb", 0.0], ["out", 0.05], [0, 0, 0.2]]
    out = {id(p): v for p, v in train_mara.parse_wd_rules(rules, m)}
    assert out[id(m.by_name("tok_embeddings.weight"))] == 0.0
    assert out[id(m.by_name("output.weight"))] == 0.05
    assert out[id(m.by_name("layers.0.attention.wq.weight"))] == 0.2  # range rule wins (last)
    assert out[id(m.by_name("layers.1.attention.wq.weight"))] == 0.1
    assert out[id(m.by_name("mtp.proj.weight"))] == 0.1              # mtp rides 'all'
    # WD 'all' covers GDN and MoE names (contrast with lr_mods below):
    assert out[id(m.by_name("layers.2.gdn_attn.q_proj.weight"))] == 0.1
    assert out[id(m.by_name("layers.2.moe.experts.w1"))] == 0.1
    # norms excluded
    assert id(m.by_name("layers.0.attention_norm.weight")) not in out
    assert id(m.by_name("norm.weight")) not in out


def test_wd_rules_coverage_guard_is_fatal(monkeypatch):
    monkeypatch.setattr(train_mara, "fatal_error",
                        lambda msg: (_ for _ in ()).throw(RuntimeError(msg)))
    with pytest.raises(RuntimeError, match="uncovered"):
        train_mara.parse_wd_rules([["all", 0.1]], StubModel())  # emb+out uncovered


# ── parse_lr_mods ──

def test_lr_mods_last_match_wins_and_range_inclusive():
    m = StubModel()
    sched1, sched2 = [[0, 1.0]], [[0, 2.0]]
    out = {id(p): s for p, s in train_mara.parse_lr_mods(
        [["all", "all", sched1], [0, 0, "ffn", sched2]], m)}
    assert out[id(m.by_name("layers.0.feed_forward.w1.weight"))] is sched2
    assert out[id(m.by_name("layers.0.attention.wq.weight"))] is sched1
    assert out[id(m.by_name("layers.1.feed_forward.w1.weight"))] is sched1


def test_lr_mods_all_covers_gdn_and_moe_params():
    """AUDIT 2026-07-11 finding #10 (MEDIUM), fixed same day: _match_type knew
    only 'attention.'/'feed_forward.', so gdn_attn/moe params escaped lr_mods
    even under 'all' (75% of attention layers on a 3:1 GDN hybrid)."""
    m = StubModel()
    out = {id(p) for p, _ in train_mara.parse_lr_mods([["all", "all", [[0, 1.0]]]], m)}
    for name in ("layers.2.gdn_attn.q_proj.weight",
                 "layers.2.moe.experts.w1",
                 "layers.2.moe.shared_experts.w1.weight",
                 "layers.2.moe.router.gate.weight"):
        assert id(m.by_name(name)) in out, f"{name} escapes lr_mods 'all'"
    # norms stay excluded, GDN or otherwise
    assert id(m.by_name("layers.2.gdn_attn.o_norm.weight")) not in out
    assert id(m.by_name("layers.0.attention_norm.weight")) not in out
    # attn/ffn split respects the new names too
    attn = {id(p) for p, _ in train_mara.parse_lr_mods([["all", "attn", [[0, 1.0]]]], m)}
    assert id(m.by_name("layers.2.gdn_attn.q_proj.weight")) in attn
    assert id(m.by_name("layers.2.moe.experts.w1")) not in attn


# ── _apply_scalar_wd_to_groups (resume WD refresh) ──

def test_resume_wd_update_reaches_all_groups():
    """AUDIT 2026-07-11 finding #5 (HIGH), fixed same day: the resume-time
    scalar WD update broke after the first differing group, leaving the rest
    on the checkpoint's old value while logging success."""
    opt = type("O", (), {})()
    opt.param_groups = [
        {"wd_group": "muon", "weight_decay": 0.01},
        {"wd_group": "adam_default", "weight_decay": 0.01},
        {"wd_group": "emb", "weight_decay": 0.01},
        {"wd_group": "norm_bias", "weight_decay": 0.0},
    ]
    updated = train_mara._apply_scalar_wd_to_groups(opt, 0.02)
    assert updated == {0.01: 3}
    assert [pg["weight_decay"] for pg in opt.param_groups] == [0.02, 0.02, 0.02, 0.0]
    # idempotent second call: nothing to update
    assert train_mara._apply_scalar_wd_to_groups(opt, 0.02) == {}


# ── _resume_config_mismatches ──

def test_resume_guard_catches_behavioral_flips():
    f = train_mara._resume_config_mismatches
    settings = type("S", (), {
        "doc_attn_mask_enabled": True, "doc_pos_reset": True,
        "doc_bos_token_id": 1, "swa_enabled": True, "swa_window": 1024,
        "swa_global_interleave": 4, "mtp_enabled": False})()
    ckpt = {"doc_attn_mask": True, "doc_pos_reset": True, "bos_token_id": 1,
            "swa_enabled": True, "swa_window": 512, "swa_global_interleave": 4,
            "mtp_enabled": False}
    assert f(ckpt, settings) == [("swa.window", 512, 1024)]
    ckpt["swa_window"] = 1024
    assert f(ckpt, settings) == []
    del ckpt["swa_window"]  # pre-field checkpoint: skipped, not flagged
    assert f(ckpt, settings) == []


def test_resume_guard_allows_mtp_boundary_mask_flip():
    """Ruled 2026-07-13: mtp.doc_boundary_mask is a loss-SHAPING knob (like
    the schedulable lambda), sanctioned to flip mid-run at a checkpoint
    boundary — it must NOT be in _RESUME_MUST_MATCH."""
    guarded = {ck for ck, _a, _l in train_mara._RESUME_MUST_MATCH}
    assert "mtp_doc_boundary_mask" not in guarded
    settings = type("S", (), {"mtp_enabled": True, "mtp_doc_boundary_mask": True})()
    ckpt = {"mtp_enabled": True, "mtp_doc_boundary_mask": False}
    assert train_mara._resume_config_mismatches(ckpt, settings) == []


def test_reassert_group_hparams_routes_by_family():
    """POLISH 2026-07-13: resume used to re-assert betas onto ALL groups
    (polluting Muon groups) and never re-asserted muon_momentum at all."""
    opt = type("O", (), {})()
    opt.param_groups = [
        {"use_muon": True, "momentum": 0.90, "beta2": 0.95, "betas": (0.9, 0.999)},
        {"use_muon": False, "betas": (0.8, 0.9)},
        {"betas": (0.8, 0.9)},  # standalone Adam-family group (no use_muon key)
    ]
    settings = type("S", (), {"muon_momentum": 0.95, "normuon_beta2": 0.99,
                              "beta1": 0.9, "beta2": 0.95})()
    train_mara._reassert_group_hparams(opt, settings)
    muon_pg, adam_pg, plain_pg = opt.param_groups
    assert muon_pg["momentum"] == 0.95
    assert muon_pg["beta2"] == 0.99
    assert "betas" not in muon_pg          # pollution stripped
    assert adam_pg["betas"] == (0.9, 0.95)
    assert plain_pg["betas"] == (0.9, 0.95)


def test_resume_guard_attrs_exist_in_settings_source():
    """Typo guard: every attr in _RESUME_MUST_MATCH must be assigned somewhere
    in Settings, or the guard compares against None forever."""
    src = open(train_mara.__file__, encoding="utf-8").read()
    for _ck, attr, _label in train_mara._RESUME_MUST_MATCH:
        assert f"self.{attr}" in src, f"_RESUME_MUST_MATCH names unknown attr {attr}"
