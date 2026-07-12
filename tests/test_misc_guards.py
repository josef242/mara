"""Small guards and formula pins from the 2026-07-13 polish pass."""
import pytest
import torch

from helpers import tiny_model


def test_scaffold_mode_without_targets_refuses():
    """scaffold_mode skips the final norm (deepest aux head owns the readout);
    the eval branch used to project the UN-normalized stream through the head
    and return silently-garbage logits (audit 2026-07-11 LOW, fixed)."""
    model = tiny_model()
    model.eval()
    with torch.no_grad(), pytest.raises(ValueError, match="scaffold_mode"):
        model(torch.randint(0, 256, (1, 8)), scaffold_mode=True)


def test_adafactor_rms_plain_tensor_formula():
    """_rms == true RMS for plain tensors (the DTensor world_size double-count
    fixed 2026-07-13 shares this code path; multi-rank parity needs a mesh and
    is rig/dist-tier)."""
    from adafactor_fsdp2 import AdafactorFSDP2
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdafactorFSDP2([p])
    t = torch.randn(64, 3)
    assert torch.allclose(opt._rms(t), t.pow(2).mean().sqrt())


def test_adamc16bit_wrapper_exposes_state_dict():
    """The AdamC wrappers aren't torch Optimizer subclasses; state_dict /
    load_state_dict now delegate to _base_optimizer (2026-07-13) so external
    tooling doesn't need the unwrap."""
    adamw_16bit = pytest.importorskip("adamw_16bit", exc_type=ImportError)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = adamw_16bit.AdamC16bit([{"params": [p], "weight_decay": 0.01,
                                   "is_normalized": True}], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()  # used to AttributeError
    assert sd["state"], "empty optimizer state after a step"
    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = adamw_16bit.AdamC16bit([{"params": [p2], "weight_decay": 0.01,
                                    "is_normalized": True}], lr=1e-3)
    opt2.load_state_dict(sd)
    st = opt2.state[p2]
    assert st["exp_avg"].dtype == torch.float16  # 16-bit re-cast rides along
