"""Environment probe: every core module imports, torch is functional.

If this file fails, nothing else in the suite is meaningful — fix the env first.
"""
import importlib

import pytest

CORE_MODULES = [
    "model_v2",
    "muon_fsdp2",
    "adamc_optimizer",
    "row_center",
    "adafactor_fsdp2",
    "body_lr_controller",
    "dataloader",
    "tokenizer_abstraction",
    "logger",
]

OPTIONAL_MODULES = {
    "adamw_16bit": "torchao",       # top-level torchao import; absent on local box
    "train_mara": "torch>=2.6",     # needs torch.distributed.fsdp.fully_shard export
}


@pytest.mark.parametrize("mod", CORE_MODULES)
def test_core_module_imports(mod):
    importlib.import_module(mod)


@pytest.mark.parametrize("mod,needs", OPTIONAL_MODULES.items())
def test_optional_module_imports(mod, needs):
    try:
        importlib.import_module(mod)
    except ImportError as e:
        pytest.skip(f"{mod} unavailable in this env (needs {needs}): {e}")


def test_torch_basic():
    import torch
    x = torch.randn(4, 4)
    assert torch.allclose(x @ torch.eye(4), x)
