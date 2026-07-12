"""Shared test config for the mara_fsdp2 + common_fsdp2 suite.

Path model: this repo (mara_fsdp2) and its sibling common_fsdp2 both hold
importable modules; train_mara.py itself inserts '../common_fsdp2' relative to
CWD, so we insert ABSOLUTE paths here to make tests runnable from any cwd.

Environment tiers (the suite runs in all of them, skipping what it can't):
  - local Windows box: torch 2.9.1 cu128 + torchao (rig parity since
    2026-07-13), 1x consumer GPU; still no triton (flex JIT) or fla
  - rigs (rig-30/rig-31): torch 2.9.x, 8x GPU, full deps (fla: rig-30 only)
Use `pytest.importorskip(..., exc_type=ImportError)` for env-dependent module
imports, and the `gpu` / `dist` markers for hardware-dependent tests.
"""
import os
import sys
from pathlib import Path

import pytest

# CPU-portable CCE: the fused 'cce' Triton kernel can't run on CPU tensors, so
# on the rigs (where cut_cross_entropy imports) any CPU training-forward test
# would crash inside the loss. WD_CCE_IMPL is cce_loss's call-time override;
# torch_compile is the pure-PyTorch impl — same math, any device. GPU-marked
# tests that specifically exercise the fused kernel can monkeypatch it back.
os.environ.setdefault("WD_CCE_IMPL", "torch_compile")

TESTS_DIR = Path(__file__).resolve().parent
MARA_ROOT = TESTS_DIR.parent
COMMON_ROOT = MARA_ROOT.parent / "common_fsdp2"

for _p in (str(MARA_ROOT), str(COMMON_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    run_dist = os.environ.get("RUN_DIST", "0") == "1"
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_dist = pytest.mark.skip(reason="set RUN_DIST=1 (and use torchrun where noted) for dist tests")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "dist" in item.keywords and not run_dist:
            item.add_marker(skip_dist)


@pytest.fixture(scope="session")
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
