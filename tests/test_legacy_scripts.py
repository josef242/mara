"""Subprocess wrappers around the pre-suite standalone test scripts.

These scripts predate the pytest suite; each self-checks and exits nonzero on
failure, which is all a wrapper needs. They are kept in their original
locations (paths are NAS-shared with the rigs) and run with cwd set to their
home directory because several use `sys.path.insert(0, ".")`.

Excluded (no failure signal — print-only): common_fsdp2/test_logger.py and
the old lr-schedule probe (now tools/lr_schedule_probe.py). Excluded
(torchrun required): tests/manual/test_row_center_dist.py and
tests/manual/test_gdn.py's FSDP2 half — run those by hand on a rig;
`norecursedirs = manual` keeps pytest from collecting them.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

MARA_ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = MARA_ROOT.parent / "common_fsdp2"

# flex_attention needs a working triton JIT — present on the rigs, absent on
# the local Windows box (its nvrtc fallback is also broken there).
HAS_TRITON = importlib.util.find_spec("triton") is not None
needs_flex = pytest.mark.skipif(not HAS_TRITON, reason="flex_attention JIT (triton) unavailable")

# (script path, marks)
LEGACY_SCRIPTS = [
    (COMMON_ROOT / "test_doc_mask.py", [needs_flex]),
    (COMMON_ROOT / "test_swa_cache.py", []),
    (COMMON_ROOT / "test_spec_decode.py", [pytest.mark.slow]),
    (COMMON_ROOT / "test_head_gauge_projection.py", []),
    (COMMON_ROOT / "test_centered_zloss.py", []),
    (COMMON_ROOT / "test_load_banner.py", []),
    (COMMON_ROOT / "test_body_lr_controller_shadow.py", []),
]


def _param(script, marks):
    return pytest.param(script, id=script.name, marks=marks)


@pytest.mark.legacy
@pytest.mark.parametrize("script", [_param(s, m) for s, m in LEGACY_SCRIPTS])
def test_legacy_script(script):
    assert script.exists(), f"missing legacy script: {script}"
    # PYTHONUTF8: with capture_output the child sees a cp1252 pipe on Windows
    # and any script printing a non-latin char (Δ, α...) dies in print().
    proc = subprocess.run(
        [sys.executable, script.name],
        cwd=str(script.parent),
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True, text=True, timeout=1200,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-30:])
        pytest.fail(f"{script.name} exited {proc.returncode}:\n{tail}")
