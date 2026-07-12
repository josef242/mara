# mara_fsdp2 / common_fsdp2 test suite

Created 2026-07-11 alongside a deep audit of the training-system guts. The
suite pins the invariants that past incidents proved fragile (RoPE meta-init
corruption, SWA rewind, Newton-Schulz radial bias, silent side-dict drops)
and carries the audit's open bugs as `xfail(strict=True)` — when a fix lands,
the xpass ERRORS and forces the pin to be promoted to a plain assert.

## Running

Local Windows box (torch 2.5.x — trainer tests skip automatically):

    C:/Users/josef/miniconda3/envs/trainenv/python.exe -m pytest            # from v:\code\mara_fsdp2
    ... -m "not slow"                                                       # skip the 4-min spec-decode script

Rig (torch 2.9, full deps — everything runs; code is NAS-shared, no sync step):

    ssh rig-31 "cd ~/valhalla/code/mara_fsdp2 && ~/miniconda3/envs/trainenv/bin/python -m pytest"

## Environment tiers

| tier | torch | what skips |
|---|---|---|
| local Windows | 2.9.1 cu128 + torchao (upgraded 2026-07-13 for rig parity), no triton/fla | flex-attention scripts, GDN meta-init test |
| rig-30/31 | 2.9.1, 8 GPU, full deps (fla on rig-30 only) | nothing (dist tests still need `RUN_DIST=1` + torchrun) |

Markers: `gpu` (auto-skip w/o CUDA), `slow` (>60s), `dist` (needs RUN_DIST=1),
`legacy` (subprocess wrapper around a pre-suite standalone script).

## Layout

- `conftest.py` — absolute sys.path for both repos; marker plumbing
- `helpers.py` — `tiny_args()`/`tiny_model()` CPU-sized model factories
- `test_sanity.py` — env probe: every core module imports
- `test_meta_init.py` — **the RoPE-corruption regression net**: meta → to_empty
  → init_weights must touch every tensor (storage poisoned with 12345.0 so
  "lucky zero pages" can't mask a miss)
- `test_kv_cache_parity.py` — decode == full forward; exact RoPE relative
  invariance (with corruption oracle so it can't rot vacuous)
- `test_doc_positions.py` — doc_ids/doc_position_ids pure-function semantics
- `test_moe.py` — BMM==loop parity, scatter/gather round-trip+grads, router
  combine-weight/bias contract, capacity-drop bookkeeping
- `test_muon_math.py` — momentum aliasing contract, Newton-Schulz bands,
  adam_update reference, NorMuon norm preservation
- `test_optimizers.py` — AdamC decay scaling/max_lr/wd_overrides; AdamW16bit
  vs AdamW trajectory; stochastic-rounding bias (rig tier)
- `test_train_parsing.py` — lr_mods/wd-rules parsing, MTP lambda schedule,
  resume mismatch guard (rig tier)
- `test_tail_truncation.py` — PTT gradient isolation (port of the old
  test_ptt.py, which printed PASS/FAIL but never asserted; its distribution
  analysis half lives on as tools/ptt_cutpoint_analysis.py)
- `test_row_center.py` — moved from repo root 2026-07-13, collected natively
- `test_legacy_scripts.py` — subprocess wrappers over the pre-suite scripts
  (they self-check and exit nonzero); excluded: print-only probes
  (common_fsdp2/test_logger.py, tools/lr_schedule_probe.py)
- `test_legacy_pytest_shims.py` — collects the already-pytest-style legacy
  files still living in common_fsdp2 (body_lr_controller ×12)
- `manual/` — torchrun/rig-only scripts (test_gdn.py, test_row_center_dist.py),
  deliberately NOT collected (`norecursedirs = manual`); run by hand:
  `torchrun --nproc_per_node=2 tests/manual/test_gdn.py`

## Conventions

- Differential tests compare LOGITS/values, not samples, in fp32/fp64 with
  seeded tiny models (spec-decode lesson: symmetric untrained distributions
  can mask broken components — prefer adversarial/asymmetric inputs).
- A test that pins a KNOWN bug is `xfail(strict=True)` with `AUDIT <date>` and
  the file:line in the reason. Never delete one without either fixing the bug
  or consciously accepting the behavior.
- Tests that need a corrupted-state oracle must also assert the oracle FAILS
  (see test_rope_relative_invariance) so the test can't silently go vacuous.
