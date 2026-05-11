"""One-shot: backfill cfg_voc_sz into saved run configs that lost it.

Older runs had cfg_voc_sz emitted into <run>/config_*.yaml by the
"dump full settings" save path. Once that path was replaced by a
verbatim source-YAML copy, configs whose source did not declare
cfg_voc_sz (e.g. dreadnought_v2.yaml) lost the field, which causes
the dashboard's w_rms convergence chart to silently skip the run.

This script appends the missing field as a trailing
"# --- Derived fields ---" block (matching what train_mara.py now
emits going forward). Existing file content is preserved verbatim;
the script is idempotent — re-running on an already-backfilled file
is a no-op.

Usage:
    python scripts/backfill_derived_fields.py V:/code/ckpt/mara_fsdp2/dreadnought_v2

cfg_voc_sz is derived from the tok_path field using the same
round-to-1024 rule as train_mara.py. Pass --dry-run to see what
would change without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Maps a substring of tok_path to the resolved cfg_voc_sz that
# train_mara.py would compute for that tokenizer (round_up(len(enc), 1024)).
TOK_TO_VOCAB: dict[str, int] = {
    "llama":  32768,   # llama tokenizer: len=32000  → round to 32768
    "cl100k": 100352,  # tiktoken cl100k_base: len=100256 → round to 100352
}

DERIVED_HEADER = "# --- Derived fields (computed at runtime, not in source config) ---"


def cfg_voc_sz_for(tok_path: str | None) -> int | None:
    if not tok_path:
        return None
    lowered = tok_path.lower()
    for needle, voc in TOK_TO_VOCAB.items():
        if needle in lowered:
            return voc
    return None


def backfill_one(path: Path, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        return f"SKIP ({path.name}): YAML parse error: {e}"

    if "cfg_voc_sz" in parsed:
        return f"OK   ({path.name}): cfg_voc_sz already present ({parsed['cfg_voc_sz']})"

    voc = cfg_voc_sz_for(parsed.get("tok_path"))
    if voc is None:
        return f"SKIP ({path.name}): no tok_path or unknown tokenizer ({parsed.get('tok_path')!r})"

    suffix = ""
    if not text.endswith("\n"):
        suffix += "\n"
    suffix += f"\n{DERIVED_HEADER}\n"
    suffix += f"cfg_voc_sz: {voc}\n"

    if dry_run:
        return f"WOULD ({path.name}): append cfg_voc_sz: {voc}"

    path.write_text(text + suffix, encoding="utf-8")
    return f"FIX  ({path.name}): appended cfg_voc_sz: {voc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory containing config_*.yaml files")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1

    configs = sorted(run_dir.glob("config_*.yaml"))
    if not configs:
        print(f"No config_*.yaml files in {run_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {len(configs)} config files in {run_dir}{' (dry run)' if args.dry_run else ''}")
    for p in configs:
        print(backfill_one(p, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
