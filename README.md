# mara_fsdp2 — Configuration Reference

Complete reference for every training-config setting (KEEL architecture + NorMuon optimizer, FSDP2). Runs are driven by YAML files in `configs/`; every key is loaded onto the `Settings` object in `train_mara.py` (class at ~line 4931), which fills defaults and validates. Each entry below gives the setting's **type**, **default**, **values/constraints**, **what it does** (grounded in the code), **interactions**, and a real **example**.

> Generated from a source-grounded pass over `train_mara.py`, the shared modules in `common_fsdp2/`, and the example configs. Line citations refer to those files.

## Contents

1. [Run, Checkpointing & Resume](#run-checkpointing--resume)
2. [Model Architecture (KEEL transformer)](#model-architecture-keel-transformer)
3. [Data Mix & Tokenizer](#data-mix--tokenizer)
4. [Batch, GA Schedule & Training Loop](#batch-ga-schedule--training-loop)
5. [Learning-Rate Schedule](#learning-rate-schedule)
6. [Optimizer (Muon / NorMuon / AdamC / AdamW families)](#optimizer-muon--normuon--adamc--adamw-families)
7. [Weight Decay](#weight-decay)
8. [Per-Layer LR Mods & Output-Head LR](#per-layer-lr-mods--output-head-lr)
9. [Precision, FSDP, Compile & Gradient Clipping](#precision-fsdp-compile--gradient-clipping)
10. [Body Growth Control: Tangent Projection + Shadow-Norm PDR Controller](#body-growth-control-tangent-projection--shadow-norm-pdr-controller)
11. [Head Hygiene (gauge projection, z-loss, row-center)](#head-hygiene-gauge-projection-z-loss-row-center)
12. [Adaptive Weight Decay (AWD)](#adaptive-weight-decay-awd)
13. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
14. [Gated DeltaNet (GDN) Hybrid Attention](#gated-deltanet-gdn-hybrid-attention)
15. [Auxiliary Heads & Staged Interventions](#auxiliary-heads--staged-interventions)
16. [Progressive Tail Truncation](#progressive-tail-truncation)
17. [Telemetry, Tracking & Health Guards](#telemetry-tracking--health-guards)

---

## Run, Checkpointing & Resume

This section covers run identity, where checkpoints and logs live, and how a run picks up from an existing checkpoint. The core pattern: you set `run_name` + `nas_root` (durable/NAS storage) + `local_checkpoint_root` (fast local scratch); everything else (`nas_path`, `resume_checkpoint_path`, `local_checkpoint_dir`) is auto-derived from those in `Settings.__init__` (train_mara.py:4952-4973) unless you override it explicitly. Checkpoints are written locally first, then rsync-migrated to the NAS (train_mara.py:3227-3495).

### `run_name`
- **Type:** string
- **Default:** required (no default; run aborts if empty)
- **Values/constraints:** Must be non-empty. Validated at train_mara.py:5575 — if falsy, prints "Run name is required" and `sys.exit(1)`. Overridable from CLI via `--run-name`, which also rewrites `nas_path`, `resume_checkpoint_path`, and `local_checkpoint_dir` to `./log/<name>/...` (train_mara.py:5561-5565).
- **What it does:** The run's identity string. It is the leaf directory name for both durable storage (`nas_path = nas_root/run_name/`) and local scratch (`local_checkpoint_dir = local_checkpoint_root/run_name`). Also logged as the run header (train_mara.py:5740). Note the CLI `--run-name` override hardcodes `./log/` for `nas_path`, ignoring any `nas_root` you set — so prefer setting `run_name` in the YAML if you use a custom `nas_root`.
- **Interacts with:** `nas_root`/`nas_path`, `local_checkpoint_root`/`local_checkpoint_dir`, `resume_checkpoint_path` (all derived from it).
- **Example:** `run_name: dn4` (configs/dn4.yaml:20)

### `resume_training`
- **Type:** bool
- **Default:** absent → treated as falsy (fresh run). Code guards everywhere with `getattr(settings, 'resume_training', False)`.
- **Values/constraints:** If `true`, `resume_step` MUST be set or the run fatals: "resume_training is True but resume_step is not set" (train_mara.py:4961-4962).
- **What it does:** Master switch for resuming from a checkpoint. When true, `main()` calls `resume_training(...)` (train_mara.py:6437), which loads model weights, optimizer shards, RNG/dataloader state, and (if present) AWD/body-LR-controller/row-center/MoE-bias sidecar states, and sets `start_step`/`total_tokens_processed` from the checkpoint. When false, the model trains from step 1 with fresh init. Several first-step-only behaviors (e.g. head-gauge init at train_mara.py:1772) are suppressed on resume.
- **Interacts with:** `resume_step` (required when true), `resume_checkpoint_path` (derived when true), `resume_data_reset` (only consulted on the resume path), `nas_path` (source of the derived checkpoint path).
- **Example:** `resume_training: true` (configs/dn3.yaml:38); `resume_training: false` (configs/dn4.yaml:45)

### `resume_step`
- **Type:** int
- **Default:** absent (None). Required whenever `resume_training: true`.
- **Values/constraints:** Any integer step for which a checkpoint exists. Used to format the default checkpoint filename with 6-digit zero-padding: `model_step_{resume_step:06d}.pt` (train_mara.py:4965). YAML underscore integer literals are accepted (e.g. `7_000`).
- **What it does:** Identifies which saved checkpoint to resume from. It seeds the derived `resume_checkpoint_path` and is passed through as `resume_step`/`mix_step` to the dataloader/data-schedule setup (train_mara.py:5757-5772) so the data mix resumes at the right point. It is only consulted when `resume_training` is true (`resume_step = settings.resume_step if settings.resume_training else None`, train_mara.py:5757).
- **Interacts with:** `resume_training` (gates it), `resume_checkpoint_path` (derived from it + `nas_path`), data schedule/mix step.
- **Example:** `resume_step: 7_000` (configs/dn3.yaml:39)

### `resume_checkpoint_path`
- **Type:** string (filesystem path)
- **Default:** auto-derived when resuming: `f"{nas_path}model_step_{resume_step:06d}.pt"` (train_mara.py:4964-4965). Only computed when `resume_training` is true.
- **Values/constraints:** Path to a `.pt` model checkpoint. Sidecar files (RNG, optimizer shards) are looked up in the SAME directory: `os.path.dirname(resume_checkpoint_path)` (train_mara.py:3588). So if you override this to point somewhere non-standard, the per-rank RNG/optimizer/dataloader files must live alongside it.
- **What it does:** The exact file `resume_training()` loads model weights from (`torch.load(settings.resume_checkpoint_path, ...)`, train_mara.py:3511). Override it explicitly (commented example in configs/kv2.yaml:183) to resume across a `run_name`/directory boundary — e.g. fork a new run from another run's checkpoint. Missing per-rank RNG files are tolerated (topology-change path, train_mara.py:3618-3623), but the model `.pt` itself must exist.
- **Interacts with:** `nas_path` + `resume_step` (its derivation), `resume_training` (gate), the co-located optimizer/RNG sidecar files.
- **Example:** `# resume_checkpoint_path: ./log/4B-T-AC/model_step_008500.pt` (commented override, configs/kv2.yaml:183)

### `resume_data_reset`
- **Type:** string (enum)
- **Default:** `"continue"` (fallback at train_mara.py:3611-3612 when the attribute is absent).
- **Values/constraints:** One of `"continue"`, `"soft"`, `"hard"` (dataloader.py:1330-1358; unknown value raises `ValueError`). Only read on the resume path, and only when the per-rank RNG/dataloader-state file exists. Note: the dataloader lives in `v:\code\common_fsdp2\dataloader.py`, not the mara_fsdp2 tree.
- **What it does:** Controls how the dataloader's counters are reset after its saved state is restored on resume. `continue` = no reset, resume exactly where the data stream left off. `soft` = `_soft_reset()`, clears the per-group window counters and re-bases `window_start_tokens` to the current historical total (use when you changed group percentages/mix and want the sampling window to re-target without discarding historical totals). `hard` = zeroes `historical_tokens_served`, `window_tokens_served`, and `window_start_tokens` for every group while keeping shard positions (dataloader.py:1349-1355).
- **Interacts with:** `resume_training` (only consulted then), the `groups` data-mix config (soft/hard are for when you changed the mix).
- **Example:** `resume_data_reset: continue         # continue (no reset), soft (reset pct window), hard (reset all)` (configs/kv2.yaml:37)

### `nas_root`
- **Type:** string (directory path)
- **Default:** `"./log/"` (train_mara.py:4953-4954).
- **Values/constraints:** A directory path (typically the durable NAS mount). Combined with `run_name` to build `nas_path`.
- **What it does:** Base directory for durable/shared checkpoint + log storage. `nas_path` is derived as `os.path.join(nas_root, run_name) + "/"` (train_mara.py:4957). This is where checkpoints are migrated to after being written locally, and where logs are written. On the Cedar Park rigs it points at the shared NAS mount.
- **Interacts with:** `nas_path` (derived), `run_name` (leaf dir). Note: the CLI `--run-name` override ignores `nas_root` and hardcodes `./log/` (train_mara.py:5563).
- **Example:** `nas_root: /home/josef/brainbox/checkpoints/current/` (configs/dn4.yaml:176)

### `nas_path`
- **Type:** string (directory path)
- **Default:** derived — `os.path.join(nas_root, run_name) + "/"` (train_mara.py:4956-4957). Only overridden if you set it explicitly and non-None.
- **Values/constraints:** A directory path ending in `/`. If you set it manually it fully replaces the derived value.
- **What it does:** The resolved per-run durable directory. It is the logger's logdir (`logger._instance.set_logdir(settings.nas_path)`, train_mara.py:5726) — so all log files (`gen_log_file`, `train_log_file`, `val_log_file`) land under it. It is the rsync destination for checkpoint migration (`trigger_checkpoint_sync`, train_mara.py:3480-3495), where `config_<timestamp>.yaml` is saved (train_mara.py:5897), the source directory for the derived `resume_checkpoint_path`, and where `diagnostics.jsonl` is written (train_mara.py:6454). Rarely set by hand.
- **Interacts with:** `nas_root` + `run_name` (its derivation), `resume_checkpoint_path`, the three log-file settings, `local_checkpoint_dir` (migration source→dest pair).
- **Example:** `# nas_path: ./log/4B-T-AC/   # auto-derived from nas_root/run_name — can be overridden` (commented, configs/kv2.yaml:182)

### `local_checkpoint_root`
- **Type:** string (directory path)
- **Default:** `"~/checkpoints/"` (train_mara.py:4968-4969). Tilde is expanded via `os.path.expanduser` when building `local_checkpoint_dir`.
- **Values/constraints:** A directory path, `~` allowed. Should be fast local storage (checkpoints are written here first, then migrated to NAS).
- **What it does:** Base directory for the fast local scratch where checkpoints are physically written by `save_model()` before the async rsync migration to `nas_path`. `local_checkpoint_dir` is derived as `expanduser(local_checkpoint_root)/run_name` (train_mara.py:4971-4973).
- **Interacts with:** `local_checkpoint_dir` (derived), `run_name` (leaf dir), `nas_path` (migration destination).
- **Example:** `local_checkpoint_root: ~/checkpoints/` (configs/dn4.yaml:177)

### `local_checkpoint_dir`
- **Type:** string (directory path)
- **Default:** derived — `os.path.join(os.path.expanduser(local_checkpoint_root), run_name)` (train_mara.py:4971-4973). Overridden only if set explicitly and non-None.
- **Values/constraints:** A directory path. Created on demand with `os.makedirs(..., exist_ok=True)` at save time (train_mara.py:3227).
- **What it does:** The resolved per-run local scratch directory where every checkpoint artifact is written before migration: the model `.pt` (train_mara.py:3379), per-rank optimizer shards (`optimizer_step_*_rank_*.pt`, :3403), per-rank RNG/dataloader state (`rng_state_step_*_rank_*.pt`, :3423), and sidecars for AWD/body-LR/row-center/MoE-bias/EP-experts. The whole dir is rsync'd to `nas_path` afterward.
- **Interacts with:** `local_checkpoint_root` + `run_name` (derivation), `nas_path` (rsync destination).
- **Example:** `# local_checkpoint_dir: ~/checkpoints/4B-T-AC   # auto-derived — can be overridden` (commented, configs/kv2.yaml:184)

### `save_step`
- **Type:** int
- **Default:** none set in `Settings.__init__` — effectively required. Accessed directly as `settings.save_step` with no getattr fallback (train_mara.py:3198), so a missing value raises `AttributeError` at the first save check.
- **Values/constraints:** Positive integer (used as `step % settings.save_step == 0`). Must be > 0 to avoid a modulo-by-zero.
- **What it does:** Checkpoint cadence in optimizer steps. A checkpoint is saved whenever `step > 0 and (step % save_step == 0 or last_step)` (train_mara.py:3198) — i.e. every `save_step` steps plus always on the final step. Smaller = more frequent checkpoints (more resume granularity, more I/O). Logged at startup as "Save Step" (train_mara.py:6159).
- **Interacts with:** `max_steps` (last-step save), `local_checkpoint_dir`/`nas_path` (where saves go).
- **Example:** `save_step: 250` (configs/dn4.yaml:36); `save_step: 500` (configs/kv2.yaml:25)

### `gen_log_file`
- **Type:** string (bare filename)
- **Default:** No default in `Settings.__init__`. If absent, `getattr(settings, 'gen_log_file', None)` is used in the seed-from-log path (train_mara.py:1453), but it is passed directly to `logger.set_default_logfile(settings.gen_log_file)` at train_mara.py:5728 (no fallback there) — so set it in the config. The logger's own built-in default is `'log.txt'` (logger.py:59) if `set_default_logfile` is never called.
- **Values/constraints:** A BARE filename, not a path (train_mara.py:1445-1446 explicitly notes this). The logger writes it under its logdir, i.e. `nas_path/<gen_log_file>`.
- **What it does:** The default log file for `logger.print_and_log(...)` calls that don't specify a target — general run/training progress output. Set as the logger's `default_logfile` (train_mara.py:5728). The `seed_from_log` recovery path also reads it to resolve prior generation output.
- **Interacts with:** `nas_path` (the directory it lands in), `train_log_file`/`val_log_file` (sibling per-stream logs).
- **Example:** `gen_log_file: gen_log.txt` (configs/dn4.yaml:171, configs/kv2.yaml:172)

### `train_log_file`
- **Type:** string (bare filename)
- **Default:** No default in `Settings.__init__`. Accessed directly as `settings.train_log_file` (train_mara.py:2776) with no getattr fallback — effectively required if the training-log line executes.
- **Values/constraints:** A bare filename, resolved under `nas_path` by the logger.
- **What it does:** Target file for the per-step training metrics line, written via `logger.print_and_log(..., True, settings.train_log_file, silent=True)` (train_mara.py:2776) — the machine-readable training stream that dashboards parse.
- **Interacts with:** `nas_path` (directory), `gen_log_file`/`val_log_file` (sibling streams).
- **Example:** `train_log_file: train_log.txt` (configs/kv2.yaml:173)

### `val_log_file`
- **Type:** string (bare filename)
- **Default:** No default in `Settings.__init__`. Accessed directly as `settings.val_log_file` (train_mara.py:1786, 2845) and passed through helper calls (train_mara.py:146, 202) with no fallback — effectively required if validation runs.
- **Values/constraints:** A bare filename, resolved under `nas_path` by the logger.
- **What it does:** Target file for validation-metrics output (val loss / eval lines), written on the validation cadence (train_mara.py:2845, and the val helper at :202). The separate validation stream dashboards read for eval curves.
- **Interacts with:** `nas_path` (directory), `gen_log_file`/`train_log_file` (sibling streams), the validation interval settings.
- **Example:** `val_log_file: val_log.txt` (configs/kv2.yaml:174)

### `server_port`
- **Type:** int
- **Default:** `29600` (via `int(getattr(settings, 'server_port', 29600))`, train_mara.py:5722). No default is set in `Settings.__init__` — the getattr fallback is the only default.
- **Values/constraints:** A TCP port, distinct from the DDP/torchrun rendezvous port (the comment at :5722 notes "Separate port from DDP"). Coerced to int, so a string like `"8888"` also works.
- **What it does:** Port for the logger's background HTTP/log server, set via `logger._instance.set_server_port(int(logger_port))` (train_mara.py:5725). This is the endpoint the dashboard/log server binds to for live run monitoring — not the distributed-training communication port. Set distinct ports per concurrent run on the same host to avoid collisions.
- **Interacts with:** The logger server (independent of DDP init); should differ from the DDP rendezvous port and from other concurrent runs' `server_port`.
- **Example:** `server_port: 8888` (configs/dn4.yaml:174, configs/kv2.yaml:176)

---

## Model Architecture (KEEL transformer)

These knobs define the transformer geometry and the KEEL (Highway-style Post-LN) residual structure. In `train_mara.py` (~line 5847) they are read off `Settings` and passed into the `ModelArgs` dataclass (`common_fsdp2/model_v2.py:322`), which is the actual source of truth for defaults. Two access patterns matter: the core geometry keys (`cfg_layers`, `cfg_heads`, `cfg_embd`, `cfg_intermediate`, `dropout`, `norm_eps`, `use_activation_checkpointing`, `qk_norm_mode`) are read as bare `settings.X` (a key with no `Settings.__init__` default and no `getattr` fallback raises AttributeError, so it is **required from YAML** — note `qk_norm_mode` is read bare at `train_mara.py:5859` but is safe because `Settings.__init__` sets it to `None`), while the newer/optional keys (`cfg_kv_heads`, `tie_word_embeddings`, `rope_theta`, `use_keel`, `keel_alpha`, `attn_res_*`) are read with `getattr(settings, ..., default)` and fall back to the `ModelArgs` default if omitted.

### `cfg_layers`
- **Type:** int
- **Default:** required (read as `settings.cfg_layers`; no fallback in `Settings.__init__`)
- **Values/constraints:** Positive int. Maps to `ModelArgs.n_layers`. With KEEL, layer 0 stays Pre-LN and layers 1..N-1 are Post-LN (`model_v2.py:1381`). Also range-checks `auxiliary_heads` layer indices (`train_mara.py:5841`) and is the divisor for the AttnRes block count.
- **What it does:** Number of stacked transformer blocks (`Transformer` builds `n_layers` `TransformerBlock`s). Sets model depth; when `keel_alpha` is left null it also determines the KEEL scale via `n_layers * 2` (`model_v2.py:1379`).
- **Interacts with:** `keel_alpha` (auto = `n_layers*2`), `attn_res_block_size` (n_layers should be divisible by it), `auxiliary_heads.heads`.
- **Example:** `cfg_layers: 69` (from `configs/dn4.yaml`)

### `cfg_heads`
- **Type:** int
- **Default:** required (read as `settings.cfg_heads`)
- **Values/constraints:** Maps to `ModelArgs.n_heads`. **Hard constraint:** `cfg_embd % cfg_heads == 0` — `check_params()` fatals otherwise (`train_mara.py:4185`), since `head_dim = dim // n_heads` (`model_v2.py:506`). Must also be divisible by `cfg_kv_heads` for GQA (`assert args.n_heads % n_kv_heads == 0`, `model_v2.py:501`); note this divisibility is enforced only by the model assert, NOT by `check_params()` (which just bounds `n_kv_heads <= n_heads`).
- **What it does:** Number of query attention heads. Together with `cfg_embd` it fixes `head_dim` (commonly 64 or 128). Under DION-family optimizers with world_size>1 the divisibility suggestions in `suggest_fsdp_dimensions` also key off this.
- **Interacts with:** `cfg_embd` (must divide it), `cfg_kv_heads` (must be a multiple of it).
- **Example:** `cfg_heads: 40` (from `configs/dn4.yaml`)

### `cfg_kv_heads`
- **Type:** int (optional)
- **Default:** `None` (read via `getattr(settings, 'cfg_kv_heads', None)`; `ModelArgs.n_kv_heads` default is `None`)
- **Values/constraints:** When `None`, falls back to `n_heads` inside `Attention` (full MHA, `model_v2.py:500`). When set: must satisfy `cfg_heads % cfg_kv_heads == 0` (GQA assert, `model_v2.py:501`) and **cannot exceed `cfg_heads`** — `check_params()` fatals otherwise (`train_mara.py:4198`). The divisibility itself is not checked by `check_params()`; a non-dividing value fails at the model `assert`.
- **What it does:** Enables Grouped-Query Attention: the number of key/value heads is reduced to `cfg_kv_heads` while query heads stay at `cfg_heads`, with KV heads repeated `n_heads/n_kv_heads` times (`repeat_kv`, `model_v2.py:478`). Shrinks the KV cache and wk/wv projections.
- **Interacts with:** `cfg_heads` (divisibility + upper bound).
- **Example:** `cfg_kv_heads: 20` (GQA 2:1 over 40 heads, from `configs/dn4.yaml`)

### `cfg_embd`
- **Type:** int
- **Default:** required (read as `settings.cfg_embd`)
- **Values/constraints:** Maps to `ModelArgs.dim`. Must be divisible by `cfg_heads` (`check_params`, `train_mara.py:4185`). Guidance in the docstring (`train_mara.py:4095`): divisible by 64 for tensor-core efficiency. Under DION optimizers + multi-GPU it must also be divisible by world_size (`train_mara.py:4211`).
- **What it does:** Model hidden/residual-stream width. Sets the dimension of embeddings, all RMSNorms, attention projections, and the FFN input/output.
- **Interacts with:** `cfg_heads` (head_dim = cfg_embd/cfg_heads), `cfg_intermediate` (FFN expansion), optimizer world_size divisibility.
- **Example:** `cfg_embd: 2560` (from `configs/dn4.yaml`)

### `cfg_intermediate`
- **Type:** int (optional)
- **Default:** `None` → auto-computed. Read as `settings.cfg_intermediate` and passed to `ModelArgs.inner_dim`. If `None`, `FeedForward` computes `inner_dim = 128*ceil((2/3 * 4*dim)/128)` (SwiGLU-style 8/3·dim rounded up to a multiple of 128, `model_v2.py:802-805`).
- **Values/constraints:** Positive int. Under DION + multi-GPU must be divisible by world_size (`train_mara.py:4215`).
- **What it does:** Hidden dimension of the dense FFN (SwiGLU: `w1`,`w3` project dim→inner_dim, `w2` projects back). Larger = more FFN capacity/params.
- **Interacts with:** `cfg_embd`, `multiple_of` (nominally the rounding multiple — but see below), `moe_inner_dim` (expert FFN width when MoE is on).
- **Example:** `cfg_intermediate: 7680` (3× cfg_embd, from `configs/dn4.yaml`)

### `multiple_of`
- **Type:** int
- **Default:** not consumed (dead key)
- **Values/constraints:** Appears in some configs as `multiple_of: 128`, but in the current code it is **not read anywhere live** — the only references in `train_mara.py` are commented out (`train_mara.py:3313`, `4297`), and `neo_common.py:275` explicitly *excludes* it when copying config fields. The FFN's internal rounding to a multiple of 128 is hard-coded (`model_v2.py:805`), not driven by this key.
- **What it does:** Historically the FFN inner-dim rounding multiple; today it is inert config documentation. Setting it has no effect on the built model. `cfg_intermediate` is used verbatim when provided.
- **Interacts with:** `cfg_intermediate` (conceptually; no live coupling).
- **Example:** `multiple_of: 128` (present in `configs/keel_repro.yaml`, but ignored; note `configs/dn4.yaml` does NOT set it)

### `norm_eps`
- **Type:** float
- **Default:** required (read as `settings.norm_eps`; `ModelArgs.norm_eps` default is `1e-5` if it ever fell through)
- **Values/constraints:** Small positive float. Applied uniformly to every `RMSNorm` (attention_norm, ffn_norm, final norm, KEEL post-LN norms, QK-norm) via `eps=args.norm_eps`.
- **What it does:** Epsilon added under the rsqrt in RMSNorm (`model_v2.py:257`) for numerical stability. Also used as the deadband eps in the `after_rope_legacy` QK L2-norm (`model_v2.py:579`) and in the AttnRes softmax RMS (`model_v2.py:393`).
- **Interacts with:** `qk_norm_mode` (reused there), `use_keel` (post-LN norms use it).
- **Example:** `norm_eps: 1.0e-05` (from `configs/dn4.yaml`)

### `dropout`
- **Type:** float
- **Default:** required (read as `settings.dropout`; `ModelArgs.dropout` default is `0.0`)
- **Values/constraints:** `[0.0, 1.0)`. Applied only during `.training`; inference paths force `dropout_p=0.0` (`model_v2.py:751`).
- **What it does:** Single dropout rate reused for attention dropout, residual dropout, and FFN dropout (`model_v2.py:518-520`, `809`). In the SDPA/FlashAttention path it is passed as `dropout_p` when training; in the fallback manual-attention path it dropouts the scores.
- **Interacts with:** `-` (LLM pretraining configs here universally set `0.0`).
- **Example:** `dropout: 0.0` (from `configs/dn4.yaml`)

### `qk_norm_mode`
- **Type:** string | null
- **Default:** `None` (explicitly set in `Settings.__init__`, `train_mara.py:4977`)
- **Values/constraints:** One of `null`, `"before_rope"`, `"after_rope_legacy"`. **No validation** in Settings or the model — any other string silently behaves as "off" (all the branches are `== "before_rope"` / `== "after_rope_legacy"` string compares, `model_v2.py:527,568,576`).
- **What it does:** Controls query/key normalization for attention stability. `"before_rope"` (recommended) applies a *learnable* per-head `RMSNorm(head_dim)` to q and k before RoPE. `"after_rope_legacy"` applies a non-learnable L2 normalize after RoPE. `null` disables QK-norm entirely.
- **Interacts with:** `norm_eps` (used by both modes). KEEL repro runs turn it off (`qk_norm_mode: null`, paper does not use it).
- **Example:** `qk_norm_mode: "before_rope"` (from `configs/dn4.yaml`)

### `tie_word_embeddings`
- **Type:** bool
- **Default:** `True` (set in `Settings.__init__`, `train_mara.py:4981`; `ModelArgs` default also `True`)
- **Values/constraints:** bool.
- **What it does:** When `True`, the input embedding matrix (`tok_embeddings.weight`) is shared as the output LM head (no separate `output.weight` parameter) — saves memory. When `False`, an independent `output` projection is created; **required for the untied-head levers** (head_gauge_projection, output-head LR mods, DN2/DN4 head-hygiene work).
- **Interacts with:** head-gauge / output-LR features (they need an untied head), z_loss/head levers. Determines whether `output.weight` exists in the param name space.
- **Example:** `tie_word_embeddings: false` (untied, from `configs/dn4.yaml`)

### `rope_theta`
- **Type:** float
- **Default:** `500000.0` (via `getattr(settings, 'rope_theta', 500000.0)`, `train_mara.py:5861`; `ModelArgs.rope_theta` also `500000.0`)
- **Values/constraints:** Positive float. Passed into `precompute_freqs_cis(dim, end, theta)` (`model_v2.py:411`).
- **What it does:** Base frequency for Rotary Position Embeddings. Higher theta stretches the wavelength spectrum for longer-context support; the default 500000 is a long-context value. KEEL-paper reproduction uses the classic `10000.0`.
- **Interacts with:** `T`/`max_seq_len` (context length these rotations cover).
- **Example:** `rope_theta: 500_000.0` (from `configs/dn4.yaml`); `rope_theta: 10000.0` (from `configs/keel_repro.yaml`)

### `use_keel`
- **Type:** bool
- **Default:** `False` (via `getattr(settings, 'use_keel', False)`; `ModelArgs.use_keel` default `False`)
- **Values/constraints:** bool.
- **What it does:** Switches the residual topology from standard Pre-LN to KEEL "Highway-style Post-LN" (paper arXiv:2601.19895). When on, block 0 stays Pre-LN, and every later block computes `h = post_attn_norm(alpha*x + attn(norm(x)))` then `out = post_ffn_norm(alpha*h + ffn(norm(h)))` (`model_v2.py:1398-1408`), adding two extra `post_*_norm` RMSNorms per layer (layers ≥1). Enables training of very deep stacks stably.
- **Interacts with:** `keel_alpha` (the highway scale), `cfg_layers` (drives auto-alpha and the first-layer exception).
- **Example:** `use_keel: true` (from `configs/dn4.yaml`)

### `keel_alpha`
- **Type:** float | null
- **Default:** `None` → auto `n_layers * 2` (via `getattr(...,'keel_alpha', None) or (args.n_layers*2)`, `model_v2.py:1379`)
- **Values/constraints:** Positive float, or `null` for the auto value. Only consulted when `use_keel: true`. Note the `or` idiom means a config value of `0` would also fall through to the auto default.
- **What it does:** The highway scaling coefficient on the identity branch in KEEL blocks — larger alpha weights the residual/identity path more heavily against the sublayer output before the post-LN, keeping deep signal alive. The paper sets `alpha = L` (total sublayers = `n_layers*2`); the auto default follows that.
- **Interacts with:** `use_keel` (no effect unless on), `cfg_layers` (auto = 2×).
- **Example:** `keel_alpha: null` (auto → 138 for 69 layers, from `configs/dn4.yaml`); `keel_alpha: 512` (explicit, from `configs/keel_repro.yaml`)

### `use_activation_checkpointing`
- **Type:** bool
- **Default:** required (read as `settings.use_activation_checkpointing`; `ModelArgs` default is `True` if it fell through)
- **Values/constraints:** bool.
- **What it does:** When `True`, each `TransformerBlock.forward` wraps its inner compute in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` during training (`model_v2.py:1420`), trading recompute for activation memory. Off = faster steps, higher memory. Only active in `.training`; inference/cache path never checkpoints.
- **Interacts with:** `cfg_layers` (deep stacks like the 256-layer KEEL repro rely on this to fit in memory).
- **Example:** `use_activation_checkpointing: true` (from `configs/dn4.yaml`)

### `attn_res_enabled`
- **Type:** bool
- **Default:** `False` (via `getattr(settings, 'attn_res_enabled', False)`; `ModelArgs` default `False`)
- **Values/constraints:** bool.
- **What it does:** Enables Block Attention Residuals (AttnRes, Kimi 2026): the model learns a depth-wise softmax attention over the stored per-block representations and mixes them into the residual stream (`block_attn_res`, `model_v2.py:398`), letting later blocks read earlier block outputs. Implemented with activation checkpointing for memory efficiency.
- **Interacts with:** `attn_res_block_size` (defines the blocks it attends over), `cfg_layers` (should be divisible by the block size).
- **Example:** `attn_res_enabled: true` (from `configs/mini-fathom-low-lr-attnres.yaml`)

### `attn_res_block_size`
- **Type:** int
- **Default:** `8` (via `getattr(settings, 'attn_res_block_size', 8)`; `ModelArgs` default `8`)
- **Values/constraints:** Positive int; `cfg_layers` should be divisible by it (a "block" is this many consecutive layers). Only meaningful when `attn_res_enabled: true`.
- **What it does:** Number of transformer layers per AttnRes block. It partitions the depth into `n_layers / attn_res_block_size` blocks; a completed block-level representation is captured and appended for depth-wise AttnRes mixing at every block boundary, i.e. when `(i+1) % attn_res_block_size == 0` (`model_v2.py:1890`).
- **Interacts with:** `attn_res_enabled` (no effect unless on), `cfg_layers` (divisibility).
- **Example:** `attn_res_block_size: 10` (70 layers → 7 blocks of 10 layers each, from `configs/mini-fathom-low-lr-attnres.yaml`)

---

## Data Mix & Tokenizer

This section covers how the trainer selects a tokenizer and how it assembles its training corpus from on-disk `.npy` token shards. The tokenizer is built by `get_tokenizer()` in `common_fsdp2/tokenizer_abstraction.py` (called at `train_mara.py:5820`); the data corpus is described by `groups` and served by `PercentageDataLoader` / `DataMixSchedule` in `common_fsdp2/dataloader.py` (constructed at `train_mara.py:5756-5772`). Note: none of these keys get a default assigned in `Settings.__init__` — they are read directly as `settings.<key>`, so all are effectively **required** except `special_tokens`, which is read via `getattr(settings, 'special_tokens', None)` (`train_mara.py:5818`) and is therefore optional.

### `tok_kind`
- **Type:** string
- **Default:** required (read directly as `settings.tok_kind` at `train_mara.py:5821`; the `get_tokenizer` signature default `"llama"` is never reached since the trainer always passes the config value)
- **Values/constraints:** case-insensitive. Recognized: `"llama"` (in-house SentencePiece via `LlamaTokenizerAdapter`, requires `tok_path`); `"tiktoken"`/`"cl100k"`/`"o200k"`/`"p50k"`/`"r50k"` (OpenAI tiktoken via `TikTokenAdapter`); `"claude"` (numeric R2L tokenizer via `ClaudeTokenizerAdapter`, requires `tok_path`); any other string falls through to the generic HuggingFace `AutoTokenizer` branch (`HFTokenizerAdapter`), treating `tok_path` or the kind string itself as a model id/dir. See `tokenizer_abstraction.py:818-853`.
- **What it does:** Selects which tokenizer adapter is instantiated. This determines the vocab, `bos_id`/`eos_id`/`pad_id`, and the encode/decode behavior used to read the token shards. After construction the trainer computes `settings.cfg_voc_sz = round_up(len(enc), 1024)` (`train_mara.py:5828`), rounding the tokenizer's vocab up to a multiple of 1024 for GPU-friendly embedding/output dims. In practice all repo configs use either `llama` or `tiktoken`.
- **Interacts with:** `tok_path` (required for `llama`/`claude`, is the encoding name for tiktoken), `special_tokens` (only actually applied for `tiktoken` and `llama` — see below), and indirectly the model's `cfg_voc_sz`.
- **Example:** `tok_kind: llama` (from `configs/dn4.yaml`); `tok_kind: tiktoken` (from `configs/machete.yaml`)

### `tok_path`
- **Type:** string
- **Default:** required (read as `settings.tok_path` at `train_mara.py:5822`; passed as `path=`)
- **Values/constraints:** Meaning depends on `tok_kind`. For `llama`: a filesystem path to the SentencePiece model dir/file (`ValueError` raised if `None`, `tokenizer_abstraction.py:826`). For tiktoken kinds: an **encoding name string**, not a path — if omitted it defaults to `cl100k_base` (or `<kind>_base`) (`tokenizer_abstraction.py:830-834`). For `claude`: required path to the tokenizer dir (`ValueError` if `None`). For HF fallback: a model-id or local directory. Paths are `expanduser()`-ed and are relative to the trainer's working directory.
- **What it does:** Points the selected adapter at its tokenizer artifact. For `llama`/`claude`/HF it is loaded from disk; for tiktoken it names the built-in BPE encoding to fetch via `tiktoken.get_encoding()`.
- **Interacts with:** `tok_kind` (governs whether this is a path or an encoding name).
- **Example:** `tok_path: ../tokenizers/llama_tokenizer` (from `configs/dn4.yaml`); `tok_path: cl100k_base` (from `configs/machete.yaml`)

### `special_tokens`
- **Type:** string | list | dict (optional)
- **Default:** `None` (via `getattr(settings, 'special_tokens', None)` at `train_mara.py:5818`)
- **Values/constraints:** Passed through `_load_special_tokens()` (`tokenizer_abstraction.py:742-790`), which accepts: a **list** of token strings (used as-is); a **dict** mapping token→id (only the keys are used); or a **path to a JSON file** containing either a bare list, `{"special_tokens": [...] | {...}}`, or `{"tokens": [...]}`. A non-existent file path raises `FileNotFoundError`. **Only honored for `tiktoken` and `llama`** — for `claude` and HF kinds a `[warning] special_tokens not yet supported` is printed and the tokens are ignored (`tokenizer_abstraction.py:844-852`).
- **What it does:** Appends custom special tokens to the base vocabulary, assigned contiguous IDs starting after the base vocab (llama: after `base_vocab_size`; tiktoken: after `n_vocab`, after an auto-added `<pad>` if missing). These extra tokens increase `len(enc)`, which flows into the rounded `cfg_voc_sz`. In `dn4.yaml` the value points at a `tokenizer_config.json` alongside the tokenized data.
- **Interacts with:** `tok_kind` (silently ignored except for llama/tiktoken), `tok_path`; increases effective vocab / `cfg_voc_sz`.
- **Example:** `special_tokens: ../../notebooks/datasets/tokenized/llama/tokenizer_config.json` (from `configs/dn4.yaml`)

### `data_root_path`
- **Type:** string
- **Default:** required (read as `settings.data_root_path` at `train_mara.py:5766` and `:5771`)
- **Values/constraints:** A directory path (relative to the working dir in all repo configs). Must contain one subdirectory per group `name` in `groups`. The loader looks for shards at `<data_root>/<name>/*_<split>_*.npy` where `<split>` is `train` or `val` (`dataloader.py:590-592`). An **active** group (percentage > 0) with no matching shards raises `ValueError`; a 0% group with no shards is skipped with a warning (`dataloader.py:594-599`). A token-count cache `token_counts.json` is read/written under this root (`dataloader.py:585`, `_save_cache`).
- **What it does:** Root under which the `PercentageDataLoader` discovers, counts (via mmap'd `np.load`), and streams token shards for every group. Both train and val loaders share this root, differing only by the `_train_`/`_val_` filename infix.
- **Interacts with:** `groups` (each group name is a required subdirectory under this root); the tokenizer choice must match how the shards were tokenized (e.g. `cl100k/` data with `tok_kind: tiktoken`, `llama/` data with `tok_kind: llama`).
- **Example:** `data_root_path: ../../notebooks/datasets/tokenized/llama/` (from `configs/dn4.yaml`)

### `groups`
- **Type:** list (of `[name, weight]` pairs; YAML lists are coerced to tuples at `train_mara.py:4938-4940`)
- **Default:** required (read as `settings.groups`; no default set in `Settings.__init__`)
- **Values/constraints:** Each entry is `[name, weight]`. `name` is a string matching a subdir under `data_root_path`. `weight` is either a **static number** (e.g. `20.0`) or a **schedule** — a list of `[step, weight]` waypoints (e.g. `[[0, 5.0], [5000, 0.0]]`). Weights are treated as **percentages**. Validation (`dataloader.py:359-367`): the **active** groups (percentage > 0) at the initial step must sum to 100 ± 0.01, else `ValueError`; groups at exactly 0% are allowed and treated as deprecated/scheduled-inactive. Note real configs write even static weights in single-waypoint schedule form, e.g. `[stories, [[0, 28.0]]]`.
- **What it does:** Defines the data mixture. If **any** group carries a list-schedule, `DataMixSchedule.from_groups()` builds an annealing schedule (`dataloader.py:172-207`): it unions all waypoint steps, linearly interpolates each group's weight per step (`_interp_waypoints`), and at each queried step **re-normalizes** the mix to sum to exactly 100% via `_normalize()` (`dataloader.py:111-123`) — so raw waypoint values need not sum to 100 when a schedule is present; they are proportions. Groups that ramp to 0 become deprecated (stop being served); groups that ramp in from 0 come online. On resume the mix is evaluated at the resume step, not step 0 (`train_mara.py:5756-5768`), so mid-ramp shard positions restore correctly. If all weights are static, `from_groups` returns `None` and the fixed `groups` list is used directly, with the sum-to-100 rule enforced by the loader.
- **Interacts with:** `data_root_path` (each `name` is a subdir); the resume logic (`resume_step`/`resume_training`); the tokenizer (shards must be pre-tokenized with the matching tokenizer). The val loader periodically syncs its active-group percentages from the train loader (`train_mara.py:1389-1390`) so validation covers whatever is currently mixed.
- **Example (static mix):**
  ```yaml
  groups:
    - [stories,      [[0, 28.0]]]
    - [ao3,          [[0, 12.0]]]
    - [books,        [[0, 28.0]]]
    - [preselect,    [[0, 11.0]]]
    - [code_python,  [[0, 6.0]]]
    - [code_c,       [[0, 6.0]]]
    - [edufineweb_1.5TT, [[0, 9.0]]]
  ```
  (from `configs/dn4.yaml`)
- **Example (annealed mix, ramps `edufineweb` in and shifts others):**
  ```yaml
  groups:
    - [stories,      [[0, 21.0], [11000, 21.0], [13000, 15.0], [15000, 15.0], [25000, 19.0]]]
    - [edufineweb_1.5TT, [[0, 0.0], [11000, 0.0], [13000, 5.0], [15000, 5.0], [25000, 1.0]]]
    # ... other groups ...
  ```
  (from `configs/dreadnought.yaml`)

### `data_type`
- **Type:** string
- **Default:** required (read as `settings.data_type`; no default in `Settings.__init__`). Every config in the repo sets `bf16`.
- **Values/constraints:** `"bf16"` → `torch.bfloat16`, `"fp16"` → `torch.float16`, anything else → `torch.float32` (the mapping appears at `train_mara.py:158`, `1934`, `2349`, `2888-2889`, `2918-2919`). No explicit validation — an unrecognized string silently falls through to fp32.
- **What it does:** Sets the autocast compute dtype for the forward/backward passes (`torch.autocast(..., dtype=...)`) and is also passed into `compute_mfu()` (`train_mara.py:2650`) and the peak-FLOPs helpers to pick the right hardware throughput number for the MFU calculation. It governs the mixed-precision math dtype of training, not the master weight/optimizer-state precision (those are handled separately by FSDP2 `mp_policy` and the optimizer).
- **Interacts with:** MFU reporting (`compute_mfu`, `default_peak_table`, `_device_peak_flops`); orthogonal to FSDP2's `mp_policy` reduce/param dtypes.
- **Example:** `data_type: bf16` (from `configs/dn4.yaml` — and universally across the repo)

---

## Batch, GA Schedule & Training Loop

These settings define the per-GPU micro-batch geometry (`B`, `T`), the effective (global) batch-size ramp via gradient accumulation, how many optimizer steps to run, and the validation cadence. There are **two mutually-exclusive GA modes**: the modern **explicit `ga_schedule`** (used by every current config) and a **legacy automatic mode** (`target_batch_size` / `min_batch_size` / `ramp_percent`) that only runs when `ga_schedule` is absent or empty (`train_mara.py:4875`). **None** of the settings in this section have a default in `Settings.__init__` — they are only ever set by the generic YAML `setattr` loop (`train_mara.py:4936-4940`). `B`, `T`, `eval_iters`, and `val_step` are read with a hard `settings.X` access, so omitting them raises `AttributeError` at use-time (i.e. they are effectively required). `max_steps` is also never defaulted, but it is guarded by an explicit truthiness check — `if not self.max_steps: print(...); sys.exit(1)` (`train_mara.py:5579-5581`) — so omitting it produces a clean error/exit rather than an `AttributeError`.

### Effective batch size — how it's computed
The atomic unit is one micro-batch across all ranks:
`tok_per_micro = B * T * ddp_world_size` (`train_mara.py:4865`, `4768`).
The optimizer accumulates `grad_accum_steps` micro-batches before each `optimizer.step()`, so the **effective (global) batch in tokens at step `s`** is:
`eff_batch = grad_accum_schedule[s] * B * T * ddp_world_size` (`train_mara.py:5811`, `2643`).
Under `ga_schedule`, each requested `global_batch_tokens` is converted to an integer GA count by `actual_ga = max(1, round(global_batch_tokens / tok_per_micro))`, then rounded back to `actual_ga * tok_per_micro` — so the realized batch may differ slightly from what you asked for (the `diff` is printed in the schedule table). This means the realized effective batch depends on world size: change GPU count and the GA counts (hence exact token batch) shift.

### `B`
- **Type:** int
- **Default:** required (no default; hard `settings.B` access at `train_mara.py:5765`)
- **Values/constraints:** No explicit validation. Must be a positive int the GPU can fit; it is passed straight to `PercentageDataLoader(B=settings.B, ...)` (`train_mara.py:5765`, `5770`).
- **What it does:** Per-GPU micro-batch size — the number of sequences in a single forward/backward micro-step on one rank. It is one factor of `tok_per_micro = B * T * world_size`, the quantum the GA schedule accumulates into an effective batch. Larger `B` means fewer GA steps are needed to hit a target token batch, but more activation memory per micro-step.
- **Interacts with:** `T`, `ddp_world_size`, `ga_schedule`/`target_batch_size` (together set the realized effective batch and the GA counts).
- **Example:** `B: 12` (from `configs/dn4.yaml:31`)

### `T`
- **Type:** int
- **Default:** required (no default; hard `settings.T` access)
- **Values/constraints:** No explicit validation here. Passed to the data loader (`train_mara.py:5765`) and used as `max_seq_len=settings.T` for the model (`train_mara.py:5854`); must not exceed the model's positional/context capacity.
- **What it does:** Context length (sequence length in tokens) per sample. It is the second factor of `tok_per_micro`, and also configures the model's `max_seq_len` and the RoPE/attention context. Doubling `T` doubles tokens-per-micro (and roughly the attention cost), so it directly scales both throughput accounting and effective batch in tokens.
- **Interacts with:** `B`, `ddp_world_size` (token budget); model `max_seq_len` (`train_mara.py:5854`).
- **Example:** `T: 2048` (from `configs/dn4.yaml:32`)

### `ga_schedule`
- **Type:** list of `[step, global_batch_tokens]` pairs (schedule/ramp)
- **Default:** unset → falls back to legacy automatic mode (`train_mara.py:4875`, `4885`)
- **Values/constraints:** Each entry is `[start_step, desired_global_batch_in_tokens]`. Pairs are sorted by step internally, so out-of-order entries are tolerated (`train_mara.py:4793`). Each value is converted to an integer GA count `max(1, round(desired_batch / tok_per_micro))` — so a desired batch smaller than `tok_per_micro` clamps to GA=1, and the realized batch is re-quantized to `actual_ga * tok_per_micro`.
- **What it does:** Defines a **step-function ramp of effective batch size**. `build_user_defined_schedule()` (`train_mara.py:4784`) holds each pair's GA value from its `start_step` until the next pair's step (last pair holds to `max_steps`), producing a per-step `grad_accum_schedule` array of length `max_steps`. The training loop reads `grad_accum_steps = grad_accum_schedule[step]` each step (`train_mara.py:1865`) and runs that many micro-batches. The **final** GA value sets `settings.total_batch_size = grad_accum_schedule[-1] * tok_per_micro` (`train_mara.py:4884`). On resume, GA is re-indexed from `grad_accum_schedule[start_step]` (`train_mara.py:4035`). When present and non-empty, this fully overrides `target_batch_size`/`min_batch_size`/`ramp_percent`.
- **Interacts with:** `B`, `T`, `world_size` (set `tok_per_micro`, hence the realized batch and the ramp's GA counts); `max_steps` (schedule length / final hold); `output_lr_batch_adjust` (derives an output-head LR schedule from this ramp, `train_mara.py:5780-5811`); makes the legacy `target_batch_size`/`min_batch_size`/`ramp_percent` inert.
- **Example (from `configs/dn4.yaml:23-30`):**
  ```yaml
  ga_schedule:
    - [0, 125_000]
    - [2_000, 250_000]
    - [4_000, 500_000]
    - [7_000, 1_000_000]
    - [10_500, 2_000_000]
    - [14_500, 3_000_000]
    - [17_000, 4_000_000]
  ```

### `max_steps`
- **Type:** int
- **Default:** required — no default assignment in `Settings.__init__`; enforced by explicit guard `if not self.max_steps: print(...); sys.exit(1)` (`train_mara.py:5579-5581`)
- **Values/constraints:** Must be truthy (non-zero). Underscored literals like `200_000` are fine (YAML/Python int). Defines the exclusive upper bound of the training loop and the length of the GA/LR schedule arrays.
- **What it does:** Total number of optimizer steps to run. The main loop is `for step in range(start_step, settings.max_steps)` (`train_mara.py:1793`), with `last_step = (step == max_steps - 1)`. It also sizes `grad_accum_schedule = [0]*max_steps` (`train_mara.py:4790`) and is the denominator/anchor for LR schedules, ramp-percent, restart points, and the ffn-pdr-controller warmup bounds (`train_mara.py:5087-5143`). Note it counts optimizer steps, not tokens — total tokens depend on the GA ramp (logged as "Total Tokens to Process", `train_mara.py:4926`).
- **Interacts with:** `ga_schedule`/`ramp_percent` (schedule length), all LR schedule settings (`warmup_steps`, restarts, dual-plateau), `val_step`, `save_step`.
- **Example:** `max_steps: 200_000` (from `configs/dn4.yaml:33`)

### `eval_iters`
- **Type:** int
- **Default:** required (no default; hard `settings.eval_iters` access at `train_mara.py:2844`, `1785`)
- **Values/constraints:** Positive int; used directly as a loop count and as the averaging divisor (`tot /= eval_iters`, `train_mara.py:140`), so `0` would divide-by-zero. The config comment notes it is "iterations for evaluation x ranks" (`configs/kv2.yaml:23`).
- **What it does:** Number of validation micro-batches averaged **per validation group** during a validation pass. `calc_group_loss()` runs `for _ in range(eval_iters)` pulling `loader.next_batch()`, sums the loss, divides by `eval_iters`, then all-reduces the mean across ranks (`train_mara.py:116-142`). `do_validation()` calls this once per active val group (`train_mara.py:160-167`). Small values (e.g. 2) keep validation cheap; the effective validation sample count is `eval_iters * B * T * world_size` per group.
- **Interacts with:** `val_step` (how often this runs), `B`/`T`/`world_size` (tokens per eval batch), val-group configuration.
- **Example:** `eval_iters: 2` (from `configs/dn4.yaml:34`)

### `val_step`
- **Type:** int
- **Default:** required at the primary read (hard `settings.val_step` at `train_mara.py:1799`, `2841`); some peripheral reads use `getattr(settings, 'val_step', 100)` (`train_mara.py:3125`, `3866`, `6339`), implying an intended nominal of ~100.
- **Values/constraints:** Positive int. Used as `step % settings.val_step == 0` (`train_mara.py:1799`), so it is the modulo period. The FFN-pdr controller's EMA alphas/rate-limits are tuned for `val_step ≈ 100`; a value far from 100 triggers a warning because `observe()` runs at this cadence and re-times the controller (`train_mara.py:6339-6344`).
- **What it does:** Cadence (in optimizer steps) for validation and for the heavier val-cadence diagnostics. On steps where `step % val_step == 0` (or the last step), the loop forces full-depth (`is_val_step`, `train_mara.py:1799`), runs `do_validation()` (`train_mara.py:2841-2844`), and emits diagnostics/telemetry (`train_mara.py:2026`, `2250`) including body-pdr and other val-cadence-only quantities. It is also the observation cadence for the body-LR / FFN-pdr controllers.
- **Interacts with:** `eval_iters` (cost per validation), `save_step` (independent checkpoint cadence), `ffn_pdr_controller`/body-LR controller EMA tuning, `max_steps` (last-step forced val).
- **Example:** `val_step: 100` (from `configs/dn4.yaml:35`)

### `target_batch_size` *(legacy automatic-mode sibling)*
- **Type:** int (tokens)
- **Default:** required **only in automatic mode** — read as `settings.target_batch_size // tok_per_micro` (`train_mara.py:4827`, `4892`). Ignored entirely when `ga_schedule` is set.
- **Values/constraints:** Interpreted as a global token batch; integer-divided by `tok_per_micro` to get the max GA count. No standalone validation.
- **What it does:** In the legacy `build_automatic_schedule()` path (only entered when `ga_schedule` is unset/empty), this is the **peak** effective batch. The builder ramps GA through powers of two (starting from `min_batch_size`) up to `target_batch_size // tok_per_micro` over `ramp_percent * max_steps` steps, then holds at the max for the rest of training (`train_mara.py:4827-4862`). Not used by any current config in `configs/` — all use `ga_schedule`.
- **Interacts with:** `min_batch_size`, `ramp_percent` (the other two automatic-mode knobs); mutually exclusive with `ga_schedule`.
- **Example:** `target_batch_size: 2_000_000` (legacy-mode; not present in the modern configs, which use `ga_schedule` instead)

### `min_batch_size` *(legacy automatic-mode sibling)*
- **Type:** int (tokens)
- **Default:** optional — guarded by `hasattr(settings, 'min_batch_size') and settings.min_batch_size`; when absent, `min_grad_accum_steps = 1` (`train_mara.py:4830-4832`).
- **Values/constraints:** Tokens; converted to `max(1, min_batch_size // tok_per_micro)` as the smallest GA value in the automatic ramp.
- **What it does:** Sets the **starting** effective batch of the legacy automatic ramp — the power-of-two GA ladder begins at this floor instead of GA=1. Only meaningful in automatic mode (no `ga_schedule`). Inert in all current configs.
- **Interacts with:** `target_batch_size`, `ramp_percent`; ignored when `ga_schedule` is set.
- **Example:** `min_batch_size: 125_000` (legacy-mode)

### `ramp_percent` *(legacy automatic-mode sibling)*
- **Type:** float (fraction 0-1)
- **Default:** optional — reads are guarded by `hasattr(settings, 'ramp_percent')` for logging (`train_mara.py:4869`), but inside `build_automatic_schedule()` it is accessed hard as `settings.max_steps * settings.ramp_percent` (`train_mara.py:4834`), so it is required if you actually use automatic mode.
- **Values/constraints:** Fraction of `max_steps` over which the automatic GA ramp completes; `total_ramp_steps = int(max_steps * ramp_percent)`. Logged as a percentage (`train_mara.py:4915`).
- **What it does:** Controls how long the legacy automatic mode takes to climb from `min_batch_size` to `target_batch_size` (the GA ladder transitions are spread evenly across `ramp_percent * max_steps`, then it holds at max). Only consulted in automatic mode; when `ga_schedule` is present the ramp-steps log line is suppressed (`train_mara.py:4914`) and this value has no effect on the schedule.
- **Interacts with:** `target_batch_size`, `min_batch_size`, `max_steps`; mutually exclusive with `ga_schedule`.
- **Example:** `ramp_percent: 0.1` (legacy-mode; 10% of training spent ramping GA)

---

## Learning-Rate Schedule

These settings shape the *global* base LR curve that every param group inherits (per-layer `lr_mods` and the body-LR controller multiply on top). The schedule is a pure function of the global step (`get_lr(it, settings)`, train_mara.py:244), so it round-trips cleanly across kill/resume. Three schedule shapes are selectable via `lr_schedule_type`; `cosine` and `restarts` share one implementation (`get_lr_with_restarts`, train_mara.py:872), while `plateau` uses a separate dual-plateau function (`get_lr_with_dual_plateau`, train_mara.py:906) whose knobs are documented in the plateau section.

### `lr_schedule_type`
- **Type:** string
- **Default:** `restarts` — this is the fallback used by every `getattr(settings, 'lr_schedule_type', 'restarts')` call site (train_mara.py:224, 248, 962, 5174). There is no assignment in `Settings.__init__`, so if the key is omitted from YAML the attribute is absent and the `restarts` default applies. Note: essentially every real config sets this explicitly, most commonly to `cosine` or `plateau`.
- **Values/constraints:** `cosine`, `restarts`, or `plateau`. Any other value hits `fatal_error(...)` in `get_lr` (train_mara.py:273). Not validated in `Settings.__init__` — the check happens lazily on the first `get_lr` call.
- **What it does:** Routes the base-LR computation (train_mara.py:248-273). `cosine` calls `get_lr_with_restarts` with an *empty* restart list, giving a single linear-warmup-then-cosine-decay curve from `max_lr` to `min_lr` over `max_steps`. `restarts` calls the same function but passes `restart_steps`/`restart_gamma`, producing warm restarts. `plateau` calls the dual-plateau function (warmup → decay → hold → decay → hold → final decay) and requires the `first_plat_*`/`second_plat_*`/`decay_*` keys to exist (they are read unconditionally at train_mara.py:265-270 — missing keys raise AttributeError).
- **Interacts with:** `max_lr`, `min_lr`, `warmup_steps`, `max_steps` (all schedules); `restart_steps`, `restart_gamma` (only `restarts`); the `first_plat_*`/`second_plat_*`/`decay_to_*_pct` block (only `plateau`); `ffn_pdr_controller.reference.mode: auto`, which forbids `restarts` with any post-anchor restart (train_mara.py:5174-5183).
- **Example:** `lr_schedule_type: "cosine"` (from configs/dn4.yaml:70)

### `max_lr`
- **Type:** float
- **Default:** required — there is no default in `Settings.__init__`. It is read unconditionally when building the optimizer (`learning_rate=settings.max_lr`, train_mara.py:6174) and on every `get_lr` call (train_mara.py:252/257/263), so a config missing `max_lr` fails at construction/first-step. (The `min_lr` derivation at train_mara.py:4944 also reads `self.max_lr`, but only *conditionally* — inside the `if not hasattr(self, 'min_lr') or self.min_lr is None:` guard — so it is not the read that makes `max_lr` required.) Every real config sets it.
- **Values/constraints:** Positive float; no explicit range validation. Observed values span ~3.0e-4 (dn3/dn4/keel-moe) up to 4.5e-3 (fathom/machete/mf-* "paper" runs). Interpreted as the *peak* LR after warmup.
- **What it does:** The peak of the schedule. During warmup, LR ramps linearly to `max_lr` (`max_lr * (it+1)/warmup_steps`, train_mara.py:888); afterward cosine decays from `max_lr` toward `min_lr` (train_mara.py:904). It is also passed straight into `configure_optimizers` as `learning_rate=settings.max_lr` (train_mara.py:6174), so it sets the optimizer's base LR that all param groups and the body-LR controller scale from. Under `restarts`, per-cycle peaks are `max_lr * gamma**idx` (train_mara.py:902).
- **Interacts with:** `min_lr` (defaults to `max_lr*0.1` if unset); `warmup_steps` (ramp target); `restart_gamma` (decays the per-restart peak); `plateau` plateau levels are typically expressed as fractions of `max_lr`. The kv2/kv3 configs warn *not* to lower global `max_lr` as a body-growth lever because it also drops the Adam groups (kv2.yaml:124, kv3.yaml:82).
- **Example:** `max_lr: 3.0e-04` (from configs/dn4.yaml:71)

### `min_lr`
- **Type:** float
- **Default:** `max_lr * 0.1` — set in `Settings.__init__` when the attribute is absent or `None` (train_mara.py:4943-4944). So omitting it yields a 10% floor.
- **Values/constraints:** Non-negative float; no explicit range check. Real values range from a true floor (`3.0e-05`, i.e. 10% of max, in dn3/dn4) down to near-zero (`1.0e-07` in keel_repro.yaml:100). Comments in configs describe both "1% floor" and "10% floor (Moonlight)" conventions depending on the run.
- **What it does:** The asymptotic LR floor. The cosine term interpolates `min_lr + coeff*(cycle_peak_lr - min_lr)` where `coeff` runs 1→0 over the cycle (train_mara.py:899-904), so LR lands exactly on `min_lr` at `max_steps` (or at each `cycle_end` under restarts). Also the floor the dual-plateau schedule decays to in its final phase.
- **Interacts with:** `max_lr` (its default multiplier and the interpolation ceiling); `max_steps` (where the floor is reached); `restart_gamma` (with γ<1 the per-cycle peak can approach `min_lr`).
- **Example:** `min_lr: 3.0e-05` (from configs/dn4.yaml:72)

### `warmup_steps`
- **Type:** int
- **Default:** No assignment in `Settings.__init__`, but some call sites read it defensively as `int(getattr(settings, 'warmup_steps', 0) or 0)` (e.g. the LR-schedule fingerprint at train_mara.py:230), so the effective default is `0` (no warmup) when omitted. Note the main `get_lr` path passes `settings.warmup_steps` directly (train_mara.py:252/257/263) and the summary/plot code reads `settings.warmup_steps` directly (train_mara.py:965-995), so it would AttributeError if truly absent — in practice every config sets it, typically 1000-2500.
- **Values/constraints:** Non-negative int. `0` disables warmup (the ramp branch `it < warmup_steps` is never entered, so the `/warmup_steps` divide never executes and 0 is safe). With `warmup_steps > it` the ramp formula is used (train_mara.py:887-888). No upper-bound validation, but it should be well below `max_steps`.
- **What it does:** Length of the linear LR ramp at the very start. For `it < warmup_steps`, LR = `max_lr * (it+1)/warmup_steps` (train_mara.py:887-888); after that the cosine (or plateau) decay begins, with the first cosine cycle starting at `cycle_start = warmup_steps` (train_mara.py:892). Also gates other warmup-linked behavior indirectly (e.g. clip uses `clip_warmup` while `step < settings.warmup_steps`, train_mara.py:2063).
- **Interacts with:** `max_lr` (ramp target); `clip_warmup`/`clip_standard` (clip switches at `warmup_steps`, train_mara.py:2063); `restart_steps` (the first cycle spans `warmup_steps → first restart`); distinct from `health_guard_warmup_steps` and `new_layer_warmup_steps`, which are unrelated warmup windows.
- **Example:** `warmup_steps: 2000` (from configs/dn4.yaml:73)

### `restart_steps`
- **Type:** list (of int global steps) — defaults to an empty tuple; consumers coerce via `tuple(... or ())` and `[int(s) for s in ...]`.
- **Default:** `()` (empty) — set in `Settings.__init__` when absent (train_mara.py:4947-4948). No config in this repo currently sets it, so it is documented from code.
- **Values/constraints:** Iterable of ascending global step numbers. Consumed with `bisect.bisect_right` (train_mara.py:891), which assumes the list is sorted ascending — unsorted input yields wrong cycle boundaries. Only has an effect under `lr_schedule_type: restarts`; the `cosine` branch never passes it (train_mara.py:256-259). Can also be auto-populated: if `auto_restart_points: true`, it is overwritten with `[10%, 25%, 50%]` of `max_steps` (train_mara.py:4897-4902). Under `ffn_pdr_controller.reference.mode: auto`, any restart at or after `anchor_step` is a fatal error (train_mara.py:5174-5183).
- **What it does:** Global steps at which the cosine curve resets to a (possibly γ-decayed) peak and begins a fresh cosine decay. `get_lr_with_restarts` locates the active cycle from these boundaries: `cycle_start` = previous restart (or `warmup_steps`), `cycle_end` = next restart (or `max_steps`), then does a per-cycle cosine (train_mara.py:891-904). The train loop also logs a "Warm restart" line when `step in settings.restart_steps` (train_mara.py:2838-2839).
- **Interacts with:** `lr_schedule_type` (only active for `restarts`); `restart_gamma` (peak decay per restart, indexed by how many restarts have passed); `auto_restart_points` (auto-fills this list); `warmup_steps`/`max_steps` (cycle endpoints); `ffn_pdr_controller.reference.mode: auto` (forbids post-anchor restarts).
- **Example:** `restart_steps: [10000, 25000, 50000]` (illustrative — no in-repo config sets this; equivalent to `auto_restart_points: true` when `max_steps: 100000`)

### `restart_gamma`
- **Type:** float
- **Default:** `1.0` — set in `Settings.__init__` when absent (train_mara.py:4949-4950). No config in this repo sets it; documented from code.
- **Values/constraints:** Positive float, typically `0 < gamma <= 1`. `1.0` → every restart returns to full `max_lr`; `< 1.0` → each successive peak is smaller (diminishing warm restarts); no explicit range validation. Only used under `lr_schedule_type: restarts` (the `cosine` branch doesn't pass it, train_mara.py:256-259).
- **What it does:** Multiplicative decay applied to the restart peak. The peak of cycle `idx` is `cycle_peak_lr = max_lr * (gamma ** idx)`, where `idx = bisect.bisect_right(restart_steps, it)` is the number of restarts already passed (train_mara.py:891, 902). So with γ<1 the 1st restart peaks at `max_lr*γ`, the 2nd at `max_lr*γ²`, etc.; the per-cycle cosine still decays toward `min_lr`.
- **Interacts with:** `restart_steps` (the peaks it decays); `lr_schedule_type: restarts` (only path that reads it); `max_lr` (the base peak) and `min_lr` (the floor each cycle decays to).
- **Example:** `restart_gamma: 0.8` (illustrative — no in-repo config sets this; default `1.0` gives full-reset restarts)

The following settings apply when `lr_schedule_type: plateau`, selecting the dual-plateau LR schedule (warmup → cosine-decay to first plateau → hold → cosine-decay to second plateau → hold → final cosine-decay to `min_lr`); `auto_restart_points` is a separate convenience flag for the `restarts` schedule.

### `first_plat_lr`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.first_plat_lr` at train_mara.py:265 with no code fallback)
- **Values/constraints:** an absolute learning-rate value, typically between `min_lr` and `max_lr`. Not validated in code.
- **What it does:** The learning rate held during the first plateau. In Phase 2 the LR cosine-decays from `max_lr` down to `first_plat_lr` (train_mara.py:934-937), Phase 3 holds it flat at `first_plat_lr` (train_mara.py:940-941), then Phase 4 cosine-decays from `first_plat_lr` to `second_plat_lr` (train_mara.py:943-947). It is an absolute LR, not a fraction of `max_lr`.
- **Interacts with:** `lr_schedule_type` (plateau), `max_lr` (decay-from target and the % reported in logs, train_mara.py:980), `second_plat_lr`, `decay_to_first_plat_pct`, `first_plat_len_pct`.
- **Example:** `first_plat_lr: 0.5e-03` (from configs/machete.yaml)

### `decay_to_first_plat_pct`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.decay_to_first_plat_pct` at train_mara.py:266)
- **Values/constraints:** a fraction of `max_steps` in [0, 1]. Combined with `warmup_steps/max_steps`, `first_plat_len_pct`, `decay_to_second_pct`, and `second_plat_len_pct`, the sum should be < 1 so a positive final-decay remainder is left (train_mara.py:990-992). Not validated in code.
- **What it does:** The length of Phase 2 (cosine decay from `max_lr` to `first_plat_lr`), expressed as a fraction of `max_steps`. The phase runs from `warmup_steps` to `warmup_steps + int(decay_to_first_plat_pct * max_steps)` (train_mara.py:924).
- **Interacts with:** `lr_schedule_type` (plateau), `max_steps`, `warmup_steps`, `first_plat_lr`, `max_lr`.
- **Example:** `decay_to_first_plat_pct: 0.08` (from configs/machete.yaml)

### `first_plat_len_pct`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.first_plat_len_pct` at train_mara.py:267)
- **Values/constraints:** a fraction of `max_steps` in [0, 1]; see the sum constraint under `decay_to_first_plat_pct`. Not validated in code.
- **What it does:** The length of Phase 3, the flat hold at `first_plat_lr`, as a fraction of `max_steps`. The plateau spans from `decay_to_first_end` to `decay_to_first_end + int(first_plat_len_pct * max_steps)` (train_mara.py:925, 939-941).
- **Interacts with:** `lr_schedule_type` (plateau), `max_steps`, `first_plat_lr`, `decay_to_first_plat_pct`.
- **Example:** `first_plat_len_pct: 0.50` (from configs/machete.yaml)

### `decay_to_second_pct`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.decay_to_second_pct` at train_mara.py:268)
- **Values/constraints:** a fraction of `max_steps` in [0, 1]; see the sum constraint under `decay_to_first_plat_pct`. Not validated in code.
- **What it does:** The length of Phase 4 (cosine decay from `first_plat_lr` down to `second_plat_lr`), as a fraction of `max_steps`. The phase runs from `first_plat_end` to `first_plat_end + int(decay_to_second_pct * max_steps)` (train_mara.py:926, 943-947).
- **Interacts with:** `lr_schedule_type` (plateau), `max_steps`, `first_plat_lr`, `second_plat_lr`, `first_plat_len_pct`.
- **Example:** `decay_to_second_pct: 0.10` (from configs/machete.yaml)

### `second_plat_lr`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.second_plat_lr` at train_mara.py:269)
- **Values/constraints:** an absolute learning-rate value, typically between `min_lr` and `first_plat_lr`. Not validated in code.
- **What it does:** The learning rate held during the second plateau. Phase 4 cosine-decays into it, Phase 5 holds flat at `second_plat_lr` (train_mara.py:949-951), and Phase 6 cosine-decays from `second_plat_lr` down to `min_lr` (train_mara.py:953-956). It is an absolute LR, not a fraction of `max_lr`.
- **Interacts with:** `lr_schedule_type` (plateau), `min_lr` (final-decay target), `first_plat_lr`, `decay_to_second_pct`, `second_plat_len_pct`, `max_lr` (the % reported in logs, train_mara.py:981).
- **Example:** `second_plat_lr: 2.0e-04` (from configs/machete.yaml)

### `second_plat_len_pct`
- **Type:** float
- **Default:** required (when `lr_schedule_type: plateau`; read unconditionally as `settings.second_plat_len_pct` at train_mara.py:270)
- **Values/constraints:** a fraction of `max_steps` in [0, 1]; see the sum constraint under `decay_to_first_plat_pct`. Not validated in code.
- **What it does:** The length of Phase 5, the flat hold at `second_plat_lr`, as a fraction of `max_steps`. The plateau spans from `decay_to_second_end` to `decay_to_second_end + int(second_plat_len_pct * max_steps)` (train_mara.py:927, 949-951); the remaining steps (Phase 6) cosine-decay to `min_lr`, so the final-decay length is the leftover `1 - warmup% - all four plateau/decay percentages` (train_mara.py:990-992).
- **Interacts with:** `lr_schedule_type` (plateau), `max_steps`, `second_plat_lr`, `min_lr`, `decay_to_second_pct`.
- **Example:** `second_plat_len_pct: 0.20` (from configs/machete.yaml)

### `auto_restart_points`
- **Type:** bool
- **Default:** `False` (read via `getattr(settings, 'auto_restart_points', False)` at train_mara.py:4897)
- **Values/constraints:** `true` or `false`.
- **What it does:** A convenience flag for the `restarts` LR schedule. When `true`, it overwrites `settings.restart_steps` with `[10%, 25%, 50%] * max_steps`, i.e. `[int(max_steps*0.10), int(max_steps*0.25), int(max_steps*0.50)]` (train_mara.py:4897-4902), so you get three evenly-staged LR restarts without hand-listing step numbers. When `false` (default), your explicit `restart_steps` list is used as-is (defaulting to `()` if unset, train_mara.py:4947-4948). Only meaningful under `lr_schedule_type: restarts`, since `plateau`/`cosine` ignore `restart_steps`.
- **Interacts with:** `restart_steps` (which it overwrites), `max_steps`, `lr_schedule_type` (restarts), `restart_gamma`.
- **Example:** `auto_restart_points: true`

---

## Optimizer (Muon / NorMuon / AdamC / AdamW families)

The optimizer is selected by a single string field, `optimizer_type`, which fans out to one of four families (Adam, Adafactor, FSDP2-native Muon, or Microsoft DION) in `configure_optimizers.py`. The settings below are the shared knobs across those families; note that none of them are given an explicit default in `Settings.__init__` — their defaults come entirely from the `getattr(settings, 'X', default)` fallbacks at the call site in `train_mara.py:6170-6192` and inside `configure_optimizers.py`. Family-specific extras (`muonsphere_radius_scale`, `dion_kwargs`, `adafactor_beta2`, `adam16bit_state_dtype`, `cautious_weight_decay`, etc.) are visible in `configs/_optimizer_reference.yaml` and are noted under "Interacts with" where relevant.

### `optimizer_type`
- **Type:** string
- **Default:** `"adamw"` (set at `train_mara.py:4998-4999` when the key is absent)
- **Values/constraints:** Must be one of `VALID_OPTIMIZER_TYPES` (`configure_optimizers.py:19-28`), else `fatal_error` at `train_mara.py:5001-5005` (and a second guard, `raise ValueError`, in `configure_optimizers.py:302-306`). The full set, grouped:
  - **Adam family:** `adamw`, `adamw_8bit`, `adamw_16bit`, `adamc`, `adamc_8bit`, `adamc_16bit`
  - **Adafactor:** `adafactor`
  - **`DION_FAMILY`** (Microsoft DION package, `configure_optimizers.py:34`): `muon_dion`, `normuon_dion`, `dion2`
  - **`FSDP2_MUON_FAMILY`** (samsja's FSDP2-native impl, `configure_optimizers.py:35`): `muon_fsdp2`, `normuon_fsdp2`, `muonsphere_fsdp2`, `normuon_sphere_fsdp2`
  - **`MUON_FAMILY`** (`configure_optimizers.py:30-33`) = DION_FAMILY ∪ FSDP2_MUON_FAMILY (all six Muon variants).
  - Two more internal sub-sets: `NORMUON_VARIANTS = {normuon_dion, normuon_fsdp2, normuon_sphere_fsdp2}` (`configure_optimizers.py:36`) and `MUONSPHERE_VARIANTS = {muonsphere_fsdp2, normuon_sphere_fsdp2}` (`configure_optimizers.py:37`) gate `normuon_beta2` and MuonSphere params respectively.
- **What it does:** Chooses the optimizer construction path. Legacy boolean flags (`use_muon`, `use_adamc`, `use_8bit_adam`, `use_adafactor`) are hard-rejected with a migration message (`train_mara.py:4985-4996`). For all Muon-family types, only 2D attention/FFN weight matrices (`wq/wk/wv/wo/w1/w2/w3`, GDN `q_proj/k_proj/v_proj/o_proj/g_proj`, and aux-head Linears) go to the Muon update; embeddings, output head, norms, expert 3D weights, and router gates fall back to an embedded Adam group (`configure_optimizers.py:322-357, 429-447`).
- **Interacts with:** Gates almost everything in this section: `muon_momentum`/`muon_ns_steps`/`muon_adam_state_dtype` (FSDP2 + DION Muon), `normuon_beta2` (NorMuon variants only), `muonsphere_radius_scale`/`muonsphere_power_iters` (MuonSphere variants), `dion_kwargs`/`dion2_fraction`/`dion2_ef_decay` (DION), `adafactor_beta2` (adafactor), `adam16bit_state_dtype` (adamw_16bit/adamc_16bit), `cautious_weight_decay` (Adam-only — raises if set with any MUON_FAMILY type, `configure_optimizers.py:308-313`). `scs_enabled` requires an FSDP2_MUON_FAMILY type (`train_mara.py:1620`); DION types trigger an all-to-all dimension-divisibility check when multi-GPU (`train_mara.py:4206`) and have `lr_mods` disabled (`train_mara.py:6242`).
- **Example:** `optimizer_type: normuon_fsdp2` (from `configs/dn3.yaml:70`; `configs/7B-MAX.yaml:68` uses `adamc_8bit`)

### `beta1`
- **Type:** float
- **Default:** `0.9` (via `getattr(settings, 'beta1', 0.9)`, `train_mara.py:6175`; same fallback in `summarize_optimizer_settings`, `configure_optimizers.py:56`)
- **Values/constraints:** No explicit validation or clamp in code; must be a valid Adam first-moment coefficient (0 ≤ β1 < 1). Combined with `beta2` into `betas=(beta1, beta2)` passed to every Adam construction.
- **What it does:** First-moment (momentum) EMA decay for all Adam-family optimizers AND the embedded Adam group inside every Muon/DION optimizer (embeddings, output head, norms, expert/router params). It is NOT the Muon momentum — that is `muon_momentum`. Feeds `betas` into `torch.optim.AdamW`, TorchAO 8-bit, `AdamW16bit`, `AdamC`/`AdamC8bitTorchAO`/`AdamC16bit`, and the Adam param groups of MuonFSDP2/DION (`configure_optimizers.py:393, 405, 414, 485, 495, 582, 590, 597, 630, 642, 653`).
- **Interacts with:** `beta2` (paired as `betas`); `optimizer_type` (which optimizer consumes it). Ignored by `adafactor` (which uses `adafactor_beta2`) and does not affect the Muon orthogonalized update itself.
- **Example:** `beta1: 0.9` (from `configs/dn3.yaml:71`)

### `beta2`
- **Type:** float
- **Default:** `0.95` (via `getattr(settings, 'beta2', 0.95)`, `train_mara.py:6175`; same in `configure_optimizers.py:56`)
- **Values/constraints:** No explicit validation; standard Adam second-moment coefficient (0 ≤ β2 < 1). Note the default is `0.95`, not the more common `0.999`.
- **What it does:** Second-moment (variance) EMA decay for all Adam-family optimizers and the embedded Adam group of Muon/DION optimizers. Bundled with `beta1` into `betas`. On checkpoint resume, `betas` is force-reset from the current config into every Adam param group (`train_mara.py:3779`, `param_group['betas'] = (settings.beta1, settings.beta2)`), so changing it in the YAML takes effect on resume. Distinct from NorMuon's `normuon_beta2`, which controls the Muon-update second moment.
- **Interacts with:** `beta1` (paired). Does not affect the raw Muon update; NorMuon's per-neuron second moment is `normuon_beta2`, and Adafactor's is `adafactor_beta2`.
- **Example:** `beta2: 0.95` (from `configs/dn3.yaml:72`)

### `muon_momentum`
- **Type:** float
- **Default:** `0.95` (via `getattr(settings, 'muon_momentum', 0.95)`, `train_mara.py:6177`; MuonFSDP2 also defaults `group["momentum"]` to `0.95` if absent, `muon_fsdp2.py:606`)
- **Values/constraints:** No explicit validation/clamp; effective momentum coefficient, 0 ≤ β < 1. Applied via `momentum.lerp_(grad, 1-β)` then `grad.lerp_(momentum, β)` with Nesterov (`apply_momentum`, `muon_fsdp2.py:112-117`).
- **What it does:** Momentum coefficient for the Muon update path (pre–Newton-Schulz). Used by both FSDP2 Muon (passed as `momentum=` into every Muon param group, `configure_optimizers.py:466`, with `nesterov=True`) and DION Muon/NorMuon (passed as `mu=`, `configure_optimizers.py:413`). Not used for Adam params — those use `beta1`. For DION `dion2` it is ignored (dion2 takes `betas` instead, `configure_optimizers.py:400-408`). This is the classic Muon momentum on the accumulated gradient before orthogonalization.
- **Interacts with:** `optimizer_type` (only meaningful for MUON_FAMILY; for the FSDP2 Muon groups the Nesterov path is hardcoded `nesterov=True` at `configure_optimizers.py:469`). Independent of `beta1`.
- **Example:** `muon_momentum: 0.95` (from `configs/dn3.yaml:73`)

### `muon_ns_steps`
- **Type:** int
- **Default:** `5` (via `getattr(settings, 'muon_ns_steps', 5)`, `train_mara.py:6178`; MuonFSDP2 group default also `5`, `muon_fsdp2.py:610`)
- **Values/constraints:** No hard validation; must be a positive int. `5` is documented as "accurate", `3` as "faster" (`_optimizer_reference.yaml:49`). A separate read at `train_mara.py:2528` (a telemetry/probe path) coerces with `int(... or 5)`.
- **What it does:** Number of quintic Newton-Schulz iterations used to orthogonalize the momentum-buffered gradient (`zeropower_via_newtonschulz5(G, steps)`, `muon_fsdp2.py:87-110`, invoked at `muon_fsdp2.py:388`). More steps → closer to a true orthogonal `US'V^T` factor (coefficients tuned to maximize slope at zero, so it converges to `S'≈Uniform(0.5,1.5)` rather than exact `UV^T`). Passed as `ns_steps=` into the Muon param groups; DION variants do not consume this (they orthogonalize internally). Only affects FSDP2 Muon family.
- **Interacts with:** `optimizer_type` (FSDP2_MUON_FAMILY only). The NS step count is the dominant compute cost of the Muon update; unrelated to Adam-family settings.
- **Example:** `muon_ns_steps: 5` (from `configs/dn3.yaml:74`)

### `normuon_beta2`
- **Type:** float
- **Default:** `0.95` (via `getattr(settings, 'normuon_beta2', 0.95)`, `train_mara.py:6179`; MuonFSDP2 group default `0.95`, `muon_fsdp2.py:612`; the docstring at `configure_optimizers.py:281` also says default `0.95`)
- **Values/constraints:** No explicit clamp; 0 ≤ β2 < 1. Config comments characterize the window: `0.95` ≈ aggressive ~20-step window, `0.99` ≈ moderate ~100, `0.999` ≈ conservative ~1000 (`configs/7B-MUON.yaml:85`, `_optimizer_reference.yaml:52`). Only consumed when `optimizer_type` is a NorMuon variant (`normuon_dion`, `normuon_fsdp2`, `normuon_sphere_fsdp2`); silently inert otherwise.
- **What it does:** Second-moment EMA decay for NorMuon's neuron-wise normalization, applied AFTER Newton-Schulz. In `apply_normuon` (`muon_fsdp2.py:222-237`) it maintains a per-neuron running mean of squared update (`second_momentum.lerp_(v_mean, 1-β2)`), then divides the update by `sqrt(second_momentum)+1e-10` to normalize each output neuron's step magnitude, and finally rescales the whole update so its Frobenius norm matches the pre-normalization value (lines 235-236). Passed as `beta2=` into the Muon param group (`configure_optimizers.py:473`) and gated by `use_normuon` (invoked at `muon_fsdp2.py:398-399`). Note the real configs set this to `0.99` even though the code default is `0.95` — so relying on the default gives a much shorter averaging window than production runs use.
- **Interacts with:** `optimizer_type` (NorMuon variants only, `NORMUON_VARIANTS`). Distinct from `beta2` (that is the Adam-group second moment). Not the same window as `muon_momentum`.
- **Example:** `normuon_beta2: 0.99` (from `configs/dn3.yaml:75`)

### `muon_adam_state_dtype`
- **Type:** string
- **Default:** `"fp32"` (via `getattr(settings, 'muon_adam_state_dtype', 'fp32')`, `train_mara.py:6191`; MuonFSDP2 constructor default `"fp32"`, `muon_fsdp2.py:588`)
- **Values/constraints:** Must be one of `VALID_ADAM_STATE_DTYPES = {"fp32", "mixed", "fp16", "bf16"}` (`muon_fsdp2.py:17`), else `ValueError` in the MuonFSDP2 constructor (`muon_fsdp2.py:589-593`). Any value other than `"fp32"` flips `_use_16bit_adam=True` (`muon_fsdp2.py:595`) and enables the half-precision Adam-state path with stochastic rounding (`muon_fsdp2.py:732-734`). Only applies to FSDP2 Muon family (passed as `adam_state_dtype=` at `configure_optimizers.py:525`); ignored by DION and Adam families.
- **What it does:** Sets the storage precision of the exp_avg / exp_avg_sq Adam states for the NON-Muon params (embeddings, output head, norms, expert/router weights) inside the FSDP2 Muon optimizer. Mapping (`_get_adam_state_dtype`, `muon_fsdp2.py:184-198`): `fp32`→both FP32; `mixed`→FP16 exp_avg + BF16 exp_avg_sq (recommended 16-bit); `fp16`→both FP16 (exp_avg_sq can underflow); `bf16`→both BF16 (safe range, less mantissa). The 16-bit modes save ~50% of the embedded-Adam optimizer memory at the cost of precision.
- **Interacts with:** `optimizer_type` (FSDP2_MUON_FAMILY only). Hard constraint: `head_gauge_projection` requires `muon_adam_state_dtype: fp32` (the non-fused Adam path — any other value selects the fused 16-bit path where the applied update is never exposed), enforced at `train_mara.py:5532-5536` — see `configs/dn4.yaml:89`. Analogous to `adam16bit_state_dtype` for the standalone `adamw_16bit`/`adamc_16bit` optimizers. Does not affect Muon param states (those are fp32 momentum buffers regardless).
- **Example:** `muon_adam_state_dtype: fp32` (from `configs/dn3.yaml:76`; `configs/dreadnought.yaml:100` shows a commented `mixed` alternative)

# Optimizer-family config knobs

All of these are read via `getattr(settings, ...)` in the `configure_optimizers(...)` call
at `train_mara.py:6170-6192`, then consumed inside `configure_optimizers.py`. One knob
(`muon_rms_scale`) is NOT wired into the live optimizer — it is read only by the in-situ
stage-trace probe at `train_mara.py:2530` (see its entry). Constants for the optimizer
families live at `configure_optimizers.py:19-37`:
- `FSDP2_MUON_FAMILY = {muon_fsdp2, normuon_fsdp2, muonsphere_fsdp2, normuon_sphere_fsdp2}`
- `MUONSPHERE_VARIANTS = {muonsphere_fsdp2, normuon_sphere_fsdp2}`
- `NORMUON_VARIANTS = {normuon_dion, normuon_fsdp2, normuon_sphere_fsdp2}`
- `DION_FAMILY = {muon_dion, normuon_dion, dion2}`
- `MUON_FAMILY` = FSDP2_MUON_FAMILY ∪ DION_FAMILY (Muon-based types)

---

### `muonsphere_radius_scale`
- **Type:** float
- **Default:** `2.0` (`getattr` default at `train_mara.py:6184`, param default `configure_optimizers.py:239`)
- **Values/constraints:** Positive float. The spectral-sphere constraint radius scale `c` (arXiv 2601.08393). Only consumed when `optimizer_type` is a MuonSphere variant.
- **What it does:** Passed as `radius_scale=` into the FSDP2 Muon param group (`configure_optimizers.py:476`) and forwarded to the `muon_fsdp2.Muon` optimizer, where it sets the radius of the spectral-sphere retraction that replaces weight decay for MuonSphere variants. Larger `c` → larger allowed spectral radius before retraction pulls the update back. Only applied when `use_muonsphere` is true, i.e. `optimizer_type in {muonsphere_fsdp2, normuon_sphere_fsdp2}` (`configure_optimizers.py:450`, `129-135`); ignored for all other optimizer types.
- **Interacts with:** MuonSphere retraction replaces weight decay — `muon_weight_decay` is forced to `0.0` for these variants (`configure_optimizers.py:453`), and the summary logs "Weight decay = DISABLED (spectral retraction regularizes)" (`configure_optimizers.py:135`). Pairs with `muonsphere_power_iters`.
- **Example:** `muonsphere_radius_scale: 2.0`

### `muonsphere_power_iters`
- **Type:** int
- **Default:** `10` (`getattr` default at `train_mara.py:6185`, param default `configure_optimizers.py:240`)
- **Values/constraints:** Positive int. Number of power-iteration steps used to estimate the spectral norm for the sphere retraction. Only consumed by MuonSphere variants.
- **What it does:** Passed as `power_iters=` into the FSDP2 Muon param group (`configure_optimizers.py:477`) and forwarded to `muon_fsdp2.Muon`, controlling how many power iterations estimate the top singular value used by the spectral retraction. More iterations → more accurate spectral estimate at higher compute cost. Only takes effect when `optimizer_type in MUONSPHERE_VARIANTS` (`configure_optimizers.py:450`, logged at `132-134`); a no-op otherwise.
- **Interacts with:** Companion to `muonsphere_radius_scale`; both are only active for the MuonSphere variants, which disable weight decay.
- **Example:** `muonsphere_power_iters: 10`

### `muon_rms_scale`
- **Type:** bool
- **Default:** `False` (`getattr` default at `train_mara.py:2530`)
- **Values/constraints:** `True`/`False`. NOTE: this setting is PROBE-ONLY. It is read ONCE, at `train_mara.py:2530`, inside the in-situ NorMuon stage-trace diagnostic (`_rmss = bool(getattr(settings, 'muon_rms_scale', False))`), which replays `apply_scaling` on the warm momentum buffer to locate the radial-bias source. It does NOT configure the live optimizer.
- **What it does:** Selects which branch of `muon_fsdp2.apply_scaling(grad, rms_scale)` the diagnostic replays (`muon_fsdp2.py:119-127`). When `True`, uses the Moonlight RMS scaling `grad *= 0.2 * sqrt(max(rows, cols))`; when `False`, uses the Keller-Jordan scaling `grad *= max(1, rows/cols)**0.5`. IMPORTANT: the LIVE FSDP2 Muon param group hardcodes `rms_scale=True` at `configure_optimizers.py:471` regardless of this setting — so for the trace to match the real update path, `muon_rms_scale` should be set to `True`. Setting it does not change training behavior, only what the probe reports.
- **Interacts with:** Only meaningful alongside the in-situ probe (`_insitu`) path and `muon_ns_steps`/`normuon_beta2` also re-read there (`train_mara.py:2528-2529`). To reflect the production optimizer, mirror the hardcoded `rms_scale=True`.
- **Example:** `muon_rms_scale: true`

### `cautious_weight_decay`
- **Type:** bool
- **Default:** `False` (`getattr` default at `train_mara.py:6183`, param default `configure_optimizers.py:237`)
- **Values/constraints:** `True`/`False`. VALID FOR ADAM-FAMILY ONLY. Setting `True` with any Muon-family `optimizer_type` raises `ValueError` at `configure_optimizers.py:308-313` (Newton-Schulz orthogonalization destroys the coordinate-wise structure the CWD mask relies on).
- **What it does:** Cautious Weight Decay — only decay a weight coordinate where momentum and weight agree in sign. `cautious_weight_decay` is passed through to `configure_optimizers` (`train_mara.py:6183`); the guard at `configure_optimizers.py:308` rejects it for `MUON_FAMILY`. NOTE: inside the FSDP2 Muon param groups the flag is hardcoded to `cautious_weight_decay=False` (`configure_optimizers.py:474,489,499,509,519`), consistent with the guard. `summarize_optimizer_settings` logs "Cautious WD = ENABLED" when set (`configure_optimizers.py:204-206`).
- **Interacts with:** Mutually exclusive with Muon-family optimizers. Intended for the Adam/AdamC/AdamW families.
- **Example:** `cautious_weight_decay: false`

### `adafactor_beta2`
- **Type:** float or null
- **Default:** `None` (`getattr` default at `train_mara.py:6182`, param default `configure_optimizers.py:235`)
- **Values/constraints:** `None` = auto-schedule; otherwise a fixed float in (0,1). Only consumed when `optimizer_type == "adafactor"`.
- **What it does:** Passed as `beta2=` into the AdafactorFSDP2 constructor (`configure_optimizers.py:538`). When `None`, Adafactor uses its built-in decaying schedule `1 - step**-0.8` (logged "Beta2 = auto" at `configure_optimizers.py:144-145`); when set, pins the second-moment decay to a constant. The summary computes and prints a recommended value for a 10M-token half-life (`configure_optimizers.py:148-151`) to guide choosing a fixed value. Ignored for every non-Adafactor optimizer type.
- **Interacts with:** Only relevant with `optimizer_type: adafactor`. The recommended value depends on `B`, `T`, world size, and grad-accum (the tokens-per-step) at `configure_optimizers.py:148-150`.
- **Example:** `adafactor_beta2: 0.999`

### `dion_kwargs`
- **Type:** dict (mapping) or null
- **Default:** `None` (`getattr` default at `train_mara.py:6180`, param default `configure_optimizers.py:232`)
- **Values/constraints:** Nested YAML mapping of extra keyword args passed straight through to the Microsoft DION optimizer constructor (e.g. `nesterov`, `use_triton`). Only consumed when `optimizer_type in DION_FAMILY` (`muon_dion`, `normuon_dion`, `dion2`).
- **What it does:** Coalesced to `extra_kwargs = dion_kwargs or {}` (`configure_optimizers.py:373`) and merged into the DION constructor kwargs via `opt_kwargs.update(extra_kwargs)` (`configure_optimizers.py:419`) right before `DionOptimizer(param_groups, **opt_kwargs)` (`420`). Lets you set DION-specific options not surfaced as first-class settings. If non-empty it is echoed as "Extra kwargs = ..." in the summary (`configure_optimizers.py:87-89`). A no-op for all non-DION optimizer types.
- **Interacts with:** Only relevant for the DION family. Keys here can override earlier `opt_kwargs` entries (lr, mu/betas, weight_decay) because `.update()` runs last.
- **Example:**
  ```yaml
  dion_kwargs:
    nesterov: true
    use_triton: false
  ```

### `dion2_fraction`
- **Type:** float
- **Default:** `0.25` (`getattr` default at `train_mara.py:6188`, param default `configure_optimizers.py:245`)
- **Values/constraints:** Float in (0,1]. The submatrix selection fraction for DION2. Only consumed when `optimizer_type == "dion2"`.
- **What it does:** Passed as `fraction=` into the `Dion2` constructor (`configure_optimizers.py:403`). Controls what fraction of each weight matrix DION2 selects/updates per step (submatrix selection). Only referenced in the `dion2` branch of the DION-family kwargs (`configure_optimizers.py:400-408`); ignored for `muon_dion`/`normuon_dion` and all non-DION types.
- **Interacts with:** DION2-only; pairs with `dion2_ef_decay`. Can be overridden by a `fraction` key inside `dion_kwargs` (update runs after).
- **Example:** `dion2_fraction: 0.25`

### `dion2_ef_decay`
- **Type:** float
- **Default:** `0.95` (`getattr` default at `train_mara.py:6189`, param default `configure_optimizers.py:246`)
- **Values/constraints:** Float in (0,1). Error-feedback accumulator decay for DION2. Only consumed when `optimizer_type == "dion2"`.
- **What it does:** Passed as `ef_decay=` into the `Dion2` constructor (`configure_optimizers.py:404`). Sets the decay rate of DION2's error-feedback buffer, which carries the residual not applied by the submatrix selection forward to the next step. Higher decay retains error longer. Only referenced in the `dion2` branch (`configure_optimizers.py:400-408`); a no-op elsewhere.
- **Interacts with:** DION2-only; companion to `dion2_fraction`. Overridable via `dion_kwargs` (`ef_decay` key).
- **Example:** `dion2_ef_decay: 0.95`

### `adam16bit_state_dtype`
- **Type:** str
- **Default:** `"mixed"` (`getattr` default at `train_mara.py:6190`, param default `configure_optimizers.py:248`)
- **Values/constraints:** One of `{"mixed", "fp16", "bf16"}` — validated by `VALID_STATE_DTYPES` in `adamw_16bit.py:31`; an invalid value raises `ValueError` (`adamw_16bit.py:163-167`). Only consumed by the 16-bit Adam variants: `optimizer_type in {"adamw_16bit", "adamc_16bit"}`.
- **What it does:** Selects the precision of the Adam optimizer states. Passed as `state_dtype=` into `AdamW16bit` (`configure_optimizers.py:632`) and `AdamC16bit` (`configure_optimizers.py:584`), which map it in `_get_state_dtype` (`adamw_16bit.py:182-192`): `"mixed"` = FP16 exp_avg + BF16 exp_avg_sq (default), `"fp16"` = both FP16 (exp_avg_sq may underflow), `"bf16"` = both BF16 (safe range, less precision). Stochastic rounding is used on the BF16 states (`adamw_16bit.py`). Ignored for fp32/8-bit Adam and all Muon-family types (which instead use `muon_adam_state_dtype`).
- **Interacts with:** Sibling of `muon_adam_state_dtype` (`train_mara.py:6191`), which applies the same three-way precision choice to the embedded-Adam params inside Muon-family optimizers (default there is `"fp32"`). The chosen dtype is echoed in the optimizer summary (`configure_optimizers.py:163-164`, `176-177`).
- **Example:** `adam16bit_state_dtype: mixed`

---

## Weight Decay

Weight decay in this trainer is controlled by a single top-level key, `weight_decay`, which is polymorphic: it accepts either a scalar (uniform decay applied via optimizer param groups) or a *rules list* of `[target, value]` / `[start, end, value]` entries where each value can itself be a step-based schedule. The rules form is parsed by `parse_wd_rules` (train_mara.py:800) into per-parameter overrides that are written into the `wd_overrides` side-dict every step; the scalar form is passed straight into `configure_optimizers`. Norm/bias params are *always* forced to WD=0 regardless of form.

### `weight_decay`
- **Type:** float | list-of-rules (each rule value may be a scalar or a `[[step, val], ...]` schedule)
- **Default:** `0.0` (if the key is absent, `Settings.__init__` sets `self.weight_decay = 0.0` — train_mara.py:5016-5017)
- **Values/constraints:**
  - Must be an `int`/`float` or a `list`; anything else is a fatal error (train_mara.py:5015: `"weight_decay must be a number or list of rules, got <type>"`).
  - **Scalar form:** cast to `float` at normalization (train_mara.py:5010-5011). No sign/range check is performed — any float is accepted; a non-negative value (e.g. `0.002`) is the conventional/intended usage.
  - **Rules form:** a list of entries. Two entry shapes are accepted (train_mara.py:816-841):
    - `[name, value]` where `name` is one of the string targets `'emb'`, `'out'`, `'all'`.
    - `[start, end, value]` where `start`/`end` are integer layer indices (inclusive range).
    - `value` in either shape may be a scalar OR a schedule `[[step, val], [step, val], ...]`.
  - **Coverage guard (rules form only):** when `weight_decay` is a list, the optimizer's base WD is forced to `0.0`, so any non-norm param not matched by *some* rule would silently get WD=0. `parse_wd_rules` rejects this: every non-norm param must be covered by at least one rule or it's a `fatal_error` (train_mara.py:848-867), reporting which bucket (emb / head / body) is uncovered. An explicit `[emb, 0.0]` counts as coverage — only the *silent* rule-less zero is rejected.
  - **Norms/biases always WD=0:** `_is_norm(n)` (name ends in `bias`, or contains `norm` (case-insensitive) and ends in `weight`) params are skipped by every rule and never decayed (train_mara.py:813-814, 821, 839). In scalar mode the same exclusion is enforced by a dedicated `norm_bias` param group pinned at `weight_decay: 0.0` (configure_optimizers.py:573, 621).
- **What it does:**
  - **Scalar form:** the value is handed to `configure_optimizers(weight_decay=...)` (train_mara.py:6173) and lives on the optimizer param groups. No `wd_overrides` are populated. On **resume**, the restore path reconciles the saved param-group WD to the current scalar setting (train_mara.py:3766-3775): it iterates non-`norm_bias` groups once, updates `param_group['weight_decay']` if it changed, and breaks after the first (all non-norm groups share the same scalar). This is a one-time resume-time reconciliation, not a per-step live update.
  - **Rules form:** `parse_wd_rules` walks `model.named_parameters()` and, with *last-matching-rule-wins* semantics (train_mara.py:807, 841), maps each entry to the concrete params it targets, returning `(param, value_or_schedule)` tuples. `weight_decay` is then passed to the optimizer as `0.0` (train_mara.py:6173) and the real per-param values are written into the `wd_overrides` side-dict *every training step* (train_mara.py:2180-2185). If the value is a schedule list, it is linearly interpolated by step via `interpolate_lr_mod` (train_mara.py:553-565, 2183); scalar values are written verbatim (train_mara.py:2185). The optimizers (Muon/NorMuon body path + emb/head Adam paths under `normuon_fsdp2`) read `self.wd_overrides.get(id(p), group["weight_decay"])` at step time (muon_fsdp2.py:464, 528, 743, 797).
  - **Target semantics** (from the matcher, train_mara.py:823-841):
    - `emb` → params whose name contains `tok_embeddings` (input embedding table).
    - `out` → params whose name starts with `output.` (the LM head / output projection). Note: if `tie_word_embeddings=True` there is no `output.` param, so an `[out, ...]` rule matches nothing — cover the head via `emb` in that case.
    - `all` → every param whose name contains `layers.` (i.e. the transformer *body*: attention + FFN matrices — NOT emb/head). This is a common gotcha: `all` means "all body layers," not "all params." dn3.yaml:117-119 documents this explicitly.
    - `[start, end, value]` → body params in the inclusive layer-index range `start..end` (regex `layers.(\d+).`, train_mara.py:833-838); norms in-range still excluded.
- **Interacts with:**
  - **`adaptive_wd`** — AWD is initialized with the scalar `base_wd` (0.0 when rules are active, train_mara.py:6285) and the shared `wd_overrides` dict, and *multiplies* the per-param WD on top of the rule/scalar values each check interval (train_mara.py:6286, 2187-2191).
  - **`ffn_pdr_controller` / shadow-norm controller** — when engaged, the controller OWNS the FFN body-matrix WD via its radial-budget λ and writes into the same `wd_overrides` *after* the WD-rules and AWD writers (train_mara.py:2193-2201), silently overriding any WD schedule that targets FFN. `Settings.__init__` emits a loud footgun warning (a raw `print` banner) if you set an `all`/layer-range WD *schedule* while a shadow-mode controller is active (train_mara.py:5231-5253) — a flat base WD is fine (covers attention + pre-engagement window), but a schedule on FFN is silently replaced. Tune FFN WD via the controller knobs (rho, lambda_max, lambda_min) instead.
  - **`cautious_weight_decay`** — a separate optimizer flag (default `False`, train_mara.py:6183) that gates cautious-WD behavior inside the optimizer; orthogonal to the target/schedule mechanics here.
  - **`optimizer_type`** — under `normuon_fsdp2`, both the Muon body path and the emb/head Adam paths read `wd_overrides`, so all three targets (emb/out/all) take effect (dn3.yaml:118-119). Per project memory, plain AdamW/AdamW8bit/AdamW16bit do NOT read `wd_overrides`, so rules-form WD is silently ignored for those optimizers.
- **Example:**
  - Rules with a mid-run body taper (from `configs/dn3.yaml:120-123`):
    ```yaml
    weight_decay:
      - [emb, 0.02]
      - [out, 0.02]
      - [all, [[7000, 0.02], [12000, 0.002]]]   # body WD linearly annealed 0.02 -> 0.002 over steps 7000-12000
    ```
  - Rules with flat values (from `configs/dn4.yaml:82-85`):
    ```yaml
    weight_decay:
      - [emb, 0.02]
      - [out, 0.02]
      - [all, 0.002]
    ```
  - Uniform scalar form (from `configs/kv2.yaml:145` / `kv3.yaml:144`):
    ```yaml
    weight_decay: 0.002
    ```

---

## Per-Layer LR Mods & Output-Head LR

Two independent-but-coupled knobs for shaping learning rate below the global LR schedule: `lr_mods` applies hand-written per-target LR multipliers (embeddings, output head, attention/FFN per layer-range), and `output_lr_batch_adjust` auto-derives an output-head multiplier schedule from the grad-accumulation ramp. Both feed the same `lr_scale_overrides` side-dict the optimizer reads, and the head-LR paths are mutually exclusive by design.

### `lr_mods`
- **Type:** list (of rule entries; each rule ends in a `[[step, mult], ...]` waypoint schedule — a **list of `[step, mult]` pairs**, not a bare scalar)
- **Default:** `None` (`train_mara.py:5020-5021` — set to `None` if the key is absent; no LR modification)
- **Values/constraints:** Each entry is one of three shapes, parsed in `parse_lr_mods` (`train_mara.py:737-797`): `[name, schedule]` where `name` ∈ {`emb`, `out`} (emb → params whose name contains `tok_embeddings`; out → params whose name starts with `output.`); `[all, type, schedule]` where `type` ∈ {`attn`, `ffn`, `all`} across all layers (the literal first element is conventionally `all` and is otherwise ignored); `[start, end, type, schedule]` for an inclusive layer-index range. `type` matching (`_match_type`): `attn` = names containing `attention.` and not `norm`; `ffn` = names containing `feed_forward.`; `all` = attn OR ffn. Norms/biases are never scaled (the `[all, type]` path guards them via `_is_norm`, and in the range/`attn` paths norms simply don't match the type filter). The trailing `schedule` **must be a list of `[step, mult]` waypoints** — `interpolate_lr_mod` (`train_mara.py:553-565`) indexes `schedule[0][0]`, so a bare scalar multiplier is NOT supported here (unlike `weight_decay` rules, whose apply site at `train_mara.py:2182-2185` does `isinstance(wd_val, list)`; the lr_mods apply site at `train_mara.py:2086` has no such guard and would raise on a scalar). Multipliers are direct factors (1.0 = normal, 0.5 = half LR). Schedule waypoints are linearly interpolated and held flat before the first / after the last step. **Last matching rule wins** per param (`param_schedules` dict keyed by `id(param)`). Optimizer support (`train_mara.py:6241-6264`): full for the FSDP2 Muon family (`{muon_fsdp2, normuon_fsdp2, muonsphere_fsdp2, normuon_sphere_fsdp2}`); standalone Adam family gets `emb`/`out` rules plus any rule whose `type` is `all` (both the `[all, all, ...]` form and the `[start, end, all, ...]` range form) — attn/ffn-specific rules are logged and ignored because Adam groups can't differentiate within one param group; DION family is unsupported entirely (warning logged, whole feature ignored).
- **What it does:** Builds a per-parameter `scale` each step and writes it into `lr_scale_overrides[id(param)]` (`train_mara.py:2083-2087`). For Muon params the optimizer applies this scale in `Fsdp1dWork.finish` (`muon_fsdp2.py:454-455`, `462-463`) — **after** Newton-Schulz orthogonalization in `start()`. This matters: NS normalizes the update's magnitude, so scaling the *gradient* would be cancelled by NS; only a post-NS `lr_scale` actually changes Muon's step size (`effective_lr = group["lr"] * lr_scale`). The same `effective_lr` also multiplies the weight-decay term (`muon_fsdp2.py:463-477`), so a rule scaling a Muon param cools its WD by the same factor, and `lr_scale = 0.0` freezes the param entirely (no update, no WD-driven decay) — an invariant SCS relies on. For standalone Adam, scales are instead pushed onto the param-group `lr` (`train_mara.py:2089-2104`).
- **Interacts with:** `output_lr_batch_adjust` (auto-appends an `[out, schedule]` entry into `lr_mods` at `train_mara.py:5799-5801`; hard-conflicts with a manual `[out, ...]` rule — see below); SCS (runs after `lr_mods` at `train_mara.py:2106+` and can override managed params during scaffold/warmup, but defers to `lr_mods` when its own scale is 1.0); `ffn_pdr_controller` / `body_lr_controller` (writes the same `lr_scale_overrides` for FFN body params — kv3 replaced kv2's open-loop `lr_mods` FFN anneal with this closed-loop controller); `weight_decay` (shared effective-LR coupling on Muon). Not supported with DION.
- **Example:** (from `configs/kv2.yaml`, live multi-target use)
  ```yaml
  lr_mods:
    - [all, attn, [[4000, 0.776], [8000, 1.0]]]
    - [all, ffn,  [[0, 1.0], [1500, 1.0], [4290, 0.75], [5060, 0.67], [6580, 0.55],
                   [10000, 0.47], [15000, 0.42], [22000, 0.40]]]
  ```
  (from `configs/machete.yaml`, emb/out/range targets: `- [emb, [[0, 1.0],[2500, 0.9]]]`, `- [out, [[0, 1.0],[2500, 0.7]]]`, `- [1,79, all, [[0, 1.0], [2500, 0.85]]]`)

### `output_lr_batch_adjust`
- **Type:** dict (with fields `base_mult`, `exponent`, and optional `ref_batch`), or `None`
- **Default:** `None` (`train_mara.py:5026-5027` — `None` if the key is absent; no auto head-LR scaling)
- **Values/constraints:** When present, must be a dict (`train_mara.py:5030-5031`). Required fields `base_mult` and `exponent` must each be a number (int/float), else `fatal_error` (`train_mara.py:5032-5036`). Optional `ref_batch`, if given and non-`None`, must be a number (`train_mara.py:5037-5038`). **Hard conflict:** if `lr_mods` also contains an `[out, ...]` entry, training aborts with a `fatal_error` — `output_lr_batch_adjust` computes the `out` schedule automatically and fully replaces any manual `[out, ...]` rule (`train_mara.py:5039-5048`); remove one.
- **What it does:** At startup (`train_mara.py:5780-5801`, after the grad-accum schedule is expanded) it synthesizes an `[out, [[step, mult], ...]]` `lr_mods` entry so the output-head LR shrinks as the effective batch grows. Per-micro token count is `tok_per_micro = B * T * ddp_world_size`; iterating the expanded per-step schedule array, at each step where the grad-accum value `ga` changes it emits a waypoint with `mult = base_mult * (ref_batch / eff_batch) ** exponent` where `eff_batch = ga * tok_per_micro`. That entry is appended to `settings.lr_mods` (creating the list if `None`), then flows through the normal `lr_mods` path onto the head at full effective LR × mult. Rank 0 logs the full table (`train_mara.py:5803-5812`).
- **Interacts with:** `lr_mods` (injects into it; conflicts with a manual `[out, ...]` rule); `ga_schedule` / GA ramp (the config key is `ga_schedule`, expanded at runtime into the per-step `grad_accum_schedule` array this is derived from — head LR steps down each time GA steps up); `B`, `T`, world size (set `tok_per_micro`, hence `eff_batch` and the default `ref_batch`); SCS output-head freeze/warmup (SCS defers to this auto `[out]` rule only post-warmup so it isn't clobbered — `train_mara.py:2135-2144`). `head_gauge_projection` is the modern function-preserving alternative — dn4 removed `output_lr_batch_adjust` in favor of it (`configs/dn4.yaml:91-95`).
- **Example:** (from `configs/dn3.yaml`)
  ```yaml
  output_lr_batch_adjust:
    base_mult: 0.8
    exponent: 0.3
  ```

### `output_lr_batch_adjust.base_mult`
- **Type:** float (nested field of `output_lr_batch_adjust`)
- **Default:** required when `output_lr_batch_adjust` is set (`fatal_error` if missing — `train_mara.py:5032-5034`)
- **Values/constraints:** Must be a number (int/float), else `fatal_error` (`train_mara.py:5035-5036`). No range clamp; interpreted as a direct LR multiplier so typical values are ≤ 1.0 (a head-LR brake).
- **What it does:** The constant multiplier in `mult = base_mult * (ref_batch / eff_batch) ** exponent` (`train_mara.py:5787`, `5795`). At the reference batch (`eff_batch == ref_batch`) the head runs at exactly `base_mult × body LR`, so it also acts as a flat head-LR discount independent of the batch-ramp term. In the observed configs it's 0.8 (a 20% head-LR brake).
- **Interacts with:** `exponent`, `ref_batch` (all three combine in the one formula); `ga_schedule` / `grad_accum_schedule`.
- **Example:** `base_mult: 0.8` (from `configs/dn3.yaml`)

### `output_lr_batch_adjust.exponent`
- **Type:** float (nested field of `output_lr_batch_adjust`)
- **Default:** required when `output_lr_batch_adjust` is set (`fatal_error` if missing — `train_mara.py:5032-5034`)
- **Values/constraints:** Must be a number (int/float), else `fatal_error` (`train_mara.py:5035-5036`). No range check.
- **What it does:** The power on the batch-ratio term `(ref_batch / eff_batch) ** exponent` (`train_mara.py:5788`, `5795`). Since `eff_batch` grows above `ref_batch` as GA ramps up, `ref_batch/eff_batch < 1`, so a positive exponent monotonically shrinks the head-LR multiplier as the batch grows; `exponent = 0` disables the batch dependence (leaving only `base_mult`). Observed configs use 0.3.
- **Interacts with:** `base_mult`, `ref_batch`, `ga_schedule` / `grad_accum_schedule`.
- **Example:** `exponent: 0.3` (from `configs/dn3.yaml`)

### `output_lr_batch_adjust.ref_batch`
- **Type:** float / int, or `None` (optional nested field of `output_lr_batch_adjust`)
- **Default:** `None` → resolved at runtime to `grad_accum_schedule[0] * tok_per_micro`, i.e. the *starting* effective batch in tokens (`train_mara.py:5783-5786`)
- **Values/constraints:** If provided and non-`None`, must be a number (int/float), else `fatal_error` (`train_mara.py:5037-5038`). Interpreted in **tokens** (matches the `ga * B * T * world_size` units of `eff_batch`), not in sequences or micro-steps.
- **What it does:** The reference effective batch in the ratio `(ref_batch / eff_batch) ** exponent` (`train_mara.py:5786`, `5795`). It sets the pivot at which the head-LR multiplier equals `base_mult`; larger `ref_batch` pushes the whole schedule toward larger multipliers (less braking) at a given `eff_batch`. Left unset it auto-pins to the initial effective batch, so the multiplier starts at `base_mult` and only decays from there as GA ramps.
- **Interacts with:** `base_mult`, `exponent`; `grad_accum_schedule[0]`, `B`, `T`, world size (define the auto default and the `tok_per_micro` unit).
- **Example:** omitted (auto-derived) in `configs/dn3.yaml`; when set explicitly it takes a token count, e.g. `ref_batch: 524288`.

---

## Precision, FSDP, Compile & Gradient Clipping

These knobs control the numeric precision of compute vs. communication, how FSDP2 shards parameters, whether the model is `torch.compile`d, and the two-phase gradient-norm clip. Note on defaults: almost none of these are defaulted in `Settings.__init__` — they are set only by the YAML setattr loop (`train_mara.py:4936-4940`). Every real config sets them explicitly; if omitted, the code raises `AttributeError` at the read site rather than falling back to a default. The one exception is `reshard_after_forward`, which is read via `getattr(settings, 'reshard_after_forward', True)`.

### `data_type`
- **Type:** string
- **Default:** required (no default in `Settings.__init__`; read directly as `settings.data_type` at multiple sites, e.g. `train_mara.py:1934`)
- **Values/constraints:** `"bf16"` | `"fp16"` | anything else → `float32`. No validation — the string is mapped by a chained ternary `bf16 → torch.bfloat16, fp16 → torch.float16, else → torch.float32` (`train_mara.py:158`, `1934`, `2349`), so a typo like `"bfp16"` silently selects fp32.
- **What it does:** Selects the **autocast** compute dtype for the forward/backward pass. The training forward runs inside `torch.autocast(device_type, dtype=<from data_type>)` (`train_mara.py:1934`); validation and other forwards use the same mapping. This is the mixed-precision math dtype (matmul/attention accumulation policy), independent of how FSDP stores/shards params. `settings.data_type` is also passed into `compute_mfu(...)` for the MFU/peak-FLOPS estimate (call site `train_mara.py:2650`; the function itself defaults `data_type="fp16"` at `train_mara.py:4743`).
- **Interacts with:** `FSDP_param_dtype` (FSDP param storage dtype — usually set to match), `FSDP_reduce_dtype`.
- **Example:** `data_type: bf16` (from `configs/dn4.yaml`)

### `FSDP_param_dtype`
- **Type:** string
- **Default:** required (read directly as `settings.FSDP_param_dtype`, `train_mara.py:4477`)
- **Values/constraints:** `"bf16"` | `"fp16"` | else → `float32`. Same unvalidated ternary mapping; unknown strings fall through to fp32 (`train_mara.py:4476-4480`).
- **What it does:** Sets `MixedPrecisionPolicy(param_dtype=...)` for FSDP2 (`train_mara.py:4490-4494`). This is the dtype FSDP2 uses for the all-gathered (unsharded) parameters seen by the compute during forward/backward, and it is also used as `output_dtype` so FSDP module outputs match the param dtype (PyTorch 2.4+ branch; falls back to a policy without `output_dtype` on older torch, `train_mara.py:4495-4499`). The master weights held by the optimizer remain fp32; this only controls the working/compute copy. This is the dtype implicated in the body-norm-ramp investigation — the fp32-param probes flip this to fp32 while keeping the rest bf16.
- **Interacts with:** `FSDP_reduce_dtype` (gradient reduce-scatter dtype), `data_type` (autocast math dtype), `cpu_offload` (sibling FSDP key).
- **Example:** `FSDP_param_dtype: bf16` (from `configs/dn4.yaml`)

### `FSDP_reduce_dtype`
- **Type:** string
- **Default:** required (read directly as `settings.FSDP_reduce_dtype`, `train_mara.py:4484`)
- **Values/constraints:** `"bf16"` | `"fp16"` | else → `float32` (same ternary, `train_mara.py:4483-4487`).
- **What it does:** Sets `MixedPrecisionPolicy(reduce_dtype=...)` (`train_mara.py:4492`) — the dtype used for FSDP2's **gradient reduce-scatter** collective across DP ranks. Setting this to `fp32` while params stay bf16 is the standard fix for the FSDP2 bf16 reduce-scatter accumulation bias (the >2-summand rounding issue flagged as a prime suspect for the body-norm ramp). Because it's set on the same `mp_policy` used for every `fully_shard` call (layers, experts, aux heads, root), it applies uniformly to all sharded modules.
- **Interacts with:** `FSDP_param_dtype`; directly relevant to the reduce-scatter precision investigation.
- **Example:** `FSDP_reduce_dtype: bf16` (from `configs/dn4.yaml`)

### `FSDP_buffer_dtype`
- **Type:** string
- **Default:** n/a — **this key is dead / vestigial.**
- **Values/constraints:** Set to `bf16` in 34 of the 35 config files, but grep across both `train_mara.py` and `common_fsdp2/` finds **zero reads**. `MixedPrecisionPolicy` is constructed only with `param_dtype`, `reduce_dtype`, and (on torch 2.4+) `output_dtype` (`train_mara.py:4490-4499`) — never a buffer dtype.
- **What it does:** **Nothing.** It is parsed onto `self` by the YAML loop but never consumed. Buffer casting is not configured; buffers follow whatever FSDP/module defaults apply. Changing this value has no effect on a run. (Flagging honestly: this is a config-surface footgun — it looks like it controls buffer precision but is inert.)
- **Interacts with:** — (no code path)
- **Example:** `FSDP_buffer_dtype: bf16` (from `configs/dn4.yaml`; present but ignored)

### `reshard_after_forward`
- **Type:** bool
- **Default:** `True` (the only setting here with a real default: `getattr(settings, 'reshard_after_forward', True)`, `train_mara.py:4501`)
- **Values/constraints:** truthy/falsy bool. No further validation.
- **What it does:** Passed to every per-layer / per-expert / per-aux-head `fully_shard(..., reshard_after_forward=...)` call (`train_mara.py:4512`, `4514`, `4520`). When `True`, FSDP2 re-shards (frees the all-gathered params) immediately after each module's forward, minimizing peak memory at the cost of re-gathering in backward; `False` keeps params unsharded between forward and backward, trading memory for less communication. Note the **root** module is always wrapped with `reshard_after_forward=False` regardless of this setting (`train_mara.py:4521`) — that's a hardcoded FSDP2 idiom (the top-level wrap keeps its full params resident).
- **Interacts with:** `cpu_offload` (sibling `getattr(settings, 'cpu_offload', False)` → `CPUOffloadPolicy`, `train_mara.py:4502-4503`); memory-vs-throughput tradeoff.
- **Example:** `reshard_after_forward: true` (from `configs/dn4.yaml`)

### `compile_model`
- **Type:** bool
- **Default:** required (read as `settings.compile_model`, `train_mara.py:6440`)
- **Values/constraints:** truthy/falsy bool; no validation.
- **What it does:** Master gate for `torch.compile`. When true, calls `_apply_per_submodule_compile(model, settings.compile_mode, logger)` (`train_mara.py:6445`), which compiles **per-submodule** (each attention/FFN/norm, and for MoE each of experts/router/shared_experts) rather than whole-block — this avoids dynamo guard invalidation from inline activation checkpointing and keeps graphs constant under tail truncation (`_apply_per_submodule_compile` docstring, `train_mara.py:4368-4379`). It also bumps `torch._dynamo.config.cache_size_limit` to `max(16, len(model.layers)+4)` and, for MoE, enables `capture_scalar_outputs` (`train_mara.py:4385-4390`). When false, the model runs eager.
- **Interacts with:** `compile_mode`; MoE/GDN layer flags (change what gets compiled); truncation/`bypass_compile` (truncated steps can bypass the compiled graph via `model._orig_mod`, `train_mara.py:1931-1933`).
- **Example:** `compile_model: true` (from `configs/dn4.yaml`)

### `compile_mode`
- **Type:** string
- **Default:** required (read as `settings.compile_mode`, only used when `compile_model` is true; `train_mara.py:6445`)
- **Values/constraints:** Passed straight through to `torch.compile(submod, mode=compile_mode)` (`train_mara.py:4408`, `4412`) with **no local validation** — so valid values are exactly torch's compile modes (`"default"`, `"reduce-overhead"`, `"max-autotune"`, etc.); an invalid string errors inside `torch.compile`. Observed in configs: `default` (nearly all) and `reduce-overhead` (`configs/7B-MAX.yaml`).
- **What it does:** Selects the `torch.compile` optimization mode applied to every compiled submodule. `default` is the balanced mode; `reduce-overhead` uses CUDA graphs to cut Python/launch overhead (more memory, can be fragile with dynamic shapes). Only consulted when `compile_model: true`.
- **Interacts with:** `compile_model` (gate); MoE dynamic routing shapes (why `capture_scalar_outputs` is force-enabled for MoE).
- **Example:** `compile_mode: default` (from `configs/dn4.yaml`)

### `clip_warmup`
- **Type:** float
- **Default:** required (read as `settings.clip_warmup`, `train_mara.py:2063`; also printed unconditionally in the run summary, `train_mara.py:4916`)
- **Values/constraints:** float. No range validation. It is the max-norm used for grad clipping **while `step < warmup_steps`**. There is no "disable" sentinel — the value is used directly as `max_norm`, and `clip_coef = max_norm / (total_norm + 1e-6)` with grads scaled only when `clip_coef < 1.0` (`train_mara.py:1179-1182`, `1205-1208`). A `0.0` here would zero all grads; to effectively disable, set a large number.
- **What it does:** The gradient-norm clip threshold applied during LR warmup. Per step, `clip_value = clip_warmup if step < warmup_steps else clip_standard` (`train_mara.py:2063`), then `_clip_grad_norm_mixed_mesh(model, clip_value)` computes a global (all-reduced, cross-mesh-safe) L2 grad norm and rescales all grads if it exceeds the threshold. A looser warmup clip (e.g. 2.0) lets early grads through while the LR is ramping.
- **Interacts with:** `warmup_steps` (the switch point), `clip_standard` (post-warmup value); `track_clip_groups`/`track_gpm` enable the per-group `group_telemetry` branch of the clip function.
- **Example:** `clip_warmup: 2.0` (from `configs/dn4.yaml`)

### `clip_standard`
- **Type:** float
- **Default:** required (read as `settings.clip_standard`, `train_mara.py:2063`; printed in summary, `train_mara.py:4917`)
- **Values/constraints:** float; same mechanics as `clip_warmup` (used directly as `max_norm`; no disable sentinel; `0.0` zeros grads).
- **What it does:** The gradient-norm clip threshold applied for the **main phase** (`step >= warmup_steps`). Same global-norm clip path as `clip_warmup`. `1.0` is the standard value across configs. (Memory note per `keel_clip_threshold_lesson`: for from-scratch KEEL + tangent-proj runs, 1.0 can be too aggressive early — judge by median/%-clipped, not mean.)
- **Interacts with:** `warmup_steps`, `clip_warmup`; Muon/NorMuon (Newton-Schulz already normalizes update magnitude, so clipping mainly bites the Adam-family and embedding/head grads); `track_clip_groups`.
- **Example:** `clip_standard: 1.0` (from `configs/dn4.yaml`)

# FSDP config knob

### `cpu_offload`
- **Type:** bool
- **Default:** `False` (read as `getattr(settings, 'cpu_offload', False)` at `train_mara.py:4502`, `6162`, `3375`)
- **Values/constraints:** `True`/`False`. Enables FSDP2 CPU offload of parameters, gradients, and optimizer states. Used by `configs/MegaMoe-CPU.yaml`. KNOWN INCOMPATIBILITY: combining `cpu_offload` with `optimizer_type in {"adamw_8bit", "adamc_8bit"}` can cause device-mismatch errors (torchao bug) — a warning is logged at `train_mara.py:6164-6165`.
- **What it does:** When true, the FSDP2 setup builds a `CPUOffloadPolicy(pin_memory=True)` (`train_mara.py:4502-4503`) and passes it as `offload_policy=` to every `fully_shard(...)` call — inner MoE experts, outer transformer layers, aux heads, and the root model (`train_mara.py:4512-4521`). This keeps sharded params/grads/optimizer-state on pinned host memory, and FSDP streams them to the GPU on demand (H2D) during forward/backward. Model materialization then targets CPU: `mat_device = "cpu"` and `model.to_empty(device="cpu")` (`train_mara.py:4524-4528`). Because only params are managed by FSDP on CPU, all model buffers (RoPE freqs, `expert_bias`, etc.) are explicitly walked and moved back to the GPU afterward (`train_mara.py:4531-4544`). At startup, when enabled, it logs "CPU offload enabled (pin_memory=True)" (`4505-4506`) and the training-config banner prints "CPU Offload = ON (params, grads, optimizer states on CPU)" (`6162-6163`). The flag is also persisted into the checkpoint metadata (`train_mara.py:3375`).
- **Interacts with:** All `fully_shard` wraps (shares `offload_policy` with `mp_policy` / `reshard_after_forward`). Trades GPU memory for host-device transfer bandwidth (enables larger models per GPU at a throughput cost). Avoid with `adamw_8bit`/`adamc_8bit` (torchao device-mismatch bug, warned at `6164-6165`). Note: checkpoint save/restore separately use `StateDictOptions(..., cpu_offload=True)` at `train_mara.py:3241`, `3574`, `3747` — that is an unconditional state-dict offload for gathering the full state dict on CPU and is NOT gated by this setting.
- **Example:** `cpu_offload: true`

---

## Body Growth Control: Tangent Projection + Shadow-Norm PDR Controller

Two coupled mechanisms that govern how fast the Muon body matrices grow in norm. **Tangent projection** strips the (radial / norm-growing) component out of the final Muon update — Newton-Schulz leaves a small, 100%-consistent anti-radial component (`cos(update,W)≈−0.013`) that descent flips to +radial, growing `‖W‖` and starving weight decay (muon_fsdp2.py:401-446). Stripping it flattens `‖W‖` but also removes the body's natural self-anneal (pdr = `‖ΔW‖/‖W‖` shrinks as `‖W‖` grows). The **ffn_pdr_controller** (`common_fsdp2/body_lr_controller.py`) restores that anneal as an explicit LR cut on the FFN body group, driven either by a hand-fit reference, a self-anchored LR-track reference, or an online shadow-norm reference. The two are mutually exclusive with adaptive_wd and with SCS (train_mara.py:5276-5287).

### `tangent_project`
- **Type:** bool
- **Default:** `false` (no Settings.__init__ entry — always read via `getattr(settings, 'tangent_project', False)`, e.g. train_mara.py:1489, 6186; there is no validation block, so an absent key silently means off)
- **Values/constraints:** `true`/`false`. No explicit validation.
- **What it does:** Master gate for the radial-projection block in the FSDP Muon path (`Fsdp1dWork.finish`, muon_fsdp2.py:411). When on, after NS + scale + NorMuon it computes the global (all-reduced over shards) radial coefficient `c = ⟨U,W⟩/‖W‖²` and does `U ← U − f·c·W` on the body matrices, removing the fraction `f` of the update's radial component. It also populates `radial_stats[id(param)] = (‖W‖, γ)` (γ = `−⟨U,W⟩/‖W‖²`, the free radial-growth rate) which the shadow controller consumes. Only active on the FSDP/DTensor path (`Fsdp1dWork.finish`); the single-device Muon path (`SingelDeviceWork`) has no tangent block, so shadow modes fatal-guard against plain tensors (train_mara.py:1547-1556).
- **Interacts with:** `tangent_project_strength` (the fraction f), `tangent_project_preserve_norm`; **required `true`** for `ffn_pdr_controller.reference.mode` = `auto`, `auto_shadow_growth`, `auto_shadow_partial` (train_mara.py:5146, 5189).
- **Example:** `tangent_project: true` (from configs/dn4.yaml:100, kv3.yaml:150)

### `tangent_project_strength`
- **Type:** float **or** schedule `[[step, val], ...]`
- **Default:** `1.0` (train_mara.py:5295-5296)
- **Values/constraints:** Scalar must be in `[0,1]`; schedule entries must each be `[step, val]` with `val` in `[0,1]` and **strictly ascending steps** (train_mara.py:5298-5317). Not a bool (bool is explicitly rejected). Schedule is linearly interpolated per step (`interpolate_lr_mod`), flat-extrapolated past the ends.
- **What it does:** The partial-projection fraction f applied inside the tangent block (`_c = c·f`, muon_fsdp2.py:432-433). `f=1` = full projection (flat `‖W‖`, original behavior); `f<1` leaves `(1−f)` of the radial component so `‖W‖` grows at `(1−f)` of its natural rate; `f=0` = no projection (free growth). The train loop writes the interpolated per-step value into each Muon body group's `tangent_project_strength` (train_mara.py:2057-2061). The canonical "grow-then-clamp" recipe holds `f=0` early to bank the early regularization benefit, then ramps `f→1` to lock the body. It is also the freeze-point signal fed to the controller as `f_now`.
- **Interacts with:** `tangent_project` (only meaningful when on); gates controller engagement in shadow modes (engagement is f-gated, not step-gated); `auto`/`auto_shadow_growth` **require the schedule to terminate at `f≈1.0`** or Settings fatals (the body must freeze; train_mara.py:5164-5169, 5265-5270); `reference.anchor_step: 'auto'` derives the anchor from the earliest step this schedule reaches its terminal f (train_mara.py:5126-5137).
- **Example:** `tangent_project_strength: [[0, 0.0], [12_000, 0.0], [24_000, 1.0]]` (grow free to 12k, anneal f 0→1 over 12k–24k, frozen after; configs/dn4.yaml:102). Scalar form e.g. `tangent_project_strength: 1.0`.

### `tangent_project_preserve_norm`
- **Type:** bool
- **Default:** `false` (no Settings.__init__ entry — read via `getattr(..., False)`, train_mara.py:6187, muon_fsdp2.py:434; no validation)
- **Values/constraints:** `true`/`false`.
- **What it does:** When on, the tangent block rescales the projected update back to its pre-projection Frobenius norm: it measures `‖U‖` (all-reduced) before the `U ← U − c·W` subtraction and multiplies by `‖U‖_before/‖U‖_after` afterward (muon_fsdp2.py:434-446). This keeps the update's magnitude constant while only changing its direction (removes the radial component but re-inflates the tangential part to compensate). Left off in the reference configs — projecting without preserving norm is the intended "remove radial growth" behavior.
- **Interacts with:** `tangent_project` (only applies when on); orthogonal to `tangent_project_strength`.
- **Example:** `tangent_project_preserve_norm: false` (configs/dn4.yaml:101, kv3.yaml:151)

---

The remaining settings are all sub-keys of the **`ffn_pdr_controller`** dict. The whole dict collapses to `None` unless `enabled: true` (train_mara.py:5062-5066); when `None` the controller object's `current_multiplier()` returns 1.0 and everything below is inert. It **actuates only the dense `feed_forward.w1/w2/w3` Muon matrices** — the trainer raises `RuntimeError` on a MoE model (expert params it can't actuate) or if it finds 0 dense FFN params (train_mara.py:1534-1543).

### `ffn_pdr_controller.enabled`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true`/`false`. If `false` (or the whole key absent), the dict is set to `None` and skipped.
- **What it does:** Turns the FFN-only pdr feedback controller on. When on, every step the train loop writes the controller's held multiplier `m` into `lr_scale_overrides[id(p)]` for the FFN body params; at the val-step cadence it calls `observe(...)` with the fresh FFN-median pdr to update `m`. `enabled: false` ⇒ `observe()` no-ops and `m` stays 1.0 (body_lr_controller.py:129, 134, 372-373).
- **Interacts with:** Mutually exclusive with `adaptive_wd` (AWD moves `‖W‖`, the pdr denominator) and with SCS `auxiliary_heads.compute_inactive_layers:false` (would un-freeze SCS-frozen FFN params) — both fatal (train_mara.py:5276-5287).
- **Example:** `ffn_pdr_controller:\n  enabled: true` (configs/kv3.yaml:105-106, dn4.yaml:110-111)

### `ffn_pdr_controller.reference.mode`
- **Type:** string
- **Default:** `"knots"` (body_lr_controller.py:161; validation default at train_mara.py:5103)
- **Values/constraints:** One of `knots`, `auto`, `auto_shadow_growth`, `auto_shadow_partial` (train_mara.py:5105-5107). If `reference` is present it must be a dict.
- **What it does:** Selects how the target pdr trajectory `r(t)` is produced. **`knots`** (legacy/hand-fit): `r = interp(knots, tok_m)`, an operator-supplied token→pdr curve, closed with a feedforward inversion `m = r/K_ema` (+ optional PI trim). **`auto`** (self-anchored LR-track, Q11): holds `m=1` until the body freezes (`f→1` at/after `anchor_step`), latches the frozen-body plant gain `K_anchor` from `anchor_samples` post-freeze pdr readings, then rides `r(t) = K_anchor·lr(t)/lr_anchor` — no knots. **`auto_shadow_growth`** (Q12, grow-then-clamp): builds an online shadow norm `S` (the free-growth counterfactual `‖W‖`) and commands `m = exp(g·median_i[log(R_i/S_i)])` where `R=‖W‖`; at `f=1` it either hands off to the LR-track tail or continues the R/S law (see `freeze_handoff`). **`auto_shadow_partial`**: same shadow law but first-class for a run whose `f` terminates below 1.0 — never latches/hands off, stays in the ramp law forever. Which knobs apply is noted per-setting below.
- **Interacts with:** `knots`/`blend_from` (knots mode); `anchor_step`/`anchor_samples`/`anchor_*_band`/`warmup_step` (auto mode); `rho`/`lambda_*`/`m_min_full`/`glide_gain`/`freeze_handoff` (shadow modes). `auto` and shadow modes all require `tangent_project: true`.
- **Example:** `reference:\n    mode: auto_shadow_growth` (configs/dn4.yaml:124-125). Omitting `mode:` uses `knots` (configs/kv3.yaml:126 supplies only `knots`+`blend_from`).

### `ffn_pdr_controller.reference.knots`
- **Type:** list of `[tok_m, pdr]` pairs
- **Default:** `[]` (required in `knots` mode)
- **Values/constraints:** Required and non-empty for `mode: knots` (train_mara.py:5108-5111). x (tok_m, tokens-in-millions) must be **strictly ascending**; y (target pdr) must be **> 0** (body_lr_controller.py:196-202). **Rejected** as a stale key in shadow modes — fatal if present (train_mara.py:5219-5224).
- **What it does:** The hand-fit reference pdr curve: `reference(tok_m) = interp(knots, tok_m)`, flat-extrapolated past the ends (body_lr_controller.py:29-42, 276-280). Typically the actual recorded FFN-median pdr from a prior run, sampled to the run horizon. The last knot's y sets the plateau the controller holds for the remainder of the run.
- **Interacts with:** `blend_from` (crossfades a second curve into this one over an early window); ignored entirely in `auto`/shadow modes.
- **Example:** `knots: [[197, 1.66e-3], [393, 2.94e-3], ..., [26000, 1.26e-3]]` (configs/kv3.yaml:127-129)

### `ffn_pdr_controller.reference.blend_from`
- **Type:** dict `{knots: [...], start_tok_m, end_tok_m}`
- **Default:** absent (`None`)
- **Values/constraints:** Only meaningful in `knots` mode. Its own `knots` obey the same ascending-x / positive-y rules; `start_tok_m` must be **< `end_tok_m`** (else the smoothstep degenerates to a hard step — fatal, train_mara.py:5271-5275, body_lr_controller.py:204-206).
- **What it does:** Optional second knot curve smoothstep-crossfaded **into** the main `knots` over `[start_tok_m, end_tok_m]`: `reference = (1−a)·interp(blend_from.knots) + a·interp(knots)` with `a = smoothstep(tok_m, t0, t1)` (body_lr_controller.py:276-280). Used by from-scratch runs to ride a high early-plasticity pdr curve before settling onto the mature glide.
- **Interacts with:** `knots` (the target it fades into); rejected in shadow modes.
- **Example:** `blend_from:\n  knots: [[197, 3.42e-3], [400, 3.10e-3], [600, 2.90e-3]]\n  start_tok_m: 197\n  end_tok_m: 575` (configs/kv3.yaml:130-133)

### `ffn_pdr_controller.reference.anchor_step`
- **Type:** int **or** the string `"auto"` (`auto` mode only)
- **Default:** `0` (body_lr_controller.py:164); **required** in `auto` mode (train_mara.py:5115-5118)
- **Values/constraints:** In `auto` mode: an int, or `"auto"` to derive it from the `tangent_project_strength` schedule (earliest step reaching terminal f; requires the schedule to be a `[[step,val],...]` list — train_mara.py:5119-5137). Must be `> warmup_step` and `< max_steps` (train_mara.py:5138-5143). Rejected as a stale key in shadow modes (train_mara.py:5219-5224).
- **What it does:** The freeze point at which `auto` mode begins collecting the frozen-body pdr to latch its reference. Before it (and before `f≥1`), `m` is held at 1.0. The controller only starts filling the anchor buffer once `step ≥ anchor_step` **and** the body is truly frozen (`f_now ≥ 1`) (body_lr_controller.py:396-408).
- **Interacts with:** `anchor_samples`, `tangent_project_strength` (freeze schedule), `warmup_step`, `max_steps`; irrelevant to `knots`/shadow modes.
- **Example:** `reference:\n    mode: auto\n    anchor_step: auto` (derives from the f-schedule; train_mara.py:5119-5137). No shipped config in `configs/` currently uses `auto` mode — dn3/kv3 use `knots`, dn4 uses `auto_shadow_growth`.

### `ffn_pdr_controller.reference.anchor_samples`
- **Type:** int
- **Default:** `8` (body_lr_controller.py:165)
- **Values/constraints:** `auto` mode only; must be `>= 1` (train_mara.py:5144-5145). Rejected as stale in shadow modes.
- **What it does:** How many post-freeze pdr samples the `auto` controller collects (at the val-step cadence) before latching. `K_anchor = geometric_mean(collected pdr) / m_anchor` (with `m_anchor=1` throughout collection); Math recommends a 5–10 sample window (body_lr_controller.py:283-296, 405-408).
- **Interacts with:** `anchor_step`, the anchor sanity bands below.
- **Example:** `anchor_samples: 8` (default; `auto`-mode only).

### `ffn_pdr_controller.reference.anchor_warn_band`
- **Type:** list `[lo, hi]`
- **Default:** `[0.6, 1.4]` (body_lr_controller.py:166-167)
- **Values/constraints:** `auto` mode only; two floats. No dedicated Settings validation (consumed directly by the controller). Rejected as stale in shadow modes.
- **What it does:** At latch, if the geometric-mean `pdr_anchor` divided by the trailing pre-freeze pdr EMA falls **outside** `[lo, hi]`, the controller sets a HEALTH-WARNING string the trainer logs (body_lr_controller.py:311-313). A soft sanity check that the captured anchor is plausible relative to the pre-freeze trajectory.
- **Interacts with:** `anchor_fatal_band` (harder band), `anchor_abs_warn`.
- **Example:** `anchor_warn_band: [0.6, 1.4]` (default).

### `ffn_pdr_controller.reference.anchor_fatal_band`
- **Type:** list `[lo, hi]`
- **Default:** `[0.35, 2.0]` (body_lr_controller.py:168-169)
- **Values/constraints:** `auto` mode only; two floats. Rejected as stale in shadow modes.
- **What it does:** Like the warn band but harder: if `pdr_anchor / pre_freeze_pdr_EMA` is outside `[lo, hi]` the controller sets `anchor_fatal`, which the trainer escalates to a `fatal_error` (likely a capture bug — wrong anchor_step or an unsettled plant; body_lr_controller.py:307-310, train_mara.py:3143-3145).
- **Interacts with:** `anchor_warn_band`, `anchor_abs_warn`.
- **Example:** `anchor_fatal_band: [0.35, 2.0]` (default).

### `ffn_pdr_controller.reference.anchor_abs_warn`
- **Type:** list `[lo, hi]`
- **Default:** `[3.0e-4, 1.0e-2]` (body_lr_controller.py:170-173)
- **Values/constraints:** `auto` mode only; two floats. Rejected as stale in shadow modes.
- **What it does:** An **absolute** plausibility band on `pdr_anchor` (independent of the pre-freeze baseline). If outside, appends a warning (body_lr_controller.py:321-324). Deliberately loose — the scale-free relative band is the real bug-catcher; this one just avoids nuisance-warning on a hotter frozen body or a different model scale.
- **Interacts with:** the two relative bands above; also the sole guard when there were no genuine pre-freeze samples (body_lr_controller.py:314-320).
- **Example:** `anchor_abs_warn: [3.0e-4, 1.0e-2]` (default).

### `ffn_pdr_controller.m_min_full`
- **Type:** float
- **Default:** `0.20` (body_lr_controller.py:217)
- **Values/constraints:** **Shadow modes only.** Must be in `(0, m_max]` (train_mara.py:5200-5202). In shadow modes it **replaces `m_floor`**, and setting `m_floor` there is fatal (train_mara.py:5228-5230).
- **What it does:** Base of the f-aware floor: the minimum multiplier at any `f` is `m_min(f) = 1 − f·(1 − m_min_full)`, so the floor is 1.0 at `f=0` and `m_min_full` at `f=1` (body_lr_controller.py:531, 598-599). Prevents the shadow controller from cutting the body LR too hard early while `f` is still low.
- **Interacts with:** `m_max` (upper bound of the clamp), `freeze_handoff` (a long no-handoff tail can keep pushing `m` toward this floor as S outgrows R — dn4 raises it to 0.30 for that reason).
- **Example:** `m_min_full: 0.30` (configs/dn4.yaml:112)

### `ffn_pdr_controller.m_floor`
- **Type:** float
- **Default:** `0.30` (body_lr_controller.py:143; validation default train_mara.py:5069)
- **Values/constraints:** Must be in `(0, 1]` and `<= m_max` (train_mara.py:5069-5077). **Fatal if set in shadow modes** — inert there, replaced by `m_min_full` (train_mara.py:5228-5230). Applies to `knots` and `auto` modes.
- **What it does:** Hard lower clamp on the commanded multiplier `m` in knots/auto modes (`m_clamped = max(m_floor, min(m_max, m_cmd))`, body_lr_controller.py:452). Kept above 0 deliberately — `lr_scale=0` would fully freeze the FFN body; the floor plus the lower-rail alarm catches "out of cooling authority" instead.
- **Interacts with:** `m_max`, `alarm_pdr_ratio`/`alarm_consecutive` (the lower-rail alarm fires when pinned at this floor and pdr still `> alarm_pdr_ratio·r`).
- **Example:** `m_floor: 0.30` (configs/kv3.yaml:108, dn3.yaml:136)

### `ffn_pdr_controller.m_max`
- **Type:** float
- **Default:** `1.0` (body_lr_controller.py:144; validation train_mara.py:5072)
- **Values/constraints:** Must be `<= 1.0` — **no body-LR amplification allowed** (train_mara.py:5072-5074); and `m_floor`/`m_min_full` must be `<= m_max`.
- **What it does:** Hard upper clamp on `m`. At the default 1.0 the controller is cuts-only: it can throttle the FFN body LR down toward the floor but never boost it above base. This is what makes the LR-track and shadow references "unreachable-from-above" trigger the informational upper-rail alarm rather than amplifying.
- **Interacts with:** `m_floor`/`m_min_full`, `upper_alarm_margin` (upper rail trips when pinned at `m_max` yet the unclamped demand wants `> 1+margin`).
- **Example:** `m_max: 1.0` (configs/kv3.yaml:109, dn4.yaml:114)

### `ffn_pdr_controller.rho`
- **Type:** float
- **Default:** `0.20` (body_lr_controller.py:218)
- **Values/constraints:** **Shadow modes only.** Must be in `(0, 1]` (train_mara.py:5193-5195). Requires `tangent_project: true`.
- **What it does:** The radial-budget fraction in the shadow-mode body weight-decay law `λ_body = clamp(λ_min, λ_max, ρ·(1−f)·γ_EMA)`, where `γ_EMA` is the EMA of the median per-matrix free radial-growth rate (body_lr_controller.py:348-354). The controller **owns the FFN body WD** in shadow modes: it writes `λ_body` into `wd_overrides` for the FFN matrices each cadence, so `rho` says "spend at most this fraction of the remaining radial-growth budget on WD." A body WD *schedule* on the FFN matrices is silently overridden — the trainer prints a loud banner (train_mara.py:5231-5253).
- **Interacts with:** `lambda_max`, `lambda_min` (the clamp), `tangent_project_strength` (the `(1−f)` factor); knots/auto modes ignore it (they don't touch body WD).
- **Example:** `rho: 0.20` (configs/dn4.yaml:115)

### `ffn_pdr_controller.lambda_max`
- **Type:** float
- **Default:** `0.02` (body_lr_controller.py:219)
- **Values/constraints:** **Shadow modes only.** Must satisfy `0 <= lambda_min <= lambda_max` (train_mara.py:5196-5199).
- **What it does:** Upper clamp on the radial-budget WD `λ_body`. It is also the value used **before any γ sample exists** (`_gamma_ema is None` → return `lambda_max`, body_lr_controller.py:349-352) — i.e. the high-early-WD regularization prior while radial growth is still alive.
- **Interacts with:** `rho`, `lambda_min`; also drives the shadow norm's decay via `λ_S = clamp(λ_min, λ_max, ρ·γ_EMA)` (the f=0 counterfactual WD).
- **Example:** `lambda_max: 0.02` (configs/dn4.yaml:116)

### `ffn_pdr_controller.lambda_min`
- **Type:** float
- **Default:** `0.002` (body_lr_controller.py:220)
- **Values/constraints:** **Shadow modes only.** Must satisfy `0 <= lambda_min <= lambda_max` (train_mara.py:5196-5199).
- **What it does:** Lower clamp on the radial-budget WD `λ_body` — the residual WD once the body is clamped (`f→1` drives the raw `ρ(1−f)γ` term toward 0, so the law floors at `lambda_min`; body_lr_controller.py:348-354).
- **Interacts with:** `rho`, `lambda_max`.
- **Example:** `lambda_min: 0.002` (configs/dn4.yaml:117)

### `ffn_pdr_controller.k_ema_alpha`
- **Type:** float
- **Default:** `0.15` (body_lr_controller.py:145)
- **Values/constraints:** Must be in `(0, 1]` (train_mara.py:5078-5081).
- **What it does:** Smoothing factor for the log-space EMA of the plant gain `K = pdr/m` (`_logK ← (1−α)·_logK + α·log(K_inst)`, body_lr_controller.py:440-441). `K_ema` is the denominator in the feedforward inversion `m_ff = r/K_ema`. In shadow modes it also smooths `γ_EMA` (the radial-budget input, body_lr_controller.py:345-346). Higher α = faster tracking / noisier; per-observe (per val-step), so it must be re-tuned when the batch/token-per-sample changes.
- **Interacts with:** `pdr_ema_alpha` (a separate PV smoother), `val_step` (the cadence these alphas are tuned against). dn3 raises both to 0.25 because its bigger batch makes each sample a much longer token-space memory.
- **Example:** `k_ema_alpha: 0.15` (kv3.yaml:110); `0.25` (dn3.yaml:142, dn4.yaml:118)

### `ffn_pdr_controller.pdr_ema_alpha`
- **Type:** float
- **Default:** `0.15` (body_lr_controller.py:146)
- **Values/constraints:** Must be in `(0, 1]` (train_mara.py:5078-5081).
- **What it does:** Smoothing factor for the log-space EMA of the observed pdr (`_pdr_ema`), which is the process-variable used **only** by the PI trim's log-error term `e = log(r/_pdr_ema)` (body_lr_controller.py:437-444). Separate from `k_ema_alpha` (which smooths K for the feedforward path) so the two aren't redundant. Also builds the pre-freeze pdr EMA baseline in `auto` mode.
- **Interacts with:** `kp`/`ki`/`kd` (the trim that consumes `_pdr_ema`); `k_ema_alpha`.
- **Example:** `pdr_ema_alpha: 0.15` (kv3.yaml:111); `0.25` (dn3.yaml:143)

### `ffn_pdr_controller.rate_up`
- **Type:** float
- **Default:** `0.02` (body_lr_controller.py:148)
- **Values/constraints:** Must be `>= 0` (train_mara.py:5084-5085).
- **What it does:** Asymmetric rate limit on *increasing* `m` per observe: `m_cmd ≤ m·(1+rate_up)` (body_lr_controller.py:450, 597). Kept small so the controller reheats the body LR slowly and doesn't chase pdr noise upward.
- **Interacts with:** `rate_down` (the faster cooling limit).
- **Example:** `rate_up: 0.02` (kv3.yaml:113, dn4.yaml:120)

### `ffn_pdr_controller.rate_down`
- **Type:** float
- **Default:** `0.05` (body_lr_controller.py:147)
- **Values/constraints:** Must be in `[0, 1)` (train_mara.py:5082-5083).
- **What it does:** Asymmetric rate limit on *decreasing* `m` per observe: `m_cmd ≥ m·(1−rate_down)` (body_lr_controller.py:450, 597). Larger than `rate_up` ("cool fast, reheat slow") so the controller can throttle a too-hot body quickly.
- **Interacts with:** `rate_up`.
- **Example:** `rate_down: 0.05` (kv3.yaml:112, dn4.yaml:119)

### `ffn_pdr_controller.glide_gain`
- **Type:** float
- **Default:** `1.0` (body_lr_controller.py:225)
- **Values/constraints:** **Shadow modes only.** Must be in `(0, 5]` (train_mara.py:5203-5206); Math's preferred band `[1.0, 1.3]`.
- **What it does:** Exponent on the shadow ratio: `m = exp(g·median_i[log(R_i/S_i)]) = geomean(R/S)^g` (body_lr_controller.py:559). `g=1.0` = pure R/S (track the free-growth counterfactual exactly). `g>1` **steepens** the cut → cooler reference → pulls the controlled pdr toward a fully-free-growth run (compensates for the shadow norm running a hair warm since it's built from the already-cooled trajectory). A tuning dial, inert at the default.
- **Interacts with:** `m_min_full` (steeper glide can peg `m` at the floor early), `rho`/`lambda_*` (the shadow norm's own WD decay).
- **Example:** `glide_gain: 1.25` (configs/dn4.yaml:121)

### `ffn_pdr_controller.freeze_handoff`
- **Type:** bool
- **Default:** `true` (body_lr_controller.py:230; validation default train_mara.py:5207)
- **Values/constraints:** **Shadow modes only.** Must be a real bool (train_mara.py:5207-5210). For `auto_shadow_partial` it is a **no-op** (partial never hands off) — the trainer prints a note (train_mara.py:5215-5218).
- **What it does:** For `auto_shadow_growth`, gates what happens at `f=1`. `true`: latch `r_freeze = K_ema·m` and `lr_freeze` once, then switch to the Q11 LR-track tail `r(t) = r_freeze·lr/lr_freeze`, `m = r/K_ema` (body_lr_controller.py:575-583, 605-610). `false` (NO-HANDOFF): keep commanding `m = median(R/S)` into the frozen phase — S keeps growing while R is frozen, so `m` keeps falling — a continuous anneal with **no law-switch/kink** (the body is still fully frozen via the f=1 clamp). The flag is config-authoritative (not restored from the checkpoint) but its checkpointed value is compared on resume: if it was **flipped across a resume while the controller has already engaged** (`shadow_active`/`frozen`/`f>0`) the trainer **fatals** (silent law-switch / r_freeze rebase); flipping it before engagement (f=0, no history) is allowed with only a HEALTH-WARNING, and a checkpoint predating the field warns that the value cannot be verified (body_lr_controller.py:709-712, 231, 758-761; guard at train_mara.py:3840-3863).
- **Interacts with:** `tangent_project_strength` (must reach f=1 for growth mode), `m_min_full` (no-handoff runs may need a higher floor), `reference.mode` (only meaningful for `auto_shadow_growth`).
- **Example:** `freeze_handoff: false` (NO-HANDOFF continuous anneal; configs/dn4.yaml:122)

### `ffn_pdr_controller.acts_on_attn`
- **Type:** bool
- **Default:** `false` (body_lr_controller.py:236; validation default train_mara.py:5211)
- **Values/constraints:** **Shadow modes only.** Must be a real bool (train_mara.py:5211-5214).
- **What it does:** When `true`, the trainer broadcasts the **same** FFN-computed multiplier `m` to the attention matrices (`wq/wk/wv/wo`) as well as the FFN body, so attn rides the same free-growth glide and the attn↔ffn lock-step is preserved (train_mara.py:1523-1529). The controller itself is unchanged (one `m`, computed FFN-only). Crucially, the radial-budget WD `λ` is **NOT** broadcast — attention keeps its flat base WD; only the LR multiplier is shared.
- **Interacts with:** the pdr id-set the `m` is written to (`_pdr_m_ids` = FFN ∪ attn when on); `rho`/`lambda_*` (FFN-only regardless).
- **Example:** `acts_on_attn: true` (configs/dn4.yaml:123)

### `ffn_pdr_controller.warmup_step`
- **Type:** int
- **Default:** `1500` (body_lr_controller.py:135; validation default train_mara.py:5086)
- **Values/constraints:** Must be in `[0, max_steps)` (train_mara.py:5086-5090). **Fatal if set in shadow modes** — engagement there is f-gated, not step-gated (train_mara.py:5225-5227). Applies to `knots` and `auto` modes.
- **What it does:** In knots/auto modes the controller holds `m=1.0` (loop frozen) until `step >= warmup_step`, then engages (body_lr_controller.py:376-380). Lets the LR warmup / early plasticity settle before the controller starts steering.
- **Interacts with:** `anchor_step` (must be `> warmup_step` in auto mode).
- **Example:** `warmup_step: 1500` (kv3.yaml:107); `7000` (dn3.yaml:134)

### `ffn_pdr_controller.kp`
- **Type:** float
- **Default:** `0.0` (body_lr_controller.py:208)
- **Values/constraints:** No range validation. Non-shadow modes (the PI trim runs in knots/auto; shadow modes command `m` from R/S directly).
- **What it does:** Proportional gain of the optional log-space PI trim layered on top of the feedforward inversion: `trim = exp(kp·e + I + kd·d)` on log-error `e = log(r/pdr_ema)` (body_lr_controller.py:89-101). With `kp=ki=kd=0` the trim is an exact no-op (`1.0`) — the run-1 default is feedforward-only. Enable only if live tracking shows a persistent bias the feedforward can't remove.
- **Interacts with:** `ki`, `kd`, `integral_clamp`, `pdr_ema_alpha` (supplies the PV).
- **Example:** `kp: 0.0` (kv3.yaml:116, dn3.yaml:146)

### `ffn_pdr_controller.ki`
- **Type:** float
- **Default:** `0.0` (body_lr_controller.py:208)
- **Values/constraints:** No range validation.
- **What it does:** Integral gain of the PI trim. Accumulates the log-error into `I` (clamped to `±integral_clamp`) with anti-windup (the integral is frozen when the output is railed in the error's direction; body_lr_controller.py:94, 108-111, 454-458). `0.0` = no integral action.
- **Interacts with:** `kp`, `kd`, `integral_clamp`.
- **Example:** `ki: 0.0` (kv3.yaml:117, dn3.yaml:147)

### `ffn_pdr_controller.kd`
- **Type:** float
- **Default:** `0.0` (body_lr_controller.py:209)
- **Values/constraints:** No range validation.
- **What it does:** Derivative gain of the PI trim. Uses derivative-on-PV (`d = pv − prev_pv`, not error) to avoid setpoint-change kick (body_lr_controller.py:95-101). `0.0` = no derivative action.
- **Interacts with:** `kp`, `ki`.
- **Example:** `kd: 0.0` (kv3.yaml:118, dn3.yaml:148)

### `ffn_pdr_controller.integral_clamp`
- **Type:** float
- **Default:** `0.5` (body_lr_controller.py:210)
- **Values/constraints:** Must be `>= 0` (train_mara.py:5093-5094).
- **What it does:** Symmetric clamp on the PI trim's accumulated integral `I` (`I ∈ [−integral_clamp, +integral_clamp]`, body_lr_controller.py:94). Bounds how far the integral term can push the trim `exp(...)`; part of the anti-windup design.
- **Interacts with:** `ki`, `kp`, `kd`.
- **Example:** `integral_clamp: 0.5` (kv3.yaml:119, dn3.yaml:149)

### `ffn_pdr_controller.force_m`
- **Type:** float or `null`
- **Default:** `None` (body_lr_controller.py:141-142)
- **Values/constraints:** No range validation. Any float, or absent/`null` for normal operation.
- **What it does:** DEBUG open-loop override. When set, `current_multiplier()` returns this **fixed** value and the entire feedback loop is bypassed (body_lr_controller.py:328-329). For actuator probes — pin a large cut and watch whether pdr responds. Leave unset for a real run.
- **Interacts with:** overrides everything (`enabled`, all references) at the `current_multiplier()` level.
- **Example:** `force_m: 0.5` (debug only; not in any shipped config).

### `ffn_pdr_controller.alarm_pdr_ratio`
- **Type:** float
- **Default:** `1.1` (body_lr_controller.py:213)
- **Values/constraints:** Must be `> 0` (train_mara.py:5095-5096).
- **What it does:** Lower-rail (hot-body) alarm threshold. When `m` is pinned at the floor (`m_floor` in knots/auto, the f-aware `m_min(f)` in shadow) yet the observed `pdr > alarm_pdr_ratio · r`, the controller is out of cooling authority — it counts consecutive such samples and raises `alarm` (logged as "base-LR-too-high") once the count reaches `alarm_consecutive` (body_lr_controller.py:467-472, 614-620).
- **Interacts with:** `alarm_consecutive`, `m_floor`/`m_min_full`.
- **Example:** `alarm_pdr_ratio: 1.1` (kv3.yaml:136)

### `ffn_pdr_controller.upper_alarm_margin`
- **Type:** float
- **Default:** `0.05` (body_lr_controller.py:174)
- **Values/constraints:** Must be `>= 0` (train_mara.py:5097-5098). Note this is a **top-level** controller key (`cfg.get('upper_alarm_margin')`), not under `reference`.
- **What it does:** Upper-rail alarm threshold (Q11). When `m` is pinned at `m_max` yet the **unclamped** feedforward demand `m_ff_raw > 1 + upper_alarm_margin` for `alarm_consecutive` samples, the reference is unreachable from below — the body is cooler than target and amplification is forbidden. Informational (`upper_alarm`, "no-upward-authority"), **not** a hot-body condition — worrying only if the run also underfits (body_lr_controller.py:477-482).
- **Interacts with:** `m_max`, `alarm_consecutive`.
- **Example:** `upper_alarm_margin: 0.05` (default; no shipped config overrides it).

### `ffn_pdr_controller.authority_low_m`
- **Type:** float
- **Default:** `0.5` (body_lr_controller.py:212)
- **Values/constraints:** No dedicated range validation (consumed by the controller).
- **What it does:** Sets the `inspect` advisory flag: `inspect = (m < authority_low_m and tok_m < blend_end)` — i.e. `m` has been cut below this level while still inside the early blend region (knots mode). A soft "low-m-early" warning on the `[ffn-ctrl]` log line, not a control action (body_lr_controller.py:463-464, 683).
- **Interacts with:** `reference.blend_from` (defines the early region it checks against); mostly relevant to knots mode.
- **Example:** `authority_low_m: 0.5` (configs/kv3.yaml:135)

### `ffn_pdr_controller.alarm_consecutive`
- **Type:** int
- **Default:** `3` (body_lr_controller.py:214)
- **Values/constraints:** Must be `>= 1` (train_mara.py:5091-5092).
- **What it does:** Number of consecutive qualifying samples required before either the lower-rail (`alarm`) or upper-rail (`upper_alarm`) latches (body_lr_controller.py:471, 481, 619). Debounces the alarms against single noisy pdr readings.
- **Interacts with:** `alarm_pdr_ratio`, `upper_alarm_margin`.
- **Example:** `alarm_consecutive: 3` (configs/kv3.yaml:137)

### `ffn_pdr_controller.cadence`
- **Type:** int
- **Default:** n/a — **deprecated and ignored**
- **Values/constraints:** None enforced; if present in a config the trainer prints a note that it is ignored (train_mara.py:6345-6347). The controller explicitly documents there is no `cadence` field (body_lr_controller.py:136-138).
- **What it does:** Nothing. `observe()` is driven by the trainer at the `val_step` cadence, so **`val_step` is the control cadence**. The EMA alphas and rate limits are per-observe (per val-step) and were tuned for `val_step ~100`. A stray `cadence:` key is a no-op. (dn3.yaml:135 still carries `cadence: 100` for documentation; it has no effect.)
- **Interacts with:** `val_step` (the real cadence), `k_ema_alpha`/`pdr_ema_alpha`/`rate_*` (all per-observe).
- **Example:** `cadence: 100` (configs/dn3.yaml:135 — present but ignored; do not rely on it).

---

## Head Hygiene (gauge projection, z-loss, row-center)

The LM readout head `output.weight` (`[V, D]`) carries a **CE-invisible common-mode gauge**: adding the same scalar to every vocab row shifts all logits equally and cancels in softmax/CE. Left alone it accumulates (bloating raw `logZ`/`‖W‖`, adding bf16 precision risk). This section documents the three levers that keep the head gauge-clean or ceiling its centered log-partition. All ship **OFF/inert by default**, all require an **untied, bias-free** head, and `head_gauge_projection` vs `row_center_head` are **mutually exclusive**. Design + math live in `docs/DN4_HEAD_HYGIENE_SPEC.md`; the only in-repo config that exercises them is `configs/dn4.yaml`.

### `head_gauge_projection.enabled`
- **Type:** bool (accepts a flat top-level bool `head_gauge_projection: true`, or a nested dict `{enabled, init_row_center}`)
- **Default:** `false` (key absent → normalized to `{enabled: false, init_row_center: false}` by `_head_gauge_cfg`, train_mara.py:346)
- **Values/constraints:** When true, several hard guards fire in `Settings.__init__` (train_mara.py:5526): fatal if `tie_word_embeddings: true`; fatal unless `muon_adam_state_dtype: fp32`; fatal if `row_center_head` is also enabled (mutually exclusive). At optimizer wiring (train_mara.py:6206) it also fatals if there's no output head, if the head has a bias, if the optimizer doesn't expose a `head_gauge_ids` hook (i.e. isn't a Muon-family / `MuonFSDP2` optimizer), if 16-bit Adam is active, if the matched head param isn't 2-D, if the matched head is in a Muon (not Adam) group, or if it can't match **exactly one** head param in the optimizer's param groups.
- **What it does:** This is **Lever 1** (dn4). Each optimizer step, inside the NorMuon **non-fused** (fp32-Adam) path, it removes the vocab-row mean from the head's *applied* update `U = m̂/√v̂` before the weight step: `U ← U − 1·mean_v(U)^T` (`_project_head_update_gauge_`, muon_fsdp2.py:167; hook at 785). Because projection is linear, the CE-visible centered head `P(W)` evolves only by `P(U)` while the gauge component never accumulates. It deliberately does **not** touch `exp_avg`/`exp_avg_sq` or do post-step weight surgery (that's the rejected `row_center_head` path), and adds **no checkpointed state**. The row-mean is computed in fp32 and subtracted in place from the fp32 update (`_subtract_row_mean_`, row_center.py:98 — stochastic-rounding write-back only engages for a bf16 buffer, and this path's `U` is fp32 because the fp32-state guard forces it). Per the spec this is training-time *hygiene* — it does NOT lower loss, `logZ_c`, or `‖W_c‖` (that's Lever 2).
- **Interacts with:** `muon_adam_state_dtype` (must be `fp32` — any other value selects the fused 16-bit Adam path where `U` is never exposed → silent no-op, hence the fatal guard); `tie_word_embeddings` (must be false); `row_center_head` (mutually exclusive); `z_loss` (independent — Lever 1 vs Lever 2); `optimizer_type` (needs a Muon-family optimizer exposing `head_gauge_ids`).
- **Example:** `head_gauge_projection: { enabled: true, init_row_center: true }` (configs/dn4.yaml:137)

### `head_gauge_projection.init_row_center`
- **Type:** bool (nested key under `head_gauge_projection`)
- **Default:** `false` when using the dict form; but note the flat-bool shorthand `head_gauge_projection: true` sets **both** `enabled` and `init_row_center` to true (train_mara.py:353-354).
- **Values/constraints:** Only meaningful when `head_gauge_projection.enabled` is true. No independent validation.
- **What it does:** One-time **weight-only** row-center of the head at init, applied exactly once when `start_step == 1` and not resuming (train_mara.py:1772), via `_row_center_head_step(..., want_exp_avg=False)`. It subtracts the current vocab-row mean from `output.weight` so the run *starts* from a gauge-clean ("clean birth") head, before any optimizer state exists. On resume past step 1 it's skipped (the stored head is already centered). This is a distinct gate from the legacy `row_center_head` init path — it does no exp_avg surgery.
- **Interacts with:** `head_gauge_projection.enabled` (no effect unless enabled); `resume_training`/`start_step` (skipped on mid-run resume).
- **Example:** `init_row_center: true` (configs/dn4.yaml:139 — "one-time weight-only gauge clean at step 1")

### `z_loss.enabled`
- **Type:** bool (nested key inside the `z_loss` dict; the whole `z_loss` value must be a dict or absent)
- **Default:** `None`/off. If `z_loss` is absent, or is a dict with `enabled: false`, it collapses to `None` and the loss path is byte-for-byte identical to baseline (train_mara.py:5358). A non-dict `z_loss` is a fatal error.
- **Values/constraints:** When true, the sub-keys (`alpha`, `backend`, `target`, `tau`, `warmup`, `warmdown`) are validated (train_mara.py:5363-5446).
- **What it does:** Master switch for the log-partition (confidence) penalty on the LM head. When on, the model computes a z-loss term each **training** step (skipped in eval; model_v2.py:1941) and the trainer folds `alpha_eff(step) * z` into the loss, where `alpha_eff` comes from the resume-safe pure-function schedule `get_zloss_alpha(step, settings)` (train_mara.py:276). The penalized quantity depends on `target` (raw `mean(logZ**2)` vs the centered deadband).
- **Interacts with:** all `z_loss.*` sub-keys; `row_center_head` (raw z-loss + active row-centering would turn raw z-loss into *centered* z-loss — guarded, see `allow_row_center_with_z_loss`); `tie_word_embeddings` (required false when `target: centered`).
- **Example:** `z_loss: { enabled: false, target: centered, tau: 128, alpha: 1.0e-5, backend: fp32_accum }` (configs/dn4.yaml:144 — built and staged **off**)

### `z_loss.target`
- **Type:** string
- **Default:** `'raw'` (train_mara.py:5385; also the model-side default `_zloss_target = 'raw'`, model_v2.py:1532)
- **Values/constraints:** must be `'raw'` or `'centered'` (fatal otherwise). `'centered'` additionally **requires** a numeric non-bool `z_loss.tau` and **requires an untied head** (`tie_word_embeddings: false`) — both are fatal-checked (train_mara.py:5388-5398), because centering shapes `logZ_c` via `mu(output.weight)` and on a tied head that would also regularize the input embeddings.
- **What it does:** Selects the penalized quantity. `'raw'` penalizes `mean(logZ**2)` (the legacy confidence penalty — not gauge-invariant; it can be minimized by pushing the common-mode gauge, which is the pathology this section exists to avoid). `'centered'` is **Lever 2**: a gauge-invariant deadband ceiling `L_zc = alpha * mean(relu(logZ_c − tau)**2)` on the *centered* log-partition `logZ_c = logZ − h·mu` (model_v2.py:158 `_centered_zloss_deadband`, forward branch at 1972). The centered gradient sums to zero over vocab → zero common-mode gradient, so it shapes the real (centered) distribution without touching the gauge.
- **Interacts with:** `z_loss.tau` (required when centered); `tie_word_embeddings` (must be false when centered); `head_gauge_projection`/`row_center_head` (Lever 2 is the "centered-head control" separate from gauge hygiene).
- **Example:** `target: centered` (configs/dn4.yaml:146)

### `z_loss.tau`
- **Type:** float
- **Default:** `0.0` (model-side default `_zloss_tau = 0.0`, model_v2.py:1533; read via `settings.z_loss.get('tau', 0.0)` at train_mara.py:5921)
- **Values/constraints:** Only used/required when `target: centered`, where it must be a numeric non-bool (fatal otherwise, train_mara.py:5390). Ignored when `target: raw`. Spec reference band: `tau ≈ 120–128` (from dn3's observed `logZ_c` band; DN4 spec:177).
- **What it does:** The **deadband ceiling** on the centered log-partition `logZ_c`. The penalty is `max(0, logZ_c − tau)**2`, so as long as `logZ_c` stays under `tau` the loss is identically zero (inactive) — it's a ceiling, not constant pressure. Only when `logZ_c` exceeds `tau` does it push back. `logZ_c` runs much lower than raw `logZ` (~108 vs ~500 in dn3), which is why the centered `tau` is O(120) not O(500).
- **Interacts with:** `z_loss.target` (only meaningful when `centered`); `z_loss.alpha` (the deadband gain).
- **Example:** `tau: 128` (configs/dn4.yaml:147)

### `z_loss.alpha`
- **Type:** float (may be scheduled indirectly via `warmup`/`warmdown`)
- **Default:** `0.0` when the key is absent (`z.get('alpha', 0.0)`, train_mara.py:295); but when `z_loss.enabled` is true, `alpha` is validated and must be a **non-negative number** (fatal if missing/negative, train_mara.py:5366).
- **Values/constraints:** `>= 0`. Effective per-step value = `get_zloss_alpha(step, settings)`, which applies the warmup ramp (`0→alpha`) and warmdown ramp (`alpha→0`) around it; before warmup start it's 0, after warmdown end it's exactly 0.
- **What it does:** The scalar coefficient multiplying the z-loss term added to the training loss. For `target: raw` it weights `mean(logZ**2)`; for `target: centered` it weights the deadband `mean(relu(logZ_c − tau)**2)`. Because the centered penalty acts on the much smaller `logZ_c`, the centered `alpha` is roughly ~24x the raw equivalent per the dn4 config comment. Spec reference band for centered: `alpha ≈ 5e-6–1e-5` (DN4 spec:177).
- **Interacts with:** `z_loss.warmup`/`z_loss.warmdown` (schedule the effective alpha); `z_loss.target` (sets what it weights).
- **Example:** `alpha: 1.0e-5` (configs/dn4.yaml:148)

### `z_loss.backend`
- **Type:** string
- **Default:** `'fp32_accum'` (train_mara.py:5377 and 5917)
- **Values/constraints:** must be `'bf16'` or `'fp32_accum'` (fatal otherwise, train_mara.py:5378).
- **What it does:** Precision/memory tradeoff for the "option-D" z-loss gradient. CCE 25.4.3 has no `return_lse`, so `logZ` is reconstructed as `CE_none + target_logit` — a bf16 catastrophic-cancellation risk. `'fp32_accum'` (default) runs the CCE accumulators in fp32 in the backward → grad cosine ~0.999 vs fp32 truth, at ~+0.45 GB head memory. `'bf16'` is the lightest-memory path (grad cosine ~0.990, fine for a small annealed regularizer). The trainer maps this onto the model flag `_zloss_fp32_accum` (`None`=off | `False`=bf16 | `True`=fp32_accum; train_mara.py:5913-5917).
- **Interacts with:** `z_loss.target` (the centered path also uses this fp32-accum flag; the blind-review fix forms the centered target logit directly in fp32 to dodge the O(400) gauge-scale cancellation).
- **Example:** `backend: fp32_accum` (configs/dn4.yaml:149)

### `z_loss.warmup`
- **Type:** dict — *sibling key discovered inside the `z_loss` block* (`{enabled, start_step, duration_steps, shape}`)
- **Default:** absent → no warmup (full `alpha` applies immediately).
- **Values/constraints:** must be a dict (fatal otherwise). When `enabled: true`: `start_step` a non-negative int, `duration_steps` a **positive** int, `shape` in `{'cosine', 'linear'}` (all fatal-checked, train_mara.py:5399-5420).
- **What it does:** Ramps the z-loss coefficient `0 → alpha` over `[start_step, start_step+duration_steps)` (train_mara.py:296-311). `cosine` (default) is a zero-slope-ends onset; `linear` is `alpha * progress`. At/before `start_step` alpha is 0; at/after the end it's the full `alpha`. Pure function of the global step, so it's resume-safe with no extra checkpoint state.
- **Interacts with:** `z_loss.alpha` (the ramp target); `z_loss.warmdown` (composes as a later ramp down).
- **Example:** `warmup: { enabled: true, start_step: 0, duration_steps: 2000, shape: cosine }` (schema per get_zloss_alpha; not present in dn4.yaml)

### `z_loss.warmdown`
- **Type:** dict — *sibling key discovered inside the `z_loss` block* (`{enabled, start_step, duration_steps, shape}`)
- **Default:** absent → no warmdown.
- **Values/constraints:** same validation shape as `warmup` (dict; `start_step >= 0` int; `duration_steps > 0` int; `shape` in `{'cosine','linear'}`; train_mara.py:5425-5445).
- **What it does:** Ramps `alpha → 0` over a half-open `[start, start+duration)` window and returns **exactly 0** from `start+duration` onward (train_mara.py:316-328). This exactness is load-bearing for staged transitions: it lets raw z-loss reach exactly zero *before* `row_center_head`'s schedule goes nonzero at the same step, so the overlap guardrail (train_mara.py:5509) can prove the two are temporally disjoint.
- **Interacts with:** `z_loss.warmup` (composes multiplicatively on the post-warmup alpha); `row_center_head` + `allow_row_center_with_z_loss` (the disjoint-schedule guardrail relies on warmdown hitting exactly 0).
- **Example:** `warmdown: { enabled: true, start_step: 10000, duration_steps: 1000, shape: cosine }` (schema per get_zloss_alpha; not present in dn4.yaml)

### `row_center_head.enabled`
- **Type:** bool (accepts a flat top-level bool `row_center_head: true` for steady-state, or a nested dict `{enabled, warmup}` for staged warmup)
- **Default:** `false` (key absent → `False`; `_row_center_cfg` normalizes to `{enabled: false, warmup: None}`, train_mara.py:332). A value that is neither bool nor dict is a fatal error.
- **Values/constraints:** When enabled, **fatal** if `tie_word_embeddings: true` (train_mara.py:5495 — subtracting the row-mean from a tied head also shifts the input embeddings, not function-preserving). Also **mutually exclusive** with `head_gauge_projection` (fatal, checked from the head_gauge side at 5537). And a **step-active overlap** guard with `z_loss`: fatal if any global step has both row-center `s(step) > 0` and z-loss `alpha_eff(step) > 0` (unless `allow_row_center_with_z_loss: true`), because that would make raw z-loss behave as centered z-loss (train_mara.py:5509).
- **What it does:** This is the **legacy** head-gauge lever. Each step (post-`optimizer.step()`, under eager, after `capture_updates()`) it subtracts the vocab-row mean `mu` from `output.weight` **and** from the Adam first moment `exp_avg` (`_row_center_head_step`, train_mara.py:1257 → `row_center_head_`, row_center.py:120; per-step call at train_mara.py:2632), removing the CE-invisible gauge and preventing it from regrowing. Function-preserving on the model *output* but not on the optimizer trajectory (hence the staged warmup). The DN4 spec rejects the exp_avg projection as unclean (a row-mean in `m` isn't pure gauge after `/√v`) — `head_gauge_projection` supersedes it. In dn4 this is kept **off**.
- **Interacts with:** `head_gauge_projection` (mutually exclusive — enable exactly one); `z_loss` (step-active overlap guard, plus `allow_row_center_with_z_loss` override); `tie_word_embeddings` (must be false); `row_center_head.warmup` (staged ramp).
- **Example:** `row_center_head: { enabled: false }` (configs/dn4.yaml:153 — the failed retrofit, left off)

### `row_center_head.warmup`
- **Type:** dict — *sibling key discovered under `row_center_head`* (`{enabled, start_step, duration_steps, shape}`)
- **Default:** absent → steady-state full projection from step 0 (flat-bool form implies `s ≡ 1`).
- **Values/constraints:** must be a dict (fatal otherwise). When `enabled: true`: `start_step >= 0` int, `duration_steps > 0` int, `shape` in `{'cosine','linear'}` (train_mara.py:5466-5479).
- **What it does:** Staged **target-gauge** warmup for mid-run adoption (avoids the cold-projection optimizer shock). A schedule scalar `s(step) ∈ [0,1]` (`get_row_center_s`, train_mara.py:482) pins the stored gauge to `(1−s)·mu0` (and exp_avg to `(1−s)·mbar0`), where `mu0`/`mbar0` are captured once at `start_step` and checkpoint-persisted. `s=0` is a no-op (gauge stays at captured value), `s=1` is fully centered. When warmup is on, the hard pre-first-forward projection is **suppressed** — the schedule is the sole path to centered (train_mara.py:1752).
- **Interacts with:** `row_center_head.enabled`; `z_loss` (the overlap guard scans this schedule's active range vs z-loss alpha's).
- **Example:** `row_center_head: { enabled: true, warmup: { enabled: true, start_step: 8000, duration_steps: 2000, shape: cosine } }` (schema per get_row_center_s; not present in dn4.yaml)

### `allow_row_center_with_z_loss`
- **Type:** bool — *sibling key discovered in the same validation block* (top-level, not nested)
- **Default:** `false` (train_mara.py:5486)
- **Values/constraints:** No type validation beyond the default; treated as a boolean escape hatch.
- **What it does:** Escape hatch that **disables** the fatal step-active overlap guard between `row_center_head` and `z_loss` (train_mara.py:5509). When true, a run may have both row-center `s>0` and z-loss `alpha_eff>0` at the same step — which deliberately turns raw z-loss into **centered** z-loss (a real regularizer, not the inert gauge). Reserved for a deliberate ablation; the startup banner explicitly flags it as `allow_row_center_with_z_loss=TRUE — z-loss runs as CENTERED z-loss (deliberate ablation)` (train_mara.py:6142).
- **Interacts with:** `row_center_head` + `z_loss` (only relevant when both are enabled with overlapping schedules).
- **Example:** `allow_row_center_with_z_loss: true` (ablation only; not in dn4.yaml)

### `transition_health_guard`
- **Type:** bool — *sibling key discovered in the same validation block* (top-level)
- **Default:** `false` (train_mara.py:5490; fatal if set to a non-bool)
- **Values/constraints:** Must be a bool.
- **What it does:** Guardrail 5 — an **advisory** transition-health monitor for staged head-hygiene transitions (z-loss warmdown into row-center warmup). It emits WARNING logs only; it takes **no auto-action**. Off by default; flipped on for a deliberate staged transition to get health telemetry during the handoff. In dn4 it's set true even though the levers it monitors are staged off.
- **Interacts with:** `z_loss.warmdown` / `row_center_head.warmup` (the staged transition it watches).
- **Example:** `transition_health_guard: true` (configs/dn4.yaml:156)

---

## Adaptive Weight Decay (AWD)

AWD is an optional closed-loop controller (`adaptive_wd.py`, class `AdaptiveWD` — a bare top-level module in the `mara_fsdp2` repo, imported via `from adaptive_wd import AdaptiveWD`, train_mara.py:6283; it is NOT under `common_fsdp2/`) that watches per-component gradient or weight norms and *multiplies* the base weight decay for that component — it never sets WD directly. In the training loop the order is: clip -> lr_mods -> `awd.compute_and_update(step)` -> WD rules populate `wd_overrides` -> `awd.apply_multipliers()` -> `optimizer.step()` (train_mara.py:2174-2191). The whole feature is off unless the `adaptive_wd` dict has `enabled: true`, and it is **mutually exclusive** with `ffn_pdr_controller` (both drive the body; train_mara.py:5276-5278 hard-errors if both are set). Note: only `check_interval`/`smoothing`/`groups` and the per-group fields are read (with `.get()` defaults) inside `AdaptiveWD.__init__`; the top-level `enabled` is consumed earlier in Settings.__init__ (train_mara.py:5050-5055 collapses the block to `None` when `enabled` is falsey) and is never re-read by `AdaptiveWD`. Because the per-group fields use `.get()` defaults with no validation, a typo'd group key silently takes its default rather than erroring.

### `adaptive_wd`
- **Type:** dict (or absent/None)
- **Default:** `None` (train_mara.py:5051-5055 sets `None` if the key is absent, or if present but `enabled` is not true)
- **Values/constraints:** Must be a dict when present. Recognized keys: `enabled`, `check_interval`, `smoothing`, `groups`. Cannot coexist with `ffn_pdr_controller` (train_mara.py:5276-5278 -> `fatal_error`). `base_wd` handed to the controller is the scalar `weight_decay` if it is a number, else `0.0` (train_mara.py:6285) — so if `weight_decay` is a *rules list*, AWD's fallback base is 0 and it only scales params that already have a positive per-param WD from the rules.
- **What it does:** The container for the whole AWD subsystem. When non-None, train_mara.py constructs `AdaptiveWD(raw_model, settings.adaptive_wd, ddp_rank, ddp_world_size, ddp, base_wd, wd_overrides)` (train_mara.py:6282-6286) and logs `"Adaptive WD: N groups, check every K steps"` (train_mara.py:6288). State (EMA, multipliers, prev_w_norms, PID integral/prev_pv) is checkpointed and restored on resume (train_mara.py:3429-3431, 3786-3789; `_STATE_VERSION = 3`).
- **Interacts with:** `weight_decay` (base it multiplies; scalar vs rules changes the fallback base), `ffn_pdr_controller` (mutually exclusive), `wd_overrides` side-dict.
- **Example:**
  ```yaml
  adaptive_wd:
    enabled: true
    check_interval: 50
    groups: [...]
  ```
  (from `configs/mf2-adam.yaml`)

### `adaptive_wd.enabled`
- **Type:** bool
- **Default:** `False` (via `.get('enabled', False)`, train_mara.py:5054)
- **Values/constraints:** Only `true` activates AWD; any falsey value collapses the whole block to `None`.
- **What it does:** Master on/off. It is read only in Settings.__init__ (train_mara.py:5054) to decide whether to keep the dict; `AdaptiveWD` itself never reads it (by the time it is constructed, the block is known-enabled).
- **Interacts with:** `adaptive_wd` (gates the whole block).
- **Example:** `enabled: true` (from `configs/snake-egg.yaml`)

### `adaptive_wd.check_interval`
- **Type:** int
- **Default:** `50` (adaptive_wd.py:66)
- **Values/constraints:** Steps between updates. No explicit range validation. Cold-start special case: `compute_and_update` fires immediately on the first call regardless of interval while there is no EMA state yet (adaptive_wd.py:271: `if step % self.check_interval != 0 and len(self._ema) > 0`), so a fresh run/resume seeds metrics right away.
- **What it does:** `compute_and_update(step)` returns early (no update) unless `step % check_interval == 0` — except on cold start (adaptive_wd.py:271). On a firing step it computes gradient norms and/or weight norms (with a batched `all_reduce(SUM)` across FSDP ranks, adaptive_wd.py:258/293) and updates each group's multiplier. `growth_rate`/`out_emb_growth` metrics measure the *change* in weight norm across one interval, so this also sets the growth window.
- **Interacts with:** `groups[].metric` (growth metrics use the interval as their delta window), `smoothing` (EMA is applied per firing).
- **Example:** `check_interval: 50` (from `configs/mf3-adam.yaml`)

### `adaptive_wd.smoothing`
- **Type:** float
- **Default:** `0.9` (adaptive_wd.py:67)
- **Values/constraints:** EMA factor in `[0,1)`; higher = more smoothing. No explicit validation. **Applies only to the threshold metrics** (`g_norm`, `ratio`, `growth_rate`, `out_emb_growth`): `ema = smoothing*ema + (1-smoothing)*raw` (adaptive_wd.py:452). The `w_rms_target` PID path ignores this and uses its own `pid_smoothing` instead (adaptive_wd.py:387-397).
- **What it does:** Smooths the raw metric before it is compared to engage/ease thresholds, damping reaction to per-interval noise.
- **Interacts with:** `groups[].metric` (only affects the four non-PID metrics), `pid_smoothing` (the PID analog for `w_rms_target`).
- **Example:** `smoothing: 0.9` (documented default in `configs/_optimizer_reference.yaml`:167; the shipped `w_rms_target` configs omit it since it is unused by PID)

### `adaptive_wd.groups`
- **Type:** list of dicts
- **Default:** `[]` (adaptive_wd.py:71 — no groups means AWD does nothing)
- **Values/constraints:** Each element is a group config parsed into a fixed schema (adaptive_wd.py:72-100). Multiple groups may target overlapping components; `_component_to_groups` collects all groups per component and each is processed (adaptive_wd.py:344-346). A single component ends up with one multiplier that the last-processed matching group writes.
- **What it does:** Declares which components to control, with what metric/target, and with what actuator limits. `_match_groups` (adaptive_wd.py:217) resolves each group's `target`/`sublayer` to concrete component names (`emb`, `out`, `L{i}.attn`, `L{i}.ffn`).
- **Interacts with:** all `groups[].*` sub-fields below.
- **Example:**
  ```yaml
  groups:
    - target: [0, 71]
      sublayer: all
      metric: w_rms_target
      target_value: 0.3
      min_wd_multiplier: 0.2
      max_wd_multiplier: 14.0
      kp: 90.0
      ki: 50.0
      kd: 0.0
      pid_smoothing: 0.0
  ```
  (from `configs/mf3-adam.yaml`)

### `adaptive_wd.groups[].target`
- **Type:** string or 2-element list
- **Default:** required (`g['target']` is accessed with no default, adaptive_wd.py:73 — a missing `target` raises KeyError)
- **Values/constraints:** `"emb"`, `"out"`, or `[start, end]` an **inclusive** layer range (adaptive_wd.py:224-240). `"out"` only resolves to a component when the model is untied (tied `output.weight is tok_embeddings.weight` -> no `out` component, adaptive_wd.py:158-163). Out-of-range or otherwise unmatched targets yield an empty component list (group is silently inert).
- **What it does:** Selects which model component(s) this group controls. For a `[start,end]` range, each layer index `i` in the range contributes `L{i}.attn` and/or `L{i}.ffn` depending on `sublayer`.
- **Interacts with:** `sublayer` (only meaningful for range targets).
- **Example:** `target: [54, 71]` (a terminal-layer band, from `configs/snake-egg.yaml`); also `target: out`, `target: emb`.

### `adaptive_wd.groups[].sublayer`
- **Type:** string
- **Default:** `'all'` (adaptive_wd.py:74)
- **Values/constraints:** `'all'`, `'attn'`, or `'ffn'`. Only consulted for `[start,end]` range targets (adaptive_wd.py:233-240); ignored for `emb`/`out`.
- **What it does:** Within a layer range, restricts control to attention blocks (`L{i}.attn`), FFN blocks (`L{i}.ffn`), or both. `L{i}.attn` bundles wq/wk/wv/wo (+ g_proj if gated / GDN projections); `L{i}.ffn` bundles w1/w2/w3 (or MoE expert + shared-expert params) (adaptive_wd.py:166-208).
- **Interacts with:** `target` (range form).
- **Example:** `sublayer: all` (from `configs/mf2-adam.yaml`)

### `adaptive_wd.groups[].metric`
- **Type:** string
- **Default:** `'g_norm'` (adaptive_wd.py:75)
- **Values/constraints:** One of `g_norm`, `ratio`, `growth_rate`, `out_emb_growth`, `w_rms_target` (adaptive_wd.py:350-439). Anything else -> the component is skipped (`else: continue`, adaptive_wd.py:439-440). Determines which inputs are gathered: `g_norm`/`ratio` need gradients; `growth_rate`/`out_emb_growth`/`w_rms_target` need weight norms (adaptive_wd.py:104-106).
- **What it does:**
  - `g_norm`: raw per-component gradient norm (adaptive_wd.py:351).
  - `ratio`: component gradient norm / mean layer gradient norm — relative concentration (adaptive_wd.py:353).
  - `growth_rate`: this component's Δw_norm / mean layer Δw_norm over one interval; skipped if mean growth ≤ 0 (adaptive_wd.py:358-361).
  - `out_emb_growth`: out Δw_norm / emb Δw_norm; skipped if emb growth ≤ 0 (adaptive_wd.py:362-365).
  - `w_rms_target`: PID setpoint controller driving `w_rms = w_norm/sqrt(num_params)` toward `target_value` (adaptive_wd.py:366-437). This is the only mode used in the shipped `w_rms_target` configs.
  The first four are "threshold" metrics using `engage_above`/`ease_below`/`aggression`/`recovery`; `w_rms_target` uses the PID gains instead and bypasses the global EMA.
- **Interacts with:** dictates which of the other group fields matter (`target_value` vs `target_ratio`; threshold vs PID fields).
- **Example:** `metric: w_rms_target` (from `configs/snake-egg.yaml`); reference config also shows `metric: g_norm` and `metric: ratio`.

### `adaptive_wd.groups[].target_value`
- **Type:** float (or None)
- **Default:** `None` (adaptive_wd.py:76)
- **Values/constraints:** Used by `g_norm` and `w_rms_target` metrics. For `w_rms_target` it must be `> 0` or the component is skipped (adaptive_wd.py:381). For `g_norm` a `None` target skips adjustment (adaptive_wd.py:459).
- **What it does:** The setpoint. In `w_rms_target` mode it is the desired `w_norm/sqrt(num_params)`; error = `(pv - target_value)/target_value` feeds the PID (adaptive_wd.py:400). In `g_norm` mode it is the reference the "excess" is measured against once `engage_above` is crossed (adaptive_wd.py:466). Changing it on resume is safe because the D term uses `prev_pv` not `prev_error`.
- **Interacts with:** `metric` (`g_norm`/`w_rms_target`), PID gains, `engage_above`/`ease_below` (g_norm).
- **Example:** `target_value: 0.7` (from `configs/mf2-adam.yaml`); `0.3` in mf3/snake-egg.

### `adaptive_wd.groups[].target_ratio`
- **Type:** float (or None)
- **Default:** `None` (adaptive_wd.py:77)
- **Values/constraints:** Used by the ratio-style threshold metrics: `ratio`, `growth_rate`, `out_emb_growth` (adaptive_wd.py:458). `None` -> no adjustment.
- **What it does:** The reference against which "excess" is measured for ratio-family metrics; once `smoothed > engage_above`, `mult *= (1 + aggression*(smoothed - target_ratio))` (adaptive_wd.py:466-467). Ignored by `g_norm` (uses `target_value`) and `w_rms_target` (uses PID).
- **Interacts with:** `metric` (ratio/growth_rate/out_emb_growth), `engage_above`, `aggression`.
- **Example:** `target_ratio: 8.0` (from the reference-config `ratio` group in `configs/_optimizer_reference.yaml`:194)

### `adaptive_wd.groups[].tolerance`
- **Type:** float (or None)
- **Default:** `None` (adaptive_wd.py:78)
- **Values/constraints:** Parsed into the group dict but **never read** anywhere in `adaptive_wd.py`. It is effectively a dead/reserved field.
- **What it does:** Nothing in the current code — parsed and stored but unused. Setting it has no effect (engage/ease are governed by `engage_above`/`ease_below`, not `tolerance`).
- **Interacts with:** — (inert)
- **Example:** not used in any shipped config; safe to omit.

### `adaptive_wd.groups[].engage_above`
- **Type:** float (or None)
- **Default:** `None` (adaptive_wd.py:79)
- **Values/constraints:** Threshold metrics only (`g_norm`, `ratio`, `growth_rate`, `out_emb_growth`). If `None` (or `ease_below` is None), the threshold adjustment is skipped entirely (adaptive_wd.py:463-464: `if engage is None or ease is None: continue`). Ignored by `w_rms_target`.
- **What it does:** When the smoothed metric exceeds `engage_above`, AWD *increases* the multiplier: `mult *= (1 + aggression*excess)` where `excess = smoothed - target` (adaptive_wd.py:465-467). Sets the upper trigger of the control band.
- **Interacts with:** `ease_below` (lower trigger), `aggression`, `target_value`/`target_ratio`.
- **Example:** `engage_above: 10.0` (ratio group, `configs/_optimizer_reference.yaml`:195)

### `adaptive_wd.groups[].ease_below`
- **Type:** float (or None)
- **Default:** `None` (adaptive_wd.py:80)
- **Values/constraints:** Threshold metrics only. If `None` (or `engage_above` is None), threshold adjustment is skipped (adaptive_wd.py:463-464). Ignored by `w_rms_target`.
- **What it does:** When the smoothed metric drops below `ease_below`, AWD *decreases* the multiplier by `mult *= recovery` (adaptive_wd.py:468-469). Between `ease_below` and `engage_above` is the dead zone, where the multiplier drifts back toward 1.0 (adaptive_wd.py:470-476).
- **Interacts with:** `engage_above`, `recovery`.
- **Example:** `ease_below: 5.0` (ratio group, `configs/_optimizer_reference.yaml`:196)

### `adaptive_wd.groups[].min_wd_multiplier`
- **Type:** float
- **Default:** `0.2` (adaptive_wd.py:81)
- **Values/constraints:** Lower clamp on the multiplier. Set `< 1.0` to make AWD *bidirectional* (it can lower WD to let underdeveloped layers grow), or `1.0` for a one-way ratchet (only ever increases WD). For the legacy (non-PID) `w_rms_target` path it also seeds the cold-start multiplier (adaptive_wd.py:130-139) — but that block is a no-op when `kp>0` or `ki>0` (i.e. PID enabled; adaptive_wd.py:133).
- **What it does:** Applied as the floor in `mult = max(min_wd_multiplier, min(mult, max_wd_multiplier))` for every metric (adaptive_wd.py:435, 478). In the PID path it also drives anti-windup: if `mult < min` and `error < 0`, the integral is frozen (adaptive_wd.py:427-428).
- **Interacts with:** `max_wd_multiplier`, PID anti-windup, `weight_decay` base (final WD = base * mult).
- **Example:** `min_wd_multiplier: 0.2` (from `configs/mf2-adam.yaml`)

### `adaptive_wd.groups[].max_wd_multiplier`
- **Type:** float
- **Default:** `15.0` (adaptive_wd.py:82)
- **Values/constraints:** Upper clamp. Shipped configs use `7.0` (mf2) or `14.0` (mf3/snake-egg); reference doc examples use 3–10.
- **What it does:** Ceiling in `mult = max(min, min(mult, max_wd_multiplier))` (adaptive_wd.py:435, 478). PID anti-windup: if `mult > max` and `error > 0`, the integral is frozen so it does not wind up while saturated (adaptive_wd.py:425-426). This is the hard cap on how aggressively AWD can decay a runaway component.
- **Interacts with:** `min_wd_multiplier`, PID `integral_max`/anti-windup, `weight_decay` base.
- **Example:** `max_wd_multiplier: 14.0` (from `configs/mf3-adam.yaml`)

### `adaptive_wd.groups[].aggression`
- **Type:** float
- **Default:** `0.1` (adaptive_wd.py:83)
- **Values/constraints:** Threshold metrics only (`g_norm`/`ratio`/`growth_rate`/`out_emb_growth`); unused by `w_rms_target`.
- **What it does:** Growth rate of the multiplier when engaged: `mult *= (1 + aggression * excess)` per firing (adaptive_wd.py:467). Larger = faster WD increase above the engage threshold.
- **Interacts with:** `engage_above`, `target_value`/`target_ratio`, `recovery`.
- **Example:** `aggression: 0.15` (out-head g_norm group, `configs/_optimizer_reference.yaml`:177)

### `adaptive_wd.groups[].recovery`
- **Type:** float
- **Default:** `0.95` (adaptive_wd.py:84)
- **Values/constraints:** Threshold metrics only; typically slightly `< 1.0`. Unused by `w_rms_target`.
- **What it does:** Multiplicative decay applied when the metric is below `ease_below` (`mult *= recovery`, adaptive_wd.py:469) and, in the dead zone, to pull the multiplier back toward 1.0 from either side (`mult *= recovery` if `>1`, `mult = min(mult/recovery, 1.0)` if `<1`; adaptive_wd.py:470-476). Closer to 1.0 = slower relaxation.
- **Interacts with:** `ease_below`, `aggression`.
- **Example:** `recovery: 0.98` (emb g_norm group, `configs/_optimizer_reference.yaml`:188)

### `adaptive_wd.groups[].kp`
- **Type:** float
- **Default:** `90.0` (adaptive_wd.py:95)
- **Values/constraints:** `w_rms_target` (PID) only. High by design because the normalized error is small (e.g. 0.67 when w_rms is 1.5x target), so a large gain is needed to produce meaningful multiplier force (adaptive_wd.py:85-95 comment). If `kp>0` or `ki>0`, the group is treated as PID-configured and the legacy cold-start seeding is skipped (adaptive_wd.py:133).
- **What it does:** Proportional term in `mult = 1 + kp*error + ki*integral + kd*d_pv`, where `error = (pv - target_value)/target_value` (adaptive_wd.py:400, 420). Immediate response to the current deviation.
- **Interacts with:** `ki`, `kd`, `integral_max`, `pid_smoothing`, `target_value`, `min/max_wd_multiplier`.
- **Example:** `kp: 90.0` (from `configs/mf2-adam.yaml`)

### `adaptive_wd.groups[].ki`
- **Type:** float
- **Default:** `50.0` (adaptive_wd.py:96)
- **Values/constraints:** `w_rms_target` only. Paired with a large `integral_max` so the integral can reach the steady-state multiplier (comment cites ~11x for target 0.09, base_wd 0.1). Anti-windup freezes accumulation while the output is clamped (adaptive_wd.py:425-428).
- **What it does:** Integral gain. Integral accumulates `error` per firing, clamped to `[-integral_max, integral_max]` (adaptive_wd.py:411), and contributes `ki * integral` to the multiplier (adaptive_wd.py:420). Eliminates steady-state offset — the term that holds the long-run WD level.
- **Interacts with:** `integral_max` (clamp), `kp`, `kd`, anti-windup via `min/max_wd_multiplier`.
- **Example:** `ki: 50.0` (from `configs/mf3-adam.yaml`)

### `adaptive_wd.groups[].kd`
- **Type:** float
- **Default:** `0.0` (adaptive_wd.py:97)
- **Values/constraints:** `w_rms_target` only. Every shipped config sets `0.0`; the code/comment note derivative action does not help because the plant is slow at ~100-step intervals (adaptive_wd.py:93-94).
- **What it does:** Derivative gain, applied to the **rate of change of PV**, not error: `d_pv = (pv - prev_pv)/target_value`, contributing `kd * d_pv` (adaptive_wd.py:416-420). Using PV (not error) avoids a setpoint kick if `target_value` is changed on resume. Would brake overshoot if nonzero.
- **Interacts with:** `kp`, `ki`, `pid_smoothing` (PV path), `target_value`.
- **Example:** `kd: 0.0` (from `configs/snake-egg.yaml`)

### `adaptive_wd.groups[].integral_max`
- **Type:** float
- **Default:** `50.0` (adaptive_wd.py:98)
- **Values/constraints:** `w_rms_target` only; the symmetric anti-windup clamp `[-integral_max, +integral_max]` on the accumulated integral (adaptive_wd.py:411).
- **What it does:** Bounds how far the integral can wind, which (with `ki`) caps the integral's contribution to the multiplier and limits recovery lag after a long saturation. Must be large enough that `ki*integral_max` can reach the desired steady-state multiplier or the controller cannot hold target.
- **Interacts with:** `ki` (product sets integral authority), `min/max_wd_multiplier` (saturation-driven freeze).
- **Example:** `integral_max: 50.0` — shipped configs omit it (rely on the 50.0 default); documented in the module.

### `adaptive_wd.groups[].pid_smoothing`
- **Type:** float
- **Default:** `0.0` (adaptive_wd.py:99)
- **Values/constraints:** `w_rms_target` only. `0.0` = use raw `w_rms` as the PID process variable (fastest response); `>0` applies an EMA `pv = a*pv + (1-a)*raw` (adaptive_wd.py:387-396). **Note:** `configs/_optimizer_reference.yaml`:156 claims a default of 0.6, but the actual code default is `0.0` — trust the code. This is the PID analog of the top-level `smoothing` (which the PID path ignores).
- **What it does:** Smooths the measured `w_rms` before it enters the PID error/derivative computation. Raise it only if measurement noise causes controller oscillation (adaptive_wd.py:384-386 comment); the shipped configs all use 0.0 because `w_rms` is a slow, stable weight-norm signal.
- **Interacts with:** `kp`/`kd` (they act on the smoothed PV), `smoothing` (separate, for threshold metrics).
- **Example:** `pid_smoothing: 0.0` (from `configs/mf2-adam.yaml`)

---

## Mixture of Experts (MoE)

Token-choice top-K MoE with DeepSeek-style sigmoid routing, an optional always-on shared expert, aux-loss-free (bias-based) load balancing, capacity-based token dropping, and Expert Parallel (2D FSDP+EP mesh). Every setting here is a *model-config* field defined on the `ModelArgs` dataclass in `model_v2.py:346-364` — none are given defaults inside `Settings.__init__` (that constructor just does a generic `setattr(self, key, value)` for every YAML key at `train_mara.py:4936-4940`, so any `moe_*` key present in the YAML lands on the settings object verbatim). The `ModelArgs(...)` build block at `train_mara.py:5847-5893` then forwards **most** of them into the model config via `getattr(settings, 'moe_*', <default>)`.

**⚠️ Wiring gap (real bug, not just a doc issue):** three fields — `moe_aux_balance_coeff`, `moe_bias_before_score`, and `moe_shared_overlap` — are read by `model_v2.py` from `ModelArgs` (`args.moe_aux_balance_coeff` at `model_v2.py:1064`, `args.moe_bias_before_score` at `1065`, `getattr(args, 'moe_shared_overlap', ...)` at `1095`) but are **NOT passed into the `ModelArgs(...)` constructor** in `train_mara.py` (they are absent from the build block at `5865-5879`, and nothing patches `model_cfg` afterward). So from a mara config these three YAML keys are silently ignored — the model always uses the dataclass defaults (`0.0`, `False`, `False`) no matter what the YAML says. The only thing that reads the YAML value for two of them is a rank-0 log line (`train_mara.py:6428-6431`) which reads `model_cfg.*` — i.e. it reports the dataclass default, not the YAML value. For all other MoE settings the YAML default and the dataclass default are the same. Per-layer MoE-vs-dense selection happens in `TransformerBlock.__init__` (`model_v2.py:1353-1362`); routing/experts live in `TokenChoiceTopKRouter`, `GroupedExperts`, and `MoE` (`model_v2.py:899-1258`).

### `moe_enabled`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true`/`false`. When `false`, all other `moe_*` keys are inert. Setting `ep_degree > 1` without `moe_enabled: true` is a hard config error (`fatal_error`, `train_mara.py:4248-4253`).
- **What it does:** Master switch for the whole feature. When true, each `TransformerBlock` whose layer index passes the head/tail/interleave test replaces its dense `FeedForward` with an `MoE` module (`model_v2.py:1357-1364`); `_ffn()` then dispatches to `self.moe` instead of `self.feed_forward` (`model_v2.py:1385-1387`). Also gates checkpoint save/restore of expert weights and `expert_bias`, the load-balancing hook, and the EP mesh setup.
- **Interacts with:** every other setting in this section; `ep_degree` (EP requires it); `inner_dim` (dense-layer FFN size).
- **Example:** `moe_enabled: true` (from `configs/keel-moe.yaml`)

### `moe_num_experts`
- **Type:** int
- **Default:** `8`
- **Values/constraints:** Total (global) expert count. Under EP it must be divisible by `ep_degree` and `>= ep_degree`, else hard errors (`train_mara.py:4255-4268`). Each rank holds `num_local_experts = num_experts // ep_degree` (`model_v2.py:1058`).
- **What it does:** Sets how many experts the router chooses among and how many expert weight-slots exist. Expert weights are 3D `nn.Parameter`s in `GroupedExperts` — `w1`/`w3` of shape `(num_local_experts, expert_hidden, dim)` and `w2` of shape `(num_local_experts, dim, expert_hidden)` (`model_v2.py:904-909`); the router's gate is `Linear(dim, num_experts)` and always sees all experts even under EP (`model_v2.py:1061-1062`). More experts increases total params but not active params per token (only `top_k` fire).
- **Interacts with:** `ep_degree` (divisibility), `moe_top_k` (fraction active), `moe_capacity_factor` (capacity = f(N·top_k/num_experts)), `moe_load_balance_coeff` (`expert_bias` buffer is sized `num_experts`).
- **Example:** `moe_num_experts: 16` (from `configs/MegaMoe.yaml`); `7` in `configs/keel-moe.yaml`

### `moe_top_k`
- **Type:** int
- **Default:** `2`
- **Values/constraints:** `1 <= top_k <= num_experts`. No explicit validation in code; a bad value would fail inside `torch.topk`.
- **What it does:** Number of experts each token is routed to. The router takes `topk(scores_for_choice, k=top_k)` per token (`model_v2.py:977`); each token's hidden state is duplicated `top_k` times through the sorted-token machinery (`token_indices_sorted // top_k`, `model_v2.py:1178`), and the `top_k` expert outputs are summed (score-before) or score-weighted-summed (score-after) back into one vector (`model_v2.py:1246-1254`). Drives active-params-per-token = shared + `top_k`×expert (`_compute_active_params`, `train_mara.py:4561-4590`).
- **Interacts with:** `moe_num_experts`, `moe_capacity_factor` (capacity ∝ top_k), `moe_score_before_experts`/`moe_route_norm`/`moe_route_scale` (how the `top_k` scores are combined).
- **Example:** `moe_top_k: 2` (both real MoE configs)

### `moe_num_shared_experts`
- **Type:** int
- **Default:** `1`
- **Values/constraints:** `0` disables the shared expert; `>0` builds one. Note it is NOT a count of separate modules — it scales the *width* of a single shared FFN.
- **What it does:** When `>0`, builds one dense `FeedForward` of hidden size `expert_hidden * moe_num_shared_experts` that runs on every token and is added to the routed output (`model_v2.py:1067-1070`, `1256-1257`). So `2` means one shared FFN twice as wide, not two experts. When `0`, `self.shared_experts is None` and only routed experts contribute.
- **Interacts with:** `moe_inner_dim`/`inner_dim` (sets `expert_hidden`, the base width), `moe_shared_overlap` (only meaningful when a shared expert exists under EP).
- **Example:** `moe_num_shared_experts: 1` (both real MoE configs)

### `moe_inner_dim`
- **Type:** int (nullable)
- **Default:** `null` / `None`
- **Values/constraints:** Expert FFN hidden dim. When `None`, falls back to `args.inner_dim`, then to `_compute_default_inner_dim(dim)` (`model_v2.py:1053`). Under EP with divisible experts there's no constraint on this value itself, but configs pick it divisible by expert count for clean sharding.
- **What it does:** Hidden width of each routed expert's SwiGLU FFN (`w1/w3: (E, inner, dim)`, `w2: (E, dim, inner)`). Also the base width the shared expert is scaled from. Lets MoE experts be narrower/wider than the dense layers' `inner_dim`; logged as "Expert hidden" (`train_mara.py:5946-5947`).
- **Interacts with:** `inner_dim` (fallback + dense-layer width), `moe_num_shared_experts` (shared width = this × count), `moe_num_experts` (total expert params).
- **Example:** `moe_inner_dim: 3072` (from `configs/MegaMoe.yaml`); `768` in `configs/keel-moe.yaml`

### `moe_score_func`
- **Type:** string
- **Default:** `"sigmoid"`
- **Values/constraints:** `"sigmoid"` or `"softmax"`. Any value other than `"sigmoid"` takes the softmax branch (no explicit whitelist validation — `model_v2.py:966-969`).
- **What it does:** How router gate logits become routing scores. `"sigmoid"` applies per-expert `sigmoid(logits.float())` (DeepSeek-V3 style, independent per expert); any other value applies `softmax(logits.float(), dim=1)` (competitive across experts). Scores are computed in fp32 regardless (`model_v2.py:966-969`).
- **Interacts with:** `moe_route_norm` (renormalizing top-k scores matters more for sigmoid, which doesn't sum to 1), `moe_bias_before_score` (bias shifts the sigmoid/softmax operating point), `moe_route_scale`.
- **Example:** `moe_score_func: sigmoid` (both real MoE configs)

### `moe_score_before_experts`
- **Type:** bool
- **Default:** `true`
- **Values/constraints:** `true`/`false`.
- **What it does:** Controls where the routing score multiplies the token. When `true`, the score scales the expert *input* (`routed_input = (routed_input.float() * scores_sorted.unsqueeze(1)).to(x.dtype)`, `model_v2.py:1179-1180`) and expert outputs are plain-summed (`model_v2.py:1248-1249`). When `false`, experts see the unscaled token and outputs are score-weighted via a `bmm` at combine time (`model_v2.py:1250-1254`). Mathematically similar but differs in where the scaling/precision lands.
- **Interacts with:** `moe_route_norm`, `moe_route_scale` (they shape the same score), `moe_top_k`.
- **Example:** `moe_score_before_experts: true` (dataclass default; not overridden in the real configs)

### `moe_route_norm`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true`/`false`.
- **What it does:** When `true`, the selected top-k scores are renormalized to sum to 1 per token before use: `top_scores = top_scores / (top_scores.sum(-1, keepdim=True) + 1e-20)` (`model_v2.py:980-981`). Mainly relevant for sigmoid scoring, where the raw top-k scores don't sum to 1; with softmax they already roughly do. Applied before `moe_route_scale`.
- **Interacts with:** `moe_score_func` (most useful with `sigmoid`), `moe_route_scale` (applied after norm), `moe_top_k`.
- **Example:** `moe_route_norm: false` (dataclass default; not set in the real configs)

### `moe_route_scale`
- **Type:** float
- **Default:** `1.0`
- **Values/constraints:** Any float; a multiplicative gain on routing scores.
- **What it does:** Multiplies the selected top-k scores after optional normalization: `top_scores = top_scores * route_scale` (`model_v2.py:982`). Uniformly scales the contribution of routed experts (an overall gain on the MoE branch's magnitude). `1.0` is a no-op.
- **Interacts with:** `moe_route_norm` (norm runs first), `moe_score_func`, `moe_score_before_experts`.
- **Example:** `moe_route_scale: 1.0` (dataclass default; not set in the real configs)

### `moe_load_balance_coeff`
- **Type:** float (nullable)
- **Default:** `1e-3`
- **Values/constraints:** non-`None` enables aux-loss-free balancing; `null`/`None` disables it entirely (no `expert_bias` buffer is created — `model_v2.py:1075-1080`). This IS forwarded from settings (unlike `moe_aux_balance_coeff`/`moe_bias_before_score`).
- **What it does:** Step size for DeepSeek's aux-loss-free load balancing. An `expert_bias` fp32 buffer (size `num_experts`) is nudged each optimizer step by `coeff * sign(mean(tokens_per_expert) - tokens_per_expert)`, then de-meaned (`delta = delta - delta.mean()`), in the balance hook (`train_mara.py:6396-6398`): under-used experts get positive bias, over-used get negative, steering future routing without adding a loss-gradient. Also arms the whole balance hook + per-step CV/drop telemetry (`train_mara.py:6352-6432`). The bias buffer is checkpointed explicitly (`moe_bias_step_*.pt`, `train_mara.py:3459-3467`).
- **Interacts with:** `moe_bias_before_score` (where the bias is applied in the router), `moe_aux_balance_coeff` (independent, gradient-based balancing you can stack on top), `moe_num_experts` (buffer size).
- **Example:** `moe_load_balance_coeff: 0.001` (both real MoE configs)

### `moe_aux_balance_coeff`
- **Type:** float
- **Default:** `0.0`
- **Values/constraints:** `0.0` disables; `>0` enables. Applied only in `training` mode (`model_v2.py:990`). **⚠️ NOT wired through in mara:** `train_mara.py` never passes this into the `ModelArgs(...)` constructor, so the model always sees the dataclass default `0.0` regardless of the YAML — the value in `configs/keel-moe.yaml`/`configs/MegaMoe.yaml` has **no effect** on training and is not even reflected in the log (the rank-0 log at `train_mara.py:6428-6429` reads `model_cfg.moe_aux_balance_coeff`, i.e. the default `0.0`, so the "aux_balance_loss=" part never prints).
- **What it does (when a value actually reaches ModelArgs):** Weight on a DeepSeek-V3-style differentiable auxiliary balance loss `coeff * num_experts * sum(f_i * P_i)`, where `f_i` is the fraction of tokens routed to expert i and `P_i` the mean router probability (`model_v2.py:988-995`). Unlike the bias trick, this backpropagates through the router gate, giving it a direct gradient to diversify. The per-forward loss is stashed on `_last_aux_loss` for the Transformer to collect and add to the training loss.
- **Interacts with:** `moe_load_balance_coeff` (complementary; both can be on), `moe_num_experts`, `moe_top_k` (enter `f_i`).
- **Example:** `moe_aux_balance_coeff: 0.0001` appears in both real MoE configs but is currently a **no-op** due to the wiring gap; `0.0` = off (dataclass default, which is what the model always gets).

### `moe_bias_before_score`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true`/`false`. Real configs flag `true` as "DO NOT USE". **⚠️ NOT wired through in mara:** `train_mara.py` never passes this into the `ModelArgs(...)` constructor, so the model always sees the dataclass default `false` regardless of the YAML — the value in the real configs is only honored by the router when it actually reaches `ModelArgs`, which it doesn't from a mara config.
- **What it does (when a value actually reaches ModelArgs):** Placement of the load-balancing `expert_bias`. When `false` (recommended), bias is added to *scores* only for the top-k selection (`scores_for_choice = scores + expert_bias`) but the gathered `top_scores` use the unbiased scores (`model_v2.py:971-978`) — selection-only, gradient-clean. When `true` (old-style), bias is added to the raw logits before sigmoid/softmax (`model_v2.py:962-964`), shifting the nonlinearity's operating point and thereby altering routing-score gradient magnitudes.
- **Interacts with:** `moe_load_balance_coeff` (this only matters when `expert_bias` exists), `moe_score_func`.
- **Example:** `moe_bias_before_score: false` (both real MoE configs, with a "DO NOT USE" warning on `true`) — but since it's not forwarded, the model gets `false` either way.

### `moe_interleave_step`
- **Type:** int
- **Default:** `1`
- **Values/constraints:** `>=1`. `1` = every eligible layer is MoE; `2` = every other; N = every Nth.
- **What it does:** Sparsifies MoE across depth. A layer is MoE only if `(layer_id - moe_n_dense_layers) % interleave == 0` (in addition to passing the head/tail dense tests) — `model_v2.py:1356-1361`. Lets you interleave MoE and dense layers to trade capacity for compute/memory.
- **Interacts with:** `moe_n_dense_layers` (offset the modulo is measured from), `moe_n_tail_dense_layers`, `n_layers`.
- **Example:** `moe_interleave_step: 1` (both real MoE configs — every eligible layer is MoE)

### `moe_n_dense_layers`
- **Type:** int
- **Default:** `0`
- **Values/constraints:** `>=0`. Number of leading layers forced dense.
- **What it does:** The first `moe_n_dense_layers` transformer blocks (`layer_id < n_dense`) keep a dense `FeedForward` and are never MoE (`model_v2.py:1357-1361`). Common practice to keep early layers dense for stability. Also becomes the origin for the `moe_interleave_step` modulo.
- **Interacts with:** `moe_interleave_step`, `moe_n_tail_dense_layers`, `n_layers`.
- **Example:** `moe_n_dense_layers: 3` (both real MoE configs)

### `moe_n_tail_dense_layers`
- **Type:** int
- **Default:** `0`
- **Values/constraints:** `>=0`. Number of trailing layers forced dense.
- **What it does:** The last `moe_n_tail_dense_layers` blocks (`layer_id >= n_layers - n_tail_dense`) stay dense (`model_v2.py:1360`). Used to "coalesce experts before output" — keep near-output layers dense so representations converge before the LM head. Reported in the MoE config log as "tail-dense (synth)" (`train_mara.py:5935-5937`).
- **Interacts with:** `moe_n_dense_layers`, `moe_interleave_step`, `n_layers`.
- **Example:** `moe_n_tail_dense_layers: 3` (from `configs/MegaMoe.yaml`); `2` in `configs/keel-moe.yaml`

### `moe_capacity_factor`
- **Type:** float
- **Default:** `0.0`
- **Values/constraints:** `0.0` disables capacity dropping (all tokens processed); `>0` caps tokens per expert. Typical `1.0`-`1.5`. Training-only — eval never drops (`if self.capacity_factor > 0 and self.training`, `model_v2.py:1137`).
- **What it does:** Enables capacity-based token dropping in training. Per-expert capacity = `ceil(capacity_factor * N * top_k / num_experts)` (`model_v2.py:1139-1141`); overflow tokens (lowest-scoring for that expert) are dropped, kept-scores are unbiased-rescaled (`scale = (sum_all/sum_keep).clamp(max=10.0)`) to preserve expected magnitude, and dropped slots get a sentinel expert id (`model_v2.py:1136-1173`). Crucially it also enables the *padded-BMM* compiled training path (static `(E, capacity, dim)` shapes via `_bmm_capacity`, `model_v2.py:1183-1189`); with `0.0` there is no fixed capacity and experts run the dynamic-shape for-loop path (`use_bmm = self._bmm_capacity is not None and self.training`, `1189`). Dropped-token % is logged as the `drp` tag (`train_mara.py:2662`).
- **Interacts with:** `moe_top_k`, `moe_num_experts` (both in the capacity formula), `ep_degree` (`_bmm_capacity = per_rank_cap * ep_degree`), batch/seq (N = tokens per step).
- **Example:** `moe_capacity_factor: 1.25` (both real MoE configs); `0.0` = no dropping (default)

### `ep_degree`
- **Type:** int (nullable in YAML)
- **Default:** `null` in YAML → auto = `WORLD_SIZE` when `moe_enabled`, else `1` (`train_mara.py:5699-5704`). Dataclass field default is `1`.
- **Values/constraints:** Must divide `world_size` (`setup_ddp` assert, `train_mara.py:4056-4058`) and divide `moe_num_experts`, and be `<= moe_num_experts` (`train_mara.py:4255-4268`). Requires `moe_enabled: true` if `>1`. Set `ep_degree: 1` to explicitly disable EP even with multiple GPUs.
- **What it does:** Expert Parallel degree — how many ranks the experts are sharded across. `>1` builds a 2D device mesh with dim names `("fsdp", "ep")` (`train_mara.py:4062`), gives each rank `num_experts // ep_degree` local experts, and routes tokens with all-to-all dispatch/combine (`MoE._ep_dispatch`/`_ep_combine`). At `null`/auto it defaults to full `world_size` for a single node (fast intra-node all-to-all). Unlike the other `moe_*` fields, `ep_degree` is computed as a local variable in `main()` and passed positionally into `ModelArgs(ep_degree=ep_degree)` (`train_mara.py:5879`), so its auto-default path is honored. Expert weights are consolidated across EP ranks at checkpoint time (`train_mara.py:3254-3290`).
- **Interacts with:** `moe_num_experts` (divisibility + local count), `moe_shared_overlap` (only relevant when `>1`), `moe_capacity_factor` (`_bmm_capacity` scales by ep_degree), `world_size`.
- **Example:** `ep_degree: null` (auto = world size, both real MoE configs); `ep_degree: 1` to force-disable

### `moe_shared_overlap`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true`/`false`. Only takes effect under EP with a shared expert present (`self._ep_mesh is not None and self._shared_overlap and self.shared_experts is not None`, `model_v2.py:1198`). **⚠️ NOT wired through in mara:** `train_mara.py` never passes this into the `ModelArgs(...)` constructor, so `MoE.__init__` reads it via `getattr(args, 'moe_shared_overlap', False)` (`model_v2.py:1095`) and always gets the dataclass default `False` regardless of the YAML — the value in `configs/MegaMoe.yaml` has **no effect**.
- **What it does (when a value actually reaches ModelArgs):** Latency optimization. When `true`, the shared-expert FFN is computed on a side CUDA stream that overlaps with the routed experts' EP all-to-all round-trip (`model_v2.py:1198-1213`), hiding shared-expert compute behind communication. Described in configs as an optimization "for cards with more SMs" — the shared work must fit in spare SMs to be a win. No effect without EP or without a shared expert.
- **Interacts with:** `ep_degree` (must be `>1`), `moe_num_shared_experts` (must be `>0`).
- **Example:** `moe_shared_overlap: true` (from `configs/MegaMoe.yaml`); `false` in `configs/keel-moe.yaml` — but currently a no-op due to the wiring gap.

---

## Gated DeltaNet (GDN) Hybrid Attention

GDN replaces most softmax-attention layers with FLA's `GatedDeltaNet` (a gated linear-attention / delta-rule layer) while keeping a minority of full-attention layers, giving a hybrid stack (default 3:1 GDN:softmax). It is fully opt-in: when `gdn_enabled=False` none of the other `gdn_*` keys do anything. When enabled, the layer split is decided at construction time in `model_v2.py:1326-1351`, GDN requires the FLA library (lazily imported in `_try_import_gdn`, `model_v2.py:231`), and — a side effect worth knowing — the surviving full-attention layers gain a sigmoid output gate (`use_gate=True`), adding a `g_proj` weight per attention layer.

### `gdn_enabled`
- **Type:** bool
- **Default:** `False` (`train_mara.py:5544-5545`; `ModelArgs.gdn_enabled`, `model_v2.py:366`)
- **Values/constraints:** `true`/`false`. No explicit validation; when `true`, the FLA library must be installed or construction raises `ImportError` (`model_v2.py:238-242`). Orthogonal to `moe_enabled` — GDN and MoE are independent per-layer booleans and can be combined.
- **What it does:** Master switch for the hybrid attention stack. When true, each `TransformerBlock` decides per layer whether to instantiate `GatedDeltaNet` (`self.gdn_attn`) or the softmax `Attention` module, using `gdn_interleave_step` (`model_v2.py:1328-1330`). It also flips `use_gate=True` on every remaining full-attention layer, so those layers apply `out = out * sigmoid(g_proj(x))` before `wo` (`model_v2.py:513-516`, `648-650`). GDN layers carry no KV cache — inference routes through the plain forward and FLA's internal recurrent state (`model_v2.py:1435-1437`, and cache-setup code skips GDN layers).
- **Interacts with:** `gdn_interleave_step` (the split), all other `gdn_*` keys (only consumed when true), `moe_enabled` (orthogonal), `use_keel` / `norm_eps` (KEEL alpha-scaling wraps `_attn`; `norm_eps` is passed to GDN as `norm_eps`).
- **Example:** `gdn_enabled: true` (from `configs/MegaMoe-GDN.yaml:79`)

### `gdn_interleave_step`
- **Type:** int
- **Default:** `4` (`train_mara.py:5547`; `ModelArgs`, `model_v2.py:367`)
- **Values/constraints:** Positive integer. No explicit range validation. The rule is `use_gdn = (layer_id % gdn_step != gdn_step - 1)` (`model_v2.py:1330`), so within each window of `gdn_step` layers the LAST one is full-attention and the rest are GDN. Step `4` ⇒ layers 3, 7, 11, … are softmax and the other 3 in 4 are GDN (the documented 3:1 ratio); step `1` would make every layer full-attention (no GDN).
- **What it does:** Sets the spacing of the full-attention layers among the GDN layers. Note the semantics: it is "every Nth layer is full-attention, rest are GDN" (config comment and `model_v2.py:367`), i.e. GDN is the majority. Startup logging prints "every {step}th layer is full-attention" (`train_mara.py:5961`).
- **Interacts with:** `gdn_enabled` (only consulted when enabled); indirectly `n_layers` (determines how many of each type you actually get).
- **Example:** `gdn_interleave_step: 4` (from `configs/keel-moe-GDN-L.yaml:79`)

### `n_gdn_heads`
- **Type:** int (nullable)
- **Default:** `None` (`train_mara.py:5549`; `ModelArgs`, `model_v2.py:368`)
- **Values/constraints:** Positive int or null. When `None`/unset it falls back to `args.n_heads` (`model_v2.py:1334`: `getattr(args, 'n_gdn_heads', None) or args.n_heads`). No further validation here; FLA imposes its own head/dim consistency.
- **What it does:** Number of heads passed as `num_heads` to `GatedDeltaNet` (`model_v2.py:1338`). Combined with `gdn_head_dim` and `gdn_v_expand`, it sets the GDN layer's projection widths and the size of the recurrent delta-rule state (state scales with `head_dim²` per head). It does not have to equal the softmax `n_heads`.
- **Interacts with:** `gdn_head_dim`, `gdn_v_expand`, `n_heads` (fallback + relative sizing), `dim` (FLA input width is `hidden_size=args.dim`).
- **Example:** `n_gdn_heads: 16` (from `configs/MegaMoe-GDN.yaml:81`) — note: the inline comment "8 heads × 256" in the shipped configs is stale; the load-bearing values are `n_gdn_heads: 16`, `gdn_head_dim: 128`.

### `gdn_head_dim`
- **Type:** int (nullable)
- **Default:** `None` (`train_mara.py:5551`; `ModelArgs`, `model_v2.py:369`)
- **Values/constraints:** Positive int or null. When `None`/unset it falls back to **`256`** (the FLA default), NOT to `dim // n_heads` (`model_v2.py:1335`: `getattr(args, 'gdn_head_dim', None) or 256`). The same `or 256` fallback is used only for the startup log (`train_mara.py:5958`).
- **What it does:** The q/k head dimension passed as `head_dim` to `GatedDeltaNet` (`model_v2.py:1339`). Larger head_dim means a larger delta-rule state matrix (state is ~`head_dim²` per head), which the shipped configs note is generally better for the delta rule at the cost of memory/compute.
- **Interacts with:** `n_gdn_heads` (total GDN width ≈ `n_gdn_heads × head_dim`), `gdn_v_expand` (value width = `head_dim × expand`), `dim`.
- **Example:** `gdn_head_dim: 128` (from `configs/keel-moe-GDN-L.yaml:81`)

### `gdn_v_expand`
- **Type:** float
- **Default:** `2.0` (`train_mara.py:5553`; `ModelArgs`, `model_v2.py:370`)
- **Values/constraints:** Positive float. No explicit validation. Passed straight through to FLA.
- **What it does:** Value-expansion ratio passed as `expand_v` to `GatedDeltaNet` (`model_v2.py:1340`). The per-head value dimension is `head_dim × gdn_v_expand`, so with `gdn_head_dim: 128` and expand `2.0` each head's value width is 256 (config comment says 512 per head — that assumes head_dim 256). Widens the value/state path relative to q/k.
- **Interacts with:** `gdn_head_dim` (value dim = `head_dim × expand`), `n_gdn_heads`.
- **Example:** `gdn_v_expand: 2.0` (from `configs/MegaMoe-GDN.yaml:83`)

### `gdn_short_conv_kernel`
- **Type:** int
- **Default:** `4` (`train_mara.py:5555`; `ModelArgs`, `model_v2.py:371`)
- **Values/constraints:** Positive int. No explicit validation. Passed as `conv_size` to `GatedDeltaNet` alongside `use_short_conv=True` (`model_v2.py:1341,1344`).
- **What it does:** Kernel size of GDN's causal short convolution over q/k/v, which injects local (short-range) context before the linear-attention recurrence. This is where the `gdn_attn.q_conv1d/k_conv1d/v_conv1d` params come from (grouped to Adam per the project's param-grouping notes).
- **Interacts with:** `gdn_enabled` only (the short conv is always on for GDN layers here, `use_short_conv=True`).
- **Example:** `gdn_short_conv_kernel: 4` (from `configs/keel-moe-GDN-L.yaml:83`)

### `gdn_mode`
- **Type:** string
- **Default:** `'chunk'` (`train_mara.py:5557`; `ModelArgs`, `model_v2.py:372`)
- **Values/constraints:** FLA mode string. In practice `'chunk'` (parallel/chunked scan, used for training) or `'fused_recurrent'` (step-wise recurrence, for inference), per the config comments and `model_v2.py:372`. No validation in this codebase — the string is passed verbatim to FLA (`model_v2.py:1342`), which enforces valid values.
- **What it does:** Selects FLA's execution kernel for the delta-rule recurrence. `'chunk'` gives the training-time parallel form; `'fused_recurrent'` is the sequential inference form.
- **Interacts with:** `gdn_enabled` only.
- **Example:** `gdn_mode: chunk` (from `configs/MegaMoe-GDN.yaml:85`)

---

## Auxiliary Heads & Staged Interventions

This trainer exposes exactly one staged-intervention config block: `auxiliary_heads`. It serves two related modes. (1) As **plain aux next-token heads** it attaches extra RMSNorm+Linear readouts at intermediate layers and folds their weighted CE into the objective, distributing readout-shaping pressure across the body. (2) With `compute_inactive_layers: false` the same block becomes **Scaffolded Cascading Supervision (SCS)** — the trainer truncates the forward at the deepest aux tap currently at weight >= 1.0, uses that tap's CE as the effective LM loss, and freezes the still-inactive tail + main head via `lr_scale=0` until the schedule cascades them online. There is **no separate `scs:` or `tst:` top-level key** — SCS is entirely a sub-mode of `auxiliary_heads` (grepping `getattr(settings, 'scs'|'tst')` returns nothing), and Token Superposition Training (TST) exists only as a git branch, not in this tree.

### `auxiliary_heads`
- **Type:** dict (or `None`)
- **Default:** `None`
- **Values/constraints:** If absent or `enabled` is falsy, `Settings.__init__` collapses it to `None` (train_mara.py:5322-5326), so downstream code only checks `settings.auxiliary_heads is not None`. When it is a dict with `enabled: true`, the recognized sub-keys are `heads` (required), `compute_inactive_layers`, `new_layer_warmup_steps`, `new_layer_lr_multiplier`. Light validation runs in Settings (train_mara.py:5319-5350) but only touches `compute_inactive_layers` (and, under SCS, the two warmup knobs) — it does **not** validate `heads`; the full `heads` parse + layer range-check happens later in `parse_aux_heads_config` (train_mara.py:568), and the layer-range check at trainer setup (train_mara.py:5839-5843) needs `cfg_layers`/`n_layers`.
- **What it does:** Top-level container for intermediate-depth prediction heads. When enabled, `model_v2.py` (lines 1509-1512) builds one `AuxHead(dim, vocab_size, norm_eps)` per listed layer (a fresh RMSNorm + bias-free Linear, ~one LM-head's worth of params each — the startup log at train_mara.py:5982-5985 prints "M per head"), captures the residual stream at each tap during forward, and stashes per-head CE in `model._last_aux_loss_tensors`. The trainer weights those by the per-head schedule and adds them to `total_loss` before backward (train_mara.py:1951-1960).
- **Interacts with:** `compute_inactive_layers` switches it into SCS mode; SCS additionally gates on `optimizer_type` (must be in `FSDP2_MUON_FAMILY`, non-Sphere), `tie_word_embeddings` (must be false), and is mutually exclusive with `ffn_pdr_controller` (train_mara.py:5279-5287) and the truncator (SCS takes precedence — line 1822). Aux heads participate in `z_loss` (each fired head stashes its own zloss so the deepest SCS tap can be selected, model_v2.py:2036-2050).
- **Example:**
```yaml
auxiliary_heads:
  enabled: true
  heads:
    - layer: 50
      weight: [[11000, 0.0], [11700, 0.01], [13250, 0.0277], [15000, 0.03]]
```
(from `configs/dreadnought_v2.yaml`)

### `auxiliary_heads.enabled`
- **Type:** bool
- **Default:** `false` (treated as absent → whole block collapses to `None`)
- **Values/constraints:** Read via `aux_cfg.get('enabled', False)` in both `parse_aux_heads_config` (train_mara.py:593) and Settings (train_mara.py:5325). Any falsy value makes `settings.auxiliary_heads` become `None` and `parse_aux_heads_config` return `([], {})`.
- **What it does:** Master on/off for the block. When false the model builds no `AuxHead` modules (`aux_head_layers` stays empty), no taps are captured, and the training path is byte-for-byte the baseline. When true, at least one valid `heads` entry is required or `parse_aux_heads_config` calls `fatal_error` (train_mara.py:597).
- **Interacts with:** Gates all other sub-fields — they are only read when `enabled: true`.
- **Example:** `enabled: false` (from `configs/dn3.yaml`, `configs/dn4.yaml` — both ship the block wired off)

### `auxiliary_heads.heads`
- **Type:** list of dicts, each `{layer: int, weight: <scalar | schedule>}`
- **Default:** required when `enabled: true` (no default)
- **Values/constraints:** Must be a non-empty list (train_mara.py:596-597). Each entry must be a dict containing both `layer` and `weight` keys (train_mara.py:601-602). Duplicate `layer` entries are rejected (train_mara.py:638-639). At trainer setup every layer index is range-checked against `cfg_layers` (train_mara.py:5839-5843); the model constructor re-checks against `n_layers` at build (model_v2.py:1504-1508).
- **What it does:** Declares which body layers get an auxiliary readout and how strongly each contributes over training. `parse_aux_heads_config` returns `(sorted_layer_list, {layer: schedule})`; the sorted layer list drives `AuxHead` construction, the schedules drive per-step weighting. Under SCS the set of layers plus their weight schedules also determines the cascade activation events (`compute_scs_activation_events`, train_mara.py:664).
- **Interacts with:** `heads[*].layer`, `heads[*].weight`; under SCS the shallowest head's schedule must reach weight >= 1.0 at step 0 or the run fatals (train_mara.py:1668-1674).
- **Example:**
```yaml
heads:
  - layer: 50
    weight: [[11000, 0.0], [11700, 0.01], [13250, 0.0277], [15000, 0.03]]
  - layer: 55
    weight: [[11000, 0.0], [11700, 0.02], [13250, 0.0554], [15000, 0.06]]
  - layer: 60
    weight: [[11000, 0.0], [11700, 0.04], [13250, 0.1109], [15000, 0.12]]
```
(from `configs/dreadnought_v2.yaml`)

### `auxiliary_heads.heads[*].layer`
- **Type:** int
- **Default:** required
- **Values/constraints:** Must be a Python `int` (train_mara.py:604-605 — non-int types rejected). Must satisfy `0 <= layer < n_layers` (checked in model_v2.py:1504-1508 at build and train_mara.py:5839-5843 at trainer setup). Duplicates across entries are a fatal error.
- **What it does:** The residual-stream depth at which this aux head taps. In the forward, when `i in self._aux_head_layer_set` the post-block hidden state `h` is captured into `aux_taps[i]` (model_v2.py:1894/1903/1909) and later run through `self.aux_heads[str(layer)]` for its CE. Under SCS the deepest layer whose weight is >= 1.0 sets the scaffold truncation depth (`active_layers = deepest_tap + 1`, train_mara.py:1818-1821).
- **Interacts with:** `n_layers` (range), sibling `weight`; deeper taps come online later in the SCS cascade.
- **Example:** `layer: 60` (from `configs/dreadnought_v2.yaml`)

### `auxiliary_heads.heads[*].weight`
- **Type:** float scalar, or schedule `[[step, val], ...]`
- **Default:** required
- **Values/constraints:** Either a number (normalized internally to `[(0, value)]`, a constant) or a list of `[step, val]` waypoints where each `step` is a non-negative int and steps within one schedule must be strictly increasing (adjacent duplicates fatal — would divide-by-zero in interpolation; train_mara.py:618-629). Every value must lie in **[0.0, 1.0]** (train_mara.py:632-637) because SCS scaffold detection and the trainer's `if w != 0` gate assume the normalized range.
- **What it does:** The coefficient multiplied onto this head's CE before it is summed into `total_loss` at the current step (train_mara.py:1956-1958). Between waypoints it linearly interpolates via `interpolate_lr_mod`. A weight of exactly 0 skips the multiply-add (the head still computes CE for logging). Under SCS, reaching weight >= 1.0 flips this head into being the live LM readout for scaffold mode (`deepest_active_tap`, train_mara.py:648) and drives the cascade activation schedule.
- **Interacts with:** Sibling `layer`; under SCS interacts with `compute_inactive_layers`, `new_layer_warmup_steps` (schedule shape determines when each compartment activates). Also selects which head feeds `z_loss` under scaffold (model_v2.py:2036-2050 stashes every head; the trainer picks `scs_deepest_tap` at train_mara.py:1988).
- **Example:** `weight: [[11000, 0.0], [11700, 0.04], [13250, 0.1109], [15000, 0.12]]` (from `configs/dreadnought_v2.yaml`)

### `auxiliary_heads.compute_inactive_layers`
- **Type:** bool
- **Default:** `true`
- **Values/constraints:** Must be a bool or Settings fatals (train_mara.py:5334-5338). Read as `_ah_cfg.get('compute_inactive_layers', True)`; `scs_enabled = aux_heads_enabled and not compute_inactive_layers` (train_mara.py:1582). Setting it `false` (i.e. enabling SCS) triggers a battery of compatibility guards: optimizer must be in `FSDP2_MUON_FAMILY` and not a Sphere variant (train_mara.py:1620-1638), `tie_word_embeddings` must be false (train_mara.py:1643-1649), and at least one head must be at weight >= 1.0 at step 0 (train_mara.py:1668-1674).
- **What it does:** `true` (default) = ordinary aux-head mode: the full network runs every step, aux heads are pure auxiliary supervision layered onto the normal main-head LM loss. `false` = **Scaffolded Cascading Supervision**: each step the trainer finds the deepest tap at weight >= 1.0, truncates forward+backward there, skips `self.norm`/`self.output` (model_v2.py:1922 guards `self.norm`; `self.output` is structurally bypassed because the scaffold training branch sets `loss = None` and never reaches the head), uses that tap's CE as the effective LM loss, and freezes all deeper layers, the main output head, and the final norm via `lr_scale_overrides = 0` (train_mara.py:2110-2161) so weight decay can't quietly erode them. As the schedule pushes deeper taps to >= 1.0, compartments activate in order until no head is at >= 1.0 — the "cascade-complete" step where the main head and tail come online end-to-end.
- **Interacts with:** `new_layer_warmup_steps` + `new_layer_lr_multiplier` (only consulted when this is false — they shape the per-compartment LR ramp); the `heads` schedules (define activation order); mutually exclusive with `ffn_pdr_controller` (train_mara.py:5285) and takes precedence over the truncator (train_mara.py:1822); `optimizer_type`, `tie_word_embeddings` (hard guards).
- **Example:** `compute_inactive_layers: false` (SCS mode; no shipped config in `configs/` currently enables it — the three configs using the block set `enabled: false` or use plain aux mode. Documented at train_mara.py:5329-5350.)

### `auxiliary_heads.new_layer_warmup_steps`
- **Type:** int
- **Default:** `0`
- **Values/constraints:** Must be a non-negative int (train_mara.py:5341-5345). Only validated and read when SCS is active (`compute_inactive_layers: false`); ignored otherwise (`scs_warmup_steps` is forced to 0, train_mara.py:1583).
- **What it does:** The number of steps over which a freshly-activated SCS compartment linearly ramps its LR from `new_layer_lr_multiplier` up to 1.0 (`scs_compartment_lr_scale`, train_mara.py:712-734). Before a compartment's activation step its params are frozen (`lr_scale=0`); from activation to activation+warmup they ramp; after, full LR. The same ramp is applied to the main output head + final norm at the cascade-complete step, so the random-init head doesn't shock the trained body on the first non-scaffold step (train_mara.py:2126-2144). `0` = instant jump to full LR at activation (no soft start).
- **Interacts with:** `new_layer_lr_multiplier` (the ramp's start value); `compute_inactive_layers` (only meaningful in SCS mode); the head weight schedules (set the activation steps).
- **Example:** SCS-only knob; not present in any shipped config. Documented default 0, referenced at train_mara.py:1583.

### `auxiliary_heads.new_layer_lr_multiplier`
- **Type:** float
- **Default:** `1.0`
- **Values/constraints:** Must be a number in **[0.0, 1.0]** (train_mara.py:5346-5350). Only validated and read under SCS (`scs_init_mult`, train_mara.py:1584); forced to 1.0 otherwise.
- **What it does:** The starting LR multiplier for a compartment at the moment it activates, before the `new_layer_warmup_steps` ramp lifts it to 1.0 (`scs_compartment_lr_scale`, train_mara.py:733). A small value (e.g. 0.1) means newly-unfrozen layers begin training gently and accelerate to full LR; `1.0` (default) plus `warmup_steps=0` means compartments snap straight to full LR with no soft start. Applies identically to the main-head/final-norm compartment at cascade completion.
- **Interacts with:** `new_layer_warmup_steps` (with warmup 0 this value is only momentarily seen); `compute_inactive_layers` (SCS-only).
- **Example:** SCS-only knob; not present in any shipped config. Documented default 1.0, referenced at train_mara.py:1584.

### `auxiliary_heads.share_lm_head`
- **Type:** bool
- **Default:** N/A — key is not consumed by the code
- **Values/constraints:** Appears in `configs/dreadnought_v2.yaml:116` (`share_lm_head: false`) but grepping the entire Python tree finds **zero read sites**. `parse_aux_heads_config` reads only `enabled` and `heads`; Settings/trainer/checkpoint code reads only `enabled`, `heads`, `compute_inactive_layers`, `new_layer_warmup_steps`, `new_layer_lr_multiplier`. It is silently ignored (unknown YAML keys inside the nested dict are never referenced).
- **What it does:** Nothing, as written. It is documentary/aspirational: each `AuxHead` always builds its own independent `RMSNorm + Linear` and explicitly does not tie to the main LM head ("No weight sharing with the main LM head — each aux head learns its own readout", model_v2.py:273, `__init__` at model_v2.py:279-282). Setting it `true` would have no effect. Treat this as a dead key; do not rely on it.
- **Interacts with:** —
- **Example:** `share_lm_head: false` (from `configs/dreadnought_v2.yaml` — present but inert)

---

## Progressive Tail Truncation

Progressive Tail Truncation (PTT) is a training-only regularizer that, on a fraction of steps, randomly cuts the forward pass short in the tail of the network — running only the first *N* layers, then applying the final norm + output head directly to that intermediate depth. This forces the head and mid layers to build stronger standalone representations via shorter gradient paths (a "light gradient injection" into earlier layers). The truncation decision is made by a step-seeded Python `random.Random(step + 42)` RNG so **all FSDP ranks make the identical decision every step** with no communication. Validation steps are always run at full depth so val loss and diagnostics stay clean.

Backing module: `v:\code\common_fsdp2\tail_truncation.py` (class `ProgressiveTailTruncation`). Config lives under a top-level `truncation:` block; it is passed whole as the `config` dict, and the model must support the `active_layers=` forward argument. `truncation.enabled: false` (or an absent block) is fully inert. PTT and SCS (`auxiliary_heads.compute_inactive_layers: false`) are mutually incompatible — when a SCS aux tap is at λ ≥ 1.0, SCS takes precedence and overrides the truncator for that step.

Both `safe_fraction` and `truncation_prob` accept either a static scalar **or** a schedule as a list of `[step, value]` waypoints, linearly interpolated between waypoints (flat-held before the first / after the last). `depth_power`, `loss_weight`, and `bypass_compile` are static scalars only.

Mechanism recap (per step, when enabled): interpolate `safe_fraction` and `truncation_prob` for the current step → `safe_layer = int(n_layers * safe_fraction)` → with prob `truncation_prob`, pick a cut point in `[safe_layer, n_layers-1]` biased by `depth_power`; otherwise run full depth. Layers `0..safe_layer-1` (the "safe zone") always run; the truncation zone is `safe_layer..n_layers-1`.

### `truncation.enabled`
- **Type:** bool
- **Default:** `false`
- **Values/constraints:** `true` / `false`
- **What it does:** Master on/off switch. When `false`, `get_truncation_point()` always returns `n_layers` (full depth) — the block is completely inert and adds no overhead.
- **Interacts with:** Gates all other sub-keys. On validation steps the trainer forces full depth regardless of this flag.
- **Example:** `enabled: true`

### `truncation.safe_fraction`
- **Type:** float in [0.0, 1.0], or schedule `[[step, frac], ...]`
- **Default:** `0.75`
- **Values/constraints:** Fraction of the network that is always run (never truncated). `safe_layer = int(n_layers * safe_fraction)`; the truncatable tail is `n_layers - safe_layer` layers. A value of `1.00` means zero truncation zone (nothing can be cut). Schedule values are linearly interpolated; before the first waypoint / after the last it is held flat.
- **What it does:** Sets the boundary between the always-run "safe zone" (layers `0 .. safe_layer-1`) and the "truncation zone" (layers `safe_layer .. n_layers-1`) from which cut points are drawn.
- **Interacts with:** Combined with `n_layers` (model depth) to compute the zone. If the resulting zone size ≤ 0, no truncation happens. Typically ramped from `1.00` up during warmup so truncation only begins after the network stabilizes. The startup log prints the resulting Safe Zone / Trunc Zone layer ranges at the final scheduled fraction.
- **Example:** `safe_fraction: [[0, 1.00], [1000, 0.80]]`  (full depth until step 1000, then reserve top 20% as truncatable)

### `truncation.truncation_prob`
- **Type:** float in [0.0, 1.0], or schedule `[[step, prob], ...]`
- **Default:** `0.25`
- **Values/constraints:** Per-step probability of truncating at all. `0.0` disables truncation for that step (full depth). Schedule values are linearly interpolated / flat-held outside the waypoint range.
- **What it does:** First-stage Bernoulli gate: on each step, with probability `truncation_prob` a truncated pass is performed; otherwise a normal full-depth pass runs. Controls how often the shorter gradient path is exercised.
- **Interacts with:** Independent of `safe_fraction` (which sets *where* the cut lands). Commonly held at `0.00` during warmup then ramped to a small value (0.05–0.20) — a "light touch" so most steps remain full-depth. If ≤ 0, full depth is forced.
- **Example:** `truncation_prob: [[0, 0.00], [1000, 0.00], [2500, 0.20]]`  (no truncation until step 1000, ramping to 20% by step 2500)

### `truncation.depth_power`
- **Type:** float
- **Default:** `2.0`
- **Values/constraints:** ≥ 1.0 in practice. `1.0` = uniform over the zone; higher = biased toward *shallow* cuts (cut points near the end of the network, i.e. running more layers). Applied as `position = u ** (1.0 / depth_power)` with `u ~ Uniform(0,1)`.
- **What it does:** Second-stage shape control: once a truncation is triggered, this biases *where* in the truncation zone the cut lands. Larger power concentrates cuts near the top of the network (fewer layers dropped); `1.0` spreads them uniformly.
- **Interacts with:** Only matters when a truncation fires (gated by `truncation_prob`) and when the zone is non-empty (gated by `safe_fraction`). See the empirical table in `mini-fathom-ptt.yaml` (e.g. 1.0 uniform mean-cut ≈ 55.5; 3.0 strongly shallow-biased mean-cut ≈ 62.5). Live configs use a near-uniform `1.1–1.2`.
- **Example:** `depth_power: 1.1`

### `truncation.loss_weight`
- **Type:** float
- **Default:** `1.0`
- **Values/constraints:** Any float (typically 0.0–1.0). Applied only on truncated steps; full-depth steps always use weight `1.0`.
- **What it does:** Multiplier on the training loss for truncated passes (`get_loss_weight`), letting you down- or up-weight the contribution of the shorter-path objective relative to full-depth steps. The trainer scales both `total_loss` and the logged `main_loss` by this factor before `backward()`.
- **Interacts with:** Applied on top of the normal `/ grad_accum_steps` division. Set to `1.0` in all shipped configs (equal weighting). Forced to `1.0` internally when SCS scaffold mode overrides the truncator.
- **Example:** `loss_weight: 1.0`

### `truncation.bypass_compile`
- **Type:** bool
- **Default:** `true`
- **Values/constraints:** `true` / `false`
- **What it does:** On a truncated step, when the model has a top-level `_orig_mod` (i.e. a single whole-model `torch.compile` wrapper), routes the forward through the un-compiled `_orig_mod` to avoid caching a separate compiled graph for every distinct truncation point (which would thrash the compile cache / recompile).
- **Interacts with:** Largely a no-op under this repo's **per-submodule** compile strategy: with per-submodule compile there is no top-level `_orig_mod` wrapper, so running fewer layers doesn't change any individual compiled graph and the bypass branch (`hasattr(model, '_orig_mod')`) simply never triggers. The trainer notes `bypass_compile` is unnecessary in that mode. Only relevant if you compile the whole model as one graph.
- **Example:** `bypass_compile: false`

---

**Example configs using this feature** (all under `v:\code\mara_fsdp2\configs\`):
- `fathom-y.yaml` — `enabled: true`, `safe_fraction: [[0,1.00],[1000,0.80]]`, `truncation_prob: [[0,0.00],[1000,0.00],[2500,0.20]]`, `depth_power: 1.1`, `loss_weight: 1.0` (80-layer KEEL)
- `mf2-ptt.yaml` — `safe_fraction: [[0,1.00],[3000,0.90]]`, `truncation_prob: [[0,0.00],[3000,0.00],[5000,0.10]]`, `depth_power: 1.1`, `loss_weight: 1.0`
- `mini-fathom-ptt.yaml` — 3-waypoint schedules `safe_fraction: [[0,1.00],[1500,0.60],[2000,0.85]]`, `truncation_prob: [[0,0.00],[1500,0.20],[2000,0.10]]`, `depth_power: 1.1`, `loss_weight: 1.0`, `bypass_compile: false` (70-layer; includes the depth_power tuning table in comments)
- `mf-revenge.yaml` — `safe_fraction: [[0,1.00],[4000,0.80]]`, `truncation_prob: [[0,0.00],[4000,0.00],[5000,0.10]]`, `depth_power: 1.2`, `loss_weight: 1.0`
- `snake-egg.yaml` — full block present but **commented out** (documents the intended "5% light-touch" usage)

---

## Telemetry, Tracking & Health Guards

Opt-in diagnostic and monitoring knobs. Each adds a Dashboard-parseable field to the status line or fires an advisory WARNING; none change training math or take auto-action (Josef keeps manual control). All but `transition_health_guard` are read via `getattr(settings, ..., default)` and have **no formal Settings.__init__ default or validation** — an unrecognized value type will not be caught at config-load time, it just flows into the read site. Enable them per-run in your config.

### `track_gpm`
- **Type:** bool
- **Default:** `False` (read via `getattr` at train_mara.py:1436; no Settings default)
- **Values/constraints:** truthy/falsy. Any truthy value constructs the tracker; there is no type validation.
- **What it does:** Turns on the Gradient Productivity Metric on the status line (` | gpm: +0.31/+0.25`). Per logged step (rank 0 only) it appends `(grad_norm, loss)` to a rolling deque and computes a lag-1 Spearman rank correlation between the median-detrended grad-norm and the next step's loss drop, over a short and a long window (`GPMTracker`, train_mara.py:361). Positive => big gradients here are productive; ~0 => noise; negative => anti-productive. On resume (`start_step > 1`) it warm-starts the buffer by re-parsing `st:/ls:/nrm:` lines from `gen_log_file` under `nas_path` so GPM-L has no restart seam (train_mara.py:1442-1475); the seed is best-effort and never breaks a resume. Shows ` | gpm: pending` until it has ≥5 points.
- **Interacts with:** `gpm_window_short`, `gpm_window_long` (its window sizes); `gen_log_file` + `nas_path` (resume warm-start source).
- **Example:** `track_gpm: true` (dn4.yaml:49)

### `gpm_window_short`
- **Type:** int
- **Default:** `15` (train_mara.py:1439)
- **Values/constraints:** positive int; coerced via `int(...)`. Only read when `track_gpm` is truthy.
- **What it does:** Length of the short/responsive GPM window (GPM-S, the first number in `gpm: S/L`). A shorter window reacts faster to a productivity shift; the S-vs-L gap is itself the read (S>L => productivity rising, S<L => dipping — see the `GPMTracker` docstring at train_mara.py:373). GPM-S needs ≥5 buffered points before it reports.
- **Interacts with:** `track_gpm` (gated by it); `gpm_window_long` (the paired long window).
- **Example:** Not set in any tracked config — the default `15` is used everywhere `track_gpm` is on.

### `gpm_window_long`
- **Type:** int
- **Default:** `101` (train_mara.py:1440)
- **Values/constraints:** positive int; coerced via `int(...)`. Only read when `track_gpm` is truthy. Sizes the underlying deque to `w_long + 2` (train_mara.py:385), so this also bounds how much history the tracker holds and how many resume points are warm-started.
- **What it does:** Length of the long/stable-baseline GPM window (GPM-L, the second number in `gpm: S/L`). Gives the slow-moving productivity baseline the short window is compared against.
- **Interacts with:** `track_gpm` (gated by it); `gpm_window_short` (the paired short window).
- **Example:** Not set in any tracked config — the default `101` is used everywhere `track_gpm` is on.

### `track_clip_groups`
- **Type:** bool
- **Default:** `False` (`bool(getattr(...))` at train_mara.py:1481; no Settings default)
- **Values/constraints:** truthy/falsy, coerced with `bool()`.
- **What it does:** "Probe B" per-group clip telemetry. When on, the grad-clip pass calls `_clip_grad_norm_mixed_mesh(model, clip_value, group_telemetry=True)` (train_mara.py:2078) and appends a Dashboard-parseable tag to the status line showing the global grad-norm split by group plus the clip coefficient: ` | gn_body: … | gn_head: … | gn_emb: … | gn_other: … | clip_c: …` (train_mara.py:2762-2766). This shows *who* the global clip is actually firing on — the Muon body is clip-invariant while the Adam groups are magnitude-sensitive. Folded into the existing clip pass, so it is essentially free.
- **Interacts with:** `clip_warmup` / `clip_standard` (the clip value being reported on); group mapping via `_clip_group_of`.
- **Example:** `track_clip_groups: true` (dn4.yaml:50, kv2.yaml:48)

### `track_subgroup_pdr`
- **Type:** bool
- **Default:** `False` (`bool(getattr(...))` at train_mara.py:6451; no Settings default)
- **Values/constraints:** truthy/falsy, coerced with `bool()`. Passed to `LayerDiagnostics(track_subgroups=...)`.
- **What it does:** Adds finer per-subgroup update-ratio (pdr = ‖dW‖/‖W‖) telemetry inside the diagnostics block (kv3 QK-norm / FFN-split investigation). When on, `LayerDiagnostics` also tracks the attention split (wq/wk vs wv/wo) and the FFN split (w1/w3 vs w2) and, at diagnostic cadence, logs one line of the four subgroup medians: `[subgroup pdr] attn qk=… vo=… | ffn w1w3=… w2=…` (diagnostics.py:1054-1061). Opt-in because it roughly doubles the transient snapshot VRAM at diagnostic cadence (diagnostics.py:110-112). It is telemetry only — the four subgroup fields are never read anywhere in train_mara.py, so it does not feed the FFN pdr controller.
- **Interacts with:** `LayerDiagnostics` cadence (emits on validation/diagnostic steps); conceptually the `ffn_pdr_controller.*` subgroup-escalation decision (this is the telemetry you read to decide whether w1/w3 and w2 need separate control), but it does not drive it.
- **Example:** `track_subgroup_pdr: true` (dn3.yaml:174, kv3.yaml:49)

### `transition_health_guard`
- **Type:** bool
- **Default:** `False` (formally defaulted + validated in Settings.__init__, train_mara.py:5490-5493)
- **Values/constraints:** must be a `bool` — any non-bool triggers `fatal_error("transition_health_guard must be a bool, ...")` at config load (train_mara.py:5493). This is the only setting in this section with real type validation.
- **What it does:** "Guardrail 5", an advisory transition health guard. Emits loud `[HEALTH WARNING @ step]` log lines only — never auto-pauses, auto-checkpoints, or intervenes. Two checks, thresholds calibrated from the hard-branch nrm=5.61 event and the family calibration (Nexus #156/#157): (1) grad-norm, per logged step — warns on `nrm > 5` (single-step spike) or `nrm > 3` for ≥3 consecutive steps (train_mara.py:2785-2797); (2) centered geometry, at validation cadence — warns on `effective_rank_c < 7` (approaching low-rank collapse) or `spectral_concentration_c > 0.45` (train_mara.py:2972-2983). The geometry check requires the centered-geometry diagnostic (`cg_diag`) to be present and runs rank-0 only.
- **Interacts with:** `health_guard_warmup_steps` (suppresses the grad-norm check for the first N steps); the centered-geometry diagnostics (must be computed for the geometry warnings to fire); `val_step` (cadence of the geometry check).
- **Example:** `transition_health_guard: true` (dn4.yaml:156, kv2.yaml:67)

### `health_guard_warmup_steps`
- **Type:** int
- **Default:** `100` (`int(getattr(...))` at train_mara.py:1431; no Settings default/validation)
- **Values/constraints:** int (coerced via `int(...)`); only meaningful when `transition_health_guard` is on. Compared as `step >= health_guard_warmup_steps` to gate the grad-norm warning.
- **What it does:** Suppresses the grad-norm half of the health guard for the first N steps (train_mara.py:2785). During LR warmup — from-scratch or a fresh resume — grad-norm is expectedly high and noisy, so warning every step there is pure noise that trains you to ignore the line before a real spike. Set it to roughly cover your LR-ramp / early-noise window (kv2 uses 1000, keelhaul 150, dn3 7100). Note: it gates **only** the grad-norm check, not the val-cadence geometry (eff_rank / spec_conc) check.
- **Interacts with:** `transition_health_guard` (no effect unless that is on); `warmup_steps` (the LR ramp this window is meant to cover).
- **Example:** `health_guard_warmup_steps: 1000` (kv2.yaml:72). Note dn4.yaml enables the guard but omits this key, so it defaults to `100`.
