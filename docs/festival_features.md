# Festival Features & Telemetry

Compact reference for the three "festival" training features and the stats they
emit — for building dashboard charts. (As of checkpoint schema **v4.0**, every
checkpoint self-describes which of these it used; see *Config flags* below.)

---

## The three features

| Feature | What it does | Result |
|---|---|---|
| **doc_attn_mask** (+ `reset_positions`) | Tokens attend only within their own document (causal ∧ same-doc via FlexAttention). With `reset_positions`, RoPE positions restart at each document boundary (separator = tokenizer BOS **id 1**). Kills cross-document attention/position bleed. | −15.5 mnats vs control, faster |
| **swa** (sliding-window attention) | Hybrid 3:1 local:global. Local layers attend a causal window of `window` tokens; every `global_interleave`-th layer stays full-causal. Composes with the doc mask. | loss-neutral, faster |
| **mtp** (multi-token prediction) | DeepSeek-style: one extra transformer block predicts the token at **t+2** (one beyond next) through a shared norm/head. Auxiliary objective `λ·mtp_loss` folded into total loss; **headline `ls:`/val stay pure t+1 CE**. Also powers self-speculative decoding at inference. | −13.3 mnats, <1% cost |

---

## Config flags (also embedded in v4.0 checkpoint `config`)

| Flag (checkpoint config key) | Type | Wizard value |
|---|---|---|
| `doc_attn_mask` | bool | true |
| `doc_pos_reset` | bool | true |
| `bos_token_id` | int | 1 |
| `swa_enabled` | bool | true |
| `swa_window` | int | 512 |
| `swa_global_interleave` | int | 4 |
| `mtp_enabled` | bool | true |

Plus `checkpoint_version: "4.0"` and `rope_fixed: true`. A run's feature set can
be read straight from any of its checkpoints — no run-yaml needed.

---

## New telemetry

Only **MTP** produces a per-step time series. SWA is purely architectural (no
runtime metric); doc-mask emits a one-time boot check.

### MTP loss — the chartable series

Same underlying value in all three sinks: `mtp_accum` = the **t+2 cross-entropy**,
averaged over grad-accum micro-steps and all-reduced (mean) across ranks. It is
the **raw, unweighted** MTP CE (the λ weight is applied only to the objective, not
to what's logged). Present only when `mtp_enabled`.

| Sink | Field / format | Cadence | Notes |
|---|---|---|---|
| `gen_log.txt` (human) | ` \| mtp: 3.1234` (`.4f`) | every logged step | eyeball line |
| `train_log.txt` (machine) | `\|mtp=3.123456` (`.6f`) appended to the pipe line | every logged step | **parse this for charts** |
| `diagnostics.jsonl` | `"mtp": { … }` (see below) | val cadence (`val_step`) | structured |

**`diagnostics.jsonl` → `"mtp"` block:**

| Key | Meaning |
|---|---|
| `loss` | all-reduced t+2 CE (same value as the `\|mtp=` above) |
| `gap` | `mtp_loss − main_loss` = t+2 CE minus t+1 CE — the **plan-ahead difficulty margin** (how much harder predicting 2-ahead is than 1-ahead; watch it shrink as the model learns to plan) |
| `lambda` | the objective weight `mtp_loss_weight` (0.3) — constant unless retuned |
| `w_norm` | MTP module aggregate weight norm (global) — tracks the head's growth |

**Suggested charts:** `mtp.loss` vs tokens (overlay the headline `ls:` to see the
gap visually); `mtp.gap` vs tokens (the key learning-to-plan signal); `mtp.w_norm`
vs tokens.

### doc-mask — one-time boot sanity (not a series)

Logged once around step 20 to `gen_log.txt`:

```
] [doc-mask] stream check OK: <N> separators in <M> windows (~0.NN docs-started/window at T=<T>)
```

Fatal-errors instead if the configured `bos_token_id` never appears (mask would be
a silent no-op). The `docs-started/window` ratio is a useful one-off data-shape
stat but has no time dimension.

### swa — no runtime metric

Architectural only. Chartable facts live in the config (`swa_window`,
`swa_global_interleave`) — e.g. derived "local vs global layer count".

---

## `train_log.txt` line format (for the parser)

Pipe-delimited, base fields first, optional `|key=value` tags appended only when
the corresponding feature is active (so older/plainer runs have fewer fields —
parse defensively by key, not position, past field 8):

```
step|loss|ppl|lr|norm|dt|total_tokens|tok_per_s   [|mtp=…][|zloss=…|logZ=…|…][|rc_muW=…|…][|z_a_eff=…|rc_s=…][ | gpm: …]
```

Base 8: `step`, `loss` (t+1 CE), `ppl`, `lr`, `norm` (pre-clip grad norm), `dt`
(s), `total_tokens`, `tok_per_s`. The festival-relevant tag is **`|mtp=<t+2 CE>`**.
(`zloss`/`logZ`/`rc_*` tags belong to the separate z-loss and row-center
interventions, not the festival set.)
