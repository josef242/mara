# Multi-Step Recurrent MTP — Design Spec

Status: **design only, not implemented.** This captures the design + all research
so a future implementation is turnkey. The λ-schedule sibling feature IS
implemented (see `get_mtp_lambda` in `train_mara.py`); this doc is the *other*
knob the frontier survey surfaced: turning our single MTP module into a
**k-token drafter** by weight-tied recurrent unrolling.

---

## 1. Goal

Extend our MTP module from predicting **one** extra token (t+2) to drafting **k**
future tokens (t+2 … t+k+1) — *without adding parameters* — by reusing the single
MTP block **recurrently, with weights tied across steps**. This lengthens the
self-speculative-decode draft window (currently k=1 → 1 token per verify) toward
the frontier's 3–7, for higher decode throughput.

Non-goal: it is **not** a training-quality feature per se (D=1 is the universal
pretraining choice). It's a **decode-speed** feature. Keep it opt-in.

---

## 2. Where we are today

| Piece | File / symbol | Current behavior |
|---|---|---|
| MTP module | `model_v2.py: MTPModule` (~L1616) | one block; `forward(h, next_emb, …)` = 1 step |
| Training loss | `model_v2.py` (~L2454–2473) | one step: predict t+2, `_last_mtp_loss` = CE(mtp, targets≪1) |
| Inference draft | `model_v2.py: mtp_decode_chunk` (~L2147) | produces **1** draft token |
| Spec engine | `neo_common.py: spec_generate` + inlined engine in `stream_generate_kv`; `_spec_verify_step` | **k=1** self-speculative (draft 1, verify, accept/reject) |
| MTP KV cache | `setup_caches` mtp branch; `_mtp_cache_len` | sized/tracked for the 1-step rollout |

The module equation today (single step):
```
h¹ᵢ = Block( proj( [RMSNorm(hᵢ) ; RMSNorm(Emb(t_{i+1}))] ) )   → logits for t_{i+2}
```
(`hᵢ` = trunk pre-final-norm hidden; readout via the SHARED final norm + head.)

---

## 3. Frontier landscape (survey, condensed)

Primary sources: DeepSeek-V3 [arXiv:2412.19437], GLM-5 [arXiv:2602.15763] +
GLM-5.2 blog (huggingface.co/blog/zai-org/glm-52-blog), MiniMax-M2
[arXiv:2510/2605.26494], Nemotron-3 [arXiv:2604.12374], Qwen3-Next blog,
EAGLE-3 [arXiv:2503.01840].

| Model | Draft horizon | How | Notable |
|---|---|---|---|
| DeepSeek-V3 | 1 (t+2) | D=1 single module | reference; discard-at-inference or ~85–90% accept, 1.8× TPS |
| MiniMax-M2 | 3 | grow D=1→3 in decay phase via **weight-COPY** (distinct modules) | freeze-then-joint; frozen-trunk-only was *worse* |
| **GLM-5 / 5.2** | **3 / 7** | **ONE module, weights TIED across steps**, unrolled | GLM-5.2 accept-length 5.47; adds rejection-sampling + TV loss + IndexShare/KV-Share |
| Nemotron-3 | 2 | shared-weight unified head, recursive | accept-length 3.45 > DeepSeek-R1's 2.70; constant λ=0.3 |
| Qwen3-Next | 1, multi-step | ONE module + "multi-step training for train/infer consistency" | λ not published |

**The convergent lesson:** the newest models (GLM-5.2, Nemotron-3, Qwen3-Next)
get longer drafts by **weight-tied recurrent unrolling of a single module**, and
they explicitly train the module under the *same* multi-step rollout it runs at
inference (train/inference consistency). This is cheaper (one module's params)
and posts higher acceptance than DeepSeek's stack-a-module-per-token. It
supersedes MiniMax-M2's weight-copy approach for our purposes. Our per-module
anatomy already matches GLM-5.2 component-for-component (enorm/hnorm/eh_proj/
shared head), so only the *unrolling* is new.

---

## 4. Core design decision: weight-tied recurrent unrolling

Add one config knob `mtp.n_steps: k` (default **1** = today's behavior). At k>1,
the SAME `MTPModule` is applied k times in sequence, each step conditioning on
the previous step's hidden state:

```
h⁰ᵢ = hᵢ                                                  (trunk hidden)
for j in 1..k:
    hʲᵢ = Block( proj( [RMSNorm(hʲ⁻¹ᵢ) ; RMSNorm(Emb(uⱼ))] ) )   → logits for t_{i+1+j}
```
where `uⱼ` is the intermediate token at step j (see §5 for teacher-forced vs
own-prediction). **Same block/proj/norms every step (weight-tied)** → zero new
parameters. Gradient flows through the whole chain into the trunk (no detach), as
today. Sequential/causal-chain (DeepSeek/GLM style), NOT parallel Medusa heads.

---

## 5. Training-side design

Loop the module k times; accumulate one CE loss per step against progressively
shifted targets; average (DeepSeek's `ℒ_MTP = (λ/k)·Σⱼ ℒʲ`). λ is already
schedulable via `get_mtp_lambda`.

- **Target alignment:** step j predicts `t_{i+1+j}` = `targets` shifted left by j
  (last j columns padded/ignored). Step 1 = today's t+2.
- **The critical decision — what feeds `Emb(uⱼ)`:**
  - **(A) Teacher-forced** (DeepSeek D>1 sequential): `uⱼ` = the *true* token
    `t_{i+j}`. Simple, one clean backward. BUT train≠inference — at inference the
    module conditions on its OWN prior draft, not the truth. Lower acceptance.
  - **(B) Training-time test** (Qwen3-Next / GLM-5.2 "consistency", EAGLE-3):
    unroll feeding the module's own step-(j−1) prediction back as `uⱼ`. Matches
    inference exactly → higher acceptance. Costs: a per-step argmax/sample in the
    graph and care with gradient flow through the fed-back token (typically
    straight-through or stop-grad on the discrete token, keep grad on the hidden).
  - **Recommendation:** ship (A) first (small delta from current code, correct,
    a real t+3… training signal), then add (B) as `mtp.multistep_consistency:
    true` once (A) is validated. GLM-5.2's gains are largely from (B) + rejection
    sampling; (A) alone already gives a usable multi-token drafter.
- **KV within the unroll:** each step's Block attends the trunk context + the
  prior MTP steps' K/V (the deep step's cache is built from prior steps — GLM-5.2
  "KV-Share"). At train time this is standard teacher-forced attention over the
  concatenated positions.
- **Gradient:** keep the no-detach flow into the trunk (our defining choice;
  MiniMax-M2 confirmed frozen-trunk is worse). With weight-tying, step j's grad
  also accumulates into the shared block (like a tiny RNN unrolled k steps —
  watch for grad magnitude scaling with k; may want `1/k` already handled by the
  loss average, but check block-output variance).

---

## 6. Inference-side design

Our spec engine (`_spec_verify_step` + the k=1 loop in `spec_generate` /
`stream_generate_kv`) already does single-token speculative sampling. Generalize
to a k-token chain:

1. **Draft:** unroll `mtp_decode_chunk` k steps to produce a k-token draft
   `d₁…d_k` (each step: previous MTP hidden + Emb(previous draft) → next draft),
   reusing the tied weights. This is `generate_forward`(trunk) once for `h_pre`,
   then k tiny MTP steps.
2. **Verify:** ONE trunk forward over `[cur, d₁, …, d_k]` yields the trunk
   distributions `p₀…p_k`. Apply the standard **chain speculative-sampling**
   accept rule: walk the draft, accept `dⱼ` w.p. `min(1, p(dⱼ)/q(dⱼ))` until the
   first rejection; on rejection resample from the residual `max(0, p−q)`; if all
   k accepted, sample one bonus token from `p_k`. (This is exactly our
   `_spec_verify_step` applied along the chain — extend it to return the accepted
   prefix length.)
3. **Cache:** the MTP block cache and `_mtp_cache_len` must hold the k-step
   unroll; the trunk cache appends the accepted prefix (append-only ring rules we
   already enforce — mind the SWA rewind guard `>=` fix and the reject phantom).
4. **Greedy** stays bit-identical (accept iff argmax matches, per step).

Throughput: tokens/verify ≈ `1 + Σ accept` (vs today's `1 + accept`). Frontier
accept-lengths at k=7: ~5.5 (GLM-5.2).

---

## 7. Code touchpoints (exact map)

- **Config / Settings** (`train_mara.py` MTP block, ~L5888): add `n_steps`
  (int, default 1) and optional `multistep_consistency` (bool) to `_mt_known` +
  validation. Persist to checkpoint config (the festival-field whitelist) so
  inference self-describes the draft depth. **Guard:** changing `n_steps` across
  a resume is a baked-in change → add it to `_RESUME_MUST_MATCH`.
- **Model init** (`model_v2.py` ~L1767): `MTPModule` is unchanged (weight-tied
  means still ONE module). Only the *number of unroll steps* changes; no new
  params, so the paired-init discipline is unaffected. Add `n_steps` to
  `ModelArgs`.
- **Training loss** (`model_v2.py` ~L2454): wrap the current single-step block in
  a `for j in range(n_steps)` loop; accumulate `Σ CE / n_steps` into
  `_last_mtp_loss`; feed teacher-forced (A) or own-pred (B) `Emb(uⱼ)`.
- **Inference draft** (`model_v2.py: mtp_decode_chunk` ~L2147): return a k-token
  draft chain instead of 1; add the MTP KV bookkeeping for the unroll.
- **Spec engine** (`neo_common.py`): generalize the accept/reject loop from a
  single draft to a chain; `_spec_verify_step` extended to walk the chain and
  return accepted-prefix length. Keep greedy bit-identical.
- **Caches** (`model_v2.py: setup_caches` mtp branch; `_mtp_cache_len`,
  `min_rolling_cache_len`): size the MTP block cache for `n_steps`; keep the
  append-only + `>=` rewind-guard invariants (see the SWA-reject-phantom fix).
- **Telemetry** (`docs/festival_features.md`): add a `mtp.n_steps` /
  accept-length-per-step chart hook once implemented.
- **Tests** (`common_fsdp2/test_spec_decode.py`): extend the greedy-parity,
  reuse-equivalence, and mutation/power tests to k>1 (the k=1 tests are the
  base case).

---

## 8. Implementation phases

1. **Config + train loop (teacher-forced, mode A).** Add `n_steps`, loop the
   loss. Validate: training runs, `mtp.loss` reasonable, t+3… targets aligned,
   grad flows. Cheapest, lowest risk. *No inference change yet* (still discard or
   k=1 decode).
2. **Inference k-step draft + chain verify.** Generalize `mtp_decode_chunk` +
   the spec engine to k tokens; extend tests (greedy bit-identical, sampled
   distribution-exact per the chain rule, reuse bit-exact). Measure accept-length.
3. **Training-time-test consistency (mode B).** Feed own predictions back in
   training; expect the accept-length jump GLM-5.2 reports. Optional:
   rejection-sampling refinement.
4. **(Optional, later) GLM-5.2 extras:** end-to-end TV loss, KV-Share — diminishing
   returns, DSA-coupled; skip unless chasing the last few % acceptance.

---

## 9. Open decisions (to make at implementation time)

- **k (n_steps):** GLM-5=3, GLM-5.2=7, Nemotron=2. Start k=2–3; 7 is aggressive
  and only pays off with mode-B training. Cost scales ~linearly in the tiny MTP
  block, negligible vs the trunk.
- **Mode A vs B first:** recommend A → validate → B. B is where the acceptance
  wins live but adds graph complexity.
- **Grad scaling with k:** the `1/k` loss average handles the loss magnitude;
  verify the tied block's output variance / update norm stays sane as k grows
  (it's an unrolled recurrence).
- **Whether to keep training-quality D=1 separate from decode-k:** you could
  train the loss with k=1 (pure t+2 signal, cheapest) but *unroll k at inference*
  — but train/inference mismatch tanks acceptance (the whole point of mode B). So
  couple them: if you want decode-k, train with n_steps=k (mode B).

---

## 10. References

- DeepSeek-V3, §2.2 MTP + §4.2 λ schedule — arXiv:2412.19437
- GLM-5 report (weight-shared 3-step MTP, DeepSeek comparison, Table 2) —
  arXiv:2602.15763 ; GLM-5.2 blog (7-step, accept-length ablation) —
  huggingface.co/blog/zai-org/glm-52-blog
- MiniMax-M2 (K=1→3 weight-copy, freeze-then-joint) — arXiv:2510/2605.26494
- Nemotron-3 (shared-weight recursive head) — arXiv:2604.12374
- Qwen3-Next blog (multi-step train/infer consistency)
- EAGLE-3 (training-time-test multi-step unrolling) — arXiv:2503.01840
- Our current MTP: `common_fsdp2/model_v2.py` (MTPModule, mtp_decode_chunk,
  training loss); spec engine in `neo_common.py`; λ-schedule `get_mtp_lambda` in
  `mara_fsdp2/train_mara.py`; feature overview `docs/festival_features.md`.
