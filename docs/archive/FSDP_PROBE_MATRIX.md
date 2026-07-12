# FSDP2 Body-Ramp — Experiment Matrix (2026-06-23)

## ⚠️ OBSOLETE — this whole matrix targeted a non-bug. The cause is Newton-Schulz, not FSDP.
**Resolution (see WD_WASTE_ANALYSIS.md banner + newton_schulz_radial_finding):** the lean these
toggles chased is the Newton-Schulz UPDATE (the `cos_grad_W` probe read `.grad` post-`optimizer.step`,
where Muon overwrote it with the NS update). The RAW gradient is radial-null; "single-card/unsharded
null" just means no Muon step ran. So FSDP is NOT the culprit — no FSDP1 rewrite, none of E1-E9
needed. The pre/post-step probe (raw=−0.0000029 vs post-NS=−0.01285) closed it. Fix = tangent-project
the Muon update (validated). Kept as history; the toggle matrix is moot.

---

Goal (ORIGINAL, now obsolete): localize the ~60% **FSDP2 execution-path residual** (the anti-radial
body-gradient lean that survives identical-data and full-fp32, but is null single-card/unsharded).
Per Math Agent: run ALL meaningful toggles before any FSDP1 rewrite. Each isolates ONE variable;
compare `cos(g,W)` (in-situ, global-aggregated — audited correct) vs the baselines.

## Baselines (DONE, trustworthy)
| # | config | cos(g,W) | note |
|---|---|---|---|
| B1 | single-card, exact rank-0 tokens, unsharded | **+0.000003** | NULL — the reference |
| B2 | 8-GPU FSDP2, bf16, diverse data | **−0.0129** | the canonical lean |
| B3 | 8-GPU FSDP2, bf16, replicated data (all=rank0) | **−0.0075** | data-independent residual |
| B4 | 8-GPU FSDP2, **fp32** params+fwd+CCE (TF32 on), diverse | **−0.0159** | precision ruled out... |

## THE MATRIX (each row = flip ONE thing vs B3 replicated-data, 8-GPU, bf16, unless noted)
Replicated data is the cleanest base (removes the ~40% data component → isolates the path).

| # | experiment | flag / change | isolates | status |
|---|---|---|---|---|
| E1 | **TF32 disabled** (true IEEE fp32) | `WD_NO_TF32=1` + fp32 config | residual precision (TF32 ~10-bit mantissa) — the one precision caveat left | BUILT, next |
| E2 | **world=1 FSDP2** | `--nproc_per_node=1` | FSDP wrapper/hooks vs REAL multi-shard | ran, OOM'd at optimizer.step (did all 1465 mb) — RETRY w/ lower mem |
| E3 | **torch.compile OFF** | `compile_model: false` | compile/DTensor graph-capture interaction | TODO |
| E4 | **activation checkpoint OFF** | `use_activation_checkpointing: false` | FSDP act-ckpt recompute (re-gather/different views in backward) | TODO |
| E5 | **reshard_after_forward = TRUE** | config (currently false) | param lifetime / reshard+re-gather between fwd/bwd | TODO |
| E6 | **no-shard / DDP-style** (replicated params) | FSDP2 no-shard mesh, or DDP | distributed data-parallel WITHOUT sharding | TODO (feasibility?) |
| E7 | **finite-difference under FSDP2** | `L(e^±εW)` per body class, ε∈{1e-4,3e-4,1e-3} | is the lean a real FWD loss-surface deriv (A), or backward/hook/measurement (B)? | TODO — Math Agent's DECISIVE test |
| E8 | **layerwise forward equivalence** | capture x_l unsharded vs FSDP2 same tokens | where (if) forward states diverge → first culprit layer | TODO |
| E9 | **FSDP1 same probe** | FSDP1 FlatParameter wrap | does FlatParameter reconstruction avoid it? | LAST — diagnostic only, NO rewrite |

## Read-outs / decision logic
- **E1 (TF32 off):** still leans → precision TRULY ruled out (B4 confirmed). Vanishes → it was
  TF32 all along (precision back in play, big simplification).
- **E7 (finite-diff):** the linchpin. FD negative & matches ⟨g,W⟩ → FSDP2 **forward loss surface**
  itself differs (→ E8 layerwise). FD null but backward dot negative → it's **backward/autograd/
  hooks/measurement**, not forward physics → strongly implicates FSDP2 impl. (Math Agent insists
  on this BEFORE any FSDP1 decision.)
- **E2/E3/E4/E5:** whichever toggle moves the lean names the mechanism (wrapper / compile /
  checkpoint / reshard). A one-line winner beats an FSDP1 migration.
- **E9 (FSDP1):** only switch the stack if FSDP1 ≈ 0 AND DDP/no-shard ≈ 0 AND finite-diff
  confirms it's not measurement AND a short FSDP1 branch shows lower body-norm slope w/o
  throughput/stability cost (Math Agent's 5-point bar).

## Memory note (rig-30, 24GB 3090s)
world=1 (E2) holds the WHOLE model + fp32 optimizer state on ONE card → OOMs at optimizer.step
even though it completes the forward/backward. Fixes: lower T further, or the probe could skip
optimizer.step (it only needs cos(.grad,W) BEFORE the step — the step is for the ΔW
decomposition we don't strictly need for E2). **Consider a probe mode that captures cos and
exits BEFORE optimizer.step** — removes the step-spike OOM for all single-card/world=1 runs.

## Policy (unchanged, Math Agent concurs — MORE firmly now)
No body-WD / no renorm / no head-WD until the residual source is pinned. Increasing body WD now
could be compensating a training-SYSTEM artifact, not model physics — would hide the source.
