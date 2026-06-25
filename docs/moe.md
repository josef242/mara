# MoE Implementation Plan for Mara FSDP2

## Source of Truth: TorchTitan (cloned at `v:\code\torchtitan`)

All implementation details below are verified against actual TorchTitan source code,
not blog posts or summaries.

---

## Key TorchTitan Files (read and analyzed)

| File | What it contains |
|------|-----------------|
| `torchtitan/models/moe/moe.py` | MoE module: GroupedExperts, TokenChoiceTopKRouter, TokenReorderer, MoE class |
| `torchtitan/distributed/expert_parallel.py` | ExpertParallel (All-to-All dispatcher), TensorParallel, ExpertTensorParallel, DeepEPExpertParallel |
| `torchtitan/distributed/parallel_dims.py` | ParallelDims dataclass, mesh construction (dense_mesh, sparse_mesh, dataloading_mesh) |
| `torchtitan/models/llama4/infra/parallelize.py` | `apply_fsdp()`, `apply_moe_ep_tp()`, `apply_compile()` — the actual FSDP+EP composition |
| `torchtitan/models/deepseek_v3/model/args.py` | DeepSeekV3ModelArgs with MoEArgs |
| `torchtitan/models/deepseek_v3/infra/parallelize.py` | DeepSeek V3 parallelization (imports from llama4) |
| `torchtitan/models/moe/moe_deepep.py` | DeepEP variant of MoE (alternative dispatcher) |
| `torchtitan/models/moe/utils.py` | `_permute`, `_unpermute`, `indices_padding_wrapper` |

---

## Architecture: How TorchTitan MoE Actually Works

### GroupedExperts (NOT ModuleList)

Expert weights are **3D tensors**, not a ModuleList of FeedForward:

```python
class GroupedExperts(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, use_grouped_mm):
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
```

This is critical — it enables:
1. EP sharding on dim 0 (expert dimension) via `distribute_tensor(param, mesh, [Shard(0)])`
2. Batched computation via `torch._grouped_mm` (SM90+ only) or for-loop fallback
3. FSDP sharding on dim 1 when `efsdp * ep > num_experts`

### Router: TokenChoiceTopKRouter

```python
class TokenChoiceTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, ...):
        self.gate = nn.Linear(dim, num_experts, bias=gate_bias)

    def forward(self, x, expert_bias=None):
        scores = self.gate(x)  # (bs*slen, num_experts)
        scores = torch.sigmoid(scores.float())  # DeepSeek-style sigmoid
        scores_for_choice = scores + expert_bias  # aux-loss-free balancing
        _, selected = torch.topk(scores_for_choice, k=top_k, dim=-1)
        top_scores = scores.gather(dim=1, index=selected)  # original scores, not biased
        # ... normalize, scale
        num_tokens_per_expert = torch.histc(selected.view(-1), bins=num_experts, ...)
        return top_scores, selected, num_tokens_per_expert
```

Key detail: `expert_bias` is used ONLY for routing decisions, NOT for gating values.
The actual weights come from original unbiased scores.

### Load Balancing: Aux-Loss-Free

```python
# In MoE.__init__:
self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
self.register_buffer("tokens_per_expert", torch.zeros(num_experts, dtype=torch.float32), persistent=False)

# In MoE.forward:
with torch.no_grad():
    self.tokens_per_expert.add_(num_tokens_per_expert)

# Updated OUTSIDE the model in optimizer pre-hook (not in forward/backward):
# expert_bias += coeff * sign(mean_tokens - tokens_per_expert)
```

### MoE Forward Flow

```
Input: (bs, slen, dim)
  → flatten to (bs*slen, dim)
  → Router: scores, selected_experts, num_tokens_per_expert
  → Reorderer: sort tokens by expert assignment
  → Gather: routed_input = x[sorted_token_indices]
  → Score (optional): routed_input *= top_scores (before experts)
  → Experts: routed_output = GroupedExperts(routed_input, num_tokens_per_expert)
  → Shared experts: shared_out = FeedForward(x) (runs in parallel conceptually)
  → Unsort: scatter routed_output back to original positions
  → Sum top-k: reduce over top_k dimension
  → Add: out = shared_out + routed_out
  → reshape to (bs, slen, dim)
```

### Shared Experts

NOT a ModuleList. Single FeedForward with scaled hidden dim:
```python
self.shared_experts = FeedForward(dim=dim, hidden_dim=hidden_dim * num_shared_experts)
```

---

## Expert Parallel: How All-to-All Actually Works

### ExpertParallel extends ParallelStyle

Applied via `distribute_module()`, NOT as a wrapper:

```python
# In apply_moe_ep_tp():
experts_plan = ExpertParallel()
parallelize_module(
    module=transformer_block.moe.experts,
    device_mesh=ep_mesh,
    parallelize_plan=experts_plan,
)
```

### _partition_fn: Shard Weights on Expert Dim

```python
def _partition_fn(self, name, mod, device_mesh):
    for param_name, param in mod.named_parameters(recurse=False):
        dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
        mod.register_parameter(param_name, dist_param)
```

With EP=2 and 8 experts: each GPU gets experts [0-3] or [4-7].

### _token_dispatch: All-to-All

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    ep_degree = device_mesh.shape[0]

    # 1. Exchange token counts between EP ranks
    num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert, ...)

    # 2. Compute split sizes
    input_splits = num_tokens_per_expert.view(ep_degree, -1).sum(dim=1)
    output_splits = num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1)

    # 3. All-to-All: send tokens to expert-owning ranks
    routed_input = all_to_all_single_autograd(routed_input, output_splits, input_splits, ...)

    # 4. Permute for local expert alignment
    routed_input, permuted_indices, ... = _permute(routed_input, ...)

    return routed_input, num_tokens_per_expert_group
```

### _token_combine: Reverse All-to-All

```python
def _token_combine(self, mod, routed_output, device_mesh):
    routed_output = _unpermute(routed_output, self.input_shape, self.permuted_indices)
    routed_output = all_to_all_single_autograd(routed_output, input_splits, output_splits, ...)
    return routed_output
```

---

## Mesh Setup for MoE

### ParallelDims creates 3 global meshes from flat world mesh:

```python
dense_mesh:  (pp, dp_replicate, fsdp, tp)       # non-MoE layers
sparse_mesh: (pp, dp_replicate, efsdp, ep, etp)  # MoE layers
dataloading: (pp, batch, cp, tp)                  # data loading
```

Where `efsdp = dp_shard * cp * tp / (etp * ep)`

### For 8 GPUs, EP=2, no TP/PP/CP:

```
dp_shard=8, ep=2, etp=1
efsdp = 8 * 1 * 1 / (1 * 2) = 4

dense_mesh:  (fsdp=8)
sparse_mesh: (efsdp=4, ep=2)
```

Non-MoE layers: FSDP across all 8 GPUs.
MoE experts: EP shards on expert dim across 2 groups of 4, FSDP within each group of 4.

---

## FSDP + EP Composition

### Two-level FSDP wrapping for MoE layers:

```python
# From llama4/parallelize.py apply_fsdp():
for layer_id, transformer_block in model.layers.items():
    if transformer_block.moe_enabled and ep_degree > 1:
        # 1. FSDP the routed experts on the smaller edp_mesh
        fully_shard(
            transformer_block.moe.experts,
            mesh=edp_mesh,  # (efsdp=4) or (dp_replicate, efsdp)
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            shard_placement_fn=lambda param: Shard(1),  # if efsdp*ep > num_experts
        )

    # 2. FSDP the whole block on the full dp_mesh
    fully_shard(
        transformer_block,
        mesh=dp_mesh,  # (fsdp=8) or (dp_replicate, fsdp)
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
```

### Explicit Prefetching (when EP enabled)

D2H syncs in EP interfere with FSDP's implicit prefetching, so explicit prefetch chains
are set up for both forward and backward:

```python
# Forward: each block prefetches the next (including next block's experts if MoE)
transformer_block.set_modules_to_forward_prefetch([next_block, next_block.moe.experts])

# Backward: reversed order
transformer_block.set_modules_to_backward_prefetch([prev_block, prev_block.moe.experts])
```

---

## torch.compile with MoE

### Per-submodule compilation (NOT whole block)

```python
# MoE layers: compile each submodule separately
for attr_name, submod in moe.named_children():
    if attr_name == "experts":
        continue  # DO NOT compile experts (FSDP hooks cause graph break)
    moe.attr_name = torch.compile(submod, backend=backend, fullgraph=True)

# Non-MoE layers: compile whole block
transformer_block = torch.compile(transformer_block, backend=backend, fullgraph=True)
```

### Required settings:
```python
torch._dynamo.config.capture_scalar_outputs = True  # for dynamic token routing shapes

# When EP enabled, mark dynamic shapes:
torch._dynamo.mark_dynamic(x, 0)  # dynamic number of tokens per expert
```

---

## Critical Hardware Constraint: SM86 vs SM90

`torch._grouped_mm` requires **SM90+** (Hopper/Ada Lovelace).
RTX 3090 is **SM86** (Ampere).

**Fallback**: `_run_experts_for_loop` — Python loop over experts with individual matmuls.
Functional but ~10-50x slower than grouped_mm for many experts.

**Mitigation options**:
1. Use for-loop (simplest, works everywhere)
2. Write custom Triton kernel for batched expert GEMMs on SM86
3. Use fewer experts (8-16) to limit the loop overhead
4. Pad and use a single stacked matmul (requires all experts get same #tokens — only with forced balance)

---

## Integration with Mara FSDP2

### Parameter Naming Changes

Current (dense FeedForward):
```
layers.{i}.feed_forward.w1.weight  → nn.Linear, shape (hidden_dim, dim)
layers.{i}.feed_forward.w2.weight  → nn.Linear, shape (dim, hidden_dim)
layers.{i}.feed_forward.w3.weight  → nn.Linear, shape (hidden_dim, dim)
```

MoE (GroupedExperts):
```
layers.{i}.moe.experts.w1  → nn.Parameter, shape (num_experts, hidden_dim, dim)
layers.{i}.moe.experts.w2  → nn.Parameter, shape (num_experts, dim, hidden_dim)
layers.{i}.moe.experts.w3  → nn.Parameter, shape (num_experts, hidden_dim, dim)
layers.{i}.moe.router.gate.weight  → nn.Linear, shape (num_experts, dim)
layers.{i}.moe.shared_experts.w1.weight  → nn.Linear (if shared experts)
```

### Optimizer Classification Updates Needed

`is_muon_param(name)` currently checks for `.w1.`, `.w2.`, `.w3.` — will match expert params.
But expert weights are 3D tensors. **Muon requires 2D weights** for Newton-Schulz orthogonalization.

Options:
1. **Exclude expert params from Muon** (treat as Adam) — safest, avoids 3D tensor issues
2. **Reshape experts to 2D** for Muon — risky, changes the math
3. **Run Muon on each expert slice** — complex, not how Muon optimizer expects params

Recommendation: **Option 1** — route expert params to Adam. Router `gate` already goes to Adam (no `.w1.`/`.w2.`/`.w3.` in name).

```python
def is_muon_param(name):
    # Existing check
    if '.weight' not in name:
        return False
    if not any(lt in name for lt in ['wq.', 'wk.', 'wv.', 'wo.', 'w1.', 'w2.', 'w3.']):
        return False
    # NEW: exclude 3D expert params (they have .w1 not .w1.weight because nn.Parameter not nn.Linear)
    if '.experts.' in name:
        return False
    return True
```

Actually — the naming is `moe.experts.w1` (no `.weight` suffix since it's nn.Parameter, not nn.Linear).
So the existing `.w1.` check won't match `experts.w1` (no trailing dot). But `w1.` WOULD match
`shared_experts.w1.weight`. Need to verify carefully.

### WD Rules / lr_mods

Expert params need new target names in WD rules and lr_mods:
- `expert` or `moe` target for expert weights
- Router and shared expert weights can use existing `all` target

### AWD Integration

New component types for AWD:
- `L{i}.moe` — grouped expert params (or break down further per-expert if needed)
- Shared experts would map to existing `L{i}.ffn` pattern

### Diagnostics

Add MoE-specific metrics:
- Expert utilization (tokens_per_expert distribution)
- Load balance coefficient
- Router entropy

---

## Phased Implementation Plan

### Phase 1: Model Changes (model_v2.py) — NO EP ✅ COMPLETE

Add MoE to model_v2.py with **replicated experts** (no EP, no multi-dim mesh).
This isolates routing bugs from distributed bugs.

Files changed:
- `common_fsdp2/model_v2.py`: Added GroupedExperts, TokenChoiceTopKRouter, MoE classes
- `common_fsdp2/model_v2.py`: Modified ModelArgs (MoE fields), TransformerBlock (`moe_enabled`, `_ffn()` dispatch)
- `mara_fsdp2/train_mara.py`: MoE config in Settings → ModelArgs, checkpoint config, load balancing hook, startup logging
- `mara_fsdp2/configure_optimizers.py`: Added `adam_default` group for expert 3D params
- `mara_fsdp2/adaptive_wd.py`: MoE layer handling in `_register_components`
- `common_fsdp2/diagnostics.py`: MoE layer handling in `_get_ffn_params`

Tested: `moe_test.yaml` — 6 layers (1 dense + 5 MoE), 4 experts, top_k=2, 1 shared expert.
7× RTX 3060 (12GB), NorMuon FSDP2 optimizer. Loss decreasing: 10.81 → 10.56 in 3 steps.

#### Phase 1 Bugs Found & Fixed

1. **`summarize_model()` crash** — Accessed `block.feed_forward.w1.weight` on MoE layers that don't have `feed_forward`. Fixed by checking `block.moe_enabled` and reading from `block.moe.experts`.

2. **CCE dtype mismatch** — Hidden states were fp32, output weight was bf16. FSDP2 MixedPrecisionPolicy may not cast 3D `nn.Parameter` like it does `nn.Linear.weight`. Fixed with explicit `h_flat.to(out_dtype)` cast and `output_dtype` in MixedPrecisionPolicy.

3. **`classify_param` 'default' trap (CRITICAL)** — Expert 3D params (`moe.experts.w1`, no `.weight` suffix) correctly failed `is_muon_param()`, but `classify_param()` returned `'default'` as fallback. The FSDP2 Muon grouping code routed ALL `'default'` params to `muon_params` with `use_muon=True`. Newton-Schulz on 3D `(4, 3136, 2240)` tensors caused an infinite hang with 100% GPU utilization. **Fix**: Use `is_muon_param(n)` directly in param grouping, add `adam_default` group for non-Muon/non-special params. Same fix applied to AdamC path (`is_normalized_param` + `unnormalized_params` group). DION family was already correct (uses `is_muon_param` directly).

4. **`step == 0` debug timing never fired** — `start_step=1` so `step` starts at 1, not 0. Fixed to use `step == start_step`.

5. **CCE disabled on Windows** — `os.name == "nt"` means the Triton CCE kernel is skipped; fallback uses `hidden @ weight.t()` + `F.cross_entropy`. Not a bug, but important to know when debugging.

### Phase 2: Expert Parallel (2D Mesh) ✅ COMPLETE

Distribute experts across GPUs via EP. Each rank owns `num_experts // ep_degree` local experts.
Tokens dispatched via all-to-all, processed by local experts, combined back via all-to-all.

**Architecture:**
```
2D mesh: (efsdp, ep) = (1, 7) for 7 GPUs with EP=7
dp_mesh: (7,) — dense FSDP (attention, norms, shared_experts, router gate, embeddings)
ep_mesh: mesh_2d["ep"] — all-to-all communication
edp_mesh: mesh_2d["fsdp"] — inner FSDP for experts (prevents outer dp_mesh from touching them)
```

**Files changed:**
- `common_fsdp2/model_v2.py`: Added `ep_degree` to ModelArgs, `_AllToAllSingleAutograd` (differentiable all-to-all), `_permute_for_ep` / `_unpermute_for_ep` (reorder tokens between rank-grouped and expert-grouped layouts), `MoE.set_ep_mesh()`, `MoE._ep_dispatch()` / `_ep_combine()`, EP branch in `MoE.forward`
- `mara_fsdp2/train_mara.py`: `setup_ddp()` — create 2D mesh `(efsdp, ep)` when `ep_degree > 1`, `create_and_shard_model()` — two-level FSDP wrapping (inner experts on edp_mesh, outer layer on dp_mesh), `_clip_grad_norm_mixed_mesh()` — gradient clipping for mixed DTensor meshes, EP validation + logging
- `mara_fsdp2/configs/moe_test.yaml`: 7 experts, `ep_degree: 7`
- `configure_optimizers.py`: No changes (expert params already in `adam_default` group)
- `muon_fsdp2.py`: No changes (Adam on any mesh works the same)

**Tested:** `moe_test.yaml` — 12 layers (1 dense + 11 MoE), 7 experts, EP=7, top_k=2, 1 shared expert.
7× RTX 3060 (12GB), NorMuon FSDP2 optimizer. Loss decreasing, training stable.

#### Phase 2 Bugs Found & Fixed

1. **FSDP bf16 count rounding (CRITICAL)** — Inner `fully_shard(experts, edp_mesh, mp_policy)` casts ALL float module inputs to bf16 via the pre-forward hook. Token count tensors from `_permute_for_ep` (float32) were rounded by bf16's 7-bit mantissa: `7346.0 → 7360.0`, `6994.0 → 7008.0`. This caused `split_with_sizes` mismatches in `GroupedExperts.forward`. **Fix**: Return `local_num_tpe` as `torch.int64` — FSDP mixed-precision only casts floating-point tensors, so int64 passes through unmodified.

2. **Mixed-mesh `clip_grad_norm_`** — PyTorch's `clip_grad_norm_` uses `torch.stack` on per-param gradient norms (DTensor scalars). When params live on different meshes (`dp_mesh` vs `edp_mesh`), `torch.stack` fails with "All operands must have the same mesh". **Fix**: Custom `_clip_grad_norm_mixed_mesh()` that extracts `._local_tensor` from each DTensor grad, computes local norm², does a single `dist.all_reduce(SUM)` across all ranks, then clips. This is correct because: FSDP sharded params have 1/N of the gradient per rank (sum = full norm²), and EP expert params have unique experts per rank (sum = total expert norm²).

### Phase 3: Compile ✅ COMPLETE

Per-submodule `torch.compile` for all layers (MoE + dense).

**Approach** (`_apply_per_submodule_compile` in train_mara.py):
- All layers compiled per-submodule (not whole-block) to avoid conflicts with inline `checkpoint()`
- Dense layers: compile `attention`, `feed_forward`, `attention_norm`, `ffn_norm` individually
- MoE layers: compile `attention`, `attention_norm`, `ffn_norm`, `moe.router`, `moe.shared_experts` — skip `moe.experts` (FSDP hooks on GroupedExperts break the graph)
- `torch._dynamo.config.capture_scalar_outputs = True` — required for dynamic token routing shapes
- `torch._dynamo.config.cache_size_limit = max(16, num_layers + 4)` — default 8 too small for many layers with distinct compiled submodules

**Bugs found:**
1. Whole-block compile + inline `checkpoint()` causes dynamo guard invalidation (new closures each call). Fix: per-submodule compile for ALL layers.
2. Default `cache_size_limit = 8` too small — each layer's compiled submodule is a unique `OptimizedModule` instance at the same call site. With 12 layers, dynamo needs 12 cache entries. Fix: increase limit to `num_layers + 4`.

### Phase 3b: Shared Expert Overlap (optional, config-gated)

Overlap `shared_experts` FFN with EP dispatch/combine using a CUDA side stream.

**Config**: `moe_shared_overlap: true` (default: `false`)

**Idea**: shared_experts has zero dependency on EP, so it could run concurrently with the EP round-trip on a side CUDA stream.

**Result on 7× RTX 3060**: ~10% slower (SM contention on 28 SMs, memory bandwidth saturation at 360 GB/s, stream sync overhead × 11 MoE layers). Reverted to off-by-default. May benefit larger GPUs (3090: 82 SMs, 936 GB/s) where NCCL leaves spare SMs idle.

**Implementation** (in `MoE.forward`, model_v2.py):
- Gated by `self._shared_overlap` (from `ModelArgs.moe_shared_overlap`)
- Lazy-created `self._shared_stream` (one `torch.cuda.Stream` per MoE module, reused across steps)
- Side stream `wait_stream(main)` before launch — ensures `x_flat` is ready
- Main stream `wait_stream(side)` after EP combine — ensures `shared_out` is computed before addition
- When disabled (default): zero overhead, identical to serial path

### Phase 3c: CPU-GPU Sync Reduction

Eliminate redundant GPU→CPU synchronization in the EP dispatch/combine path.

**Problem**: Each MoE layer's EP path had 5 GPU→CPU pipeline stalls per forward pass (× 11 MoE layers = 55 stalls/step). Three were redundant (same data converted multiple times) and one could be made async.

**Sync audit (before)**:
```
SYNC 1: _ep_dispatch       → num_tpe.int().cpu().tolist()           → input_splits
SYNC 2: _ep_dispatch       → num_tpe_received.int().cpu().tolist()  → output_splits
SYNC 3: _permute_for_ep    → num_tpe_received.int().tolist()        → REDUNDANT (=SYNC 2)
SYNC 4: GroupedExperts.fwd  → local_num_tpe.int().tolist()           → derivable from SYNC 2
SYNC 5: _unpermute_for_ep  → num_tpe_received...int().tolist()      → REDUNDANT (=SYNC 2)
```

**After**: 1 blocking sync + 1 async per layer (down from 5 blocking):
- SYNC 1 → async `non_blocking=True` D2H, overlapped with count all-to-all
- SYNC 2 → kept (unavoidable — need received counts)
- SYNC 3 → eliminated (pass pre-computed `rcv_counts` list to `_permute_for_ep`)
- SYNC 4 → eliminated (pass `local_counts` list to `GroupedExperts.forward` via `_counts` kwarg)
- SYNC 5 → eliminated (pre-compute `_ep_unpermute_counts` from `rcv_counts` in Python)

**Changes**:
- `_permute_for_ep` / `_unpermute_for_ep`: take `counts` list directly (no GPU tensor), no internal `.tolist()` sync
- `GroupedExperts.forward`: optional `_counts` kwarg skips `.int().tolist()` sync in EP path
- `_ep_dispatch`: async D2H for input_splits, single sync for rcv_counts, derive all other lists in Python
- Non-EP path unchanged (still uses `.int().tolist()` in GroupedExperts — only 1 sync per layer there)

### Phase 4: Integration

#### 4a: Expert Utilization Diagnostics ✅ COMPLETE

**Per-step** — `bal:` field in training log line:
```
st:     5 | ls: 9.937500 | ... | bal: 0.031
```
Shows average coefficient of variation (CV = std/mean) across all MoE layers. Lower = better balanced. 0.0 = perfect.

**Validation-time** — detailed per-layer table:
```
  === MoE Expert Utilization @ step 50 (CV = std/mean, lower = better) ===
    layer |   min%   max%      CV |           bias range
    ----- |  -----  -----  ------ |  -------------------
    L   1 |  13.8   15.1  0.0310 | [-0.0030, +0.0020]
    L   2 |  14.0   14.6  0.0150 | [-0.0010, +0.0010]
    ...
      avg |                0.023 |
```

**Implementation** (in `_update_expert_bias` hook, train_mara.py):
- After all-reduce of `tokens_per_expert`, compute per-layer CV and token % distribution
- Store in `moe_stats` dict (nonlocal, updated each step)
- Per-step log reads `moe_stats['avg_cv']`
- Validation-time iterates `moe_stats['per_layer']` for detailed table

#### 4b: Checkpoint, AWD, Config Validation — COMPLETE

**Checkpoint: expert_bias buffer persistence** — already works. `expert_bias` is registered with
`persistent=True` in `MoE.__init__`, so `get_model_state_dict()` / `set_model_state_dict()` handle
save/load automatically. `tokens_per_expert` is intentionally non-persistent (zeroed each step).

**AWD component mapping for expert params** — `_build_component_map()` in `adaptive_wd.py` groups
MoE expert 3D params + shared expert params into `L{i}.ffn` components (same as dense FFN layers).
With EP, expert `p.numel()` is local (per-rank), but `all_reduce(SUM)` aggregates norms globally.
Fix: expert numel is multiplied by `ep_degree` so `w_rms_target` uses the correct global param count.

**Config validation** — in `validate_model_config()` (train_mara.py):
- `ep_degree > 1` without `moe_enabled` → error
- `num_experts % ep_degree != 0` → error
- `ep_degree > num_experts` → error
- `world_size % ep_degree != 0` → assert in `setup_ddp()`

#### 4c: Parameter Counting & Startup Logs — COMPLETE

**Bug fix: total params undercounted with EP** — `summarize_model()` used
`sum(p.numel() for p in model.parameters())`, but with EP each rank holds only
`num_local_experts` worth of expert params. Fix: add back `(ep_degree - 1) × expert_numel`
per MoE layer for the global total. Also fixed `num_experts` field (was reading
`experts.num_experts` = local count, now reads `moe.num_experts` = global count).

**New startup log fields** (in `print_model_summary`):
- `PER-RANK` — local param count when EP distributes experts (only shown with EP > 1)
- `EXPERTS` — shows `N total (M local, EP=K)` with EP, plain `N` without
- `ACTIVE/TOKEN` — params active per forward token: only `top_k/num_experts` fraction
  of expert params fire, plus all dense/attention/shared/router params. Shows as count + % of total.

---

## Working Test Configs

### Phase 1 (replicated experts, no EP)
```yaml
moe_enabled: true
moe_num_experts: 4             # 4 experts (keep low for SM86 for-loop)
moe_top_k: 2
moe_num_shared_experts: 1
moe_score_func: sigmoid
moe_load_balance_coeff: 0.001
moe_n_dense_layers: 1
moe_interleave_step: 1
moe_inner_dim: 3136
compile_model: false
# ep_degree: 1 (default, omit or set to 1)
```

### Phase 2 (Expert Parallel, EP=7)
```yaml
moe_enabled: true
moe_num_experts: 7             # Must be divisible by ep_degree
moe_top_k: 2
moe_num_shared_experts: 1
moe_score_func: sigmoid
moe_load_balance_coeff: 0.001
moe_n_dense_layers: 1
moe_interleave_step: 1
moe_inner_dim: 3136            # Must be divisible by world_size (for Muon shared_experts)
ep_degree: 7                   # 7 GPUs, 1 expert per GPU
compile_model: false
# cfg_embd and cfg_intermediate must also be divisible by world_size
```

---

## What the Other Agent Got Wrong

1. **"NO FSDP on experts"** — WRONG. TorchTitan DOES FSDP experts on `edp_mesh`.
2. **"ExpertParallel wrapper"** — It's a `ParallelStyle` applied via `distribute_module()`.
3. **"configure_optimizers: zero changes needed"** — WRONG. Expert weights are 3D nn.Parameter, not nn.Linear. Muon classifier needs updates. The `classify_param` fallback silently routed expert params to Muon, causing an infinite hang.
4. **"Copy ~150 lines"** — The EP code is ~400 lines including permutation utils, and depends on DTensor internals.
5. **"DeepEP dispatcher"** — DeepEP is a separate optional backend requiring custom ops. Standard EP uses plain `all_to_all_single_autograd`.
6. **Code snippets with ModuleList of FeedForward** — TorchTitan uses 3D tensors (GroupedExperts), NOT ModuleList. This is a fundamental architecture difference.
