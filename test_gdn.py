"""
Gated DeltaNet compatibility tests for our training stack.

Tests FLA library installation, SM86 Triton kernels, FSDP2 wrapping,
torch.compile, and hybrid layer patterns — all BEFORE integration.

Usage:
    python test_gdn.py                  # single GPU tests only
    torchrun --nproc_per_node=2 test_gdn.py   # include FSDP2 tests
"""

import sys
import time
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
PASSED = 0
FAILED = 0
SKIPPED = 0


def result(name, ok, msg="", skip=False):
    global PASSED, FAILED, SKIPPED
    if skip:
        SKIPPED += 1
        print(f"  [SKIP] {name}: {msg}")
    elif ok:
        PASSED += 1
        print(f"  [PASS] {name}{f': {msg}' if msg else ''}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}: {msg}")


# ===== 1. Environment checks =====
print("\n=== 1. Environment ===")

# GPU info
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    sm = f"sm{gpu.major}{gpu.minor}"
    result("CUDA GPU", True, f"{gpu.name} ({sm}, {gpu.total_memory / 1e9:.1f}GB)")
    result("SM86 check", gpu.major == 8 and gpu.minor == 6,
           f"Got {sm} — {'matches your RTX 3090s' if gpu.major == 8 and gpu.minor == 6 else 'different arch'}")
else:
    result("CUDA GPU", False, "No CUDA available")
    print("Cannot continue without CUDA.")
    sys.exit(1)

# PyTorch version
result("PyTorch version", True, torch.__version__)

# Triton
try:
    import triton
    result("Triton", True, triton.__version__)
except ImportError:
    result("Triton", False, "Not installed — FLA requires triton >= 3.0")

# causal-conv1d
try:
    import causal_conv1d
    ver = getattr(causal_conv1d, '__version__', 'unknown')
    result("causal-conv1d", True, ver)
except ImportError:
    result("causal-conv1d", False, "Not installed — FLA requires causal-conv1d >= 1.4.0")

# FLA
try:
    import fla
    ver = getattr(fla, '__version__', 'unknown')
    result("FLA library", True, ver)
except ImportError:
    result("FLA library", False,
           "Not installed. Run: pip install -U git+https://github.com/fla-org/flash-linear-attention")
    print("\nCannot continue without FLA.")
    sys.exit(1)


# ===== 2. Basic GatedDeltaNet forward/backward =====
print("\n=== 2. Basic Forward/Backward ===")

from fla.layers import GatedDeltaNet

B, T, D, H = 4, 512, 512, 4  # batch, seq, dim, heads

try:
    gdn = GatedDeltaNet(hidden_size=D, num_heads=H, mode='chunk').to(DTYPE).to(DEVICE)
    x = torch.randn(B, T, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y, *rest = gdn(x)
    result("Forward pass", True, f"output shape={y.shape}")

    loss = y.sum()
    loss.backward()
    result("Backward pass", True, f"grad shape={x.grad.shape}")
except Exception as e:
    result("Forward/Backward", False, str(e))

# Longer sequence
try:
    x_long = torch.randn(2, 2048, D, dtype=DTYPE, device=DEVICE)
    y_long, *_ = gdn(x_long)
    result("Seq len 2048", True, f"output shape={y_long.shape}")
except Exception as e:
    result("Seq len 2048", False, str(e))

# Check memory usage
torch.cuda.reset_peak_memory_stats()
try:
    x_mem = torch.randn(4, 2048, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y_mem, *_ = gdn(x_mem)
    y_mem.sum().backward()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    result("Memory footprint", True, f"peak={peak_mb:.0f}MB for B=4, T=2048, D={D}")
except Exception as e:
    result("Memory footprint", False, str(e))


# ===== 3. torch.compile compatibility =====
print("\n=== 3. torch.compile ===")

torch.cuda.empty_cache()

# Full module compile
try:
    gdn_compiled = torch.compile(
        GatedDeltaNet(hidden_size=D, num_heads=H, mode='chunk').to(DTYPE).to(DEVICE),
        mode="default"
    )
    x = torch.randn(B, T, D, dtype=DTYPE, device=DEVICE)
    y, *_ = gdn_compiled(x)
    result("torch.compile (full module)", True)
except Exception as e:
    result("torch.compile (full module)", False, str(e)[:200])

# Per-submodule compile (matches our strategy)
try:
    gdn_sub = GatedDeltaNet(hidden_size=D, num_heads=H, mode='chunk').to(DTYPE).to(DEVICE)
    # Compile individual submodules like we do in _apply_per_submodule_compile
    for name, mod in gdn_sub.named_children():
        if isinstance(mod, nn.Linear):
            compiled = torch.compile(mod, mode="default")
            setattr(gdn_sub, name, compiled)
    x = torch.randn(B, T, D, dtype=DTYPE, device=DEVICE)
    y, *_ = gdn_sub(x)
    result("torch.compile (per-submodule)", True)
except Exception as e:
    result("torch.compile (per-submodule)", False, str(e)[:200])

# Compile with backward
try:
    gdn_bwd = torch.compile(
        GatedDeltaNet(hidden_size=D, num_heads=H, mode='chunk').to(DTYPE).to(DEVICE),
        mode="default"
    )
    x = torch.randn(B, T, D, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y, *_ = gdn_bwd(x)
    y.sum().backward()
    result("torch.compile + backward", True)
except Exception as e:
    result("torch.compile + backward", False, str(e)[:200])


# ===== 4. Hybrid block pattern =====
print("\n=== 4. Hybrid Block Pattern ===")


class HybridBlock(nn.Module):
    """Minimal test: alternate between GatedDeltaNet and standard attention."""
    def __init__(self, dim, n_heads, use_gdn=True):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.use_gdn = use_gdn
        if use_gdn:
            self.attn = GatedDeltaNet(hidden_size=dim, num_heads=n_heads, mode='chunk')
        else:
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        if self.use_gdn:
            out, *_ = self.attn(h)
        else:
            mask = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=h.device, dtype=h.dtype)
            out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        return x + out


class HybridModel(nn.Module):
    """3:1 hybrid: 3 GDN blocks per 1 softmax attention block."""
    def __init__(self, n_layers=8, dim=256, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        gdn_count = 0
        attn_count = 0
        for i in range(n_layers):
            use_gdn = (i % 4 != 3)  # layers 0,1,2=GDN, 3=attn, 4,5,6=GDN, 7=attn
            self.layers.append(HybridBlock(dim, n_heads, use_gdn=use_gdn))
            if use_gdn:
                gdn_count += 1
            else:
                attn_count += 1
        self.dim = dim
        print(f"  Hybrid model: {gdn_count} GDN + {attn_count} softmax attn layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


try:
    hybrid = HybridModel(n_layers=8, dim=256, n_heads=4).to(DTYPE).to(DEVICE)
    x = torch.randn(4, 512, 256, dtype=DTYPE, device=DEVICE, requires_grad=True)
    y = hybrid(x)
    y.sum().backward()
    result("Hybrid forward+backward", True, f"output shape={y.shape}")
except Exception as e:
    result("Hybrid forward+backward", False, str(e)[:200])


# ===== 5. FSDP2 compatibility =====
print("\n=== 5. FSDP2 Compatibility ===")

import os
is_distributed = int(os.environ.get("RANK", -1)) >= 0

if not is_distributed:
    result("FSDP2 wrapping", True, skip=True, msg="Run with torchrun for FSDP2 tests")
    result("FSDP2 + compile", True, skip=True, msg="Run with torchrun for FSDP2 tests")
else:
    import torch.distributed as dist
    from torch.distributed import init_process_group, destroy_process_group
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    dp_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    mp_policy = MixedPrecisionPolicy(param_dtype=DTYPE, reduce_dtype=DTYPE)

    # Test FSDP2 wrapping of hybrid model
    try:
        hybrid_fsdp = HybridModel(n_layers=8, dim=256, n_heads=4)

        # Per-layer sharding (matches our strategy)
        for layer in hybrid_fsdp.layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(hybrid_fsdp, mesh=dp_mesh, mp_policy=mp_policy)

        # Materialize
        hybrid_fsdp = hybrid_fsdp.to_empty(device=device)
        for p in hybrid_fsdp.parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, std=0.02)
        for name, buf in hybrid_fsdp.named_buffers():
            if buf is not None:
                torch.nn.init.zeros_(buf)

        x = torch.randn(4, 512, 256, dtype=DTYPE, device=device, requires_grad=True)
        y = hybrid_fsdp(x)
        y.sum().backward()
        if rank == 0:
            result("FSDP2 wrapping", True, "per-layer sharding works")
    except Exception as e:
        if rank == 0:
            result("FSDP2 wrapping", False, str(e)[:200])

    # Test FSDP2 + compile
    try:
        hybrid_fsdp2 = HybridModel(n_layers=8, dim=256, n_heads=4)
        for layer in hybrid_fsdp2.layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(hybrid_fsdp2, mesh=dp_mesh, mp_policy=mp_policy)
        hybrid_fsdp2 = hybrid_fsdp2.to_empty(device=device)
        for p in hybrid_fsdp2.parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, std=0.02)
        for name, buf in hybrid_fsdp2.named_buffers():
            if buf is not None:
                torch.nn.init.zeros_(buf)

        # Compile GDN layers only (skip softmax attn, like we skip MoE experts)
        for layer in hybrid_fsdp2.layers:
            if layer.use_gdn:
                layer.attn = torch.compile(layer.attn, mode="default")

        x = torch.randn(4, 512, 256, dtype=DTYPE, device=device, requires_grad=True)
        y = hybrid_fsdp2(x)
        y.sum().backward()
        if rank == 0:
            result("FSDP2 + compile (GDN layers)", True)
    except Exception as e:
        if rank == 0:
            result("FSDP2 + compile (GDN layers)", False, str(e)[:200])

    destroy_process_group()


# ===== 6. Throughput comparison =====
print("\n=== 6. Throughput (single GPU) ===")

torch.cuda.empty_cache()


def bench(name, module, x, warmup=3, iters=10):
    """Time forward+backward."""
    for _ in range(warmup):
        y = module(x)
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        module.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = module(x)
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    result(name, True, f"{dt:.1f}ms per fwd+bwd")
    return dt


D_bench, H_bench = 512, 8
gdn_bench = GatedDeltaNet(hidden_size=D_bench, num_heads=H_bench, mode='chunk').to(DTYPE).to(DEVICE)

# Compare at different sequence lengths
for T_bench in [512, 1024, 2048, 4096]:
    try:
        x_bench = torch.randn(4, T_bench, D_bench, dtype=DTYPE, device=DEVICE, requires_grad=True)
        bench(f"GDN T={T_bench}", gdn_bench, x_bench)
    except Exception as e:
        result(f"GDN T={T_bench}", False, str(e)[:100])

# ===== 7. GatedDeltaNet internals =====
print("\n=== 7. GatedDeltaNet Internals ===")

gdn_inspect = GatedDeltaNet(hidden_size=512, num_heads=8, mode='chunk').to(DTYPE).to(DEVICE)
print(f"  Module structure:")
for name, mod in gdn_inspect.named_modules():
    if name:  # skip root
        pcount = sum(p.numel() for p in mod.parameters(recurse=False))
        if pcount > 0:
            print(f"    {name}: {mod.__class__.__name__} ({pcount:,} params)")

# Check for buffers (important for FSDP/CPU offload)
bufs = list(gdn_inspect.named_buffers())
if bufs:
    print(f"  Buffers ({len(bufs)}):")
    for name, buf in bufs:
        print(f"    {name}: {buf.shape} {buf.dtype} {buf.device}")
else:
    print(f"  No buffers (good for FSDP)")

# Check what extra outputs the forward returns
x_test = torch.randn(2, 128, 512, dtype=DTYPE, device=DEVICE)
outputs = gdn_inspect(x_test)
print(f"  Forward returns: {len(outputs)} values")
for i, o in enumerate(outputs):
    if o is None:
        print(f"    [{i}]: None")
    elif isinstance(o, torch.Tensor):
        print(f"    [{i}]: Tensor {o.shape} {o.dtype}")
    else:
        print(f"    [{i}]: {type(o).__name__}")


# ===== Summary =====
print(f"\n{'='*50}")
print(f"Results: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
if FAILED == 0:
    print("All tests passed!")
else:
    print(f"{FAILED} test(s) failed — review output above")
print(f"{'='*50}")
