#!/usr/bin/env python3
"""
check_flash_attn.py — Flash Attention diagnostic for training rigs

Checks:
  1. Whether flash-attn is installed (and which version)
  2. GPU compute capability compatibility
  3. PyTorch SDPA backend availability (flash, memory-efficient, math)
  4. xformers status for comparison
  5. Functional smoke test with actual tensors
  6. FLA (flash-linear-attention) status

Usage:
    python check_flash_attn.py
"""
import sys
import subprocess

# ─── Colors ───────────────────────────────────────────────────────────────────
G = "\033[92m"   # green
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
C = "\033[96m"   # cyan
B = "\033[1m"    # bold
RST = "\033[0m"

def ok(msg):   print(f"  {G}✓{RST} {msg}")
def warn(msg): print(f"  {Y}⚠{RST} {msg}")
def fail(msg): print(f"  {R}✗{RST} {msg}")
def head(msg): print(f"\n{B}{C}{'─'*60}\n  {msg}\n{'─'*60}{RST}")

# ─── 1. Python & PyTorch basics ──────────────────────────────────────────────
head("Environment")
print(f"  Python:  {sys.version.split()[0]}  ({sys.executable})")

try:
    import torch
    ok(f"PyTorch:  {torch.__version__}")
    ok(f"CUDA:     {torch.version.cuda or 'N/A'}")
except ImportError:
    fail("PyTorch not installed!")
    sys.exit(1)

if not torch.cuda.is_available():
    fail("CUDA not available — can't test Flash Attention.")
    sys.exit(1)

# ─── 2. GPU inventory & compute capability ───────────────────────────────────
head("GPU Inventory")

FA2_MIN_CC = 75   # SM 7.5+ (Turing and above)
FA2_BEST_CC = 80  # SM 8.0+ (Ampere and above, supports bf16)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    cc = props.major * 10 + props.minor
    cc_str = f"{props.major}.{props.minor}"
    name = props.name
    vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)

    status = ""
    if cc >= FA2_BEST_CC:
        status = f"{G}[FA2 optimal — Ampere+]{RST}"
    elif cc >= FA2_MIN_CC:
        status = f"{Y}[FA2 supported — Turing]{RST}"
    else:
        status = f"{R}[FA2 NOT supported — need SM≥7.5]{RST}"

    print(f"  GPU {i}: {name}  |  SM {cc_str}  |  {vram_gb:.1f} GB  {status}")

# ─── 3. Flash Attention package ──────────────────────────────────────────────
head("Flash Attention (flash-attn package)")

flash_attn_ver = None
try:
    import flash_attn
    flash_attn_ver = getattr(flash_attn, "__version__", "unknown")
    ok(f"flash-attn installed: v{flash_attn_ver}")

    # Check for FA2 specific imports
    try:
        from flash_attn import flash_attn_func
        ok("flash_attn_func available (FA2 core)")
    except ImportError:
        warn("flash_attn_func not found — may be an older version")

    try:
        from flash_attn import flash_attn_varlen_func
        ok("flash_attn_varlen_func available (variable-length / packing support)")
    except ImportError:
        warn("flash_attn_varlen_func not found")

    try:
        from flash_attn.bert_padding import pad_input, unpad_input
        ok("bert_padding utilities available (unpad/pad for efficiency)")
    except ImportError:
        warn("bert_padding utilities not found (optional)")

except ImportError:
    fail("flash-attn NOT installed")
    print(f"      Install with: pip install flash-attn --no-build-isolation")

# ─── 4. PyTorch SDPA backends ───────────────────────────────────────────────
head("PyTorch SDPA Backends (torch.nn.functional.scaled_dot_product_attention)")

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    ok("SDPA API available (torch.nn.attention)")
except ImportError:
    # Older torch
    try:
        from torch.backends.cuda import (
            flash_sdp_enabled,
            mem_efficient_sdp_enabled,
            math_sdp_enabled,
        )
        ok("SDPA available (legacy torch.backends.cuda API)")
    except ImportError:
        warn("SDPA backend query not available — torch may be too old")

# Probe each backend with a real tensor
print()
print(f"  {B}Backend probe (actual dispatch test):{RST}")

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = "cuda:0"
B_size, H, S, D = 2, 8, 256, 64

q = torch.randn(B_size, H, S, D, dtype=dtype, device=device)
k = torch.randn(B_size, H, S, D, dtype=dtype, device=device)
v = torch.randn(B_size, H, S, D, dtype=dtype, device=device)

# Use the modern API (torch 2.8+) with fallback to legacy
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _use_new_api = True
    backends = {
        "Flash (FA2)":                      [SDPBackend.FLASH_ATTENTION],
        "Memory-efficient (CuDNN/xformers)":[SDPBackend.EFFICIENT_ATTENTION],
        "Math (fallback)":                  [SDPBackend.MATH],
    }
    # Also probe CuDNN backend if available (torch 2.8+)
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        backends["CuDNN Flash"] = [SDPBackend.CUDNN_ATTENTION]
except ImportError:
    _use_new_api = False
    backends = {
        "Flash (FA2)":                      {"enable_flash": True,  "enable_math": False, "enable_mem_efficient": False},
        "Memory-efficient (xformers-like)": {"enable_flash": False, "enable_math": False, "enable_mem_efficient": True},
        "Math (fallback)":                  {"enable_flash": False, "enable_math": True,  "enable_mem_efficient": False},
    }

sdpa_flash_works = False  # Track whether SDPA's built-in flash backend works

for name, flags in backends.items():
    try:
        if _use_new_api:
            with sdpa_kernel(flags):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            with torch.backends.cuda.sdp_kernel(**flags):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        ok(f"{name}: works  (output shape: {tuple(out.shape)})")
        if "Flash" in name and "CuDNN" not in name:
            sdpa_flash_works = True
    except RuntimeError as e:
        err_short = str(e).split('\n')[0][:80]
        fail(f"{name}: {err_short}")
    except Exception as e:
        fail(f"{name}: {type(e).__name__}: {e}")

# ─── 5. xformers status ─────────────────────────────────────────────────────
head("xformers")

try:
    import xformers
    xf_ver = getattr(xformers, "__version__", "unknown")
    ok(f"xformers installed: v{xf_ver}")
    try:
        from xformers.ops import memory_efficient_attention
        ok("memory_efficient_attention available")
    except ImportError:
        warn("memory_efficient_attention not importable")
    except Exception as e:
        fail(f"xformers C++/CUDA extensions broken: {type(e).__name__}")
        # Check for version mismatch
        try:
            import torch
            xf_torch = getattr(xformers, '_C', None)
            warn(f"xformers v{xf_ver} likely built for a different PyTorch/CUDA combo.")
            print(f"      You're on PyTorch {torch.__version__} — reinstall xformers to match:")
            print(f"      pip install -U xformers --index-url https://download.pytorch.org/whl/cu128")
        except Exception:
            pass
except ImportError:
    warn("xformers not installed (optional if using FA2 via SDPA)")
except Exception as e:
    fail(f"xformers import crashed: {type(e).__name__}: {str(e)[:100]}")
    warn("This usually means xformers was built for a different PyTorch/CUDA version.")
    print(f"      Reinstall: pip install -U xformers --index-url https://download.pytorch.org/whl/cu128")

# ─── 6. FLA (flash-linear-attention) status ──────────────────────────────────
head("FLA (flash-linear-attention)")

try:
    import fla
    fla_ver = getattr(fla, "__version__", "unknown")
    ok(f"fla installed: v{fla_ver}")
    try:
        from fla.layers import GatedDeltaNet
        ok("GatedDeltaNet layer importable")
    except ImportError:
        warn("GatedDeltaNet not importable — may need newer fla version")
except ImportError:
    fail("fla NOT installed")
    print(f"      Install with: pip install flash-linear-attention --no-build-isolation")

# ─── 7. Triton status (needed for FLA kernels) ──────────────────────────────
head("Triton")

try:
    import triton
    triton_ver = getattr(triton, "__version__", "unknown")
    ok(f"triton installed: v{triton_ver}")
except ImportError:
    warn("triton not installed (required for FLA kernels)")
    print(f"      Usually installed automatically with PyTorch or fla")

# ─── 8. Summary & recommendations ───────────────────────────────────────────
head("Recommendations")

props = torch.cuda.get_device_properties(0)
cc = props.major * 10 + props.minor

if flash_attn_ver:
    # Parse major version
    try:
        fa_major = int(flash_attn_ver.split('.')[0])
    except (ValueError, IndexError):
        fa_major = 0

    if fa_major >= 2:
        ok("Flash Attention 2 package installed — available as both standalone and SDPA backend.")
        print(f"      In your model, you can use either:")
        print(f"        • flash_attn.flash_attn_func() directly (+ varlen packing support)")
        print(f"        • F.scaled_dot_product_attention() which auto-dispatches to flash")
    else:
        warn(f"Flash Attention v{flash_attn_ver} detected — consider upgrading to v2.x+")
        print(f"      pip install -U flash-attn --no-build-isolation")
elif sdpa_flash_works:
    ok("SDPA Flash backend is working — you're already getting Flash Attention performance!")
    print(f"      PyTorch's built-in SDPA dispatches through a flash kernel automatically.")
    print(f"      The standalone flash-attn package is NOT required unless you need:")
    print(f"        • flash_attn_varlen_func() for variable-length sequence packing")
    print(f"        • Fine-grained control over flash kernel parameters")
    print(f"      Your F.scaled_dot_product_attention(is_causal=True) calls are optimal as-is.")
else:
    if cc >= FA2_MIN_CC:
        print(f"  {R}→ Your GPU (SM {cc//10}.{cc%10}) supports Flash Attention 2 but neither{RST}")
        print(f"  {R}  the standalone package nor SDPA flash backend are working!{RST}")
        print(f"    Install: pip install flash-attn --no-build-isolation")
        print(f"    Or upgrade PyTorch for built-in SDPA flash support.")
    else:
        print(f"  GPU compute capability SM {cc//10}.{cc%10} doesn't support FA2.")
        print(f"  Use xformers memory_efficient_attention or PyTorch SDPA math fallback.")

# Check if bf16 is supported
if torch.cuda.is_bf16_supported():
    ok("bf16 supported — use with Flash Attention for best throughput.")
else:
    warn("bf16 NOT supported — Flash Attention will use fp16.")

print()
