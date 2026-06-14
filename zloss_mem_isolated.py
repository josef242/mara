"""Isolated real-shape peak-memory for each z-loss variant (rig).

RUN ON THE RIG:  python zloss_mem_isolated.py

Each variant is measured in a FRESH SUBPROCESS so allocator fragmentation /
reserved-memory carryover from earlier variants cannot distort the peak (which
made variant D spuriously OOM when run 4th in one process). Each child measures
exactly one config from a clean CUDA context and prints one line.

Shapes = dreadnought_v2 head: N=B*T=24576, V=32000, D=2560, bf16.
"""
import os
import sys
import subprocess

N, V, D = 24576, 32000, 2560
PAD = 0

# (label, accum_fp32, fp32_dot) — None config = baseline CE-only
CONFIGS = [
    ("baseline_CE_only", None, None),
    ("A_default_bf16dot", False, False),
    ("B_python_fp32",     False, True),
    ("C_accum_flags",     True,  False),
    ("D_accum_plus_fp32", True,  True),
]

CHILD = r'''
import torch, torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy as lce
N,V,D,PAD = {N},{V},{D},{PAD}
accum = {accum}; fp32_dot = {fp32}; baseline = {baseline}; alpha = {alpha}
DT = torch.bfloat16
torch.manual_seed(0)
e0 = torch.randn(N,D,device="cuda",dtype=DT)
c0 = (torch.randn(V,D,device="cuda",dtype=DT)/(D**0.5))
tgt = torch.randint(1,V,(N,),device="cuda"); tgt[:64]=PAD

def safe_dot(e,c,targets,ig,f32):
    valid = targets!=ig
    st = targets.masked_fill(~valid,0)
    rows = c.index_select(0,st)
    return (e.float()*rows.float()).sum(-1) if f32 else (e*rows).sum(-1)

def mz(lse,targets,ig):
    f=lse.float(); k=(targets!=ig).to(f.dtype); d=k.sum().clamp_min(1.0)
    return (f*f*k).sum()/d

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
e = e0.clone().requires_grad_(True); c = c0.clone().requires_grad_(True)
if baseline:
    lce(e,c,tgt,reduction="mean",ignore_index=PAD).backward()
else:
    ce = lce(e,c,tgt,reduction="mean",ignore_index=PAD)
    kw = dict(reduction="none", ignore_index=PAD)
    if accum: kw.update(accum_e_fp32=True, accum_c_fp32=True)
    ce_none = lce(e,c,tgt,**kw)
    if fp32_dot: ce_none = ce_none.float()
    lse = ce_none + safe_dot(e,c,tgt,PAD,fp32_dot)   # reconstruction ALWAYS runs
    z = mz(lse,tgt,PAD)
    (ce + alpha*z).backward()                        # alpha=0 -> z still in graph
torch.cuda.synchronize()
peak = (torch.cuda.max_memory_allocated()-base)/1e9
print(f"{{peak:.4f}}")
'''


def run_child(accum, fp32, baseline, alpha):
    code = CHILD.format(N=N, V=V, D=D, PAD=PAD,
                        accum=accum if accum is not None else False,
                        fp32=fp32 if fp32 is not None else False,
                        baseline=baseline, alpha=alpha)
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            err = out.stderr.strip().splitlines()
            last = err[-1] if err else "(no stderr)"
            return None, last
        return float(out.stdout.strip().splitlines()[-1]), None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


def main():
    print(f"Isolated real-shape memory  N={N} V={V} D={D} bf16  "
          f"([N,V] fp32 logits we avoid = {N*V*4/1e9:.2f} GB)")
    print(f"{'variant':<22} {'alpha=0 peak':>14} {'alpha>0 peak':>14}   notes")
    base = None
    for label, accum, fp32 in CONFIGS:
        if label.startswith("baseline"):
            p, err = run_child(None, None, True, 0.0)
            base = p
            print(f"{label:<22} {('%.4f GB'%p) if p else 'FAIL: '+str(err):>14} "
                  f"{'-':>14}")
            continue
        p0, e0 = run_child(accum, fp32, False, 0.0)
        pa, ea = run_child(accum, fp32, False, 1e-4)
        def fmt(p, e):
            if p is None:
                return f"OOM/FAIL"
            d = f" (+{p-base:.3f})" if base else ""
            return f"{p:.4f} GB{d}"
        print(f"{label:<22} {fmt(p0,e0):>14} {fmt(pa,ea):>14}")
        if e0 or ea:
            print(f"    err: {e0 or ea}")
    print("\nEach variant measured in a clean subprocess — no cross-variant "
          "fragmentation. If D fits here, its earlier OOM was allocator carryover.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
