"""z-loss Option-D: 4-variant precision gate + real-shape memory (rig).

RUN ON THE RIG:  python zloss_variants_rig.py

Per Rook #125: the python-side fp32 reconstruction does NOT fix the gradient,
because the catastrophic cancellation happens INSIDE the CCE kernel computing
ce_none = logZ - logit_target (bits already lost before we get the fp32 result).
The untested lever is CCE's OWN backward accumulation flags
(accum_e_fp32 / accum_c_fp32), which force fp32 accumulation of the e/c gradient
contractions INSIDE the kernel -- exactly where the large target-class term that
cancels against our explicit target-logit term is computed.

Tests 4 variants of the gradient against the fp32 ANALYTIC TRUTH
  gL = (2*logZ/Nk) * softmax(L),  ge_truth = gL @ c,  gc_truth = gL.T @ e   (fp32):

  A. current Option D : ce_none (CCE default flags) + bf16 target dot
  B. python fp32 recon: ce_none.float() + fp32 target dot          (expected: ~no help)
  C. CCE accum flags  : accum_e_fp32=True, accum_c_fp32=True + bf16 target dot   <-- the bet
  D. combined         : accum flags + fp32 target dot

Reports grad_e/grad_c (cos, norm-rel), zloss scalar, and real-shape peak memory
at dreadnought_v2's head shape (N=24576, V=32000, D=2560). Also exercises the
safe_targets ignore_index guard, and a memory case where alpha=0 STILL runs the
reconstruction (no short-circuit).

DECISION RULE (Rook #125): adopt a variant as the default backend ONLY if its
gradient agreement is strictly better than A, zloss no worse, and real-shape
memory delta acceptable. If none beat A, ship A and document the floor.
"""
import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16
PAD = 0


def metrics(a, b):
    a, b = a.float(), b.float()
    nrel = (a - b).norm().item() / b.norm().clamp_min(1e-12).item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return nrel, cos


def masked_zloss(lse, targets, pad_id):
    f = lse.float()
    keep = (targets != pad_id).to(f.dtype)
    den = keep.sum().clamp_min(1.0)
    return (f * f * keep).sum() / den


def safe_gather_dot(e, c, targets, ignore_index, fp32):
    """target logit = (e . c[target]) with an ignore_index-safe gather.

    c.index_select crashes / mis-gathers if targets carries ignore_index
    (e.g. -100). Clamp ignored positions to row 0 for the gather; those rows
    are masked out of the zloss downstream so the bogus value never matters.
    """
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    # NOTE (vocab-parallel): this manual gather does NOT inherit CCE's
    # vocab-parallel target remapping / rank-local handling. If VP is ever
    # enabled, this must mirror CCE's target handling.
    c_rows = c.index_select(0, safe_targets)
    if fp32:
        return (e.float() * c_rows.float()).sum(-1)
    return (e * c_rows).sum(-1)


def analytic_truth(e_f, c_f, targets, pad_id):
    L = e_f @ c_f.t()
    logZ = torch.logsumexp(L, dim=-1)
    sm = torch.softmax(L, dim=-1)
    keep = (targets != pad_id).float()
    Nk = keep.sum().clamp_min(1.0)
    w = (2.0 * logZ / Nk) * keep
    gL = (w[:, None] * sm) * keep[:, None]
    return logZ, gL @ c_f, gL.t() @ e_f


def optionD_lse(lce, e, c, targets, pad_id, accum_fp32, fp32_dot):
    kw = dict(reduction="none", ignore_index=pad_id)
    if accum_fp32:
        kw.update(accum_e_fp32=True, accum_c_fp32=True)
    ce_none = lce(e, c, targets, **kw)
    if fp32_dot:
        ce_none = ce_none.float()
    tgt = safe_gather_dot(e, c, targets, pad_id, fp32_dot)
    return ce_none + tgt


VARIANTS = [
    ("A current (default flags, bf16 dot)", False, False),
    ("B python fp32 recon",                 False, True),
    ("C CCE accum_*_fp32 (bf16 dot)",       True,  False),
    ("D accum flags + fp32 dot",            True,  True),
]


def correctness(lce):
    torch.manual_seed(0)
    N, V, D = 512, 4096, 512
    e0 = torch.randn(N, D, device=DEV, dtype=DT)
    c0 = (torch.randn(V, D, device=DEV, dtype=DT) / (D ** 0.5))
    targets = torch.randint(1, V, (N,), device=DEV); targets[:32] = PAD
    e_f, c_f = e0.float(), c0.float()
    logZ_t, ge_truth, gc_truth = analytic_truth(e_f, c_f, targets, PAD)
    keep = (targets != PAD).float(); Nk = keep.sum().clamp_min(1.0)
    zT = ((logZ_t ** 2) * keep).sum().item() / Nk.item()

    print(f"\n{'='*78}\nCORRECTNESS vs fp32 analytic truth (N={N},V={V},D={D},bf16)   truth zloss={zT:.6e}\n{'='*78}")
    print(f"  {'variant':<34}  {'grad_e (nrel, cos)':<26}  {'grad_c (nrel, cos)':<26}  zloss")
    results = {}
    for label, accum, fp32 in VARIANTS:
        try:
            e = e0.clone().requires_grad_(True); c = c0.clone().requires_grad_(True)
            lse = optionD_lse(lce, e, c, targets, PAD, accum, fp32)
            z = masked_zloss(lse, targets, PAD); z.backward()
            ne, ce = metrics(e.grad.float(), ge_truth)
            nc, cc = metrics(c.grad.float(), gc_truth)
            zrel = abs(z.item() - zT) / (abs(zT) + 1e-12)
            print(f"  {label:<34}  ({ne:.3e}, {ce:.5f})    ({nc:.3e}, {cc:.5f})    "
                  f"{z.item():.5e} (rel {zrel:.1e})")
            results[label] = dict(ne=ne, ce=ce, nc=nc, cc=cc, z=z.item(), zrel=zrel)
        except Exception as ex:
            print(f"  {label:<34}  ERROR {type(ex).__name__}: {ex}")
            results[label] = None
    return results


def memory(lce):
    """Real-shape (dreadnought_v2 head) peak-memory deltas. alpha=0 case STILL
    runs the reconstruction (no short-circuit), per Rook #125 refinement 1."""
    N, V, D = 24576, 32000, 2560
    print(f"\n{'='*78}\nREAL-SHAPE MEMORY  N={N} V={V} D={D} (dreadnought head, B3*T8192)\n{'='*78}")
    logits_gb = N * V * 4 / 1e9
    print(f"  [N,V] fp32 logits (the thing we AVOID): {logits_gb:.2f} GB")
    print(f"  [N,D] target gather (what Option D adds): {N*D*2/1e9:.3f} GB bf16 / {N*D*4/1e9:.3f} GB fp32")

    torch.manual_seed(0)
    e0 = torch.randn(N, D, device=DEV, dtype=DT)
    c0 = (torch.randn(V, D, device=DEV, dtype=DT) / (D ** 0.5))
    targets = torch.randint(1, V, (N,), device=DEV); targets[:64] = PAD

    def peak(fn):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn(); torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1e9

    def ce_only():  # baseline: enabled=false, just CE
        e = e0.clone().requires_grad_(True); c = c0.clone().requires_grad_(True)
        lce(e, c, targets, reduction="mean", ignore_index=PAD).backward()

    def zloss_variant(accum, fp32, alpha):
        e = e0.clone().requires_grad_(True); c = c0.clone().requires_grad_(True)
        ce = lce(e, c, targets, reduction="mean", ignore_index=PAD)
        lse = optionD_lse(lce, e, c, targets, PAD, accum, fp32)  # reconstruction ALWAYS runs
        z = masked_zloss(lse, targets, PAD)
        (ce + alpha * z).backward()                              # alpha=0 -> z still in graph

    try:
        b = peak(ce_only)
        print(f"\n  baseline CE-only (enabled=false)          peak delta {b:.3f} GB")
        for label, accum, fp32 in VARIANTS:
            try:
                p0 = peak(lambda: zloss_variant(accum, fp32, 0.0))
                pa = peak(lambda: zloss_variant(accum, fp32, 1e-4))
                print(f"  {label:<34} alpha=0 {p0:.3f} GB (+{p0-b:.3f})   "
                      f"alpha>0 {pa:.3f} GB (+{pa-b:.3f})")
            except RuntimeError as ex:
                print(f"  {label:<34} OOM/failed: {type(ex).__name__}")
    except RuntimeError as ex:
        print(f"  baseline failed: {ex}")


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA."); return 0
    try:
        from cut_cross_entropy import linear_cross_entropy as lce
        import inspect
        sig = inspect.signature(lce)
        has_accum = "accum_e_fp32" in sig.parameters and "accum_c_fp32" in sig.parameters
    except Exception as ex:
        print(f"SKIP: cut_cross_entropy import failed: {ex}"); return 0
    import cut_cross_entropy as cce
    print(f"cut_cross_entropy {getattr(cce,'__version__','?')} | torch {torch.__version__} "
          f"| {torch.cuda.get_device_name(0)}")
    print(f"accum_e_fp32/accum_c_fp32 in signature: {has_accum}  "
          f"{'(variant C/D valid)' if has_accum else '(C/D will fall back to default flags!)'}")

    res = correctness(lce)
    memory(lce)

    print(f"\n{'='*78}\nDECISION\n{'='*78}")
    A = res.get("A current (default flags, bf16 dot)")
    if A is None:
        print("  variant A failed — cannot compare."); return 1
    base_cos = min(A["ce"], A["cc"])
    best = "A"; best_cos = base_cos
    for label, r in res.items():
        if r is None or label.startswith("A"):
            continue
        cos = min(r["ce"], r["cc"])
        better = cos > best_cos + 1e-4 and r["zrel"] <= A["zrel"] * 1.5
        tag = "  <-- strictly better" if better else ""
        if better and cos > best_cos:
            best = label[0]; best_cos = cos
    print(f"  A (current bf16) grad cos floor: {base_cos:.5f}  (norm-rel ge {A['ne']:.2e}, gc {A['nc']:.2e})")
    for label, r in res.items():
        if r is None or label.startswith("A"):
            continue
        print(f"  {label[0]}: grad cos {min(r['ce'],r['cc']):.5f}  "
              f"{'BETTER than A' if min(r['ce'],r['cc']) > base_cos + 1e-4 else 'no better than A'}")
    if best != "A":
        print(f"\n  => Variant {best} measurably improves the gradient. Adopt it IF its")
        print(f"     real-shape memory delta (above) is acceptable. C (accum flags) is")
        print(f"     free memory-wise; D adds the [N,D] fp32 operand.")
    else:
        print(f"\n  => No variant beats A. SHIP A (current bf16 Option D) and document the")
        print(f"     measured precision floor: grad cosine ~{base_cos:.4f}, "
              f"norm-rel ~{max(A['ne'],A['nc']):.0e}. Acceptable for a small annealed")
        print(f"     regularizer (alpha~1e-4). Custom kernel stays deferred (OLMo trap).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
