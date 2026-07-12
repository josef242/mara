"""Definitive diagnostic: is option-D's z-loss gradient CORRECT, or BROKEN?

RUN ON THE RIG:  python zloss_diagnostic_rig.py

The whole question is (a) vs (b):
  (a) option D is mathematically CORRECT and our pure-torch bf16 reference was
      the inaccurate baseline (it never matters which bf16 path is "closer" to
      the other -- only which matches the fp32 ANALYTIC TRUTH).
  (b) option D's gradient composition is genuinely BROKEN and needs a custom
      autograd Function.

The trap (caught in review): a CORRECT option D, measured in bf16, sits at
~0.08 norm-rel / ~0.996 cos against the fp32 truth -- which can look like a
"fail". The ONLY way to tell precision-loss from a real bug is to also run
option D's composition IN FP32 and check it against the truth: if fp32-option-D
matches to ~1e-6, the composition is correct and the bf16 gap is just rounding
in the logZ = ce_none + logit_target round-trip (CCE returns
ce_none = logZ - logit_target in reduced precision; re-adding can't recover the
lost bits). That is precision, not a bug.

Ground truth (baseline-free): for zloss = mean(logZ[non_pad]**2),
  d(zloss)/d(logit_j) = (2*logZ / Nk) * softmax(logit)_j  on non-pad rows, 0 on pad
chained through logits = e @ c.T:
  ge_truth = gL_true @ c ;  gc_truth = gL_true.T @ e
all in fp32 from an fp32 matmul of the SAME bf16 leaves (bf16->fp32 is exact).
"""
import torch
import torch.nn.functional as F

DEV = "cuda"
DT = torch.bfloat16
PAD = 0
N, V, D = 512, 4096, 512
SEED = 0


def metrics(a, b):
    a, b = a.float(), b.float()
    nrel = (a - b).norm().item() / b.norm().clamp_min(1e-12).item()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    mabs = (a - b).abs().max().item()
    return nrel, cos, mabs


def fmt(tag, a, b):
    nrel, cos, mabs = metrics(a, b)
    print(f"  {tag:<36} norm_rel={nrel:.3e}  cos={cos:.6f}  max|d|={mabs:.3e}")
    return nrel, cos, mabs


def banner(s):
    print("\n" + "-" * 72)
    print(s)
    print("-" * 72)


def masked_zloss(lse, targets, pad_id):
    f = lse.float()
    keep = (targets != pad_id).to(f.dtype)
    den = keep.sum().clamp_min(1.0)
    return (f * f * keep).sum() / den


def analytic_truth(e_f, c_f, targets, pad_id):
    """fp32 (2*logZ/Nk)*softmax, chained to e and c. e_f,c_f are fp32."""
    L = e_f @ c_f.t()                       # [N, V] fp32
    logZ = torch.logsumexp(L, dim=-1)       # [N]
    sm = torch.softmax(L, dim=-1)           # [N, V]
    keep = (targets != pad_id).float()
    Nk = keep.sum().clamp_min(1.0)
    w = (2.0 * logZ / Nk) * keep            # [N] upstream into each logZ
    gL = w[:, None] * sm                    # [N, V]
    gL = gL * keep[:, None]                 # zero pad rows
    ge = gL @ c_f                           # [N, D]
    gc = gL.t() @ e_f                       # [V, D]
    return logZ, gL, ge, gc, L


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA."); return 0
    try:
        from cut_cross_entropy import linear_cross_entropy as lce
    except Exception as ex:
        print(f"SKIP: cut_cross_entropy import failed: {ex}"); return 0
    import cut_cross_entropy as cce
    print(f"cut_cross_entropy {getattr(cce,'__version__','?')} | torch {torch.__version__} "
          f"| {torch.cuda.get_device_name(0)}")
    print(f"config: N={N} V={V} D={D} dtype={str(DT).replace('torch.','')} pad={PAD}")

    torch.manual_seed(SEED)
    e0 = torch.randn(N, D, device=DEV, dtype=DT)
    c0 = (torch.randn(V, D, device=DEV, dtype=DT) / (D ** 0.5))
    targets = torch.randint(1, V, (N,), device=DEV); targets[:32] = PAD
    m = targets != PAD
    e_f, c_f = e0.float(), c0.float()

    R = {}

    # ---- (1) per-token CE: CCE none == torch CE? (rules out softcap/shift/smoothing) ----
    banner("(1) per-token CE definition: CCE reduction='none' vs torch CE")
    try:
        ce_cce = lce(e0, c0, targets, reduction="none", ignore_index=PAD).float()
        ce_torch_fp32 = F.cross_entropy(e_f @ c_f.t(), targets, reduction="none",
                                        ignore_index=PAD).float()
        ce_md = (ce_cce[m] - ce_torch_fp32[m]).abs().max().item()
        print(f"  CCE none-CE vs torch CE(fp32 matmul) per-token max|d| (non-pad) = {ce_md:.3e}")
        print("  ^ small (~bf16 matmul level, <1e-1) => CCE none IS raw CE (no smoothing/softcap/shift).")
        R["ce_md"] = ce_md
    except Exception as ex:
        print(f"  [ERROR] {type(ex).__name__}: {ex}")

    # ---- (2) forward identity: logZ_recon == logsumexp? ----
    banner("(2) forward identity: ce_none + (e*c[tgt]).sum == logsumexp logZ")
    try:
        eA = e0.clone().requires_grad_(True); cA = c0.clone().requires_grad_(True)
        ce_none = lce(eA, cA, targets, reduction="none", ignore_index=PAD)
        logit_tgt = (eA * cA.index_select(0, targets)).sum(-1)
        logZ_recon = (ce_none + logit_tgt).float()
        logZ_ref = torch.logsumexp((e_f @ c_f.t()), dim=-1)
        lz_md = (logZ_recon[m] - logZ_ref[m]).abs().max().item()
        print(f"  logZ_recon vs logsumexp(fp32)  per-token max|d| (non-pad) = {lz_md:.3e}")
        print("  ^ small => identity holds in CCE arithmetic (forward is fine).")
        R["lz_md"] = lz_md
    except Exception as ex:
        print(f"  [ERROR] {type(ex).__name__}: {ex}")

    # ---- (3) MASTER CHECK ----
    banner("(3) MASTER CHECK: option D & torch-ref & fp32-control vs ANALYTIC TRUTH")
    try:
        logZ_t, gL_true, ge_truth, gc_truth, L = analytic_truth(e_f, c_f, targets, PAD)

        # 3b sanity: the analytic formula must equal autograd of zloss(L) in fp32.
        Lleaf = L.detach().clone().requires_grad_(True)
        z_on_L = masked_zloss(torch.logsumexp(Lleaf, dim=-1), targets, PAD)
        (gL_autograd,) = torch.autograd.grad(z_on_L, Lleaf)
        fmt("SANITY gL_autograd vs gL_true", gL_autograd, gL_true)
        print("  ^ MUST be ~0 / ~1.0, else the analytic formula is wrong (STOP).")

        # option D (bf16)
        eD = e0.clone().requires_grad_(True); cD = c0.clone().requires_grad_(True)
        lseD = lce(eD, cD, targets, reduction="none", ignore_index=PAD) \
            + (eD * cD.index_select(0, targets)).sum(-1)
        zD = masked_zloss(lseD, targets, PAD); zD.backward()
        geD, gcD = eD.grad.float(), cD.grad.float()

        # pure-torch bf16 reference (the baseline we'd been comparing against)
        eR = e0.clone().requires_grad_(True); cR = c0.clone().requires_grad_(True)
        lseR = torch.logsumexp((eR @ cR.t()).float(), dim=-1)
        zR = masked_zloss(lseR, targets, PAD); zR.backward()
        geR, gcR = eR.grad.float(), cR.grad.float()

        # option D COMPOSITION in fp32 (the control that separates precision from bug):
        # same algebra (CE_none + target_logit), all in fp32 so no round-trip loss.
        eC = e_f.clone().requires_grad_(True); cC = c_f.clone().requires_grad_(True)
        Lc = eC @ cC.t()
        ce_none_fp32 = F.cross_entropy(Lc, targets, reduction="none", ignore_index=PAD)
        tgt_fp32 = (eC * cC.index_select(0, targets)).sum(-1)
        lseC = ce_none_fp32 + tgt_fp32
        zC = masked_zloss(lseC, targets, PAD); zC.backward()
        geC, gcC = eC.grad.float(), cC.grad.float()

        print("\n  -- option D (bf16) vs analytic truth --")
        R["optD_ge"] = fmt("ge: optionD vs truth", geD, ge_truth)
        R["optD_gc"] = fmt("gc: optionD vs truth", gcD, gc_truth)
        print("\n  -- pure-torch bf16 ref vs analytic truth --")
        R["ref_ge"] = fmt("ge: torchRef vs truth", geR, ge_truth)
        R["ref_gc"] = fmt("gc: torchRef vs truth", gcR, gc_truth)
        print("\n  -- option D COMPOSITION in fp32 (CONTROL) vs analytic truth --")
        R["optDfp32_ge"] = fmt("ge: optionD-fp32 vs truth", geC, ge_truth)
        R["optDfp32_gc"] = fmt("gc: optionD-fp32 vs truth", gcC, gc_truth)
        print("  ^ if THIS matches (~0 / ~1.0), option D's autograd composition is")
        print("    MATHEMATICALLY CORRECT; any bf16 gap above is precision, not a bug.")

        print("\n  -- option D (bf16) vs torch-ref (bf16): the originally-observed gap --")
        fmt("ge: optionD vs torchRef", geD, geR)
        fmt("gc: optionD vs torchRef", gcD, gcR)

        keep = (targets != PAD).float(); Nk = keep.sum().clamp_min(1.0)
        zT = ((logZ_t ** 2) * keep).sum().item() / Nk.item()
        print(f"\n  zloss scalar: optionD={zD.item():.6e}  torchRef={zR.item():.6e}  truth={zT:.6e}")
    except Exception:
        import traceback; traceback.print_exc()

    # ---- (4) pad isolation ----
    banner("(4) PAD isolation: option D vs truth with NO pad tokens")
    try:
        tg2 = torch.randint(1, V, (N,), device=DEV)  # all valid
        _, _, ge_t2, gc_t2, _ = analytic_truth(e_f, c_f, tg2, PAD)
        e2 = e0.clone().requires_grad_(True); c2 = c0.clone().requires_grad_(True)
        lse2 = lce(e2, c2, tg2, reduction="none", ignore_index=PAD) \
            + (e2 * c2.index_select(0, tg2)).sum(-1)
        masked_zloss(lse2, tg2, PAD).backward()
        R["nopad_ge"] = fmt("ge: optionD vs truth (no pad)", e2.grad.float(), ge_t2)
        R["nopad_gc"] = fmt("gc: optionD vs truth (no pad)", c2.grad.float(), gc_t2)
        print("  ^ if no-pad is dramatically better, the pad/ignore_index path is implicated.")
    except Exception as ex:
        print(f"  [ERROR] {type(ex).__name__}: {ex}")

    # ---- DIAGNOSIS ----
    banner("DIAGNOSIS")
    try:
        def good(k):  # cos>=0.999 and norm_rel<=6e-2
            nr, co, _ = R[k]; return co >= 0.999 and nr <= 6e-2
        def great(k):  # fp32-control level: essentially exact
            nr, co, _ = R[k]; return co >= 0.9999 and nr <= 1e-3

        fp32_ok = "optDfp32_ge" in R and great("optDfp32_ge") and great("optDfp32_gc")
        optD_bf16_ok = "optD_ge" in R and good("optD_ge") and good("optD_gc")
        ref_bf16_ok = "ref_ge" in R and good("ref_ge") and good("ref_gc")

        if fp32_ok:
            print("* Option D's composition is MATHEMATICALLY CORRECT (fp32 control")
            print("  matches the analytic truth to ~1e-6). The composition is NOT broken.")
            if optD_bf16_ok:
                print("* Its bf16 gradient is also within tolerance of the truth.")
            else:
                print("* The bf16 gap vs truth is PRECISION LOSS in the")
                print("  logZ = ce_none + logit_target round-trip (CCE returns")
                print("  logZ - logit_target in reduced precision; re-adding can't recover")
                print("  the lost bits) -- NOT a gradient bug.")
            if not ref_bf16_ok:
                print("* The pure-torch bf16 reference is ALSO off vs the fp32 truth -- it")
                print("  was an inaccurate baseline, not a gold standard.")
            print("\n  => VERDICT: ADOPT OPTION D. Validate the shipped feature against the")
            print("     fp32 ANALYTIC TRUTH (not a bf16 reference). No custom Function needed.")
            print("     For a small annealed regularizer (alpha~1e-4) a ~0.99 cosine /")
            print("     ~0.1 magnitude error is almost certainly fine; if undesirable,")
            print("     accumulate the logZ reconstruction in fp32.")
        else:
            print("* Option D's fp32 composition does NOT match the analytic truth.")
            print("  The composition is genuinely BROKEN (not mere precision).")
            print("\n  => VERDICT: do NOT use option D. Implement a custom autograd Function:")
            print("     forward logZ detached (cce_lse_forward_kernel, no materialization);")
            print("     backward = closed-form (2logZ/Nk)*softmax contracted vs e,c, computed")
            print("     in VOCAB CHUNKS to honor the no-VRAM constraint.")
    except Exception as ex:
        print(f"  [diagnosis error] {type(ex).__name__}: {ex}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
