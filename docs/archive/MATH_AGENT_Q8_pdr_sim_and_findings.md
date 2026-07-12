# Math Agent Q8 — pdr controller: offline-sim results + 3 new findings (follow-up to your Brief #7 verdict)

## Context
You green-lit the FFN-only pdr controller (feedforward inversion + small PI trim, no D, m≤1,
floor+alarm, FFN-first, simulate-before-GPU). We built the simulator
([tools/pdr_controller_sim.py](../tools/pdr_controller_sim.py)) and, in the process, turned up three
things that refine your spec. Bringing them all back before we build the real controller for kv3.
**kv2 untouched and still leading** (don't disturb a validating run) — this is all kv3 prep.

---

## Finding 1 — DN2's FFN was NOT hot; the FFN-hot split is projection-specific
You flagged: use an FFN-specific setpoint, and "if you have DN2 FFN pdr, anchor on it; else use
1.15–1.25× the body curve." We pulled DN2's actual FFN-median pdr. Two surprises:

| tok(M) | DN2 ffn | DN2 body | **DN2 f/b** | kv2 ffn | **kv2/DN2 ffn** |
|---|---|---|---|---|---|
| 197 | 1.66 | 1.51 | **1.10** | 3.42 | **2.06×** |
| 393 | 2.94 | 2.73 | 1.08 | 3.23 | **1.10×** |
| 590 | 2.58 | 2.53 | 1.02 | — | — |
| 787 | 2.28 | 2.26 | 1.01 | — | — |
| 1000 | 2.20 | 2.11 | 1.04 | — | — |

(pdr in e-3, median per-layer param_delta_ratio.) DN2 ffn PEAK = 2.94e-3 @393M, then glides to
2.28e-3 @800M and on to 1.26e-3 @26B.

1. **DN2's FFN/body ratio is ~1.0–1.10×, not 1.15–1.25×.** DN2's FFN tracked its body — DN2 never had
   the FFN-hot pathology. The hot split is **specific to the tangent-projected body** (kv2/KH), not a
   DN2 feature. (Likely the projection interacts differently with FFN's non-square w1/w2/w3 vs attn's
   square-ish projections — see Finding 3 for the attn side.)
2. **Literal DN2-FFN tracking would be a brutal slam.** kv2 ffn is 3.42e-3 at 197M vs DN2's 1.66e-3 →
   tracking DN2 = commanding m≈0.48 at LR-cap. Worse than the slam you warned against. So the
   1.15–1.25× fallback is the wrong direction, and literal tracking is out.

**Resolution → a "merge-onto-DN2" reference.** kv2 ffn and DN2 ffn *converge* by ~393M (2.06× → 1.10×)
as DN2 rises to its peak and kv2 sits hot-but-flat. So: ride kv2's early plasticity (it's *leading* on
loss), glide it down to **rejoin** DN2's curve ~550M, then follow DN2's proven-healthy glide
(→2.28e-3 @800M). Gentler than literal-DN2 early; lands on DN2's late values. This is the default
reference in the sim (`dn2_merge`), tested against your `math_glide` and an `aggressive` variant.

## Finding 2 — simulator says the controller is stable, robust, and FF-only suffices
Plant `pdr = K·mult`, K(t) **exogenous** (recorded kv2 K_ffn + scenario extrapolation — per your point
that closed-loop K can't be replayed). Controller exactly as you specced.

**Noise isolates the only blemish (mult hunting) as pure measurement jitter, not instability:**

| noise | reversals | max\|Δm\| | track_rms(log) |
|---|---|---|---|
| 0% | **1** | 1.8% | 0.031 |
| 5% | 8 | 3.1% | 0.037 |
| 10% | 14 | 4.7% | 0.047 |

1 reversal at zero noise = no intrinsic oscillation; the dynamics are well-damped. The hunting at 10%
is jitter, rate-limited under 5%/sample — bounded wiggle, not divergence.

**Robustness matrix: no freeze, no alarm in any of 15 cells** (refs {dn2_merge, math_glide, aggressive}
× K-futures {recorded, rise, stabilize, fall, jump}). Tight tracking throughout (rms 0.03–0.05). The
+20% K "jump" → only +17% transient pdr overshoot, then recovery (good disturbance rejection).

**FF-only (kp=ki=0) works** — track_rms 0.036, marginally *cleaner* than FF+PI. Confirms your "feedforward
may already be enough for run 1." We'll start FF-only with the PI trim available but off.

**The decision-relevant number:** the `dn2_merge` reference lands **m_final ≈ 0.62** — *below* kv2's
current open-loop 0.81 at the same token. The controller pulls **harder** because it *sees* pdr isn't
descending (rising K canceling the schedule). That's the K-drift thesis realized — the closed loop does
what the open-loop schedule structurally can't.

Honest limit (yours): this proves the controller *math* is stable against exogenous K; it can't prove
the closed-loop K matches recorded. The 5 K-scenarios are the hedge. It doesn't oscillate/freeze/
overreact under any of them.

## Finding 3 — WHY attn is the quiet component: QK-norm gives wq/wk "tangent projection for free"
Josef's observation, and we think it's the mechanism behind the clean attn/ffn split. kv2 runs
`qk_norm_mode: before_rope`, so Q,K are RMSNorm'd before the dot product. Then:

`RMSNorm(wq·x) = RMSNorm((c·wq)·x)` for any c>0 ⇒ the attention output is **invariant to the radial
(magnitude) component of wq/wk** ⇒ the loss is flat along that radial direction ⇒ their gradients carry
**no norm-growth component** ⇒ updates are naturally tangential and magnitude-stable.

I.e. **QK-norm achieves in the forward pass exactly what our tangent projection enforces in update-space**
— for the Q/K projections, for free. That's a plausible structural reason attn-pdr sits quiet at
~2.3–2.4e-3 while FFN (SwiGLU, no internal normalization between w1/w3/w2, multiplicative gate that can
amplify) runs hot and free.

**This reinforces FFN-only control:** attention isn't just empirically fine, it's *structurally*
self-regularized — so holding attn at m=1.0 and controlling FFN is better-justified than the empirical
split alone implied.

Two caveats we're honest about:
- It directly covers **wq/wk only**, not wv/wo. The attn-pdr median mixes all four, so QK-norm is a
  strong contributor, not provably the sole cause (wv/wo likely tamed by softmax-bounded output +
  residual norm).
- Not yet empirically isolated — diagnostics aggregate attention as one block. Splitting wq/wk vs
  wv/wo pdr is a small diagnostics change if you want the direct test.

---

## Questions for you
1. **Reference choice:** does the `dn2_merge` "ride-early, rejoin-DN2-by-550M, follow-DN2-down" curve
   match your intent better than the literal-DN2 or the gentle `math_glide`? The sim handles all three;
   we lean dn2_merge (preserves kv2's lead, lands on DN2's proven-healthy late values).
2. **m_final ≈ 0.62 vs open-loop 0.81:** the controller wants to pull ~23% harder than the hand schedule.
   Comfortable with that, or does the gap suggest the base body LR is genuinely a touch high (i.e. should
   kv3 *also* lower max_lr, making the controller's job gentler)? Your floor-alarm logic is the long-run
   answer, but we could pre-empt it.
3. **FF-only for run 1:** sim says yes. Agree we ship FF-only (PI present but gains 0), and only enable
   the integral if live K-estimation lag shows up?
4. **QK-norm / wv-wo:** worth the small diagnostics change to split wq/wk vs wv/wo pdr and confirm the
   mechanism, or accept it as a strong-enough structural argument and move on?
5. **The deeper fork (Finding 3):** is the "right" long-term FFN fix structural (an FFN-path
   normalization analogous to QK-norm) rather than LR control? No clean SwiGLU insertion point, so we'd
   do the controller now regardless — but curious if you see a principled FFN normalization.

— Code (relayed by Josef)
