"""Cross-attention for Janus bridges and merge.

No RoPE — both streams have already applied their own positional encoding in
self-attention. Adding RoPE here would compound and is theoretically muddy
(see Janus v1 doc §3.4).

Causal mask: at query position p, attend only to key/value positions ≤ p in
the other stream. Both streams are the same length (same depth, same B/T),
so SDPA's `is_causal=True` does the right thing.

QK norm is applied when configured ("before_rope" mode), even though no RoPE
follows — the norm operation is sensible on its own as a stability primitive
(see Nexus #91 Q4).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import RMSNorm, repeat_kv


class JanusCrossAttention(nn.Module):
    """Causal cross-attention with no RoPE.

    Args:
        q_dim: hidden size of the query stream
        kv_dim: hidden size of the key/value stream (typically == q_dim)
        n_heads: number of query heads
        head_dim: per-head dim (q and kv use the same head_dim)
        kv_bandwidth: total K/V projection dim. Must be a multiple of head_dim.
            Default None = q_dim (full bandwidth, same as Q).
            Lower values reduce bridge K/V capacity (thin callosum ablation).
        qk_norm_mode: "before_rope" | None
        norm_eps: RMSNorm epsilon
        dropout: attention/residual dropout
        out_dim: output projection dim. Default None = q_dim.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        n_heads: int,
        head_dim: int,
        kv_bandwidth: Optional[int] = None,
        qk_norm_mode: Optional[str] = "before_rope",
        norm_eps: float = 1.0e-5,
        dropout: float = 0.0,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        if kv_bandwidth is None:
            kv_bandwidth = q_dim
        if kv_bandwidth % head_dim != 0:
            raise ValueError(
                f"kv_bandwidth ({kv_bandwidth}) must be a multiple of head_dim ({head_dim})"
            )
        self.n_kv_heads = kv_bandwidth // head_dim
        if n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.n_rep = n_heads // self.n_kv_heads
        self.dropout = dropout

        if out_dim is None:
            out_dim = q_dim

        self.wq = nn.Linear(q_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(kv_dim, self.n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(kv_dim, self.n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, out_dim, bias=False)

        self.qk_norm_mode = qk_norm_mode
        if qk_norm_mode == "before_rope":
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, q_stream: torch.Tensor, kv_stream: torch.Tensor) -> torch.Tensor:
        """Compute causal cross-attention.

        Args:
            q_stream: [B, S, q_dim] — query side
            kv_stream: [B, S, kv_dim] — key/value side. Must be the same S.
        Returns:
            [B, S, out_dim]
        """
        if q_stream.shape[1] != kv_stream.shape[1]:
            raise ValueError(
                f"q and kv must have equal seq_len for causal cross-attn; "
                f"got {q_stream.shape[1]} vs {kv_stream.shape[1]}"
            )
        bsz, seqlen, _ = q_stream.shape

        xq = self.wq(q_stream).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(kv_stream).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(kv_stream).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.qk_norm_mode == "before_rope":
            # Despite the name, we apply it here too — it's a stability primitive
            # independent of RoPE (per Nexus #91 Q4).
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.transpose(1, 2)  # [B, H, S, D]
        if self.n_rep > 1:
            k = repeat_kv(xk, self.n_rep).transpose(1, 2)
            v = repeat_kv(xv, self.n_rep).transpose(1, 2)
        else:
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            xq, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # q_pos attends to kv_pos <= q_pos
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out
