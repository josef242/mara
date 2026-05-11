"""Self-attention for Janus stacks.

Single class handles both local (windowed, causal) and global (full, causal)
self-attention. Window is controlled by `window_size` arg: None or 0 = full
causal, positive int = banded causal where position p attends to positions
[max(0, p - window_size + 1), p].

Training path only (no KV cache). Mirrors `model_v2.Attention` structure for
familiarity but is intentionally minimal — no flash-attn branch, no gated
softmax (Janus doesn't use GDN hybrid here), no cache.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# common_fsdp2 is on sys.path via train_mara's setup
from model_v2 import RMSNorm, apply_rotary_emb, repeat_kv


class JanusSelfAttention(nn.Module):
    """Multi-head self-attention with optional sliding-window causal mask.

    Args:
        dim: hidden size
        n_heads: total query heads
        n_kv_heads: KV heads (for GQA; must divide n_heads)
        max_seq_len: used to size the cached attention mask buffer
        window_size: None or 0 = full causal attention. Positive int = banded
            causal: position p attends to [max(0, p-window_size+1), p].
        qk_norm_mode: "before_rope" | None
        norm_eps: RMSNorm epsilon (used when qk_norm_mode == "before_rope")
        dropout: attention/residual dropout
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        window_size: Optional[int] = None,
        qk_norm_mode: Optional[str] = "before_rope",
        norm_eps: float = 1.0e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size if (window_size and window_size > 0) else None
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.qk_norm_mode = qk_norm_mode
        if qk_norm_mode == "before_rope":
            self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-build attention mask buffer. Shape: [max_seq_len, max_seq_len].
        # For window=None: standard causal (lower-triangular).
        # For window=W: banded causal — position p attends to [p-W+1, p].
        # Stored as additive bias: 0 for attended positions, -inf for masked.
        mask = self._build_mask(max_seq_len, self.window_size)
        # Register as non-persistent buffer (will be recreated on resume).
        self.register_buffer("attn_mask", mask, persistent=False)

    @staticmethod
    def _build_mask(seq_len: int, window: Optional[int]) -> torch.Tensor:
        # base causal: position q attends only to k <= q
        q_idx = torch.arange(seq_len).unsqueeze(1)   # [S, 1]
        k_idx = torch.arange(seq_len).unsqueeze(0)   # [1, S]
        allow = (k_idx <= q_idx)                     # causal
        if window is not None:
            allow = allow & (k_idx > (q_idx - window))  # within window of width `window`
        mask = torch.zeros(seq_len, seq_len)
        mask.masked_fill_(~allow, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.qk_norm_mode == "before_rope":
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # [B, H, S, D] layouts for SDPA
        xq = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # GQA: expand if SDPA's enable_gqa is unavailable. Keep it simple — always
        # materialize. This matches the manual fallback in model_v2.Attention.
        if self.n_rep > 1:
            k = repeat_kv(xk, self.n_rep).transpose(1, 2)
            v = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # Slice the prebuilt mask to current seqlen. Shape [S, S] broadcasts over
        # [B, H, S, S]. Cast to query dtype so SDPA stays in the activation dtype.
        mask = self.attn_mask[:seqlen, :seqlen].to(dtype=xq.dtype)

        out = F.scaled_dot_product_attention(
            xq, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # causality is in the mask
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out
