"""BilateralBlock — one layer in a bilateral stack.

The block exposes three sublayer methods (`apply_self_attn`, `apply_bridge`,
`apply_ffn`) rather than a single forward. This is because at a bridge layer,
both stacks need each other's post-self-attention hidden state — synchronization
that has to live in the stack-level loop, not inside the block.

KEEL post-LN follows the pattern in `model_v2.TransformerBlock`:
  - First block in stack (layer_id == 0): standard Pre-LN, no alpha scaling.
  - Subsequent blocks: `LN_post(alpha * x + sublayer(LN_pre(x)))` per sublayer.

Per Nexus #91 Q2, alpha is per-stack with bridges counted:
  alpha_stack = 2 * L_bilateral + n_bridges (full: 104, mini: 52)

No detach on either side at the bridge (Nexus #89 §3.4 / #91 emphasis) — both
streams' gradients flow through the cross-attention K/V projections.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model_v2 import RMSNorm, FeedForward

from .window_attention import JanusSelfAttention
from .cross_attention import JanusCrossAttention


class BilateralBlock(nn.Module):
    """One layer in either the local or global bilateral stack.

    Args:
        layer_id: within-stack index (0 = first layer; 0 uses plain Pre-LN)
        is_global: True for the global stack (full causal self-attn), False
            for the local stack (windowed causal self-attn)
        is_bridge: True if this layer carries a bridge cross-attention sublayer
        dim: hidden size (d_stack)
        n_heads, n_kv_heads, head_dim: attention head config
        inner_dim: FFN intermediate size
        window_size_local: window for local self-attn (only used when is_global=False)
        max_seq_len: used to size the self-attn mask buffer
        bridge_kv_bandwidth: K/V projection dim for the bridge (default = dim)
        keel_alpha: KEEL post-LN scaling factor (only used when use_keel=True
            and layer_id > 0)
        use_keel: enable KEEL post-LN
        qk_norm_mode: "before_rope" | None
        norm_eps: RMSNorm epsilon
        dropout: attention/residual dropout
    """

    def __init__(
        self,
        layer_id: int,
        is_global: bool,
        is_bridge: bool,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        inner_dim: int,
        window_size_local: int,
        max_seq_len: int,
        bridge_kv_bandwidth: Optional[int],
        keel_alpha: float,
        use_keel: bool,
        qk_norm_mode: Optional[str],
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.is_global = is_global
        self.is_bridge = is_bridge
        self.use_keel = use_keel
        self.use_keel_post = use_keel and layer_id > 0
        self.keel_alpha = keel_alpha

        window = None if is_global else window_size_local
        self.self_attn = JanusSelfAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            window_size=window,
            qk_norm_mode=qk_norm_mode,
            norm_eps=norm_eps,
            dropout=dropout,
        )
        self.attn_norm_pre = RMSNorm(dim, eps=norm_eps)

        if is_bridge:
            self.bridge = JanusCrossAttention(
                q_dim=dim,
                kv_dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                kv_bandwidth=bridge_kv_bandwidth,
                qk_norm_mode=qk_norm_mode,
                norm_eps=norm_eps,
                dropout=dropout,
                out_dim=dim,
            )
            self.bridge_norm_pre_q = RMSNorm(dim, eps=norm_eps)
            self.bridge_norm_pre_kv = RMSNorm(dim, eps=norm_eps)

        self.feed_forward = FeedForward(dim=dim, inner_dim=inner_dim, dropout=dropout)
        self.ffn_norm_pre = RMSNorm(dim, eps=norm_eps)

        if self.use_keel_post:
            self.attn_norm_post = RMSNorm(dim, eps=norm_eps)
            self.ffn_norm_post = RMSNorm(dim, eps=norm_eps)
            if is_bridge:
                self.bridge_norm_post = RMSNorm(dim, eps=norm_eps)

    # ------------------------------------------------------------------
    # Sublayer applications (with KEEL math factored out)
    # ------------------------------------------------------------------
    def apply_self_attn(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        a = self.self_attn(self.attn_norm_pre(x), freqs_cos, freqs_sin)
        if self.use_keel_post:
            return self.attn_norm_post(self.keel_alpha * x + a)
        return x + a

    def apply_bridge(
        self,
        x_attn: torch.Tensor,
        other_x_attn: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_bridge:
            raise RuntimeError("apply_bridge called on a non-bridge block")
        # NO detach on either side — bidirectional gradient flow is intentional.
        b = self.bridge(
            self.bridge_norm_pre_q(x_attn),
            self.bridge_norm_pre_kv(other_x_attn),
        )
        if self.use_keel_post:
            return self.bridge_norm_post(self.keel_alpha * x_attn + b)
        return x_attn + b

    def apply_ffn(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feed_forward(self.ffn_norm_pre(x))
        if self.use_keel_post:
            return self.ffn_norm_post(self.keel_alpha * x + f)
        return x + f


class TrunkBlock(nn.Module):
    """One layer in the synthesis trunk — monolithic full-attention with KEEL.

    Identical structure to a non-bridge bilateral block, but separated as its
    own class for clarity and to keep `BilateralBlock` focused on the bilateral
    phase. Uses the trunk's alpha and RoPE theta.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        inner_dim: int,
        max_seq_len: int,
        keel_alpha: float,
        use_keel: bool,
        qk_norm_mode: Optional[str],
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.use_keel = use_keel
        self.use_keel_post = use_keel and layer_id > 0
        self.keel_alpha = keel_alpha

        self.self_attn = JanusSelfAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            window_size=None,                # full causal
            qk_norm_mode=qk_norm_mode,
            norm_eps=norm_eps,
            dropout=dropout,
        )
        self.attn_norm_pre = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, inner_dim=inner_dim, dropout=dropout)
        self.ffn_norm_pre = RMSNorm(dim, eps=norm_eps)

        if self.use_keel_post:
            self.attn_norm_post = RMSNorm(dim, eps=norm_eps)
            self.ffn_norm_post = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        a = self.self_attn(self.attn_norm_pre(x), freqs_cos, freqs_sin)
        if self.use_keel_post:
            h = self.attn_norm_post(self.keel_alpha * x + a)
        else:
            h = x + a
        f = self.feed_forward(self.ffn_norm_pre(h))
        if self.use_keel_post:
            return self.ffn_norm_post(self.keel_alpha * h + f)
        return h + f
