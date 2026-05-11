"""Merge operation — fuses local + global stacks into the trunk input.

Three variants, selected by `merge_op` in JanusConfig:

- cross_attn_fuse (default): local as query, global as K/V. Causal cross-attn
  projects both streams into d_synth, then adds a global-side residual and
  RMSNorms. See Janus v1 doc §3.5.

- concat_project: simple [local; global] → Linear(2*d_stack -> d_synth) → RMSNorm.

- gated_mix: per-token learned gate σ(g) mixes the two streams after projecting
  each independently into d_synth.

Asymmetry in cross_attn_fuse: local is the query (drives token-aligned position
flow into the trunk), global is the key/value (contributes context). The trunk
needs token-aligned hidden states, so query=local is the principled choice.

Causal mask: at position p in the query (local) stream, attend only to global
positions ≤ p. SDPA's `is_causal=True` handles this since both streams have
the same sequence length.

No RoPE — both streams have already applied positional encoding in their own
self-attention (see Janus v1 doc §3.4 / §5).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v2 import RMSNorm

from .cross_attention import JanusCrossAttention


class CrossAttnFuseMerge(nn.Module):
    """Default merge: cross-attention with local=Q, global=K/V, plus global residual.

    All projections produce d_synth-dim outputs. Heads are configured to match
    the trunk's head structure (n_heads_trunk, head_dim_trunk) for natural
    alignment with the trunk that consumes our output.
    """

    def __init__(
        self,
        d_stack: int,
        d_synth: int,
        n_heads: int,
        head_dim: int,
        qk_norm_mode: Optional[str],
        norm_eps: float,
        dropout: float,
    ):
        super().__init__()
        if n_heads * head_dim != d_synth:
            raise ValueError(
                f"n_heads*head_dim ({n_heads*head_dim}) must equal d_synth ({d_synth})"
            )
        self.cross_attn = JanusCrossAttention(
            q_dim=d_stack,
            kv_dim=d_stack,
            n_heads=n_heads,
            head_dim=head_dim,
            kv_bandwidth=d_synth,           # full bandwidth at trunk dim
            qk_norm_mode=qk_norm_mode,
            norm_eps=norm_eps,
            dropout=dropout,
            out_dim=d_synth,
        )
        # Global-side residual projection (global → d_synth) added after cross-attn.
        self.global_residual = nn.Linear(d_stack, d_synth, bias=False)
        self.out_norm = RMSNorm(d_synth, eps=norm_eps)

    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        merged = self.cross_attn(h_local, h_global)
        merged = merged + self.global_residual(h_global)
        return self.out_norm(merged)


class ConcatProjectMerge(nn.Module):
    """Concat + Linear projection. Simple alternative to cross_attn_fuse."""

    def __init__(self, d_stack: int, d_synth: int, norm_eps: float):
        super().__init__()
        self.proj = nn.Linear(2 * d_stack, d_synth, bias=False)
        self.out_norm = RMSNorm(d_synth, eps=norm_eps)

    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        merged = self.proj(torch.cat([h_local, h_global], dim=-1))
        return self.out_norm(merged)


class GatedMixMerge(nn.Module):
    """Per-token sigmoid gate mixes independently-projected streams."""

    def __init__(self, d_stack: int, d_synth: int, norm_eps: float):
        super().__init__()
        self.proj_local = nn.Linear(d_stack, d_synth, bias=False)
        self.proj_global = nn.Linear(d_stack, d_synth, bias=False)
        self.gate = nn.Linear(2 * d_stack, 1, bias=True)
        self.out_norm = RMSNorm(d_synth, eps=norm_eps)

    def forward(self, h_local: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_local, h_global], dim=-1)))  # [B, S, 1]
        merged = g * self.proj_local(h_local) + (1.0 - g) * self.proj_global(h_global)
        return self.out_norm(merged)


def build_merge(
    merge_op: str,
    d_stack: int,
    d_synth: int,
    n_heads: int,
    head_dim: int,
    qk_norm_mode: Optional[str],
    norm_eps: float,
    dropout: float,
) -> nn.Module:
    if merge_op == "cross_attn_fuse":
        return CrossAttnFuseMerge(
            d_stack=d_stack, d_synth=d_synth,
            n_heads=n_heads, head_dim=head_dim,
            qk_norm_mode=qk_norm_mode, norm_eps=norm_eps, dropout=dropout,
        )
    if merge_op == "concat_project":
        return ConcatProjectMerge(d_stack=d_stack, d_synth=d_synth, norm_eps=norm_eps)
    if merge_op == "gated_mix":
        return GatedMixMerge(d_stack=d_stack, d_synth=d_synth, norm_eps=norm_eps)
    raise ValueError(f"unknown merge_op: {merge_op}")
