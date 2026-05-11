"""JanusModel — bilateral asymmetric transformer with synthesis trunk.

Top-level model that composes:
  1. Token embeddings
  2. Bilateral phase: two parallel stacks (local windowed, global full) with
     periodic cross-attention bridges and bidirectional gradient flow
  3. Merge: fuse local + global at d_synth
  4. Synthesis trunk: monolithic full-attention transformer at d_synth
  5. Output heads: NTP + MTP{2,3,4,5} with weighted loss

See `janus_design.md` (memory) for design refs (Nexus #89 design, #91 decisions).

Forward returns logits when targets is None; returns (None, loss_dict) when
targets is provided, where loss_dict carries the total loss plus per-head
breakdown (loss_ntp, loss_mtp_h2, ...) for diagnostics.

Training-only path. KV caching for inference is not implemented in v1.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from model_v2 import RMSNorm, precompute_freqs_cis, cce_loss

from .config import JanusConfig
from .layers.bilateral_block import BilateralBlock, TrunkBlock
from .layers.merge import build_merge


class JanusModel(nn.Module):
    """Bilateral asymmetric transformer with synthesis trunk + multi-horizon heads."""

    def __init__(self, config: JanusConfig):
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError(
                f"JanusConfig.vocab_size must be set (got {config.vocab_size}). "
                "Fill from tokenizer at build time."
            )
        self.config = config

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_stack)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ------------------------------------------------------------------
        # Bilateral phase: two parallel stacks, layer-aligned with bridges
        # ------------------------------------------------------------------
        bridge_indices = set(config.bridge_layer_indices)
        self.local_blocks = nn.ModuleList()
        self.global_blocks = nn.ModuleList()
        for layer_id in range(config.L_bilateral):
            is_bridge = layer_id in bridge_indices
            self.local_blocks.append(self._build_bilateral_block(
                layer_id=layer_id, is_global=False, is_bridge=is_bridge,
            ))
            self.global_blocks.append(self._build_bilateral_block(
                layer_id=layer_id, is_global=True, is_bridge=is_bridge,
            ))

        # ------------------------------------------------------------------
        # Merge
        # ------------------------------------------------------------------
        self.merge = build_merge(
            merge_op=config.merge_op,
            d_stack=config.d_stack,
            d_synth=config.d_synth,
            n_heads=config.n_heads_trunk,
            head_dim=config.head_dim_trunk,
            qk_norm_mode=config.qk_norm_mode,
            norm_eps=config.norm_eps,
            dropout=config.dropout,
        )

        # ------------------------------------------------------------------
        # Synthesis trunk
        # ------------------------------------------------------------------
        self.trunk_blocks = nn.ModuleList()
        for layer_id in range(config.L_synthesis):
            self.trunk_blocks.append(TrunkBlock(
                layer_id=layer_id,
                dim=config.d_synth,
                n_heads=config.n_heads_trunk,
                n_kv_heads=config.n_kv_heads_trunk,
                inner_dim=config.inner_dim_trunk,
                max_seq_len=config.max_seq_len,
                keel_alpha=config.keel_alpha_trunk,
                use_keel=config.use_keel,
                qk_norm_mode=config.qk_norm_mode,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
            ))
        self.trunk_norm = RMSNorm(config.d_synth, eps=config.norm_eps)

        # ------------------------------------------------------------------
        # Output heads (NTP + MTP). Each head is an independent linear.
        # NTP predicts token at p+1; MTP[h] predicts token at p+h, h in {2..5}.
        # ------------------------------------------------------------------
        self.ntp_head = nn.Linear(config.d_synth, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(config.d_synth, config.vocab_size, bias=False)
            for h in config.mtp_horizons
        })
        self.mtp_horizons = tuple(config.mtp_horizons)
        self.head_norm = RMSNorm(config.d_synth, eps=config.norm_eps)

        # ------------------------------------------------------------------
        # RoPE frequency tables
        #   - Local stack uses theta_local
        #   - Global stack and trunk share theta_global (same theta per #91)
        # Both share head_dim=64 in the default config.
        # ------------------------------------------------------------------
        freqs_cos_local, freqs_sin_local = precompute_freqs_cis(
            dim=config.head_dim_stack,
            end=config.max_seq_len,
            theta=config.rope_theta_local,
        )
        self.register_buffer("freqs_cos_local", freqs_cos_local, persistent=False)
        self.register_buffer("freqs_sin_local", freqs_sin_local, persistent=False)

        freqs_cos_global, freqs_sin_global = precompute_freqs_cis(
            dim=config.head_dim_stack,
            end=config.max_seq_len,
            theta=config.rope_theta_global,
        )
        self.register_buffer("freqs_cos_global", freqs_cos_global, persistent=False)
        self.register_buffer("freqs_sin_global", freqs_sin_global, persistent=False)

        # Trunk RoPE: always register as its own buffer. (Aliasing as a plain
        # Python attribute would break on .to('cuda') — only registered buffers
        # get moved.) Memory cost is negligible: max_seq_len * head_dim/2 floats.
        freqs_cos_trunk, freqs_sin_trunk = precompute_freqs_cis(
            dim=config.head_dim_trunk,
            end=config.max_seq_len,
            theta=config.rope_theta_trunk,
        )
        self.register_buffer("freqs_cos_trunk", freqs_cos_trunk, persistent=False)
        self.register_buffer("freqs_sin_trunk", freqs_sin_trunk, persistent=False)

        # ------------------------------------------------------------------
        # Init weights
        # ------------------------------------------------------------------
        self.apply(self._init_weights)
        # Output-projection-style scaled init (w3, wo, output heads). Use the
        # combined depth as the denominator since residual variance accumulates
        # across both stacks and the trunk.
        total_depth = config.L_bilateral + config.L_synthesis
        output_std = 0.02 / math.sqrt(2 * total_depth)
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=output_std)
            elif "ntp_head.weight" == pn or pn.startswith("mtp_heads."):
                nn.init.normal_(p, mean=0.0, std=output_std)

    # ----------------------------------------------------------------------
    # Bridge-control hook (for diagnostics / split-brain probes)
    # ----------------------------------------------------------------------
    # When non-None, replaces the output of every bridge cross-attention with
    # zero. Used by the "zero bridges" causality test and by split-brain
    # diagnostics. The model still produces output and gradients still flow.
    # ----------------------------------------------------------------------
    def _build_bilateral_block(self, layer_id: int, is_global: bool, is_bridge: bool) -> BilateralBlock:
        c = self.config
        return BilateralBlock(
            layer_id=layer_id,
            is_global=is_global,
            is_bridge=is_bridge,
            dim=c.d_stack,
            n_heads=c.n_heads_stack,
            n_kv_heads=c.n_kv_heads_stack,
            head_dim=c.head_dim_stack,
            inner_dim=c.inner_dim_stack,
            window_size_local=c.window_size_local,
            max_seq_len=c.max_seq_len,
            bridge_kv_bandwidth=c.bridge_kv_bandwidth,
            keel_alpha=c.keel_alpha_stack,
            use_keel=c.use_keel,
            qk_norm_mode=c.qk_norm_mode,
            norm_eps=c.norm_eps,
            dropout=c.dropout,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------------------
    # One bilateral layer's forward (with cross-stream synchronization at the
    # bridge). Separated out so we can wrap it with activation checkpointing.
    # ----------------------------------------------------------------------
    def _forward_one_bilateral_layer(
        self,
        local_block: BilateralBlock,
        global_block: BilateralBlock,
        local_h: torch.Tensor,
        global_h: torch.Tensor,
        zero_bridge: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Both stacks compute self-attn first, simultaneously.
        local_attn = local_block.apply_self_attn(
            local_h, self.freqs_cos_local, self.freqs_sin_local,
        )
        global_attn = global_block.apply_self_attn(
            global_h, self.freqs_cos_global, self.freqs_sin_global,
        )

        if local_block.is_bridge and not zero_bridge:
            # NO detach — bidirectional gradient flow is the whole point.
            local_after_bridge = local_block.apply_bridge(local_attn, global_attn)
            global_after_bridge = global_block.apply_bridge(global_attn, local_attn)
        else:
            local_after_bridge = local_attn
            global_after_bridge = global_attn

        local_h = local_block.apply_ffn(local_after_bridge)
        global_h = global_block.apply_ffn(global_after_bridge)
        return local_h, global_h

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        zero_local_at_merge: bool = False,
        zero_global_at_merge: bool = False,
        zero_bridges: bool = False,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            tokens: [B, S] input token ids
            targets: [B, S] targets for NTP+MTP loss. If None, return logits.
            zero_local_at_merge: split-brain probe — replace h_local at merge with zeros
            zero_global_at_merge: split-brain probe — replace h_global at merge with zeros
            zero_bridges: split-brain probe — zero all bridge cross-attn outputs

        Returns:
            (logits_dict, None) if targets is None — logits keyed "ntp" and "h{h}"
            (None, loss_dict)   if targets is not None — keys: "loss" (total),
                                "loss_ntp", "loss_mtp_h{h}" for each h in mtp_horizons
        """
        cfg = self.config
        bsz, seqlen = tokens.shape
        if seqlen > cfg.max_seq_len:
            raise ValueError(
                f"input seqlen {seqlen} exceeds max_seq_len {cfg.max_seq_len}"
            )

        # ----- Embedding shared by both stacks -----
        h_embed = self.embed_dropout(self.tok_embeddings(tokens))
        local_h = h_embed
        global_h = h_embed

        # ----- Bilateral phase -----
        for layer_id in range(cfg.L_bilateral):
            local_block = self.local_blocks[layer_id]
            global_block = self.global_blocks[layer_id]
            do_ckpt = (
                cfg.use_activation_checkpointing
                and self.training
                and (
                    cfg.ckpt_every_bilateral_layer
                    or (local_block.is_bridge and cfg.ckpt_bridge_layers)
                )
            )
            if do_ckpt:
                local_h, global_h = cp.checkpoint(
                    self._forward_one_bilateral_layer,
                    local_block, global_block, local_h, global_h, zero_bridges,
                    use_reentrant=False,
                )
            else:
                local_h, global_h = self._forward_one_bilateral_layer(
                    local_block, global_block, local_h, global_h, zero_bridges,
                )

        # ----- Split-brain probes (at merge boundary) -----
        if zero_local_at_merge:
            local_h = torch.zeros_like(local_h)
        if zero_global_at_merge:
            global_h = torch.zeros_like(global_h)

        # ----- Merge -----
        do_merge_ckpt = (
            cfg.use_activation_checkpointing and self.training and cfg.ckpt_merge
        )
        if do_merge_ckpt:
            h = cp.checkpoint(self.merge, local_h, global_h, use_reentrant=False)
        else:
            h = self.merge(local_h, global_h)

        # ----- Trunk -----
        freqs_cos_t = self.freqs_cos_trunk
        freqs_sin_t = self.freqs_sin_trunk
        for layer_id, block in enumerate(self.trunk_blocks):
            do_ckpt = (
                cfg.use_activation_checkpointing
                and self.training
                and cfg.ckpt_every_trunk_layer
            )
            if do_ckpt:
                h = cp.checkpoint(block, h, freqs_cos_t, freqs_sin_t, use_reentrant=False)
            else:
                h = block(h, freqs_cos_t, freqs_sin_t)

        h = self.head_norm(h)  # final pre-head norm

        # ----- Outputs / loss -----
        if targets is None:
            logits = {"ntp": self.ntp_head(h)}
            for hor in self.mtp_horizons:
                logits[f"h{hor}"] = self.mtp_heads[f"h{hor}"](h)
            return logits, None

        loss_dict = self._compute_loss(h, targets)
        return None, loss_dict

    def _compute_loss(self, h: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute NTP + MTP weighted loss with edge masking.

        For NTP, target at position p is tokens[p+1]; for MTP[h], target at p is
        tokens[p+h]. Positions where p+h >= seqlen have no target and are
        excluded.

        Implemented with CCE (cce_loss from model_v2) per head to avoid
        materializing full [B, S, V] logits tensors.
        """
        cfg = self.config
        bsz, seqlen, _ = h.shape

        # NTP: position p predicts targets[p+1]. Shift hidden states / targets.
        ntp_h = h[:, :-1].contiguous().view(-1, h.shape[-1])
        ntp_t = targets[:, 1:].contiguous().view(-1)
        loss_ntp = cce_loss(
            ntp_h, self.ntp_head.weight, ntp_t,
            reduction="mean",
        )

        total = loss_ntp
        per_head = {"loss_ntp": loss_ntp.detach()}

        for hor in self.mtp_horizons:
            # Position p predicts targets[p+hor]. Valid range: p in [0, seqlen-hor).
            valid = seqlen - hor
            if valid <= 0:
                # Sequence too short for this horizon — skip but log a zero loss.
                zero = torch.zeros((), device=h.device, dtype=h.dtype)
                per_head[f"loss_mtp_h{hor}"] = zero
                continue
            mtp_h = h[:, :valid].contiguous().view(-1, h.shape[-1])
            mtp_t = targets[:, hor:hor + valid].contiguous().view(-1)
            loss_h = cce_loss(
                mtp_h, self.mtp_heads[f"h{hor}"].weight, mtp_t,
                reduction="mean",
            )
            per_head[f"loss_mtp_h{hor}"] = loss_h.detach()
            total = total + cfg.mtp_weight(hor) * loss_h

        per_head["loss"] = total
        return per_head
