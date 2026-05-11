"""Janus configuration.

All settled defaults reflect the decisions in Nexus #91 (Rook's reply on the
v1 design clarifying questions). See `janus_design.md` (memory) for the full
decision log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class JanusConfig:
    # -- Bilateral phase -----------------------------------------------------
    L_bilateral: int = 48
    d_stack: int = 1024
    n_heads_stack: int = 16
    n_kv_heads_stack: int = 16              # no GQA (matches mf-low-lr)
    inner_dim_stack: int = 1664             # 1.667x d_stack rounded to multiple_of=128

    # -- Asymmetry -----------------------------------------------------------
    window_size_local: int = 256
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 100_000.0    # was 500K in v1 doc; revised in #91

    # -- Bridges -------------------------------------------------------------
    bridge_frequency: int = 6               # every Nth bilateral layer is a bridge
    bridge_kv_bandwidth: Optional[int] = None  # None = full (= d_stack)

    # -- Merge ---------------------------------------------------------------
    merge_op: str = "cross_attn_fuse"       # cross_attn_fuse | concat_project | gated_mix
    d_synth: int = 1280

    # -- Synthesis trunk -----------------------------------------------------
    L_synthesis: int = 8
    n_heads_trunk: int = 20                 # head_dim 64 at d=1280
    n_kv_heads_trunk: int = 20              # no GQA
    inner_dim_trunk: int = 2176             # 1.667x d_synth rounded to multiple_of=128
    rope_theta_trunk: float = 100_000.0     # revised from 500K per #91

    # -- KEEL ----------------------------------------------------------------
    # Per-stack with bridges counted: alpha_stack = 2*L_bilateral + n_bridges,
    # alpha_trunk = 2*L_synthesis. Auto-computed if None.
    use_keel: bool = True
    keel_alpha_stack: Optional[float] = None
    keel_alpha_trunk: Optional[float] = None

    # -- Norm / dropout ------------------------------------------------------
    norm_eps: float = 1.0e-5
    dropout: float = 0.0
    qk_norm_mode: str = "before_rope"       # applied in self- and cross-attention

    # -- Output heads --------------------------------------------------------
    vocab_size: int = -1                    # filled in by tokenizer at build time
    mtp_horizons: Tuple[int, ...] = (2, 3, 4, 5)
    mtp_loss_weighting: str = "inverse"     # "inverse" -> w_h = 1/h; "uniform"; "geometric"

    # -- Sequence ------------------------------------------------------------
    max_seq_len: int = 2048

    # -- Training infrastructure --------------------------------------------
    use_activation_checkpointing: bool = True
    ckpt_every_bilateral_layer: bool = False  # False = checkpoint only at bridges
    ckpt_bridge_layers: bool = True
    ckpt_merge: bool = True
    ckpt_every_trunk_layer: bool = True

    # -- Output-head LR (batch-adjusted, per Dreadnought v2) ----------------
    output_lr_base_mult: float = 0.8
    output_lr_exponent: float = 0.3

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.d_stack % self.n_heads_stack != 0:
            raise ValueError(
                f"d_stack ({self.d_stack}) must be divisible by n_heads_stack ({self.n_heads_stack})"
            )
        if self.d_synth % self.n_heads_trunk != 0:
            raise ValueError(
                f"d_synth ({self.d_synth}) must be divisible by n_heads_trunk ({self.n_heads_trunk})"
            )
        if self.n_heads_stack % self.n_kv_heads_stack != 0:
            raise ValueError("n_heads_stack must be divisible by n_kv_heads_stack")
        if self.n_heads_trunk % self.n_kv_heads_trunk != 0:
            raise ValueError("n_heads_trunk must be divisible by n_kv_heads_trunk")
        if 1 in self.mtp_horizons:
            raise ValueError(
                "mtp_horizons must not contain 1 — h=1 would duplicate NTP. "
                "See Janus v1 doc §3.7."
            )
        if self.merge_op not in ("cross_attn_fuse", "concat_project", "gated_mix"):
            raise ValueError(f"unknown merge_op: {self.merge_op}")

        # Auto-fill KEEL alphas (per Nexus #91 Q2)
        n_bridges = self.bridge_layer_count
        if self.use_keel and self.keel_alpha_stack is None:
            self.keel_alpha_stack = 2.0 * self.L_bilateral + n_bridges
        if self.use_keel and self.keel_alpha_trunk is None:
            self.keel_alpha_trunk = 2.0 * self.L_synthesis

        # Auto-fill bridge K/V bandwidth (full = d_stack)
        if self.bridge_kv_bandwidth is None:
            self.bridge_kv_bandwidth = self.d_stack

    @property
    def head_dim_stack(self) -> int:
        return self.d_stack // self.n_heads_stack

    @property
    def head_dim_trunk(self) -> int:
        return self.d_synth // self.n_heads_trunk

    @property
    def bridge_layer_indices(self) -> Tuple[int, ...]:
        """Layer indices (0-based) within the bilateral phase that carry a bridge sublayer.

        Doc says "every 6th layer is a bridge layer" with the last bilateral layer
        always being a bridge. With L=48 and k=6, this yields {5, 11, 17, 23, 29, 35, 41, 47}
        in 0-indexed terms (i.e. layers 6, 12, ... 48 in 1-indexed).
        """
        k = self.bridge_frequency
        # 1-indexed: {k, 2k, ..., L}
        return tuple(i - 1 for i in range(k, self.L_bilateral + 1, k))

    @property
    def bridge_layer_count(self) -> int:
        return len(self.bridge_layer_indices)

    def is_bridge_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.bridge_layer_indices

    def mtp_weight(self, h: int) -> float:
        if self.mtp_loss_weighting == "inverse":
            return 1.0 / h
        if self.mtp_loss_weighting == "uniform":
            return 1.0
        if self.mtp_loss_weighting == "geometric":
            return 0.5 ** (h - 1)
        raise ValueError(f"unknown mtp_loss_weighting: {self.mtp_loss_weighting}")


# ----------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------

def janus_mini() -> JanusConfig:
    """Janus-mini pre-flight config (per Nexus #91 Q8).

    Sized at ~450M including heads — large enough that bridge representations
    have room to develop, small enough for fast iteration on a single GPU.
    """
    return JanusConfig(
        L_bilateral=24,
        d_stack=768,
        n_heads_stack=12,
        n_kv_heads_stack=12,
        inner_dim_stack=1280,          # 1.667x 768 ≈ 1280
        window_size_local=256,
        rope_theta_local=10_000.0,
        rope_theta_global=100_000.0,
        bridge_frequency=6,            # bridges at layers 6, 12, 18, 24 (1-indexed)
        merge_op="cross_attn_fuse",
        d_synth=896,
        L_synthesis=4,
        n_heads_trunk=14,
        n_kv_heads_trunk=14,
        inner_dim_trunk=1536,          # 1.667x 896 ≈ 1493 → 1536 (multiple_of=128)
        rope_theta_trunk=100_000.0,
        use_keel=True,
        # KEEL alphas: stack=2*24+4=52, trunk=2*4=8 (auto-computed)
    )


def janus_full() -> JanusConfig:
    """Janus full config (per Nexus #91 updated spec).

    Target ~1.0-1.1B params total. Compares loss-over-tokens against mf-low-lr.
    """
    return JanusConfig(
        L_bilateral=48,
        d_stack=1024,
        n_heads_stack=16,
        n_kv_heads_stack=16,
        inner_dim_stack=1664,
        window_size_local=256,
        rope_theta_local=10_000.0,
        rope_theta_global=100_000.0,
        bridge_frequency=6,            # bridges at layers 6, 12, 18, 24, 30, 36, 42, 48
        merge_op="cross_attn_fuse",
        d_synth=1280,
        L_synthesis=8,
        n_heads_trunk=20,
        n_kv_heads_trunk=20,
        inner_dim_trunk=2176,
        rope_theta_trunk=100_000.0,
        use_keel=True,
        # KEEL alphas: stack=2*48+8=104, trunk=2*8=16 (auto-computed)
    )
