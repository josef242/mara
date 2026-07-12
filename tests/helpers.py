"""Tiny-model factories shared across the suite.

Everything here is sized to run in seconds on CPU. Keep configs deterministic:
callers seed torch themselves when they need reproducibility.
"""


def tiny_args(**overrides):
    """A minimal CPU-friendly ModelArgs. Overrides are applied as attributes,
    so any ModelArgs field can be set (matches how train_mara configures it)."""
    from model_v2 import ModelArgs
    cfg = ModelArgs()
    cfg.dim = 64
    cfg.n_layers = 2
    cfg.n_heads = 4
    cfg.vocab_size = 256
    cfg.max_seq_len = 64
    cfg.dropout = 0.0
    cfg.use_activation_checkpointing = False
    cfg.pad_id = 0
    cfg.use_keel = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def tiny_model(**overrides):
    import torch
    from model_v2 import Transformer
    torch.manual_seed(overrides.pop("seed", 1234))
    return Transformer(tiny_args(**overrides))
