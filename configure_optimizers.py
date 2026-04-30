# configure_optimizers.py
"""
Optimizer configuration for FSDP2 training.

Supports:
- AdamW (standard fused, or 8-bit via torchao)
- AdamC (corrected weight decay for normalized layers, or 8-bit via torchao)
- Adafactor (FSDP2-compatible, memory efficient with stochastic rounding)
- Muon/NorMuon/MuonSphere via FSDP2-native implementation
- Muon/NorMuon/Dion2 via Microsoft DION package
"""

import inspect
import torch

# ---------------------------------------------------------------------------
# Optimizer type taxonomy
# ---------------------------------------------------------------------------
VALID_OPTIMIZER_TYPES = {
    # Adam family
    "adamw", "adamw_8bit", "adamw_16bit", "adamc", "adamc_8bit", "adamc_16bit",
    # Adafactor
    "adafactor",
    # Microsoft DION package
    "muon_dion", "normuon_dion", "dion2",
    # FSDP2-native Muon (samsja's implementation)
    "muon_fsdp2", "normuon_fsdp2", "muonsphere_fsdp2", "normuon_sphere_fsdp2",
}

MUON_FAMILY = {
    "muon_dion", "normuon_dion", "dion2",
    "muon_fsdp2", "normuon_fsdp2", "muonsphere_fsdp2", "normuon_sphere_fsdp2",
}
DION_FAMILY = {"muon_dion", "normuon_dion", "dion2"}
FSDP2_MUON_FAMILY = {"muon_fsdp2", "normuon_fsdp2", "muonsphere_fsdp2", "normuon_sphere_fsdp2"}
NORMUON_VARIANTS = {"normuon_dion", "normuon_fsdp2", "normuon_sphere_fsdp2"}
MUONSPHERE_VARIANTS = {"muonsphere_fsdp2", "normuon_sphere_fsdp2"}


def summarize_optimizer_settings(settings, ddp_world_size, grad_accum, logger, model=None):
    """
    Print a summary of optimizer settings to the logger.

    Args:
        settings: Training settings object with optimizer_type and related config
        ddp_world_size: Number of distributed processes
        grad_accum: Gradient accumulation steps (for calculating tokens per step)
        logger: Logger with print_and_log method
        model: Optional model to count param groups for Muon
    """
    def log(msg):
        logger.print_and_log(msg)

    optimizer_type = settings.optimizer_type
    learning_rate = settings.max_lr
    betas = (getattr(settings, 'beta1', 0.9), getattr(settings, 'beta2', 0.95))

    # Count Muon vs Adam params if model provided
    muon_param_count = 0
    adam_param_count = 0
    if model is not None and optimizer_type in MUON_FAMILY:
        for n, p in model.named_parameters():
            if '.weight' in n and any(lt in n for lt in ['wq.', 'wk.', 'wv.', 'wo.', 'w1.', 'w2.', 'w3.',
                                                                'q_proj.', 'k_proj.', 'v_proj.', 'o_proj.', 'g_proj.']):
                muon_param_count += 1
            else:
                adam_param_count += 1

    # -------------------------------------------------------------------------
    # DION variants (Microsoft package)
    # -------------------------------------------------------------------------
    if optimizer_type in DION_FAMILY:
        opt_name = {"muon_dion": "Muon", "normuon_dion": "NorMuon", "dion2": "Dion2"}[optimizer_type]
        log(f"  ] Optimizer = Microsoft DION {opt_name}")
        if model:
            log(f"  ] {opt_name} params = {muon_param_count}, AdamW params = {adam_param_count}")
        log(f"  ] {opt_name} LR = {learning_rate} (auto-scaled by DION)")
        if optimizer_type in ("muon_dion", "normuon_dion"):
            muon_momentum = getattr(settings, 'muon_momentum', 0.95)
            log(f"  ] {opt_name} Momentum (mu) = {muon_momentum}")
        log(f"  ] AdamW LR (embeddings/norms) = {learning_rate}")
        dion_kwargs = getattr(settings, 'dion_kwargs', None)
        if dion_kwargs:
            log(f"  ] Extra kwargs = {dion_kwargs}")

    # -------------------------------------------------------------------------
    # FSDP2-native Muon variants
    # -------------------------------------------------------------------------
    elif optimizer_type in FSDP2_MUON_FAMILY:
        opt_names = {
            "muon_fsdp2": "Muon-FSDP2",
            "normuon_fsdp2": "NorMuon-FSDP2",
            "muonsphere_fsdp2": "MuonSphere-FSDP2",
            "normuon_sphere_fsdp2": "NorMuon-Sphere-FSDP2",
        }
        opt_name = opt_names[optimizer_type]
        muon_momentum = getattr(settings, 'muon_momentum', 0.95)
        muon_ns_steps = getattr(settings, 'muon_ns_steps', 5)

        log(f"  ] Optimizer = {opt_name}")
        if model:
            log(f"  ] Muon params = {muon_param_count}, AdamW params = {adam_param_count}")
        log(f"  ] Muon LR = {learning_rate}")
        log(f"  ] Muon Momentum = {muon_momentum}")
        log(f"  ] NS steps = {muon_ns_steps}")
        muon_adam_sd = getattr(settings, 'muon_adam_state_dtype', 'fp32')
        log(f"  ] AdamW LR (embeddings/norms) = {learning_rate}")
        if muon_adam_sd != "fp32":
            log(f"  ] Adam state dtype = {muon_adam_sd} (16-bit with stochastic rounding)")
        else:
            log(f"  ] Adam state dtype = fp32")

        if optimizer_type in NORMUON_VARIANTS:
            normuon_beta2 = getattr(settings, 'normuon_beta2', 0.95)
            log(f"  ] NorMuon neuron-wise normalization enabled")
            log(f"  ] NorMuon beta2 = {normuon_beta2}")

        if optimizer_type in MUONSPHERE_VARIANTS:
            radius_scale = getattr(settings, 'muonsphere_radius_scale', 2.0)
            power_iters = getattr(settings, 'muonsphere_power_iters', 10)
            log(f"  ] MuonSphere spectral retraction ENABLED (arXiv 2601.08393)")
            log(f"  ] Radius scale (c) = {radius_scale}")
            log(f"  ] Power iterations = {power_iters}")
            log(f"  ] Weight decay = DISABLED (spectral retraction regularizes)")

    # -------------------------------------------------------------------------
    # Adafactor
    # -------------------------------------------------------------------------
    elif optimizer_type == "adafactor":
        adafactor_beta2 = getattr(settings, 'adafactor_beta2', None)
        log(f"  ] Optimizer = AdafactorFSDP2 (memory efficient)")
        log(f"  ] LR = {learning_rate}")
        if adafactor_beta2 is None:
            log(f"  ] Beta2 = auto (1 - step^-0.8)")
        else:
            log(f"  ] Beta2 = {adafactor_beta2}")
        tokens_per_step = settings.B * settings.T * ddp_world_size * grad_accum
        half_life_tokens = 10_000_000
        recommended_beta2 = 0.5 ** (1.0 / (half_life_tokens / tokens_per_step))
        log(f"  ] Recommended beta2 = {recommended_beta2:.6f} (10M token half-life)")

    # -------------------------------------------------------------------------
    # AdamC
    # -------------------------------------------------------------------------
    elif optimizer_type in ("adamc", "adamc_8bit", "adamc_16bit"):
        variant = {
            "adamc": "AdamC",
            "adamc_8bit": "AdamC8bit (torchao)",
            "adamc_16bit": "AdamC16bit",
        }[optimizer_type]
        if optimizer_type == "adamc_16bit":
            sd = getattr(settings, 'adam16bit_state_dtype', 'mixed')
            log(f"  ] Optimizer = {variant} (state_dtype={sd})")
        else:
            log(f"  ] Optimizer = {variant}")
        log(f"  ] LR = {learning_rate}")
        log(f"  ] Beta1 = {betas[0]:.2f}")
        log(f"  ] Beta2 = {betas[1]:.2f}")

    # -------------------------------------------------------------------------
    # Standard AdamW
    # -------------------------------------------------------------------------
    elif optimizer_type in ("adamw", "adamw_8bit", "adamw_16bit"):
        if optimizer_type == "adamw_16bit":
            sd = getattr(settings, 'adam16bit_state_dtype', 'mixed')
            log(f"  ] Optimizer = AdamW16bit (state_dtype={sd}, FSDP2-compatible)")
        elif optimizer_type == "adamw_8bit":
            log(f"  ] Optimizer = AdamW8bit (torchao, FSDP2-compatible)")
        else:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            fused = fused_available and torch.cuda.is_available()
            log(f"  ] Optimizer = AdamW (fused={fused})")
        log(f"  ] LR = {learning_rate}")
        log(f"  ] Beta1 = {betas[0]:.2f}")
        log(f"  ] Beta2 = {betas[1]:.2f}")

    # Common settings — weight decay display
    wd = settings.weight_decay
    def _wd_fmt(v):
        if isinstance(v, list):
            return " \u2192 ".join(f"{val}@{s}" for s, val in v)
        return f"{v:.4f}"
    if isinstance(wd, list):
        log(f"  ] Weight Decay = rules ({len(wd)} entries)")
        for entry in wd:
            if isinstance(entry[0], str):
                log(f"  ]   {entry[0]}: {_wd_fmt(entry[1])}")
            else:
                log(f"  ]   layers {entry[0]}-{entry[1]}: {_wd_fmt(entry[2])}")
        log(f"  ]   Norms/biases: 0.0000 (hardcoded)")
    else:
        log(f"  ] Weight Decay = {wd:.4f}")
    cautious_wd = getattr(settings, 'cautious_weight_decay', False)
    if cautious_wd:
        log(f"  ] Cautious WD  = ENABLED")

    # LR modifiers summary
    lr_mods = getattr(settings, 'lr_mods', None)
    if lr_mods:
        log(f"  ] LR Modifiers = {len(lr_mods)} rules")
        for entry in lr_mods:
            if isinstance(entry[0], str) and len(entry) == 2:
                log(f"  ]   {entry[0]}: {entry[1]}")
            elif isinstance(entry[0], str) and len(entry) == 3:
                log(f"  ]   {entry[0]} {entry[1]}: {entry[2]}")
            else:
                log(f"  ]   layers {entry[0]}-{entry[1]} {entry[2]}: {entry[3]}")


def configure_optimizers(
    model,
    optimizer_type,
    weight_decay,
    learning_rate,
    betas=(0.9, 0.95),
    device_type="cuda",
    # Muon-family parameters
    muon_momentum=0.95,
    muon_ns_steps=5,
    normuon_beta2=0.95,
    dion_kwargs=None,
    distributed_mesh=None,
    # Adafactor-specific
    adafactor_beta2=None,
    # Weight decay options
    cautious_weight_decay=False,
    # MuonSphere settings
    muonsphere_radius_scale=2.0,
    muonsphere_power_iters=10,
    # Dion2-specific
    dion2_fraction=0.25,
    dion2_ef_decay=0.95,
    # 16-bit Adam state dtype
    adam16bit_state_dtype="mixed",
    # Muon Adam state precision
    muon_adam_state_dtype="fp32",
):
    """
    Configure optimizer for FSDP2 training.

    Args:
        model: The model to optimize
        optimizer_type: Which optimizer to use. Valid values:
            Adam family:
              - "adamw": Standard AdamW (fused if CUDA)
              - "adamw_8bit": TorchAO 8-bit AdamW
              - "adamw_16bit": 16-bit AdamW (FP16/BF16 states via torchao base)
              - "adamc": AdamC (corrected weight decay for normalized layers)
              - "adamc_8bit": AdamC with TorchAO 8-bit
              - "adamc_16bit": AdamC with 16-bit states (FP16/BF16)
              - "adafactor": FSDP2-compatible Adafactor
            Microsoft DION package:
              - "muon_dion": DION Muon (Newton-Schulz orthogonalization)
              - "normuon_dion": DION NorMuon (neuron-wise normalization)
              - "dion2": DION2 (submatrix selection)
            FSDP2-native Muon (samsja's implementation):
              - "muon_fsdp2": Muon with DTensor gather/scatter
              - "normuon_fsdp2": NorMuon with DTensor + neuron-wise norm
              - "muonsphere_fsdp2": MuonSphere (spectral sphere optimization)
              - "normuon_sphere_fsdp2": NorMuon + Sphere combined
        weight_decay: Scalar weight decay for all non-norm param groups (0.0 when rules-based WD is active).
        learning_rate: Learning rate (used for all optimizer types)
        betas: Adam beta coefficients (beta1, beta2)
        device_type: Device type ('cuda', 'cpu')
        muon_momentum: Momentum for Muon/NorMuon (default: 0.95)
        muon_ns_steps: Newton-Schulz iterations (default: 5)
        normuon_beta2: Second moment coefficient for NorMuon variants (default: 0.95)
        dion_kwargs: Dict of passthrough kwargs for DION variants (nesterov, use_triton, etc.)
        distributed_mesh: DeviceMesh for distributed training
        adafactor_beta2: Beta2 for Adafactor. None=auto-schedule, or fixed value
        cautious_weight_decay: CWD - only decay weights where momentum and weight agree.
                               Valid for Adam-family only. Raises error with Muon-family.
        muonsphere_radius_scale: Spectral sphere constraint radius scale (default: 2.0)
        muonsphere_power_iters: Power iterations for spectral estimation (default: 10)
        dion2_fraction: Submatrix fraction for Dion2 (default: 0.25)
        dion2_ef_decay: Error feedback decay for Dion2 (default: 0.95)
        adam16bit_state_dtype: State dtype for 16-bit Adam variants. Options:
            "mixed" (default): FP16 for exp_avg, BF16 for exp_avg_sq
            "fp16": Both in FP16 (caution: exp_avg_sq may underflow)
            "bf16": Both in BF16 (safe range, less precision)
        muon_adam_state_dtype: State dtype for Adam params in Muon-family optimizers.
            "fp32" (default): Standard FP32 states (backward compatible)
            "mixed": FP16 for exp_avg, BF16 for exp_avg_sq (with stochastic rounding)
            "fp16": Both in FP16
            "bf16": Both in BF16
    """
    # ── Validation ──────────────────────────────────────────────────
    if optimizer_type not in VALID_OPTIMIZER_TYPES:
        raise ValueError(
            f"Unknown optimizer_type: '{optimizer_type}'. "
            f"Valid options: {sorted(VALID_OPTIMIZER_TYPES)}"
        )

    if cautious_weight_decay and optimizer_type in MUON_FAMILY:
        raise ValueError(
            f"cautious_weight_decay=True is incompatible with optimizer_type='{optimizer_type}'. "
            "Muon's Newton-Schulz orthogonalization destroys coordinate-wise structure, "
            "making the CWD mask essentially random."
        )

    # ── Param classification helpers ────────────────────────────────
    # MoE param naming: expert 3D weights are nn.Parameter named "moe.experts.w1"
    # (no .weight suffix) — the .weight check below excludes them automatically.
    # Shared expert 2D weights "moe.shared_experts.w1.weight" DO match → Muon. Correct.
    # Router gate "moe.router.gate.weight" doesn't match wq/wk/etc → Adam. Correct.
    # GDN projection names: q_proj, k_proj, v_proj, o_proj, g_proj are large 2D matrices → Muon.
    # Excluded: a_proj, b_proj (tiny gate scalars), conv1d (no match), o_norm (caught by is_no_decay).
    _muon_patterns = ['wq.', 'wk.', 'wv.', 'wo.', 'w1.', 'w2.', 'w3.',
                      'q_proj.', 'k_proj.', 'v_proj.', 'o_proj.', 'g_proj.']

    def is_muon_param(name):
        """Parameters suitable for Muon: 2D weight matrices in attention/FFN."""
        return '.weight' in name and any(lt in name for lt in _muon_patterns)

    def is_normalized_param(name):
        """Weights in attention/FFN (not output layer) — for AdamC corrected WD."""
        return not name.startswith('output.') and '.weight' in name and any(
            lt in name for lt in _muon_patterns
        )

    def is_no_decay(name):
        """Biases and layer norm weights — no weight decay."""
        return name.endswith("bias") or ("norm" in name.lower() and name.endswith("weight"))

    def classify_param(name):
        """Classify parameter into weight decay group."""
        if is_no_decay(name):
            return 'norm_bias'
        if is_muon_param(name):
            return 'default'
        if 'tok_embeddings' in name:
            return 'embeddings'
        if name.startswith('output.'):
            return 'output_head'
        return 'default'  # fallback

    # ── Weight decay (scalar for all non-norm groups) ────────────────
    wd_value = float(weight_decay)

    # =====================================================================
    # DION family: muon_dion, normuon_dion, dion2
    # =====================================================================
    if optimizer_type in DION_FAMILY:
        if optimizer_type == "muon_dion":
            from dion import Muon as DionOptimizer
        elif optimizer_type == "normuon_dion":
            from dion import NorMuon as DionOptimizer
        elif optimizer_type == "dion2":
            from dion import Dion2 as DionOptimizer

        extra_kwargs = dion_kwargs or {}

        muon_params = []
        adam_params = []
        for n, p in model.named_parameters():
            if is_muon_param(n):
                muon_params.append(p)
            else:
                adam_params.append(p)

        param_groups = [
            {
                "params": muon_params,
                "lr": learning_rate,
                "use_muon": True,
            },
            {
                "params": adam_params,
                "lr": learning_rate,
                "algorithm": "adamw",
                "betas": betas,
                "weight_decay": 0.0,
                "use_muon": False,
            },
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        if optimizer_type == "dion2":
            opt_kwargs = dict(
                lr=learning_rate,
                fraction=dion2_fraction,
                ef_decay=dion2_ef_decay,
                betas=betas,
                weight_decay=wd_value,
                distributed_mesh=distributed_mesh,
            )
        else:
            # muon_dion or normuon_dion
            opt_kwargs = dict(
                lr=learning_rate,
                mu=muon_momentum,
                betas=betas,
                weight_decay=wd_value,
                distributed_mesh=distributed_mesh,
            )

        opt_kwargs.update(extra_kwargs)
        optimizer = DionOptimizer(param_groups, **opt_kwargs)
        return optimizer

    # =====================================================================
    # FSDP2-native Muon family: muon_fsdp2, normuon_fsdp2, muonsphere_fsdp2, normuon_sphere_fsdp2
    # =====================================================================
    if optimizer_type in FSDP2_MUON_FAMILY:
        from muon_fsdp2 import Muon as MuonFSDP2

        muon_params, adam_default_params, embed_params, output_params, norm_params = [], [], [], [], []
        for n, p in model.named_parameters():
            group = classify_param(n)
            if group == 'norm_bias':
                norm_params.append(p)
            elif group == 'embeddings':
                embed_params.append(p)
            elif group == 'output_head':
                output_params.append(p)
            elif is_muon_param(n):
                muon_params.append(p)
            else:
                # Everything else: expert 3D weights, router gate, etc. → Adam
                adam_default_params.append(p)

        use_normuon = optimizer_type in NORMUON_VARIANTS
        use_muonsphere = optimizer_type in MUONSPHERE_VARIANTS

        # MuonSphere: weight decay handled by spectral retraction
        muon_weight_decay = 0.0 if use_muonsphere else wd_value

        param_groups = [
            dict(
                params=muon_params,
                lr=learning_rate,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                use_muon=True,
                nesterov=True,
                ns_steps=muon_ns_steps,
                rms_scale=True,
                use_normuon=use_normuon,
                beta2=normuon_beta2,
                cautious_weight_decay=False,
                use_muonsphere=use_muonsphere,
                radius_scale=muonsphere_radius_scale,
                power_iters=muonsphere_power_iters,
                wd_group='default',
            ),
            dict(
                params=adam_default_params,
                lr=learning_rate,
                betas=betas,
                eps=1e-10,
                weight_decay=wd_value,
                use_muon=False,
                cautious_weight_decay=False,
                wd_group='adam_default',
            ),
            dict(
                params=embed_params,
                lr=learning_rate,
                betas=betas,
                eps=1e-10,
                weight_decay=wd_value,
                use_muon=False,
                cautious_weight_decay=False,
                wd_group='embeddings',
            ),
            dict(
                params=output_params,
                lr=learning_rate,
                betas=betas,
                eps=1e-10,
                weight_decay=wd_value,
                use_muon=False,
                cautious_weight_decay=False,
                wd_group='output_head',
            ),
            dict(
                params=norm_params,
                lr=learning_rate,
                betas=betas,
                eps=1e-10,
                weight_decay=0.0,
                use_muon=False,
                cautious_weight_decay=False,
                wd_group='norm_bias',
            ),
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = MuonFSDP2(param_groups, adam_state_dtype=muon_adam_state_dtype)
        return optimizer

    # =====================================================================
    # Adafactor
    # =====================================================================
    if optimizer_type == "adafactor":
        from adafactor_fsdp2 import AdafactorFSDP2

        all_params = list(model.parameters())
        optimizer = AdafactorFSDP2(
            all_params,
            lr=learning_rate,
            beta2=adafactor_beta2,
            weight_decay=wd_value,
        )
        return optimizer

    # =====================================================================
    # AdamC family: adamc, adamc_8bit, adamc_16bit
    # =====================================================================
    if optimizer_type in ("adamc", "adamc_8bit", "adamc_16bit"):
        from adamc_optimizer import AdamC, AdamC8bitTorchAO

        normalized_params, unnormalized_params, embed_params, output_params, norm_params = [], [], [], [], []
        for n, p in model.named_parameters():
            group = classify_param(n)
            if group == 'norm_bias':
                norm_params.append(p)
            elif group == 'embeddings':
                embed_params.append(p)
            elif group == 'output_head':
                output_params.append(p)
            elif is_normalized_param(n):
                normalized_params.append(p)
            else:
                # Expert 3D weights, router gate, etc. — standard WD, no correction
                unnormalized_params.append(p)

        optimizer_grouped_parameters = [
            {"params": normalized_params, "weight_decay": wd_value, "is_normalized": True, "wd_group": "default"},
            {"params": unnormalized_params, "weight_decay": wd_value, "is_normalized": False, "wd_group": "adam_default"},
            {"params": embed_params, "weight_decay": wd_value, "is_normalized": False, "wd_group": "embeddings"},
            {"params": output_params, "weight_decay": wd_value, "is_normalized": False, "wd_group": "output_head"},
            {"params": norm_params, "weight_decay": 0.0, "is_normalized": False, "wd_group": "norm_bias"},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

        if optimizer_type == "adamc_16bit":
            from adamw_16bit import AdamC16bit
            optimizer = AdamC16bit(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                weight_decay=wd_value,
                state_dtype=adam16bit_state_dtype,
            )
        elif optimizer_type == "adamc_8bit":
            optimizer = AdamC8bitTorchAO(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                weight_decay=wd_value,
            )
        else:
            optimizer = AdamC(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=betas,
                weight_decay=wd_value,
            )
        return optimizer

    # =====================================================================
    # AdamW family: adamw, adamw_8bit, adamw_16bit
    # =====================================================================
    decay_params, embed_params, output_params, norm_params = [], [], [], []
    for n, p in model.named_parameters():
        group = classify_param(n)
        if group == 'norm_bias':
            norm_params.append(p)
        elif group == 'embeddings':
            embed_params.append(p)
        elif group == 'output_head':
            output_params.append(p)
        else:  # default — transformer layer weights
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": wd_value, "wd_group": "default"},
        {"params": embed_params, "weight_decay": wd_value, "wd_group": "embeddings"},
        {"params": output_params, "weight_decay": wd_value, "wd_group": "output_head"},
        {"params": norm_params, "weight_decay": 0.0, "wd_group": "norm_bias"},
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

    if optimizer_type == "adamw_16bit":
        from adamw_16bit import AdamW16bit
        optimizer = AdamW16bit(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=wd_value,
            state_dtype=adam16bit_state_dtype,
        )
    elif optimizer_type == "adamw_8bit":
        try:
            from torchao.optim import AdamW8bit as TorchAOAdamW8bit
        except ImportError:
            from torchao.prototype.low_bit_optim import AdamW8bit as TorchAOAdamW8bit
        optimizer = TorchAOAdamW8bit(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=wd_value,
        )
    else:
        # Standard AdamW
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

    return optimizer
