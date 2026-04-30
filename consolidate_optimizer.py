#!/usr/bin/env python3
"""Consolidate sharded optimizer state into a single full state dict.

Run this on the SAME cluster topology that produced the checkpoint (e.g. 7 GPUs).
It loads the per-rank optimizer shards, gathers them into a single full state dict
via get_optimizer_state_dict(full_state_dict=True, cpu_offload=True), and saves the
result to a new file that can be loaded on ANY world_size.

Usage:
    torchrun --nproc_per_node=7 consolidate_optimizer.py --config configs/mini-fathom-low-lr.yaml

    # Or with explicit step (overrides config's resume_step):
    torchrun --nproc_per_node=7 consolidate_optimizer.py --config configs/mini-fathom-low-lr.yaml --step 24000

Output:
    <checkpoint_dir>/optimizer_step_NNNNNN_full.pt   (rank 0 only)
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from datetime import timedelta

# Add common_fsdp2 to path
common_path = '../common_fsdp2'
if common_path not in sys.path:
    sys.path.insert(0, common_path)

from model_v2 import Transformer, ModelArgs
from tokenizer_abstraction import get_tokenizer
from configure_optimizers import configure_optimizers


def main():
    parser = argparse.ArgumentParser(description="Consolidate sharded optimizer state")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (overrides config)")
    args = parser.parse_args()

    # --- Load settings (reuse train_mara's Settings class) ---
    # Import Settings from train_mara
    sys.path.insert(0, os.path.dirname(__file__))
    from train_mara import Settings
    settings = Settings.from_yaml(args.config)
    if args.step is not None:
        settings.resume_step = args.step
    assert hasattr(settings, 'resume_step') and settings.resume_step is not None, \
        "resume_step must be set (in config or via --step)"
    settings.resume_training = True  # trigger path derivation

    step = settings.resume_step

    # --- Initialize DDP ---
    assert int(os.environ.get('RANK', -1)) != -1, "Must be launched with torchrun"
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    init_process_group(backend='nccl', timeout=timedelta(minutes=10),
                       device_id=torch.device(device))
    dp_mesh = init_device_mesh("cuda", (ddp_world_size,))

    if ddp_rank == 0:
        print(f"Consolidating optimizer state for step {step} ({ddp_world_size} ranks)")

    # --- Build tokenizer (for vocab size) ---
    enc = get_tokenizer(
        settings.tok_kind,
        path=settings.tok_path,
        special_tokens=getattr(settings, 'special_tokens', None),
    )
    def round_up(x, multiple=128):
        return (x + multiple - 1) // multiple * multiple
    settings.cfg_voc_sz = round_up(len(enc), 1024)

    # --- Determine EP degree (same logic as train_mara main) ---
    ep_degree = getattr(settings, 'ep_degree', None)
    if ep_degree is None:
        if getattr(settings, 'moe_enabled', False):
            ep_degree = ddp_world_size
        else:
            ep_degree = 1

    # For this script, we only handle ep_degree=1 (no EP).
    # EP models have separate expert consolidation files already.
    assert ep_degree == 1, (
        f"This script handles dense models (ep_degree=1). "
        f"Got ep_degree={ep_degree}. EP expert weights are already consolidated separately."
    )

    # --- Build model config (mirrors train_mara main) ---
    model_cfg = ModelArgs(
        dim=settings.cfg_embd,
        inner_dim=settings.cfg_intermediate,
        n_heads=settings.cfg_heads,
        n_kv_heads=getattr(settings, 'cfg_kv_heads', None),
        n_layers=settings.cfg_layers,
        vocab_size=settings.cfg_voc_sz,
        max_seq_len=settings.T,
        dropout=settings.dropout,
        use_activation_checkpointing=settings.use_activation_checkpointing,
        norm_eps=settings.norm_eps,
        pad_id=enc.pad_id,
        qk_norm_mode=settings.qk_norm_mode,
        tie_word_embeddings=getattr(settings, 'tie_word_embeddings', True),
        rope_theta=getattr(settings, 'rope_theta', 500000.0),
        use_keel=getattr(settings, 'use_keel', False),
        keel_alpha=getattr(settings, 'keel_alpha', None),
        moe_enabled=getattr(settings, 'moe_enabled', False),
        moe_num_experts=getattr(settings, 'moe_num_experts', 8),
        moe_top_k=getattr(settings, 'moe_top_k', 2),
        moe_num_shared_experts=getattr(settings, 'moe_num_shared_experts', 1),
        moe_score_func=getattr(settings, 'moe_score_func', 'sigmoid'),
        moe_score_before_experts=getattr(settings, 'moe_score_before_experts', True),
        moe_route_norm=getattr(settings, 'moe_route_norm', False),
        moe_route_scale=getattr(settings, 'moe_route_scale', 1.0),
        moe_load_balance_coeff=getattr(settings, 'moe_load_balance_coeff', 1e-3),
        moe_interleave_step=getattr(settings, 'moe_interleave_step', 1),
        moe_n_dense_layers=getattr(settings, 'moe_n_dense_layers', 0),
        moe_n_tail_dense_layers=getattr(settings, 'moe_n_tail_dense_layers', 0),
        moe_capacity_factor=getattr(settings, 'moe_capacity_factor', 0.0),
        moe_inner_dim=getattr(settings, 'moe_inner_dim', None),
        ep_degree=ep_degree,
        gdn_enabled=getattr(settings, 'gdn_enabled', False),
        gdn_interleave_step=getattr(settings, 'gdn_interleave_step', 4),
        n_gdn_heads=getattr(settings, 'n_gdn_heads', None),
        gdn_head_dim=getattr(settings, 'gdn_head_dim', None),
        gdn_v_expand=getattr(settings, 'gdn_v_expand', 2.0),
        gdn_short_conv_kernel=getattr(settings, 'gdn_short_conv_kernel', 4),
        gdn_mode=getattr(settings, 'gdn_mode', 'chunk'),
        attn_res_enabled=getattr(settings, 'attn_res_enabled', False),
        attn_res_block_size=getattr(settings, 'attn_res_block_size', 8),
    )

    # --- Create model on meta device + FSDP shard ---
    if ddp_rank == 0:
        print("  ] Creating model on meta device...")

    target_dtype = (
        torch.bfloat16 if settings.FSDP_param_dtype == 'bf16'
        else torch.float16 if settings.FSDP_param_dtype == 'fp16'
        else torch.float32
    )
    reduce_dtype = (
        torch.bfloat16 if settings.FSDP_reduce_dtype == 'bf16'
        else torch.float16 if settings.FSDP_reduce_dtype == 'fp16'
        else torch.float32
    )

    with torch.device('meta'):
        model = Transformer(model_cfg)

    try:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=target_dtype, reduce_dtype=reduce_dtype, output_dtype=target_dtype)
    except TypeError:
        mp_policy = MixedPrecisionPolicy(param_dtype=target_dtype, reduce_dtype=reduce_dtype)

    reshard_after_forward = getattr(settings, 'reshard_after_forward', True)
    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=False)

    # Materialize
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    model = model.to_empty(device=device)
    torch.set_default_dtype(old_dtype)

    # Init weights (needed so optimizer param shapes are correct before loading)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if hasattr(model, 'init_weights'):
        model.init_weights()
    else:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    if ddp_rank == 0:
        print("  ] Model created and sharded.")

    # --- Load model weights ---
    if ddp_rank == 0:
        print("  ] Loading model weights...")
    checkpoint_path = settings.resume_checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint["model"]

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_model_state_dict(model, state_dict, options=options)
    del state_dict, checkpoint
    torch.cuda.empty_cache()
    dist.barrier()

    if ddp_rank == 0:
        print("  ] Model weights loaded.")

    # --- Create optimizer (same config as training) ---
    if ddp_rank == 0:
        print("  ] Creating optimizer...")
    optimizer = configure_optimizers(
        model=model,
        optimizer_type=settings.optimizer_type,
        weight_decay=settings.weight_decay if isinstance(settings.weight_decay, (int, float)) else 0.0,
        learning_rate=settings.max_lr,
        betas=(getattr(settings, 'beta1', 0.9), getattr(settings, 'beta2', 0.95)),
        device_type="cuda",
        muon_momentum=getattr(settings, 'muon_momentum', 0.95),
        muon_ns_steps=getattr(settings, 'muon_ns_steps', 5),
        normuon_beta2=getattr(settings, 'normuon_beta2', 0.95),
        dion_kwargs=getattr(settings, 'dion_kwargs', None),
        distributed_mesh=dp_mesh,
        adafactor_beta2=getattr(settings, 'adafactor_beta2', None),
        cautious_weight_decay=getattr(settings, 'cautious_weight_decay', False),
        muonsphere_radius_scale=getattr(settings, 'muonsphere_radius_scale', 2.0),
        muonsphere_power_iters=getattr(settings, 'muonsphere_power_iters', 10),
        dion2_fraction=getattr(settings, 'dion2_fraction', 0.25),
        dion2_ef_decay=getattr(settings, 'dion2_ef_decay', 0.95),
        adam16bit_state_dtype=getattr(settings, 'adam16bit_state_dtype', 'mixed'),
        muon_adam_state_dtype=getattr(settings, 'muon_adam_state_dtype', 'fp32'),
    )

    # --- Load sharded optimizer state ---
    if ddp_rank == 0:
        print("  ] Loading sharded optimizer state...")

    pth = os.path.dirname(checkpoint_path)
    optim_shard_path = os.path.join(pth, f"optimizer_step_{step:06d}_rank_{ddp_rank}.pt")
    assert os.path.exists(optim_shard_path), f"Missing optimizer shard: {optim_shard_path}"

    optim_shard = torch.load(optim_shard_path, map_location="cpu")
    load_optim = getattr(optimizer, '_base_optimizer', optimizer)

    # Preserve param_group keys (same as resume_training does)
    pg_defaults = [{k: v for k, v in pg.items() if k != 'params'} for pg in load_optim.param_groups]
    set_optimizer_state_dict(model, load_optim, optim_shard,
                            options=StateDictOptions(full_state_dict=False))
    del optim_shard

    for pg, defaults in zip(load_optim.param_groups, pg_defaults):
        for k, v in defaults.items():
            if k not in pg:
                pg[k] = v

    dist.barrier()
    if ddp_rank == 0:
        print("  ] Sharded optimizer state loaded.")

    # --- Gather full optimizer state and save ---
    if ddp_rank == 0:
        print("  ] Gathering full optimizer state to CPU (this may take a moment)...")

    full_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_optim_sd = get_optimizer_state_dict(model, load_optim, options=full_options)

    if ddp_rank == 0:
        out_path = os.path.join(pth, f"optimizer_step_{step:06d}_full.pt")
        print(f"  ] Saving full optimizer state to {out_path}...")
        torch.save(full_optim_sd, out_path)
        # Report size
        file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  ] Done! Saved {file_size_mb:.1f} MB")
        print(f"\n  To resume on a different world_size, the resume code will")
        print(f"  detect this file and load with full_state_dict=True.")

    del full_optim_sd
    torch.cuda.empty_cache()
    dist.barrier()

    destroy_process_group()
    if ddp_rank == 0:
        print("  ] Consolidation complete.")


if __name__ == "__main__":
    main()
