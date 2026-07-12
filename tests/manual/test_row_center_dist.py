"""Distributed test for row-centering on a VOCAB-SHARDED head (req #3).

Proves the GLOBAL mean is correct under sharding, and that a naive PER-SHARD
mean would subtract different offsets per vocab region (changing probabilities).

Run on a multi-GPU box (e.g. a rig), NOT the Windows CPU env:
    torchrun --nproc_per_node=2 test_row_center_dist.py

Uses a DTensor sharded on dim 0 (vocab) over the default process group, which
is exactly how FSDP2 lays out output.weight after fully_shard(model, dp_mesh).
"""
import os
import sys
sys.path.insert(0, "../common_fsdp2")

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, DTensor
from row_center import row_center_head_, _global_row_mean


def main():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(dev)
    mesh = init_device_mesh(dev.type, (world,))

    V, D = 4096, 256
    # Build the SAME full weight on every rank (seeded), with a real gauge.
    torch.manual_seed(123)
    W_full = (torch.randn(V, D) * 0.5 + 0.6).to(dev)

    # Single-device reference: center a private full copy.
    W_ref = W_full.clone()
    ref_tel = row_center_head_(W_ref)  # plain tensor path

    # Sharded path: distribute W over the vocab dim, then center via our op.
    W_dt = distribute_tensor(W_full.clone(), mesh, [Shard(0)])
    assert isinstance(W_dt, DTensor)
    dt_tel = row_center_head_(W_dt)

    # Gather the sharded result back to full and compare to the single-dev ref.
    W_dt_full = W_dt.full_tensor()
    max_diff = (W_dt_full - W_ref).abs().max().item()

    # What a WRONG per-shard mean would have produced (local-only mean):
    local = W_full.clone()
    shard_rows = V // world
    s = rank * shard_rows
    e = V if rank == world - 1 else s + shard_rows
    local_mu = local[s:e].float().mean(0)            # per-shard mean (WRONG)
    global_mu, _ = _global_row_mean(W_dt, 0)         # correct global mean
    pershard_vs_global = (local_mu - global_mu).norm().item()

    if rank == 0:
        print(f"world={world}")
        print(f"  [global centering] sharded == single-device ref?  max|diff|={max_diff:.2e}")
        print(f"  [telemetry] ref mu_pre={ref_tel['mu_w_pre']:.4f}  dt mu_pre={dt_tel['mu_w_pre']:.4f}  "
              f"(match: {abs(ref_tel['mu_w_pre']-dt_tel['mu_w_pre'])<1e-3})")
        print(f"  [sharded] mu_w_post={dt_tel['mu_w_post']:.2e} (should be ~0)")
        print(f"  [per-shard WOULD differ] ||local_mu - global_mu||={pershard_vs_global:.4f} "
              f"(nonzero => per-shard means subtract DIFFERENT offsets => changes probs)")
        ok = (max_diff < 1e-3) and (dt_tel['mu_w_post'] < 1e-3) and (pershard_vs_global > 1e-3)
        print("\nRESULT:", "PASS" if ok else "FAIL")
        if not ok:
            dist.destroy_process_group()
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
