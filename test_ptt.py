"""
Statistical analysis of Progressive Tail Truncation cut-point distributions.

Usage: python test_ptt.py
"""
import sys
sys.path.insert(0, '../common_fsdp2')

from tail_truncation import ProgressiveTailTruncation

N_LAYERS = 70
N_STEPS = 50_000
START_STEP = 0


def analyze(name: str, config: dict):
    ptt = ProgressiveTailTruncation(n_layers=N_LAYERS, config=config)
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")

    truncated_steps = []
    cut_points = []  # only truncated steps

    for step in range(START_STEP, START_STEP + N_STEPS):
        al = ptt.get_truncation_point(step)
        if al < N_LAYERS:
            truncated_steps.append(step)
            cut_points.append(al)

    total = N_STEPS
    n_trunc = len(truncated_steps)
    print(f"Total steps: {total}")
    print(f"Truncated:   {n_trunc} ({100*n_trunc/total:.1f}%)")
    print(f"Full-depth:  {total - n_trunc} ({100*(total-n_trunc)/total:.1f}%)")

    if not cut_points:
        print("  (no truncations occurred)")
        return

    # Zone info
    final_safe_frac = ptt._safe_schedule[-1][1]
    safe_layer = int(N_LAYERS * final_safe_frac)
    zone_size = N_LAYERS - safe_layer
    print(f"\nZone: layers {safe_layer}-{N_LAYERS-1} ({zone_size} layers)")
    print(f"Min cut: {min(cut_points)}  Max cut: {max(cut_points)}  "
          f"Mean cut: {sum(cut_points)/len(cut_points):.1f}")

    # Histogram -- bucket by relative depth in zone
    # "shallow" = near n_layers (high layer number), "deep" = near safe_layer (low layer number)
    n_buckets = min(zone_size, 5)
    bucket_size = zone_size / n_buckets
    buckets = [0] * n_buckets
    for cp in cut_points:
        relative = cp - safe_layer
        b = min(int(relative / bucket_size), n_buckets - 1)
        buckets[b] += 1

    print(f"\nDistribution across truncation zone (deep -> shallow):")
    for i in range(n_buckets):
        lo = safe_layer + int(i * bucket_size)
        hi = safe_layer + int((i + 1) * bucket_size) - 1
        count = buckets[i]
        pct = 100 * count / n_trunc if n_trunc else 0
        bar = '#' * int(pct / 2)
        label = "DEEP" if i == 0 else "SHALLOW" if i == n_buckets - 1 else ""
        print(f"  L{lo:02d}-L{hi:02d}: {count:5d} ({pct:5.1f}%) {bar}  {label}")

    # Show first 20 truncation events
    print(f"\nFirst 20 truncation events:")
    for step, cp in zip(truncated_steps[:20], cut_points[:20]):
        depth_pct = 100 * (N_LAYERS - cp) / zone_size
        print(f"  step {step:5d}: cut at layer {cp} ({depth_pct:.0f}% of zone removed)")


# ── Test configs ──────────────────────────────────────────────────

# 1) Uniform (power=1)
analyze("depth_power=1.0 (uniform)", {
    'enabled': True,
    'safe_fraction': 0.60,
    'truncation_prob': 0.25,
    'depth_power': 1.0,
})

# 2) Shallow-biased (power=2, default)
analyze("depth_power=2.0 (shallow-biased)", {
    'enabled': True,
    'safe_fraction': 0.60,
    'truncation_prob': 0.25,
    'depth_power': 2.0,
})

# 3) Strongly shallow-biased (power=3)
analyze("depth_power=3.0 (strongly shallow)", {
    'enabled': True,
    'safe_fraction': 0.60,
    'truncation_prob': 0.25,
    'depth_power': 3.0,
})

# 4) Sweet spot? (power=1.5)
analyze("depth_power=1.5 (mild shallow bias)", {
    'enabled': True,
    'safe_fraction': 0.60,
    'truncation_prob': 0.25,
    'depth_power': 1.5,
})

# 5) Scheduled config
analyze("depth_power=2.0 -- scheduled", {
    'enabled': True,
    'safe_fraction': [[0, 1.0], [1000, 0.60]],
    'truncation_prob': [[0, 0.0], [1000, 0.25]],
    'depth_power': 2.0,
})


# ── Gradient verification ─────────────────────────────────────────
import torch
from model_v2 import Transformer, ModelArgs

def test_gradient_flow():
    """Verify that skipped layers receive NO gradients on truncated steps."""
    print(f"\n{'='*70}")
    print(f" GRADIENT VERIFICATION TEST")
    print(f"{'='*70}")

    n_layers = 8
    active = 5  # Only run layers 0-4, skip layers 5-7

    cfg = ModelArgs()
    cfg.dim = 64
    cfg.n_layers = n_layers
    cfg.n_heads = 4
    cfg.vocab_size = 256
    cfg.max_seq_len = 32
    cfg.dropout = 0.0
    cfg.use_activation_checkpointing = False
    cfg.pad_id = 0
    cfg.use_keel = False

    model = Transformer(cfg)
    model.train()

    tokens = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    # Full forward (baseline)
    model.zero_grad()
    _, loss_full = model(tokens, targets)
    loss_full.backward()

    full_grads = {}
    for i, layer in enumerate(model.layers):
        g = layer.feed_forward.w1.weight.grad
        full_grads[i] = g.norm().item() if g is not None else 0.0

    print(f"\n  FULL FORWARD (all {n_layers} layers):")
    for i in range(n_layers):
        status = "HAS GRAD" if full_grads[i] > 0 else "NO GRAD"
        print(f"    Layer {i}: grad_norm = {full_grads[i]:.6f}  [{status}]")

    # Truncated forward
    model.zero_grad()
    _, loss_trunc = model(tokens, targets, active_layers=active)
    loss_trunc.backward()

    print(f"\n  TRUNCATED FORWARD (active_layers={active}, skipping layers {active}-{n_layers-1}):")
    all_correct = True
    for i, layer in enumerate(model.layers):
        g = layer.feed_forward.w1.weight.grad
        grad_norm = g.norm().item() if g is not None else 0.0
        should_have_grad = i < active
        has_grad = grad_norm > 0
        ok = has_grad == should_have_grad
        status = "HAS GRAD" if has_grad else "NO GRAD"
        check = "OK" if ok else "FAIL!"
        print(f"    Layer {i}: grad_norm = {grad_norm:.6f}  [{status}]  {check}")
        if not ok:
            all_correct = False

    # Check norm and output head always get gradients
    norm_grad = model.norm.weight.grad
    norm_ok = norm_grad is not None and norm_grad.norm().item() > 0
    print(f"\n    Norm:   grad_norm = {norm_grad.norm().item() if norm_ok else 0:.6f}  "
          f"[{'HAS GRAD' if norm_ok else 'NO GRAD'}]  {'OK' if norm_ok else 'FAIL!'}")

    out_grad = model.output.weight.grad
    out_ok = out_grad is not None and out_grad.norm().item() > 0
    print(f"    Output: grad_norm = {out_grad.norm().item() if out_ok else 0:.6f}  "
          f"[{'HAS GRAD' if out_ok else 'NO GRAD'}]  {'OK' if out_ok else 'FAIL!'}")

    embed_grad = model.tok_embeddings.weight.grad
    embed_ok = embed_grad is not None and embed_grad.norm().item() > 0
    print(f"    Embed:  grad_norm = {embed_grad.norm().item() if embed_ok else 0:.6f}  "
          f"[{'HAS GRAD' if embed_ok else 'NO GRAD'}]  {'OK' if embed_ok else 'FAIL!'}")

    all_correct = all_correct and norm_ok and out_ok and embed_ok
    print(f"\n  RESULT: {'PASS -- skipped layers get no gradients' if all_correct else 'FAIL'}")

test_gradient_flow()
