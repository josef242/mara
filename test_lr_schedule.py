"""Quick test: visualize the plateau LR schedule to verify correctness."""
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_lr_with_dual_plateau(it, max_lr, min_lr, warmup_steps, max_steps,
                             first_plat_lr, decay_to_first_plat_pct, first_plat_len_pct,
                             decay_to_second_pct, second_plat_lr, second_plat_len_pct):
    # Calculate phase boundaries
    decay_to_first_end = warmup_steps + int(decay_to_first_plat_pct * max_steps)
    first_plat_end = decay_to_first_end + int(first_plat_len_pct * max_steps)
    decay_to_second_end = first_plat_end + int(decay_to_second_pct * max_steps)
    second_plat_end = decay_to_second_end + int(second_plat_len_pct * max_steps)

    # Phase 1: Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # Phase 2: Cosine decay from max_lr to first_plat_lr
    if it < decay_to_first_end:
        progress = (it - warmup_steps) / (decay_to_first_end - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return first_plat_lr + coeff * (max_lr - first_plat_lr)

    # Phase 3: First plateau
    if it < first_plat_end:
        return first_plat_lr

    # Phase 4: Cosine decay from first_plat_lr to second_plat_lr
    if it < decay_to_second_end:
        progress = (it - first_plat_end) / (decay_to_second_end - first_plat_end)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return second_plat_lr + coeff * (first_plat_lr - second_plat_lr)

    # Phase 5: Second plateau
    if it < second_plat_end:
        return second_plat_lr

    # Phase 6: Final cosine decay to min_lr
    progress = (it - second_plat_end) / (max_steps - second_plat_end)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (second_plat_lr - min_lr)


def get_lr_cosine(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    progress = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# --- Config from keel-moe-GDN-L.yaml ---
max_steps = 200_000
max_lr = 3.0e-04
min_lr = 3.0e-05
warmup_steps = 1000

# Plateau params
first_plat_lr = 2.0e-04
decay_to_first_plat_pct = 0.05
first_plat_len_pct = 0.45
decay_to_second_pct = 0.12
second_plat_lr = 6.0e-05
second_plat_len_pct = 0.25

# Print phase boundaries
d1_end = warmup_steps + int(decay_to_first_plat_pct * max_steps)
p1_end = d1_end + int(first_plat_len_pct * max_steps)
d2_end = p1_end + int(decay_to_second_pct * max_steps)
p2_end = d2_end + int(second_plat_len_pct * max_steps)

print(f"Phase boundaries:")
print(f"  Warmup:        0 → {warmup_steps:,}")
print(f"  Decay→P1:  {warmup_steps:,} → {d1_end:,}")
print(f"  Plateau 1: {d1_end:,} → {p1_end:,}  @ {first_plat_lr:.2e}")
print(f"  Decay→P2:  {p1_end:,} → {d2_end:,}")
print(f"  Plateau 2: {d2_end:,} → {p2_end:,}  @ {second_plat_lr:.2e}")
print(f"  Final:     {p2_end:,} → {max_steps:,}  → {min_lr:.2e}")
print(f"  Sum check: {p2_end + (max_steps - p2_end):,} = {max_steps:,}")

# Generate LR curves
steps = list(range(max_steps))
lr_plateau = [get_lr_with_dual_plateau(s, max_lr, min_lr, warmup_steps, max_steps,
              first_plat_lr, decay_to_first_plat_pct, first_plat_len_pct,
              decay_to_second_pct, second_plat_lr, second_plat_len_pct) for s in steps]
lr_cosine = [get_lr_cosine(s, max_lr, min_lr, warmup_steps, max_steps) for s in steps]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(steps, lr_cosine, label='Cosine', alpha=0.7, linewidth=1.5)
ax1.plot(steps, lr_plateau, label='Plateau', alpha=0.9, linewidth=2)
ax1.axhline(y=first_plat_lr, color='gray', linestyle=':', alpha=0.4, label=f'P1 = {first_plat_lr:.2e}')
ax1.axhline(y=second_plat_lr, color='gray', linestyle='--', alpha=0.4, label=f'P2 = {second_plat_lr:.2e}')
for x in [warmup_steps, d1_end, p1_end, d2_end, p2_end]:
    ax1.axvline(x=x, color='lightgray', linestyle=':', alpha=0.5)
ax1.set_ylabel('Learning Rate')
ax1.set_title('Plateau vs Cosine LR Schedule')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative energy (integral)
import itertools
cum_plateau = list(itertools.accumulate(lr_plateau))
cum_cosine = list(itertools.accumulate(lr_cosine))
ax2.plot(steps, cum_cosine, label=f'Cosine (total={cum_cosine[-1]:.1f})', alpha=0.7)
ax2.plot(steps, cum_plateau, label=f'Plateau (total={cum_plateau[-1]:.1f}, {cum_plateau[-1]/cum_cosine[-1]*100:.0f}%)', alpha=0.9)
ax2.set_xlabel('Step')
ax2.set_ylabel('Cumulative LR')
ax2.set_title('LR Energy (cumulative sum)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_schedule_test.png', dpi=150)
plt.show()
print("\nSaved to lr_schedule_test.png")
