# QK-Clip Implementation Synopsis

## Overview

QK-Clip is a stability mechanism developed by Moonshot AI for their Kimi K2 model to address **exploding attention logits** during Muon optimizer training. It operates as a post-optimizer weight adjustment that scales Q and K projection matrices when attention scores exceed a threshold.

### The Problem It Solves

When training with Muon (or similar aggressive optimizers), attention logits can grow unbounded:
- Logits exceeding 1000+ cause softmax to become numerically one-hot
- This leads to massive gradients when the model is confidently wrong
- Results in loss spikes and potential training divergence
- Kimi observed this at 9B+ scale; it becomes worse with scale

### Key Insight

Rather than clipping the attention logits directly (which distorts attention distributions), QK-Clip scales the **source weights** (W_q and W_k) to prevent logits from exceeding a threshold. This is a negative feedback controller that intervenes only when needed.

---

## Algorithm

### Step 1: Monitor Attention Logits

During the forward pass, track the maximum attention logit per head:

```python
# Inside attention forward pass
attention_logits = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
max_logit_per_head = attention_logits.max(dim=-1).values.max(dim=-1).values  # [num_heads]
```

### Step 2: Compute Scaling Factor

After the optimizer step, for each attention head h:

```python
tau = 100.0  # threshold (configurable)
S_max_h = max_logit_per_head[h]

if S_max_h > tau:
    gamma_h = tau / S_max_h  # gamma < 1, will shrink weights
else:
    gamma_h = 1.0  # no action needed
```

### Step 3: Scale Q and K Weights

Apply the scaling to the projection matrices:

```python
# For standard MHA (Multi-Head Attention):
W_q[h] *= sqrt(gamma_h)  # Scale query weights for head h
W_k[h] *= sqrt(gamma_h)  # Scale key weights for head h

# The sqrt split ensures: new_logit = (Q * sqrt(γ)) @ (K * sqrt(γ))^T = Q @ K^T * γ
# So if gamma = tau / S_max, new_logit = old_logit * (tau / S_max) ≈ tau
```

### Why sqrt(gamma) on both Q and K?

The attention score is `Q @ K^T`. If we want to scale the result by γ:
- `(Q * sqrt(γ)) @ (K * sqrt(γ))^T = Q @ K^T * γ`

This distributes the correction evenly between Q and K, preserving the relative geometry.

---

## Implementation for Standard Multi-Head Attention

```python
import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

class QKClip:
    """
    QK-Clip mechanism for attention stability.
    
    Monitors attention logits and scales Q/K projection weights
    when logits exceed threshold to prevent training instability.
    """
    
    def __init__(
        self,
        tau: float = 100.0,
        enabled: bool = True,
        log_clips: bool = True,
    ):
        """
        Args:
            tau: Maximum allowed attention logit. Kimi K2 used 100.0.
                 Lower values (30-50) are also safe per their experiments.
            enabled: Whether QK-Clip is active.
            log_clips: Whether to log when clipping occurs.
        """
        self.tau = tau
        self.enabled = enabled
        self.log_clips = log_clips
        
        # Statistics tracking
        self.clip_counts = {}  # layer_name -> count of clips
        self.max_logits_history = {}  # layer_name -> list of max logits
    
    def compute_max_logits(
        self,
        Q: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        K: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        scale: float,
    ) -> torch.Tensor:
        """
        Compute max attention logit per head (for monitoring).
        
        Returns:
            max_logits: [num_heads] tensor of max logits per head
        """
        # Compute attention scores
        # Q: [B, H, S, D], K: [B, H, S, D]
        # scores: [B, H, S, S]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Get max per head across batch and sequence dimensions
        max_per_head = scores.abs().amax(dim=(0, 2, 3))  # [num_heads]
        
        return max_per_head
    
    @torch.no_grad()
    def maybe_clip_weights(
        self,
        layer_name: str,
        W_q: nn.Parameter,  # Query projection weight
        W_k: nn.Parameter,  # Key projection weight
        max_logits: torch.Tensor,  # [num_heads] max logit per head
        num_heads: int,
    ) -> Tuple[int, float]:
        """
        Apply QK-Clip to Q and K projection weights if needed.
        
        Args:
            layer_name: Name of the layer (for logging)
            W_q: Query projection weight, shape [num_heads * head_dim, embed_dim]
                 or [embed_dim, embed_dim] depending on implementation
            W_k: Key projection weight, same shape as W_q
            max_logits: Maximum attention logit per head from forward pass
            num_heads: Number of attention heads
            
        Returns:
            Tuple of (num_heads_clipped, max_logit_value)
        """
        if not self.enabled:
            return 0, max_logits.max().item()
        
        max_logit_value = max_logits.max().item()
        
        # Track statistics
        if layer_name not in self.max_logits_history:
            self.max_logits_history[layer_name] = []
        self.max_logits_history[layer_name].append(max_logit_value)
        
        # Check which heads need clipping
        needs_clip = max_logits > self.tau
        
        if not needs_clip.any():
            return 0, max_logit_value
        
        num_clipped = needs_clip.sum().item()
        
        # Compute per-head scaling factors
        gamma = torch.clamp(self.tau / max_logits, max=1.0)  # [num_heads]
        sqrt_gamma = torch.sqrt(gamma)
        
        # Determine head dimension
        if W_q.dim() == 2:
            # Shape: [num_heads * head_dim, embed_dim] (common layout)
            head_dim = W_q.shape[0] // num_heads
            
            for h in range(num_heads):
                if needs_clip[h]:
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    
                    # Scale both Q and K by sqrt(gamma) for this head
                    W_q.data[start_idx:end_idx] *= sqrt_gamma[h]
                    W_k.data[start_idx:end_idx] *= sqrt_gamma[h]
        
        # Track clip counts
        if layer_name not in self.clip_counts:
            self.clip_counts[layer_name] = 0
        self.clip_counts[layer_name] += num_clipped
        
        if self.log_clips:
            print(f"[QK-Clip] {layer_name}: clipped {num_clipped}/{num_heads} heads, "
                  f"max_logit={max_logit_value:.1f} -> ~{self.tau}")
        
        return num_clipped, max_logit_value
    
    def get_statistics(self) -> Dict:
        """Return clipping statistics for logging."""
        return {
            "clip_counts": self.clip_counts.copy(),
            "max_logits": {
                k: v[-1] if v else 0 
                for k, v in self.max_logits_history.items()
            }
        }


class AttentionWithQKClip(nn.Module):
    """
    Example attention module with QK-Clip integration.
    
    Shows how to integrate QK-Clip into a standard attention implementation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qk_clip: Optional[QKClip] = None,
        layer_name: str = "attn",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_name = layer_name
        
        # Projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # QK-Clip (optional)
        self.qk_clip = qk_clip
        
        # Store max logits from forward pass for post-optimizer clipping
        self.last_max_logits: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Track max logits for QK-Clip (detached, no grad needed)
        if self.qk_clip is not None and self.training:
            with torch.no_grad():
                self.last_max_logits = scores.abs().amax(dim=(0, 2, 3))  # [num_heads]
        
        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_o(out)
        
        return out
    
    def apply_qk_clip(self) -> Tuple[int, float]:
        """
        Apply QK-Clip after optimizer step.
        Call this after optimizer.step() in training loop.
        
        Returns:
            Tuple of (num_heads_clipped, max_logit_value)
        """
        if self.qk_clip is None or self.last_max_logits is None:
            return 0, 0.0
        
        return self.qk_clip.maybe_clip_weights(
            layer_name=self.layer_name,
            W_q=self.W_q.weight,
            W_k=self.W_k.weight,
            max_logits=self.last_max_logits,
            num_heads=self.num_heads,
        )
```

---

## Integration into Training Loop

```python
# Initialize QK-Clip
qk_clip = QKClip(tau=100.0, enabled=True, log_clips=True)

# Create model with QK-Clip enabled in attention layers
model = MyTransformer(qk_clip=qk_clip)

# Training loop
for step, batch in enumerate(dataloader):
    # Forward pass (attention layers store max_logits internally)
    loss = model(batch)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Apply QK-Clip AFTER optimizer step
    total_clipped = 0
    max_logit = 0
    for layer in model.layers:
        if hasattr(layer.attn, 'apply_qk_clip'):
            clipped, logit = layer.attn.apply_qk_clip()
            total_clipped += clipped
            max_logit = max(max_logit, logit)
    
    # Log periodically
    if step % 100 == 0:
        print(f"Step {step}: max_attn_logit={max_logit:.1f}, heads_clipped={total_clipped}")
```

---

## Configuration Options

```yaml
# Recommended config settings to expose
qk_clip:
  enabled: true                    # Toggle QK-Clip on/off
  tau: 100.0                       # Threshold for max attention logit
                                   # Kimi K2 used 100, but 30-50 also works
  log_clips: true                  # Log when clipping occurs
  log_frequency: 100               # How often to log statistics
  warmup_steps: 0                  # Optional: disable during warmup
```

---

## Important Implementation Notes

### 1. When to Apply

QK-Clip is applied **after** the optimizer step, not during. The sequence is:
1. Forward pass → track max logits
2. Backward pass → compute gradients
3. Optimizer step → update weights
4. **QK-Clip → scale Q/K weights if needed**

### 2. Per-Head Granularity

Clipping is done **per-head**, not globally. This is important because:
- Some heads may be stable while others aren't
- Global clipping would unnecessarily penalize well-behaved heads
- Kimi found only ~13% of heads ever triggered clipping

### 3. The sqrt Split

Both W_q and W_k are scaled by `sqrt(gamma)`, not `gamma`. This is because:
- Attention logit = Q @ K^T
- Scaling Q by sqrt(γ) and K by sqrt(γ) results in logit scaled by γ
- This preserves the relative contribution of Q and K

### 4. MLA (Multi-head Latent Attention) Considerations

Kimi K2 uses MLA which has shared vs per-head components:
- **Wq_c, Wk_c** (per-head context): Scale by sqrt(γ)
- **Wq_r** (per-head rotary query): Scale by γ
- **Wk_r** (shared rotary key): **Do NOT scale** - affects all heads

For standard MHA, you only need to worry about W_q and W_k per head.

### 5. Numerical Stability

```python
# Avoid division by zero
gamma = torch.clamp(self.tau / (max_logits + 1e-8), max=1.0)

# Ensure gamma doesn't go below some minimum (optional)
gamma = torch.clamp(gamma, min=0.1, max=1.0)
```

### 6. Interaction with QK-Norm

If you already use QK-Norm (normalizing Q and K before dot product):
- QK-Norm bounds the logits to [-1, 1] * seq_len approximately
- You may not need QK-Clip, but it can still help as a safety net
- They can coexist - QK-Norm is a forward-pass operation, QK-Clip is post-optimizer

### 7. Gradient Checkpointing

If using gradient checkpointing, make sure the max_logits are captured in the first forward pass (not the recomputed one):

```python
if not torch.is_grad_enabled():  # During recomputation
    return  # Don't overwrite stored max_logits
```

---

## Monitoring and Diagnostics

Add these to your training diagnostics:

```python
@dataclass
class QKClipDiagnostics:
    step: int
    max_logit_per_layer: Dict[str, float]
    heads_clipped_per_layer: Dict[str, int]
    total_heads_clipped: int
    clip_triggered: bool

# Log at each step
diagnostics = QKClipDiagnostics(
    step=step,
    max_logit_per_layer={name: qk_clip.max_logits_history[name][-1] 
                         for name in qk_clip.max_logits_history},
    heads_clipped_per_layer=qk_clip.clip_counts,
    total_heads_clipped=sum(qk_clip.clip_counts.values()),
    clip_triggered=any(v > 0 for v in qk_clip.clip_counts.values()),
)
```

### Warning Thresholds

| max_logit Range | Status | Action |
|-----------------|--------|--------|
| 10-30 | 🟢 Healthy | None |
| 30-50 | 🟡 Warm | Monitor more frequently |
| 50-80 | 🟠 Concerning | Consider lowering tau |
| 80-100 | 🔴 At threshold | QK-Clip should be triggering |
| 100+ | 💀 Problem | QK-Clip not working or tau too high |

---

## References

1. **Kimi K2 Technical Report**: https://arxiv.org/pdf/2507.20534
2. **Kimi K2 Blog Post**: https://moonshotai.github.io/Kimi-K2/
3. **Community Implementation**: https://github.com/kyegomez/MuonClip
4. **Fireworks Blog Explanation**: https://fireworks.ai/blog/muonclip

---

## Summary

QK-Clip is a simple but effective mechanism:
1. Track max attention logit per head during forward pass
2. After optimizer step, check if any head exceeded threshold τ
3. If so, scale W_q and W_k for that head by sqrt(τ / max_logit)
4. This bounds future logits to approximately τ

The key insight is targeting the **weights** (the cause) rather than the **logits** (the symptom), and doing so with surgical per-head precision rather than global intervention.
