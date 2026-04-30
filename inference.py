"""
Minimal single-GPU inference script for Mara FSDP2 checkpoints.

Usage:
    python inference.py --checkpoint path/to/model_step_000100.pt --prompt "Once upon a time"
    python inference.py --checkpoint path/to/model_step_000100.pt --prompt "Hello" --max_tokens 256 --temperature 0.8

Notes:
    - Checkpoints saved with EP > 1 (Expert Parallel) require the updated save_model
      that consolidates expert params from all ranks.  Old EP checkpoints (before
      consolidation was added) will fail to load — re-save from training to fix.
    - Tokenizer paths in the checkpoint are relative to the training script dir
      (mara_fsdp2/).  Use --tok_path to override if needed.
"""

import sys
import os
import argparse
import dataclasses

import torch
import torch.nn.functional as F

# Add common_fsdp2 to path for model + tokenizer imports
common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common_fsdp2')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

from model_v2 import Transformer, ModelArgs
from tokenizer_abstraction import get_tokenizer


def load_checkpoint(path: str):
    """Load a Mara FSDP2 checkpoint and return (state_dict, config_dict, metadata)."""
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    state_dict = ckpt["model"]
    metadata = {
        "step": ckpt.get("step", 0),
        "tok_kind": ckpt.get("tok_kind", "llama"),
        "tok_path": ckpt.get("tok_path"),
        "special_tokens": ckpt.get("special_tokens"),
        "checkpoint_version": ckpt.get("checkpoint_version", "unknown"),
        "total_tokens": ckpt.get("total_tokens_processed", 0),
    }

    print(f"  Step: {metadata['step']}, Version: {metadata['checkpoint_version']}")
    print(f"  Config: {config['n_layers']}L, dim={config['dim']}, heads={config['n_heads']}, vocab={config['vocab_size']}")
    if config.get("moe_enabled", False):
        ep = config.get("ep_degree", 1)
        print(f"  MoE: {config['moe_num_experts']} experts, top-{config['moe_top_k']}, ep_degree={ep}")

    # EP handling: main checkpoint only has rank 0's local experts.
    # The consolidated ep_experts file (saved alongside) has ALL experts.
    ep_degree = config.get("ep_degree", 1)
    if ep_degree > 1:
        # Look for consolidated EP experts file next to the checkpoint
        ckpt_dir = os.path.dirname(path)
        step = metadata["step"]
        ep_path = os.path.join(ckpt_dir, f"ep_experts_step_{step:06d}.pt")
        if os.path.exists(ep_path):
            print(f"  EP checkpoint (ep_degree={ep_degree}): loading consolidated experts from {os.path.basename(ep_path)}")
            ep_experts = torch.load(ep_path, map_location="cpu", weights_only=True)
            # Overlay consolidated expert params into state dict
            for key, val in ep_experts.items():
                state_dict[key] = val
            del ep_experts
            print(f"  Overlaid {sum(1 for k in state_dict if '.moe.experts.' in k)} expert params")
        else:
            print(f"\n  ERROR: Checkpoint has ep_degree={ep_degree} — expert params are incomplete.")
            print(f"  Missing consolidated file: {os.path.basename(ep_path)}")
            print(f"  Re-save from training with the updated save_model to fix.")
            sys.exit(1)

    return state_dict, config, metadata


def build_model(config: dict) -> Transformer:
    """Build a Transformer from a checkpoint config dict."""
    config = dict(config)
    config["ep_degree"] = 1
    config["use_activation_checkpointing"] = False
    config["dropout"] = 0.0

    # Filter to known ModelArgs fields (future-proof against new config keys)
    known_fields = {f.name for f in dataclasses.fields(ModelArgs)}
    config = {k: v for k, v in config.items() if k in known_fields}

    args = ModelArgs(**config)
    model = Transformer(args)
    return model


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: list[int],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_id: int | None = None,
) -> list[int]:
    """Autoregressive generation with KV caching."""
    device = next(model.parameters()).device
    max_seq = model.params.max_seq_len

    if len(prompt_tokens) > max_seq - 1:
        print(f"  Prompt truncated from {len(prompt_tokens)} to {max_seq - 1} tokens")
        prompt_tokens = prompt_tokens[-(max_seq - 1):]

    model.setup_caches(max_batch_size=1, max_seq_len=max_seq)

    # Prefill
    tokens_t = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    logits = model.generate_forward(tokens_t, start_pos=0)

    generated = list(prompt_tokens)
    pos = len(prompt_tokens)

    for _ in range(max_new_tokens):
        if pos >= max_seq:
            break

        next_token = _sample(logits[0, -1, :], temperature, top_k, top_p)
        generated.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

        next_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits = model.generate_forward(next_t, start_pos=pos)
        pos += 1

    model.clear_caches()
    return generated


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample a single token from logits. All math in float32 for stability."""
    logits = logits.float()

    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k: keep only the top_k highest logits
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        logits[logits < topk_vals[-1]] = float('-inf')

    # Top-p (nucleus): keep smallest set of tokens whose cumulative prob >= top_p
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Shift right so the token that crosses the threshold is kept
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove] = float('-inf')
        # Scatter back to original positions
        logits = torch.zeros_like(logits).scatter(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def main():
    parser = argparse.ArgumentParser(description="Mara FSDP2 single-GPU inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model_step_NNNNNN.pt")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--tok_path", default=None, help="Override tokenizer path")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load checkpoint
    state_dict, config, meta = load_checkpoint(args.checkpoint)

    # Build model and load weights
    model = build_model(config)
    try:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            # Filter out non-persistent buffers (expected to be missing)
            real_missing = [k for k in result.missing_keys
                           if not k.endswith('.tokens_per_expert')]
            if real_missing:
                print(f"  WARNING: Missing keys: {real_missing}")
        if result.unexpected_keys:
            print(f"  WARNING: Unexpected keys: {result.unexpected_keys}")
    except RuntimeError as e:
        print(f"\n  ERROR loading state dict: {e}")
        print(f"  This may indicate an EP checkpoint that wasn't consolidated.")
        print(f"  Re-save from training with the updated save_model to fix.")
        sys.exit(1)
    del state_dict

    model = model.to(device=args.device, dtype=dtype)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded on {args.device} ({args.dtype}), {total_params:,} params")

    # Load tokenizer — stored paths are relative to the training script dir (mara_fsdp2/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = args.tok_path or meta["tok_path"]
    if tok_path and not os.path.isabs(tok_path):
        tok_path = os.path.join(script_dir, tok_path)
    special_tokens = meta["special_tokens"]
    if special_tokens and not os.path.isabs(special_tokens):
        special_tokens = os.path.join(script_dir, special_tokens)
    enc = get_tokenizer(meta["tok_kind"], path=tok_path, special_tokens=special_tokens)
    print(f"Tokenizer: {meta['tok_kind']}, vocab={len(enc)}")

    # Encode and generate (BOS expected by LLaMA-family models)
    prompt_tokens = enc.encode(args.prompt, bos=True)
    if not prompt_tokens:
        print("ERROR: Empty prompt — nothing to generate from.")
        sys.exit(1)
    print(f"\nPrompt ({len(prompt_tokens)} tokens): {args.prompt}")
    print("-" * 60)

    output_tokens = generate(
        model, prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=enc.eos_id,
    )

    # Decode: strip BOS to avoid SentencePiece leading-space artifact.
    # SentencePiece uses ▁ (U+2581) for word-boundary spaces and strips the
    # leading ▁ from the first piece in a decode call.  When BOS is the first
    # token, it produces no text but "consumes" the first-piece status, so the
    # next real token's ▁ becomes an unwanted leading space.
    decode_tokens = [t for t in output_tokens if t != enc.bos_id]
    output_text = enc.decode(decode_tokens)
    print(output_text)
    print("-" * 60)
    new_tokens = len(output_tokens) - len(prompt_tokens)
    print(f"Generated {new_tokens} new tokens")


if __name__ == "__main__":
    main()
