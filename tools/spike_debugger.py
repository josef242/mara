# spike_debugger.py
import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle

class SpikeDebugger:
    def __init__(self, 
                 output_dir: str,
                 threshold: float = 3.0,
                 ddp_rank: int = 0,
                 tokenizer = None,
                 max_samples_per_file: int = 100):
        """
        Debug gradient norm spikes by capturing problematic batches.
        
        Args:
            output_dir: Directory to save debug files
            threshold: Gradient norm threshold to trigger capture
            ddp_rank: DDP rank (only rank 0 saves)
            tokenizer: Tokenizer for decoding samples
            max_samples_per_file: Max spike samples before rotating file
        """
        self.output_dir = output_dir
        self.threshold = threshold
        self.ddp_rank = ddp_rank
        self.tokenizer = tokenizer
        self.max_samples = max_samples_per_file
        
        self.spike_count = 0
        self.spike_history = []
        self.current_file_idx = 0
        
        # Create debug directory
        if self.ddp_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Track statistics
        self.stats = {
            'total_steps': 0,
            'total_spikes': 0,
            'spikes_by_group': {},
            'spikes_by_shard': {},
            'spike_magnitudes': [],
            'normal_norms': []
        }
        
    def check_and_capture(self,
                         step: int,
                         norm: float,
                         loss: float,
                         lr: float,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train_loader,
                         grad_accum_steps: int) -> bool:
        """
        Check if norm exceeds threshold and capture debug info.
        Returns True if spike detected.
        """
        self.stats['total_steps'] += 1
        
        if norm < self.threshold:
            self.stats['normal_norms'].append(float(norm))
            return False
            
        # Spike detected!
        self.stats['total_spikes'] += 1
        self.stats['spike_magnitudes'].append(float(norm))
        
        if self.ddp_rank != 0:
            return True  # Only rank 0 saves
            
        # Get current shard info
        current_shard = train_loader.current_shard_name()
        current_group = train_loader.groups[train_loader._cur_group_idx].name
        current_position = train_loader._cur_tok_pos
        
        # Update group/shard statistics
        self.stats['spikes_by_group'][current_group] = \
            self.stats['spikes_by_group'].get(current_group, 0) + 1
        self.stats['spikes_by_shard'][current_shard] = \
            self.stats['spikes_by_shard'].get(current_shard, 0) + 1
        
        # Prepare spike data
        spike_data = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'norm': float(norm),
            'loss': float(loss),
            'lr': float(lr),
            'grad_accum_steps': grad_accum_steps,
            'shard': current_shard,
            'group': current_group,
            'shard_position': current_position,
            'batch_shape': list(x.shape),
            'tokens': {
                'x': x.cpu().numpy().tolist(),  # Convert to list for JSON
                'y': y.cpu().numpy().tolist()
            }
        }
        
        # Decode samples if tokenizer available
        if self.tokenizer is not None:
            spike_data['decoded_samples'] = self._decode_samples(x, y)
            
        # Save to current file
        self._save_spike(spike_data)
        
        # Log summary
        print(f"\n🚨 SPIKE DETECTED at step {step}!")
        print(f"   Norm: {norm:.4f} (threshold: {self.threshold})")
        print(f"   Loss: {loss:.6f}, LR: {lr:.2e}")
        print(f"   Group: {current_group}, Shard: {current_shard}")
        print(f"   Position in shard: {current_position:,}")
        
        if 'decoded_samples' in spike_data:
            print(f"   First sequence preview: {spike_data['decoded_samples'][0]['preview']}")
        
        return True
        
    def _decode_samples(self, x: torch.Tensor, y: torch.Tensor, 
                       num_samples: int = 3, preview_len: int = 200) -> List[Dict]:
        """Decode a few samples from the batch for inspection."""
        samples = []
        num_to_decode = min(num_samples, x.shape[0])
        
        for i in range(num_to_decode):
            try:
                # Get tokens for this sequence
                x_tokens = x[i].cpu().numpy()
                y_tokens = y[i].cpu().numpy()
                
                # Decode input and target
                x_text = self.tokenizer.decode(x_tokens)
                y_text = self.tokenizer.decode(y_tokens)
                
                # Create preview (first N chars)
                preview = x_text[:preview_len]
                if len(x_text) > preview_len:
                    preview += "..."
                    
                samples.append({
                    'sequence_idx': i,
                    'input_text': x_text,
                    'target_text': y_text,
                    'preview': preview,
                    'token_count': len(x_tokens)
                })
            except Exception as e:
                samples.append({
                    'sequence_idx': i,
                    'error': f"Failed to decode: {str(e)}",
                    'token_count': len(x[i])
                })
                
        return samples
        
    def _save_spike(self, spike_data: Dict[str, Any]):
        """Save spike data to file."""
        # Rotate files if needed
        if self.spike_count >= self.max_samples:
            self.current_file_idx += 1
            self.spike_count = 0
            self.spike_history = []
            
        self.spike_history.append(spike_data)
        self.spike_count += 1
        
        # Save as JSON
        json_path = os.path.join(
            self.output_dir, 
            f'spikes_batch_{self.current_file_idx:03d}.json'
        )
        with open(json_path, 'w') as f:
            json.dump(self.spike_history, f, indent=2)
            
        # Also save raw tensors as pickle for exact reproduction
        pickle_path = os.path.join(
            self.output_dir,
            f'spike_tensors_{spike_data["step"]:06d}.pkl'
        )
        tensor_data = {
            'step': spike_data['step'],
            'x': torch.tensor(spike_data['tokens']['x']),
            'y': torch.tensor(spike_data['tokens']['y']),
            'metadata': {
                'norm': spike_data['norm'],
                'shard': spike_data['shard'],
                'group': spike_data['group']
            }
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump(tensor_data, f)
            
    def save_summary(self):
        """Save summary statistics."""
        if self.ddp_rank != 0:
            return
            
        # Calculate statistics
        if self.stats['normal_norms']:
            avg_normal = np.mean(self.stats['normal_norms'])
            std_normal = np.std(self.stats['normal_norms'])
        else:
            avg_normal = std_normal = 0.0
            
        if self.stats['spike_magnitudes']:
            avg_spike = np.mean(self.stats['spike_magnitudes'])
            max_spike = np.max(self.stats['spike_magnitudes'])
        else:
            avg_spike = max_spike = 0.0
            
        summary = {
            'total_steps': self.stats['total_steps'],
            'total_spikes': self.stats['total_spikes'],
            'spike_rate': self.stats['total_spikes'] / max(1, self.stats['total_steps']),
            'threshold': self.threshold,
            'normal_norm_stats': {
                'mean': avg_normal,
                'std': std_normal,
                'count': len(self.stats['normal_norms'])
            },
            'spike_stats': {
                'mean': avg_spike,
                'max': max_spike,
                'count': len(self.stats['spike_magnitudes'])
            },
            'spikes_by_group': self.stats['spikes_by_group'],
            'spikes_by_shard': self.stats['spikes_by_shard']
        }
        
        summary_path = os.path.join(self.output_dir, 'spike_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📊 Spike Summary saved to {summary_path}")
        print(f"   Total spikes: {summary['total_spikes']} / {summary['total_steps']} steps")
        print(f"   Spike rate: {summary['spike_rate']:.2%}")
        if self.stats['spikes_by_group']:
            print("   Spikes by group:")
            for group, count in sorted(self.stats['spikes_by_group'].items()):
                print(f"     - {group}: {count}")

    """
    Notes for integration:

    1. At top of train_mara.py:
    from spike_debugger import SpikeDebugger


    2. Then modify your train_loop function:

    def train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_world_size, start_step, 
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule
    ):
    
    # Initialize spike debugger
    debug_dir = os.path.join(logger._instance.get_dir(), "spike_debug")
    spike_debugger = SpikeDebugger(
        output_dir=debug_dir,
        threshold=3.0,  # Adjust based on your observations
        ddp_rank=ddp_rank,
        tokenizer=enc  # Pass the tokenizer for decoding
    )

    # Do a baseline validation before training
    if start_step == 1:
        do_validation(model, val_loader, device, settings.eval_iters, 0, ddp_rank, settings.val_log_file, total_tokens_processed, ddp, ddp_world_size, settings.data_type, device_type)

    for step in range(start_step, settings.max_steps):
        t0 = time.time()
        last_step = (step == settings.max_steps - 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        grad_accum_steps = grad_accum_schedule[step]
        
        # Store batches for potential debugging
        batch_x_list = []
        batch_y_list = []
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # Store for debugging (only first micro-batch for memory efficiency)
            if micro_step == 0:
                batch_x_list.append(x)
                batch_y_list.append(y)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if settings.data_type=="bf16" else torch.float16 if settings.data_type=="fp16" else torch.float32):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        lr = get_lr(step, settings.max_lr, settings.min_lr, settings.warmup_steps,settings.max_steps, settings.restart_steps, settings.restart_gamma)
        clip_value = get_clip_value(step, lr, settings)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # Check for spikes and capture debug info
        if batch_x_list and batch_y_list:
            spike_detected = spike_debugger.check_and_capture(
                step=step,
                norm=norm.item(),
                loss=loss_accum.item(),
                lr=lr,
                x=batch_x_list[0],  # Use first micro-batch
                y=batch_y_list[0],
                train_loader=train_loader,
                grad_accum_steps=grad_accum_steps
            )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        # Rest of your training loop remains the same...
        if ddp_rank == 0:
            dt = time.time() - t0
            tokens_per_step = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size)
            total_tokens_processed += tokens_per_step
            tokens_per_sec = tokens_per_step / dt

            mfu = compute_mfu(tokens_per_sec, flops_per_token, ddp_world_size, settings.data_type) * 100
        
            ppl = math.exp(loss_accum.item())
            logger.print_and_log(
                f"st: {step:5d} | ls: {loss_accum.item():.6f} | ppl: {ppl:.2f} | lr: {lr:.4e} | nrm: {norm:.4f} | dt: {dt*1000:.2f}ms | t_tk: {total_tokens_processed:11d} | tok/s: {tokens_per_sec:.2f} | MFU: {mfu:.0f}%",
            )

            logger.print_and_log(
                f"{step:5d}|{loss_accum.item():.6f}|{ppl:.2f}|{lr:.4e}|{norm:.4f}|{dt*1000:.2f}|{total_tokens_processed:11d}|{tokens_per_sec:.2f}",
                True, settings.train_log_file, silent=True
            )

        # ... rest of training loop ...
        
    # Save summary at the end
    spike_debugger.save_summary()


    3. Make sure to pass the tokenizer to train_loop:
    In your main function, modify the train_loop call:

    # Make enc accessible to train_loop by passing it
    train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_world_size, start_step, 
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule, enc  # Add enc parameter
    )

    And update the train_loop signature to accept it:
    def train_loop(
        model, optimizer, train_loader, val_loader, device, ddp, ddp_rank, ddp_world_size, start_step, 
        total_tokens_processed, model_cfg, flops_per_token, settings, device_type, grad_accum_schedule, enc  # Add enc
    ):

    4. Analysis tools:
    Once you have some spike data, you can analyze it:

    # analyze_spikes.py
    import json
    import glob

    def analyze_spike_patterns(debug_dir):
        # Load all spike files
        spike_files = glob.glob(f"{debug_dir}/spikes_batch_*.json")
        all_spikes = []
        
        for file in spike_files:
            with open(file, 'r') as f:
                all_spikes.extend(json.load(f))
        
        # Analyze patterns
        print(f"Total spikes captured: {len(all_spikes)}")
        
        # Group analysis
        by_group = {}
        for spike in all_spikes:
            group = spike['group']
            by_group[group] = by_group.get(group, [])
            by_group[group].append(spike['norm'])
        
        print("\nSpikes by group:")
        for group, norms in by_group.items():
            print(f"  {group}: {len(norms)} spikes, max norm: {max(norms):.2f}")
        
        # Look for common patterns in decoded text
        if all_spikes and 'decoded_samples' in all_spikes[0]:
            print("\nSample previews from spikes:")
            for spike in all_spikes[:5]:  # First 5
                print(f"\nStep {spike['step']} (norm={spike['norm']:.2f}):")
                print(f"  {spike['decoded_samples'][0]['preview']}")
    """