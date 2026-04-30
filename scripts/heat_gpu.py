#!/usr/bin/env python3
"""
Heat up a specific GPU to identify it physically.
Run after reboot when all 8 GPUs are visible.
The target GPU will climb to 70-80°C while others stay idle at ~25°C.
Feel the backplates or use an IR thermometer to find the hot one.
"""
import torch
import time
import subprocess
import sys

TARGET_GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 6

print(f"=== HEATING GPU {TARGET_GPU} FOR PHYSICAL IDENTIFICATION ===")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Target: cuda:{TARGET_GPU} = {torch.cuda.get_device_name(TARGET_GPU)}")

# Show starting temps
print("\nStarting temperatures:")
subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,temperature.gpu", "--format=csv"])

print(f"\nPutting sustained load on GPU {TARGET_GPU}...")
print("All other GPUs should stay cool.")
print("Feel the backplates or use IR thermometer to find the hot card.")
print("Press Ctrl+C to stop.\n")

x = torch.randn(10000, 10000, device=f'cuda:{TARGET_GPU}')

try:
    step = 0
    while True:
        # Sustained matrix multiply to generate heat
        for _ in range(100):
            x = torch.mm(x, x)
            x = x / x.norm()  # prevent overflow
        
        step += 1
        if step % 5 == 0:
            print(f"\n[{step * 100} iterations] Current temperatures:")
            subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,temperature.gpu", "--format=csv"])
            
except KeyboardInterrupt:
    print("\n\nStopped. Final temperatures:")
    subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,temperature.gpu", "--format=csv"])
    print(f"\nThe hot card is GPU {TARGET_GPU}. Pull it or swap its riser.")