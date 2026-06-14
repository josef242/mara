#!/bin/bash
# 3090 low power control script

for gpu in {0..7}; do
    sudo nvidia-smi -i ${gpu} -pl 250
    echo "GPU $gpu: Power level set to 250w"
done
