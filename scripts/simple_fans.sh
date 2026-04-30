#!/bin/bash
# Simple GPU Fan Control Script

# Set fan speeds for each GPU (0-7)
#        4  0  1  2  3  5  6  7
SPEEDS=(90 75 80 75 80 90 90 90)

echo "Setting GPU fan speeds..."

# Check if X server is already running on :99
if ! xdpyinfo -display :99 >/dev/null 2>&1; then
    echo "Starting X server on display :99..."
    sudo X :99 -ac -noreset &
    sleep 3
else
    echo "X server already running on :99"
fi

# Set display
export DISPLAY=:99

# Enable fan control for all 8 GPUs
for gpu in {0..7}; do
    sudo DISPLAY=:99 nvidia-settings -a "[gpu:${gpu}]/GPUFanControlState=1" 2>/dev/null
    echo "GPU $gpu: Fan control enabled"
done

# Set fan speeds (each GPU has 2 fans)
for gpu in {0..7}; do
    fan1=$((gpu * 2))
    fan2=$((gpu * 2 + 1))
    speed=${SPEEDS[$gpu]}
    
    sudo DISPLAY=:99 nvidia-settings -a "[fan:${fan1}]/GPUTargetFanSpeed=${speed}" 2>/dev/null
    sudo DISPLAY=:99 nvidia-settings -a "[fan:${fan2}]/GPUTargetFanSpeed=${speed}" 2>/dev/null
    echo "GPU $gpu: Fans set to ${speed}%"
done

echo ""
echo "Waiting for 5 seconds to stabilize..."
sleep 5
echo "Done! Current status:"
nvidia-smi --query-gpu=index,name,fan.speed,temperature.gpu --format=csv

sleep 5
sudo killall Xorg
