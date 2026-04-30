#!/bin/bash

# Training environment setup with configurable conda environment

# Set conda environment name - use first argument or default to "trainenv"
CONDA_ENV="trainenv"
SESSION_NAME="test"

echo "Setting up environment: $CONDA_ENV"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if tmux session exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Attaching to existing '$SESSION_NAME' session..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session '$SESSION_NAME' with conda env '$CONDA_ENV'..."
    
    # Create new detached tmux session
    tmux new-session -d -s $SESSION_NAME
    
    # Send commands to the tmux session
    tmux send-keys -t $SESSION_NAME "eval \"\$(conda shell.bash hook)\"" Enter
    tmux send-keys -t $SESSION_NAME "conda deactivate" Enter
    tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" Enter
    sleep 0.5   # Let conda finish
    tmux send-keys -t $SESSION_NAME "cd ~/valhalla/code/mara" Enter
    tmux send-keys -t $SESSION_NAME "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" Enter
    tmux send-keys -t $SESSION_NAME "clear" Enter
    tmux send-keys -t $SESSION_NAME "echo '===================================='" Enter
    tmux send-keys -t $SESSION_NAME "echo 'Training environment ready!'" Enter
    tmux send-keys -t $SESSION_NAME "echo \"Conda env: \$CONDA_DEFAULT_ENV\"" Enter
    tmux send-keys -t $SESSION_NAME "echo \"Directory: \$(pwd)\"" Enter
    tmux send-keys -t $SESSION_NAME "echo '===================================='" Enter
    tmux send-keys -t $SESSION_NAME "echo 'VERIFY SWAP! command: sudo swapon ~/swapfile'" Enter
    tmux send-keys -t $SESSION_NAME "echo 'SET FANS SPEEDS! command: sudo ./simple_fans.sh'" Enter
    
    # Wait a moment for commands to execute
    sleep 1
    
    # Attach to the session
    tmux attach-session -t $SESSION_NAME
fi
