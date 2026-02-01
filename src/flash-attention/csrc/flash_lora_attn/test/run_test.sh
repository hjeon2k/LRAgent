#!/bin/bash

# Run script for Flash LoRA Attention test
# Sets up library paths before running the test

# Select GPU (change this to use a different GPU)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

TORCH_DIR="/home/hyeongju.ha/anaconda3/envs/multillmagents/lib/python3.10/site-packages/torch"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# Add PyTorch and CUDA libraries to library path
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "=== Running Flash LoRA Attention Test ==="
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Library path: $LD_LIBRARY_PATH"
echo ""

# Run the test
# ./test_flash_fwd > test_output.txt
compute-sanitizer ./test_flash_fwd > test_output.txt
# ./test_flash_fwd_origin >> test_output.txt