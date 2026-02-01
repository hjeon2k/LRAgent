#!/bin/bash

# Compile script for Flash LoRA Attention test
# This script compiles the test_flash_fwd.cu file with all necessary dependencies

set -e  # Exit on error

# Configuration
CUDA_ARCH="86"  # RTX A6000 uses sm_86. Use 80 for A100, 89 for RTX 4090
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CUTLASS_DIR="../../cutlass/include"
CUTE_DIR="$CUTLASS_DIR/cute"
TORCH_DIR="/home/hyeongju.ha/anaconda3/envs/multillmagents/lib/python3.10/site-packages/torch"

# Parallelization settings
NUM_THREADS=$(nproc)  # Get number of CPU cores
USE_CCACHE=true  # Set to false if ccache is not available

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Flash LoRA Attention Test Compilation ==="
echo "CUDA Home: $CUDA_HOME"
echo "CUDA Architecture: sm_$CUDA_ARCH"
echo "Parallel threads: $NUM_THREADS"

# Check for ccache
CCACHE_PREFIX=""
if [ "$USE_CCACHE" = true ] && command -v ccache &> /dev/null; then
    CCACHE_PREFIX="ccache"
    echo -e "${GREEN}ccache: enabled${NC}"
else
    echo -e "${YELLOW}ccache: not available (install with: sudo apt-get install ccache)${NC}"
fi
echo ""

# Check if CUDA is available
if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo -e "${RED}Error: nvcc not found at $CUDA_HOME/bin/nvcc${NC}"
    echo "Please set CUDA_HOME environment variable"
    exit 1
fi

# Check CUTLASS
if [ ! -d "$CUTLASS_DIR" ]; then
    echo -e "${RED}Error: CUTLASS not found at $CUTLASS_DIR${NC}"
    echo "Expected directory structure: src/flash-attention/csrc/cutlass"
    exit 1
fi

# Source directory
SRC_DIR="../src"

# Include paths
INCLUDES=(
    "-I$SRC_DIR"
    "-I$CUTLASS_DIR"
    "-I$CUTE_DIR"
    "-I$CUDA_HOME/include"
    "-I$TORCH_DIR/include"
    "-I$TORCH_DIR/include/torch/csrc/api/include"
)

# Compiler flags
NVCC_FLAGS=(
    "-std=c++17"
    "-O3"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"
    "-Xcompiler=-fPIC"
    "-Xcompiler=-Wno-psabi"
    "-Xcompiler=-fno-strict-aliasing"
    "--generate-code=arch=compute_${CUDA_ARCH},code=[sm_${CUDA_ARCH},compute_${CUDA_ARCH}]"
    # Parallelization and optimization flags
    "--threads=$NUM_THREADS"  # Parallel compilation (CUDA 11.2+)
    "-Xptxas=-v"  # Verbose PTX assembler output
    "-Xptxas=-O3"  # PTX optimization level
    # Compilation speed optimizations
    "--keep-device-functions"  # Keep all device functions (faster compile)
    "-lineinfo"  # Line info for profiling (can be removed for faster compile)
)

# Library paths
LIBRARY_PATHS=(
    "-L$TORCH_DIR/lib"
    "-L$CUDA_HOME/lib64"
)

# Libraries to link
LIBRARIES=(
    "-lc10"
    "-lc10_cuda"
    "-ltorch"
    "-ltorch_cuda"
    "-lcudart"
)

# Source files to compile
SOURCE_FILES=(
    "test_flash_fwd.cu"
)

# Output binary
OUTPUT="test_flash_fwd"

echo "Compiling test..."
echo "Compiler: ${CCACHE_PREFIX:+$CCACHE_PREFIX }nvcc"
echo ""

# Start timer
START_TIME=$(date +%s)

# Compile with optional ccache
NVCC_CMD="$CUDA_HOME/bin/nvcc"
if [ -n "$CCACHE_PREFIX" ]; then
    export CCACHE_DIR="${HOME}/.ccache"
    export CCACHE_MAXSIZE="5G"
    NVCC_CMD="$CCACHE_PREFIX $NVCC_CMD"
fi

if eval $NVCC_CMD \
    "${NVCC_FLAGS[@]}" \
    "${INCLUDES[@]}" \
    "${SOURCE_FILES[@]}" \
    "${LIBRARY_PATHS[@]}" \
    "${LIBRARIES[@]}" \
    -o "$OUTPUT" \
    2>&1 | tee compilelog.txt; then
    
    END_TIME=$(date +%s)
    COMPILE_TIME=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}✓ Compilation successful!${NC}"
    echo "Binary: ./$OUTPUT"
    echo "Compilation time: ${COMPILE_TIME}s"
    
    # Show ccache stats if available
    if [ -n "$CCACHE_PREFIX" ]; then
        echo ""
        echo "=== ccache statistics ==="
        ccache -s | grep -E "cache hit|cache miss|files in cache"
    fi
    
    echo ""
    echo "Run with: ./$OUTPUT"
else
    echo ""
    echo -e "${RED}✗ Compilation failed${NC}"
    echo "Check compilelog.txt for details"
    exit 1
fi
