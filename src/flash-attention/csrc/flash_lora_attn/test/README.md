# Flash LoRA Attention Test

This directory contains functionality tests for the Flash LoRA Attention forward pass.

## Test Configuration

- **Head dimension**: 128
- **Sequence length**: 1024 (both Q and KV)
- **Batch size**: 1
- **Query heads**: 32
- **KV heads**: 8 (Grouped Query Attention)
- **LoRA dimension**: 8
- **Block size**: 128x128
- **Warps**: 4

## Files

- `test_flash_fwd.cu` - Main test code for `run_flash_fwd` function
- `compile.sh` - Build script for compiling the test
- `README.md` - This file

## Building

### Prerequisites

- CUDA Toolkit (>= 11.8)
- GPU with compute capability >= 8.0 (Ampere or later)
- CUTLASS library (included in flash-attention)

### Compilation

```bash
cd /path/to/flash-attention/csrc/flash_lora_attn/test
./compile.sh
```

The script will:
1. Check for CUDA installation
2. Verify CUTLASS headers
3. Compile with appropriate flags for your architecture
4. Generate `test_flash_fwd` binary

### Architecture Configuration

By default, the script compiles for sm_80 (A100, RTX 3090). To change:

```bash
# Edit compile.sh and change CUDA_ARCH variable
CUDA_ARCH="86"  # For RTX 3080
CUDA_ARCH="89"  # For RTX 4090
```

## Running

```bash
./test_flash_fwd
```

### Expected Output

```
Using GPU: <GPU Name>
Compute Capability: 8.x

=== Flash LoRA Attention Forward Test ===
Config:
  Batch size: 1
  Q heads: 32
  KV heads: 8 (GQA)
  Sequence length (Q): 1024
  Sequence length (KV): 1024
  Head dimension: 128
  LoRA dimension: 8

Initializing tensors with random data...
Running Flash Attention forward kernel...
✓ Kernel execution successful!

=== Output Statistics ===
✓ Output is valid (no NaN/Inf)
  Max absolute value: <value>
  Min absolute value: <value>
  Mean value: <value>

=== Test Complete ===
```

## What the Test Does

1. **Allocates GPU memory** for Q, K, V, O, LoRA_V, LoRA_B tensors
2. **Initializes tensors** with random normal distribution (mean=0, std=0.02)
3. **Sets up Flash_fwd_params** with proper strides and dimensions
4. **Launches run_flash_fwd** template function with:
   - Kernel_traits: `Flash_fwd_kernel_traits<128, 128, 128, 4, 8, false, false, half>`
   - Is_dropout: false
   - Is_causal: false
5. **Validates output** by checking for NaN/Inf and computing statistics

## Troubleshooting

### Compilation Errors

**CUTLASS not found:**
```bash
# Ensure CUTLASS is in the correct location
ls ../../../../include/cutlass
```

**CUDA not found:**
```bash
# Set CUDA_HOME if not in default location
export CUDA_HOME=/usr/local/cuda-12.0
./compile.sh
```

### Runtime Errors

**Invalid architecture:**
- Check that your GPU has compute capability >= 8.0
- Verify CUDA_ARCH in compile.sh matches your GPU

**Out of memory:**
- Reduce batch size or sequence length in test_flash_fwd.cu
- Current config uses ~200MB GPU memory

**Kernel launch failed:**
- Check `dmesg` for GPU errors
- Verify CUDA driver version matches toolkit version

## Extending the Test

To test different configurations, modify `test_flash_fwd.cu`:

```cpp
// Change dimensions
constexpr int kHeadDim = 64;      // 64, 128, 256
constexpr int kLoRADim = 16;      // Must be multiple of 8
constexpr int seqlen_q = 2048;    // Any length
constexpr int num_heads_q = 16;   // Must be divisible by num_heads_kv

// Enable causal masking
constexpr bool Is_causal = true;

// Enable dropout
constexpr bool Is_dropout = true;
params.p_dropout = 0.1f;
```

## Performance Notes

This is a **functionality test**, not a benchmark. For performance testing:
- Use longer sequences (4096+)
- Multiple iterations with warm-up
- Compare against baseline Flash Attention
- Profile with `nsys` or `ncu`
