/******************************************************************************
 * Test for Flash LoRA Attention Forward Pass
 * 
 * Configuration:
 * - Head dim: 128
 * - KV length: 1024
 * - Batch: 1
 * - Query heads: 32
 * - KV heads: 8 (GQA)
 * - LoRA dim: 8
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

// Flash Attention includes
#include "../src/flash.h"
#include "../src/kernel_traits.h"
#include "../src/flash_fwd_launch_template.h"
#include "../src/static_switch.h"

// Avoid redefinition of CHECK_CUDA
#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

template<typename T>
void fill_random(T* data, size_t size, float scale = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, scale);
    
    std::vector<float> host_data(size);
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = dist(gen);
    }
    
    std::vector<T> host_data_converted(size);
    for (size_t i = 0; i < size; ++i) {
        host_data_converted[i] = static_cast<T>(host_data[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(data, host_data_converted.data(), 
                          size * sizeof(T), cudaMemcpyHostToDevice));
}

void test_flash_splitkv_fwd() {
    using Element = cutlass::half_t;
    constexpr int kHeadDim = 128;
    constexpr int kBlockM = 64;  // SplitKV uses 64
    constexpr int kBlockN = 128; // Standard for kHeadDim=128
    constexpr int kNWarps = 4;
    constexpr int kLoRADim = 8;
    constexpr int batch_size = 1;
    constexpr int num_heads_q = 32;
    constexpr int num_heads_kv = 8;
    constexpr int seqlen_q = 1;     // Decoding scenario
    constexpr int seqlen_k = 1020;
    constexpr int num_splits = 4;   // Split KV into 4 parts
    
    std::cout << "=== Flash LoRA Attention SplitKV Forward Test ===" << std::endl;
    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Q heads: " << num_heads_q << std::endl;
    std::cout << "  KV heads: " << num_heads_kv << " (GQA)" << std::endl;
    std::cout << "  Sequence length (Q): " << seqlen_q << " (Decoding)" << std::endl;
    std::cout << "  Sequence length (KV): " << seqlen_k << std::endl;
    std::cout << "  kBlockM: " << kBlockM << std::endl;
    std::cout << "  kBlockN: " << kBlockN << std::endl;
    std::cout << "  Head dimension: " << kHeadDim << std::endl;
    std::cout << "  LoRA dimension: " << kLoRADim << std::endl;
    std::cout << "  Num splits: " << num_splits << std::endl;
    std::cout << std::endl;
    
    // Calculate sizes
    const size_t q_size = batch_size * seqlen_q * num_heads_q * kHeadDim;
    const size_t k_size = batch_size * seqlen_k * num_heads_kv * kHeadDim;
    const size_t v_size = batch_size * seqlen_k * num_heads_kv * kHeadDim;
    const size_t o_size = batch_size * seqlen_q * num_heads_q * kHeadDim;
    const size_t lora_v_size = batch_size * seqlen_k * kLoRADim;
    const size_t lora_b_size = num_heads_kv * kLoRADim * kHeadDim;
    
    // SplitKV accumulator sizes
    const size_t oaccum_size = num_splits * batch_size * num_heads_q * seqlen_q * kHeadDim;
    const size_t lseaccum_size = num_splits * batch_size * num_heads_q * seqlen_q;
    
    // Allocate device memory
    Element *d_q, *d_k, *d_v, *d_o;
    Element *d_lora_v, *d_lora_b;
    float *d_softmax_lse, *d_softmax_lseaccum, *d_oaccum;
    
    CHECK_CUDA(cudaMalloc(&d_q, q_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_k, k_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_v, v_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_o, o_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_lora_v, lora_v_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_lora_b, lora_b_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_softmax_lse, batch_size * num_heads_q * seqlen_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_lseaccum, lseaccum_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_oaccum, oaccum_size * sizeof(float)));
    
    // Initialize with random data
    std::cout << "Initializing tensors with random data..." << std::endl;
    fill_random(d_q, q_size, 0.02f);
    fill_random(d_k, k_size, 0.02f);
    fill_random(d_v, v_size, 0.02f);
    fill_random(d_lora_v, lora_v_size, 0.02f);
    fill_random(d_lora_b, lora_b_size, 0.02f);
    CHECK_CUDA(cudaMemset(d_o, 0, o_size * sizeof(Element)));
    CHECK_CUDA(cudaMemset(d_softmax_lse, 0, batch_size * num_heads_q * seqlen_q * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_softmax_lseaccum, 0, lseaccum_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_oaccum, 0, oaccum_size * sizeof(float)));
    
    // Setup Flash_fwd_params - zero initialize to avoid garbage values
    FLASH_NAMESPACE::Flash_fwd_params params;
    memset(&params, 0, sizeof(params));
    
    // QKV pointers
    params.q_ptr = d_q;
    params.k_ptr = d_k;
    params.v_ptr = d_v;
    params.o_ptr = d_o;
    
    // LoRA pointers
    params.lora_v_ptr = d_lora_v;
    params.lora_b_ptr = d_lora_b;
    
    // Strides (assuming contiguous layout: [batch, seqlen, heads, head_dim])
    params.q_batch_stride = seqlen_q * num_heads_q * kHeadDim;
    params.q_row_stride = num_heads_q * kHeadDim;
    params.q_head_stride = kHeadDim;
    
    params.k_batch_stride = seqlen_k * num_heads_kv * kHeadDim;
    params.k_row_stride = num_heads_kv * kHeadDim;
    params.k_head_stride = kHeadDim;
    
    params.v_batch_stride = seqlen_k * num_heads_kv * kHeadDim;
    params.v_row_stride = num_heads_kv * kHeadDim;
    params.v_head_stride = kHeadDim;
    
    params.o_batch_stride = seqlen_q * num_heads_q * kHeadDim;
    params.o_row_stride = num_heads_q * kHeadDim;
    params.o_head_stride = kHeadDim;
    
    // LoRA strides
    params.lora_v_batch_stride = seqlen_k * kLoRADim;
    params.lora_v_row_stride = kLoRADim;
    
    params.lora_b_head_stride = kLoRADim * kHeadDim;
    params.lora_b_row_stride = kHeadDim;
    
    // Dimensions
    params.b = batch_size;
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = ((seqlen_q + kBlockM - 1) / kBlockM) * kBlockM;
    params.seqlen_k_rounded = ((seqlen_k + kBlockN - 1) / kBlockN) * kBlockN;
    params.d = kHeadDim;
    params.d_rounded = kHeadDim;
    params.lora_dim = kLoRADim;
    
    // Softmax scale
    params.scale_softmax = 1.0f / std::sqrt(float(kHeadDim));
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;  // log2(e) for exp2 computation
    
    // LSE pointers (splitKV specific)
    params.softmax_lse_ptr = d_softmax_lse;
    params.softmax_lseaccum_ptr = d_softmax_lseaccum;
    params.oaccum_ptr = d_oaccum;
    
    // SplitKV parameters
    params.num_splits = num_splits;
    params.total_q = batch_size * seqlen_q;
    params.is_seqlens_k_cumulative = true;
    
    // No special features for this test
    params.p_ptr = nullptr;  // No softmax output
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.leftpad_k = nullptr;
    params.blockmask = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;
    params.cache_batch_idx = nullptr;  // Add this
    params.block_table = nullptr;  // Add this
    params.block_table_batch_stride = 0;  // Add this
    params.page_block_size = 1;  // Add this (set to 1 to avoid division by zero)
    params.p_dropout = 0.0f;  // No dropout
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_bf16 = false;
    params.is_causal = false;  // Non-causal attention
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;
    params.rotary_dim = 0;
    params.is_rotary_interleaved = false;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    std::cout << "Running Flash Attention SplitKV forward kernel..." << std::endl;
    
    // Run the kernel
    using Kernel_traits_split = Flash_fwd_kernel_traits<
        kHeadDim, kBlockM, kBlockN, kNWarps, kLoRADim, false, false, Element>;
    
    constexpr bool Is_causal = false;
    
    try {
        // Start timing
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        FLASH_NAMESPACE::run_flash_splitkv_fwd<Kernel_traits_split, Is_causal>(params, stream);
        
        // Stop timing
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "✓ Kernel execution successful!" << std::endl;
        std::cout << "  Execution time: " << milliseconds << " ms" << std::endl;
        
        // Calculate performance metrics
        const double flops = 2.0 * batch_size * num_heads_q * seqlen_q * seqlen_k * kHeadDim; // Q@K^T
        const double flops_v = 2.0 * batch_size * num_heads_q * seqlen_q * seqlen_k * kHeadDim; // P@V
        const double total_flops = flops + flops_v;
        const double tflops = (total_flops / (milliseconds * 1e-3)) / 1e12;
        
        std::cout << "  Performance: " << tflops << " TFLOPS" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Kernel execution failed: " << e.what() << std::endl;
        CHECK_CUDA(cudaGetLastError());
    }
    
    // Cleanup events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Verify output (simple sanity check)
    std::vector<Element> h_o(o_size);
    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(Element), 
                          cudaMemcpyDeviceToHost));
    
    // Check for NaN or Inf
    bool has_nan_inf = false;
    float max_val = 0.0f;
    float min_val = 1e10f;
    double sum = 0.0;
    
    for (size_t i = 0; i < o_size; ++i) {
        float val = static_cast<float>(h_o[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
        max_val = std::max(max_val, std::abs(val));
        min_val = std::min(min_val, std::abs(val));
        sum += val;
    }
    
    std::cout << std::endl;
    std::cout << "=== Output Statistics ===" << std::endl;
    if (has_nan_inf) {
        std::cout << "✗ Output contains NaN or Inf!" << std::endl;
    } else {
        std::cout << "✓ Output is valid (no NaN/Inf)" << std::endl;
        std::cout << "  Max absolute value: " << max_val << std::endl;
        std::cout << "  Min absolute value: " << min_val << std::endl;
        std::cout << "  Mean value: " << sum / o_size << std::endl;
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));
    CHECK_CUDA(cudaFree(d_lora_v));
    CHECK_CUDA(cudaFree(d_lora_b));
    CHECK_CUDA(cudaFree(d_softmax_lse));
    CHECK_CUDA(cudaFree(d_softmax_lseaccum));
    CHECK_CUDA(cudaFree(d_oaccum));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    std::cout << std::endl;
    std::cout << "=== SplitKV Test Complete ===" << std::endl;
}

void test_flash_fwd() {
    using Element = cutlass::half_t;
    constexpr int kHeadDim = 128;
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 128;
    constexpr int kNWarps = 4;
    constexpr int kLoRADim = 8;
    constexpr int batch_size = 1;
    constexpr int num_heads_q = 32;
    constexpr int num_heads_kv = 8;
    constexpr int seqlen_q = 1024;
    constexpr int seqlen_k = 1024;
    
    std::cout << "=== Flash LoRA Attention Forward Test ===" << std::endl;
    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Q heads: " << num_heads_q << std::endl;
    std::cout << "  KV heads: " << num_heads_kv << " (GQA)" << std::endl;
    std::cout << "  Sequence length (Q): " << seqlen_q << std::endl;
    std::cout << "  Sequence length (KV): " << seqlen_k << std::endl;
    std::cout << "  kBlockM: " << kBlockM << std::endl;
    std::cout << "  kBlockN: " << kBlockN << std::endl;
    std::cout << "  Head dimension: " << kHeadDim << std::endl;
    std::cout << "  LoRA dimension: " << kLoRADim << std::endl;
    std::cout << std::endl;
    
    // Calculate sizes
    const size_t q_size = batch_size * seqlen_q * num_heads_q * kHeadDim;
    const size_t k_size = batch_size * seqlen_k * num_heads_kv * kHeadDim;
    const size_t v_size = batch_size * seqlen_k * num_heads_kv * kHeadDim;
    const size_t o_size = batch_size * seqlen_q * num_heads_q * kHeadDim;
    const size_t lora_v_size = batch_size * seqlen_k * kLoRADim;
    const size_t lora_b_size = num_heads_kv * kLoRADim * kHeadDim;
    
    // Allocate device memory
    Element *d_q, *d_k, *d_v, *d_o;
    Element *d_lora_v, *d_lora_b;
    float *d_softmax_lse;
    
    CHECK_CUDA(cudaMalloc(&d_q, q_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_k, k_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_v, v_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_o, o_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_lora_v, lora_v_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_lora_b, lora_b_size * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&d_softmax_lse, batch_size * num_heads_q * seqlen_q * sizeof(float)));
    
    // Initialize with random data
    std::cout << "Initializing tensors with random data..." << std::endl;
    fill_random(d_q, q_size, 0.02f);
    fill_random(d_k, k_size, 0.02f);
    fill_random(d_v, v_size, 0.02f);
    fill_random(d_lora_v, lora_v_size, 0.02f);
    fill_random(d_lora_b, lora_b_size, 0.02f);
    CHECK_CUDA(cudaMemset(d_o, 0, o_size * sizeof(Element)));
    CHECK_CUDA(cudaMemset(d_softmax_lse, 0, batch_size * num_heads_q * seqlen_q * sizeof(float)));
    
    // Setup Flash_fwd_params - zero initialize to avoid garbage values
    FLASH_NAMESPACE::Flash_fwd_params params;
    memset(&params, 0, sizeof(params));
    
    // QKV pointers
    params.q_ptr = d_q;
    params.k_ptr = d_k;
    params.v_ptr = d_v;
    params.o_ptr = d_o;
    
    // LoRA pointers
    params.lora_v_ptr = d_lora_v;
    params.lora_b_ptr = d_lora_b;
    
    // Strides (assuming contiguous layout: [batch, seqlen, heads, head_dim])
    params.q_batch_stride = seqlen_q * num_heads_q * kHeadDim;
    params.q_row_stride = num_heads_q * kHeadDim;
    params.q_head_stride = kHeadDim;
    
    params.k_batch_stride = seqlen_k * num_heads_kv * kHeadDim;
    params.k_row_stride = num_heads_kv * kHeadDim;
    params.k_head_stride = kHeadDim;
    
    params.v_batch_stride = seqlen_k * num_heads_kv * kHeadDim;
    params.v_row_stride = num_heads_kv * kHeadDim;
    params.v_head_stride = kHeadDim;
    
    params.o_batch_stride = seqlen_q * num_heads_q * kHeadDim;
    params.o_row_stride = num_heads_q * kHeadDim;
    params.o_head_stride = kHeadDim;
    
    // LoRA strides
    params.lora_v_batch_stride = seqlen_k * kLoRADim;
    params.lora_v_row_stride = kLoRADim;
    
    params.lora_b_head_stride = kLoRADim * kHeadDim;
    params.lora_b_row_stride = kHeadDim;
    
    // Dimensions
    params.b = batch_size;
    params.h = num_heads_q;
    params.h_k = num_heads_kv;
    params.h_h_k_ratio = num_heads_q / num_heads_kv;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = ((seqlen_q + kBlockM - 1) / kBlockM) * kBlockM;
    params.seqlen_k_rounded = ((seqlen_k + kBlockN - 1) / kBlockN) * kBlockN;
    params.d = kHeadDim;
    params.d_rounded = kHeadDim;
    params.lora_dim = kLoRADim;
    params.total_q = batch_size * seqlen_q;
    
    // Softmax scale
    params.scale_softmax = 1.0f / std::sqrt(float(kHeadDim));
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;
    
    // LSE pointer
    params.softmax_lse_ptr = d_softmax_lse;
    params.softmax_lseaccum_ptr = nullptr;
    
    // No special features for this test
    params.p_ptr = nullptr;  // No softmax output
    params.oaccum_ptr = nullptr;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.leftpad_k = nullptr;
    params.blockmask = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.p_dropout = 0.0f;  // No dropout
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.is_bf16 = false;
    params.is_causal = false;  // Non-causal attention
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;
    params.rotary_dim = 0;
    params.is_rotary_interleaved = false;
    params.num_splits = 0;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    std::cout << "Running Flash Attention forward kernel..." << std::endl;
    
    // Run the kernel
    using Kernel_traits = Flash_fwd_kernel_traits<
        kHeadDim, kBlockM, kBlockN, kNWarps, kLoRADim, false, false, Element>;
    
    constexpr bool Is_dropout = false;
    constexpr bool Is_causal = false;
    
    try {
        // Start timing
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        FLASH_NAMESPACE::run_flash_fwd<Kernel_traits, Is_dropout, Is_causal>(params, stream);
        
        // Stop timing
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << "✓ Kernel execution successful!" << std::endl;
        std::cout << "  Execution time: " << milliseconds << " ms" << std::endl;
        
        // Calculate performance metrics
        const double flops = 2.0 * batch_size * num_heads_q * seqlen_q * seqlen_k * kHeadDim; // Q@K^T
        const double flops_v = 2.0 * batch_size * num_heads_q * seqlen_q * seqlen_k * kHeadDim; // P@V
        const double total_flops = flops + flops_v;
        const double tflops = (total_flops / (milliseconds * 1e-3)) / 1e12;
        
        std::cout << "  Performance: " << tflops << " TFLOPS" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Kernel execution failed: " << e.what() << std::endl;
        CHECK_CUDA(cudaGetLastError());
    }
    
    // Cleanup events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Verify output (simple sanity check)
    std::vector<Element> h_o(o_size);
    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(Element), 
                          cudaMemcpyDeviceToHost));
    
    // Check for NaN or Inf
    bool has_nan_inf = false;
    float max_val = 0.0f;
    float min_val = 1e10f;
    double sum = 0.0;
    
    for (size_t i = 0; i < o_size; ++i) {
        float val = static_cast<float>(h_o[i]);
        if (std::isnan(val) || std::isinf(val)) {
            has_nan_inf = true;
            break;
        }
        max_val = std::max(max_val, std::abs(val));
        min_val = std::min(min_val, std::abs(val));
        sum += val;
    }
    
    std::cout << std::endl;
    std::cout << "=== Output Statistics ===" << std::endl;
    if (has_nan_inf) {
        std::cout << "✗ Output contains NaN or Inf!" << std::endl;
    } else {
        std::cout << "✓ Output is valid (no NaN/Inf)" << std::endl;
        std::cout << "  Max absolute value: " << max_val << std::endl;
        std::cout << "  Min absolute value: " << min_val << std::endl;
        std::cout << "  Mean value: " << sum / o_size << std::endl;
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_o));
    CHECK_CUDA(cudaFree(d_lora_v));
    CHECK_CUDA(cudaFree(d_lora_b));
    CHECK_CUDA(cudaFree(d_softmax_lse));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    std::cout << std::endl;
    std::cout << "=== Test Complete ===" << std::endl;
}

int main() {
    // Check CUDA device
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    if (prop.major < 8) {
        std::cerr << "Flash Attention requires compute capability >= 8.0 (Ampere)" << std::endl;
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "   Running Standard Flash Attention    " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    test_flash_fwd();
    
    std::cout << std::endl << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "     Running SplitKV Flash Attention   " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    test_flash_splitkv_fwd();
    
    return 0;
}
