/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/unfusedGenMLASoftmax.h"

#include <cuda_fp16.h>

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cuda_runtime_api.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace unfused_gen_mla_softmax
{

// Helper function for warp-wide max reduction
inline __device__ float warpReduceMax(float val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t mask = 0xffffffffu;
    asm volatile("redux.sync.max.f32 %0, %0, %1;\n" : "+f"(val) : "r"(mask));
#else
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
#endif
    return val;
}

// Helper function for warp-wide sum reduction
inline __device__ float warpReduceSum(float val)
{
    // With _xor_ all the threads have the reduced value in the end.
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for block-wide max reduction
inline __device__ float blockReduceMax(float* shared, float val)
{
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // All the warps have the max this way.
    val = (lane < blockDim.x / 32) ? shared[lane] : -INFINITY;
    return warpReduceMax(val);
}

// Helper function for block-wide sum reduction
inline __device__ float blockReduceSum(float* shared, float val)
{
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // All the warps have the sum this way.
    val = (lane < blockDim.x / 32) ? shared[lane] : 0.f;
    return warpReduceSum(val);
}

template <typename T>
__global__ void basic_unfused_gen_mla_softmax_kernel(T const* input, T* output, int const q_seq_len,
    int const max_kv_seq_len, float const softmax_scale, int const* seqLensKvPtr)
{
    return;
}

#ifdef ENABLE_BF16
template <>
__global__ void basic_unfused_gen_mla_softmax_kernel<__nv_bfloat16>(__nv_bfloat16 const* input, __nv_bfloat16* output,
    int const q_seq_len, int const max_kv_seq_len, float const softmax_scale, int const* seqLensKvPtr)
{

    int q_id = blockIdx.x;
    int valid_kv_len = seqLensKvPtr[0];

    if (q_id >= q_seq_len)
        return;

    // Shared memory for the reductions - we have at most 64 warps per block
    __shared__ float s_max[64], s_sum[64];

    int const col = threadIdx.x;

    // Initialize reduction variables
    float max_val = -INFINITY;
    float sum_val = 0.0f;

    // First pass: find max value
    for (int i = col; i < max_kv_seq_len; i += blockDim.x)
    {
        float val = i < valid_kv_len ? static_cast<float>(input[q_id * max_kv_seq_len + i]) * softmax_scale : -INFINITY;
        max_val = fmaxf(max_val, val);
    }

    // Perform block-wide max reduction
    max_val = blockReduceMax(s_max, max_val);

    // Second pass: compute exp and sum
    for (int i = col; i < max_kv_seq_len; i += blockDim.x)
    {
        float val = i < valid_kv_len ? static_cast<float>(input[q_id * max_kv_seq_len + i]) * softmax_scale : -INFINITY;
        val = i < valid_kv_len ? expf(val - max_val) : 0.0f;
        sum_val += val;
        output[q_id * max_kv_seq_len + i] = static_cast<__nv_bfloat16>(val);
    }

    // Perform block-wide sum reduction
    sum_val = blockReduceSum(s_sum, sum_val);

    // Third pass: normalize
    for (int i = col; i < valid_kv_len; i += blockDim.x)
    {
        float val = static_cast<float>(output[q_id * max_kv_seq_len + i]);
        output[q_id * max_kv_seq_len + i] = static_cast<__nv_bfloat16>(val / (sum_val + 1e-5f));
    }
}
#endif

template <typename T>
void invokeUnfusedGenMLASoftmax(T const* input, T* output, int const* seqLensKvPtr, int q_seq_len, int max_kv_seq_len,
    float softmax_scale, cudaStream_t stream)
{
    dim3 grid(q_seq_len);
    dim3 block(256);

    basic_unfused_gen_mla_softmax_kernel<T>
        <<<grid, block, 0, stream>>>(input, output, q_seq_len, max_kv_seq_len, softmax_scale, seqLensKvPtr);
}

template void invokeUnfusedGenMLASoftmax<float>(float const* input, float* output, int const* seqLensKvPtr,
    int q_seq_len, int max_kv_seq_len, float softmax_scale, cudaStream_t stream);

template void invokeUnfusedGenMLASoftmax<half>(half const* input, half* output, int const* seqLensKvPtr, int q_seq_len,
    int max_kv_seq_len, float softmax_scale, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeUnfusedGenMLASoftmax<__nv_bfloat16>(__nv_bfloat16 const* input, __nv_bfloat16* output,
    int const* seqLensKvPtr, int q_seq_len, int max_kv_seq_len, float softmax_scale, cudaStream_t stream);
#endif

} // namespace unfused_gen_mla_softmax
} // namespace kernels
} // namespace tensorrt_llm
