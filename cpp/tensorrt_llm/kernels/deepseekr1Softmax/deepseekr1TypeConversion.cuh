/*
 * Adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.h
 * Copyright (c) 2023, Tri Dao.
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
 *
 * Not a contribution
 * Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
 * NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tensorrt_llm
{
namespace kernels
{

// Convert half to float
inline __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

// Convert float to half
inline __device__ uint16_t float_to_half(float f)
{
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// Convert half2 to float2
inline __device__ float2 half2_to_float2(uint32_t x)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Use native instruction on Ampere and later
    float2 f;
    asm volatile("cvt.f32.f16x2 %0, %1;\n" : "=f"(f.x), "=f"(f.y) : "r"(x));
    return f;
#else
    // Optimized version for older architectures
    float2 f;
    asm volatile(
        "{\n"
        "    .reg .b16 lo, hi;\n"
        "    mov.b32 {lo, hi}, %2;\n"
        "    cvt.f32.f16 %0, lo;\n"
        "    cvt.f32.f16 %1, hi;\n"
        "}\n"
        : "=f"(f.x), "=f"(f.y)
        : "r"(x));
    return f;
#endif
}

// Convert float2 to half2
inline __device__ uint32_t float2_to_half2(float2 f)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(f.y), "f"(f.x));
#else
    uint16_t lo = float_to_half(f.x);
    uint16_t hi = float_to_half(f.y);
    asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(c) : "h"(lo), "h"(hi));
#endif
    return c;
}

#ifdef ENABLE_BF16
// Convert bfloat16 to float
inline __device__ float bf16_to_float(uint16_t h)
{
    float f;
    asm volatile("mov.b32 %0, {0, %1};\n" : "=f"(f) : "h"(h));
    return f;
}

// Convert float to bfloat16
inline __device__ uint16_t float_to_bf16(float f)
{
    return __float2bfloat16(f);
}

// Convert bfloat162 to float2
inline __device__ float2 bf162_to_float2(uint32_t x)
{
    float2 res;
    asm volatile(
        "{\n"
        "    .reg .b16 lo, hi;\n"
        "    mov.b32 {lo, hi}, %2;\n"
        "    mov.b32 %0, {0, lo};\n"
        "    mov.b32 %1, {0, hi};\n"
        "}\n"
        : "=f"(res.x), "=f"(res.y)
        : "r"(x));
    return res;
}

// Convert float2 to bfloat162
inline __device__ uint32_t float2_to_bf162(float2 f)
{
    uint32_t c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(c) : "f"(f.y), "f"(f.x));
#else
    uint16_t* px = reinterpret_cast<uint16_t*>(&f.x);
    uint16_t* py = reinterpret_cast<uint16_t*>(&f.y);
    uint16_t value = px[1];
    uint16_t value2 = py[1];

    if (px[0] == 0x8000)
    {
        if ((value & 0x1) == 1)
            value++;
    }
    else if (px[0] > 0x8000)
    {
        value++;
    }

    if (py[0] == 0x8000)
    {
        if ((value2 & 0x1) == 1)
            value2++;
    }
    else if (py[0] > 0x8000)
    {
        value2++;
    }

    uint32_t high = reinterpret_cast<uint32_t&>(value2);
    c = (high << 16) | value;
#endif
    return c;
}
#endif // ENABLE_BF16

// Template function to convert to float
template <typename T>
inline __device__ float to_float(T x);

template <>
inline __device__ float to_float<half>(half x)
{
    return half_to_float(reinterpret_cast<uint16_t&>(x));
}

template <>
inline __device__ float to_float<float>(float x)
{
    return x;
}

#ifdef ENABLE_BF16
template <>
inline __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x)
{
    return bf16_to_float(reinterpret_cast<uint16_t&>(x));
}
#endif

// Template function to convert from float
template <typename T>
inline __device__ T from_float(float x);

template <>
inline __device__ half from_float<half>(float x)
{
    uint16_t h = float_to_half(x);
    return *reinterpret_cast<half*>(&h);
}

template <>
inline __device__ float from_float<float>(float x)
{
    return x;
}

#ifdef ENABLE_BF16
template <>
inline __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x)
{
    uint16_t h = float_to_bf16(x);
    return *reinterpret_cast<__nv_bfloat16*>(&h);
}
#endif

// template function to convert to float2
template <typename T>
inline __device__ float2 to_float2(T x);

template <>
inline __device__ float2 to_float2<half2>(half2 x)
{
    return half2_to_float2(reinterpret_cast<uint32_t&>(x));
}

template <>
inline __device__ float2 to_float2<float2>(float2 x)
{
    return x;
}

#ifdef ENABLE_BF16
template <>
inline __device__ float2 to_float2<__nv_bfloat162>(__nv_bfloat162 x)
{
    return bf162_to_float2(reinterpret_cast<uint32_t&>(x));
}
#endif

// template function to convert from float2
template <typename T>
inline __device__ T from_float2(float2 x);

template <>
inline __device__ half2 from_float2<half2>(float2 x)
{
    uint32_t h = float2_to_half2(x);
    return *reinterpret_cast<half2*>(&h);
}

template <>
inline __device__ float2 from_float2<float2>(float2 x)
{
    return x;
}

#ifdef ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 from_float2<__nv_bfloat162>(float2 x)
{
    uint32_t h = float2_to_bf162(x);
    return *reinterpret_cast<__nv_bfloat162*>(&h);
}
#endif

} // namespace kernels
} // namespace tensorrt_llm
