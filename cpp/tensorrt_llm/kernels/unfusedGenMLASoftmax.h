/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include <cstdint>

namespace tensorrt_llm
{
namespace kernels
{
namespace unfused_gen_mla_softmax
{
template <typename T>
void invokeUnfusedGenMLASoftmax(T const* input, T* output, int const* seqLensKvPtr, int q_seq_len, int max_kv_seq_len,
    float softmax_scale, int max_q_seq_len, int num_heads, cudaStream_t stream);

} // namespace unfused_gen_mla_softmax
} // namespace kernels
} // namespace tensorrt_llm
