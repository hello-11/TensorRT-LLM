/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/request.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime::decoder
{
class DecoderState;
} // namespace tensorrt_llm::runtime::decoder

namespace tensorrt_llm::batch_manager
{
class RuntimeBuffers;
class DecoderInputBuffers;

class GenerateRequestOptions : Algorithm
{
public:
    constexpr static auto name{"GenerateRequestOptions"};

    using SizeType32 = runtime::SizeType32;
    using ITensor = runtime::ITensor;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using BufferManager = runtime::BufferManager;
    template <typename T>
    using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

    GenerateRequestOptions(bool speculativeDecodingFastLogits, bool isLeaderInOrchMode, bool isNormalizeLogProbs)
        : mSpeculativeDecodingFastLogits(speculativeDecodingFastLogits)
        , mIsLeaderInOrchMode(isLeaderInOrchMode)
        , mIsNormalizeLogProbs(isNormalizeLogProbs)
    {
    }

    std::tuple<ITensor::SharedPtr, std::vector<runtime::decoder_batch::Request>, std::vector<runtime::SamplingConfig>>
    operator()(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests,
        BufferManager const& bufferManager, nvinfer1::DataType logitsType, DecoderInputBuffers& inputBuffers,
        runtime::decoder::DecoderState& decoderState, SizeType32 beamWidth, runtime::CudaStream const& stream,
        OptionalRef<RuntimeBuffers const> buffers = std::nullopt) const;

private:
    [[nodiscard]] std::vector<runtime::decoder_batch::Request> createDecoderRequests(
        RequestVector const& finishedContextRequests, TensorPtr const& inputIds,
        executor::DecodingConfig const& decodingConfig, BufferManager const& bufferManager,
        nvinfer1::DataType logitsType, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        OptionalRef<RuntimeBuffers const> buffers) const;

    [[nodiscard]] std::shared_ptr<runtime::ITensor> retrieveDraftLogits(runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, std::shared_ptr<runtime::ITensor> const& tensor,
        BufferManager const& bufferManager) const;

    bool mSpeculativeDecodingFastLogits;
    bool mIsLeaderInOrchMode;
    bool mIsNormalizeLogProbs;
};

} // namespace tensorrt_llm::batch_manager
