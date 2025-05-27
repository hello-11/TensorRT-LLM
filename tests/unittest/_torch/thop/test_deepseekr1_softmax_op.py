# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from itertools import product

import torch
from parameterized import parameterized
from utils.util import unittest_name_func

import tensorrt_llm


class TestSoftmaxOp(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(
            product(
                [1],  # batch sizes
                [4],  # num_MTP_tokens
                [16],  # num_heads
                [2048, 1024, 391],  # hidden dimensions
                ['bfloat16'],  # data types
            )),
        name_func=unittest_name_func)
    def test_deepseekr1_softmax_basic(self, batch_size, num_MTP_tokens,
                                      num_heads, hidden_dim, dtype):
        """Test basic softmax functionality with various configurations."""
        device = "cuda"
        torch_dtype = getattr(torch, dtype)

        rows = batch_size * num_MTP_tokens * num_heads
        cols = hidden_dim
        # Generate random input data
        input_data = torch.randn(rows, cols, device=device, dtype=torch_dtype)

        # Calculate reference output using PyTorch
        ref_output = torch.softmax(input_data, dim=-1)

        # Run TensorRT-LLM softmax
        output = torch.ops.trtllm.deepseekr1_softmax(input_data, dim=-1)

        # Compare results
        atol = {"bfloat16": 1e-2}
        torch.testing.assert_close(output,
                                   ref_output,
                                   rtol=1e-2,
                                   atol=atol[dtype])


if __name__ == "__main__":
    unittest.main()
