# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from neuronx_distributed_inference.models.config import NeuronConfig


class Lfm2NeuronConfig(NeuronConfig):
    """
    Neuron-specific configuration for LFM2 model
    """
    def __init__(
        self,
        tp_degree: int = 8,
        batch_size: int = 1,
        seq_len: int = 2048,
        use_fp16: bool = True,
        **kwargs
    ):
        super().__init__(
            tp_degree=tp_degree,
            batch_size=batch_size,
            **kwargs
        )
        self.seq_len = seq_len
        self.use_fp16 = use_fp16
