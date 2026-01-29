# coding=utf-8
# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# Ported to AWS Neuron by Amazon Web Services
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
"""InternLM3 Neuron configuration"""

from typing import List, Type
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


class InternLM3NeuronConfig(InferenceConfig):
    """
    Configuration class for InternLM3 Neuron model.
    Reference: transformers/src/transformers/models/internlm3/configuration_internlm3.py::InternLM3Config
    """
    
    def __init__(
        self,
        vocab_size=128512,
        hidden_size=4096,
        intermediate_size=10240,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=50000000,
        rope_scaling=None,
        qkv_bias=False,
        attention_dropout=0.0,
        bias=False,
        head_dim=128,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
        ]
    
    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load configuration from pretrained model directory."""
        import json
        import os
        
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        config_dict.update(kwargs)
        return cls(**config_dict)
