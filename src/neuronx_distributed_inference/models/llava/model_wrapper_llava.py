# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    ModelWrapper,
)


class ModelWrapperLlava(ModelWrapper):
    """
    A class that wraps the Llava model for context encoding, speculation, and token generation tasks.
    This class overrides input_generator() to provide additional pixel_values in the sample inputs for tracing
    """

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            pixel_values = torch.zeros(
                (
                    self.neuron_config.batch_size,
                    3,
                    self.config.vision_config.image_size,
                    self.config.vision_config.image_size,
                ),
                dtype=self.neuron_config.torch_dtype,
            )
            text_embedding_indices = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            image_embedding_indices = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            inputs.append(
                (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    pixel_values,
                    text_embedding_indices,
                    image_embedding_indices,
                )
            )

        return inputs

    def pad_to_max_compiled_seq(self, *args):
        args = super().pad_to_max_compiled_seq(*args)

        # text_embedding_indices, image_embedding_indices
        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[5:]
            pad_lengths = [self.neuron_config.max_context_length - arg.shape[1] for arg in to_pad]

            # We use this pad value to "torch.scatter_" garbage elements to the last position of embeddings
            # Because of the attention mask, this garbage elements won't be used
            pad_val = self.neuron_config.max_context_length - 1
            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_len in zip(to_pad, pad_lengths)
            ]

            # In different batches, the lengths of the valid part of args are different
            # The unused values were prefilled with -1 and we want to change it to pad_val
            padded_args = [torch.where(arg == -1, pad_val, arg) for arg in padded_args]
            return (*args[:5], *padded_args)
        return args
