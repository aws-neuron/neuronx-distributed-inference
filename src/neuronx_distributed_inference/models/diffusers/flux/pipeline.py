# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# This implementation is derived from the Diffusers library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
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

import functools
import logging

from diffusers import FluxPipeline, FluxFillPipeline, FluxControlPipeline

logger = logging.getLogger(__name__)


class NeuronFluxPipeline(FluxPipeline):
    @functools.wraps(FluxPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert kwargs.get("lora_scale") is None, "NeuronFluxPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxPipeline.__call__)
    def __call__(self, *args, **kwargs):
        assert (
            kwargs.get("ip_adapter_image") is None
        ), "NeuronFluxPipeline does not support ip_adapter_image input."
        assert (
            kwargs.get("ip_adapter_image_embeds") is None
        ), "NeuronFluxPipeline does not support ip_adapter_image_embeds input."
        assert (
            kwargs.get("negative_ip_adapter_image") is None
        ), "NeuronFluxPipeline does not support negative_ip_adapter_image input."
        assert (
            kwargs.get("negative_ip_adapter_image_embeds") is None
        ), "NeuronFluxPipeline does not support negative_ip_adapter_image_embeds input."

        with self.transformer.image_rotary_emb_cache_context():
            return super().__call__(*args, **kwargs)


class NeuronFluxFillPipeline(FluxFillPipeline):
    @functools.wraps(FluxFillPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert (
            kwargs.get("lora_scale") is None
        ), "NeuronFluxFillPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxFillPipeline.__call__)
    def __call__(self, *args, **kwargs):
        assert (
            kwargs.get("ip_adapter_image") is None
        ), "NeuronFluxFillPipeline does not support ip_adapter_image input."
        assert (
            kwargs.get("ip_adapter_image_embeds") is None
        ), "NeuronFluxFillPipeline does not support ip_adapter_image_embeds input."
        assert (
            kwargs.get("negative_ip_adapter_image") is None
        ), "NeuronFluxFillPipeline does not support negative_ip_adapter_image input."
        assert (
            kwargs.get("negative_ip_adapter_image_embeds") is None
        ), "NeuronFluxFillPipeline does not support negative_ip_adapter_image_embeds input."

        with self.transformer.image_rotary_emb_cache_context():
            return super().__call__(*args, **kwargs)


class NeuronFluxControlPipeline(FluxControlPipeline):
    @functools.wraps(FluxControlPipeline.encode_prompt)
    def encode_prompt(self, *args, **kwargs):
        assert (
            kwargs.get("lora_scale") is None
        ), "NeuronFluxControlPipeline does not support LoRA."
        return super().encode_prompt(*args, **kwargs)

    @functools.wraps(FluxControlPipeline.__call__)
    def __call__(self, *args, **kwargs):
        with self.transformer.image_rotary_emb_cache_context():
            return super().__call__(*args, **kwargs)
