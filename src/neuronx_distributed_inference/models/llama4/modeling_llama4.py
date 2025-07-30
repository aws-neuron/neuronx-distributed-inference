# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

import copy
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    destroy_model_parallel,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from neuronx_distributed.utils.utils import hardware
from neuronx_distributed.trace.trace import get_sharded_checkpoint

from torch_neuronx.utils import get_platform_target
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper, IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS
from neuronx_distributed_inference.models.llama4.modeling_llama4_text import (
    NeuronLlama4TextForCausalLM,
    NeuronLlama4TextModel,
)
from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import (
    VISION_MAX_NUM_CHUNKS,
    Llama4VisionModelWrapper,
    NeuronLlama4ForImageEncoding,
    NeuronLlama4VisionEmbeddings,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_llama4_vision_encoder_buckets,
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)
from neuronx_distributed_inference.models.image_to_text_model_base import ImageToTextInferenceConfig, NeuronBaseForImageToText
from neuronx_distributed_inference.models.model_wrapper import (CONTEXT_ENCODING_MODEL_TAG,
                                                                TOKEN_GENERATION_MODEL_TAG,
                                                                VISION_ENCODER_MODEL_TAG)
from neuronx_distributed_inference.modules.autobucketing import generate_buckets

logger = logging.getLogger("Neuron")

_HARDWARE = hardware(get_platform_target())


class Llama4NeuronConfig(MoENeuronConfig):
    """
    TODO: This class only inherits MoENeuronConfig and should be removed.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)


class Llama4InferenceConfig(ImageToTextInferenceConfig):
    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )
        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Llama4 does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Llama4 does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Llama4 does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Llama4 does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Llama4 does not yet support fused speculation.")

        # Calculate the vision chink size and save in text_config
        # This is needed by by text model wrapper's input_generator to initilize
        # the correct shapes for vision arguments
        self.text_config.vision_encoder_chunk_size = self.get_chunk_size()

    def get_chunk_size(self):
        num_patches = self.vision_config.image_size // self.vision_config.patch_size
        downsample_ratio = int(round(1.0 / (self.vision_config.pixel_shuffle_ratio**2)))
        return pow(num_patches, 2) // downsample_ratio

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list

        return [
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.pad_token_id",
            "text_config.vocab_size",
            "text_config.max_position_embeddings",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.hidden_act",
            "text_config.num_local_experts",
            "text_config.num_experts_per_tok",
            "vision_config.image_size",
            "vision_config.patch_size",
            "vision_config.num_hidden_layers",
            "vision_config.num_channels",
            "vision_config.hidden_size",
            "vision_config.num_attention_heads",
            "vision_config.rope_theta",
            "vision_config.vision_output_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Llama4NeuronConfig]:
        return Llama4NeuronConfig


class NeuronLlama4ForCausalLM(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronLlama4TextModel
    vision_model_cls = NeuronLlama4VisionEmbeddings

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Llama4VisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        self.compile_tag = VISION_ENCODER_MODEL_TAG

        new_config = copy.deepcopy(self.config)
        new_config.neuron_config = copy.deepcopy(self.vision_config.neuron_config)
        if new_config.neuron_config.enable_bucketing:
            if new_config.neuron_config.buckets is None:
                new_config.neuron_config.buckets = generate_llama4_vision_encoder_buckets(
                    self.neuron_config.dp_degree, VISION_MAX_NUM_CHUNKS
                )
        else:
            new_config.neuron_config.buckets = generate_buckets(
                VISION_MAX_NUM_CHUNKS, VISION_MAX_NUM_CHUNKS
            )
        self.vision_config.neuron_config.buckets = new_config.neuron_config.buckets
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=False,
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # text model state dict convertion
        state_dict = NeuronLlama4TextForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.text_config
        )

        # vision model state dict convertion
        state_dict = NeuronLlama4ForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.vision_config
        )

        return state_dict

    def _convert_input_dict_to_ordered_tuple(self, input_dict: Dict[str, Any]):
        """
        Utility function to convert input dictionary to ordered tuple
        based on outputs of _get_model_outputs
        """
        args = []

        for key in IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS:
            if key in input_dict and input_dict[key] is not None:
                arg = input_dict[key]
            else:
                arg = torch.empty(0)
            args.append(arg)

        return tuple(args)

    def get_padding_length(self, buckets, input_ids):
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
        ]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, _ = input_ids.shape
        if (
            (pixel_values is not None)
            and (vision_mask is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):  # call vision encoder
            assert (
                vision_mask.dtype == torch.bool
            ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            pad_limit = self.get_padding_length(self.config.neuron_config.buckets, input_ids)
            # Pad vision mask to VISION_MAX_NUM_CHUNKS

            vision_mask = pad_positions(
                vision_mask, pad_limit, (pad_limit - 1)
            )

            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
            ).to(self.text_config.neuron_config.torch_dtype)

            # flatten vision embeddings
            embedding_dim = vision_embeddings.shape[-1]
            vision_embeddings = vision_embeddings.view(-1, embedding_dim).unsqueeze(0)

            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)
        else:
            pad_limit = self.get_padding_length(self.config.neuron_config.buckets, input_ids)

            vision_embeddings, vision_mask = self.text_model_wrapper.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1)
            )

        output_token = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )
        return output_token

    @classmethod
    def get_config_cls(cls):
        return Llama4InferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Llama4ForConditionalGeneration

        return Llama4ForConditionalGeneration.from_pretrained(model_path, **kwargs)

    def to_cpu(self):
        """
        Initialize CPU versions of both text and vision models with different parallelism configurations,
        shard and load their weights, and assign to respective model wrappers.
        This function as of now only supports TP DEGREE of 1 in vision and text.
        """
        os.environ["NXD_CPU_MODE"] = "1"

        # Validation checks
        if self.neuron_config.torch_dtype == torch.bfloat16 and (
            self.neuron_config.tp_degree > 1 or self.neuron_config.ve_tp_degree > 1
        ):
            raise NotImplementedError(
                "The gloo backend does not natively support bfloat16, please proceed with float32 dtype instead."
            )
        if self.neuron_config.speculation_length > 0:
            raise NotImplementedError("Speculation is not yet supported for CPU inference.")

        # destroy distributed process if already started
        if model_parallel_is_initialized():
            destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Initialize distributed processing
        if "WORLD_SIZE" in os.environ:
            assert (
                int(os.environ["WORLD_SIZE"]) == self.neuron_config.world_size
            ), "Total number of processes does not match implied world size from NeuronConfig inputs."
            torch.distributed.init_process_group("gloo")
        if not torch.distributed.is_initialized():
            if self.neuron_config.world_size == 1:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                torch.distributed.init_process_group(
                    backend="gloo",
                    world_size=1,
                    rank=0,
                )
            else:
                raise RuntimeError("Please initialize parallel processing via 'torchrun'.")

        # Initialize model parallel for vision and text model. We only support TP Degree 1 at this point.
        initialize_model_parallel(
            tensor_model_parallel_size=self.neuron_config.tp_degree,
            pipeline_model_parallel_size=1,  # No pipeline parallelism for vision encoder
            expert_model_parallel_size=1,  # No expert parallelism for vision encoder
            skip_collective_init=True,
        )

        # Initialize and load vision model with vision-specific config
        vision_base_model = self.vision_model_cls(self.config)
        vision_base_model = vision_base_model.to(
            self.vision_config.neuron_config.torch_dtype
        )

        vision_model_sd = (
            self.checkpoint_loader_fn()
        )  # You might need a separate loader for vision weights
        if self.vision_config.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                vision_model_sd,
                vision_base_model,
                torch.distributed.get_rank(),
                self.vision_config.neuron_config.tp_degree,
            )

        vision_base_model.load_state_dict(vision_model_sd, strict=False)

        # Initialize and load text model with text-specific config
        text_base_model = self.text_model_cls(self.config.text_config)
        text_base_model = text_base_model.to(self.config.text_config.neuron_config.torch_dtype)

        text_model_sd = self.checkpoint_loader_fn()
        if self.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                text_model_sd,
                text_base_model,
                torch.distributed.get_rank(),
                self.neuron_config.tp_degree,
            )
        text_base_model.load_state_dict(text_model_sd, strict=False)

        # Assign models to their respective wrappers
        for model_wrapper in self.text_models:
            model_wrapper.model = text_base_model

        for model_wrapper in self.vision_models:
            model_wrapper.model = vision_base_model

        self.eval()

    # Wraps NeuronBaseForCausalLM.enable_context_encoding() to add compile_tag.
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    # Wraps NeuronBaseForCausalLM.enable_token_generation() to add compile_tag.
    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self) -> str:
        logical_nc_config = self.text_config.neuron_config.logical_nc_config

        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O2"
        elif self.compile_tag == VISION_ENCODER_MODEL_TAG:
            return f"-O1 --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap' " \
                   f"--auto-cast=none --lnc={logical_nc_config}"
        else:
            raise ValueError(f"get_compiler_args() Invalid compile tag encountered: {self.compile_tag}")

        args = f"--auto-cast=none --model-type=transformer '--tensorizer-options=--enable-ccop-compute-overlap " \
               f"--cc-pipeline-tiling-factor=1 --vectorize-strided-dma --enable-scalar-dge-vectorization' " \
               f"--lnc={logical_nc_config} {optimization_level} "
        return args
