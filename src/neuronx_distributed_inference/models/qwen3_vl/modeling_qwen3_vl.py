# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3-VL model for NxD Inference."""

import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from neuronx_distributed_inference.modules.autobucketing import generate_buckets
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLTextForCausalLM,
    NeuronQwen3VLTextModel,
    NeuronQwen3VLTextModelWrapper,
)
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLForImageEncoding,
    NeuronQwen3VLVisionModel,
    NeuronQwen3VLVisionModelWrapper,
)
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group

logger = logging.getLogger("Neuron")

# From HF checkpoint preprocessor_config.json
# We use these constants to calculate resized image height and width after pre-processing
# Fixed image height and width will not be needed after we support dynamic image resolution by bucketing on seq dim
# TODO: remove after we support dynamic image resolution
IMAGE_SIZE_SHORTEST_EDGE = 65536
IMAGE_SIZE_LONGEST_EDGE = 16777216


# TODO: Use NeuronConfig after dynamic image resolution is supported
class Qwen3VLNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class Qwen3VLInferenceConfig(ImageToTextInferenceConfig):
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
        # validate text and vision supported configs
        self.validate_vision_model_supported_configs()

        # copy deepstack_visual_indexes from vision to text config to know which CTE layer should perform deepstack process
        setattr(
            self.text_config,
            "deepstack_visual_indexes",
            copy.deepcopy(self.vision_config.deepstack_visual_indexes),
        )

        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Qwen3VL does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Qwen3VL does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Qwen3VL does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Qwen3VL does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Qwen3VL does not yet support fused speculation.")
        # Qwen3 VL position_id is 3D (3, bs, sl) that does not work with DP and CP in NeuronAttentionBase out-of-box
        if self.text_config.neuron_config.attention_dp_degree > 1:
            raise ValueError("Qwen3VL does not yet support attention data parallel")
        if self.text_config.neuron_config.cp_degree > 1:
            raise ValueError("Qwen3VL does not yet support context parallel")
        if self.text_config.neuron_config.seq_len > 10240:
            # Need to increase CC buffer size for All-Reduce for long seq len
            os.environ["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = f"{140 * 1024 * 1024}"

        if self.neuron_config.flash_decoding_enabled:
            # For Qwen3VL, we use REPLICATE_TO_TP_DEGREE as the sharding_strategy
            # Hence attn_heads are padded to become divisible by tp_degree
            num_attn_heads, num_kv_heads = (
                self.text_config.num_attention_heads,
                self.text_config.num_key_value_heads,
            )
            num_attn_heads = (
                num_attn_heads // self.neuron_config.tp_degree + 1
            ) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

        # Bucket on vision seq len to support dynamic image resolution
        if not self.vision_config.neuron_config.enable_bucketing:
            VISION_SEQ_LENGTH = self.vision_config.neuron_config.seq_len
            # Default vision bucket to a single bucket of VISION_SEQ_LENGTH
            self.vision_config.neuron_config.enable_bucketing = True
            self.vision_config.neuron_config.buckets = generate_buckets(
                VISION_SEQ_LENGTH, VISION_SEQ_LENGTH
            )

        logger.info(f"Bucketing Qwen3 VL vision model on sequence length. Buckets are {self.vision_config.neuron_config.buckets}")

        # Ensure text max context length >= vision max seq length
        # Qwen3VLVisionPatchMerger compress the vision seq len before passing vision embeddings to text model
        vision_seq_len_to_text_model = self.vision_config.neuron_config.seq_len // (self.vision_config.spatial_merge_size**2)
        assert self.text_config.neuron_config.max_context_length >= vision_seq_len_to_text_model, \
            f"Text model max context length {self.text_config.neuron_config.max_context_length} must be no less than compressed vision model sequence length {vision_seq_len_to_text_model}"

    def validate_vision_model_supported_configs(self):
        Qwen3VL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "mlp_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
            "attn_kernel_enabled",
        ]
        for unsupported_config in Qwen3VL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG:
            # attn_kernel_enabled defaults to None, and None means enabled
            if getattr(self.vision_config.neuron_config, unsupported_config, False) is not False:
                setattr(self.vision_config.neuron_config, unsupported_config, False)
                logger.warning(
                    f"Qwen3VL vision model does not yet support '{unsupported_config}'. Will be disabled."
                )

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list

        return [
            "text_config",
            "vision_config",
            "text_config.attention_bias",
            "text_config.attention_dropout",
            "text_config.num_attention_heads",
            "text_config.bos_token_id",
            "text_config.dtype",
            "text_config.eos_token_id",
            "text_config.head_dim",
            "text_config.hidden_act",
            "text_config.hidden_size",
            "text_config.initializer_range",
            "text_config.intermediate_size",
            "text_config.max_position_embeddings",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.rms_norm_eps",
            "text_config.rope_scaling",
            "text_config.rope_theta",
            "text_config.vocab_size",
            "vision_config.deepstack_visual_indexes",
            "vision_config.depth",
            "vision_config.hidden_act",
            "vision_config.hidden_size",
            "vision_config.in_channels",
            "vision_config.initializer_range",
            "vision_config.intermediate_size",
            "vision_config.num_heads",
            "vision_config.num_position_embeddings",
            "vision_config.out_hidden_size",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.temporal_patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Qwen3VLNeuronConfig


class NeuronQwen3VLForCausalLM(NeuronBaseForImageToText):

    text_model_cls = NeuronQwen3VLTextModel
    vision_model_cls = NeuronQwen3VLVisionModel

    text_model_wrapper = NeuronQwen3VLTextModelWrapper
    vision_model_wrapper = NeuronQwen3VLVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )
        self.rope_deltas = None

    def _count_images_per_batch_line(self, input_ids, attention_mask):
        """Count the number of images in each batch line.

        The HF preprocessor inserts <vision_start><image_token>...<vision_end> for each image.
        We count <vision_start> followed by <image_token> to get the image count per batch line.

        This count is used by _split_vision_inputs_by_batch_line to split the flattened
        pixel_values and image_grid_thw (which the HF preprocessor merges across all batch lines)
        back into per-batch-line groups. For example:
          - BS=2, batch_line_0 has 2 images, batch_line_1 has 1 image
          - Returns [2, 1]
          - image_grid_thw has 3 rows total: rows 0-1 belong to batch_line_0, row 2 to batch_line_1
          - pixel_values patches are split by summing T*H*W for each image group

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            List[int]: number of images per batch line
        """
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        images_per_batch_line = []

        for i in range(input_ids.shape[0]):
            ids = input_ids[i]
            if attention_mask is not None:
                ids = ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
            if vision_start_indices.numel() == 0:
                images_per_batch_line.append(0)
            else:
                vision_tokens = ids[vision_start_indices + 1]
                num_images = (vision_tokens == image_token_id).sum().item()
                images_per_batch_line.append(num_images)

        return images_per_batch_line

    def _split_vision_inputs_by_batch_line(self, pixel_values, image_grid_thw, images_per_batch_line):
        """Split flattened pixel_values and image_grid_thw into per-batch-line groups.

        The HF preprocessor flattens all images from all batch lines into:
        - pixel_values: [total_patches, patch_dim] — all images concatenated
        - image_grid_thw: [total_images, 3] — one row per image across all batch lines

        This method uses the per-batch-line image counts to split them:
          - images_per_batch_line = [2, 1] means batch_line_0 has 2 images, batch_line_1 has 1
          - image_grid_thw rows 0:2 go to batch_line_0, row 2:3 to batch_line_1
          - Each image's patch count = T * H * W from its grid_thw row
          - pixel_values is split by cumulative patch counts

        Args:
            pixel_values: [total_patches, patch_dim]
            image_grid_thw: [total_images, 3]
            images_per_batch_line: List[int]

        Returns:
            List[Tuple[Tensor|None, Tensor|None]]: (pixel_values_i, grid_thw_i) per batch line
        """
        result = []
        image_offset = 0
        patch_offset = 0

        for num_images in images_per_batch_line:
            if num_images == 0:
                result.append((None, None))
                continue

            grid_thw_i = image_grid_thw[image_offset : image_offset + num_images]
            num_patches = grid_thw_i.prod(dim=1).sum().item()
            pixel_values_i = pixel_values[patch_offset : patch_offset + num_patches]

            result.append((pixel_values_i, grid_thw_i))

            image_offset += num_images
            patch_offset += num_patches

        return result

    def forward_atomic_prefill(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values,
        image_grid_thw,
        input_capture_hook=None,
        tensor_capture_hook=None,
    ):
        """Execute VE → CTE atomically for a single batch line.

        Ensures vision embeddings written to the on-device buffer by VE are
        immediately consumed by CTE before the next batch line's VE call
        can overwrite them.

        Args:
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
            position_ids: [1, seq_len]
            seq_ids: [1]
            sampling_params: [1, num_params]
            pixel_values: [num_patches, patch_dim] for this batch line, or None
            image_grid_thw: [num_images, 3] for this batch line, or None

        Returns:
            Tuple[CausalLMOutputWithPast, Tensor]: output and rope_deltas for this batch line
        """
        pad_limit = self.get_padding_length(input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            # Compute vision mask for this batch line
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            # Call VE — writes vision embeddings to on-device buffer
            vision_embeddings, deepstack_vision_embeds = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_grid_thw
            )
        else:
            # Text-only batch line
            vision_embeddings, vision_mask, deepstack_vision_embeds = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        # Compute rotary position ids for this batch line
        rotary_position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        # Call CTE — immediately reads vision embeddings from on-device buffer
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            rotary_position_ids=rotary_position_ids,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )

        return output, rope_deltas

    @staticmethod
    def concat_causal_lm_outputs(outputs_list):
        """Concatenate CausalLMOutputWithPast from multiple batch lines.

        Args:
            outputs_list: List[CausalLMOutputWithPast]

        Returns:
            CausalLMOutputWithPast with concatenated logits/tokens
        """
        concatenated_logits = []
        concatenated_hidden_states = []
        concatenated_tokens = []

        for output in outputs_list:
            if isinstance(output.logits, torch.Tensor):
                concatenated_logits.append(output.logits)
            if isinstance(output.hidden_states, torch.Tensor):
                concatenated_hidden_states.append(output.hidden_states)
            elif isinstance(output.hidden_states, list):
                concatenated_hidden_states.extend(output.hidden_states)
            if hasattr(output, 'tokens') and isinstance(output.tokens, torch.Tensor):
                concatenated_tokens.append(output.tokens)

        concatenated_logits = torch.cat(concatenated_logits, dim=0) if len(concatenated_logits) > 0 else None
        concatenated_tokens = torch.cat(concatenated_tokens, dim=0) if len(concatenated_tokens) > 0 else None

        concatenated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatenated_output.tokens = concatenated_tokens
        return concatenated_output

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return f"--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 \
                --internal-max-instruction-limit=15000000"

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return f"--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 \
                --internal-max-instruction-limit=15000000"

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return ["pixel_values", "image_grid_thw"]

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            # Enable pipeline execution for vision model
            pipeline_execution=True,
            return_ranked_to_cpu=False,
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)

        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:

        state_dict = NeuronQwen3VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )
        state_dict = NeuronQwen3VLTextForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.text_config
        )

        return state_dict

    def get_padding_length(self, input_ids):
        # vision inputs should be padded to context encoding model bucket
        buckets = self.context_encoding_model.config.neuron_config.buckets

        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Copied from transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_rope_index()
        Qwen3VL use timestamps rather than absolute time position ids.
        """

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                llm_positions = llm_positions.to(total_input_ids.dtype)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        vision_attention_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if (
            (pixel_values is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0  # check empty pixel_values
        ):
            # Vision+Text Prefill: use atomic prefill to ensure VE→CTE
            # execute back-to-back per batch line, preventing vision embedding
            # overwrites in the shared on-device buffer.
            batch_size = input_ids.shape[0]

            # Split flattened vision inputs into per-batch-line groups
            images_per_batch_line = self._count_images_per_batch_line(input_ids, attention_mask)
            vision_inputs_per_bl = self._split_vision_inputs_by_batch_line(
                pixel_values, image_grid_thw, images_per_batch_line
            )

            # Ensure seq_ids are available for atomic prefill.
            # Each batch line's CTE call must write to a distinct KV cache slot.
            # When seq_ids is None (e.g. HF generate() path), create explicit
            # seq_ids so batch_line_0 → slot 0, batch_line_1 → slot 1, etc.
            if seq_ids is None:
                seq_ids = torch.arange(batch_size)

            outputs = []
            rope_deltas_list = []
            for index in range(batch_size):
                pv_i, grid_thw_i = vision_inputs_per_bl[index]
                output, rope_deltas = self.forward_atomic_prefill(
                    input_ids[index].unsqueeze(0),
                    attention_mask[index].unsqueeze(0) if attention_mask is not None else None,
                    position_ids[index].unsqueeze(0) if position_ids is not None else None,
                    seq_ids[index].unsqueeze(0),
                    sampling_params[index].unsqueeze(0) if sampling_params is not None else None,
                    pv_i,
                    grid_thw_i,
                    input_capture_hook=input_capture_hook,
                    tensor_capture_hook=tensor_capture_hook,
                )
                outputs.append(output)
                rope_deltas_list.append(rope_deltas)

            # Store rope_deltas for all batch lines (used in decode phase)
            self.rope_deltas = torch.cat(rope_deltas_list, dim=0)
            return self.concat_causal_lm_outputs(outputs)

        # Text-only Prefill or Decode phase
        pad_limit = self.get_padding_length(input_ids)
        vision_embeddings, vision_mask, deepstack_vision_embeds = (
            self.text_model_wrapper.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1),
            )
        )

        """
        Qwen3 VL has a new multimodal ROPE in time, height, and width.
        Vision tokens of the same time/height/width position will have the same position id on the corresponding axis.
        In NeuronQwen3VLAttention, `position_ids` is only used to populate and update KV cache. The shape is (bs, seqlen).
        `rotary_position_ids` is used for rope calculation in NeuronQwen3VLRotaryEmbedding. The shape is (3, bs, seqlen).
        3 for time, height, and width axes.

        - Text-only mode
            - rotary_position_id = position_ids
            - rope_deltas = 0
        - Vision+text mode:
            - Prefill: rotary_position_id, rope_deltas = self.get_rope_index()
            - Decode: rotary_position_id = position_ids.add(delta)
        """
        if input_ids.shape[-1] > 1:
            # Prefill phase
            rotary_position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw=None,  # TODO: support video input
                attention_mask=attention_mask,
            )  # shape (3, 1, seq_len) 3 is time, height, width
            self.rope_deltas = rope_deltas

        else:
            # Decode phase
            batch_size, seq_length = input_ids.shape

            """
            Original HF impl in decode stage:

            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            """

            if self.rope_deltas is not None:
                delta = self.rope_deltas.to(input_ids.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            else:
                delta = 0

            # position_ids is already [batch_size, seq_length] with correct absolute
            # positions per batch line. Skip .view(1,-1).expand() which incorrectly
            # merges batch and seq dims when batch_size > 1.
            rotary_position_ids = copy.deepcopy(position_ids)
            rotary_position_ids = rotary_position_ids.add(delta)
            rotary_position_ids = rotary_position_ids.unsqueeze(0).expand(3, -1, -1)

        output_token = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            rotary_position_ids=rotary_position_ids,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )
        return output_token

    @classmethod
    def get_config_cls(cls):
        return Qwen3VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        return NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts, images, processor, role, config
        )
