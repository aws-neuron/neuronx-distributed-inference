import logging
import torch
import copy
from typing import Dict, List, Optional, Type, Callable, Union, Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import ImageToTextInferenceConfig, NeuronBaseForImageToText
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import NeuronQwen2VLForImageEncoding, NeuronQwen2VisionModel, Qwen2VLVisionModelWrapper
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import NeuronQwen2VLTextModel, Qwen2VLTextModelWrapper, NeuronQwen2VLTextForCausalLM

from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
)
from neuronx_distributed_inference.models.qwen2_vl.utils.constants import (
    DEFAULT_PIXELS_PER_IMAGE
)

logger = logging.getLogger("Neuron")

QWEN2_VL_TEXT_CONFIG_KEYS = [
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "vocab_size",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "rope_scaling",
    "hidden_act",
    "bos_token_id",
    "eos_token_id",
    "qkv_bias",
    "o_bias",
    "intermediate_size",
    "vision_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
]


class Qwen2VLInferenceConfig(ImageToTextInferenceConfig):
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

        # Qwen2-VL-specific config settings
        self.add_special_config()

        # validate text and vision supported configs
        self.validate_model_supported_configs()

    def validate_model_supported_configs(self):
        # check self.config contain self.config.text_config
        for key in QWEN2_VL_TEXT_CONFIG_KEYS:
            if hasattr(self, key):
                assert getattr(self, key) == getattr(self.text_config, key)

        QWEN2VL_TEXT_MODEL_UNSUPPORTED_NEURON_CONFIG = [
            "is_block_kv_layout",
            "is_prefix_caching",
            "is_chunked_prefill",
            "is_medusa",
            "enable_fused_speculation"
        ]

        for unsupported_config in QWEN2VL_TEXT_MODEL_UNSUPPORTED_NEURON_CONFIG:
            if getattr(self.text_config.neuron_config, unsupported_config, False) is not False:
                setattr(self.text_config.neuron_config, unsupported_config, False)
                logger.warning(f"Qwen2VL text model does not yet support '{unsupported_config}'. Will be disabled.")

        QWEN2VL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "qkv_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
        ]
        for unsupported_config in QWEN2VL_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG:
            # attn_kernel_enabled defaults to None, and None means enabled
            if getattr(self.vision_config.neuron_config, unsupported_config, False) is not False:
                setattr(self.vision_config.neuron_config, unsupported_config, False)
                logger.warning(f'Qwen2VL vision config validation: {unsupported_config} unsupported, disabling')

        # assertions
        # make sure max vision pixels no greater than text bucket
        if self.vision_config.neuron_config.buckets and self.text_config.neuron_config.buckets:
            assert (self.vision_config.neuron_config.buckets[-1] * DEFAULT_PIXELS_PER_IMAGE // 4) <= self.text_config.neuron_config.buckets[-1]
        assert self.vision_config.neuron_config.fused_qkv is True

    def add_special_config(self):
        # qwen2_vl-specific config setting
        self.num_cores_per_group = 1

        self.qkv_bias = True
        self.o_bias = False

        self.vision_config.head_dim = self.vision_config.embed_dim // self.vision_config.num_heads

        # copy to text_config
        for key in QWEN2_VL_TEXT_CONFIG_KEYS:
            setattr(self.text_config, key, getattr(self, key))

        self.pad_token_id = self.text_config.pad_token_id

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list

        return [
            "text_config",
            "vision_config",
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
            "vision_config.depth",
            "vision_config.mlp_ratio",
            "vision_config.num_heads",
            "vision_config.in_channels",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.temporal_patch_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen2VLForCausalLM(NeuronBaseForImageToText):

    text_model_cls = NeuronQwen2VLTextModel
    vision_model_cls = NeuronQwen2VisionModel

    text_model_wrapper = Qwen2VLTextModelWrapper
    vision_model_wrapper = Qwen2VLVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return f"--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return f"--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
            "image_grid_thw"
        ]

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=False
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)

        return model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: Qwen2VLInferenceConfig
    ) -> dict:

        state_dict = NeuronQwen2VLForImageEncoding.convert_hf_to_neuron_state_dict(state_dict, inference_config)
        state_dict = NeuronQwen2VLTextForCausalLM.convert_hf_to_neuron_state_dict(state_dict, inference_config.text_config)

        return state_dict

    def get_padding_length(self, input_ids):
        # vision inputs should be padded to context encoding model bucket
        buckets = self.context_encoding_model.config.neuron_config.buckets

        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

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
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        pad_limit = self.get_padding_length(input_ids)
        if (
            (pixel_values is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0  # check empty pixel_values
        ):  # call vision encoder
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(
                vision_mask, pad_limit, (pad_limit - 1)
            )

            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_grid_thw
            )

        else:
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
            input_capture_hook=input_capture_hook,
            tensor_capture_hook=tensor_capture_hook,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )
        return output_token

    @classmethod
    def get_config_cls(cls):
        return Qwen2VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        return NeuronQwen2VLForImageEncoding.prepare_input_args(
            prompts, images, processor, role, config
        )
