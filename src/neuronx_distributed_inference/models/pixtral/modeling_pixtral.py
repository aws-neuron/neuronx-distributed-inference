import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

import neuronx_distributed_inference.modules.autobucketing as autobucketing
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaModel, NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.models.image_to_text_model_base import ImageToTextInferenceConfig, NeuronBaseForImageToText
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)
from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import (
    PixtralVisionModelWrapper,
    NeuronPixtralForImageEncoding,
    NeuronPixtralVisionModel,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
logger = logging.getLogger("Neuron")


class PixtralInferenceConfig(ImageToTextInferenceConfig):
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

        if self.text_config.neuron_config.is_block_kv_layout:
            raise ValueError("Pixtral does not yet support block_kv_layout.")
        if self.text_config.neuron_config.is_prefix_caching:
            raise ValueError("Pixtral does not yet support prefix_caching.")
        if self.text_config.neuron_config.is_chunked_prefill:
            raise ValueError("Pixtral does not yet support chunked_prefill.")
        if self.text_config.neuron_config.is_medusa:
            raise ValueError("Pixtral does not yet support medusa.")
        if self.text_config.neuron_config.enable_fused_speculation:
            raise ValueError("Pixtral does not yet support fused speculation.")

        if self.neuron_config.flash_decoding_enabled:
            # For pixtral, we use REPLICATE_TO_TP_DEGREE as the sharding_strategy
            # Hence attn_heads are padded to become divisible by tp_degree
            num_attn_heads, num_kv_heads = self.text_config.num_attention_heads, self.text_config.num_key_value_heads
            num_attn_heads = (num_attn_heads // self.neuron_config.tp_degree + 1) * self.neuron_config.tp_degree
            self.text_config.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def validate_vision_model_supported_configs(self):
        PIXTRAl_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "attn_kernel_enabled",
            "fused_qkv",
            "qkv_kernel_enabled",
            "mlp_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
        ]
        for unsupported_config in PIXTRAl_VISION_MODEL_UNSUPPORTED_NEURON_CONFIG:
            # attn_kernel_enabled defaults to None, and None means enabled
            if getattr(self.vision_config.neuron_config, unsupported_config, False) is not False:
                setattr(self.vision_config.neuron_config, unsupported_config, False)
                logger.warning(f"Pixtral vision model does not yet support '{unsupported_config}'. Will be disabled.")

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list

        return [
            "text_config",
            "vision_config",
            "multimodal_projector_bias",
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
            "vision_config.image_size",
            "vision_config.patch_size",
            "vision_config.num_hidden_layers",
            "vision_config.num_channels",
            "vision_config.hidden_size",
            "vision_config.num_attention_heads",
            "vision_config.rope_theta",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronPixtralTextModel(NeuronLlamaModel):
    """
    The neuron version of the Pixtral Text Model
    """
    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        # Concat vision and text embeddings during context encoding
        # Both inputs_embeds and vision_embeddings should be of the same shape: [BS, Total tokens (image + text), Hidden]
        # And vision_mask should of the shape [BS, Total tokens (image + text), 1]
        # Entries in vision_mask with value `True` represent vision tokens and with value `False` represent text tokens
        # For text-only inputs, vision_mask should be all `False`
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


class NeuronPixtralForCausalLM(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronPixtralTextModel
    vision_model_cls = NeuronPixtralVisionModel

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = PixtralVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls):
        return PixtralInferenceConfig

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            # neuron_config.buckets default to neuron_config.seq_len is not given. For vision we want to do auto-bucketing here
            if new_config.vision_config.neuron_config.buckets == [new_config.vision_config.neuron_config.seq_len] or \
                    new_config.vision_config.neuron_config.buckets is None:
                # 1024 vision seq len corresponds to a single 512x512 image. Smaller bucket size does not make sense in real life.
                if new_config.vision_config.neuron_config.seq_len > 1024:
                    new_config.vision_config.neuron_config.buckets = autobucketing.generate_buckets(
                        1024, new_config.vision_config.neuron_config.seq_len
                    )
                else:
                    new_config.vision_config.neuron_config.buckets = [new_config.vision_config.neuron_config.seq_len]
        # This should not be needed as in vision modeling code we should always use vision_config.neuron_config as vision model's neuron config
        # added this line just to add insurance to avoid mix-up
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # text model state dict convertion
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            if 'language_model.model.' in dict_key:
                new_key = dict_key.replace('language_model.model.', "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            replacement_atten_key = attention_keys[atten_key]
                            new_key = new_key.replace(atten_key, replacement_atten_key)
                new_state_dict[new_key] = state_dict[dict_key]
            elif 'language_model.' in dict_key:
                new_key = dict_key.replace('language_model.', "")
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]
        state_dict = NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(
            new_state_dict, inference_config.text_config
        )

        # vision model state dict convertion
        state_dict = NeuronPixtralForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )

        return state_dict

    def get_padding_length(self, input_ids):
        # vision inputs should be padded to context encoding model bucket
        buckets = self.context_encoding_model.config.neuron_config.buckets

        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
            "image_sizes",
        ]

    def concat_causal_lm_outputs(self, outputs_list):
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
        concatenated_tokens = torch.cat(concatenated_tokens, dim=0) if len(concatenated_tokens) else None

        concatentated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatentated_output.tokens = concatenated_tokens
        return concatentated_output

    def forward_atomic_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None
    ):
        if image_sizes is None:
            assert len(pixel_values.shape) == 4, "Pixel value shape is expected to be [batch_size, num_channels, img_height, img_width]"
            img_hight = pixel_values.shape[2]
            img_width = pixel_values.shape[3]
            image_sizes = torch.tensor([[img_hight, img_width]], dtype=torch.int32)

        if vision_mask is None:
            vision_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
        # Convert vision mask from bool to indices
        assert (
            vision_mask.dtype == torch.bool
        ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"
        vision_mask = generate_positions_from_mask(vision_mask.squeeze())

        vision_embeddings = self.vision_encoder_model(
            pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_sizes
        ).to(self.text_config.neuron_config.torch_dtype)

        # Pad vision embeddings and vision mask to corresponding text bucket
        pad_limit = self.get_padding_length(input_ids)
        print(f"vision_mask shape: {vision_mask.shape}, pad_limit: {pad_limit}, input_ids shape: {input_ids.shape}")
        vision_mask = pad_positions(
            vision_mask, pad_limit, (pad_limit - 1)
        )
        vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def get_batch_line_mm_input(self, mm_input, index):
        if mm_input is None:
            return None
        elif isinstance(mm_input, list):
            return mm_input[index]
        elif isinstance(mm_input, torch.Tensor):
            return mm_input[index].unsqueeze(0)
        else:
            raise ValueError(f"Unsupported type for mm_input:{type(mm_input)}, expecting list, tensor, or None.")

    def check_empty_pixel_values(self, pixel_values):
        print(f"supported type for pixel_values {type(pixel_values)}.")
        if pixel_values is None:
            return True
        elif isinstance(pixel_values, torch.Tensor):
            return (pixel_values.sum() == 0)
        elif isinstance(pixel_values, list):
            # return False (not empty) if any pixel_value has sum != 0
            for pixel_value in pixel_values:
                if (pixel_value.sum() != 0):
                    return False
            # return True (empty) if all pixel_values are empty
            return True
        else:
            raise ValueError(f"Unsupported type for pixel_values {type(pixel_values)}, expecting list, tensor, or None.")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (
            input_ids.shape[-1] > 1
            and not self.check_empty_pixel_values(pixel_values)
        ):  # call vision encoder
            outputs = []
            for index in range(input_ids.shape[0]):
                outputs.append(
                    self.forward_atomic_prefill(
                        input_ids[index].unsqueeze(0),
                        attention_mask[index].unsqueeze(0) if (attention_mask is not None) else attention_mask,
                        position_ids[index].unsqueeze(0) if (position_ids is not None) else position_ids,
                        seq_ids[index].unsqueeze(0) if (seq_ids is not None) else seq_ids,
                        sampling_params[index].unsqueeze(0) if (sampling_params is not None) else sampling_params,
                        self.get_batch_line_mm_input(pixel_values, index),
                        self.get_batch_line_mm_input(vision_mask, index),
                        self.get_batch_line_mm_input(image_sizes, index),
                    )
                )
            return self.concat_causal_lm_outputs(outputs)
        else:
            pad_limit = self.get_padding_length(input_ids)
            vision_embeddings, vision_mask = self.text_model_wrapper.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1)
            )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
            )

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)

    def to_cpu(self):
        raise NotImplementedError("to_cpu() is not implemented")
