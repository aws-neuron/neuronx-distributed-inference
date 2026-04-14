# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5-Omni support for NXD inference.
#
# Provides two modes:
#   1. Text-only (Thinker): NeuronQwen25OmniForCausalLM
#      - Reuses Qwen2 decoder with thinker.model.* prefix remapping
#   2. Multimodal (Vision + Text): NeuronQwen25OmniMultimodalForCausalLM
#      - Vision encoder: Qwen2.5-Omni ViT (SwiGLU, RMSNorm, separate QKV)
#      - Text decoder: Qwen2-VL text model (multimodal RoPE)
#
# Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B

"""Qwen2.5-Omni model for NXD inference."""

import copy
import gc
import logging
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import (
    NeuronQwen2ForCausalLM,
    NeuronQwen2Model,
    convert_state_dict_to_fused_qkv,
)

logger = logging.getLogger("Neuron")


# Attributes to extract from the thinker's text_config to the top-level config.
_TEXT_CONFIG_ATTRS = [
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
    "hidden_act",
    "tie_word_embeddings",
    "max_window_layers",
    "use_sliding_window",
    "sliding_window",
]


class Qwen25OmniInferenceConfig(InferenceConfig):
    """Inference config for Qwen2.5-Omni (Thinker text component).

    Handles the nested config structure: the HF config has attributes under
    thinker_config.text_config that we need at the top level for NxDI.
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Qwen2.5-Omni text model has QKV bias but no output projection bias
        self.qkv_bias = True
        self.o_bias = False

        # Extract text config attributes from nested thinker_config
        if hasattr(self, "thinker_config"):
            thinker_cfg = self.thinker_config
            # When loaded from saved JSON, thinker_config is a plain dict
            if isinstance(thinker_cfg, dict):
                thinker_cfg = SimpleNamespace(**thinker_cfg)
                self.thinker_config = thinker_cfg

            text_cfg = thinker_cfg.text_config
            if isinstance(text_cfg, dict):
                text_cfg = SimpleNamespace(**text_cfg)
                thinker_cfg.text_config = text_cfg

            # Text config attributes always take precedence over top-level
            # defaults from PretrainedConfig (e.g. tie_word_embeddings defaults
            # to True at the top level but is False in text_config).
            for attr in _TEXT_CONFIG_ATTRS:
                if hasattr(text_cfg, attr):
                    setattr(self, attr, getattr(text_cfg, attr))

            # Set pad_token_id from thinker_config
            if hasattr(thinker_cfg, "pad_token_id"):
                if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
                    self.pad_token_id = thinker_cfg.pad_token_id

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
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen25OmniForCausalLM(NeuronQwen2ForCausalLM):
    """Qwen2.5-Omni Thinker text model for Causal LM on Neuron.

    Reuses the Qwen2 model architecture since the Thinker's text backbone
    is architecturally identical to Qwen2.5. The main differences are:
      - Weight keys are prefixed with 'thinker.model.' / 'thinker.lm_head.'
      - Non-text weights (talker, token2wav, audio_tower, visual) are discarded
    """

    _model_cls = NeuronQwen2Model
    _STATE_DICT_MODEL_PREFIX = "thinker.model."

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        """Load the full Qwen2.5-Omni model from HuggingFace.

        Note: We load the full model and filter to thinker text weights
        in convert_hf_to_neuron_state_dict.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[Qwen25OmniInferenceConfig]:
        return Qwen25OmniInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, Any],
        config: Qwen25OmniInferenceConfig,
    ) -> Dict[str, Any]:
        """Convert Qwen2.5-Omni state dict to NxDI format.

        1. Keep only thinker text model weights (discard talker, token2wav, audio, visual)
        2. Map 'thinker.lm_head.' -> 'lm_head.'
        3. Apply standard Qwen2 conversions (fused QKV, rank utils, etc.)
        """
        neuron_config = config.neuron_config

        # Filter: keep only thinker text weights and lm_head
        # After base-class prefix stripping, 'thinker.model.*' becomes '*'
        # but 'thinker.lm_head.*' is NOT stripped, so handle it here.
        keys_to_remove = []
        keys_to_rename = {}
        for key in state_dict:
            if key.startswith("thinker.lm_head."):
                # Map thinker.lm_head.weight -> lm_head.weight
                new_key = key.replace("thinker.lm_head.", "lm_head.", 1)
                keys_to_rename[key] = new_key
            elif not key.startswith(("layers.", "embed_tokens.", "norm.", "lm_head.")):
                # After base-class prefix stripping, valid text keys start with
                # layers.*, embed_tokens.*, norm.*, or lm_head.*
                # Everything else (talker, token2wav, audio_tower, visual) should be removed
                keys_to_remove.append(key)

        # Apply renames
        for old_key, new_key in keys_to_rename.items():
            state_dict[new_key] = state_dict.pop(old_key)

        # Remove non-text weights
        for key in keys_to_remove:
            del state_dict[key]

        gc.collect()
        logger.info(
            "Filtered state dict to %d thinker text model weights", len(state_dict)
        )

        # Add rank utilities (same as Qwen2)
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    def get_compiler_args(self):
        compiler_args = (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            "--auto-cast=none "
            "--model-type transformer "
            "-O1"
        )
        compiler_args += (
            " --tensorizer-options='"
            "--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 "
            "--vectorize-strided-dma'"
        )
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args


# ---------------------------------------------------------------------------
# Multimodal (Vision + Text) support
# ---------------------------------------------------------------------------

# Keys from thinker_config.text_config to copy to the top-level multimodal config
_MULTIMODAL_TEXT_CONFIG_KEYS = [
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
    "image_token_id",
    "vision_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
]


class Qwen25OmniMultimodalInferenceConfig(ImageToTextInferenceConfig):
    """Inference config for Qwen2.5-Omni multimodal (vision + text).

    Handles the nested config structure where text_config and vision_config
    are under thinker_config. Extracts them to the top level as required
    by ImageToTextInferenceConfig.
    """

    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        # Extract text_config and vision_config from thinker_config
        # The HF config nests them: thinker_config.text_config, thinker_config.vision_config
        thinker = kwargs.get("thinker_config", None)
        if thinker is not None:
            if hasattr(thinker, "__dict__") and not isinstance(thinker, dict):
                thinker = vars(thinker)
            if isinstance(thinker, dict):
                if "text_config" not in kwargs and "text_config" in thinker:
                    tc = thinker["text_config"]
                    kwargs["text_config"] = (
                        vars(tc) if hasattr(tc, "__dict__") and not isinstance(tc, dict) else tc
                    )
                if "vision_config" not in kwargs and "vision_config" in thinker:
                    vc = thinker["vision_config"]
                    kwargs["vision_config"] = (
                        vars(vc) if hasattr(vc, "__dict__") and not isinstance(vc, dict) else vc
                    )
                # Extract audio_config from thinker_config
                if "audio_config" not in kwargs and "audio_config" in thinker:
                    ac = thinker["audio_config"]
                    kwargs["audio_config"] = (
                        vars(ac) if hasattr(ac, "__dict__") and not isinstance(ac, dict) else ac
                    )
                # Extract special token IDs from thinker_config
                for token_key in [
                    "image_token_index", "audio_token_index", "video_token_index",
                    "audio_start_token_id", "audio_end_token_id",
                    "vision_start_token_id", "vision_end_token_id",
                    "vision_token_id", "pad_token_id",
                ]:
                    if token_key in thinker and token_key not in kwargs:
                        kwargs[token_key] = thinker[token_key]

        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        self.add_special_config()
        self.validate_model_supported_configs()

    def add_special_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False

        # Vision config derived attributes
        self.vision_config.head_dim = (
            self.vision_config.embed_dim // self.vision_config.num_heads
        )
        self.vision_config.num_cores_per_group = 1

        # Vision encoder MUST use separate Q/K/V (not fused)
        if getattr(self.vision_config.neuron_config, "fused_qkv", True):
            self.vision_config.neuron_config.fused_qkv = False
            logger.info(
                "Qwen2.5-Omni vision encoder: set fused_qkv=False "
                "(separate Q/K/V projections)"
            )

        # Copy text config keys to top-level (for compatibility)
        for key in _MULTIMODAL_TEXT_CONFIG_KEYS:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))

        # Map Qwen2.5-Omni token IDs to Qwen2-VL compatible names
        if hasattr(self, "image_token_index"):
            self.image_token_id = self.image_token_index
            self.text_config.image_token_id = self.image_token_index
        if hasattr(self, "video_token_index"):
            self.video_token_id = self.video_token_index
            self.text_config.video_token_id = self.video_token_index
        if hasattr(self, "audio_token_index"):
            self.audio_token_id = self.audio_token_index
            self.text_config.audio_token_id = self.audio_token_index

        # Set pad_token_id
        if hasattr(self, "pad_token_id"):
            self.text_config.pad_token_id = self.pad_token_id

        # Store audio_config as SimpleNamespace for attribute access
        if hasattr(self, "audio_config") and isinstance(self.audio_config, dict):
            self.audio_config = SimpleNamespace(**self.audio_config)

    def validate_model_supported_configs(self):
        # Disable unsupported features for text model
        unsupported_text = [
            "is_prefix_caching",
            "is_chunked_prefill",
            "is_medusa",
            "enable_fused_speculation",
        ]
        for cfg_name in unsupported_text:
            if getattr(self.text_config.neuron_config, cfg_name, False):
                setattr(self.text_config.neuron_config, cfg_name, False)
                logger.warning(
                    f"Qwen2.5-Omni text model does not support "
                    f"'{cfg_name}'. Disabled."
                )

        # Disable unsupported features for vision model
        unsupported_vision = [
            "sequence_parallel_enabled",
            "flash_decoding_enabled",
            "qkv_kernel_enabled",
        ]
        for cfg_name in unsupported_vision:
            if getattr(self.vision_config.neuron_config, cfg_name, False):
                setattr(self.vision_config.neuron_config, cfg_name, False)
                logger.warning(
                    f"Qwen2.5-Omni vision model does not support "
                    f"'{cfg_name}'. Disabled."
                )

    def get_required_attributes(self) -> List[str]:
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
            "vision_config.embed_dim",
            "vision_config.num_heads",
            "vision_config.in_channels",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.temporal_patch_size",
            "vision_config.out_hidden_size",
            "vision_config.intermediate_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen25OmniMultimodalForCausalLM(NeuronBaseForImageToText):
    """Qwen2.5-Omni multimodal model (vision encoder + text decoder) on Neuron.

    Reuses Qwen2-VL text model components (same mRoPE architecture) and
    the Qwen2.5-Omni vision encoder (SwiGLU, RMSNorm, separate QKV).
    """

    # Import lazily to avoid circular imports
    @staticmethod
    def _get_text_model_cls():
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
            NeuronQwen2VLTextModel,
        )
        return NeuronQwen2VLTextModel

    @staticmethod
    def _get_text_model_wrapper():
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
            Qwen2VLTextModelWrapper,
        )
        return Qwen2VLTextModelWrapper

    @staticmethod
    def _get_vision_model_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_vision import (
            NeuronQwen25OmniVisionModel,
        )
        return NeuronQwen25OmniVisionModel

    @staticmethod
    def _get_vision_model_wrapper():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_vision import (
            Qwen25OmniVisionModelWrapper,
        )
        return Qwen25OmniVisionModelWrapper

    @staticmethod
    def _get_audio_encoder_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_audio import (
            NeuronQwen25OmniAudioEncoder,
        )
        return NeuronQwen25OmniAudioEncoder

    @staticmethod
    def _get_talker_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_talker import (
            NeuronQwen25OmniTalker,
        )
        return NeuronQwen25OmniTalker

    @staticmethod
    def _get_neuron_talker_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_talker import (
            NeuronQwen25OmniTalkerForCausalLM,
        )
        return NeuronQwen25OmniTalkerForCausalLM

    @staticmethod
    def _get_talker_config_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_talker import (
            TalkerInferenceConfig,
        )
        return TalkerInferenceConfig

    @staticmethod
    def _get_thinker_projection_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_talker import (
            ThinkerToTalkerProjection,
        )
        return ThinkerToTalkerProjection

    @staticmethod
    def _get_token2wav_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_token2wav import (
            NeuronQwen25OmniToken2Wav,
        )
        return NeuronQwen25OmniToken2Wav

    @staticmethod
    def _get_neuron_token2wav_cls():
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_token2wav import (
            NeuronQwen25OmniToken2WavWithNeuronDiT,
        )
        return NeuronQwen25OmniToken2WavWithNeuronDiT

    def __init__(self, *args, **kwargs):
        super().__init__(
            self._get_text_model_cls(),
            self._get_vision_model_cls(),
            self._get_text_model_wrapper(),
            self._get_vision_model_wrapper(),
            *args,
            **kwargs,
        )
        self.audio_encoder = None
        self.talker = None
        self.token2wav = None
        self.speaker_map = {}

    def get_vision_compiler_args(self) -> str:
        return (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 ' -O1 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_compiler_args(self) -> str:
        return (
            "--enable-saturate-infinity "
            "--enable-mixed-precision-accumulation "
            "--auto-cast=none --model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 "
            "--vectorize-strided-dma' "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "vision_mask", "image_grid_thw"]

    def enable_vision_encoder(
        self, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        new_config = copy.deepcopy(self.config)
        self.vision_encoder_model = self._get_vision_model_wrapper()(
            config=new_config,
            model_cls=self._get_vision_model_cls(),
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=False,
        )
        self.vision_models.append(self.vision_encoder_model)

    def enable_audio_encoder(self, state_dict=None):
        """Initialize the audio encoder (CPU frontend/postprocessor + Neuron transformer).

        The audio encoder is a hybrid CPU+Neuron module:
          - CPU: Conv1d frontend, positional embeddings, chunking
          - Neuron (TP=4): 32 transformer layers with block-diagonal attention
          - CPU: AvgPool, LayerNorm, Linear projection

        The CPU components are loaded from the converted state dict.
        The Neuron transformer must be compiled/loaded separately via
        compile_audio_encoder() and load_audio_encoder().

        Args:
            state_dict: Converted state dict containing audio_tower.* keys
                (already split into frontend.*, transformer.*, postprocessor.*).
                If None, the encoder is created with random weights.
        """
        audio_config = getattr(self.config, "audio_config", None)
        if audio_config is None:
            logger.warning(
                "No audio_config found in model config. "
                "Audio encoder will not be initialized."
            )
            return

        AudioEncoderCls = self._get_audio_encoder_cls()
        dtype = torch.bfloat16
        if hasattr(self.config, "neuron_config"):
            dtype = getattr(self.config.neuron_config, "torch_dtype", dtype)

        if state_dict is not None:
            self.audio_encoder = AudioEncoderCls.from_pretrained_state_dict(
                audio_config, state_dict, dtype=dtype
            )
        else:
            self.audio_encoder = AudioEncoderCls(audio_config, dtype=dtype)

        self.audio_encoder.eval()
        logger.info(
            "Audio encoder initialized (CPU frontend/postprocessor, "
            "Neuron transformer pending compile/load)"
        )

    def compile_audio_encoder(self, compiled_model_path, audio_neuron_config=None):
        """Compile the audio encoder transformer layers on Neuron.

        Args:
            compiled_model_path: Path to save compiled model artifacts.
            audio_neuron_config: Optional NeuronConfig for audio transformer.
                If None, creates a default config with TP matching the text model.
        """
        if self.audio_encoder is None:
            raise RuntimeError("Call enable_audio_encoder() first")

        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_audio import (
            AudioEncoderInferenceConfig,
            NeuronQwen25OmniForAudioEncoding,
        )

        audio_config = getattr(self.config, "audio_config", None)
        if isinstance(audio_config, dict):
            from types import SimpleNamespace
            audio_config = SimpleNamespace(**audio_config)

        if audio_neuron_config is None:
            # Default: match text model TP degree, reasonable seq_len buckets
            tp_degree = self.neuron_config.tp_degree
            audio_neuron_config = NeuronConfig(
                tp_degree=tp_degree,
                torch_dtype=self.neuron_config.torch_dtype,
                batch_size=1,
                # Audio seq_len buckets: typical values after conv
                buckets=[256, 512, 1024, 1500],
            )

        audio_inf_config = AudioEncoderInferenceConfig(
            neuron_config=audio_neuron_config,
            audio_config=vars(audio_config) if hasattr(audio_config, '__dict__') else audio_config,
        )

        audio_app = NeuronQwen25OmniForAudioEncoding(audio_inf_config)
        audio_app.compile(compiled_model_path)

        logger.info("Audio encoder transformer compiled to %s", compiled_model_path)
        return audio_app

    def load_audio_encoder(self, compiled_model_path, audio_app=None):
        """Load compiled audio encoder transformer layers.

        Args:
            compiled_model_path: Path to compiled model artifacts.
            audio_app: Optional pre-compiled NeuronQwen25OmniForAudioEncoding.
                If None, creates and loads from compiled_model_path.
        """
        if self.audio_encoder is None:
            raise RuntimeError("Call enable_audio_encoder() first")

        if audio_app is None:
            from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_audio import (
                AudioEncoderInferenceConfig,
                NeuronQwen25OmniForAudioEncoding,
            )
            # Load from compiled artifacts
            audio_app = NeuronQwen25OmniForAudioEncoding.load(compiled_model_path)

        self.audio_encoder.transformer = audio_app.model
        logger.info("Audio encoder transformer loaded from %s", compiled_model_path)

    def enable_talker(self, state_dict=None, use_neuron=False):
        """Initialize the Talker model.

        The Talker converts Thinker hidden states into codec tokens for
        speech synthesis.

        Args:
            state_dict: Converted state dict containing talker keys.
                If None, the Talker is created with random weights.
            use_neuron: If True, use the Neuron-compiled Talker
                (NeuronQwen25OmniTalkerForCausalLM). Requires separate
                compilation with compile_talker() / load_talker().
                If False (default), use the CPU-based HF wrapper.
        """
        talker_config = getattr(self.config, "talker_config", None)
        if talker_config is None:
            logger.warning(
                "No talker_config found in model config. "
                "Talker will not be initialized."
            )
            return

        if use_neuron:
            # Neuron Talker — just mark for later compile/load
            logger.info(
                "Neuron Talker mode enabled. Call compile_talker() or "
                "load_talker() to activate."
            )
            self._talker_use_neuron = True
            self._talker_state_dict = state_dict
            # Initialize CPU projection for thinker states
            if state_dict is not None:
                ProjCls = self._get_thinker_projection_cls()
                self.thinker_to_talker_proj = ProjCls.from_state_dict(state_dict)
                logger.info("Thinker→Talker CPU projection initialized")
        else:
            # CPU Talker (default)
            TalkerCls = self._get_talker_cls()
            dtype = torch.bfloat16

            if state_dict is not None:
                self.talker = TalkerCls.from_pretrained_state_dict(
                    talker_config, state_dict, dtype=dtype
                )
            else:
                self.talker = TalkerCls(talker_config, dtype=dtype)

            logger.info("Talker initialized on CPU")

    def compile_talker(self, compiled_model_path, talker_neuron_config=None):
        """Compile the Talker on Neuron.

        Creates a NeuronQwen25OmniTalkerForCausalLM, converts the state
        dict (fusing embedding + projection), and compiles.

        Recommended neuron_config:
          - tp_degree=4 (12 Q heads / 4 = 3 per rank)
          - batch_size=1
          - seq_len=4096 (max codec tokens)

        Args:
            compiled_model_path: Path to save compiled model
            talker_neuron_config: NeuronConfig for Talker compilation.
                If None, creates a default config with TP=4.
        """
        talker_config = getattr(self.config, "talker_config", None)
        if talker_config is None:
            raise RuntimeError("No talker_config in model config")

        NeuronTalkerCls = self._get_neuron_talker_cls()
        TalkerConfigCls = self._get_talker_config_cls()

        if talker_neuron_config is None:
            from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_talker import (
                TalkerNeuronConfig,
            )
            talker_neuron_config = TalkerNeuronConfig(
                tp_degree=4,
                batch_size=1,
                seq_len=4096,
                torch_dtype=torch.bfloat16,
            )

        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        inference_config = TalkerConfigCls(
            neuron_config=talker_neuron_config,
            load_config=load_pretrained_config(hf_config=talker_config),
        )

        app = NeuronTalkerCls(self.model_path, config=inference_config)
        app.compile(compiled_model_path)
        logger.info("Talker compiled to %s", compiled_model_path)

    def load_talker(self, compiled_model_path, talker_app=None):
        """Load a compiled Talker from disk.

        Args:
            compiled_model_path: Path to compiled model artifacts.
            talker_app: Optional pre-compiled NeuronQwen25OmniTalkerForCausalLM.
        """
        if talker_app is None:
            NeuronTalkerCls = self._get_neuron_talker_cls()
            talker_app = NeuronTalkerCls.load(compiled_model_path)

        self.talker = talker_app
        logger.info("Neuron Talker loaded from %s", compiled_model_path)

    def enable_token2wav(
        self, state_dict=None, speaker_dict_path=None, use_neuron_dit=False
    ):
        """Initialize the Token2Wav vocoder.

        Token2Wav converts codec tokens from the Talker into audio
        waveforms using DiT + BigVGAN.

        Args:
            state_dict: Converted state dict containing token2wav keys.
                If None, Token2Wav is created with random weights.
            speaker_dict_path: Path to spk_dict.pt for speaker conditioning.
                If provided, loads the speaker map for audio generation.
            use_neuron_dit: If True, use the Neuron-accelerated version
                (DiT on Neuron, ODE loop + BigVGAN on CPU).
                Call compile_token2wav_dit() / load_token2wav_dit() to compile.
        """
        token2wav_config = getattr(self.config, "token2wav_config", None)
        if token2wav_config is None:
            logger.warning(
                "No token2wav_config found in model config. "
                "Token2Wav will not be initialized."
            )
            return

        if use_neuron_dit:
            Token2WavCls = self._get_neuron_token2wav_cls()
        else:
            Token2WavCls = self._get_token2wav_cls()

        if state_dict is not None:
            self.token2wav = Token2WavCls.from_pretrained_state_dict(
                token2wav_config, state_dict
            )
        else:
            self.token2wav = Token2WavCls(token2wav_config)

        if speaker_dict_path is not None:
            self.speaker_map = Token2WavCls.load_speaker_dict(speaker_dict_path)
            logger.info(
                "Loaded %d speakers: %s",
                len(self.speaker_map),
                list(self.speaker_map.keys()),
            )

        mode = "CPU (Neuron DiT capable)" if use_neuron_dit else "CPU"
        logger.info("Token2Wav initialized on %s (float32)", mode)

    def compile_token2wav_dit(self, compiled_path, **kwargs):
        """Compile the Token2Wav DiT on Neuron.

        Requires enable_token2wav(use_neuron_dit=True) to be called first.

        Args:
            compiled_path: Directory to save compiled DiT
            **kwargs: Additional args passed to compile_dit()
                (max_codec_len, max_mel_len, batch_size)
        """
        if self.token2wav is None:
            raise RuntimeError("Call enable_token2wav(use_neuron_dit=True) first")
        if not hasattr(self.token2wav, "compile_dit"):
            raise RuntimeError(
                "Token2Wav is not Neuron DiT capable. "
                "Call enable_token2wav(use_neuron_dit=True)"
            )
        self.token2wav.compile_dit(compiled_path, **kwargs)
        logger.info("Token2Wav DiT compiled to %s", compiled_path)

    def load_token2wav_dit(self, compiled_path):
        """Load a compiled Token2Wav DiT.

        Args:
            compiled_path: Directory containing compiled DiT
        """
        if self.token2wav is None:
            raise RuntimeError("Call enable_token2wav(use_neuron_dit=True) first")
        if not hasattr(self.token2wav, "load_dit"):
            raise RuntimeError(
                "Token2Wav is not Neuron DiT capable. "
                "Call enable_token2wav(use_neuron_dit=True)"
            )
        self.token2wav.load_dit(compiled_path)
        logger.info("Token2Wav DiT loaded from %s", compiled_path)

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict,
        inference_config: Qwen25OmniMultimodalInferenceConfig,
        include_talker: bool = False,
        include_token2wav: bool = False,
    ) -> dict:
        """Convert Qwen2.5-Omni full state dict to NxDI format.

        1. Remap thinker.* prefixes to Qwen2-VL compatible format
        2. Apply vision encoder conversion (separate Q/K/V, SwiGLU MLP)
        3. Apply audio encoder conversion (strip prefix, cast dtype)
        4. Apply text model conversion (fused QKV, attention key renames)
        5. Optionally strip talker.* and token2wav.* prefixes

        Args:
            state_dict: Full HF state dict
            inference_config: Multimodal inference config
            include_talker: If True, include talker keys (stripped prefix)
            include_token2wav: If True, include token2wav keys (stripped prefix)
        """
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_audio import (
            NeuronQwen25OmniAudioEncoder,
        )
        from neuronx_distributed_inference.models.qwen25_omni.modeling_qwen25_omni_vision import (
            NeuronQwen25OmniForImageEncoding,
        )
        from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text import (
            NeuronQwen2VLTextForCausalLM,
        )

        # Step 1: Remap thinker.* prefixes, optionally keep talker/token2wav
        remapped = {}
        talker_state = {}
        token2wav_state = {}
        for key, value in state_dict.items():
            if key.startswith("thinker.model."):
                remapped["model." + key[len("thinker.model."):]] = value
            elif key.startswith("thinker.lm_head."):
                remapped[key[len("thinker."):]] = value
            elif key.startswith("thinker.visual."):
                remapped["visual." + key[len("thinker.visual."):]] = value
            elif key.startswith("thinker.audio_tower."):
                remapped["audio_tower." + key[len("thinker.audio_tower."):]] = value
            elif key.startswith("talker.") and include_talker:
                talker_state[key[len("talker."):]] = value
            elif key.startswith("token2wav.") and include_token2wav:
                token2wav_state[key[len("token2wav."):]] = value

        del state_dict
        gc.collect()

        logger.info(
            "Remapped %d thinker keys%s%s",
            len(remapped),
            " + %d talker keys" % len(talker_state) if talker_state else "",
            " + %d token2wav keys" % len(token2wav_state) if token2wav_state else "",
        )

        # Step 2: Vision encoder conversion
        remapped = NeuronQwen25OmniForImageEncoding.convert_hf_to_neuron_state_dict(
            remapped, inference_config
        )

        # Step 3: Audio encoder conversion (strip prefix, cast dtype)
        audio_dtype = getattr(
            inference_config, "torch_dtype", torch.bfloat16
        )
        if hasattr(inference_config, "neuron_config"):
            audio_dtype = getattr(
                inference_config.neuron_config, "torch_dtype", audio_dtype
            )
        remapped = NeuronQwen25OmniAudioEncoder.convert_hf_to_neuron_state_dict(
            remapped, dtype=audio_dtype
        )

        # Step 4: Text model conversion
        remapped = NeuronQwen2VLTextForCausalLM.convert_hf_to_neuron_state_dict(
            remapped, inference_config.text_config
        )

        # Step 5: Merge talker and token2wav state dicts
        if talker_state:
            for k, v in talker_state.items():
                remapped["talker." + k] = v
            logger.info("Included %d talker keys in output", len(talker_state))
        if token2wav_state:
            for k, v in token2wav_state.items():
                remapped["token2wav." + k] = v
            logger.info("Included %d token2wav keys in output", len(token2wav_state))

        return remapped

    def get_padding_length(self, input_ids):
        """Get the context encoding bucket size for given input_ids."""
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
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
            generate_positions_from_mask,
            pad_positions,
        )

        pad_limit = self.get_padding_length(input_ids)
        is_context_encoding = input_ids.shape[-1] > 1

        # --- Audio encoding (CPU frontend + Neuron transformer + CPU postprocessor) ---
        audio_embeddings = None
        audio_positions = None
        if (
            input_features is not None
            and self.audio_encoder is not None
            and is_context_encoding
        ):
            audio_token_id = getattr(
                self.config, "audio_token_id", None
            ) or getattr(self.config, "audio_token_index", 151646)

            with torch.no_grad():
                # Prepare audio features (same as HF get_audio_features)
                if feature_attention_mask is not None:
                    audio_feature_lengths = feature_attention_mask.sum(-1)
                    # (batch, mel_len, n_mels) -> mask -> (n_mels, valid_len)
                    input_features_flat = input_features.permute(0, 2, 1)[
                        feature_attention_mask.bool()
                    ].permute(1, 0)
                else:
                    input_features_flat = input_features.squeeze(0).permute(1, 0)
                    audio_feature_lengths = torch.tensor(
                        [input_features_flat.shape[1]], dtype=torch.long
                    )

                aftercnn_lens, audio_output_lens = (
                    self.audio_encoder._get_feat_extract_output_lengths(
                        audio_feature_lengths
                    )
                )
                audio_embeddings = self.audio_encoder(
                    input_features_flat,
                    feature_lens=audio_feature_lengths,
                    aftercnn_lens=aftercnn_lens,
                )

                # Find audio token positions for scattering
                audio_mask_bool = (input_ids == audio_token_id)
                if audio_mask_bool.any() and audio_embeddings is not None:
                    audio_positions = generate_positions_from_mask(
                        audio_mask_bool.squeeze()
                    )

        # --- Vision encoding (Neuron) ---
        vision_embeddings = None
        vision_positions = None
        if (
            (pixel_values is not None)
            and is_context_encoding
            and pixel_values.sum() != 0
        ):
            image_token_id = getattr(self.config, "image_token_id", None) or getattr(
                self.config, "image_token_index", 151655
            )
            vision_mask_bool = (input_ids == image_token_id)
            if vision_mask_bool.any():
                vision_positions = generate_positions_from_mask(
                    vision_mask_bool.squeeze()
                )
                vision_embeddings = self.vision_encoder_model(
                    pixel_values.to(self.vision_config.neuron_config.torch_dtype),
                    image_grid_thw,
                )

        # --- Combine multimodal embeddings for scattering ---
        # The text model's encode_vision_to_input scatters embeddings at
        # specified positions. We combine audio + vision into one tensor.
        if audio_embeddings is not None and vision_embeddings is not None:
            # Both audio and vision present
            # audio_embeddings is on CPU, vision_embeddings may be on XLA
            # The model wrapper handles device transfer, so keep on CPU
            all_embeddings = torch.cat([
                vision_embeddings.cpu() if vision_embeddings.is_cuda else vision_embeddings,
                audio_embeddings,
            ], dim=0)
            all_positions = torch.cat([vision_positions, audio_positions])
            vision_embeddings = all_embeddings
            vision_mask = pad_positions(all_positions, pad_limit, (pad_limit - 1))
        elif audio_embeddings is not None and audio_positions is not None:
            # Audio only, no vision
            vision_embeddings = audio_embeddings
            vision_mask = pad_positions(audio_positions, pad_limit, (pad_limit - 1))
        elif vision_embeddings is not None and vision_positions is not None:
            # Vision only, no audio
            vision_mask = pad_positions(vision_positions, pad_limit, (pad_limit - 1))
        else:
            # No multimodal input - use dummy embeddings
            vision_embeddings, vision_mask = self._get_text_model_wrapper().get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1),
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
        return Qwen25OmniMultimodalInferenceConfig
