"""Qwen3-Omni-30B-A3B-Instruct multimodal model for NxD Inference.

Combines:
  - Qwen3-VL vision encoder (reused directly)
  - Qwen3-Omni audio encoder (Conv2d + transformer on Neuron)
  - MoE text decoder (MRoPE attention + sparse MoE FFN)

All neural network components run on Neuron.
"""

import copy
import gc
import logging
import math
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
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
)
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import (
    NeuronQwen3VLForImageEncoding,
    NeuronQwen3VLVisionModel,
    NeuronQwen3VLVisionModelWrapper,
)
from neuronx_distributed_inference.modules.autobucketing import generate_buckets

from modeling_qwen3_omni_text import (
    NeuronQwen3OmniTextModel,
    NeuronQwen3OmniTextModelWrapper,
    convert_qwen3_omni_text_hf_to_neuron,
)
from modeling_qwen3_omni_audio import (
    AudioEncoderInferenceConfig,
    NeuronQwen3OmniAudioEncoder,
    NeuronQwen3OmniForAudioEncoding,
)

logger = logging.getLogger("Neuron")


class Qwen3OmniMoEInferenceConfig(ImageToTextInferenceConfig):
    """Inference config for Qwen3-Omni multimodal model.

    Handles the nested config structure:
      Qwen3OmniMoeConfig -> thinker_config -> text_config, vision_config, audio_config

    Combines ImageToTextInferenceConfig (vision + text) with MoE settings
    from Qwen3MoeInferenceConfig.
    """

    @staticmethod
    def _extract_thinker_config(obj):
        thinker = getattr(obj, "thinker_config", None)
        if thinker is None:
            return
        if hasattr(thinker, "__dict__") and not isinstance(thinker, dict):
            thinker = vars(thinker)
        if not isinstance(thinker, dict):
            return

        def _to_dict(x):
            if hasattr(x, "__dict__") and not isinstance(x, dict):
                return vars(x)
            return x

        if not hasattr(obj, "text_config") and "text_config" in thinker:
            obj.text_config = _to_dict(thinker["text_config"])
        if not hasattr(obj, "vision_config") and "vision_config" in thinker:
            obj.vision_config = _to_dict(thinker["vision_config"])
        if not hasattr(obj, "audio_config") and "audio_config" in thinker:
            obj.audio_config = _to_dict(thinker["audio_config"])
        for token_key in [
            "audio_token_id", "image_token_id", "video_token_id",
            "audio_start_token_id", "vision_start_token_id", "vision_end_token_id",
            "vision_token_id", "pad_token_id", "position_id_per_seconds",
        ]:
            if token_key in thinker and not hasattr(obj, token_key):
                setattr(obj, token_key, thinker[token_key])

    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        # Extract sub-configs from thinker_config if present
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
                if "audio_config" not in kwargs and "audio_config" in thinker:
                    ac = thinker["audio_config"]
                    kwargs["audio_config"] = (
                        vars(ac) if hasattr(ac, "__dict__") and not isinstance(ac, dict) else ac
                    )
                for token_key in [
                    "audio_token_id", "image_token_id", "video_token_id",
                    "audio_start_token_id", "vision_start_token_id", "vision_end_token_id",
                    "vision_token_id", "pad_token_id",
                ]:
                    if token_key in thinker and token_key not in kwargs:
                        kwargs[token_key] = thinker[token_key]

        # Wrap load_config to extract thinker sub-configs
        original_load_config = load_config
        if original_load_config is not None:
            extract = self._extract_thinker_config
            def _wrapped_load_config(self_inner):
                original_load_config(self_inner)
                extract(self_inner)
            load_config = _wrapped_load_config

        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )

        self._add_moe_config()
        self._add_special_config()
        self._validate_supported_configs()

    def _add_moe_config(self):
        """Apply MoE-specific settings to text_config (from Qwen3MoeInferenceConfig)."""
        tc = self.text_config

        # num_local_experts alias for initialize_moe_module
        if hasattr(tc, "num_experts") and not hasattr(tc, "num_local_experts"):
            tc.num_local_experts = tc.num_experts
        tc.n_shared_experts = 0

        # GLU MLP required for MoE
        tc.neuron_config.glu_mlp = True

        # Router config
        tc.neuron_config.router_config.dtype = torch.float32
        tc.neuron_config.router_config.act_fn = "softmax"
        tc.neuron_config.disable_numeric_cc_token = True
        if hasattr(tc, "norm_topk_prob") and tc.norm_topk_prob:
            tc.neuron_config.normalize_top_k_affinities = True

        # Set intermediate_size to moe_intermediate_size for MoE layers
        if hasattr(tc, "moe_intermediate_size"):
            tc.intermediate_size = tc.moe_intermediate_size

        # Intermediate size padding for MoE
        moe_tp_degree = tc.neuron_config.moe_tp_degree
        if hasattr(tc, "moe_intermediate_size"):
            I_TP = tc.moe_intermediate_size // moe_tp_degree
            if getattr(tc.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
                if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                    padded = (
                        math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                        * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                        * moe_tp_degree
                    )
                    tc.moe_intermediate_pad_size = max(padded - tc.moe_intermediate_size, 0)
                    tc.moe_intermediate_size = padded

            # MoE fused NKI kernel
            I_TP = tc.moe_intermediate_size // moe_tp_degree
            if (
                getattr(tc.neuron_config, "moe_fused_nki_kernel_enabled", False)
                and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0
            ):
                tc.moe_fused_nki_kernel_enabled = True

    def _add_special_config(self):
        """Copy vision/text attributes and apply validation."""
        # Copy deepstack_visual_indexes from vision to text config
        if hasattr(self.vision_config, "deepstack_visual_indexes"):
            self.text_config.deepstack_visual_indexes = copy.deepcopy(
                self.vision_config.deepstack_visual_indexes
            )

        # MRoPE section from text_config
        if hasattr(self.text_config, "rope_scaling"):
            rs = self.text_config.rope_scaling
            if isinstance(rs, dict) and "mrope_section" in rs:
                self.text_config.mrope_section = rs["mrope_section"]

        # Vision config derived attributes
        if hasattr(self.vision_config, "hidden_size") and hasattr(self.vision_config, "num_heads"):
            self.vision_config.head_dim = (
                self.vision_config.hidden_size // self.vision_config.num_heads
            )
        self.vision_config.num_cores_per_group = 1

        # Vision encoder uses fused QKV (HF stores qkv.weight, conversion maps to Wqkv)
        self.vision_config.neuron_config.fused_qkv = True

        # Copy token IDs to top-level and text_config
        for attr in [
            "image_token_id", "video_token_id", "audio_token_id",
            "vision_start_token_id", "vision_end_token_id",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self.text_config, attr, val)

        # Pad token
        if hasattr(self, "pad_token_id"):
            self.text_config.pad_token_id = self.pad_token_id

        # Qwen3 MoE text: no QKV bias, no output bias
        self.text_config.attention_bias = False
        self.text_config.qkv_bias = False
        self.text_config.o_bias = False

        # Store audio_config as SimpleNamespace
        if hasattr(self, "audio_config") and isinstance(self.audio_config, dict):
            self.audio_config = SimpleNamespace(**self.audio_config)

        # Vision bucketing
        if not self.vision_config.neuron_config.enable_bucketing:
            VISION_SEQ_LENGTH = self.vision_config.neuron_config.seq_len
            self.vision_config.neuron_config.enable_bucketing = True
            self.vision_config.neuron_config.buckets = generate_buckets(
                VISION_SEQ_LENGTH, VISION_SEQ_LENGTH
            )

        if self.text_config.neuron_config.seq_len > 10240:
            os.environ["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = f"{140 * 1024 * 1024}"

    def _validate_supported_configs(self):
        unsupported_text = [
            "is_block_kv_layout", "is_prefix_caching", "is_chunked_prefill",
            "is_medusa", "enable_fused_speculation",
        ]
        for cfg_name in unsupported_text:
            if getattr(self.text_config.neuron_config, cfg_name, False):
                setattr(self.text_config.neuron_config, cfg_name, False)
                logger.warning(f"Qwen3-Omni text model does not support '{cfg_name}'. Disabled.")

        if self.text_config.neuron_config.attention_dp_degree > 1:
            raise ValueError("Qwen3-Omni does not support attention data parallel")
        if self.text_config.neuron_config.cp_degree > 1:
            raise ValueError("Qwen3-Omni does not support context parallel")

        unsupported_vision = [
            "sequence_parallel_enabled", "flash_decoding_enabled",
            "mlp_kernel_enabled",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_enabled",
            "qkv_kernel_enabled", "attn_kernel_enabled",
        ]
        for cfg_name in unsupported_vision:
            if getattr(self.vision_config.neuron_config, cfg_name, False) is not False:
                setattr(self.vision_config.neuron_config, cfg_name, False)

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.vocab_size",
            "text_config.rms_norm_eps",
            "text_config.rope_theta",
            "text_config.moe_intermediate_size",
            "text_config.num_experts",
            "text_config.num_experts_per_tok",
            "vision_config.deepstack_visual_indexes",
            "vision_config.depth",
            "vision_config.hidden_size",
            "vision_config.num_heads",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig


class NeuronQwen3OmniForCausalLM(NeuronBaseForImageToText):
    """Qwen3-Omni multimodal model (vision + audio + MoE text) on Neuron.

    - Vision encoder: Qwen3-VL ViT (reused directly)
    - Audio encoder: Conv2d + Neuron transformer + proj1/GELU/proj2
    - Text decoder: MRoPE attention + MoE FFN with deepstack
    """

    text_model_cls = NeuronQwen3OmniTextModel
    vision_model_cls = NeuronQwen3VLVisionModel

    text_model_wrapper = NeuronQwen3OmniTextModelWrapper
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
        self.audio_encoder = None
        self._cached_neuron_state_dict = None

    def checkpoint_loader_fn(self, mmap: bool = False):
        """Convert the full state dict once, split into text-only and
        vision-only shards, and return a fresh shallow copy to the caller.

        The underlying ModelBuilder mutates and deletes keys from the returned
        dict during `preprocess_checkpoint` (it drops everything not in the
        target model's state_dict). Sharing one dict across text + vision
        builders would delete the vision keys on the first pass. Each call
        therefore returns its own dict, while tensor storage is shared
        between them.
        """
        if self._cached_neuron_state_dict is None:
            sd = super().checkpoint_loader_fn(mmap=mmap)
            # Partition keys by owning model. The same tensor memory is
            # shared between partitions; each builder will drop the keys
            # it does not own during preprocess_checkpoint.
            text_prefixes = ("layers.", "embed_tokens.", "lm_head.", "norm.", "rank_util.")
            vision_prefixes = ("blocks.", "patch_embed.", "merger.",
                               "deepstack_merger_list.", "rotary_pos_emb.", "pos_embed.")
            text_sd, vision_sd = {}, {}
            for k, v in sd.items():
                if any(k.startswith(p) for p in text_prefixes):
                    text_sd[k] = v
                elif any(k.startswith(p) for p in vision_prefixes):
                    vision_sd[k] = v
            self._cached_neuron_state_dict = {"text": text_sd, "vision": vision_sd}
            del sd

        # Return a fresh shallow copy so preprocess_checkpoint's .pop calls
        # don't mutate the cache. Tensors are shared (same underlying storage).
        cache = self._cached_neuron_state_dict
        return {**cache["text"], **cache["vision"]}

    def free_cached_state_dict(self):
        """Call after all builders have finished sharding to release CPU memory."""
        self._cached_neuron_state_dict = None

    # --- Vision encoder ---

    def get_vision_compiler_args(self) -> str:
        cc = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc}' -O1 "
            f"--internal-max-instruction-limit=15000000"
        )

    def get_compiler_args(self) -> str:
        cc = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc}' -O1 "
            f"--internal-max-instruction-limit=15000000"
        )

    def get_required_kwargs(self) -> List[str]:
        return [
            "pixel_values", "image_grid_thw",
            "input_features", "feature_attention_mask",
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
            return_ranked_to_cpu=False,
        )
        self.vision_models.append(self.vision_encoder_model)

    # --- Audio encoder ---

    def enable_audio_encoder(self, state_dict=None):
        audio_config = getattr(self.config, "audio_config", None)
        if audio_config is None:
            logger.warning("No audio_config found. Audio encoder not initialized.")
            return

        dtype = torch.bfloat16
        if hasattr(self.config, "neuron_config"):
            dtype = getattr(self.config.neuron_config, "torch_dtype", dtype)

        if state_dict is not None:
            self.audio_encoder = NeuronQwen3OmniAudioEncoder.from_pretrained_state_dict(
                audio_config, state_dict, dtype=dtype
            )
            # Stash transformer weights for compile / load (checkpoint_loader_fn).
            self._audio_transformer_state_dict = {
                k: v for k, v in state_dict.items() if k.startswith("transformer.")
            }
        else:
            self.audio_encoder = NeuronQwen3OmniAudioEncoder(audio_config, dtype=dtype)
            self._audio_transformer_state_dict = None

        self.audio_encoder.eval()
        logger.info("Audio encoder initialized (Neuron transformer pending compile/load)")

    def compile_audio_encoder(self, compiled_model_path, audio_neuron_config=None):
        if self.audio_encoder is None:
            raise RuntimeError("Call enable_audio_encoder() first")

        audio_config = getattr(self.config, "audio_config", None)
        if isinstance(audio_config, dict):
            audio_config = SimpleNamespace(**audio_config)

        if audio_neuron_config is None:
            tp_degree = self.neuron_config.tp_degree
            audio_neuron_config = NeuronConfig(
                tp_degree=tp_degree,
                torch_dtype=self.neuron_config.torch_dtype,
                batch_size=1,
                buckets=[256, 512, 1024, 1500, 2048, 3000],
            )

        audio_inf_config = AudioEncoderInferenceConfig(
            neuron_config=audio_neuron_config,
            audio_config=vars(audio_config) if hasattr(audio_config, "__dict__") else audio_config,
        )

        # Pass the already-loaded transformer weights so checkpoint_loader_fn can return them.
        transformer_sd = getattr(self, "_audio_transformer_state_dict", None)

        audio_app = NeuronQwen3OmniForAudioEncoding(
            model_path=self.model_path,
            config=audio_inf_config,
            transformer_state_dict=transformer_sd,
        )
        audio_app.compile(compiled_model_path)
        logger.info("Audio encoder transformer compiled to %s", compiled_model_path)
        return audio_app

    def load_audio_encoder(self, compiled_model_path, audio_app=None):
        if self.audio_encoder is None:
            raise RuntimeError("Call enable_audio_encoder() first")

        if audio_app is None:
            inf_config = AudioEncoderInferenceConfig.load(compiled_model_path)
            transformer_sd = getattr(self, "_audio_transformer_state_dict", None)
            audio_app = NeuronQwen3OmniForAudioEncoding(
                model_path=self.model_path,
                config=inf_config,
                transformer_state_dict=transformer_sd,
            )
            audio_app.is_compiled = True
            audio_app.traced_model = torch.jit.load(
                compiled_model_path + "/model.pt"
            )
            for mw in audio_app.models:
                mw.model = audio_app.traced_model
            audio_app.is_loaded_to_neuron = True
            audio_app.load_weights(compiled_model_path)

        self.audio_encoder.transformer = audio_app.model
        logger.info("Audio encoder transformer loaded from %s", compiled_model_path)

    # --- Image counting and splitting (from Qwen3-VL) ---

    def _count_images_per_batch_line(self, input_ids, attention_mask):
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

    # --- Rope index (from Qwen3-VL) ---

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute 3D MRoPE position IDs (copied from Qwen3-VL)."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = getattr(self.config, "video_token_id", None)
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, input_ids_i in enumerate(total_input_ids):
                input_ids_i = input_ids_i[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_i[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                if video_token_id is not None:
                    video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_i.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id is not None and video_token_id in input_tokens and remain_videos > 0:
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
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
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

    # --- Atomic prefill ---

    def forward_atomic_prefill(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values,
        image_grid_thw,
        audio_embeddings=None,
        audio_positions=None,
        input_capture_hook=None,
        tensor_capture_hook=None,
    ):
        pad_limit = self.get_padding_length(input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))

            vision_embeddings, deepstack_vision_embeds = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_grid_thw
            )
        else:
            vision_embeddings, vision_mask, deepstack_vision_embeds = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        # Merge audio embeddings into vision embeddings for scattering
        if audio_embeddings is not None and audio_positions is not None:
            if vision_embeddings is not None and vision_embeddings.numel() > 0:
                all_embeddings = torch.cat([
                    vision_embeddings.cpu() if hasattr(vision_embeddings, 'is_cuda') and vision_embeddings.is_cuda else vision_embeddings,
                    audio_embeddings,
                ], dim=0)
                all_positions = torch.cat([
                    generate_positions_from_mask(
                        (input_ids == self.config.image_token_id).squeeze()
                    ),
                    audio_positions,
                ])
                vision_embeddings = all_embeddings
                vision_mask = pad_positions(all_positions, pad_limit, (pad_limit - 1))
            else:
                vision_embeddings = audio_embeddings
                vision_mask = pad_positions(audio_positions, pad_limit, (pad_limit - 1))

        rotary_position_ids, rope_deltas = self.get_rope_index(
            input_ids, image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

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
            if hasattr(output, "tokens") and isinstance(output.tokens, torch.Tensor):
                concatenated_tokens.append(output.tokens)

        concatenated_logits = torch.cat(concatenated_logits, dim=0) if concatenated_logits else None
        concatenated_tokens = torch.cat(concatenated_tokens, dim=0) if concatenated_tokens else None

        concatenated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatenated_output.tokens = concatenated_tokens
        return concatenated_output

    # --- Main forward ---

    def get_padding_length(self, input_ids):
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
        is_context_encoding = input_ids.shape[-1] > 1
        # Ensure prompt_len < chosen bucket so pad_positions' fill_value
        # (pad_limit - 1) lands on a padding token rather than the last real
        # token. When prompt_len equals a bucket boundary (e.g. 256), the
        # scatter in encode_vision_to_input overwrites position pad_limit-1
        # with a zero audio embedding, corrupting the prompt.
        if is_context_encoding:
            pad_limit = self.get_padding_length(input_ids)
            if input_ids.shape[1] == pad_limit:
                pad_token_id = getattr(self.config, "pad_token_id", None) or 151645
                input_ids = torch.cat([
                    input_ids,
                    torch.full((input_ids.shape[0], 1), pad_token_id,
                               dtype=input_ids.dtype, device=input_ids.device),
                ], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros((attention_mask.shape[0], 1),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device),
                    ], dim=1)
                if position_ids is not None:
                    position_ids = torch.cat([
                        position_ids,
                        position_ids[:, -1:] + 1,
                    ], dim=1)
        pad_limit = self.get_padding_length(input_ids)

        # --- Audio encoding ---
        audio_embeddings = None
        audio_positions = None
        if (
            input_features is not None
            and self.audio_encoder is not None
            and is_context_encoding
        ):
            audio_token_id = getattr(self.config, "audio_token_id", 151646)

            with torch.no_grad():
                if feature_attention_mask is not None:
                    audio_feature_lengths = feature_attention_mask.sum(-1)
                    input_features_flat = input_features.permute(0, 2, 1)[
                        feature_attention_mask.bool()
                    ].permute(1, 0)
                else:
                    input_features_flat = input_features.squeeze(0).permute(1, 0)
                    audio_feature_lengths = torch.tensor(
                        [input_features_flat.shape[1]], dtype=torch.long
                    )

                audio_embeddings = self.audio_encoder(
                    input_features_flat,
                    feature_lens=audio_feature_lengths,
                )

                audio_mask_bool = (input_ids == audio_token_id)
                if audio_mask_bool.any() and audio_embeddings is not None:
                    audio_positions = generate_positions_from_mask(
                        audio_mask_bool.squeeze()
                    )

        # --- Vision + Text prefill with atomic batching ---
        has_vision = (
            pixel_values is not None
            and is_context_encoding
            and pixel_values.sum() != 0
        )

        if has_vision:
            batch_size = input_ids.shape[0]
            images_per_batch_line = self._count_images_per_batch_line(input_ids, attention_mask)
            vision_inputs_per_bl = self._split_vision_inputs_by_batch_line(
                pixel_values, image_grid_thw, images_per_batch_line
            )

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
                    audio_embeddings=audio_embeddings,
                    audio_positions=audio_positions,
                    input_capture_hook=input_capture_hook,
                    tensor_capture_hook=tensor_capture_hook,
                )
                outputs.append(output)
                rope_deltas_list.append(rope_deltas)

            self.rope_deltas = torch.cat(rope_deltas_list, dim=0)
            return self.concat_causal_lm_outputs(outputs)

        # --- Text-only or audio-only prefill, or decode ---
        vision_embeddings_combined = None
        vision_mask_combined = None

        if audio_embeddings is not None and audio_positions is not None:
            vision_embeddings_combined = audio_embeddings
            vision_mask_combined = pad_positions(audio_positions, pad_limit, (pad_limit - 1))

        if vision_embeddings_combined is None:
            vision_embeddings_combined, vision_mask_combined, deepstack_vision_embeds = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )
        else:
            _, _, deepstack_vision_embeds = (
                self.text_model_wrapper.get_dummy_vision_inputs(
                    config=self.text_config,
                    input_ids=input_ids,
                    n_active_tokens=pad_limit,
                    fill_value=(pad_limit - 1),
                )
            )

        # Compute rotary position IDs
        if is_context_encoding:
            rotary_position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            if self.rope_deltas is not None:
                delta = self.rope_deltas.to(input_ids.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            else:
                delta = 0
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
            vision_embeddings=vision_embeddings_combined,
            vision_mask=vision_mask_combined,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )
        return output_token

    # --- HF model loading and state dict conversion ---

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict,
        inference_config: Qwen3OmniMoEInferenceConfig,
    ) -> dict:
        """Convert Qwen3-Omni full state dict to NxDI format.

        HF keys: thinker.visual.*, thinker.audio_tower.*, thinker.model.*, thinker.lm_head.*

        Step 0: Remap thinker.visual.* -> visual.* so Qwen3-VL vision conversion works.
        Step 1: Vision encoder conversion (strips visual.* prefix, remaps attn keys)
        Step 2: Audio encoder conversion (strips thinker.audio_tower.*, splits into frontend/transformer/postprocessor)
        Step 3: MoE text model conversion (strips thinker.model.*, attention remap, expert stacking)
        """
        # Step 0: Remap thinker.visual.* -> visual.* and map Qwen3-Omni vision
        # merger/merger_list names to Qwen3-VL's merger/deepstack_merger_list schema:
        #   merger.ln_q.*      -> merger.norm.*
        #   merger.mlp.0.*     -> merger.linear_fc1.*
        #   merger.mlp.2.*     -> merger.linear_fc2.*
        #   merger_list.N.*    -> deepstack_merger_list.N.*  (with same ln_q/mlp remap)
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("thinker.visual."):
                new_key = "visual." + key[len("thinker.visual."):]
                if new_key.startswith("visual.merger_list."):
                    new_key = "visual.deepstack_merger_list." + new_key[len("visual.merger_list."):]
                new_key = new_key.replace(".ln_q.", ".norm.")
                new_key = new_key.replace(".mlp.0.", ".linear_fc1.")
                new_key = new_key.replace(".mlp.2.", ".linear_fc2.")
                remapped[new_key] = value
            else:
                remapped[key] = value
        state_dict = remapped

        # Step 1: Vision encoder conversion (Qwen3-VL: strips visual.*, remaps attn)
        state_dict = NeuronQwen3VLForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )

        # Step 2: Audio encoder conversion
        audio_dtype = getattr(
            inference_config.neuron_config, "torch_dtype", torch.bfloat16
        )
        state_dict = NeuronQwen3OmniAudioEncoder.convert_hf_to_neuron_state_dict(
            state_dict, dtype=audio_dtype
        )

        # Step 3: MoE text model conversion
        state_dict = convert_qwen3_omni_text_hf_to_neuron(
            state_dict, inference_config.text_config
        )

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Qwen3OmniMoEInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        return NeuronQwen3VLForImageEncoding.prepare_input_args(
            prompts, images, processor, role, config
        )
