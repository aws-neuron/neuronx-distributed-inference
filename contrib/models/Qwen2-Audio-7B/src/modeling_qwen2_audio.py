"""
Qwen2-Audio model for NeuronX Distributed Inference.

Contains:
  - Audio encoder (Whisper-like, 32 layers) using NeuronAttentionBase
  - Text LM (Qwen2 7B) using NeuronBaseModel
  - Top-level NeuronQwen2AudioForConditionalGeneration

Both encoder and LM run entirely on Neuron hardware.

NOTE ON NAMING: The NXDI framework uses 'vision_*' names internally
because it was built for image models. We cannot rename framework internals.
Our public API uses 'audio_*' names. Comments mark where framework names
leak through.
"""

import copy, logging, math, torch
from typing import List, Optional, Tuple, Type

from torch import nn
from torch.nn import LayerNorm
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper, VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import NeuronQwen2DecoderLayer, get_rmsnorm_cls
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase

from .configuration_qwen2_audio import Qwen2AudioMultimodalConfig

logger = logging.getLogger("Neuron")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class _AudioAttention(NeuronAttentionBase):
    """Whisper-style multi-head attention. No rotary embeddings."""
    def __init__(self, config):
        super().__init__(
            config=config, hidden_size=config.d_model,
            num_attention_heads=config.encoder_attention_heads,
            num_key_value_heads=config.encoder_attention_heads,
            head_dim=config.d_model // config.encoder_attention_heads,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sequence_parallel_enabled=False, rotary_emb=None,
            qkv_bias=True, o_bias=True,
        )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        return Q, K, cos_cache, sin_cache


class _AudioMLP(nn.Module):
    def __init__(self, d_model, ffn_dim, dtype):
        super().__init__()
        self.fc1 = ColumnParallelLinear(d_model, ffn_dim, gather_output=False, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = RowParallelLinear(ffn_dim, d_model, input_is_parallel=True, dtype=dtype, reduce_dtype=dtype)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _AudioEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype = config.neuron_config.torch_dtype
        self.self_attn_layer_norm = LayerNorm(config.d_model, eps=1e-5, dtype=dtype)
        self.self_attn = _AudioAttention(config)
        self.final_layer_norm = LayerNorm(config.d_model, eps=1e-5, dtype=dtype)
        self.mlp = _AudioMLP(config.d_model, config.encoder_ffn_dim, dtype)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states))[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.final_layer_norm(hidden_states))
        return residual + hidden_states


class NeuronQwen2AudioEncoderModel(nn.Module):
    """
    Audio encoder: conv → 32 transformer layers → avg_pool → norm → projector.
    Input:  (1, 128, 3000) mel spectrogram
    Output: (1, seq_len, 4096) projected audio embeddings (padded to text seq_len)
    """
    def __init__(self, config: InferenceConfig):
        super().__init__()
        ac = getattr(config, "audio_config", None) or config.vision_config
        dtype = ac.neuron_config.torch_dtype
        text_cfg = getattr(config, "text_config", None)
        self._text_seq_len = text_cfg.neuron_config.seq_len if text_cfg else None

        self.conv1 = nn.Conv1d(ac.num_mel_bins, ac.d_model, kernel_size=3, padding=1).to(dtype)
        self.conv2 = nn.Conv1d(ac.d_model, ac.d_model, kernel_size=3, stride=2, padding=1).to(dtype)
        self.embed_positions = nn.Embedding(ac.max_source_positions, ac.d_model).to(dtype)
        self.blocks = nn.ModuleList([_AudioEncoderLayer(ac) for _ in range(ac.encoder_layers)])
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.layer_norm = LayerNorm(ac.d_model, dtype=dtype)
        text_hidden = getattr(config, "hidden_size", None) or getattr(config, "text_config", config).hidden_size
        self.projector = nn.Linear(ac.d_model, text_hidden, bias=True).to(dtype)

    def forward(self, input_features):
        x = torch.nn.functional.gelu(self.conv1(input_features))
        x = torch.nn.functional.gelu(self.conv2(x))
        h = x.permute(0, 2, 1) + self.embed_positions.weight
        for block in self.blocks:
            h = block(h)
        h = self.avg_pooler(h.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.projector(self.layer_norm(h))
        if self._text_seq_len and out.shape[1] < self._text_seq_len:
            out = torch.nn.functional.pad(out, (0, 0, 0, self._text_seq_len - out.shape[1]))
        return out


class _AudioEncoderWrapper(ModelWrapper):
    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        vc = getattr(self.config, "audio_config", None) or self.config.vision_config
        return [(torch.ones(1, 128, 3000, dtype=vc.neuron_config.torch_dtype),)]

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def forward(self, input_features):
        if self.model is None:
            raise RuntimeError("Forward called before load")
        return self._forward(input_features)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT MODEL (Qwen2 LM with audio embedding merge)
# ═══════════════════════════════════════════════════════════════════════════════

class _Qwen2AudioTextModel(NeuronBaseModel):
    """Qwen2 LM that merges audio embeddings during context encoding."""

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        # Framework method — 'vision' args carry audio embeddings
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    def setup_attr_for_model(self, config):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id,
            dtype=config.neuron_config.torch_dtype, shard_across_embedding=True, pad=True)
        self.layers = nn.ModuleList([NeuronQwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, pad=True,
            gather_output=not self.on_device_sampling)


class _Qwen2AudioTextWrapper(ImageToTextModelWrapper):
    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        bs, seq = input_ids.shape[0], input_ids.shape[-1]
        if seq > 1:
            return (torch.zeros(bs, config.neuron_config.seq_len, config.hidden_size, dtype=config.neuron_config.torch_dtype),
                    torch.full((bs, n_active_tokens, 1), fill_value=fill_value, dtype=torch.int32))
        return torch.zeros((0), dtype=config.neuron_config.torch_dtype), torch.zeros((0), dtype=torch.bool)


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class NeuronQwen2AudioForConditionalGeneration(NeuronBaseForImageToText):
    """
    Full Qwen2-Audio on Neuron: audio encoder + language model.

    Both run entirely on Neuron hardware. Audio embeddings are merged into
    text embeddings inside the compiled graph via scatter_by_index_put.
    """

    text_model_cls = _Qwen2AudioTextModel
    vision_model_cls = NeuronQwen2AudioEncoderModel  # framework name → audio encoder
    text_model_wrapper = _Qwen2AudioTextWrapper
    vision_model_wrapper = _AudioEncoderWrapper       # framework name → audio encoder wrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls, self.vision_model_cls,
            self.text_model_wrapper, self.vision_model_wrapper,
            *args, **kwargs)

    @property
    def audio_encoder(self):
        return self.vision_encoder_model

    @property
    def audio_config(self):
        return self.vision_config

    def get_compiler_args(self):
        return ("--enable-saturate-infinity --enable-mixed-precision-accumulation "
                "--auto-cast=none --model-type transformer -O1 "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma' "
                "--internal-hlo2tensorizer-options='--verify-hlo=true'")

    def get_vision_compiler_args(self):
        # Framework method — compiles the audio encoder
        return ("--auto-cast=none --model-type=transformer "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2' -O1 "
                "--enable-saturate-infinity --enable-mixed-precision-accumulation "
                "--internal-hlo2tensorizer-options='--verify-hlo=true'")

    def get_required_kwargs(self):
        return ["audio_features"]

    def enable_vision_encoder(self, enable_wlt_optimization=True, **kw):
        # Framework method — initializes the audio encoder
        cfg = copy.deepcopy(self.config)
        self.vision_encoder_model = self.vision_model_wrapper(
            config=cfg, model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG, compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=kw,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True, return_ranked_to_cpu=False)
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2AudioForConditionalGeneration
        return Qwen2AudioForConditionalGeneration.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        text_cfg = getattr(config, "text_config", config)
        result = _convert_text_weights(state_dict, text_cfg)
        if hasattr(config, "vision_config"):
            result.update(_convert_encoder_weights(state_dict, config))
        return result

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                seq_ids=None, sampling_params=None, audio_features=None,
                input_capture_hook=None, tensor_capture_hook=None, **kwargs):
        from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
            generate_positions_from_mask, pad_positions)

        pad_limit = self._get_bucket_size(input_ids)

        if audio_features is not None and input_ids.shape[-1] > 1 and audio_features.sum() != 0:
            audio_token_id = getattr(self.config, "audio_token_index", 151646)
            mask = (input_ids == audio_token_id).unsqueeze(-1).to(torch.bool)
            audio_positions = pad_positions(generate_positions_from_mask(mask.squeeze()), pad_limit, pad_limit - 1)

            enc_result = self.audio_encoder(audio_features.to(self.audio_config.neuron_config.torch_dtype))
            if isinstance(enc_result, list) and isinstance(enc_result[0], list):
                audio_embeddings = enc_result[0][0].detach().cpu()
            elif isinstance(enc_result, list):
                audio_embeddings = enc_result[0].detach().cpu()
            else:
                audio_embeddings = enc_result.detach().cpu()
        else:
            audio_embeddings, audio_positions = self.text_model_wrapper.get_dummy_vision_inputs(
                config=self.text_config, input_ids=input_ids,
                n_active_tokens=pad_limit, fill_value=(pad_limit - 1))

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, seq_ids=seq_ids, sampling_params=sampling_params,
            input_capture_hook=input_capture_hook, tensor_capture_hook=tensor_capture_hook,
            vision_embeddings=audio_embeddings, vision_mask=audio_positions)

    def _get_model_outputs(self, input_ids, attention_mask, position_ids,
                           seq_ids, sampling_params, prev_hidden, adapter_ids,
                           vision_embeddings, vision_mask, deepstack_vision_embeds,
                           medusa_args, llava_args, **kwargs):
        rotary_position_ids = kwargs.get("rotary_position_ids") or torch.empty(0)
        args = (input_ids, attention_mask, position_ids, seq_ids, sampling_params,
                *(torch.empty(0) for _ in range(16)),
                rotary_position_ids, vision_embeddings, vision_mask)

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(*args)
            self.kv_cache_populated = True
            return outputs, self.context_encoding_model.is_neuron()
        outputs = self.token_generation_model(*args)
        return outputs, self.token_generation_model.is_neuron()

    def _get_bucket_size(self, input_ids):
        for val in self.context_encoding_model.config.neuron_config.buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    @classmethod
    def get_config_cls(cls):
        return Qwen2AudioMultimodalConfig


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_text_weights(state_dict, config):
    """HF language_model.* → Neuron text model keys."""
    nc = config.neuron_config
    sd = {}
    for key, value in state_dict.items():
        if key.startswith("language_model.model."):
            nk = key[len("language_model.model."):]
        elif key.startswith("language_model.lm_head."):
            nk = key[len("language_model."):]
        else:
            continue
        for p in ("q_proj", "k_proj", "v_proj"):
            if f".self_attn.{p}." in nk:
                nk = nk.replace(f".self_attn.{p}.", f".self_attn.qkv_proj.{p}.")
                break
        if ".self_attn.o_proj." in nk:
            nk = nk.replace(".self_attn.o_proj.", ".self_attn.o_proj.o_proj.")
        sd[nk] = value.clone()

    if nc.vocab_parallel:
        sd["embed_tokens.rank_util.rank"] = torch.arange(0, nc.local_ranks_size, dtype=torch.int32)
    for i in range(config.num_hidden_layers):
        sd[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, nc.tp_degree, dtype=torch.int32)
    sd["rank_util.rank"] = torch.arange(0, nc.tp_degree, dtype=torch.int32)
    return sd


def _convert_encoder_weights(state_dict, config):
    """HF audio_tower.* + multi_modal_projector.* → Neuron encoder keys (blocks.*)."""
    ac = getattr(config, "audio_config", None) or config.vision_config
    dtype = ac.neuron_config.torch_dtype
    sd = {}
    for key, value in state_dict.items():
        if key.startswith("audio_tower."):
            nk = key[len("audio_tower."):]
            if nk.startswith("layers."):
                nk = "blocks." + nk[len("layers."):]
        elif key.startswith("multi_modal_projector.linear."):
            nk = key.replace("multi_modal_projector.linear.", "projector.")
        else:
            continue
        nk = nk.replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        if ".fc1." in nk and "blocks." in nk:
            nk = nk.replace(".fc1.", ".mlp.fc1.")
        if ".fc2." in nk and "blocks." in nk:
            nk = nk.replace(".fc2.", ".mlp.fc2.")
        sd[nk] = value.clone().detach().contiguous().to(dtype)

    # k_proj has no bias in HF — add zero bias for NeuronAttentionBase
    for i in range(ac.encoder_layers):
        if f"blocks.{i}.self_attn.k_proj.bias" not in sd:
            sd[f"blocks.{i}.self_attn.k_proj.bias"] = torch.zeros(ac.d_model, dtype=dtype)
    return sd


__all__ = ["NeuronQwen2AudioForConditionalGeneration", "NeuronQwen2AudioEncoderModel"]
