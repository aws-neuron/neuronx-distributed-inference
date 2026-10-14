"""Qwen3-Omni Talker MoE transformer on Neuron.

Talker architecture (config.talker_config.text_config):
  - 20 layers, hidden=1024, heads=16, kv_heads=2 (GQA g=8), head_dim=128
  - 128 experts (moe_intermediate=384), top-6, norm_topk_prob=True
  - shared_expert (intermediate=768) gated by sigmoid(shared_expert_gate(x))
  - q_norm / k_norm (per-head-dim RMSNorm)
  - MRoPE theta=1e6, mrope_section=[24,20,20], interleaved

The HF talker pipeline:
  inputs_embeds  ──► self.model (20 MoE layers) ──► codec_head ──► codec token
                    (this file traces this block)

text_projection / hidden_projection / code_predictor / codec_head stay on CPU
and are orchestrated by host Python. This file wraps only the 20-layer MoE
body into Neuron, following the thinker pattern.

The NxDI stock MoE module (`initialize_moe_module`) ties shared_expert size to
`config.intermediate_size`; Qwen3-Omni talker has different sizes (MoE=384,
shared=768), so we wrap the MoE block and add a separate SharedExpertSwiGLU.
"""
import copy
import logging
import math
import os
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

# Reuse the thinker's Qwen3-VL attention (MRoPE, q/k-norm, GQA) — same exact
# attention architecture as the talker.
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
    NeuronQwen3VLAttention,
    get_rmsnorm_cls,
)

logger = logging.getLogger("Neuron")


# -----------------------------------------------------------------------------
# Shared expert (not provided by initialize_moe_module when intermediate sizes
# differ between routed and shared experts).
# -----------------------------------------------------------------------------

class SharedExpertSwiGLU(nn.Module):
    """Shared SwiGLU MLP with a sigmoid gate — matches HF Qwen3-Omni talker."""

    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden = config.hidden_size
        inter = config.shared_expert_intermediate_size
        dtype = config.neuron_config.torch_dtype
        self.gate_proj = ColumnParallelLinear(
            hidden, inter, bias=False, gather_output=False, dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden, inter, bias=False, gather_output=False, dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            inter, hidden, bias=False, input_is_parallel=True, dtype=dtype,
        )
        # Output is a single sigmoid gate per token; output_size=1 isn't
        # divisible by TP so we keep this one replicated (plain nn.Linear).
        self.gate = nn.Linear(hidden, 1, bias=False, dtype=dtype)

    def forward(self, x):
        y = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        g = torch.sigmoid(self.gate(x).to(y.dtype))
        return g * y


# -----------------------------------------------------------------------------
# Talker decoder layer: reuse thinker pattern, add shared_expert on top of
# routed MoE.
# -----------------------------------------------------------------------------

class NeuronTalkerDecoderLayer(nn.Module):
    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3VLAttention(config)

        rmsnorm_cls = get_rmsnorm_cls()
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)

        # Routed experts via NxDI (driven off config.intermediate_size=MOE_INTER)
        self.mlp = initialize_moe_module(config=config)
        # Separate shared expert (different intermediate size)
        self.shared_expert = SharedExpertSwiGLU(config)

        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states

        qkv_fused_rmsnorm = None
        if self.input_layernorm is not None:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rotary_position_ids=rotary_position_ids,
            rmsnorm=qkv_fused_rmsnorm,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        norm_out = self.post_attention_layernorm(hidden_states)
        is_speculative_decoding = (
            self.config.neuron_config.enable_fused_speculation
            and not self.config.neuron_config.is_prefill_stage
        )
        routed = self.mlp(norm_out, padding_mask, is_speculative_decoding=is_speculative_decoding)[0]
        shared = self.shared_expert(norm_out)
        hidden_states = residual + routed + shared

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# -----------------------------------------------------------------------------
# Talker transformer (NeuronBaseModel)
# -----------------------------------------------------------------------------

class NeuronTalkerModel(NeuronBaseModel):
    """Talker MoE transformer on Neuron.

    Input: inputs_embeds [B, S, H] — the host passes the already-computed
    sum of text/code embeddings (HF's prepare_inputs_for_generation output).
    We expose this via the `vision_embeddings` slot of ImageToTextModelWrapper
    so existing input/output plumbing works without re-tracing the framework.
    """

    def setup_attr_for_model(self, config):
        self.on_device_sampling = (
            config.neuron_config.on_device_sampling_config is not None
        )
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config):
        self.padding_idx = getattr(config, "pad_token_id", 0) or 0
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            # embed_tokens is used for lookup during token-generation autoregressive
            # stepping (when host passes input_ids instead of inputs_embeds).
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
            # codec_head on Neuron (avoids a big CPU matmul every step)
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size, self.hidden_size, self.padding_idx,
            )
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        self.layers = nn.ModuleList(
            [NeuronTalkerDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def get_model_output(
        self,
        input_ids=None,
        inputs_embeds=None,
        vision_embeddings=None,
        vision_mask=None,
        is_for_context_encoding: bool = False,
        adapter_ids=None,
        **kwargs,
    ):
        # We apply vision-embed injection ourselves (both prefill and decode)
        # and pass through to super() with vision_*=None so the upstream does
        # not try to re-inject (its shape expectations don't match ours).
        if (
            vision_embeddings is not None
            and vision_mask is not None
            and vision_embeddings.numel() > 0
        ):
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if vision_embeddings.dtype != self.config.neuron_config.torch_dtype:
                vision_embeddings = vision_embeddings.to(self.config.neuron_config.torch_dtype)
            inputs_embeds = self.encode_vision_to_input(
                inputs_embeds, vision_embeddings, vision_mask
            )
            vision_embeddings = None
            vision_mask = None

        return super().get_model_output(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
            is_for_context_encoding=is_for_context_encoding,
            adapter_ids=adapter_ids,
            **kwargs,
        )

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        """Replace inputs_embeds with the host-provided embedding wherever the
        mask says valid. Unlike thinker's scatter-by-index, we accept that
        host already built the full per-position embedding, and just write
        wherever the mask says to.

        Talker's vision_embeddings comes in full seq_len size (4096) while
        inputs_embeds matches the bucket size (say 64). We need to crop to
        the bucket's active region.
        """
        if vision_embeddings is None or vision_embeddings.numel() == 0:
            return inputs_embeds
        S_in = inputs_embeds.shape[1]
        # Accept either bucket-sized or seq_len-sized vision_embeddings;
        # always take the leading S_in positions.
        if vision_embeddings.shape[1] >= S_in:
            ve = vision_embeddings[:, :S_in, :]
        else:
            pad = torch.zeros(
                inputs_embeds.shape[0], S_in - vision_embeddings.shape[1],
                inputs_embeds.shape[2], dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            ve = torch.cat([vision_embeddings, pad], dim=1)
        if vision_mask is None or vision_mask.numel() == 0:
            return ve  # full replacement
        # vision_mask shape: [B, n_active_tokens, 1], same sizing convention
        vm = vision_mask
        if vm.shape[1] >= S_in:
            vm = vm[:, :S_in, :]
        mask_bool = vm.bool()
        return torch.where(mask_bool, ve, inputs_embeds)


# -----------------------------------------------------------------------------
# Model wrapper: provides dummy vision/deepstack tensors during tracing
# -----------------------------------------------------------------------------

class NeuronTalkerModelWrapper(ImageToTextModelWrapper):
    """Input generator that emits dummy vision_embeddings at token-generation
    time too (so the traced NEFF bakes in the ADD/REPLACE path).
    """

    _ROTARY_POSITION_IDS_INDEX = 21

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        B, S = input_ids.shape
        if S > 1:
            vision_embeddings = torch.zeros(
                B, config.neuron_config.seq_len, config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(B, n_active_tokens, 1), fill_value=fill_value, dtype=torch.int32,
            )
        else:
            vision_embeddings = torch.zeros(
                B, 1, config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            vision_mask = torch.full(
                size=(B, 1, 1), fill_value=fill_value, dtype=torch.int32,
            )
        # Talker has no deepstack — pass empty tensor
        deepstack_vision_embeds = torch.zeros(
            (0), dtype=config.neuron_config.torch_dtype
        )
        return vision_embeddings, vision_mask, deepstack_vision_embeds

    def input_generator(self):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )
            input_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )
            ve, vm, ds = self.get_dummy_vision_inputs(
                config=self.config, input_ids=input_ids,
                n_active_tokens=n_active_tokens, fill_value=0,
            )
            rotary_position_ids = torch.zeros(
                (3, self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32,
            )
            if self.tag in (CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG):
                inputs.append((
                    input_ids, attention_mask, position_ids, seq_ids, sampling_params,
                    torch.empty(0),  # prev_hidden
                    torch.empty(0),  # adapter_ids
                    torch.empty(0),  # accepted_indices
                    torch.empty(0),  # current_length
                    torch.empty(0),  # medusa_mask
                    torch.empty(0),  # scatter_index
                    torch.empty(0),  # slot_mapping
                    torch.empty(0),  # active_block_table
                    torch.empty(0),  # num_queries
                    torch.empty(0),  # computed_context_lens
                    torch.empty(0),  # tile_q_indices
                    torch.empty(0),  # tile_block_tables
                    torch.empty(0),  # tile_masks
                    torch.empty(0),  # inputs_embeds
                    torch.empty(0),  # kv_cache
                    torch.empty(0),  # active_mask
                    rotary_position_ids,  # 21
                    ve,                   # 22
                    vm,                   # 23
                    ds,                   # 24
                ))
            else:
                raise ValueError(f"Unsupported tag: {self.tag}")
        return inputs


# -----------------------------------------------------------------------------
# InferenceConfig
# -----------------------------------------------------------------------------

class TalkerInferenceConfig(InferenceConfig):
    """Config wrapping the talker.text_config + MoE-specific Neuron fields."""

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return MoENeuronConfig

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size", "num_attention_heads", "num_hidden_layers",
            "num_key_value_heads", "vocab_size", "rms_norm_eps", "rope_theta",
            "moe_intermediate_size", "num_experts", "num_experts_per_tok",
            "shared_expert_intermediate_size",
        ]

    def add_derived_config(self):
        self.num_cores_per_group = 1
        # Talker attention defaults
        self.attention_bias = False
        self.qkv_bias = False
        self.o_bias = False
        # Explicit head_dim (already 128 in HF config)
        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = 128
        # MRoPE section (from rope_scaling)
        rs = getattr(self, "rope_scaling", None) or {}
        self.mrope_section = rs.get("mrope_section", [24, 20, 20])

        # MoE adapters for initialize_moe_module: intermediate_size must be
        # the MoE expert intermediate (NOT shared expert).
        self.intermediate_size = self.moe_intermediate_size
        # num_local_experts alias
        if not hasattr(self, "num_local_experts"):
            self.num_local_experts = self.num_experts
        # No shared experts via initialize_moe_module — we handle those
        # ourselves in SharedExpertSwiGLU because of different intermediate
        # size from the routed experts.
        self.n_shared_experts = 0
        # GLU MLP required for experts
        self.neuron_config.glu_mlp = True
        # Router config
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"
        self.neuron_config.disable_numeric_cc_token = True
        if getattr(self, "norm_topk_prob", True):
            self.neuron_config.normalize_top_k_affinities = True


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------

class NeuronTalkerForCausalLM(NeuronBaseForCausalLM):
    """Autoregressive talker on Neuron.

    Usage is similar to thinker: compile() then load(); then use adapter.generate
    from host Python, with the host filling in `inputs_embeds` (via
    vision_embeddings) for each prefill/decode call.
    """

    _model_cls = NeuronTalkerModel

    @classmethod
    def get_config_cls(cls):
        return TalkerInferenceConfig

    def get_model_wrapper_cls(self):
        return NeuronTalkerModelWrapper

    def get_required_kwargs(self) -> List[str]:
        return ["vision_embeddings", "vision_mask"]

    def get_compiler_args(self) -> str:
        cc = self.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc}' -O1 "
            f"--internal-max-instruction-limit=15000000"
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # Talker uses a separate codec_head (maps hidden -> codec vocab),
        # never tied to embed_tokens.
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        # We never instantiate the full HF model to get state — we read
        # safetensors directly via get_state_dict override.
        raise NotImplementedError("Use get_state_dict / checkpoint_loader_fn")

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config) -> dict:
        """Read only talker.* tensors from the HF safetensors shards and
        run our conversion. Avoids loading the full 30B model.
        """
        import json as _json
        from safetensors.torch import safe_open

        index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
        talker_raw = {}
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = _json.load(f)["weight_map"]
            wanted_shards = {
                weight_map[k] for k in weight_map
                if k.startswith("talker.") and not k.startswith("talker.code_predictor.")
            }
            for shard in sorted(wanted_shards):
                with safe_open(os.path.join(model_name_or_path, shard), framework="pt") as sf:
                    for k in sf.keys():
                        if k.startswith("talker.") and not k.startswith("talker.code_predictor."):
                            talker_raw[k[len("talker."):]] = sf.get_tensor(k)
        else:
            for fname in sorted(os.listdir(model_name_or_path)):
                if fname.endswith(".safetensors"):
                    with safe_open(os.path.join(model_name_or_path, fname), framework="pt") as sf:
                        for k in sf.keys():
                            if k.startswith("talker.") and not k.startswith("talker.code_predictor."):
                                talker_raw[k[len("talker."):]] = sf.get_tensor(k)

        return convert_talker_hf_to_neuron(talker_raw, config)


# -----------------------------------------------------------------------------
# Weight conversion: HF 'talker.*' → Neuron keys
# -----------------------------------------------------------------------------

def convert_talker_hf_to_neuron(hf_sd: Dict[str, torch.Tensor], config: TalkerInferenceConfig) -> Dict:
    """Convert HF talker state dict to the key layout our module expects.

    HF keys (present in root state_dict):
      model.embed_codec.weight  (actually: model.codec_embedding.weight? verify)
      model.layers.{l}.input_layernorm.weight
      model.layers.{l}.post_attention_layernorm.weight
      model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{l}.self_attn.{q,k}_norm.weight
      model.layers.{l}.mlp.gate.weight
      model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight
      model.layers.{l}.mlp.shared_expert.{gate,up,down}_proj.weight
      model.layers.{l}.mlp.shared_expert_gate.weight
      model.norm.weight
      codec_head.weight
      (code_predictor and projections also present but dropped here — stay on CPU)
    """
    import gc
    num_experts = config.num_experts
    moe_inter = config.moe_intermediate_size
    hidden = config.hidden_size
    tp = config.neuron_config.tp_degree

    out: Dict[str, torch.Tensor] = {}

    # Embedding table (codec token embedding used for talker token generation
    # is model.codec_embedding). But HF's talker uses `get_input_embeddings()`
    # which returns model.codec_embedding (the talker's own codec vocab).
    # Keep the key neutral: "embed_tokens.weight".
    if "model.codec_embedding.weight" in hf_sd:
        out["embed_tokens.weight"] = hf_sd["model.codec_embedding.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
    elif "model.embed_tokens.weight" in hf_sd:
        out["embed_tokens.weight"] = hf_sd["model.embed_tokens.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()

    # codec_head → lm_head
    if "codec_head.weight" in hf_sd:
        out["lm_head.weight"] = hf_sd["codec_head.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()

    # Final norm
    if "model.norm.weight" in hf_sd:
        out["norm.weight"] = hf_sd["model.norm.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()

    for l in range(config.num_hidden_layers):
        base = f"model.layers.{l}"
        tgt = f"layers.{l}"

        # Norms
        out[f"{tgt}.input_layernorm.weight"] = hf_sd[f"{base}.input_layernorm.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.post_attention_layernorm.weight"] = hf_sd[f"{base}.post_attention_layernorm.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()

        # Attention: NeuronQwen3VLAttention expects qkv_proj.{q,k,v}_proj + o_proj.o_proj
        # format — match the thinker convert logic.
        out[f"{tgt}.self_attn.qkv_proj.q_proj.weight"] = hf_sd[f"{base}.self_attn.q_proj.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.self_attn.qkv_proj.k_proj.weight"] = hf_sd[f"{base}.self_attn.k_proj.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.self_attn.qkv_proj.v_proj.weight"] = hf_sd[f"{base}.self_attn.v_proj.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.self_attn.o_proj.o_proj.weight"] = hf_sd[f"{base}.self_attn.o_proj.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        # q_norm / k_norm map to q_layernorm / k_layernorm in thinker naming
        out[f"{tgt}.self_attn.q_layernorm.weight"] = hf_sd[f"{base}.self_attn.q_norm.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.self_attn.k_layernorm.weight"] = hf_sd[f"{base}.self_attn.k_norm.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()
        out[f"{tgt}.self_attn.rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)

        # Routed MoE: gate → router.linear_router; stack experts into gate_up_proj/down_proj
        out[f"{tgt}.mlp.router.linear_router.weight"] = hf_sd[f"{base}.mlp.gate.weight"].to(
            config.neuron_config.torch_dtype
        ).contiguous()

        dtype = config.neuron_config.torch_dtype
        gate_up = torch.empty(num_experts, hidden, 2 * moe_inter, dtype=dtype)
        down = torch.empty(num_experts, moe_inter, hidden, dtype=dtype)
        for e in range(num_experts):
            gw = hf_sd[f"{base}.mlp.experts.{e}.gate_proj.weight"]
            uw = hf_sd[f"{base}.mlp.experts.{e}.up_proj.weight"]
            dw = hf_sd[f"{base}.mlp.experts.{e}.down_proj.weight"]
            gate_up[e, :, :moe_inter].copy_(gw.T.to(dtype))
            gate_up[e, :, moe_inter:].copy_(uw.T.to(dtype))
            down[e].copy_(dw.T.to(dtype))
        out[f"{tgt}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up
        out[f"{tgt}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down

        # Shared expert (SharedExpertSwiGLU: gate_proj/up_proj/down_proj/gate)
        out[f"{tgt}.shared_expert.gate_proj.weight"] = hf_sd[f"{base}.mlp.shared_expert.gate_proj.weight"].to(dtype).contiguous()
        out[f"{tgt}.shared_expert.up_proj.weight"] = hf_sd[f"{base}.mlp.shared_expert.up_proj.weight"].to(dtype).contiguous()
        out[f"{tgt}.shared_expert.down_proj.weight"] = hf_sd[f"{base}.mlp.shared_expert.down_proj.weight"].to(dtype).contiguous()
        out[f"{tgt}.shared_expert.gate.weight"] = hf_sd[f"{base}.mlp.shared_expert_gate.weight"].to(dtype).contiguous()

        gc.collect()

    out["rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)
    return out
