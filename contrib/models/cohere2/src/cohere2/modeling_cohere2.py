# coding=utf-8
# Copyright 2024 Cohere Inc. HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Cohere2 model for NXD inference."""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)06d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

from typing import Any, Dict, Optional, List, Tuple, Type
import warnings

from neuronxcc.nki.language import nc
from neuronxcc.nki._private_kernels.mlp import (
    mlp_isa_kernel,
    quant_mlp_isa_kernel,
)
import neuronx_distributed
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region, 
    reduce_scatter_to_sequence_parallel_region, 
    reduce_from_tensor_model_parallel_region
)
from neuronx_distributed.parallel_layers import utils 
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.modules.attention.attention_base import FlashAttentionStrategy
from neuronx_distributed_inference.modules.attention.gqa import GroupQueryAttention_O
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import (
    move_heads_front,
    preprocess_quantized_linear_layer,
    repeat_kv,
    transpose_parallel_linear_layer,
    )
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.generation.sampling import Sampler
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group
import torch
from torch import nn, ones, float32, rsqrt, FloatTensor
from torch.distributed import ProcessGroup
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import Cohere2ForCausalLM
from transformers.activations import ACT2FN

from cohere2.hybrid_kv_cache_manager import HybridKVCacheManager
from cohere2.utils.rope import Cohere2RotaryEmbedding, apply_rotary_position_embedding
from cohere2.utils.qkv import GroupQueryAttentionQKVWithoutRMSKernel, convert_state_dict_to_fused_qkv
from cohere2.nki import (
    flash_fwd, FlashConfig, DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE, MIN_SLIDING_WINDOW_SEQ_TILE_SIZE
)


class Cohere2NeuronConfig(NeuronConfig):
    pass


class Cohere2InferenceConfig(InferenceConfig):

    def get_required_attributes(self) -> List[str]:
        return [
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "attention_bias",
            "sliding_window",
            "sliding_window_pattern",
            "rope_theta",
            "intermediate_size",
            "hidden_act",
            "logit_scale"
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Cohere2NeuronConfig

    def add_derived_config(self):
        # From LlamaInferenceConfig
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                tp_degree=self.neuron_config.tp_degree
            )


class NeuronCohere2LayerNorm(nn.Module):
    """
    https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/cohere2/modeling_cohere2.py#L107
    """
    def __init__(self, hidden_size=None, eps=1e-5, bias=False):
        """The hidden size can be a tuple or an int. """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: FloatTensor) -> FloatTensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(float32) * hidden_states
        return hidden_states.to(input_dtype)


class NeuronCohere2Attention(NeuronAttentionBase):
    def __init__(self, 
                 config: Cohere2InferenceConfig, 
                 block_idx: int, 
                 tensor_model_parallel_group: Optional[ProcessGroup] = None,
                 ) -> None:
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            rotary_emb=None,
            rms_norm_eps=None,
            use_qk_norm=False,
            clip_qkv=None,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
            sequence_parallel_enabled=False,
            attention_chunk_size=None,
            tensor_model_parallel_group=tensor_model_parallel_group,
            )

        # Neuron config
        self.neuron_config = config.neuron_config
        self.torch_dtype = self.neuron_config.torch_dtype
        self.padding_side = self.neuron_config.padding_side

        # Attention layer config
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.o_bias = self.qkv_bias = self.config.attention_bias
        self.sliding_window_pattern = self.config.sliding_window_pattern
        self.clip_qkv = None

        # Optimization: Fused QKV
        self.fused_qkv = self.neuron_config.fused_qkv

        if not parallel_state.model_parallel_is_initialized():
            warnings.warn(
                "No initialized distributed environment was found. "
                "Falling back to a local setup. "
                "Distributed optimizations (TP, PP, EP, SP) will not be available.",
                UserWarning
            )

            self.tp_degree = 1
        else:
            # Optimization: Tensor & Sequence parallelism
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()

        # Optimization: Sequence parallelism
        # As collective communications are handled at the decoder block level, sequence parallelism is disabled at the 
        # attention layer level to ensure it is disabled in QGA parallel layers so that they don't call these collective 
        # operation redundantly. In other words, inputs to QKV layers are always already all-gathered and of shape [B,S,H].
        # Disabling SP at the attention layer level also ensures that q_len is not multiplied by TP in NeuronAttentionBase.forward
        self.sequence_parallel_enabled = False
        self.sequence_dimension = None

        # Initialize the QKVO distributed linear layers
        self.init_gqa_properties()

        # To avoid duplicate all-gather (TP+SP) or all-reduce (TP) calls due to the parallel layout of the decoder block, 
        # these operations are performed once for the MLP & attention layer at the decoder block level. By setting reduce_output 
        # to False for the RowParallelLinear layer of the GQA attention layer, we ensure these collectives are not needlessly
        # called by the GQA O layer. 
        self.o_proj.o_proj.reduce_output = False

        # Specific to Cohere2
        self.sliding_window_enabled = (block_idx + 1) % self.sliding_window_pattern != 0
        self.sliding_window_size = self.config.sliding_window if self.sliding_window_enabled else None
        self.flash_decoding_enabled = False if self.sliding_window_enabled else self.neuron_config.flash_decoding_enabled

        # Initialize RoPE
        self.rotary_emb = None
        if self.sliding_window_enabled:
            self.rotary_emb = Cohere2RotaryEmbedding(
                head_dim=self.head_dim,
                rope_theta=self.config.rope_theta,
            ) 

    def init_gqa_properties(self) -> None:
        self.qkv_proj = GroupQueryAttentionQKVWithoutRMSKernel(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            qkv_kernel_nbsd_layout=self.neuron_config.qkv_kernel_nbsd_layout,
            on_cpu=self.neuron_config.on_cpu,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
            out_proj_kernel_enabled=False, # Not supported at the moment
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_layernorm = nn.LayerNorm(self.head_dim) if self.qk_layernorm else None
        self.k_layernorm = nn.LayerNorm(self.head_dim) if self.qk_layernorm else None
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_nc_config = self.neuron_config.logical_nc_config

    def prep_qkv_tensors(
            self,
            position_ids: torch.LongTensor,
            hidden_states: torch.FloatTensor,
            past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor],
            adapter_ids: Optional[torch.FloatTensor] = None,
            cos_cache: Optional[torch.FloatTensor] = None,
            sin_cache: Optional[torch.FloatTensor] = None,
            rmsnorm: Optional[torch.FloatTensor] = None,
            skip_rope: Optional[torch.BoolTensor] = False,
            residual: Optional[torch.FloatTensor] = None,
            use_polar_compatible_rope: Optional[torch.BoolTensor] = False,
    ) -> Tuple[torch.FloatTensor]:
        """We override this function to ensure Cohere2's apply_rotary_position_embedding implementation is called
        """
        # Vs. NeuronAttentionBase implementation: If SP is enabled, hidden_states have already been all-gathered at the 
        # decoder block level, q_len therefore already equals total sequence length and therefore don't need to multiply
        # it by the TP degree
        bsz, q_len, _ = hidden_states.size()
        
        Q, K, V, residual = self.qkv_proj(
            hidden_states=hidden_states, 
            rmsnorm=rmsnorm, 
            adapter_ids=adapter_ids, 
            residual=residual
            )

        # Change layout: BSHD -> BHSD
        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if not skip_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_position_embedding(q=Q, k=K, cos=cos_cache, sin=sin_cache)

        return Q, K, V, cos_cache, sin_cache, residual

    def _perform_prefill(self,
                        Q: torch.FloatTensor, 
                        K: torch.FloatTensor, 
                        V: torch.FloatTensor, 
                        q_len: int, 
                        bsz: int, 
                        attention_mask: torch.LongTensor,
                        ) -> Tuple[torch.FloatTensor, FlashAttentionStrategy]:
        flash_attn_strategy = self.get_flash_attention_strategy(q_len, attention_mask is not None)
        if flash_attn_strategy in (FlashAttentionStrategy.UNSHARDED_KERNEL, FlashAttentionStrategy.SHARDED_KERNEL) \
            and self.sliding_window_enabled and q_len > self.sliding_window_size:
                K_active = repeat_kv(K, self.num_key_value_groups)
                V_active = repeat_kv(V, self.num_key_value_groups)
                batch_size, n_head, seq_len, _ = Q.shape
                Q, K_active = Q.permute(0, 1, 3, 2), K_active.permute(0, 1, 3, 2)  # BHSD -> BHDS
                config = FlashConfig() if seq_len >= DEFAULT_SLIDING_WINDOW_SEQ_TILE_SIZE else FlashConfig(seq_tile_size=MIN_SLIDING_WINDOW_SEQ_TILE_SIZE)
                attn_output = flash_fwd[batch_size, n_head](Q, K_active, V_active, window_size=(self.sliding_window_size - 1, -1), config=config)
                return attn_output, flash_attn_strategy
        else:
            return super().perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        

    def _attention_context_encode(self,
                                 Q: torch.FloatTensor, 
                                 K: torch.FloatTensor, 
                                 V: torch.FloatTensor, 
                                 q_len: int, 
                                 bsz: int, 
                                 attention_mask: torch.LongTensor,
                                 past_key_value: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
                                 active_mask: Optional[torch.LongTensor] = None,
                                 ) -> Tuple[torch.FloatTensor]:
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        else:
            attn_output, flash_attn_strategy = self.perform_prefix_prefill(Q, K, V, q_len, bsz, attention_mask, past_key_value, active_mask)
        if self.flash_decoding_enabled:
            K, V = self._filter_kv_for_flash_decoding(K, V, q_len, Q)

        swa_kernel_enabled = flash_attn_strategy in (FlashAttentionStrategy.UNSHARDED_KERNEL, FlashAttentionStrategy.SHARDED_KERNEL) \
            and self.sliding_window_enabled and q_len > self.sliding_window_size
        if flash_attn_strategy == FlashAttentionStrategy.NONE or swa_kernel_enabled:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        return attn_output, K, V

        
class NeuronCohere2MLP(torch.nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.logical_nc_config = self.neuron_config.logical_nc_config

        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # Optimization: Sequence parallelism
        # As collective communications are handled at the decoder block level, sequence parallelism is disabled at the 
        # MLP layer level to ensure it is disabled in its parallel layers so that they don't call these collective 
        # operation redundantly. In other words, inputs to MLP layers are always already all-gathered and of shape [B,S,H].
        self.sequence_parallel_enabled = False
        self.sequence_dimension = None

        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        
        # Quantization 
        self.activation_quantization_type = self.neuron_config.activation_quantization_type
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound

        if (self.mlp_kernel_enabled or self.quantized_mlp_kernel_enabled) and self.logical_nc_config == 1:
            # On Trn1/Inf2, we can call the unsharded MLP kernel but it requires that intermediate_size/TP <= 4096
            assert self.intermediate_size // self.tp_degree <= 4096

        if parallel_state.model_parallel_is_initialized():
            tp_degree = self.neuron_config.tp_degree
            if self.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                self.intermediate_size += (
                    utils.get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                reduce_output=False, # Avoid redundant reduce operations (TP: all-reduce, TP+SP: reduce-scatter) since already performed at the decoder block level
                tensor_model_parallel_group=get_tp_group(config),
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )

            if self.mlp_kernel_enabled:
                if self.quantized_mlp_kernel_enabled:
                    setattr(self.gate_proj, "post_create_quantized_module_hook", preprocess_quantized_linear_layer)
                    setattr(self.up_proj, "post_create_quantized_module_hook", preprocess_quantized_linear_layer)
                    setattr(self.down_proj, "post_create_quantized_module_hook", preprocess_quantized_linear_layer)
                else:
                    # Transpose the weights to the layout expected by kernels
                    self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _native_mlp(self, x: torch.FloatTensor, adapter_ids=None) -> torch.FloatTensor:
        gate_proj_output = (
            self.gate_proj(x)
            if not is_lora_module(self.gate_proj)
            else self.gate_proj(x, adapter_ids)
        )
        up_proj_output = (
            self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        )
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        return output

    def _kernel_enabled_mlp(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mlp_fwd_nki_kernel = nki_jit()(mlp_isa_kernel)

        # Init output tensor
        output = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        # Since we don't use the fused RMSNorm, RMSNorm weigths are set to zero
        norm_weights, norm_eps = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device), 1e-05

        if self.logical_nc_config == 2:
            # Call to the sharded kernel -> Only works on Trn2
            spmd_grid = (nc(self.logical_nc_config),)
            mlp_fwd_nki_kernel[spmd_grid](
                x,
                norm_weights,
                self.gate_proj.weight.data,
                self.up_proj.weight.data,
                self.down_proj.weight.data,
                output,
                fused_rmsnorm=None,
                eps=norm_eps,
                kernel_name="MLP",
            )
        else: 
            # Call to the unsharded kernel
            mlp_fwd_nki_kernel(
                x,
                norm_weights,
                self.gate_proj.weight.data,
                self.up_proj.weight.data,
                self.down_proj.weight.data,
                output,
                fused_rmsnorm=None,
                eps=norm_eps,
                kernel_name="MLP",
            )
        return output
    
    def _kernel_enabled_quantized_mlp(self, x: torch.FloatTensor) -> torch.FloatTensor:
        spmd_grid = (nc(self.logical_nc_config),)
        mlp_fwd_nki_kernel = nki_jit()(quant_mlp_isa_kernel)

        # Init output tensor
        output = torch.zeros(x.shape, dtype=x.dtype, device=x.device)

        # Since we don't use the fused RMSNorm, RMSNorm weigths are set to zero
        norm_weights, norm_eps = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device), 1e-05

        if self.logical_nc_config == 2:
            # Call to the sharded kernel -> Only works on Trn2
            spmd_grid = (nc(self.logical_nc_config),)
            mlp_fwd_nki_kernel[spmd_grid](
                x,
                norm_weights,
                self.gate_proj.weight.data,
                self.gate_proj.scale,
                self.up_proj.weight.data,
                self.up_proj.scale,
                self.down_proj.weight.data,
                self.down_proj.scale,
                self.quantize_clamp_bound,
                output,
                fused_rmsnorm=None,
                eps=norm_eps,
                kernel_name="MLP",
            )
        else:
            # Call to the unsharded kernel
            mlp_fwd_nki_kernel(
                x,
                norm_weights,
                self.gate_proj.weight.data,
                self.gate_proj.scale,
                self.up_proj.weight.data,
                self.up_proj.scale,
                self.down_proj.weight.data,
                self.down_proj.scale,
                self.quantize_clamp_bound,
                output,
                fused_rmsnorm=None,
                eps=norm_eps,
                kernel_name="MLP",
            )
        return output

    def forward(self, x: torch.FloatTensor, adapter_ids=None) -> torch.FloatTensor:
        if self.mlp_kernel_enabled:
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(x=x)
            return self._kernel_enabled_mlp(x=x)
        else:
            return self._native_mlp(x=x, adapter_ids=adapter_ids)


class NeuronCohere2DecoderLayer(nn.Module):
    def __init__(self, 
                 config: InferenceConfig, 
                 layer_idx: int,
                 tensor_model_parallel_group: Optional[ProcessGroup] = None,
                 ):
        super().__init__()

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
        elif neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = (
                neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_group()
            )
        else:
            self.tensor_model_parallel_group = None

        self.self_attn = NeuronCohere2Attention(
            config=config, 
            block_idx=layer_idx,
            tensor_model_parallel_group=self.tensor_model_parallel_group
        )
        self.mlp = NeuronCohere2MLP(config)
        self.input_layernorm = NeuronCohere2LayerNorm(
            hidden_size=config.hidden_size, 
            eps=config.layer_norm_eps
        )

        # TODO: EAGLE speculative decoding
        # if (
        #     not config.neuron_config.is_eagle_draft
        #     or config.neuron_config.enable_eagle_draft_input_norm
        # ):
            # self.input_layernorm = get_rmsnorm_cls()(
            #     config.hidden_size,
            #     eps=config.rms_norm_eps,
            # )

        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.reduce_dtype = config.neuron_config.rpl_reduce_dtype

         # Specific to Cohere2
        self.sliding_window_enabled = (layer_idx + 1) % config.sliding_window_pattern != 0
        self.sliding_window_size = config.sliding_window
        self.n_positions = config.neuron_config.n_positions
        self.padding_side = config.neuron_config.padding_side

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Tuple[torch.BoolTensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # If SP enabled, SP region and hidden_states of shape [B, S/TP, H], 
        # else, non-parallel region and hidden_states of shape [B, S, H]
        residual = hidden_states

        # Kernel use may involve norm fusion -> add if clause + check whether compatible with SP
        hidden_states = self.input_layernorm(hidden_states)

        if self.tensor_model_parallel_group is not None and self.sequence_parallel_enabled:
            # Transition from SP region to TP region (all-gather)
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        if self.sliding_window_enabled:
            _, attention_mask = attention_mask
        else:
            attention_mask, _ = attention_mask

        # TP region - hidden_states of shape [B, S, H]
        hidden_states_attention, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            adapter_ids=adapter_ids,
            **kwargs
        )

        hidden_states_mlp = self.mlp(
            hidden_states,
            adapter_ids=adapter_ids,
        )

        hidden_states_output = hidden_states_attention + hidden_states_mlp

        original_dtype = hidden_states_output.dtype
        if self.tensor_model_parallel_group is not None:
            hidden_states_output = hidden_states_output.to(self.reduce_dtype)
            if self.sequence_parallel_enabled:
                # Transition from TP region to SP region (reduce-scatter)
                hidden_states_output = reduce_scatter_to_sequence_parallel_region(
                    hidden_states_output, 
                    self.sequence_dimension, 
                    process_group=self.tensor_model_parallel_group,
                )
            else:
                # Transition from TP region to non-parallel region (all-reduce)
                hidden_states_output = reduce_from_tensor_model_parallel_region(
                    hidden_states_output, 
                    process_group=self.tensor_model_parallel_group,
                )

        # If SP enabled, SP region and hidden_states of shape [B, S/TP, H], 
        # else, non-parallel region and hidden_states of shape [B, S, H]
        hidden_states_output = hidden_states_output.to(original_dtype)

        hidden_states = residual + hidden_states_output

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronCohere2LMHead(torch.nn.Module):

    def __init__(self, config: InferenceConfig, on_device_sampling_enabled: bool) -> None:
        super().__init__()
        self.logit_scale = config.logit_scale
        if parallel_state.model_parallel_is_initialized():
            self._lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not on_device_sampling_enabled,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self._lm_head = torch.nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        logits = self._lm_head(hidden_states)
        return logits * self.logit_scale
    

class NeuronCohere2Model(NeuronBaseModel):

    def init_inference_optimization(self, config: InferenceConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config.neuron_config)
        self.kv_mgr = HybridKVCacheManager(config=config, num_kv_head=self.num_key_value_heads)

    def setup_attr_for_model(self, config: InferenceConfig) -> None:
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.max_length = config.neuron_config.max_length
        self.sliding_window_pattern = config.sliding_window_pattern
        self.sliding_window_size = config.sliding_window

    def init_model(self, config: InferenceConfig) -> None:
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
        
        self.lm_head = NeuronCohere2LMHead(
            config=config, 
            on_device_sampling_enabled=self.on_device_sampling
            )

        self.layers = nn.ModuleList(
            [NeuronCohere2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NeuronCohere2LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _create_attn_mask_for_context_processing(self, 
                                                 attention_mask_2d: torch.LongTensor, 
                                                 has_sliding_window: bool, 
                                                 **kwargs) -> torch.BoolTensor:
        """Create a 4D attention mask for context processing (prefill).

        Examples of input zero-padded 2D attention masks (batch_size=2, bucket_size=10), 0 = masked token:
        * Left-padding:
        [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
        * Right-padding:
        [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]

        Args:
            attention_mask_2d (torch.LongTensor): Zero-padded 2D attention mask of shape [batch_size, bucket_size]
            has_sliding_window (bool): Whether or not to add sliding window masking.

        Returns:
            torch.BoolTensor: 4D attention mask of shape [batch_size, 1, bucket_size, bucket_size]
        """
        # 2D global attention mask of shape [bucket_size, bucket_size]
        tri_attn_mask_2d = torch.full((self.n_positions, self.n_positions), True, device=attention_mask_2d.device)\
            .tril(diagonal=0)

        if has_sliding_window and (self.n_positions > self.sliding_window_size):
            sliding_window_mask_2d = torch.logical_not(
                torch.full((self.n_positions, self.n_positions), True, device=attention_mask_2d.device)\
                    .tril(diagonal=-self.sliding_window_size))
            tri_attn_mask_2d = torch.logical_and(tri_attn_mask_2d, sliding_window_mask_2d)

        # Expand to 4D attention mask of shape [batch_size, 1, max_total_seq_len, max_total_seq_len]
        attn_mask_4d = tri_attn_mask_2d[None, None, :, :]\
            .expand(self.batch_size, 1, self.n_positions, self.n_positions)

        if self.padding_side == "left":
            padding_mask_4d = attention_mask_2d[:, None, None, :]\
                .expand(self.batch_size, 1, self.n_positions, self.n_positions)\
                .to(dtype=torch.bool)
            attn_mask_4d = torch.logical_and(attn_mask_4d, padding_mask_4d)
        return attn_mask_4d
    
    def _create_attn_mask_for_token_generation(self,
                                               attention_mask_2d: torch.LongTensor, 
                                               position_ids: torch.LongTensor,
                                               has_sliding_window: bool, 
                                               **kwargs) -> torch.BoolTensor:
        """Create a 4D attention mask for token generation.
        The output 4D attention mask is required to be of shape (B, 1, len_q, len_k), i.e. (B, 1, 1, len_k) since the 
        query tensor is of length 1 during the token generation phase.
        In the token generation phase, the 4D attention mask is used for computing the attention scores using the query and 
        the K cache slice. len_k is therefore the length of the K cache slice, i.e. the bucket size `n_positions`. If the 
        layer is a sliding window attention layer and if the bucket size is larger than the window size, then the K cache 
        slice (and therefore len_k) has the same length as the window size. 
        The output 4D attention mask must be consistent with the K cache slice. In the case of sliding window layers in 
        particular, it must account for the fact that the K cache has possibly been rolled.

        Examples of inputs & outputs with sliding window enabled for batch_size=2, bucket_size=10, sliding_window_size=6), 
        0 = masked token:
        * Left-padding:
          - attention_mask_2d
            [[0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
          - position_ids
            [[4],
             [7]]
          - output attention mask (2D-slice)
            [[0, 0, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1]]
        * Right-padding:
          - attention_mask_2d
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
          - position_ids
            [[4],
             [7]]
          - output attention mask (2D-slice)
            [[1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1]]

        Args:
            attention_mask_2d (torch.LongTensor): Zero-padded 2D attention mask of shape [batch_size, bucket_size]
            position_ids (torch.LongTensor): Position IDs tensor of shape [batch_size, 1]
            has_sliding_window (bool): Whether or not to account for sliding window masking.

        Returns:
            torch.BoolTensor: 4D attention mask of shape [batch_size, 1, 1, bucket_size] for global 
            attention, [batch_size, 1, 1, sliding_window_size] for sliding-window attention
        """
        if has_sliding_window and (self.n_positions > self.sliding_window_size):
            if self.padding_side == "left":
                max_position_ids = torch.max(position_ids)[None, None].expand(self.batch_size, -1)
                #max_position_ids = torch.amax(position_ids, keepdim=True).expand(self.batch_size, -1)
            else:
                max_position_ids = position_ids
            
            offset = torch.clamp(max_position_ids - self.sliding_window_size, min=0)
            index = torch.arange(self.sliding_window_size, device=attention_mask_2d.device)[None, :] + offset
            attn_mask_2d = torch.gather(attention_mask_2d, dim=1, index=index).to(dtype=torch.bool)

            if self.padding_side == "left":
                leftmost_token_mask = torch.full_like(offset, False, dtype=torch.bool, device=attention_mask_2d.device)
            else:
                leftmost_token_mask = position_ids < torch.full_like(position_ids, self.sliding_window_size, dtype=position_ids.dtype, device=attention_mask_2d.device)
            
            sliding_window_mask_2d = torch.full((self.batch_size, self.sliding_window_size-1), True, device=attention_mask_2d.device)
            sliding_window_mask_2d = torch.cat([leftmost_token_mask, sliding_window_mask_2d], dim=1)
            attn_mask_2d = torch.logical_and(attn_mask_2d, sliding_window_mask_2d)
            
            attn_mask_4d = attn_mask_2d[:, None, None, :]
        else:
            attn_mask_4d = attention_mask_2d[:, None, None, :].to(dtype=torch.bool)
        return attn_mask_4d

    def create_attn_mask(self, 
                         attention_mask: torch.LongTensor,
                         is_for_context_encoding: bool, 
                         is_for_speculation: bool, 
                         **kwargs) -> Tuple[torch.BoolTensor]:
        """Create 4D attention masks of shape [batch_size, 1, query_len, key_len] for models with both sliding-window and 
        global attention layers:
            - For context processing masks: 
                - query_len=bucket_size=n_positions
                - key_len=bucket_size=n_positions
            - For token generation masks:
                - query_len=1 
                - key_len=bucket_size=n_positions if bucket_size<sliding_window_size, otherwise: key_len=sliding_window_size 

        Args:
            attention_mask (torch.FloatTensor): Zero-padded 2D attention mask 0=padded/masked token, 1=attended token of shape 
            [batch_size, bucket_size]. If context encoding phase, padded to neuron_config.context_max_length or to context 
            bucket size. If token generation phase, padded to neuron_config.max_length or token generation bucket size. 
            Cf. `ModelWrapper.pad_to_max_compiled_seq`.
            is_for_context_encoding (bool): Whether masks are to be used by the context encoding model. If not, generate masks
            for token generation.
            is_for_speculation (bool): Whether speculation is enabled.

        Returns:
            Tuple[torch.BoolTensor]: Tuple of 4D attention masks. First masks is for global attention layers, second is for 
            sliding window layers.
        """
        if is_for_context_encoding:
            return (
                self._create_attn_mask_for_context_processing(
                    attention_mask_2d=attention_mask, 
                    has_sliding_window=False
                    ),
                self._create_attn_mask_for_context_processing(
                    attention_mask_2d=attention_mask, 
                    has_sliding_window=True
                    )
            )
        elif is_for_speculation:
            raise NotImplementedError("Speculative decoding is currently not supported for sliding window models")
        elif self.is_prefix_caching:
            raise NotImplementedError("Prefix caching is currently not supported for sliding window models")
        elif self.is_chunked_prefill:
            raise NotImplementedError("Chunked prefill is currently not supported for sliding window models")
        else: # Token generation
            return (
                self._create_attn_mask_for_token_generation(
                    attention_mask_2d=attention_mask, 
                    position_ids=kwargs["position_ids"], 
                    has_sliding_window=False
                    ),
                self._create_attn_mask_for_token_generation(
                    attention_mask_2d=attention_mask, 
                    position_ids=kwargs["position_ids"], 
                    has_sliding_window=True
                    )
            )


class NeuronCohere2ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronCohere2Model

    @classmethod
    def get_config_cls(cls) -> Type:
         return Cohere2InferenceConfig

    @staticmethod
    def load_hf_model(model_path: str, **kwargs) -> Cohere2ForCausalLM:
        return Cohere2ForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: Dict[str, Any], config: InferenceConfig) -> Dict[str, Any]:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # Temporary workaround for getting rank information in traced SPMD model. 
            # Will removed and replaced with ReplicaID in HLO once compiler adds support
            # Rank ID information is required when vocab parallelism is enabled to compute masks signaling which 
            # embeddings are available locally
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )
        
        # Rank ID information is required when Flash Decoding is enabled to compute masks signaling which sequence chunk 
        # is available locally
        # Add rank information to each attention layer
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        # Add rank information at the model level
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: Dict[str, Any]) -> None:
        state_dict["lm_head._lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, config: InferenceConfig) -> Dict[str, Any]:
        q_hf_state_dict = super().generate_quantized_state_dict(model_path=model_path, config=config)
        # Required since contrary to NeuronApplicationBase.get_state_dict, NeuronApplicationBase.get_quantized_state_dict 
        # does not call NeuronApplicationBase.update_state_dict_for_tied_weights. However, get_quantized_state_dict still 
        # removes "model." prefixes.
        q_hf_state_dict["lm_head._lm_head.weight"] = q_hf_state_dict["lm_head.weight"]
        del q_hf_state_dict["lm_head.weight"]
        if "lm_head" not in config.neuron_config.modules_to_not_convert:
            q_hf_state_dict["lm_head._lm_head.scale"] = q_hf_state_dict["lm_head.scale"]
            del q_hf_state_dict["lm_head.scale"]
        return q_hf_state_dict
