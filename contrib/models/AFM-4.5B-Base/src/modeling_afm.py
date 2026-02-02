# coding=utf-8
# Copyright 2025 Arcee AI and the HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch AFM-4.5B-Base (Arcee) model for NeuronX Distributed Inference.

This implementation is based on the Arcee architecture from HuggingFace transformers
with modifications for AWS Neuron/Trainium hardware.

Key architectural features:
- Grouped Query Attention (GQA) with 20 Q heads and 4 KV heads
- Simple MLP with ReLU^2 activation (not GLU-based)
- YARN RoPE scaling for extended context (65k tokens) - FIXED IMPLEMENTATION
- RMSNorm for layer normalization
"""

import copy
import json
import logging
import math
import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import BaseGroupQueryAttention
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    """
    Initialize to the appropriate implementation of RMSNorm
    If infer on NXD -> CustomRMSNorm
    If infer on CPU -> torch.nn.RMSNorm (CustomRMSNorm does not work on CPU)
    """
    # For CPU mode, use a simple RMSNorm implementation
    if cpu_mode():
        class SimpleRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

        return SimpleRMSNorm
    else:
        return CustomRMSNorm


class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) Rotary Position Embedding for NeuronX.

    This implements the YaRN RoPE scaling mechanism that allows AFM to handle
    extended context lengths (up to 65k tokens) by applying frequency-dependent
    scaling to the rotary embedding.

    The key insight from YaRN is that different frequency dimensions should be
    scaled differently:
    - Low-frequency dimensions (high wavelength): Use interpolation (scale by factor)
    - High-frequency dimensions (low wavelength): Keep extrapolation (no scaling)
    - Middle frequencies: Linear blend between the two

    Reference: https://huggingface.co/papers/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 65536,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        device=None,
    ):
        """
        Initialize YaRN rotary embedding.

        Args:
            dim: Dimension of the rotary embedding (head_dim)
            max_position_embeddings: Maximum sequence length
            base: RoPE theta base
            rope_scaling: YaRN scaling configuration containing:
                - factor: Context extension factor (e.g., 20.0)
                - beta_fast: Fast boundary for extrapolation (default 32)
                - beta_slow: Slow boundary for interpolation (default 1)
                - mscale: Magnitude scaling factor (default 1.0)
                - original_max_position_embeddings: Original context length (e.g., 4096)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Parse YaRN configuration
        if rope_scaling is None:
            rope_scaling = {}

        self.factor = rope_scaling.get("factor", 1.0)
        self.beta_fast = rope_scaling.get("beta_fast", 32.0)
        self.beta_slow = rope_scaling.get("beta_slow", 1.0)
        self.mscale = rope_scaling.get("mscale", 1.0)
        self.original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 4096
        )

        # Compute the attention scaling factor
        self.attention_factor = self._compute_attention_factor()

        # Precompute inverse frequencies with YaRN scaling
        self.register_buffer("inv_freq", None, persistent=False)
        self._compute_inv_freq(device)

        logger.info(f"YaRNRotaryEmbedding: dim={dim}, base={base}, "
                   f"max_pos={max_position_embeddings}, "
                   f"original_max_pos={self.original_max_position_embeddings}, "
                   f"factor={self.factor}, beta_fast={self.beta_fast}, "
                   f"beta_slow={self.beta_slow}, mscale={self.mscale}, "
                   f"attention_factor={self.attention_factor:.4f}")

    def _compute_attention_factor(self) -> float:
        """
        Compute the attention scaling factor based on mscale.

        For YaRN, the attention factor helps compensate for the scaling
        applied to the rotary embeddings.
        """
        if self.factor <= 1:
            return 1.0
        return 0.1 * self.mscale * math.log(self.factor) + 1.0

    def _find_correction_dim(self, num_rotations: float) -> float:
        """
        Find the dimension based on the number of rotations.

        This is the inverse of the frequency formula to determine which
        dimension corresponds to a given rotation frequency.
        """
        return (
            self.dim * math.log(self.original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(self.base))

    def _find_correction_range(self) -> Tuple[float, float]:
        """
        Find the dimension range for the correction ramp.

        Returns the low and high dimensions that define the transition
        zone between extrapolation and interpolation.
        """
        low = self._find_correction_dim(self.beta_fast)
        high = self._find_correction_dim(self.beta_slow)
        # Clamp to valid range
        low = max(math.floor(low), 0)
        high = min(math.ceil(high), self.dim - 1)
        return low, high

    def _compute_inv_freq(self, device=None):
        """
        Compute inverse frequencies with YaRN scaling.

        The key YaRN algorithm:
        1. Compute base inverse frequencies (extrapolation)
        2. Compute scaled inverse frequencies (interpolation)
        3. Use linear ramp to blend between them based on dimension
        """
        # Find the correction range
        low, high = self._find_correction_range()

        # Create linear ramp function for blending
        # 0 = use extrapolation, 1 = use interpolation
        dim_range = torch.arange(self.dim // 2, dtype=torch.float32, device=device)

        # Linear ramp from 0 (at low) to 1 (at high)
        if low == high:
            high = low + 0.001  # Prevent division by zero
        linear_func = (dim_range - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)

        # Compute base frequencies
        pos_freqs = self.base ** (2 * dim_range / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)

        # Blend using the ramp function
        # extrapolation_factor = 1 - ramp_func (use extrapolation where ramp is 0)
        inv_freq_extrapolation_factor = 1 - ramp_func
        self.inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings with YaRN scaling.

        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            position_ids: Position indices [batch, seq_len]

        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        # Ensure inv_freq is on the correct device
        if self.inv_freq is None or self.inv_freq.device != x.device:
            self._compute_inv_freq(x.device)

        # Expand inv_freq for batch computation
        # inv_freq: [dim/2] -> [1, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float()

        # position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Compute frequencies: [batch, dim/2, seq_len]
        freqs = inv_freq_expanded @ position_ids_expanded

        # Transpose to [batch, seq_len, dim/2]
        freqs = freqs.transpose(1, 2)

        # Concatenate for full dimension: [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply attention factor scaling and convert to target dtype
        # Note: HF applies attention_factor as post-scaling to cos/sin values
        cos = (emb.cos() * self.attention_factor).to(dtype=x.dtype)
        sin = (emb.sin() * self.attention_factor).to(dtype=x.dtype)

        return cos, sin


class AFMInferenceConfig(InferenceConfig):
    """
    Configuration class for AFM (Arcee) model inference on NeuronX.

    Inherits from InferenceConfig and adds AFM-specific parameters.
    """

    def __init__(
        self,
        neuron_config: Optional[NeuronConfig] = None,
        vocab_size: int = 128004,
        hidden_size: int = 2560,
        intermediate_size: int = 18432,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        hidden_act: str = "relu2",
        max_position_embeddings: int = 65536,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        **kwargs,
    ):
        """
        Initialize AFM configuration.

        Args:
            neuron_config: NeuronX-specific configuration
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            intermediate_size: MLP intermediate dimension
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key-value heads for GQA
            head_dim: Dimension of each attention head
            hidden_act: Activation function (relu2 for AFM)
            max_position_embeddings: Maximum sequence length
            initializer_range: Weight initialization range
            rms_norm_eps: RMSNorm epsilon
            use_cache: Whether to use KV cache
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            tie_word_embeddings: Whether to tie embeddings and LM head
            rope_theta: RoPE theta parameter
            rope_scaling: RoPE scaling configuration (YARN for AFM)
            attention_bias: Whether to use bias in attention layers
            attention_dropout: Attention dropout probability
            mlp_bias: Whether to use bias in MLP layers
        """
        # Set all attributes BEFORE calling parent __init__
        # because parent calls add_derived_config()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # Additional attributes required by base class
        self.output_attentions = False
        self.output_hidden_states = False
        self.return_dict = True

        # Now call parent __init__ which will call add_derived_config()
        # If neuron_config is None, create a default one to avoid validation errors
        if neuron_config is None:
            print("[AFM Config] Warning: neuron_config is None, creating default")
            neuron_config = NeuronConfig()

        super().__init__(
            neuron_config=neuron_config,
            **kwargs
        )

    def add_derived_config(self):
        """Add derived configuration parameters."""
        self.num_cores_per_group = 1

        # Ensure head_dim is set correctly
        if not hasattr(self, 'head_dim') or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def get_required_attributes(self) -> List[str]:
        """List of required attributes for the configuration."""
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        """Return the NeuronConfig class to use."""
        return NeuronConfig

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "AFMInferenceConfig":
        """
        Load configuration from a pretrained model directory.

        Args:
            model_path: Path to the model directory containing config.json
            **kwargs: Additional arguments to override configuration

        Returns:
            AFMInferenceConfig: Configuration object
        """
        # Extract neuron_config from kwargs if it exists
        neuron_config = kwargs.pop("neuron_config", None)

        # Read config file
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Override with kwargs
        config_dict.update(kwargs)

        # Create config object
        config = cls(neuron_config=neuron_config, **config_dict)

        print(f"[AFM Config] Loaded configuration from {model_path}")
        print(f"  - Model: AFM-4.5B-Base (Arcee)")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Attention heads: {config.num_attention_heads}")
        print(f"  - KV heads: {config.num_key_value_heads} (GQA)")
        print(f"  - Vocab size: {config.vocab_size}")
        print(f"  - Max position embeddings: {config.max_position_embeddings}")
        print(f"  - RoPE scaling: {config.rope_scaling}")
        print(f"  - Activation: {config.hidden_act}")

        return config


class NeuronAFMMLP(nn.Module):
    """
    AFM MLP implementation for NeuronX.

    AFM uses a simple 2-layer MLP with ReLU^2 activation (NOT GLU-based).

    Architecture:
        x -> up_proj -> relu^2 -> down_proj -> output

    This is different from LLaMA which uses:
        x -> gate_proj -> silu -> * up_proj -> down_proj
    """

    def __init__(self, config: AFMInferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Up projection (hidden_size -> intermediate_size)
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )

        # Down projection (intermediate_size -> hidden_size)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
        )

        # ReLU^2 activation (x.relu().pow(2))
        # Note: We implement this inline in forward() for efficiency

    def forward(self, hidden_states):
        """
        Forward pass of AFM MLP.

        Args:
            hidden_states: Input tensor

        Returns:
            Tuple of (output, None) - None for compatibility with framework
        """
        # Up projection
        up_out = self.up_proj(hidden_states)

        # ReLU^2 activation: relu(x)^2
        # This is equivalent to: x.relu().pow(2)
        activated = torch.relu(up_out).pow(2)

        # Down projection
        output = self.down_proj(activated)

        return output, None


class NeuronAFMAttention(NeuronAttentionBase):
    """
    AFM Attention implementation for NeuronX with YaRN RoPE scaling.

    Uses Grouped Query Attention (GQA) with:
    - 20 query heads
    - 4 key-value heads
    - YaRN RoPE for extended context support (65k tokens)
    """

    def __init__(self, config: AFMInferenceConfig, layer_idx: int):
        # Initialize YaRN rotary embeddings with proper scaling
        # This is the key fix - use YaRNRotaryEmbedding instead of basic RotaryEmbedding
        rotary_emb = YaRNRotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Initialize base attention with AFM parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rope_theta=config.rope_theta,
            qkv_bias=config.attention_bias,
            o_bias=config.attention_bias,
            num_cores_per_group=config.num_cores_per_group,
        )

        self.layer_idx = layer_idx


class NeuronAFMDecoderLayer(nn.Module):
    """
    AFM Decoder Layer for NeuronX.

    Architecture:
        x = x + attention(norm(x))
        x = x + mlp(norm(x))
    """

    def __init__(self, config: AFMInferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Self-attention with GQA
        self.self_attn = NeuronAFMAttention(config, layer_idx)

        # MLP with ReLU^2
        self.mlp = NeuronAFMMLP(config)

        # Layer normalization
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass of AFM decoder layer.

        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key-value pairs
            residual: Residual tensor from previous layer

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        """
        # Save entry hidden states for residual
        entry_hidden_states = hidden_states

        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention - returns NeuronAttentionBaseOutput dataclass
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        # Extract outputs from attention
        hidden_states = attn_output.hidden_states if hasattr(attn_output, 'hidden_states') else attn_output[0]
        present_key_value = attn_output.present_key_value if hasattr(attn_output, 'present_key_value') else attn_output[1]
        cos_cache = attn_output.cos_cache if hasattr(attn_output, 'cos_cache') else None
        sin_cache = attn_output.sin_cache if hasattr(attn_output, 'sin_cache') else None

        # First residual connection
        residual = entry_hidden_states
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Return format: (hidden_states, present_key_value, cos_cache, sin_cache, residual)
        # Set residual to None as we've already added it
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronAFMModel(NeuronBaseModel):
    """
    AFM Base Model for NeuronX Distributed Inference.

    This is the core transformer model without the language modeling head.
    """

    def setup_attr_for_model(self, config: AFMInferenceConfig):
        """Setup attributes needed for model initialization."""
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: AFMInferenceConfig):
        """Initialize model components."""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings and lm_head
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
            )

            # Language modeling head
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )

            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # Decoder layers
        self.layers = nn.ModuleList([
            NeuronAFMDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer normalization
        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        print(f"[AFM Model] Initialized with {config.num_hidden_layers} layers (YaRN RoPE enabled)")

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.embed_tokens = value


class NeuronAFMForCausalLM(NeuronBaseForCausalLM):
    """
    AFM Causal Language Model for NeuronX Distributed Inference.

    This wraps the base model and adds the language modeling head.
    """

    _model_cls = NeuronAFMModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config: AFMInferenceConfig):
        """
        Convert HuggingFace AFM checkpoint to NeuronX format.

        Key transformations:
        1. Remove "model." prefix
        2. Transform QKV projections:
           - layers.{i}.self_attn.{q,k,v}_proj -> layers.{i}.self_attn.qkv_proj.{q,k,v}_proj
        3. Transform o_proj to nested structure (GroupQueryAttention_O has nested o_proj):
           - layers.{i}.self_attn.o_proj -> layers.{i}.self_attn.o_proj.o_proj

        Input (HF format):
        - model.embed_tokens.weight
        - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        - model.layers.{i}.mlp.{gate,up,down}_proj.weight
        - model.norm.weight
        - lm_head.weight

        Output (NeuronX format after this function):
        - embed_tokens.weight
        - layers.{i}.self_attn.qkv_proj.{q,k,v}_proj.weight
        - layers.{i}.self_attn.o_proj.o_proj.weight
        - layers.{i}.mlp.{gate,up,down}_proj.weight
        - norm.weight
        - lm_head.weight

        Args:
            state_dict: HuggingFace state dictionary
            config: AFM configuration

        Returns:
            NeuronX-format state dictionary
        """
        neuron_state_dict = {}

        print(f"[Weight Conversion] Converting HuggingFace AFM checkpoint to NeuronX format")
        print(f"  - Original keys: {len(state_dict)}")

        # Convert each weight:
        # 1. Remove "model." prefix
        # 2. Transform QKV projection keys to qkv_proj.{q,k,v}_proj
        # 3. Transform o_proj to o_proj.o_proj (matches GroupQueryAttention_O structure)
        for key, value in state_dict.items():
            # Remove "model." prefix if it exists
            if key.startswith("model."):
                neuron_key = key[6:]  # Remove "model." prefix
            else:
                neuron_key = key

            # Transform QKV projection keys to match GroupQueryAttention_QKV module structure
            if ".self_attn.q_proj." in neuron_key:
                neuron_key = neuron_key.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.q_proj.")
            elif ".self_attn.k_proj." in neuron_key:
                neuron_key = neuron_key.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.k_proj.")
            elif ".self_attn.v_proj." in neuron_key:
                neuron_key = neuron_key.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.v_proj.")
            # Note: o_proj is left as-is; preshard_hook in GroupQueryAttention_O handles the transformation

            neuron_state_dict[neuron_key] = value.clone()

        # Add rank utilities for tensor parallelism
        tp_degree = config.neuron_config.tp_degree
        for i in range(config.num_hidden_layers):
            neuron_state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        print(f"  - Converted keys: {len(neuron_state_dict)}")
        print(f"  - Added rank utilities for {config.num_hidden_layers} layers")

        return neuron_state_dict


# Export main classes
__all__ = [
    "AFMInferenceConfig",
    "YaRNRotaryEmbedding",
    "NeuronAFMMLP",
    "NeuronAFMAttention",
    "NeuronAFMDecoderLayer",
    "NeuronAFMModel",
    "NeuronAFMForCausalLM",
]
