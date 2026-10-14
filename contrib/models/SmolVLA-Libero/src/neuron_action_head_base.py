"""
neuron_action_head_base
=======================

Minimal config shim used by the three SmolVLA subgraphs.

SmolVLA is a flow-matching VLA (vision-language-action) policy, not a CausalLM,
so it cannot reuse NxDI's stock ``InferenceConfig`` — that config carries
LLM-specific fields (KV-cache layout, sequence buckets, vocab size, etc.) that
have no meaning here.

``NeuronDenoisingConfig`` exposes the small set of attributes that
``ModelWrapper.__init__()`` actually reads (``neuron_config.torch_dtype``,
``neuron_config.tp_degree``, ``neuron_config.batch_size``, ``pad_token_id``)
plus a handful of action-head-specific fields used by the per-subgraph wrappers
in ``modeling_smolvla_vision.py`` and ``modeling_smolvla_text.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


COMPILED_MODEL_FILE_NAME = "model.pt"


class NeuronDenoisingConfig:
    """Minimal config that satisfies ``ModelWrapper.__init__()``.

    LLM-specific fields are stubbed as ``None`` / ``False`` / ``0``. Action-head
    fields (``action_chunk_size`` etc.) are exposed at the top level so each
    subgraph wrapper can pull static input shapes from them.
    """

    def __init__(
        self,
        batch_size: int,
        tp_degree: int,
        action_chunk_size: int,
        action_dim: int,
        num_conditioning_tokens: int,
        conditioning_hidden_size: int,
        timestep_embed_dim: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.neuron_config = SimpleNamespace(
            # Core — required by ModelWrapper
            torch_dtype=torch_dtype,
            tp_degree=tp_degree,
            batch_size=batch_size,
            # Compiler tuning
            cc_pipeline_tiling_factor=1,
            logical_nc_config=1,
            # LLM features — all disabled (we are not a CausalLM)
            is_block_kv_layout=False,
            is_prefix_caching=False,
            is_medusa=False,
            token_generation_batches=None,
            async_mode=False,
            scratchpad_page_size=None,
            attn_block_tkg_nki_kernel_enabled=False,
            enable_long_context_mode=False,
            layer_boundary_markers=False,
            dma_order_config=None,
            enable_spill_reload_dge=False,
            target=None,
            quantized=False,
            quantization_dtype=None,
            kv_cache_quant=False,
            quantized_mlp_kernel_enabled=False,
            activation_quantization_type=None,
            enable_output_completion_notifications=False,
            # Weight loading
            save_sharded_checkpoint=True,
            start_rank_id=0,
            local_ranks_size=tp_degree,
            cast_type="config",
            # Parallelism
            pp_degree=1,
            ep_degree=1,
            world_size=tp_degree,
        )
        self.pad_token_id = 0

        # Action-head specific
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.num_conditioning_tokens = num_conditioning_tokens
        self.conditioning_hidden_size = conditioning_hidden_size
        self.timestep_embed_dim = timestep_embed_dim
