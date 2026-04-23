#!/usr/bin/env python3
"""Monkeypatch NxDI's NeuronLlamaMLP to use the NKI TKG kernel for token
generation while falling back to manual matmul for context encoding.

The built-in mlp_isa_kernel has a hard 4096 intermediate-dim limit per core,
which blocks S2 (intermediate_size=5504 at TP=1). The newer
nki_mlp_tkg_isa_kernel has NO such limit but only handles small batch*seqlen
(token gen). For CTE we do raw matmuls on the already-transposed weights.

Usage:
    import nki_mlp_patch  # must import BEFORE model.compile()
"""

import logging
import torch
import torch.nn.functional as F
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)

logger = logging.getLogger("Neuron")


def _patched_forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
    """Patched NeuronLlamaMLP.forward:
    - TKG path (batch_seqlen <= 128): use NKI TKG kernel (no dim limit)
    - CTE path (batch_seqlen > 128): manual matmul with transposed weights
      + all-reduce, bypassing mlp_isa_kernel's 4096 limit
    """
    TKG_BS_SEQLEN_THRESHOLD = 128

    if not self.mlp_kernel_enabled:
        # No kernel -- use native path
        assert rmsnorm is None and residual is None
        return (self._native_mlp(x, adapter_ids=adapter_ids), None)

    # Determine if this is a small batch*seqlen (token gen)
    if self.tensor_model_parallel_group is not None:
        tp_degree = self.tensor_model_parallel_group.size()
    else:
        tp_degree = self.config.neuron_config.tp_degree

    if self.sequence_parallel_enabled:
        real_seqlen = x.shape[1] * tp_degree
    else:
        real_seqlen = x.shape[1]

    batch_seqlen = x.shape[0] * real_seqlen
    is_small_batch_seqlen = batch_seqlen <= TKG_BS_SEQLEN_THRESHOLD

    # Import the TKG kernel check
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        _trace_nki_mlp_tkg_kernel,
    )

    use_tkg_nki_kernel = (
        _trace_nki_mlp_tkg_kernel
        and is_small_batch_seqlen
        and self.mlp_tkg_nki_kernel_enabled
    )

    if use_tkg_nki_kernel:
        # Token generation: use the NKI TKG kernel (no dimension limit)
        return self._kernel_enabled_nki_mlp_tkg(
            x, rmsnorm, residual, adapter_ids=adapter_ids
        )
    else:
        # Context encoding: manual matmul with transposed weights
        # Weights are already transposed by mlp_kernel_enabled=True init:
        #   gate_proj.weight: [H, I]  (transposed from [I, H])
        #   up_proj.weight:   [H, I]
        #   down_proj.weight: [I, H]  (transposed from [H, I])
        logger.debug("MLP: patched CTE path (manual matmul, bypassing 4096 limit)")

        # Handle fused rmsnorm if provided
        if rmsnorm is not None:
            x_normed = rmsnorm(x)
        else:
            x_normed = x

        # Handle sequence parallel
        if self.sequence_parallel_enabled:
            from neuronx_distributed_inference.modules.custom_calls import (
                gather_from_sequence_parallel_region,
            )

            x_normed = gather_from_sequence_parallel_region(
                x_normed,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )

        gate_w = self.gate_proj.weight.data  # [H, I]
        up_w = self.up_proj.weight.data  # [H, I]
        down_w = self.down_proj.weight.data  # [I, H]

        gate_out = torch.matmul(x_normed, gate_w)  # [B, S, I]
        up_out = torch.matmul(x_normed, up_w)  # [B, S, I]
        down_input = self.act_fn(gate_out) * up_out  # [B, S, I]
        output = torch.matmul(down_input, down_w)  # [B, S, H]

        # All-reduce across TP ranks
        if self.sequence_parallel_enabled:
            from neuronx_distributed_inference.modules.custom_calls import (
                reduce_scatter_to_sequence_parallel_region,
            )

            output = reduce_scatter_to_sequence_parallel_region(
                output,
                self.sequence_dimension,
                process_group=self.tensor_model_parallel_group,
            )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output,
                process_group=self.tensor_model_parallel_group,
            )

        # Handle residual add if requested
        residual_out = None
        if residual is not None:
            residual_out = residual + x_normed if rmsnorm is not None else residual + x

        return (output, residual_out)


def apply_patch():
    """Apply the monkeypatch to NeuronLlamaMLP."""
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP

    NeuronLlamaMLP.forward = _patched_forward
    logger.info(
        "NKI MLP patch applied: TKG kernel for token-gen, manual matmul for CTE"
    )


# Auto-apply on import
apply_patch()
