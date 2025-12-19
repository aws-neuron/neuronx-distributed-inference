import torch
from typing import Optional, Tuple

from neuronxcc.nki.language import nc
import neuronxcc.nki.language as nl
from neuronxcc.nki._pre_prod_kernels import RouterActFnType, ExpertAffinityScaleMode, ActFnType
from neuronxcc.nki._pre_prod_kernels.moe_token_gen import moe_token_gen_all_experts_kernel


def tokengen_moe_megakernel_forward_all_experts(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    W_router: torch.Tensor,
    W_expert_gate_up: torch.Tensor,
    W_expert_down: torch.Tensor,
    rank_id: torch.Tensor,
    W_expert_gate_up_scale: Optional[torch.Tensor] = None,
    W_expert_down_scale: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: str = 'sigmoid',
    router_pre_norm: bool = True,
    expert_affinities_scaling_mode: str = 'no_scale',
    hidden_act_fn: str = 'silu',
    router_matmul_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Token generation MOE (Mixture of Experts) computation using NKI kernel without shared expert support.

    .. note::
        Implementation details:
        The function calls the ``moe_token_gen_all_experts_kernel`` which runs with
        Logical NC Config = 2. The kernel performs fused operations including RMSNorm, router computation,
        expert selection via top-k routing, expert gate/up projections, activation functions, expert down
        projections, and final output aggregation in a single optimized kernel.

    .. note::
        This implementation supports regular experts only. Each token is routed to the top-k experts
        based on router logits, and the expert outputs are weighted by the routing probabilities.
        For models requiring shared experts that are always activated, use
        ``tokengen_moe_megakernel_forward_all_experts_with_shared_experts`` instead.

    :param hidden_states: Input hidden states from the transformer layer.
                         Shape: ``(batch_size, sequence_length, hidden_size)``
                         These are the token embeddings that will be processed through the MOE layer.
    :param gamma: RMSNorm scaling parameter (gamma) applied to hidden states before MOE computation.
                  Shape: ``(1, hidden_size)``
                  Used in the fused RMSNorm operation when ``router_pre_norm=True``.
    :param W_router: Router weight matrix that determines expert selection for each token.
                     Shape: ``(hidden_size, num_experts)``
                     Projects normalized hidden states to expert affinity scores for routing decisions.
    :param W_expert_gate_up: Combined gate and up projection weights for all local experts.
                            Shape: ``(num_local_experts, hidden_size, 2, intermediate_size)``
                            Contains both gate and up weights where the third dimension separates
                            gate weights (index 0) and up weights (index 1) for each expert.
    :param W_expert_down: Down projection weights for all local experts.
                         Shape: ``(num_local_experts, intermediate_size, hidden_size)``
                         Projects expert intermediate representations back to hidden_size.
    :param rank_id: Rank identifier tensor for distributed expert placement.
                    Shape: ``(1, 1)``
                    Used to determine which experts are local to the current device in distributed setups.
    :param W_expert_gate_up_scale: Optional quantization scale factors for expert gate/up weights.
                                  Shape: ``(num_local_experts, 2, intermediate_size)`` when provided
                                  Used for dequantization of gate/up weights during computation.
    :param W_expert_down_scale: Optional quantization scale factors for expert down weights.
                               Shape: ``(num_local_experts, hidden_size)`` when provided
                               Used for dequantization of down weights during computation.
    :param eps: Epsilon value for RMSNorm computation to prevent division by zero.
               Typically set to 1e-6 or 1e-5. Used in the fused RMSNorm operation.
    :param top_k: Number of top experts to route each token to.
                 Common values are 1, 2, or 4. Higher values increase computation but may improve quality.
                 Must be less than or equal to the total number of experts.
    :param router_act_fn: Activation function applied to router logits for expert selection.
                         Supported values: ``'sigmoid'``, ``'softmax'``
                         Controls how routing probabilities are computed from router outputs.
    :param router_pre_norm: Whether to apply RMSNorm to hidden states before router computation.
                           When True, uses ``gamma`` and ``eps`` for normalization.
                           When False, assumes input is already normalized.
    :param expert_affinities_scaling_mode: Mode for scaling expert affinity scores during routing.
                                          Supported values: ``'no_scale'``, ``'post_scale'``, ``'pre_scale'``
                                          Controls when and how scaling is applied to routing decisions.
    :param hidden_act_fn: Activation function applied within expert feed-forward networks.
                         Supported values: ``'silu'``, ``'gelu'``, ``'swish'``
                         Applied between gate/up projections and down projection in each expert.
    :param router_matmul_dtype: Data type for router matrix multiplication operations.
                               Supported values: ``torch.bfloat16``, ``torch.float16``, ``torch.float32``
                               Controls precision vs performance trade-off for routing computations.

    :return: Tuple containing:
             - ``moe_output``: MOE layer output tensor with shape ``(batch_size * sequence_length, hidden_size)``
                              Contains the weighted combination of selected expert outputs for each token
             - ``router_logits``: Router logit scores with shape ``(batch_size * sequence_length, num_experts)``
                                  Raw affinity scores before top-k selection, useful for load balancing analysis
    """
    LOGICAL_NC_CONFIG = 2  # Underlying NKI kernel runs with LNC=2

    # TODO: Add input validation for kernel parameters

    # Convert string parameters to enums using helper functions
    router_act_enum = _get_router_act_fn_enum(router_act_fn)
    scaling_mode_enum = _get_scaling_mode_enum(expert_affinities_scaling_mode)
    hidden_act_enum = _get_hidden_act_fn_enum(hidden_act_fn)

    # Convert torch dtype to nl dtype for router_mm_dtype
    router_mm_dtype_nl = _torch_dtype_to_nl_dtype(router_matmul_dtype)

    # Invoke NKI kernel
    grid = (nc(LOGICAL_NC_CONFIG),)
    output, router_logits = moe_token_gen_all_experts_kernel[grid](
        inp=hidden_states,
        gamma=gamma,
        router_weights=W_router,
        expert_gate_up_weights=W_expert_gate_up,
        expert_down_weights=W_expert_down,
        rank_id=rank_id,
        shared_expert_gate_w=None,
        shared_expert_up_w=None,
        shared_expert_down_w=None,
        expert_gate_up_weights_scale=W_expert_gate_up_scale,
        expert_down_weights_scale=W_expert_down_scale,
        shared_expert_gate_bias=None,
        shared_expert_up_bias=None,
        shared_expert_down_bias=None,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_enum,
        router_pre_norm=router_pre_norm,
        expert_affinities_scaling_mode=scaling_mode_enum,
        hidden_act_fn=hidden_act_enum,
        router_mm_dtype=router_mm_dtype_nl
    )

    return output, router_logits


def _get_router_act_fn_enum(router_act_fn: str) -> RouterActFnType:
    """Convert router activation function string to enum."""
    router_act_fn_map = {
        'sigmoid': RouterActFnType.SIGMOID,
        'softmax': RouterActFnType.SOFTMAX,
        # Add other activation functions as needed
    }

    if router_act_fn.lower() not in router_act_fn_map:
        raise ValueError(f"Unsupported router activation function: {router_act_fn}")

    return router_act_fn_map[router_act_fn.lower()]


def _get_scaling_mode_enum(expert_affinities_scaling_mode: str) -> ExpertAffinityScaleMode:
    """Convert expert affinity scaling mode string to enum."""
    scaling_mode_map = {
        'no_scale': ExpertAffinityScaleMode.NO_SCALE,
        'post_scale': ExpertAffinityScaleMode.POST_SCALE,
        'pre_scale': ExpertAffinityScaleMode.PRE_SCALE,
        # Add other modes as needed
    }

    if expert_affinities_scaling_mode.lower() not in scaling_mode_map:
        raise ValueError(f"Unsupported expert affinity scaling mode: {expert_affinities_scaling_mode}")

    return scaling_mode_map[expert_affinities_scaling_mode.lower()]


def _get_hidden_act_fn_enum(hidden_act_fn: str) -> ActFnType:
    """Convert hidden activation function string to enum."""
    hidden_act_map = {
        'silu': ActFnType.SiLU,
        'gelu': ActFnType.GELU,
        'swish': ActFnType.Swish,
        # Add other activation functions as needed
    }

    if hidden_act_fn.lower() not in hidden_act_map:
        raise ValueError(f"Unsupported hidden activation function: {hidden_act_fn}")

    return hidden_act_map[hidden_act_fn.lower()]


def _torch_dtype_to_nl_dtype(torch_dtype: torch.dtype):
    """Convert torch dtype to neuronxcc.nki.language dtype for router_mm_dtype parameter."""
    dtype_map = {
        torch.bfloat16: nl.bfloat16,
        torch.float16: nl.float16,
        torch.float32: nl.float32,
    }

    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported router_matmul_dtype: {torch_dtype}. "
                         f"Supported dtypes: {list(dtype_map.keys())}")

    return dtype_map[torch_dtype]
