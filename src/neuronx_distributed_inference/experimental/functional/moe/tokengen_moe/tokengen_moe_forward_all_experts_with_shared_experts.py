import torch
from typing import Optional, Tuple

from neuronxcc.nki.language import nc
from neuronxcc.nki._pre_prod_kernels.moe_token_gen import moe_token_gen_all_experts_kernel

from .tokengen_moe_forward_all_experts import _get_router_act_fn_enum, _get_scaling_mode_enum, _get_hidden_act_fn_enum, _torch_dtype_to_nl_dtype


def tokengen_moe_megakernel_forward_all_experts_with_shared_experts(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    W_router: torch.Tensor,
    W_expert_gate_up: torch.Tensor,
    W_expert_down: torch.Tensor,
    rank_id: torch.Tensor,
    W_shared_expert_gate: torch.Tensor,
    W_shared_expert_up: torch.Tensor,
    W_shared_expert_down: torch.Tensor,
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
    Token generation MOE (Mixture of Experts) computation using NKI kernel with shared expert support.

    .. note::
        Implementation details:
        The function calls the ``moe_token_gen_all_experts_kernel`` which runs with
        Logical NC Config = 2. The kernel performs fused operations including RMSNorm, router computation,
        expert selection via top-k routing, regular expert processing, shared expert processing,
        and final output aggregation combining both regular and shared expert contributions.

    .. note::
        This implementation supports both regular experts and shared experts. Regular experts are
        selected via top-k routing based on router logits, while shared experts are always activated
        for every token. The final output combines weighted regular expert outputs with shared expert
        outputs, providing both specialized and general processing capabilities.

    :param hidden_states: Input hidden states from the transformer layer.
                         Shape: ``(batch_size, sequence_length, hidden_size)``
                         These are the token embeddings that will be processed through the MOE layer.
    :param gamma: RMSNorm scaling parameter (gamma) applied to hidden states before MOE computation.
                  Shape: ``(1, hidden_size)``
                  Used in the fused RMSNorm operation when ``router_pre_norm=True``.
    :param W_router: Router weight matrix that determines expert selection for each token.
                     Shape: ``(hidden_size, num_experts)``
                     Projects normalized hidden states to expert affinity scores for routing decisions.
                     Only affects regular expert selection; shared experts are always activated.
    :param W_expert_gate_up: Combined gate and up projection weights for all local regular experts.
                            Shape: ``(num_local_experts, hidden_size, 2, intermediate_size)``
                            Contains both gate and up weights where the third dimension separates
                            gate weights (index 0) and up weights (index 1) for each regular expert.
    :param W_expert_down: Down projection weights for all local regular experts.
                         Shape: ``(num_local_experts, intermediate_size, hidden_size)``
                         Projects regular expert intermediate representations back to hidden_size.
    :param rank_id: Rank identifier tensor for distributed expert placement.
                    Shape: ``(1, 1)``
                    Used to determine which experts are local to the current device in distributed setups.
    :param W_shared_expert_gate: Gate projection weights for the shared expert.
                                Shape: ``(hidden_size, intermediate_size)``
                                Applied to all tokens regardless of routing decisions.
                                Works in conjunction with ``W_shared_expert_up`` for gated activation.
    :param W_shared_expert_up: Up projection weights for the shared expert.
                              Shape: ``(hidden_size, intermediate_size)``
                              Applied to all tokens and combined with gate projection via element-wise
                              multiplication after activation function application.
    :param W_shared_expert_down: Down projection weights for the shared expert.
                                Shape: ``(intermediate_size, hidden_size)``
                                Projects shared expert intermediate representation back to hidden_size.
                                Output is added to the regular expert outputs.
    :param W_expert_gate_up_scale: Optional quantization scale factors for regular expert gate/up weights.
                                  Shape: ``(num_local_experts, 2, intermediate_size)`` when provided
                                  Used for dequantization of regular expert gate/up weights during computation.
    :param W_expert_down_scale: Optional quantization scale factors for regular expert down weights.
                               Shape: ``(num_local_experts, hidden_size)`` when provided
                               Used for dequantization of regular expert down weights during computation.
    :param eps: Epsilon value for RMSNorm computation to prevent division by zero.
               Typically set to 1e-6 or 1e-5. Used in the fused RMSNorm operation.
    :param top_k: Number of top regular experts to route each token to.
                 Common values are 1, 2, or 4. Higher values increase computation but may improve quality.
                 Must be less than or equal to the total number of regular experts.
                 Does not affect shared expert activation (always active).
    :param router_act_fn: Activation function applied to router logits for regular expert selection.
                         Supported values: ``'sigmoid'``, ``'softmax'``
                         Controls how routing probabilities are computed from router outputs.
                         Does not affect shared expert processing.
    :param router_pre_norm: Whether to apply RMSNorm to hidden states before router computation.
                           When True, uses ``gamma`` and ``eps`` for normalization.
                           When False, assumes input is already normalized.
                           Affects both regular and shared expert input processing.
    :param expert_affinities_scaling_mode: Mode for scaling regular expert affinity scores during routing.
                                          Supported values: ``'no_scale'``, ``'post_scale'``, ``'pre_scale'``
                                          Controls when and how scaling is applied to routing decisions.
                                          Does not affect shared expert processing.
    :param hidden_act_fn: Activation function applied within both regular and shared expert feed-forward networks.
                         Supported values: ``'silu'``, ``'gelu'``, ``'swish'``
                         Applied between gate/up projections and down projection in all experts.
    :param router_matmul_dtype: Data type for router matrix multiplication operations.
                               Supported values: ``torch.bfloat16``, ``torch.float16``, ``torch.float32``
                               Controls precision vs performance trade-off for routing computations.
                               Does not affect shared expert computations.

    :return: Tuple containing:
             - ``moe_output``: MOE layer output tensor with shape ``(batch_size * sequence_length, hidden_size)``
                              Contains the combined output of weighted regular expert outputs and shared expert outputs
             - ``router_logits``: Router logit scores with shape ``(batch_size * sequence_length, num_experts)``
                                  Raw affinity scores for regular experts before top-k selection, useful for load balancing analysis
    """
    LOGICAL_NC_CONFIG = 2  # Underlying NKI kernel runs with LNC=2

    # TODO: Add input validation for kernel parameters

    # Convert string parameters to enums
    router_act_enum = _get_router_act_fn_enum(router_act_fn)
    scaling_mode_enum = _get_scaling_mode_enum(expert_affinities_scaling_mode)
    hidden_act_enum = _get_hidden_act_fn_enum(hidden_act_fn)

    # Convert torch dtype to nl dtype for router_mm_dtype
    router_mm_dtype_nl = _torch_dtype_to_nl_dtype(router_matmul_dtype)

    # Call the kernel with shared expert support
    grid = (nc(LOGICAL_NC_CONFIG),)
    output, router_logits = moe_token_gen_all_experts_kernel[grid](
        inp=hidden_states,
        gamma=gamma,
        router_weights=W_router,
        expert_gate_up_weights=W_expert_gate_up,
        expert_down_weights=W_expert_down,
        rank_id=rank_id,
        shared_expert_gate_w=W_shared_expert_gate,
        shared_expert_up_w=W_shared_expert_up,
        shared_expert_down_w=W_shared_expert_down,
        expert_gate_up_weights_scale=W_expert_gate_up_scale,
        expert_down_weights_scale=W_expert_down_scale,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_enum,
        router_pre_norm=router_pre_norm,
        expert_affinities_scaling_mode=scaling_mode_enum,
        hidden_act_fn=hidden_act_enum,
        router_mm_dtype=router_mm_dtype_nl
    )

    return output, router_logits
