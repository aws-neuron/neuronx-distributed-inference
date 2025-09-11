from typing import Optional

from torch import nn

from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig, MoEFusedTKGConfig
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe.moe_process_group import (
    init_tensor_expert_parallel_moe_process_groups,
    get_moe_tp_ep_group,
    get_moe_ep_group,
)
from neuronx_distributed_inference.models.config import InferenceConfig

# NOTE: MOE V2 only accepts InferenceConfig object. This requires the model config to have standardized naming (dbrx, mixtral, llama4)
#       for attributes that are being use in this method and RoutedExpertsMLPOpsConfig. This also requires modeling code to deepcopy
#       config and set the attribute `n_shared_experts`. This workaround will be removed once HF updates their config to include `n_shared_experts`.


def initialize_moe_module(
    config: InferenceConfig,
    rmsnorm: Optional[nn.Module] = None,
    init_tkg_module: bool = False,
    apply_act_fn_over_topk: bool = False,
    router_bias: bool = False,
    experts_bias: bool = False
):
    """
    Initializes and returns an MoE module corresponding to the given configuration.
    """
    enabled_hybrid_sharding = config.neuron_config.hybrid_sharding_config is not None
    moe_tkg_tensor_model_parallel_group, moe_tkg_expert_model_parallel_group, moe_cte_tensor_model_parallel_group, moe_cte_expert_model_parallel_group = \
        initialize_moe_process_group(config, enabled_hybrid_sharding)

    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        bias=router_bias,
        apply_act_fn_over_topk=apply_act_fn_over_topk,
        store_transposed_weights=init_tkg_module,  # register transposed weights for TKG kernel
    )
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(num_experts=config.num_local_experts,
                                                            hidden_size=config.hidden_size,
                                                            intermediate_size=config.intermediate_size,
                                                            top_k=config.num_experts_per_tok,
                                                            hidden_act=config.hidden_act,
                                                            bias=experts_bias,
                                                            glu_mlp=config.neuron_config.glu_mlp,
                                                            glu_type=config.neuron_config.glu_type,
                                                            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
                                                            hidden_act_bias=config.neuron_config.hidden_act_bias,
                                                            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
                                                            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
                                                            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping
                                                            ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        enabled_hybrid_sharding=enabled_hybrid_sharding,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tensor_model_parallel_group,
        cte_expert_model_parallel_group=moe_cte_expert_model_parallel_group,
        tkg_tensor_model_parallel_group=moe_tkg_tensor_model_parallel_group,
        tkg_expert_model_parallel_group=moe_tkg_expert_model_parallel_group,
    )
    if config.n_shared_experts:
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_shared_experts=config.n_shared_experts,
            hidden_act=config.hidden_act,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            fused_gate_up_projection=config.neuron_config.fused_shared_experts,
            sequence_parallel_enabled=config.neuron_config.shared_experts_sequence_parallel_enabled,
            transpose_weights=config.neuron_config.transpose_shared_experts_weights,
        )

    if init_tkg_module:
        tkg_config = MoEFusedTKGConfig(
            quantized=config.neuron_config.quantized,
            moe_fused_kernel_enabled=config.neuron_config.moe_fused_nki_kernel_enabled,
            router_topk_kernel_enabled=config.neuron_config.router_topk_nki_kernel_enabled,
            expert_mlp_kernel_enabled=config.neuron_config.expert_mlp_nki_kernel_enabled,
            shared_mlp_kernel_enabled=config.neuron_config.shared_mlp_nki_kernel_enabled,
        )
    else:
        tkg_config = None
    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=shared_experts if config.n_shared_experts else None,
        rmsnorm=rmsnorm,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        sequence_dimension=1,
        init_tkg_module=init_tkg_module,
        tkg_config=tkg_config,
    )

    # Set MoE module in eval mode
    moe.eval()
    return moe


def initialize_moe_process_group(config: InferenceConfig, enabled_hybrid_sharding: bool):
    if enabled_hybrid_sharding:
        moe_cte_tp_degree = config.neuron_config.hybrid_sharding_config.moe_cte_tp_degree
        moe_cte_ep_degree = config.neuron_config.hybrid_sharding_config.moe_cte_ep_degree
        moe_tkg_tp_degree = config.neuron_config.hybrid_sharding_config.moe_tkg_tp_degree
        moe_tkg_ep_degree = config.neuron_config.hybrid_sharding_config.moe_tkg_ep_degree
        init_tensor_expert_parallel_moe_process_groups(moe_tkg_tp_degree, moe_tkg_ep_degree, moe_cte_tp_degree, moe_cte_ep_degree)
        moe_tkg_tensor_model_parallel_group = get_moe_tp_ep_group(prefill=False)
        moe_tkg_expert_model_parallel_group = get_moe_ep_group(prefill=False)
        moe_cte_tensor_model_parallel_group = get_moe_tp_ep_group(prefill=True)
        moe_cte_expert_model_parallel_group = get_moe_ep_group(prefill=True)
    else:
        if config.neuron_config.moe_ep_degree > 1:
            moe_ep_degree = config.neuron_config.moe_ep_degree
            moe_tp_degree = config.neuron_config.moe_tp_degree
            init_tensor_expert_parallel_moe_process_groups(moe_tp_degree, moe_ep_degree, moe_tp_degree, moe_ep_degree)
            moe_tkg_tensor_model_parallel_group = get_moe_tp_ep_group(prefill=False)
            moe_tkg_expert_model_parallel_group = get_moe_ep_group(prefill=False)
            moe_cte_tensor_model_parallel_group = get_moe_tp_ep_group(prefill=True)
            moe_cte_expert_model_parallel_group = get_moe_ep_group(prefill=True)
        else:
            moe_tkg_tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            moe_tkg_expert_model_parallel_group = parallel_state.get_expert_model_parallel_group()
            moe_cte_tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            moe_cte_expert_model_parallel_group = parallel_state.get_expert_model_parallel_group()

    return moe_tkg_tensor_model_parallel_group, moe_tkg_expert_model_parallel_group, moe_cte_tensor_model_parallel_group, moe_cte_expert_model_parallel_group
