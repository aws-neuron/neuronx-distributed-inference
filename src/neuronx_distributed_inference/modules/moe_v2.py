from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.shared_experts import SharedExperts

# NOTE: MOE V2 only accepts InferenceConfig object. This requires the model config to have standardized naming (dbrx, mixtral, llama4)
#       for attributes that are being use in this method and RoutedExpertsMLPOpsConfig. This also requires modeling code to deepcopy
#       config and set the attribute `n_shared_experts`. This workaround will be removed once HF updates their config to include `n_shared_experts`.


def initialize_moe_module(config):
    """
    Initializes and returns an MoE module corresponding to the given configuration.
    """

    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
    )
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(num_experts=config.num_local_experts,
                                                            hidden_size=config.hidden_size,
                                                            intermediate_size=config.intermediate_size,
                                                            top_k=config.num_experts_per_tok,
                                                            hidden_act=config.hidden_act,
                                                            glu_mlp=config.neuron_config.glu_mlp,
                                                            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
                                                            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        dtype=config.neuron_config.torch_dtype
    )
    if config.n_shared_experts:
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_shared_experts=config.n_shared_experts,
            hidden_act=config.hidden_act,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            fused_gate_up_projection=config.neuron_config.fused_shared_experts
        )

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=shared_experts if config.n_shared_experts else None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
    )
    # Set MoE module in eval mode
    moe.eval()
    return moe
