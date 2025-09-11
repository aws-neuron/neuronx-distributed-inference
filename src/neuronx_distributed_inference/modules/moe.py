from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK


def initialize_moe_module(
    config,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    hidden_act,
    normalize_top_k_affinities=True,
):
    """
    Initializes and returns an MoE module corresponding to the given configuration.
    """
    router = RouterTopK(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
    )
    expert_mlps = ExpertMLPs(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        capacity_factor=config.neuron_config.capacity_factor,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        glu_mlp=config.neuron_config.glu_mlp,
        normalize_top_k_affinities=normalize_top_k_affinities,
        is_prefill=config.neuron_config.is_prefill_stage
    )
    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
    )
    # Set MoE module in eval mode
    moe.eval()
    return moe
