import copy
import pytest
from typing import Any, Dict
import torch
from functools import partial
from types import SimpleNamespace
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch_neuronx.utils import get_platform_target
import torch_neuronx
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.config import MoENeuronConfig, InferenceConfig, get_platform_lnc

class MoEWrapper(torch.nn.Module):
    def __init__(self, config):
        self.config=config
        self.tp_degree=config.neuron_config.tp_degree
        self.world_size = config.neuron_config.tp_degree * config.neuron_config.ep_degree
        super().__init__()
        self.moe1 = initialize_moe_module(self.config)
        self.moe2 = initialize_moe_module(self.config)
        self.moe3 = initialize_moe_module(self.config)
        self.moe4 = initialize_moe_module(self.config)
    
    def forward(self, hidden_states):
        # MoE 1
        output = self.moe1(hidden_states)

        # MoE 2
        output = self.moe2(output[0])

        # MoE 3
        output = self.moe3(output[0])

        # MoE 4
        output = self.moe4(output[0])

        return output[0]

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        prefix = prefix.removesuffix("weight")
        if f"{prefix}gate.weight" in model_state_dict:
            model_state_dict[f"{prefix}moe1.router.linear_router.weight"] = model_state_dict[
                f"{prefix}gate.weight"
            ]
            model_state_dict[f"{prefix}moe2.router.linear_router.weight"] = model_state_dict[
                f"{prefix}gate.weight"
            ]
            model_state_dict[f"{prefix}moe3.router.linear_router.weight"] = model_state_dict[
                f"{prefix}gate.weight"
            ]
            model_state_dict[f"{prefix}moe4.router.linear_router.weight"] = model_state_dict[
                f"{prefix}gate.weight"
            ]
        fuse_experts_weights(model_state_dict, prefix, self.config.n_routed_experts)
        create_spmd_ranks(
            model_state_dict=model_state_dict,
            prefix=prefix,
            world_size=parallel_state.get_world_group().size(),
            n_routed_experts=self.moe1.expert_mlps.routed_experts_mlp_config.num_experts,
            expert_model_parallel_group=self.moe1.expert_mlps.moe_expert_model_parallel_group,
        )

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
    n_routed_experts: int,
    expert_model_parallel_group,
):
    # add weight for spmd rank
    model_state_dict[f"{prefix}spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    model_state_dict[f"{prefix}moe1.expert_mlps.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    model_state_dict[f"{prefix}moe2.expert_mlps.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    model_state_dict[f"{prefix}moe3.expert_mlps.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    model_state_dict[f"{prefix}moe4.expert_mlps.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )

    if expert_model_parallel_group.size() > 1:
        expert_indices = []
        for rank in range(world_size):
            curr_expert_rank = parallel_state.get_expert_parallel_rank_from_global_rank(
                rank=rank, expert_parallel_group=expert_model_parallel_group
            )
            curr_expert_indices = parallel_state.get_experts_for_expert_parallel_rank(
                curr_expert_rank,
                total_number_of_experts=n_routed_experts,
                expert_model_parallel_size=expert_model_parallel_group.size(),
            )
            expert_indices.append(curr_expert_indices)

        model_state_dict[f"{prefix}moe1.expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )
        model_state_dict[f"{prefix}moe2.expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )
        model_state_dict[f"{prefix}moe3.expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )
        model_state_dict[f"{prefix}moe4.expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )


def fuse_experts_weights(
    model_state_dict: Dict[str, Any],
    prefix: str,
    n_routed_experts: int,
) -> None:

    down_proj_weights_list = []
    gate_up_proj_weights_list = []

    for i in range(n_routed_experts):
        down_proj_weight = (
            model_state_dict[f"{prefix}experts.{i}.down_proj.weight"].transpose(0, 1).contiguous()
        )
        down_proj_weights_list.append(down_proj_weight)
        del model_state_dict[f"{prefix}experts.{i}.down_proj.weight"]

        up_proj_weight = model_state_dict[f"{prefix}experts.{i}.up_proj.weight"]
        gate_weight = model_state_dict[f"{prefix}experts.{i}.gate_proj.weight"]
        gate_up_proj_weights_list.append(
            torch.cat((gate_weight, up_proj_weight), dim=0).transpose(0, 1).contiguous()
        )

        del model_state_dict[f"{prefix}experts.{i}.up_proj.weight"]
        del model_state_dict[f"{prefix}experts.{i}.gate_proj.weight"]


    down_proj_weights = torch.stack(down_proj_weights_list)
    model_state_dict[f"{prefix}moe1.expert_mlps.mlp_op.down_proj.weight"] = down_proj_weights
    model_state_dict[f"{prefix}moe2.expert_mlps.mlp_op.down_proj.weight"] = down_proj_weights
    model_state_dict[f"{prefix}moe3.expert_mlps.mlp_op.down_proj.weight"] = down_proj_weights
    model_state_dict[f"{prefix}moe4.expert_mlps.mlp_op.down_proj.weight"] = down_proj_weights

    gate_up_proj_weights = torch.stack(gate_up_proj_weights_list)
    model_state_dict[f"{prefix}moe1.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights
    model_state_dict[f"{prefix}moe2.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights
    model_state_dict[f"{prefix}moe3.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights
    model_state_dict[f"{prefix}moe4.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights

def _generate_random_weights(hidden_size, intermediate_size, n_routed_experts, dtype):
    checkpoint = {
        "moe1.shared_experts.down_proj.weight": (torch.rand(hidden_size, intermediate_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe1.shared_experts.up_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe1.shared_experts.gate_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe2.shared_experts.down_proj.weight": (torch.rand(hidden_size, intermediate_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe2.shared_experts.up_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe2.shared_experts.gate_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe3.shared_experts.down_proj.weight": (torch.rand(hidden_size, intermediate_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe3.shared_experts.up_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe3.shared_experts.gate_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe4.shared_experts.down_proj.weight": (torch.rand(hidden_size, intermediate_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe4.shared_experts.up_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "moe4.shared_experts.gate_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "gate.weight": (torch.rand(n_routed_experts, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        "gate.e_score_correction_bias": (torch.rand(n_routed_experts, dtype=dtype) * 2 - 1) * 0.025,
    }

    for i in range(n_routed_experts):
        expert = {
            f"experts.{i}.down_proj.weight": (torch.rand(hidden_size, intermediate_size, dtype=dtype) * 2 - 1) * 0.025,
            f"experts.{i}.up_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
            f"experts.{i}.gate_proj.weight": (torch.rand(intermediate_size, hidden_size, dtype=dtype) * 2 - 1) * 0.025,
        }
        checkpoint.update(expert)

    return checkpoint

def get_inference_config(config, seq_len, tp_degree, ep_degree, dtype, use_shard_on_intermediate_dynamic_while, use_shard_on_block_dynamic_while):
        config = SimpleNamespace(**config)
        inference_config = {
            "hidden_size": config.hidden_size,
            "hidden_act": config.hidden_act,
            "num_local_experts": config.n_routed_experts,
            "n_routed_experts": config.n_routed_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "intermediate_size": config.intermediate_size,
            "n_shared_experts": config.n_shared_experts,
            "dtype": dtype,}

        neuron_config = MoENeuronConfig(
            torch_dtype=dtype,
            tp_degree=64 if get_platform_target() == 'trn2' else 32,
            ep_degree=1,
            seq_len=seq_len,
            normalize_top_k_affinities=config.norm_topk_prob,
            disable_normalize_top_k_affinities=not config.norm_topk_prob,
            router_config={"dtype": torch.float32, "act_fn":"softmax"},
            blockwise_matmul_config={"use_block_parallel": False,
                                     "block_sharding_strategy":"HI_LO",
                                     "skip_dma_token": False,
                                     "skip_dma_weight": False,
                                     "parallelize_token_to_block_mapping": True,
                                     "use_shard_on_block_dynamic_while": use_shard_on_block_dynamic_while,
                                     "use_shard_on_intermediate_dynamic_while": use_shard_on_intermediate_dynamic_while,
                                    },
            moe_tp_degree=tp_degree,
            moe_ep_degree=ep_degree,
        )

        inference_config = InferenceConfig(
            neuron_config=neuron_config,
            **inference_config,
        )
        return inference_config

def compile_neuron_moe_model(
        sample_inputs, 
        checkpoint, 
        load_module, 
        tp_degree=1, 
        ep_degree=1, 
        torch_dtype=torch.bfloat16, 
        base_config=None, 
        use_shard_on_intermediate_dynamic_while=False, 
        use_shard_on_block_dynamic_while=False):

    seq_len = sample_inputs.shape[1]
    inference_config = get_inference_config(base_config, seq_len, tp_degree, ep_degree, torch_dtype,\
                                             use_shard_on_intermediate_dynamic_while=use_shard_on_intermediate_dynamic_while, use_shard_on_block_dynamic_while=use_shard_on_block_dynamic_while)
    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config = get_platform_lnc(),
        debug = True,
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, inference_config),
            input_output_aliases={},
        ),
        example_inputs=[(sample_inputs,)],
        compiler_args=_add_compiler_args(),
    )

    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model

def _add_compiler_args():
    """
    Over-ride function from base class for better control over compiler flags
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    # add dma optimzation flag
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    return compiler_args


def _load_module_moe(config):
    return MoEWrapper(config=config).eval()

class TestMoE:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.batch_size = 1
        self.n_shared_experts = 0

    def _get_common_model_params(self, hidden_size, n_routed_experts, intermediate_size, num_experts_per_tok, early_expert_affinity_modulation):
        return {
            "hidden_size": hidden_size,
            "hidden_act": 'silu',
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "intermediate_size": intermediate_size,
            "n_shared_experts": self.n_shared_experts,
            "norm_topk_prob": True if num_experts_per_tok > 1 else False,
            "n_group": 8,
            "topk_group": 4,
            "routed_scaling_factor": 1.0,
            "early_expert_affinity_modulation": early_expert_affinity_modulation,
        }

    def _create_models(self, model_params, tp_degree_1, ep_degree_1, tp_degree_2, ep_degree_2, dtype, use_shard_on_intermediate_dynamic_while=False, use_shard_on_block_dynamic_while=False):
        sample_inputs = torch.rand(
            self.batch_size, 
            model_params["seq_len"], 
            model_params["hidden_size"], 
            dtype=dtype
        )
        
        hf_checkpoint = _generate_random_weights(
            hidden_size=model_params["hidden_size"],
            intermediate_size=model_params["intermediate_size"],
            n_routed_experts=model_params["n_routed_experts"],
            dtype=dtype
        )
        
        neuron_checkpoint = copy.deepcopy(hf_checkpoint)
        neuron_checkpoint["moe1.router.linear_router.weight"] = copy.deepcopy(hf_checkpoint["gate.weight"])
        neuron_checkpoint["moe2.router.linear_router.weight"] = copy.deepcopy(hf_checkpoint["gate.weight"])
        neuron_checkpoint["moe3.router.linear_router.weight"] = copy.deepcopy(hf_checkpoint["gate.weight"])
        neuron_checkpoint["moe4.router.linear_router.weight"] = copy.deepcopy(hf_checkpoint["gate.weight"])
        del neuron_checkpoint["gate.weight"]
        neuron_checkpoint_ep = copy.deepcopy(neuron_checkpoint)

        common_params = self._get_common_model_params(
            model_params["hidden_size"],
            model_params["n_routed_experts"],
            model_params["intermediate_size"],
            model_params["num_experts_per_tok"],
            model_params["early_expert_affinity_modulation"],
        )

        neuron_model_TP = compile_neuron_moe_model(
            sample_inputs,
            neuron_checkpoint,
            _load_module_moe,
            tp_degree=tp_degree_1,
            ep_degree=ep_degree_1,
            torch_dtype=dtype,
            base_config=common_params
        )

        neuron_model_EP = compile_neuron_moe_model(
            sample_inputs,
            neuron_checkpoint_ep,
            _load_module_moe,
            tp_degree=tp_degree_2,
            ep_degree=ep_degree_2,
            torch_dtype=dtype,
            use_shard_on_intermediate_dynamic_while=use_shard_on_intermediate_dynamic_while,
            use_shard_on_block_dynamic_while=use_shard_on_block_dynamic_while,
            base_config=common_params
        )

        return neuron_model_TP, neuron_model_EP, sample_inputs

    @pytest.mark.parametrize("model_params", [
        {"hidden_size": 5120, "intermediate_size": 8192, "n_routed_experts": 128, "seq_len": 5120, "num_experts_per_tok": 1, "early_expert_affinity_modulation": True},
        {"hidden_size": 7168, "intermediate_size": 2048, "n_routed_experts": 128, "seq_len": 5120, "num_experts_per_tok": 1, "early_expert_affinity_modulation": False},
        {"hidden_size": 2048, "intermediate_size": 1024, "n_routed_experts": 64, "seq_len": 5120, "num_experts_per_tok": 1, "early_expert_affinity_modulation": True},
    ])
    @pytest.mark.parametrize("tp_degree_1, ep_degree_1", [
        (64, 1),
    ])
    @pytest.mark.parametrize("tp_degree_2, ep_degree_2", [
        (64, 1),
    ])
    @pytest.mark.skipif(get_platform_target() != 'trn2', reason="Test only runs on trn2 platform")
    def test_moe_configurations_shard_on_block_trn2(self, model_params, tp_degree_1, ep_degree_1, tp_degree_2, ep_degree_2):
        if model_params["n_routed_experts"] % ep_degree_2 != 0:
            pytest.skip(f"n_routed_experts ({model_params['n_routed_experts']}) must be divisible by ep_degree ({ep_degree_2})")

        neuron_model_ground_truth, neuron_model_test_target, nxdi_sample_inputs = self._create_models(
            model_params=model_params,
            tp_degree_1=tp_degree_1,
            ep_degree_1=ep_degree_1,
            tp_degree_2=tp_degree_2,
            ep_degree_2=ep_degree_2,
            dtype=torch.bfloat16,
            use_shard_on_block_dynamic_while=True,
        )

        neuron_output_ground_truth = neuron_model_ground_truth(nxdi_sample_inputs)
        neuron_output_test_target = neuron_model_test_target(nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_ground_truth, neuron_output_test_target)