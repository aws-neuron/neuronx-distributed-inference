import copy
import unittest

import torch
import torch_xla.core.xla_model as xm
from typing import Any, Dict
from functools import partial
import torch_neuronx
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.modules.moe import ExpertMLPs

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from torch import nn
import torch.nn.functional as F
from torch_neuronx.utils import get_platform_target
from neuronx_distributed_inference.models.config import get_platform_lnc

torch.manual_seed(42)

class TestConfig:
    def __init__(
        self,
        torch_dtype,
        tp_degree: int,
        ep_degree: int,
        rpl_reduce_dtype,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        hidden_act: str = 'silu',
        norm_topk_prob: bool = True,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size = intermediate_size
        self.norm_topk_prob = norm_topk_prob

        self.torch_dtype = torch_dtype
        self.tp_degree = tp_degree
        self.ep_degree = ep_degree
        self.rpl_reduce_dtype = rpl_reduce_dtype

class CPUExpertMLP(nn.Module):
    def __init__(
            self,
            n_routed_experts: int,
            topk: int,
            hidden_size: int,
            moe_inter_dim: int,
            dtype = torch.bfloat16,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.experts = nn.ModuleList([Expert(hidden_size, moe_inter_dim, dtype=dtype) for i in range(self.n_routed_experts)])
        
    def forward(self, x: torch.Tensor, weights, indices, seq_len) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.hidden_size)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y.view(shape)

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=None, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=None, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=None, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    
class ExpertMLPWrapper(torch.nn.Module):
    def __init__(self, config, world_size):
        self.config=config
        self.tp_degree=config.tp_degree
        self.world_size=world_size
        super().__init__()
        self.expert_mlp = ExpertMLPs(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            normalize_top_k_affinities=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            glu_mlp=True,
            capacity_factor=None,
            dtype=config.torch_dtype,
            logical_nc_config = get_platform_lnc(),
            enable_spmd_rank=True,
        )
    
    def forward(self, hidden_states, expert_affinities, expert_indices , seq_len):

        output = self.expert_mlp(hidden_states, expert_affinities,expert_indices,seq_len)
        output = self._reduce_output(output)
        return output
    
    def _reduce_output(self, output: torch.Tensor) -> torch.Tensor:
        original_dtype = output.dtype
        output = output.to(torch.float32)
        if parallel_state.get_expert_model_parallel_size() > 1:
            output = reduce_from_tensor_model_parallel_region(
                output,
                process_group=parallel_state.get_world_group(),
            )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output,
                process_group=parallel_state.get_tensor_model_parallel_group(as_list=False),
            )
        output = output.to(original_dtype)
        return output
    
    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        fuse_experts_weights(model_state_dict, self.config.n_routed_experts)
        create_spmd_ranks(
            model_state_dict=model_state_dict,
            prefix=prefix,
            world_size=self.world_size,
            n_routed_experts=self.config.n_routed_experts,
            expert_model_parallel_size=parallel_state.get_expert_model_parallel_size(),
        )

def fuse_experts_weights(
    model_state_dict: Dict[str, Any],
    n_routed_experts: int,
) -> None:

    down_proj_weights_list = []
    gate_up_proj_weights_list = []

    for i in range(n_routed_experts):
        down_proj_weight = (
            model_state_dict[f"experts.{i}.down_proj.weight"].transpose(0, 1).contiguous()
        )
        down_proj_weights_list.append(down_proj_weight)
        del model_state_dict[f"experts.{i}.down_proj.weight"]

        up_proj_weight = model_state_dict[f"experts.{i}.up_proj.weight"]
        gate_weight = model_state_dict[f"experts.{i}.gate_proj.weight"]
        gate_up_proj_weights_list.append(
            torch.cat((gate_weight, up_proj_weight), dim=0).transpose(0, 1).contiguous()
        )

        del model_state_dict[f"experts.{i}.up_proj.weight"]
        del model_state_dict[f"experts.{i}.gate_proj.weight"]


    down_proj_weights = torch.stack(down_proj_weights_list)
    model_state_dict["expert_mlp.mlp_op.down_proj.weight"] = down_proj_weights

    gate_up_proj_weights = torch.stack(gate_up_proj_weights_list)
    model_state_dict["expert_mlp.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
    n_routed_experts: int,
    expert_model_parallel_size: int,
):
    model_state_dict["expert_mlp.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )

    if parallel_state.get_expert_model_parallel_size() > 1:
        expert_indices = []
        for rank in range(world_size):
            curr_expert_rank = parallel_state.get_expert_parallel_rank_from_global_rank(
                rank=rank, expert_parallel_group=parallel_state.get_expert_model_parallel_group()
            )
            curr_expert_indices = parallel_state.get_experts_for_expert_parallel_rank(
                curr_expert_rank,
                total_number_of_experts=n_routed_experts,
                expert_model_parallel_size=expert_model_parallel_size,
            )
            expert_indices.append(curr_expert_indices)

        model_state_dict["expert_mlp.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )


def compile_neuron_expert_mlp_model(sample_inputs, checkpoint, load_module, tp_degree=1, ep_degree=1, world_size=32, torch_dtype=torch.bfloat16, **inference_config):
        inference_config = TestConfig(
            torch_dtype= torch_dtype,
            tp_degree = tp_degree,
            ep_degree=ep_degree,
            rpl_reduce_dtype=torch.float32,
            **inference_config,
        )

        checkpoint = copy.copy(checkpoint)

        builder = ModelBuilder(
            router=None,
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            checkpoint_loader=lambda: checkpoint,
        )

        builder.add(
            key="main",
            model_instance=BaseModelInstance(
                module_cls=partial(load_module, inference_config, world_size),
                input_output_aliases={},
            ),
            example_inputs=[sample_inputs],
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
    compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"
    return compiler_args


def _load_module_expert_mlp(config, world_size: int):
    return ExpertMLPWrapper(config=config, world_size=world_size).eval()


def _load_sample_inputs(seq_len, hidden_size, num_expert, num_experts_per_tok, dtype):
    sample_hidden_state = torch.randn(seq_len, hidden_size, dtype=dtype)
    sample_weights = torch.randn(seq_len, num_expert, dtype=dtype)
    sample_indices = torch.randint(0, num_expert, (seq_len, num_experts_per_tok), dtype=torch.int64)
    nxdi_sample_inputs = (sample_hidden_state.squeeze(0), sample_weights, sample_indices, torch.tensor(seq_len))
    hf_sample_inputs = (sample_hidden_state.squeeze(0), sample_weights, sample_indices, torch.tensor(seq_len))
    
    return hf_sample_inputs, nxdi_sample_inputs

class TestExpertMLP(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.seq_len = 1024
        self.hidden_size = 7168
        self.intermediate_size = 2048
        self.n_routed_experts = 64
        self.num_experts_per_tok = 4
        
    def _get_common_model_params(self):
        return {
            "hidden_size": self.hidden_size,
            "hidden_act": 'silu',
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "intermediate_size": self.intermediate_size,
            "norm_topk_prob": True,
        }

    def _create_models(self, tp_degree_1, ep_degree_1, tp_degree_2, ep_degree_2, dtype, world_size):
        hf_sample_inputs, nxdi_sample_inputs = _load_sample_inputs(
            self.seq_len, 
            self.hidden_size, 
            self.n_routed_experts, 
            self.num_experts_per_tok, 
            dtype
        )

        model = CPUExpertMLP(
            n_routed_experts=self.n_routed_experts,
            topk=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_inter_dim=self.intermediate_size,
            dtype=dtype,
        ).eval()

        hf_checkpoint = model.state_dict()
        neuron_checkpoint = copy.deepcopy(hf_checkpoint)
        
        common_params = self._get_common_model_params()
        # Create first model configuration
        neuron_model_TP = compile_neuron_expert_mlp_model(
            nxdi_sample_inputs,
            neuron_checkpoint,
            _load_module_expert_mlp,
            tp_degree=tp_degree_1,
            ep_degree=ep_degree_1,
            world_size=world_size,
            torch_dtype=dtype,
            **common_params
        )

        # Create second model configuration
        neuron_model_EP = compile_neuron_expert_mlp_model(
            nxdi_sample_inputs,
            neuron_checkpoint,
            _load_module_expert_mlp,
            tp_degree=tp_degree_2,
            ep_degree=ep_degree_2,
            world_size=world_size,
            torch_dtype=dtype,
            **common_params
        )
        
        return neuron_model_TP, neuron_model_EP, nxdi_sample_inputs

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP1EP32_bf16_1(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=1, 
            ep_degree_2=32,
            world_size=32,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP1EP32_bf16_2(self):
        self.n_routed_experts = 256
        self.intermediate_size = 2048
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=1, 
            ep_degree_2=32,
            world_size=32,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP1EP32_bf16_3(self):
        self.n_routed_experts = 128
        self.intermediate_size = 4096
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=1, 
            ep_degree_2=32,
            world_size=32,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP16EP2_bf16_1(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=16, 
            ep_degree_2=2,
            world_size=32,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP16EP2_bf16_2(self):
        self.n_routed_experts = 8
        self.intermediate_size = 2048
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=16, 
            ep_degree_2=2,
            world_size=32,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn1', "Test only runs on trn1 platform")
    def test_expert_mlp_TP32EP1_against_TP1EP32_fp32(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=32, 
            ep_degree_1=1,
            tp_degree_2=1, 
            ep_degree_2=32,
            world_size=32,
            dtype=torch.float32
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP, rtol = 1e-2)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP32EP2_bf16_1(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=32, 
            ep_degree_2=2,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP32EP2_bf16_2(self):
        self.n_routed_experts = 256
        self.intermediate_size = 2048
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=32, 
            ep_degree_2=2,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP32EP2_bf16_3(self):
        self.n_routed_experts = 128
        self.intermediate_size = 4096
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=32, 
            ep_degree_2=2,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP16EP4_bf16_1(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=16, 
            ep_degree_2=4,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)
 
    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP16EP4_bf16_2(self):
        self.n_routed_experts = 8
        self.intermediate_size = 4096
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=16, 
            ep_degree_2=4,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP4EP16_bf16(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=4, 
            ep_degree_2=16,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP2EP32_bf16(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=2, 
            ep_degree_2=32,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)

    @unittest.skipIf(get_platform_target() != 'trn2', "Test only runs on trn2 platform")
    def test_expert_mlp_TP64EP1_against_TP1EP64_bf16(self):
        neuron_model_TP, neuron_model_EP, nxdi_sample_inputs = self._create_models(
            tp_degree_1=64, 
            ep_degree_1=1,
            tp_degree_2=1, 
            ep_degree_2=64,
            world_size=64,
            dtype=torch.bfloat16
        )

        neuron_output_TP = neuron_model_TP(*nxdi_sample_inputs)
        neuron_output_EP = neuron_model_EP(*nxdi_sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output_TP, neuron_output_EP)