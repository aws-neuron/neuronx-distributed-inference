import unittest
import os 
import torch
from types import SimpleNamespace

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronxcc.nki._private_kernels.blockwise_mm import SkipMode

class config:
     def __init__(self,
                  hidden_size: int,
                  intermediate_size: int,
                  top_k: int,
                  num_experts: int,
                  hidden_act: str,
                  n_shared_experts: int,
                  neuron_config: MoENeuronConfig
                  ):
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        self.num_experts_per_tok=top_k
        self.num_local_experts=num_experts
        self.hidden_act=hidden_act
        self.n_shared_experts=n_shared_experts
        self.neuron_config=neuron_config

class TestInitializeMoEModule(unittest.TestCase):
    
    def setUp(self):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # if need to run distributed framework on CPU
        print("Initializing cpu env")
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "8080"
        os.environ["RANK"] = "0"
        os.environ["NXD_CPU_MODE"] = "1"
        torch.distributed.init_process_group(backend="gloo")
        parallel_state.initialize_model_parallel()

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
    def get_configs(self,**kwargs):
        configs = SimpleNamespace(**kwargs)
        neuron_config = MoENeuronConfig(batch_size=1,
                        seq_len=128,
                        torch_dtype=torch.bfloat16,
                        blockwise_matmul_config={"block_size" : configs.block_size,"use_block_parallel" : configs.use_block_parallel,"use_torch_block_wise" : configs.use_torch_block_wise, "skip_dma_token" : configs.skip_token, "skip_dma_weight": configs.skip_weight},
                        router_config={"act_fn": configs.router_act_fn,"dtype": configs.router_dtype},
                        early_expert_affinity_modulation=configs.early_expert_affinity_modulation,
                        disable_normalize_top_k_affinities=configs.disable_normalize_top_k_affinities,
                        return_expert_index=configs.return_expert_index,
                        shared_experts_sequence_parallel_enabled=configs.shared_experts_sequence_parallel_enabled,
                        logical_nc_config=1)
        return config(
            hidden_size=configs.hidden_size,
            intermediate_size=configs.intermediate_size,
            top_k=configs.top_k,
            num_experts=configs.num_experts,
            hidden_act=configs.hidden_act,
            n_shared_experts=configs.n_shared_experts,
            neuron_config=neuron_config)

    def test_init_moe_module(self):
        
        test_configs = [
            {
                "early_expert_affinity_modulation": True,
                "disable_normalize_top_k_affinities": True,
                "block_size": 64,
                "use_block_parallel": True,
                "use_torch_block_wise": True,
                "router_act_fn": "sigmoid",
                "router_dtype": torch.float32,
                "top_k":1,
                "n_shared_experts": 0,
                "return_expert_index": True,
                "shared_experts_sequence_parallel_enabled": False,
                "hidden_size": 5120,
                "intermediate_size": 8192,
                "num_experts": 1,
                "hidden_act": "silu",
                "skip_token": True,
                "skip_weight": True
            },
            {
                "early_expert_affinity_modulation": False,
                "disable_normalize_top_k_affinities": False,
                "block_size": 32,
                "use_block_parallel": False,
                "use_torch_block_wise": False,
                "router_act_fn": "softmax",
                "router_dtype": torch.float16,
                "top_k": 4,
                "n_shared_experts": 1,
                "return_expert_index": True,
                "shared_experts_sequence_parallel_enabled": True,
                "hidden_size": 128,
                "intermediate_size": 512,
                "num_experts": 16,
                "hidden_act": "silu",
                "skip_token": False,
                "skip_weight": True
            },
            {
                "early_expert_affinity_modulation": False,
                "disable_normalize_top_k_affinities": True,
                "block_size": 128,
                "use_block_parallel": True,
                "use_torch_block_wise": False,
                "router_act_fn": "softmax",
                "router_dtype": torch.bfloat16,
                "top_k":1,
                "n_shared_experts": 5,
                "return_expert_index": False,
                "shared_experts_sequence_parallel_enabled": True,
                "hidden_size": 1280,
                "intermediate_size": 128,
                "num_experts": 5,
                "hidden_act": "relu",
                "skip_token": True,
                "skip_weight": False
            }
        ]

        for test_config in test_configs:
            module_config = self.get_configs(**test_config)

            module = initialize_moe_module(module_config)

            # Test modules initialization
            assert module is not None, "Module initialization failed"
            assert isinstance(module, MoE), f"Expected module to be of type MoE, but got {type(module).__name__}"

            assert module.router is not None, "Expected router to be initialized"
            assert isinstance(module.router, RouterTopK), f"Expected router to be of type RouterTopK, but got {type(module).__name__}"

            assert module.expert_mlps is not None, "Expected expert_mlps to be initialized"
            assert isinstance(module.expert_mlps, ExpertMLPsV2), f"Expected routed experts to be of type ExpertMLPsV2, but got {type(module).__name__}"
            
            if module_config.n_shared_experts:
                assert module.shared_experts is not None, "Expected shared_experts to be initialized"
                assert isinstance(module.shared_experts, SharedExperts), f"Expected shared experts to be of type SharedExperts, but got {type(module).__name__}"
            else:
                assert module.shared_experts is None, "Expected shared_experts to be None when n_shared_experts is 0"

            # Test Router configs
            assert module.router.act_fn == module_config.neuron_config.router_config.act_fn, f"Expected router activation to be {module_config.neuron_config.router_config.act_fn} but got {module.router.act_fn}"
            assert module.router.dtype == module_config.neuron_config.router_config.dtype, f"Expected router dtype to be {module_config.neuron_config.router_config.dtype} but got {module.router.dtype}"
            
            # Test routed expert configs
            assert module.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation == module_config.neuron_config.early_expert_affinity_modulation, f"Expected early_expert_affinity_modulation to be {module_config.neuron_config.early_expert_affinity_modulation} but got {module.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation}"
            assert module.expert_mlps.routed_experts_mlp_config.normalize_top_k_affinities == module_config.neuron_config.normalize_top_k_affinities, f"Expected normalize_top_k_affinities to be {module_config.neuron_config.normalize_top_k_affinities} but got {module.expert_mlps.routed_experts_mlp_config.normalize_top_k_affinities}"
            assert module.expert_mlps.blockwise_matmul_config.block_size == module_config.neuron_config.blockwise_matmul_config.block_size, f"Expected block_size to be {module_config.neuron_config.blockwise_matmul_config.block_size} but got {module.expert_mlps.blockwise_matmul_config.block_size}"
            assert module.expert_mlps.blockwise_matmul_config.use_block_parallel == module_config.neuron_config.blockwise_matmul_config.use_block_parallel, f"Expected use_block_parallel to be {module_config.neuron_config.blockwise_matmul_config.use_block_parallel} but got {module.expert_mlps.blockwise_matmul_config.use_block_parallel}"
            skip_dma = SkipMode(module_config.neuron_config.blockwise_matmul_config.skip_dma_token, module_config.neuron_config.blockwise_matmul_config.skip_dma_weight)
            assert module.expert_mlps.blockwise_matmul_config.skip_dma == skip_dma, f"Expected use_block_parallel to be {module.expert_mlps.blockwise_matmul_config.skip_dma} but got {skip_dma}"

            # Test RoutedExpertsMLPOpsConfig
            assert module.expert_mlps.routed_experts_mlp_config.hidden_size == module_config.hidden_size, f"Expected hidden_size to be {module_config.hidden_size} but got {module.expert_mlps.routed_experts_mlp_config.hidden_size}"
            assert module.expert_mlps.routed_experts_mlp_config.intermediate_size == module_config.intermediate_size, f"Expected intermediate_size to be {module_config.intermediate_size} but got {module.expert_mlps.routed_experts_mlp_config.intermediate_size}"
            assert module.expert_mlps.routed_experts_mlp_config.top_k == module_config.num_experts_per_tok, f"Expected num_experts_per_tok to be {module_config.top_k} but got {module.expert_mlps.routed_experts_mlp_config.num_experts_per_tok}"
            assert module.expert_mlps.routed_experts_mlp_config.num_experts == module_config.num_local_experts, f"Expected num_local_experts to be {module_config.num_experts_per_tok} but got {module.expert_mlps.routed_experts_mlp_config.num_local_experts}"
            assert module.expert_mlps.routed_experts_mlp_config.hidden_act == module_config.hidden_act, f"Expected hidden_act to be {module_config.hidden_act} but got {module.expert_mlps.routed_experts_mlp_config.hidden_act}"

            # Test shared expert configs
            if module_config.n_shared_experts:
                assert module.shared_experts.num_shared_experts == module_config.n_shared_experts, f"Expected num_shared_experts to be {module_config.n_shared_experts} but got {module.shared_experts.num_shared_experts}"
                assert module.shared_experts.sequence_parallel_enabled == module_config.neuron_config.shared_experts_sequence_parallel_enabled, f"Expected num_shared_experts to be {module_config.neuron_config.shared_experts_sequence_parallel_enabled} but got {module.shared_experts.sequence_parallel_enabled}"

            assert module.return_expert_index == module_config.neuron_config.return_expert_index, f"Expected return_expert_index to be {module_config.neuron_config.return_expert_index} but got {module.return_expert_index}"

if __name__ == "__main__":
    unittest.main()