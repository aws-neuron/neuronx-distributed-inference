import os
import torch
import unittest
import argparse
import itertools
import torch_neuronx
from functools import partial
from types import SimpleNamespace

from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.modules.checkpoint import load_state_dict
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed_inference.models.config import MoENeuronConfig, InferenceConfig
from neuronx_distributed.quantization.quantization_config import get_default_expert_wise_per_channel_custom_qconfig_dict
from neuronx_distributed.modules.moe.moe_process_group import (
    init_tensor_expert_parallel_moe_process_groups,
    get_moe_tp_ep_group,
    get_moe_ep_group,
)

os.environ['NEURON_PLATFORM_TARGET_OVERRIDE'] = 'trn2'
os.environ['NEURON_LOGICAL_NC_CONFIG'] = '2'
os.environ['NEURON_RT_VIRTUAL_CORE_SIZE']='2' 
os.environ['NEURON_RT_NUM_CORES']='64'

os.environ["UNSAFE_FP8FNCAST"]="1"
os.environ["XLA_HANDLE_SPECIAL_SCALAR"]="1"

torch.manual_seed(0)
NUM_LAYERS = 1

def init_cpu_env():
    # destroy distributed process if already started
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

def destroy_cpu_env():
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os.environ["NXD_CPU_MODE"] = "0"

# Class to Initialize the ExpertMLPs model for the test
class ExpertMoEClass(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        # model = torch.nn.ModuleList([
        if config.neuron_config.moe_ep_degree > 1:
            init_tensor_expert_parallel_moe_process_groups(config.neuron_config.moe_tp_degree, config.neuron_config.moe_ep_degree, config.neuron_config.moe_tp_degree, config.neuron_config.moe_ep_degree)
            self.cte_tensor_model_parallel_group=get_moe_tp_ep_group(prefill = True)
            self.cte_expert_model_parallel_group=get_moe_ep_group(prefill = True)
            self.tkg_tensor_model_parallel_group=get_moe_tp_ep_group(prefill = False)
            self.tkg_expert_model_parallel_group=get_moe_ep_group(prefill = False)
        else:
            self.cte_tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group()
            self.cte_expert_model_parallel_group=parallel_state.get_expert_model_parallel_group()
            self.tkg_tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group()
            self.tkg_expert_model_parallel_group=parallel_state.get_expert_model_parallel_group()

        model = ExpertMLPsV2(
                routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
                    num_experts=config.num_experts,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    top_k=config.top_k,
                    hidden_act=config.hidden_act,
                    glu_mlp=config.neuron_config.glu_mlp,
                    early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
                    normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
                    enable_spmd_rank=config.neuron_config.moe_ep_degree > 1),
                blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
                dtype=config.neuron_config.torch_dtype,
                cte_tensor_model_parallel_group = self.cte_tensor_model_parallel_group,
                cte_expert_model_parallel_group = self.cte_expert_model_parallel_group,
                tkg_tensor_model_parallel_group = self.tkg_tensor_model_parallel_group,
                tkg_expert_model_parallel_group = self.tkg_expert_model_parallel_group,
            ).eval()

        if config.neuron_config.quantization_type == "expert_wise_per_channel_symmetric":
            q_config = get_default_expert_wise_per_channel_custom_qconfig_dict()
            self.module = convert(model,q_config=q_config,inplace=True)
        else:
            self.module = model

    def forward(self, hidden_states, expert_affinities, expert_index, seq_len):
        # for layer in self.module:
        hidden_states = self.module(hidden_states, expert_affinities, expert_index, seq_len)
        hidden_states = mappings.reduce_from_tensor_model_parallel_region(hidden_states, process_group=parallel_state.get_world_group())
        
        return hidden_states

def _load_module_expert_mlps(config):
    return ExpertMoEClass(config).eval()

def rand_interval(a, b, *size, dtype):
    return ((b - a) * torch.rand(*size) + a).to(dtype)

# Generate FP8 weights and scales
def quantize_fp8_per_channel(num_experts, dim1, dim2):
    tensor = torch.nn.Parameter(rand_interval(-0.03, 0.03,(num_experts, dim1, dim2),dtype=torch.bfloat16))
    fp8_max, fp8_min = 240.0, -240.0
    max_values = torch.amax(torch.abs(tensor), dim=(1,), keepdim=True)
    
    scales = max_values / fp8_max
    scales = torch.max(scales, torch.ones(scales.shape, device=scales.device) * 1e-05)
    quantized_tensor = tensor / scales
    quantized_tensor = torch.clamp(quantized_tensor, fp8_min, fp8_max)

    scale_shape = [1] * len(quantized_tensor.shape)
    scale_shape[0] = num_experts
    scale_shape[2] = quantized_tensor.shape[2]
    quantized_tensor = quantized_tensor.to(torch.float8_e4m3fn)
    return quantized_tensor, scales.to(torch.float32).view(scale_shape)

def _add_compiler_args(quant):
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1 --lnc=2"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    if quant: 
        compiler_args += " --internal-hlo2tensorizer-options=' --modular-flow-mac-threshold=10 --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'"
    else: 
        compiler_args += " --internal-hlo2tensorizer-options=' --modular-flow-mac-threshold=10  --verify-hlo=true'"
    return compiler_args

# Generate random weights for the model
def _generate_random_quant_weights(hidden_size, intermediate_size, n_routed_experts):
    # Generate weights once and reuse for all layers
    checkpoint = {}
    
    layer_prefix = "module."
    down_weight, down_scale = quantize_fp8_per_channel(n_routed_experts, intermediate_size, hidden_size)
    fuse_weight, fuse_scale = quantize_fp8_per_channel(n_routed_experts, hidden_size, intermediate_size * 2)
    checkpoint[f"{layer_prefix}mlp_op.down_proj.weight"] = down_weight
    checkpoint[f"{layer_prefix}mlp_op.gate_up_proj.weight"] = fuse_weight
    checkpoint[f"{layer_prefix}mlp_op.down_proj.scale"] = down_scale
    checkpoint[f"{layer_prefix}mlp_op.gate_up_proj.scale"] = fuse_scale
    
    return checkpoint

def dequantize_checkpoint_to_bf16(checkpoint):
    dequantized_checkpoint = {}
    
    for key, value in checkpoint.items():
        if key.endswith('.weight'):
            # Get the corresponding scale
            scale_key = key.replace('.weight', '.scale')
            if scale_key in checkpoint:
                # Dequantize the weight by multiplying with scale
                scale = checkpoint[scale_key]
                value = value.to(torch.float32)
                dequantized_weight = (value * scale).to(torch.bfloat16)
                dequantized_checkpoint[key] = dequantized_weight
        
    return dequantized_checkpoint

# Test class for the ExpertMLPs model
class TestRoutedExpertsFP8(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.dtype = torch.bfloat16
        self.tp_degree = 64
        self.checkpoint_path = None

    def set_test_args(self, path, goldens):
        """Set the path for loading checkpoints"""

        assert os.path.exists(path), f"Checkpoint path does not exist: {path}"
        assert goldens in ["cpu", "neuron"], f"goldens must be one of ['cpu', 'neuron'], got: {goldens}"

        self.checkpoint_path = path
        self.goldens = goldens

    def get_quant_config(self, on_cpu, quant, seq_len, **kwargs):
        configs = SimpleNamespace(**kwargs)
        inference_config = {
            "hidden_size": configs.hidden_size,
            "hidden_act": 'silu',
            "num_experts": configs.num_experts,
            "top_k": configs.top_k,
            "intermediate_size": configs.intermediate_size,
            "dtype": self.dtype,}
        
        neuron_config = MoENeuronConfig(
            torch_dtype=self.dtype,
            tp_degree=self.tp_degree,
            seq_len=seq_len,
            logical_nc_config = 2,
            blockwise_matmul_config=getattr(configs,"blockwise_matmul_config",{}),
            early_expert_affinity_modulation=configs.early_expert_affinity_modulation,
            disable_normalize_top_k_affinities=configs.disable_normalize_top_k_affinities,
            moe_tp_degree = 4,
            moe_ep_degree = 16,
        )
        if on_cpu:
            neuron_config.tp_degree = 1
            neuron_config.on_cpu = True
            neuron_config.moe_tp_degree = 1
            neuron_config.moe_ep_degree = 1

        if quant:
            neuron_config.quantized = True
            neuron_config.quantization_dtype = "f8e3m4",
            neuron_config.quantization_type = "expert_wise_per_channel_symmetric"

        inference_config = InferenceConfig(
            neuron_config=neuron_config,
            **inference_config,
        )
        return inference_config
    
    def _initialize_test_data(self, seq_len, config):
        expert_affinities = torch.rand(seq_len, config.num_experts, dtype=self.dtype)
        _, expert_index = torch.topk(expert_affinities, config.top_k)
        hidden_states = torch.rand(seq_len, config.hidden_size, dtype=self.dtype)
        return hidden_states, expert_affinities, expert_index

    def compile_neuron_expert_mlps_model(self, inference_config, checkpoint, load_module, seq_len):
        hidden_states, expert_affinities, expert_index = self._initialize_test_data(seq_len, inference_config)
        builder = ModelBuilder(
            router=None,
            tp_degree=self.tp_degree,
            checkpoint_loader=lambda: checkpoint,
            compiler_workdir="./test_compiler_workdir/",
            logical_nc_config=inference_config.neuron_config.logical_nc_config,
        )
        builder.add(
            key="main",
            model_instance=BaseModelInstance(
                module_cls=partial(load_module, inference_config),
                input_output_aliases={},
            ),
            example_inputs=[(hidden_states, expert_affinities, expert_index, torch.tensor(seq_len)),],
            compiler_args=_add_compiler_args(inference_config.neuron_config.quantized),
        )
        neuron_model = builder.trace(initialize_model_weights=True)
        return neuron_model

    def compile_cpu_expert_mlps_model(self, inference_config, checkpoint, load_module):
        module = load_module(inference_config)
        module.load_state_dict(checkpoint)
        return module

    def test_expert_mlp_fp8_quantization(self):
        test_configs = [
            {
                # Qwen3 configs with 16 experts
                "model_type": "Qwen3",
                "early_expert_affinity_modulation": False,
                "disable_normalize_top_k_affinities": False,
                "hidden_size": 2048,
                "intermediate_size": 768,
                "num_experts": 16,
                "top_k":8,
            },
        ]

        blockwise_configs = []
        block_sizes = [256]
        block_strategies = [ "PING_PONG"]
        skip_dma_configs = [[False, False]]
        
        # Generate all possible combinations for blockwise_matmul_config
        blockwise_configs = []
        for base_config in test_configs:
            param_combinations = itertools.product(
                block_sizes,
                block_strategies,
                skip_dma_configs
            )
            
            # for block_size, strategy, skip_token, skip_weight in param_combinations:
            for block_size, strategy, skip_dma in param_combinations:
                config = base_config.copy()
                config["blockwise_matmul_config"] = {
                    "block_size": block_size,
                    "use_block_parallel": False,
                    "block_sharding_strategy": strategy,
                    "skip_dma_token": skip_dma[0],
                    "skip_dma_weight": skip_dma[1],
                    "use_torch_block_wise": True,
                }
                blockwise_configs.append(config)

        seq_mapping = {
            1024: "blockwise"
        }

        results = []
        for seq_len in seq_mapping.keys():
            print(f"\nRunning Test for {seq_mapping[seq_len]} with config:", end="")

            if seq_mapping[seq_len] == "blockwise":
                test_configs = blockwise_configs

            for config in test_configs:
                config_str = str(config)
                print(f"\n {config}\n")

                hidden_states, expert_affinities, expert_index = self._initialize_test_data(seq_len, SimpleNamespace(**config))
                checkpoint  = None
                # Check if this is llama4 config and path is provided
                if config["model_type"] == "llama4" and self.checkpoint_path:
                    print(f"\nLoading checkpoint from given path: {self.checkpoint_path}")
                    checkpoint = load_state_dict(self.checkpoint_path)

                if checkpoint is None:
                    # For all other configs or if model path is not provided, use random weights
                    checkpoint = _generate_random_quant_weights(
                        hidden_size=config["hidden_size"], 
                        intermediate_size=config["intermediate_size"],
                        n_routed_experts=config["num_experts"]
                    )

                if self.goldens == "cpu":
                    print("\nRunning FP8 CPU Model.....")
                    init_cpu_env()
                    cpu_module = self.compile_cpu_expert_mlps_model(
                        self.get_quant_config(on_cpu=True, quant=True, seq_len=seq_len, **config), 
                        checkpoint, 
                        _load_module_expert_mlps
                    )
                    cpu_output = cpu_module(hidden_states, expert_affinities, expert_index, seq_len)
                    destroy_cpu_env()
                else:
                    dequant_checkpoint = dequantize_checkpoint_to_bf16(checkpoint)
                    print("\nRunning with dequant BF16 Model weights.....")
                    cpu_module = self.compile_neuron_expert_mlps_model(
                        self.get_quant_config(on_cpu=False, quant=False, seq_len=seq_len, **config), 
                        dequant_checkpoint, 
                        _load_module_expert_mlps,
                        seq_len
                    )
                    cpu_output = cpu_module(hidden_states, expert_affinities, expert_index, torch.tensor(seq_len))

                
                print("\nRunning FP8 Neuron Model.....")
                neuron_model_original = self.compile_neuron_expert_mlps_model(
                    self.get_quant_config(on_cpu=False, quant=True, seq_len=seq_len, **config), 
                    checkpoint, 
                    _load_module_expert_mlps, 
                    seq_len
                )
                neuron_output = neuron_model_original(hidden_states, expert_affinities, expert_index, torch.tensor(seq_len))
                print(neuron_output)

                try:
                    print("\nComparing outputs.....")
                    print(f"\nCPU Output: {cpu_output} \n \n Neuron Output: {neuron_output}\n")
                    torch_neuronx.testing.assert_close(cpu_output, neuron_output, atol=1e-2, rtol=1e-2)
                    results.append((config_str, "PASS"))
                    print("Test PASSED")
                except AssertionError as e:
                    results.append((config_str, f"FAIL: {str(e)}"))
                    print(f"Test FAILED: {str(e)}")
        
        # Print summary of results
        print("\n\n===== TEST RESULTS SUMMARY =====")
        passed = sum(1 for _, status in results if status == "PASS")
        failed = len(results) - passed
        print(f"TOTAL: {len(results)}, PASSED: {passed}, FAILED: {failed}")

        print("\n----- PASSED TESTS -----")
        for config, status in results:
            if status == "PASS":
                print(f"{config}: {status}")

        print("\n----- FAILED TESTS -----")
        for config, status in results:
            if status != "PASS":
                print(f"{config}: {status}")
        
        # If any test failed, make the test fail
        if failed > 0:
            self.fail(f"{passed} configurations passed, {failed} configurations failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--golden-type", type=str, help="Goldens for accuracy check")
    args = parser.parse_args()
    test = TestRoutedExpertsFP8()
    if args.model_path or args.golden_type:
        test.set_test_args(args.model_path, args.golden_type)
    test.test_expert_mlp_fp8_quantization()