import os
import time
import pytest
import tempfile
import torch
import torch_xla
import copy
import logging
from typing import List, Optional, Dict, Tuple
from argparse import Namespace

from transformers import GenerationConfig
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration,
    Llama4Config,
)

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig, TensorCaptureConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4 import Llama4InferenceConfig, Llama4NeuronConfig, NeuronLlama4ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.constants import TEST_PROMPT
from neuronx_distributed_inference.models.config import to_dict
from neuronx_distributed_inference.utils.tensor_capture_utils import list_capturable_modules_in_application
from torch_neuronx.testing import neuron_allclose

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
NUM_TOKENS_TO_CHECK = 16
NUM_CHUNKS = 5
BATCH_SIZE = 1
TEXT_TP_DEGREE = 64
VISION_TP_DEGERE = 16
WORLD_SIZE = 64
SEQ_LENGTH = 8192

def setup_debug_env():
    """Set up debug environment - inline version"""
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    # for trn2
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    torch_xla._XLAC._set_ir_debug(True)
    set_random_seed(0)

def rand_interval(a, b, *size):
    """Generate random tensor in interval - inline version"""
    return (b - a) * torch.rand(*size) + a

def get_llama4_config(dtype=torch.float32,
                      model_path=None):
    """Get Llama4 config - inline version"""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/llama4/config_16E_4layer.json")

    router_config = {"dtype": torch.float32, "act_fn": "sigmoid"}

    text_neuron_config = Llama4NeuronConfig(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=TEXT_TP_DEGREE,
        cp_degree=1,
        on_device_sampling_config=OnDeviceSamplingConfig(dynamic=False, top_k=1),
        world_size=WORLD_SIZE,
        capacity_factor=None,
        fused_qkv=False,
        attention_dtype=dtype,
        rpl_reduce_dtype=torch.float32,
        early_expert_affinity_modulation=True,
        disable_normalize_top_k_affinities=True,
        cast_type="as-declared",
        router_config=router_config,
        logical_neuron_cores=2,
        output_logits=True,
    )

    # Vision kernels with FP32 are known to not pass accuracy threshold. Turning off kernels in FP32.
    if dtype == torch.float32:
        use_vision_kernel = False
    else:
        use_vision_kernel = True
    vision_neuron_config = Llama4NeuronConfig(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LENGTH,
        torch_dtype=dtype,
        skip_sharding=False,
        save_sharded_checkpoint=False,
        tp_degree=VISION_TP_DEGERE,
        cp_degree=1,
        on_device_sampling_config=OnDeviceSamplingConfig(dynamic=False, top_k=1),
        dp_degree=WORLD_SIZE//VISION_TP_DEGERE,
        world_size=WORLD_SIZE,
        fused_qkv=True,
        qkv_kernel_enabled=use_vision_kernel,
        attn_kernel_enabled=use_vision_kernel,
        mlp_kernel_enabled=use_vision_kernel,
        enable_bucketing=use_vision_kernel,
        buckets=[8, 16, 88],
        logical_neuron_cores=2,
    )

    config = Llama4InferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    return config

def get_inputs(config, dtype):
    """Create test inputs for Llama4 model with vision capabilities"""
    # inputs
    text_token_len = 16
    num_vision_token_per_chunk = (config.vision_config.image_size // config.vision_config.patch_size) ** 2 * ((config.vision_config.pixel_shuffle_ratio) ** 2)
    vision_token_len = NUM_CHUNKS * int(num_vision_token_per_chunk)
    total_input_len = text_token_len + vision_token_len
    
    # construct text input tokens
    text_input_ids = torch.rand((config.neuron_config.batch_size, text_token_len)) * config.text_config.vocab_size
    
    # construct vision input tokens
    vision_input_ids = torch.full([config.neuron_config.batch_size, vision_token_len], fill_value=config.image_token_index)
    
    # assume vision tokens are before text tokens
    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1)
    input_ids = input_ids.to(dtype=torch.int32)
    attention_mask = torch.ones((config.neuron_config.batch_size, total_input_len), dtype=torch.int32)
    
    # vision inputs
    pixel_values = torch.nn.Parameter(
        rand_interval(
            -1,
            1,
            (
                NUM_CHUNKS,
                config.vision_config.num_channels,
                config.vision_config.image_size,
                config.vision_config.image_size,
            ),
        )
    ).to(dtype)
    
    vision_mask = (input_ids == config.image_token_index).unsqueeze(-1)
    vision_mask = vision_mask.to(torch.bool)
    
    return Namespace(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, vision_mask=vision_mask)

def save_checkpoint(config_path):
    """Save a model checkpoint with random weights for testing"""
    hf_config = Llama4Config.from_pretrained(config_path, torch_dtype=torch.float16)
    logger.info(f"HF config {to_dict(hf_config)}")
    hf_model = Llama4ForConditionalGeneration._from_config(hf_config)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir

def run_model(
    model_path: str, 
    config: Llama4InferenceConfig, 
    inputs: Namespace,
    tensor_capture_config: Optional[TensorCaptureConfig] = None,
    tensor_capture_dir: Optional[str] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Run a Llama4 model with or without tensor capture enabled
    
    Args:
        model_path: Path to the model
        config: Llama4InferenceConfig object
        inputs: Input tensors
        tensor_capture_config: Optional TensorCaptureConfig for enabling tensor capture
        tensor_capture_dir: Directory to save captured tensors
        
    Returns:
        Tuple containing model outputs and metadata
    """
    # Apply tensor capture config if provided - only to text_config for Llama4
    if tensor_capture_config:
        # Create a deep copy to avoid modifying the original config
        config = copy.deepcopy(config)
        config.text_config.neuron_config.tensor_capture_config = tensor_capture_config
    
    # Initialize model
    model = NeuronLlama4ForCausalLM(model_path=model_path, config=config)
    
    # Compile and load model
    compiled_model_path = os.path.join(model_path, "compiled_checkpoint")
    if tensor_capture_config:
        compiled_model_path += "_tensor_capture"
    
    print(f"Compiling model to {compiled_model_path}")
    model.compile(compiled_model_path)
    print(f"Loading model from {compiled_model_path}")
    model.load(compiled_model_path)
    
    # Create tensor capture hook if needed
    tensor_capture_hook = None
    if tensor_capture_config and tensor_capture_dir:
        tensor_capture_hook = get_tensor_capture_hook(
            capture_indices=[1, 5, 10],
            tensor_capture_save_dir=tensor_capture_dir,
        )
    
    # Run inference
    generation_model = HuggingFaceGenerationAdapter(model)
    generation_config = GenerationConfig(
        do_sample=False, 
        pad_token_id=config.text_config.pad_token_id, 
        max_new_tokens=NUM_TOKENS_TO_CHECK,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    start_time = time.time()
    outputs = generation_model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
        vision_mask=inputs.vision_mask,
        generation_config=generation_config,
        tensor_capture_hook=tensor_capture_hook
    )
    execution_time = time.time() - start_time
    
    metadata = {
        "execution_time": execution_time,
        "config": config,
        "model": model
    }
    
    return outputs, metadata

def assert_close_with_baseline_logits(baseline_logits, tensor_capture_logits):
    for i in range(len(baseline_logits)):
        all_close_summary = neuron_allclose(baseline_logits[i], tensor_capture_logits[i], rtol=5e-3, atol=1e-4)
        assert all_close_summary.allclose

def validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir):
    """
    Validate that tensor capture worked correctly and didn't affect model outputs
    
    Args:
        baseline_outputs: Outputs from model without tensor capture
        tensor_capture_outputs: Outputs from model with tensor capture
        tensor_capture_dir: Directory where captured tensors were saved
    """
    # Verify outputs are consistent across all configurations
    print("Verifying output consistency:")
    outputs_match = torch.allclose(tensor_capture_outputs.sequences, baseline_outputs.sequences)
    assert outputs_match, "Outputs do not match baseline"
    print("✓ Outputs match baseline")

    # Verify output logits are consistent across all configurations
    assert_close_with_baseline_logits(baseline_outputs.scores, tensor_capture_outputs.scores)
    print("✓ Outputs logits match baseline")

    # Verify that tensor files were created
    tensor_files = [f for f in os.listdir(tensor_capture_dir) 
                    if f.startswith("captured_tensors_")]

    print("✓ Found {len(tensor_files)} tensor capture files")
    
    # Assert that at least one tensor file was created
    assert len(tensor_files) > 0, "No tensor capture files were created"

@pytest.mark.tp64
@pytest.mark.xfail(reason="Accuracy comparison fails after transformers v4.56 upgrade, pending investigation")
def test_llama4_tensor_capture_modules_only():
    """Test tensor capture with only modules_to_capture specified for Llama4"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama4/config_16E_4layer.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create config
    config = get_llama4_config(dtype=torch.float16, model_path=model_path)
    
    # Create inputs
    inputs = get_inputs(config, dtype=torch.float16)
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        config,
        inputs
    )
    
    # Run with tensor capture - modules only
    # Use modules from the capturable_modules list if available, otherwise use default
    modules_to_capture = ['layers.0', 'layers.1.feed_forward.moe.router', 'layers.3']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        config,
        inputs,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )
    
    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())
    
    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)
    
    # Clean up
    model_tempdir.cleanup()

@pytest.mark.tp64
def test_llama4_tensor_capture_with_inputs():
    """Test tensor capture with modules_to_capture and capture_inputs enabled for Llama4"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama4/config_16E_4layer.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create config
    config = get_llama4_config(dtype=torch.float16, model_path=model_path)
    
    # Create inputs
    inputs = get_inputs(config, dtype=torch.float16)
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        config,
        inputs
    )
    
    # Run with tensor capture - modules and inputs
    # Use modules from the capturable_modules list if available, otherwise use default
    modules_to_capture = ['layers.0', 'layers.1.feed_forward.moe.router', 'layers.3']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture,
        capture_inputs=True
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        config,
        inputs,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )
    
    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())
    
    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)
    
    # Clean up
    model_tempdir.cleanup()


@pytest.mark.tp64
def test_llama4_tensor_capture_all_options():
    """Test tensor capture with all options enabled for Llama4"""
    # Load model config and save with random weights
    config_path = os.path.dirname(os.path.abspath(__file__)) + "/../models/llama4/config_16E_4layer.json"
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    
    # Create config
    config = get_llama4_config(dtype=torch.float16, model_path=model_path)
    
    # Create inputs
    inputs = get_inputs(config, dtype=torch.float16)
    
    # Create tensor capture directory
    tensor_capture_dir = tempfile.TemporaryDirectory().name
    os.makedirs(tensor_capture_dir, exist_ok=True)
    
    # Run baseline without tensor capture
    baseline_outputs, baseline_metadata = run_model(
        model_path, 
        config,
        inputs
    )
    
    # Run with tensor capture - all options
    # Use modules from the capturable_modules list if available, otherwise use default
    modules_to_capture = ['layers.0', 'layers.1.feed_forward.moe.router', 'layers.3']
    tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture,
        capture_inputs=True,
        max_intermediate_tensors=10
    )
    
    tensor_capture_outputs, tc_metadata = run_model(
        model_path, 
        config,
        inputs,
        tensor_capture_config=tensor_capture_config,
        tensor_capture_dir=tensor_capture_dir
    )
    
    print(tc_metadata['config'].to_json_string())
    print(baseline_metadata['config'].to_json_string())

    # Validate tensor capture
    validate_tensor_capture(baseline_outputs, tensor_capture_outputs, tensor_capture_dir)
    
    # Clean up
    model_tempdir.cleanup()

if __name__ == "__main__":
    setup_debug_env()
    test_llama4_tensor_capture_modules_only()
    test_llama4_tensor_capture_with_inputs()
    test_llama4_tensor_capture_all_options()
