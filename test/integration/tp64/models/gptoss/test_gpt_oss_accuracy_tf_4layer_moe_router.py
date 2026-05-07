import copy
import os
import json
import shutil
import hashlib
import subprocess
import pytest
import shlex
import time
import tempfile
from contextlib import contextmanager
from typing import Iterable, List, Dict, Optional
from pathlib import Path
import re

import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
try:
    from transformers import GptOssForCausalLM
except ImportError:
    GptOssForCausalLM = None

from neuronx_distributed_inference.models.config import (
    MoENeuronConfig,
    TensorCaptureConfig,
    OnDeviceSamplingConfig,
    TensorReplacementConfig,
)
from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (
    GptOssInferenceConfig,
    NeuronGptOssForCausalLM,
    GptOssNeuronConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.tensor_capture_utils import get_tensor_capture_hook
from neuronx_distributed.utils.tensor_capture.model_modification import (
    modify_hf_eager_model_for_tensor_capture,
)
from torch_neuronx.testing.validation import DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

# -------------------------
# Constants / Defaults
# -------------------------

CAPTURE_DIR = "~/tensor_capture_gpt_oss"

# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS16 baseline for GPT-OSS
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_baseline_tp64.json"), "r") as f:
    baseline_json = json.load(f)
BASELINE_NEURON_CONFIG = GptOssNeuronConfig(**baseline_json)

DEFAULT_PROMPTS = [
    "A pencil cost $0.50, and an eraser cost $0.25. If you bought 6 pencils and 8 erasers and paid $10, how much change would you get?"
]

# Map CPU module names → Neuron canonical names for GPT-OSS
MODULE_NAME_EQUIV = {
    "mlp.router": "feed_forward.moe.router",
}

# -------------------------
# Core Utilities
# -------------------------

def dir_missing_or_empty(path: str) -> bool:
    return (not os.path.exists(path)) or (not os.listdir(path))

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def ensure_empty_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# -------------------------
# Tensor Capture Functions
# -------------------------

def create_dense_router_hook(save_dir: str, step_counter: dict, is_generation: dict, module_id: str):
    """
    Create hook to capture dense router logits before top-k operation.
    
    For CPU models, the mlp.router module output is after top-k selection, but we need
    the dense logits before top-k to match Neuron's tensor replacement system.
    This hook manually recomputes the dense router logits from the input.
    
    Args:
        save_dir: Directory to save captured tensors
        step_counter: Dictionary tracking current generation step
        is_generation: Dictionary tracking generation phase state
        module_id: Unique identifier for the module being hooked
        
    Returns:
        Hook function that captures dense router logits and saves them to disk
    """
    def hook_fn(module, input, output):
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            hidden_states = input[0].reshape(-1, module.hidden_dim)
            router_logits = torch.nn.functional.linear(hidden_states, module.weight, module.bias)
            
            # Determine if this is CTE (step 1) or TKG (steps 2+)
            if not is_generation['started']:
                step = 1
                model_phase = "cte"

                if module_id not in is_generation['cte_seen']:
                    is_generation['cte_seen'].add(module_id)

                if len(is_generation['cte_seen']) >= is_generation['total_modules']:
                    is_generation['started'] = True
            else:
                step = step_counter.get('step', 2)
                model_phase = "tkg"
                if module_id not in step_counter.get('modules_seen_this_step', set()):
                    step_counter.setdefault('modules_seen_this_step', set()).add(module_id)
                    if len(step_counter['modules_seen_this_step']) >= is_generation['total_modules']:
                        step_counter['step'] = step + 1
                        step_counter['modules_seen_this_step'] = set()
            
            module_name = getattr(module, '_capture_name', 'unknown')
            filename = f"captured_tensors_{model_phase}_step_{step}_module_{module_name}_output_0.pt"
            filepath = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save(router_logits.detach().cpu(), filepath)
            print(f"Saved dense router logits to {filepath}")
            
        return output
    
    return hook_fn

@contextmanager
def patched_cpu_model_for_dense_router_capture(
    base_hf_model,
    modules_to_capture: List[str],
    save_dir: str,
):
    """
    Set up custom tensor capture for CPU model router modules.
    
    CPU models have mlp.router outputs after top-k selection, but tensor replacement
    requires dense router logits before top-k. This function patches the model to
    capture the dense logits by recomputing them from module inputs.
    
    Args:
        base_hf_model: The CPU HuggingFace model to patch
        modules_to_capture: List of router module names to capture (e.g., ['layers.0.mlp.router'])
        save_dir: Directory where captured tensors will be saved
        
    Returns:
        Context manager yielding the patched model with tensor capture enabled
    """
    hooks = []
    step_counter = {'step': 2}  # TKG steps start from 2
    is_generation = {
        'started': False,  # Track if we've completed CTE phase
        'cte_seen': set(),  # Track which modules have seen CTE
        'total_modules': 0  # Total number of modules to track
    }
    
    try:
        # Find and hook router modules specifically
        router_modules = []
        for name, module in base_hf_model.named_modules():
            if any(target_module in name for target_module in modules_to_capture):
                if type(module).__name__ == 'GptOssTopKRouter':
                    router_modules.append((name, module))
        
        is_generation['total_modules'] = len(router_modules)
        
        for name, module in router_modules:
            # Keep original module name format with dots
            # model.layers.0.mlp.router -> layers.0.mlp.router  
            capture_name = name.replace('model.', '')
            module._capture_name = capture_name
            module_id = name
            hook = module.register_forward_hook(create_dense_router_hook(save_dir, step_counter, is_generation, module_id))
            hooks.append(hook)
            print(f"Registered dense router hook for: {name} -> {capture_name}")
        
        yield base_hf_model
    finally:
        for hook in hooks:
            hook.remove()

        for name, module in base_hf_model.named_modules():
            if hasattr(module, '_capture_name'):
                delattr(module, '_capture_name')

# -------------------------
# Test Function
# -------------------------

@pytest.mark.tp64
@pytest.mark.parametrize(
    "model_path, neuron_config, num_tokens_to_check, divergence_tolerance",
    [
        pytest.param(
            os.getenv("GPT_OSS_MODEL_PATH"),
            BASELINE_NEURON_CONFIG,
            15,
            DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
        ),  # BS16 baseline
    ],
)
def test_gpt_oss_accuracy_with_tf_moe_router(
    model_path, neuron_config, num_tokens_to_check, divergence_tolerance
):
    if not model_path or dir_missing_or_empty(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    num_of_layers = int(cfg["num_hidden_layers"])

    BASELINE_NEURON_CONFIG.batch_size = 1
    BASELINE_NEURON_CONFIG.ctx_batch_size = 1
    BASELINE_NEURON_CONFIG.output_logits = True
    BASELINE_NEURON_CONFIG.on_device_sampling_config = OnDeviceSamplingConfig()
    neuron_cfg = BASELINE_NEURON_CONFIG

    router_modules_to_capture = []
    for i in range(num_of_layers):
        router_modules_to_capture.append(f"layers.{i}.feed_forward.moe.router")

    modules_to_capture = []
    modules_to_capture.extend(router_modules_to_capture)
    
    print(f"DEBUG: Attempting to capture modules: {modules_to_capture}")

    # capture reference tensors
    num_tokens_to_check = 512
    cpu_dir, neu_dir = run_tensor_capture(
        model_path=model_path,
        neuron_config=neuron_cfg,
        num_tokens_to_check=num_tokens_to_check,
        prompts=DEFAULT_PROMPTS,
        modules_to_capture=modules_to_capture,
    )

    # need to +1 due to 1-indexed system in tensor capture
    tf_map = {i: router_modules_to_capture for i in range(1, num_tokens_to_check + 1)}
    # run tensor-replacement using those captures
    # Use base MODULE_NAME_EQUIV - the registry will handle phase-specific mapping
    run_accuracy_with_tensor_replacement(
        model_path=model_path,
        neuron_config=neuron_cfg,
        num_tokens_to_check=512,
        prompts=DEFAULT_PROMPTS,
        cpu_dir=cpu_dir,
        neuron_dir=neu_dir,
        tf_map=tf_map,
        module_equiv=MODULE_NAME_EQUIV,
    )

# -------------------------
# Model Building Functions
# -------------------------

def build_gpt_oss_neuron(model_path: str, neuron_config: GptOssNeuronConfig):
    """
    Compile once, load once, wrap with HF generation adapter.
    Returns (raw_neuron_model, generation_adapter, tokenizer).
    """
    config = GptOssInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    neuron_model = NeuronGptOssForCausalLM(model_path, config)
    compiled_path = os.path.join(model_path, "compiled_checkpoint_accuracy")
    neuron_model.compile(compiled_path)
    neuron_model.load(compiled_path)
    
    adapter_model = HuggingFaceGenerationAdapter(neuron_model)
    return neuron_model, adapter_model, tokenizer

def capture_for_prompt(
    prompt: str,
    model,
    tokenizer,
    neuron_config: GptOssNeuronConfig,
    generation_config: GenerationConfig,
    capture_dir: str,
    use_tensor_hook: bool = False,
    num_tokens_to_check: int = 5,
):
    ensure_empty_dir(capture_dir)
    prompts = [prompt] * neuron_config.batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=num_tokens_to_check,
        min_new_tokens=num_tokens_to_check,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        generation_config=generation_config,
    )
    if use_tensor_hook:
        kwargs["tensor_capture_hook"] = get_tensor_capture_hook(tensor_capture_save_dir=capture_dir)
    outputs = model.generate(**kwargs)
    return outputs


def run_tensor_capture(
    model_path: str,
    neuron_config: GptOssNeuronConfig,
    num_tokens_to_check: int,
    modules_to_capture: List[str],
    prompts: Iterable[str] = DEFAULT_PROMPTS,
):
    """
    - Compiles + loads Neuron once.
    - Loads HF eager model once
    - For each prompt, captures CPU and Neuron tensors into separate dirs.
    """

    base_capture_root = os.path.expanduser(CAPTURE_DIR)
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)
    # enable capture in Neuron config
    neuron_config.tensor_capture_config = TensorCaptureConfig(
        modules_to_capture=modules_to_capture, max_intermediate_tensors=3
    )

    # model build
    neuron_model, hf_adapter, tokenizer = build_gpt_oss_neuron(model_path, neuron_config)
    # also get a CPU eager model using the same architecture
    cpu_hf_model = neuron_model.load_hf_model(model_path)
    
    cpu_dir, neu_dir = None, None
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)

        cpu_dir = os.path.join(base_capture_root, "tensor_capture", "cpu", p_hash)
        neu_dir = os.path.join(base_capture_root, "tensor_capture", "neuron", p_hash)

        # CPU eager capture - use dense router capture for pre-topk logits
        cpu_modules_to_capture = [
            m.replace("feed_forward.moe.router", "mlp.router")
            if "feed_forward.moe.router" in m
            else m
            for m in modules_to_capture
        ]
        
        with patched_cpu_model_for_dense_router_capture(
            cpu_hf_model,
            modules_to_capture=cpu_modules_to_capture,
            save_dir=cpu_dir,
        ) as cpu_model_patched:
            capture_for_prompt(
                prompt,
                cpu_model_patched,
                tokenizer,
                neuron_config,
                gen_cfg,
                cpu_dir,
                use_tensor_hook=False,
                num_tokens_to_check=num_tokens_to_check,
            )
            
        # Clean up any _output_1.pt files from CPU capture
        # For GPT-OSS MoE routers: _output_0.pt contains dense router logits (needed for tensor replacement)
        # _output_1.pt contains metadata/auxiliary data (not needed and causes tensor shape mismatches)
        for file_path in Path(cpu_dir).glob("*_output_1.pt"):
            file_path.unlink()
            print(f"Removed unwanted metadata file: {file_path}")
            
        # Neuron capture (adapter path uses tensor_capture_hook)
        capture_for_prompt(
            prompt,
            hf_adapter,
            tokenizer,
            neuron_config,
            gen_cfg,
            neu_dir,
            use_tensor_hook=True,
            num_tokens_to_check=num_tokens_to_check,
        )

        # Clean per-prompt state if Neuron model stashed step counters
        if hasattr(neuron_model, "_tensor_capture_step"):
            delattr(neuron_model, "_tensor_capture_step")

    print("Tensor capture complete.")
    return cpu_dir, neu_dir

# -------------------------
# Accuracy Testing Workflow
# -------------------------

def run_accuracy_with_tensor_replacement(
    model_path: str,
    neuron_config: GptOssNeuronConfig,
    num_tokens_to_check: int,
    tf_map: Dict[int, List[str]],
    prompts: Iterable[str] = DEFAULT_PROMPTS,
    # Replacement sources + mapping
    cpu_dir: str = "",
    neuron_dir: str = "",
    module_equiv: Optional[Dict[str, str]] = None,
):
    """
    Test tensor replacement effectiveness through accuracy comparison and direct tensor verification.
    
    - First runs baseline accuracy test without tensor replacement
    - Then enables tensor replacement and runs accuracy test again
    - Compares the error metrics to see if tensor replacement improves accuracy
    - Finally verifies tensor replacement actually occurred by directly comparing
      CPU ground truth tensors with Neuron tensors after replacement
    """
    from neuronx_distributed_inference.utils.exceptions import LogitMatchingValidationError
    
    base_replace_root = os.path.expanduser(CAPTURE_DIR)
    gen_cfg = GenerationConfig(do_sample=False, pad_token_id=0)

    print("\n=== BASELINE ACCURACY TEST (NO TENSOR REPLACEMENT) ===")
    # First test: baseline without tensor replacement
    baseline_neuron_model, _, tokenizer = build_gpt_oss_neuron(model_path, neuron_config)
    
    baseline_results = {}
    for idx, prompt in enumerate(prompts):
        print(f"[BASELINE {idx+1}] Prompt: {prompt[:80]}...")
        try:
            results = check_accuracy_logits(
                baseline_neuron_model,
                generation_config=gen_cfg,
                num_tokens_to_check=num_tokens_to_check,
                prompt=prompt,
                tokenizer=tokenizer,
                divergence_difference_tol=DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
                generate_fn_divergence=True,
            )
            print(f"[BASELINE {idx+1}] PASSED - No divergence detected")
            baseline_results[idx] = {"status": "PASSED", "results": results}
        except LogitMatchingValidationError as e:
            print(f"[BASELINE {idx+1}] FAILED - {str(e)}")
            baseline_results[idx] = {"status": "FAILED", "results": e.results, "error_msg": str(e)}
    
    print("\n=== TENSOR REPLACEMENT ACCURACY TEST ===")
    # Second test: with tensor replacement
    # configure replacement
    neuron_config.tensor_replacement_config = TensorReplacementConfig(
        ref_dir=os.path.expanduser(cpu_dir),
        neuron_dir=os.path.expanduser(neuron_dir),
        tf_map=tf_map,
        module_map=module_equiv or MODULE_NAME_EQUIV,
        neuron_config=copy.deepcopy(neuron_config),
    )

    # model build (compile + load once)
    neuron_model, _, tokenizer = build_gpt_oss_neuron(model_path, neuron_config)

    replacement_results = {}
    for idx, prompt in enumerate(prompts):
        print(f"[TENSOR REPLACEMENT {idx+1}] Prompt: {prompt[:80]}...")
        p_hash = hash_prompt(prompt)
        neu_dir_out = os.path.join(base_replace_root, "tensor_replace", "neuron", p_hash)

        try:
            results = check_accuracy_logits(
                neuron_model,
                generation_config=gen_cfg,
                num_tokens_to_check=num_tokens_to_check,
                prompt=prompt,
                tokenizer=tokenizer,
                divergence_difference_tol=DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
                generate_fn_divergence=True,
                tensor_capture_hook=get_tensor_capture_hook(tensor_capture_save_dir=neu_dir_out),
            )
            print(f"[TENSOR REPLACEMENT {idx+1}] PASSED - Teacher forcing successful!")
            replacement_results[idx] = {"status": "PASSED", "results": results}
        except LogitMatchingValidationError as e:
            print(f"[TENSOR REPLACEMENT {idx+1}] FAILED - {str(e)}")
            replacement_results[idx] = {"status": "FAILED", "results": e.results, "error_msg": str(e)}

        if hasattr(neuron_model, "_tensor_capture_step"):
            delattr(neuron_model, "_tensor_capture_step")

    # Compare tensors using detailed comparison logic
    print("\n=== TENSOR COMPARISON (CPU vs Neuron Replace) ===")
    
    prompt_id = hash_prompt(list(prompts)[0])
    cpu_compare_dir = os.path.join(base_replace_root, "tensor_capture", "cpu", prompt_id)
    neuron_replace_compare_dir = os.path.join(base_replace_root, "tensor_replace", "neuron", prompt_id)
    
    success, message = verify_tensor_replacement_success(cpu_compare_dir, neuron_replace_compare_dir)
    print(f"Tensor comparison result: {message}")
    
    if success:
        print("\n=== OVERALL TEST RESULT: PASSED ===")
        print("Tensor replacement shows good alignment with CPU reference")
    else:
        print("\n=== OVERALL TEST RESULT: FAILED ===")
        print("Tensor replacement does not align well with CPU reference")
        assert False, f"Tensor comparison failed: {message}"

    return neu_dir_out

# -------------------------
# Tensor Comparison Utilities
# -------------------------

def parse_tensor_filename(filename):
    """Parse tensor filename to extract step, layer, and phase information."""
    pattern_cpu = r'captured_tensors_([^_]+)_step_(\d+)_module_(.+)_output_0\.pt'
    match = re.match(pattern_cpu, filename)
    
    if match:
        return {
            'phase': match.group(1),
            'step': int(match.group(2)),
            'module': match.group(3),
            'filename': filename,
            'type': 'cpu'
        }
    
    pattern_neuron = r'captured_tensors_([^_]+)_step_(\d+)_module_(.+)_output\.pt'
    match = re.match(pattern_neuron, filename)
    
    if match:
        return {
            'phase': match.group(1),
            'step': int(match.group(2)),
            'module': match.group(3),
            'filename': filename,
            'type': 'neuron'
        }
    
    return None

def verify_tensor_replacement_success(cpu_ground_truth_dir, neuron_replaced_tensors_dir):
    """
    Verify that Neuron tensors were successfully replaced with CPU ground truth values.
    
    Uses overlap detection based on shape[0] - compares the first min(cpu_rows, neuron_rows) rows.
    """
    def load_tensors(directory, file_type):
        tensors = {}
        if not os.path.exists(directory):
            return tensors
        
        for file_path in Path(directory).glob("*.pt"):
            try:
                parsed = parse_tensor_filename(file_path.name)
                if parsed is None:
                    continue
                    
                tensor = torch.load(file_path, map_location='cpu')
                module = parsed['module']
                
                # Convert CPU module names to match Neuron naming convention for comparison
                if file_type == 'cpu':
                    module = module.replace('mlp.router', 'feed_forward.moe.router')
                
                step = parsed['step']
                if step not in tensors:
                    tensors[step] = {}
                tensors[step][module] = {
                    'tensor': tensor,
                    'phase': parsed['phase'],
                    'filename': parsed['filename'],
                    'original_module': parsed['module']
                }
            except Exception:
                continue
        return tensors
    
    def compare_tensors_with_overlap(cpu_tensor, neuron_tensor):
        """Compare tensors using overlap detection"""
        cpu_tensor = cpu_tensor.float()
        neuron_tensor = neuron_tensor.float()
        
        if len(cpu_tensor.shape) != len(neuron_tensor.shape):
            return False
        
        overlap_rows = min(cpu_tensor.shape[0], neuron_tensor.shape[0])
        
        if cpu_tensor.shape[1:] != neuron_tensor.shape[1:]:
            return False
        
        cpu_slice = cpu_tensor[:overlap_rows]
        neuron_slice = neuron_tensor[:overlap_rows]
        
        return torch.allclose(cpu_slice, neuron_slice, atol=1e-4, rtol=1e-3)
    
    cpu_tensors = load_tensors(cpu_ground_truth_dir, 'cpu')
    neuron_tensors = load_tensors(neuron_replaced_tensors_dir, 'neuron')
    
    if not cpu_tensors or not neuron_tensors:
        return False, "Missing tensor files"
    
    all_cpu_steps = set(cpu_tensors.keys())
    all_neuron_steps = set(neuron_tensors.keys())
    common_steps = sorted(all_cpu_steps & all_neuron_steps)
    
    if not common_steps:
        return False, "No common steps found"
    
    total_comparisons = 0
    close_comparisons = 0
    
    for step in common_steps:
        cpu_modules = set(cpu_tensors[step].keys())
        neuron_modules = set(neuron_tensors[step].keys())
        common_modules = cpu_modules & neuron_modules
        
        for module in common_modules:
            cpu_info = cpu_tensors[step][module]
            neuron_info = neuron_tensors[step][module]
            
            if cpu_info['phase'] == neuron_info['phase']:
                tensor1 = cpu_info['tensor']
                tensor2 = neuron_info['tensor']
                
                total_comparisons += 1
                if compare_tensors_with_overlap(tensor1, tensor2):
                    close_comparisons += 1
    
    if total_comparisons == 0:
        return False, "No tensor comparisons performed"
    
    success_rate = close_comparisons / total_comparisons
    return success_rate == 1.0, f"Success rate: {success_rate:.2%} ({close_comparisons}/{total_comparisons})"

def extract_max_errors_from_results(results: dict) -> dict:
    """Extract comprehensive error statistics from logit validation results."""
    stats = {}
    
    if 'max_top_k_errors' in results:
        for k, v in results['max_top_k_errors'].items():
            key = f"top_k_{k}" if k != 'null' else "top_k_None"
            stats[f"{key}_max_error"] = v.get('error', 0.0)
    
    if 'average_over_tokens' in results:
        avg_stats = results['average_over_tokens']
        
        if 'null' in avg_stats:
            null_stats = avg_stats['null']
            stats['mean_abs_error'] = null_stats.get('mean_abs_error', 0.0)
            stats['mean_squared_error'] = null_stats.get('mean_squared_error', 0.0)
            stats['max_abs_error'] = null_stats.get('max_abs_error', 0.0)
            stats['max_squared_error'] = null_stats.get('max_squared_error', 0.0)
        
        for k in ['1000', '50', '5']:
            if k in avg_stats:
                k_stats = avg_stats[k]
                stats[f'top_k_{k}_max_abs_error'] = k_stats.get('max_abs_error', 0.0)
                stats[f'top_k_{k}_max_squared_error'] = k_stats.get('max_squared_error', 0.0)
    
    return stats

if __name__ == "__main__":
    gpt_oss_model_local_path = "~/models/gpt-oss/"
    
    test_gpt_oss_accuracy_with_tf_moe_router(
        gpt_oss_model_local_path,
        BASELINE_NEURON_CONFIG,
        512,
        DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
    )