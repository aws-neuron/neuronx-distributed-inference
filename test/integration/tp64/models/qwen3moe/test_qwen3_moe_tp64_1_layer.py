from argparse import Namespace
import os
import json
import pytest
import tempfile
import torch
from concurrent.futures import ProcessPoolExecutor

from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.profiling import (
    get_neff_path_and_output_ntff_folder,
    run_profiler_on_neff,
)
from neuronx_distributed_inference.utils.constants import (
    CONTEXT_ENCODING_MODEL,
    TOKEN_GENERATION_MODEL,
)
from torch_neuronx.testing.validation import DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE


# Reading neuron_config test cases from jsons
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# BS16 baseline
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_baseline_tp64.json"), "r") as f:
    baseline_json = json.load(f)
BASELINE_NEURON_CONFIG = MoENeuronConfig(**baseline_json)

# BS16 TP4/CP16 for CTE, TP8/DP8 for TKG, EP32/TP2 for MOE
with open(os.path.join(CURR_DIR, "neuron_configs/bs16_sl10k_optimized.json"), "r") as f:
    optimized_json = json.load(f)
OPTIMIZED_NEURON_CONFIG = MoENeuronConfig(**optimized_json)


@pytest.mark.tp64
@pytest.mark.parametrize(
    "neuron_config, num_tokens_to_check, divergence_tolerance, check_performance, cte_device_time_threshold, tkg_device_time_threshold",
    [
        # Use 5% regression threshold
        pytest.param(BASELINE_NEURON_CONFIG, 15, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, False, 152.90*1.05, 1.25*1.05),  # BS16 baseline
        pytest.param(OPTIMIZED_NEURON_CONFIG, 16, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, True, 8.7*1.05, 0.68*1.05), # BS16 MoE EP2/TP2, TP4/CP16 for prefill, TP8/DP8 for decode attention
    ],
)
def test_1_layer_accuracy(neuron_config, num_tokens_to_check, divergence_tolerance, check_performance, cte_device_time_threshold, tkg_device_time_threshold):
    # Set random seed for reproducibility
    # Putting it inside the test script so that python and pytest command both run it
    set_random_seed(42)

    # Load model from config, and save with random weights.
    config_path = os.path.join(CURR_DIR, "config.json")

    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name

    generation_config = GenerationConfig(do_sample=False, pad_token_id=0)
    
    config = Qwen3MoeInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Use ProcessPool to release neuron cores after done logit validation so that profiling can run
    if check_performance:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                validate_accuracy,
                model_path,
                config,
                generation_config,
                divergence_tolerance,
                num_tokens_to_check
            )
            future.result()
        validate_performance(config, cte_device_time_threshold, tkg_device_time_threshold)
    else:
        validate_accuracy(model_path, config, generation_config, divergence_tolerance, num_tokens_to_check)

    # Clean up the model checkpoint only if the test passes.
    model_tempdir.cleanup()


def save_checkpoint(config_path):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.float32)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def validate_accuracy(model_path, config, generation_config, divergence_tolerance, num_tokens_to_check):
    # TEST_PROMPT_STABLE = "A pencil cost $0.50, and an eraser cost $0.25. If you bought 6 pencils and 8 erasers and paid $10, how much change would you get?"
    input_ids = torch.tensor([[   32, 46118,  2783,   400,    15,    13,    20,    15,    11,   323,
           458,  2714, 12080,  2783,   400,    15,    13,    17,    20,    13,
          1416,   498, 10788,   220,    21, 96338,   323,   220,    23,  2714,
         59730,   323,  7171,   400,    16,    15,    11,  1246,  1753,  2297,
          1035,   498,   633,    30]]* config.neuron_config.batch_size).to(dtype=torch.int32)
    input_len = input_ids.shape[1]
    attention_mask = torch.ones((config.neuron_config.batch_size, input_len), dtype=torch.int32)
    inputs = Namespace(input_ids=input_ids, attention_mask=attention_mask)

    model = NeuronQwen3MoeForCausalLM(model_path, config)
    compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
    model.compile(compiled_model_path, debug=True)
    model.load(compiled_model_path)

    check_accuracy_logits(
        model,
        generation_config=generation_config,
        # Logits matching for longer sequence length will fail, most likely because
        # experts weights are too close and the router could select different
        # experts because of numeric error.
        num_tokens_to_check=num_tokens_to_check,
        inputs=inputs,
        divergence_difference_tol=divergence_tolerance,
    )


def validate_performance(config, cte_device_time_threshold, tkg_device_time_threshold):   
    # Profile CTE NEFF
    cte_neff_path, cte_output_ntff_folder = get_neff_path_and_output_ntff_folder(CONTEXT_ENCODING_MODEL, bucket_index=None)  # default to profile largest bucket
    cte_norm_metrics, cte_neff_path, cte_ntff_path = run_profiler_on_neff(cte_neff_path, cte_output_ntff_folder, config.neuron_config.world_size)
    
    if "mfu_estimated_percent" in cte_norm_metrics:
        mfu_estimated_percent = cte_norm_metrics["mfu_estimated_percent"] * 100
        print(f"Actual MFU is {mfu_estimated_percent}%")

    cte_device_time = cte_norm_metrics["total_time"] * 1000
    print(f"Actual CTE device time is {cte_device_time} ms")
    assert cte_device_time < cte_device_time_threshold

    # Profile TKG NEFF
    tkg_neff_path, tkg_output_ntff_folder = get_neff_path_and_output_ntff_folder(TOKEN_GENERATION_MODEL, bucket_index=None) # default to profile largest bucket
    tkg_norm_metrics, tkg_neff_path, tkg_ntff_path = run_profiler_on_neff(tkg_neff_path, tkg_output_ntff_folder, config.neuron_config.world_size)
    
    if "mbu_estimated_percent" in tkg_norm_metrics:
        mbu_estimated_percent = tkg_norm_metrics["mbu_estimated_percent"] * 100
        print(f"Actual MBU is {mbu_estimated_percent}%")
    if "mbu_min_read_util_percent" in tkg_norm_metrics:
        mbu_min_read_util_percent = tkg_norm_metrics["mbu_min_read_util_percent"] * 100
        print(f"Actual MBU_MIN_READ_UTIL is {mbu_min_read_util_percent}%")

    tkg_device_time = tkg_norm_metrics["total_time"] * 1000
    print(f"Actual TKG device time is {tkg_device_time} ms")
    assert tkg_device_time < tkg_device_time_threshold


if __name__ == "__main__":
    # For easy `python test_qwen3_moe_1_layer.py` testing rather than using pytest
    test_1_layer_accuracy(BASELINE_NEURON_CONFIG, 26, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, False, 165.90*1.05, 1.25*1.05) # BS16 baseline
    test_1_layer_accuracy(OPTIMIZED_NEURON_CONFIG, 16, DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE, True, 8.7*1.05, 0.68*1.05) # BS16 Optimized, MoE EP2/TP2, TP4/CP16 for prefill, TP8/DP8 for decode attention
    