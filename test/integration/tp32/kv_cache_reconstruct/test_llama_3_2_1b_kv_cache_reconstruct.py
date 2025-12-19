import os
import random
import tempfile

import pytest
import torch
import transformers
from torch_neuronx.testing.validation import assert_close
from transformers import AutoConfig, AutoModel, GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits_v2
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.kv_cache_reconstruct_utils import (
    NeuronDeviceCache,
    reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree,
    reconstruct_neuron_cpu_kv_cache,
)
from torch.nn.modules.container import ParameterList

##############################################################################
# Constants
##############################################################################
LOGIT_CLOSE_THRESHOLD = 4e-05
KV_CACHE_CLOSE_THRESHOLD = 4e-05

# Path to Llama3.2-1B
MODEL_CONFIG_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)) + "/../models/llama/llama3.2/1b/config.json"
)


##############################################################################
# Model Loading Utilities
##############################################################################
def save_checkpoint(config_path, model_weight_dtype: torch.dtype):
    hf_config = AutoConfig.from_pretrained(config_path)
    hf_model = AutoModel.from_config(hf_config, torch_dtype=model_weight_dtype)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


def load_neuron_model(model, model_path, cpu_mode):
    if cpu_mode:
        print("\nLoading model to CPU...")
        model.to_cpu()
    else:
        print("\nCompiling and loading model to device ...")
        compiled_model_path = model_path + "/compiled_checkpoint_accuracy"
        model.compile(compiled_model_path)
        model.load(compiled_model_path)
    return model


##############################################################################
# Generation Utils
##############################################################################
def create_inputs(inference_config: InferenceConfig, input_len: int = 16):
    if not hasattr(inference_config, "vocab_size"):
        raise KeyError("Inference Config has no attribute vocab_size")

    vocab_size = inference_config.vocab_size
    max_context_len = inference_config.neuron_config.max_context_length
    batch_size = inference_config.neuron_config.batch_size

    # Create Random Prompt
    input_ids = torch.rand(batch_size, input_len) * vocab_size
    input_ids = input_ids.to(torch.int32)

    # Create Attention Mask
    attention_mask = torch.ones((batch_size, input_len), dtype=torch.int32)

    max_new_tokens = max_context_len - input_len

    return input_ids, attention_mask, max_new_tokens


##############################################################################
# Validation and Reconstruction
##############################################################################
def validate_logits_and_kv_cache(
    neuron_config: NeuronConfig, config_path: str, cpu_mode: bool = False
):
    """
    Creates model weights, loads the Neuron and HF model, and runs generation/validation
    """

    model_tempdir = save_checkpoint(config_path, model_weight_dtype=neuron_config.torch_dtype)
    model_weight_path = model_tempdir.name

    # Create Llama Configurations
    llama_config = LlamaInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_pretrained_config(model_weight_path),
    )

    # Create Neuron model
    neuron_model = NeuronLlamaForCausalLM(model_weight_path, llama_config)
    load_neuron_model(neuron_model, model_weight_path, cpu_mode)

    # Create HF Model
    hf_model = neuron_model.load_hf_model(model_weight_path)

    # Prepare inputs and configs for forward pass
    input_ids, attention_mask, max_new_tokens = create_inputs(llama_config)

    generation_config = GenerationConfig(
        do_sample=False, pad_token_id=0)

    # HF Forward Pass
    hf_outputs = hf_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )

    # Verify Logits Match
    check_accuracy_logits_v2(
        neuron_model=neuron_model,
        inputs_input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        expected_logits=torch.stack(hf_outputs.scores),
        generation_config=generation_config
    )

    # Verify KV Cache; note the neuron forward pass in `check_accuracy_logits_v2` populates the KV CAche
    _validate_kv_cache(neuron_model, hf_outputs.past_key_values, llama_config, cpu_mode)

    model_tempdir.cleanup()


def _validate_kv_cache(
    neuron_llama_model: NeuronLlamaForCausalLM,
    hf_cache: transformers.cache_utils.DynamicCache,
    inference_config: InferenceConfig,
    cpu_mode: bool,
):
    """
    Reconstructs Neuron KV Cache and validates it compared to HF Goldens using `assert_close`
    """
    if cpu_mode:
        neuron_sharded_cache: ParameterList = (
            neuron_llama_model.context_encoding_model.model.kv_mgr.past_key_values
        )

        reconstructed_neuron_k_cache, reconstructed_neuron_v_cache = reconstruct_neuron_cpu_kv_cache(
            neuron_sharded_cache
        )
    else:
        neuron_sharded_cache: NeuronDeviceCache = (
            neuron_llama_model.context_encoding_model.model.nxd_model.state
        )

        reconstructed_neuron_k_cache, reconstructed_neuron_v_cache = (
            reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(
                neuron_sharded_cache, inference_config
            )
        )

    hf_k_cache = torch.stack([layer.keys for layer in hf_cache.layers], dim=1)
    hf_v_cache = torch.stack([layer.values for layer in hf_cache.layers], dim=1)

    # Since Neuron Tensors are statically shaped, slice out the Batch and Seq Len dimension
    num_batches = hf_k_cache.shape[0]
    num_kv_cache_tokens = inference_config.neuron_config.max_context_length - 1

    reconstructed_neuron_k_cache = reconstructed_neuron_k_cache[:num_batches, :, :, :num_kv_cache_tokens, :]
    reconstructed_neuron_v_cache = reconstructed_neuron_v_cache[:num_batches, :, :, :num_kv_cache_tokens, :]

    assert_close(
        reconstructed_neuron_k_cache,
        hf_k_cache,
        rtol=KV_CACHE_CLOSE_THRESHOLD,
    )

    assert_close(
        reconstructed_neuron_v_cache,
        hf_v_cache,
        rtol=KV_CACHE_CLOSE_THRESHOLD,
    )

    print(f"All KV Cache Heads Match within {KV_CACHE_CLOSE_THRESHOLD}!")


# Note: if you run with CPU test before Neuron Test, use --forked
@pytest.mark.parametrize(
    "tp_degree, batch_size, max_context_length, model_weight_dtype, cpu_mode, rpl_reduce_dtype",
    [
        (32, 2, 256, torch.float32, False, torch.float32),
        (16, 2, 256, torch.float32, False, torch.float32),
        (1, 2, 256, torch.float32, True, torch.float32),  # Note: only TP=1 is supported for CPU
    ],
)
def test_llama3_2_1b_4layer_kv_cache_reconstruction(
    tp_degree: int,
    batch_size: int,
    max_context_length: int,
    model_weight_dtype: torch.dtype,
    cpu_mode: bool,
    rpl_reduce_dtype: torch.dtype,
):
    if cpu_mode and tp_degree != 1:
        raise ValueError("Neuron CPU Mode KV Cache aggregation currently only supports TP=1")

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=max_context_length,
        torch_dtype=model_weight_dtype,
        on_cpu=cpu_mode,
        rpl_reduce_dtype=rpl_reduce_dtype,
    )

    validate_logits_and_kv_cache(neuron_config, MODEL_CONFIG_PATH, cpu_mode=cpu_mode)


if __name__ == "__main__":
    print("Testing llama-3.2 1B with KV Cache Reconstruction")
    print(
        "This script demos only 2 tests and is to be used as an example. Please use pytest for full coverage."
    )

    # Set seed for reproduciblity
    torch.manual_seed(42)

    # Note: to avoid having to destroy the CPU environment, we run with cpu_mode=False first, then cpu_mode=True
    test_llama3_2_1b_4layer_kv_cache_reconstruction(
        tp_degree=32,
        batch_size=2,
        max_context_length=256,
        model_weight_dtype=torch.float32,
        cpu_mode=False,
        rpl_reduce_dtype=torch.float32,
    )

    test_llama3_2_1b_4layer_kv_cache_reconstruction(
        tp_degree=1,
        batch_size=2,
        max_context_length=256,
        model_weight_dtype=torch.float32,
        cpu_mode=True,
        rpl_reduce_dtype=torch.float32,
    )

    print(
        "######################################\nAll KV Cache Tests Passed!\n######################################"
    )
