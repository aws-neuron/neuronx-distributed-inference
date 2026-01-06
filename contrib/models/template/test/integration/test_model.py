"""
This sample test script demonstrates how to validate model accuracy and performance for Neuron
modeling code that works with a Huggingface checkpoint (such as Llama3.2 1B).

To validate accuracy, this test script uses logit validation, which compares output logits against
expected logits. You can provide expected logits from generating on GPU, or you can let the logit
validation tool generate expected logits on CPU.

To validate performance, this test script runs the NxDI benchmarking API, which reports the latency
and throughput for the model and each sub-model (such as context encoding and token generation).
The test script validates that the time-to-first-token (TTFT) and throughput meet given thresholds.

Note that for larger models and larger sequence lengths, this script takes a longer amount of time
to check accuracy and performance. By default, during logit validation, NxDI runs the HuggingFace
transformers model on CPU, which takes awhile for larger models. To save time, you can save the
and reuse the expected outputs by passing `expected_logits` to `check_accuracy_logits`.

See also:
* https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html#nxdi-logit-matching
* https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/onboarding-models.html#nxdi-benchmark-sampling
"""

import pytest
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.exceptions import LogitMatchingValidationError
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = "/home/ubuntu/models/Llama-3.2-1B/"
compiled_model_path = "/home/ubuntu/neuron-models/Llama-3.2-1B/"

NUM_TOKENS_TO_CHECK = 256

torch.manual_seed(0)

@pytest.mark.parametrize(
    "batch_size, seq_len, ttft_threshold, throughput_threshold",
    [
        (1, 128, 4.94, 524),
        (4, 128, 10.68, 1203),
        (8, 128, 17.16, 1866),
        (1, 8192, 71.06, 388),
        (4, 8192, 257.86, 736),
        (1, 32768, 363.20, 247),
    ]
)
def test_model_accuracy_and_performance(batch_size, seq_len, ttft_threshold, throughput_threshold):
    print(f"Testing model with parameters: {batch_size=}, {seq_len=}, {ttft_threshold=}, {throughput_threshold=}")

    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(
        model_path,
        do_sample=False,
        top_k=1,
    )
    neuron_config = NeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        max_context_length=seq_len,
        seq_len=seq_len,
        enable_bucketing=False,
        torch_dtype=torch.bfloat16,
    )
    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(compiled_model_path)
    model.load(compiled_model_path)

    # Check accuracy. This checks the accuracy of all logits at every token.
    try:
        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            num_tokens_to_check=NUM_TOKENS_TO_CHECK,
        )
    except LogitMatchingValidationError as e:
        print(e)
        raise e

    # Check that the performance is within 10% of defined thresholds.
    benchmark_report = benchmark_sampling(model, generation_config=generation_config)
    assert benchmark_report["context_encoding_model"]["latency_ms_p50"] < ttft_threshold * 1.1
    assert benchmark_report["token_generation_model"]["throughput"] < throughput_threshold * 1.1
    print(f"Test passed for parameters: {batch_size=}, {seq_len=}, {ttft_threshold=}, {throughput_threshold=}")
