import argparse
import ast
import copy
import json
from enum import Enum
from typing import Type

import torch
from neuronx_distributed.quantization.quantization_config import QuantizationType
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig, to_torch_dtype
from neuronx_distributed_inference.models.dbrx.modeling_dbrx import NeuronDbrxForCausalLM
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import NeuronMixtralForCausalLM
from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed_inference.utils.accuracy import check_accuracy, check_accuracy_logits
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

torch.manual_seed(0)

MODEL_TYPES = {
    "llama": {"causal-lm": NeuronLlamaForCausalLM},
    "mixtral": {"causal-lm": NeuronMixtralForCausalLM},
    "dbrx": {"causal-lm": NeuronDbrxForCausalLM},
}


class CheckAccuracyMode(Enum):
    SKIP_ACCURACY_CHECK = "skip-accuracy-check"
    TOKEN_MATCHING = "token-matching"
    LOGIT_MATCHING = "logit-matching"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=MODEL_TYPES.keys(), required=True)
    parser.add_argument("--task-type", type=str, required=True)
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run")
    setup_run_parser(run_parser)

    return parser.parse_args()


def setup_run_parser(run_parser: argparse.ArgumentParser):
    run_parser.add_argument("--model-path", type=str, required=True)
    run_parser.add_argument("--compiled-model-path", type=str, required=True)

    # Evaluation
    run_parser.add_argument("--benchmark", action="store_true")
    run_parser.add_argument(
        "--check-accuracy-mode",
        type=CheckAccuracyMode,
        choices=list(CheckAccuracyMode),
        default=CheckAccuracyMode.SKIP_ACCURACY_CHECK,
    )
    run_parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    run_parser.add_argument("--tol-map", type=str)

    # Generation
    run_parser.add_argument("--prompt", dest="prompts", type=str, action="append", required=True)
    run_parser.add_argument("--top-k", type=int, default=1)
    run_parser.add_argument("--do-sample", action="store_true", default=True)
    run_parser.add_argument("--pad-token-id", type=int, default=0)

    # Basic config
    run_parser.add_argument("--torch-dtype", type=to_torch_dtype)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--padding-side", type=str)
    run_parser.add_argument("--seq-len", type=int)
    run_parser.add_argument("--n-active-tokens", type=int)
    run_parser.add_argument("--n-positions", type=int)
    run_parser.add_argument("--max-context-length", type=int)
    run_parser.add_argument("--max-new-tokens", type=int)
    run_parser.add_argument("--max-length", type=int)

    # Attention
    run_parser.add_argument("--fused-qkv", action="store_true")
    run_parser.add_argument("--sequence-parallel-norm", action="store_true")

    # Continuous batching
    run_parser.add_argument("--ctx-batch-size", type=int)
    run_parser.add_argument("--tkg-batch-size", type=int)
    run_parser.add_argument("--max-batch-size", type=int)
    run_parser.add_argument("--is-continuous-batching", action="store_true")

    # On device sampling
    run_parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    run_parser.add_argument("--enable-bucketing", action="store_true")
    run_parser.add_argument("--bucket-n-active-tokens", action="store_true")

    # Quantization
    run_parser.add_argument("--quantized", action="store_true")
    run_parser.add_argument("--quantized-checkpoints-path", type=str)
    run_parser.add_argument(
        "--quantization-type", type=str, choices=[t.value for t in QuantizationType]
    )

    # MoE
    run_parser.add_argument("--capacity-factor", type=float)

    # Speculative decoding
    run_parser.add_argument("--draft-model-path", type=str)
    run_parser.add_argument("--compiled-draft-model-path", type=str)

    run_parser.add_argument(
        "--no-trace-tokengen-model", dest="trace_tokengen_model", action="store_false"
    )
    run_parser.add_argument("--speculation-length", type=int)
    run_parser.add_argument("--spec-batch-size", type=int)

    # Medusa decoding
    run_parser.add_argument("--is-medusa", action="store_true")
    run_parser.add_argument("--medusa-speculation-length", type=int)
    run_parser.add_argument("--num-medusa-heads", type=int)
    run_parser.add_argument("--medusa-tree-json", type=load_json_file, dest="medusa_tree")

    # Parallism
    run_parser.add_argument("--tp-degree", type=int)
    run_parser.add_argument("--pp-degree", type=int)
    run_parser.add_argument("--ep-degree", type=int)
    run_parser.add_argument("--world-size", type=int)
    run_parser.add_argument("--start_rank_id", type=int)
    run_parser.add_argument("--local_ranks_size", type=int)

    # lora
    run_parser.add_argument("--enable-lora", action="store_true")
    run_parser.add_argument("--max-loras", type=int)
    run_parser.add_argument("--max-lora-rank", type=int)
    run_parser.add_argument("--target-modules", nargs="+")
    run_parser.add_argument("--max-loras-on-cpu", type=int)


def load_json_file(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def run_inference(model_cls: Type[NeuronApplicationBase], args):
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

    adapter_ids = None
    if args.enable_lora:
        config_kwargs["lora_config"] = LoraServingConfig(
            max_loras=args.max_loras,
            max_lora_rank=args.max_lora_rank,
            target_modules=args.target_modules,
            max_loras_on_cpu=args.max_loras_on_cpu,
        )
        adapter_ids = torch.tensor([0, 1], dtype=torch.int64)
    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    # Initialize draft model.
    draft_model = None
    if neuron_config.speculation_length > 0 and args.draft_model_path is not None:
        # Reset speculation options to defaults for the draft model.
        draft_neuron_config = copy.deepcopy(config.neuron_config)
        draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True

        draft_config = model_cls.get_config_cls()(
            draft_neuron_config, load_config=load_pretrained_config(args.draft_model_path)
        )

        draft_model = model_cls(args.draft_model_path, draft_config)

    # Quantize model.
    if neuron_config.quantized:
        model_cls.save_quantized_state_dict(args.model_path, config)

    # Compile and save model.
    print("\nCompiling and saving model...")
    model.compile(args.compiled_model_path)
    if draft_model is not None:
        print("\nCompiling and saving draft model...")
        draft_model.compile(args.compiled_draft_model_path)

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    if draft_model is not None:
        print("\nLoading draft model to Neuron...")
        draft_model.load(args.compiled_draft_model_path)

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = ["do_sample", "top_k", "pad_token_id"]
    generation_config_kwargs = {
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    # With Medusa, the model is also the draft model.
    if neuron_config.is_medusa:
        draft_model = model

    # Check accuracy.
    run_accuracy_check(
        model,
        tokenizer,
        generation_config,
        args.check_accuracy_mode,
        args.divergence_difference_tol,
        args.tol_map,
        draft_model=draft_model,
    )

    # Generate outputs.
    run_generation(model, tokenizer, args.prompts, generation_config, draft_model=draft_model, adapter_ids=adapter_ids)

    # Benchmarking.
    if args.benchmark:
        benchmark_sampling(model, draft_model, generation_config)


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def run_generation(model, tokenizer, prompts, generation_config, draft_model=None, adapter_ids=None):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    kwargs = {}
    if draft_model is not None:
        draft_generation_model = HuggingFaceGenerationAdapter(draft_model)
        kwargs.update({"assistant_model": draft_generation_model, "do_sample": False})

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.neuron_config.max_length,
        adapter_ids=adapter_ids,
        **kwargs,
    )

    model.reset()
    if draft_model is not None:
        draft_model.reset()

    output_tokens = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(model, 
                       tokenizer, 
                       generation_config, 
                       check_accuracy_mode, 
                       divergence_difference_tol, 
                       tol_map,
                       draft_model=None):
    if model.neuron_config.is_medusa:
        # Medusa doesn't use greedy sampling, so check accuracy doesn't work.
        assert (
            check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK
        ), "Accuracy checking not supported for Medusa"

    if check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK:
        print("\nSkipping accuracy check")
    elif check_accuracy_mode == CheckAccuracyMode.TOKEN_MATCHING:
        print("\nChecking accuracy by token matching")
        check_accuracy(model, tokenizer, generation_config, draft_model=draft_model)
    elif check_accuracy_mode == CheckAccuracyMode.LOGIT_MATCHING:
        assert draft_model is None, "Logit matching not supported for speculation"
        print("\nChecking accuracy by logit matching")

        if tol_map:
            tol_map = ast.literal_eval(tol_map)

        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
        )
    else:
        raise ValueError(f"Unsupported check accuracy mode: {check_accuracy_mode}")


def main():
    args = parse_args()
    assert (
        args.task_type in MODEL_TYPES[args.model_type]
    ), f"Unsupported task: {args.model_type}/{args.task_type}"
    model_cls = MODEL_TYPES[args.model_type][args.task_type]
    run_inference(model_cls, args)


if __name__ == "__main__":
    main()
