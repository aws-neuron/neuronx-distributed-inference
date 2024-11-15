import argparse
import ast
import copy
import json
import os
import time
from enum import Enum
from typing import Type

import torch
from neuronx_distributed.quantization.quantization_config import QuantizationType
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import (
    FusedSpecNeuronConfig,
    OnDeviceSamplingConfig,
    to_torch_dtype,
)
from neuronx_distributed_inference.models.dbrx.modeling_dbrx import NeuronDbrxForCausalLM
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import NeuronMixtralForCausalLM
from neuronx_distributed_inference.modules.lora_serving import LoraServingConfig
from neuronx_distributed_inference.utils.accuracy import (
    check_accuracy,
    check_accuracy_logits,
    get_generate_outputs,
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.distributed import get_init_rank, get_init_world_size
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed

set_random_seed(0)


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
    run_parser.add_argument("--expected-outputs-path", type=validate_file_exists)
    run_parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    run_parser.add_argument("--tol-map", type=str)
    run_parser.add_argument("--num-tokens-to-check", type=int)

    # Generation
    run_parser.add_argument("--prompt", dest="prompts", type=str, action="append", required=True)
    run_parser.add_argument("--top-k", type=int, default=1)
    run_parser.add_argument("--global-topk", type=int)
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
    run_parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)

    run_parser.add_argument("--vocab-parallel", action="store_true")

    # Attention
    run_parser.add_argument("--fused-qkv", action="store_true")
    run_parser.add_argument("--sequence-parallel-enabled", action="store_true")
    run_parser.add_argument("--flash-decoding-enabled", action="store_true")

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
    run_parser.add_argument("--kv-cache-quant", action="store_true")
    run_parser.add_argument("--quantization-dtype", type=str)

    # MoE
    run_parser.add_argument("--capacity-factor", type=float)

    # Speculative decoding
    run_parser.add_argument("--draft-model-path", type=str)
    run_parser.add_argument("--draft-model-tp-degree", type=int, default=None)
    run_parser.add_argument("--compiled-draft-model-path", type=str)
    run_parser.add_argument("--enable-fused-speculation", action="store_true", default=False)
    run_parser.add_argument("--enable-eagle-speculation", action="store_true", default=False)
    run_parser.add_argument("--enable-eagle-draft-input-norm", action="store_true", default=False)

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

    # Parallelism
    run_parser.add_argument("--tp-degree", type=int)
    run_parser.add_argument("--pp-degree", type=int)
    run_parser.add_argument("--ep-degree", type=int)
    run_parser.add_argument("--world-size", type=int)
    run_parser.add_argument("--start_rank_id", type=int)
    run_parser.add_argument("--local_ranks_size", type=int)
    run_parser.add_argument(
        "--enable-torch-dist",
        action="store_true",
        help="Use torch.distributed (gloo) backend when running multi-node examples. "
        "This is useful for ensuring processes on different nodes are in sync",
    )
    run_parser.add_argument(
        "--skip-save-sharded-checkpoint", dest="save_sharded_checkpoint", action="store_false"
    )

    # async
    run_parser.add_argument("--async", action="store_true")

    # lora
    run_parser.add_argument("--enable-lora", action="store_true")
    run_parser.add_argument("--max-loras", type=int)
    run_parser.add_argument("--max-lora-rank", type=int)
    run_parser.add_argument("--target-modules", nargs="+")
    run_parser.add_argument("--max-loras-on-cpu", type=int)

    # Kernels
    run_parser.add_argument("--qkv-kernel-enabled", action="store_true")
    run_parser.add_argument("--attn-kernel-enabled", action="store_true")
    run_parser.add_argument("--mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    run_parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    run_parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    run_parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")
    run_parser.add_argument("--logical-neuron-cores", type=int, default=1)
    run_parser.add_argument("--cc-pipeline-tiling-factor", type=int, default=2)

    # optional demo arguments
    run_parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip model compilation. If this option is set, then compiled model must be "
        "present at path specified by --compiled-model-path argument",
    )
    run_parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only perform model compilation.",
    )
    run_parser.add_argument(
        "--hlo-debug",
        action="store_true",
        help="Adds metadata into the generated HLO. This metadata maps the HLO "
        "operators to the corresponding lines in the PyTorch code",
    )


def validate_file_exists(path):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise argparse.ArgumentError("Path must exist and be a file")
    return path


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

    if (args.quantized and args.quantization_dtype == "f8e4m3") or args.kv_cache_quant:
        os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"

    adapter_ids = None
    if args.enable_lora:
        config_kwargs["lora_config"] = LoraServingConfig(
            max_loras=args.max_loras,
            max_lora_rank=args.max_lora_rank,
            target_modules=args.target_modules,
            max_loras_on_cpu=args.max_loras_on_cpu,
        )
        adapter_ids = torch.tensor([0, 1], dtype=torch.int32)
    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    # Initialize draft model.
    draft_model = None
    if neuron_config.speculation_length > 0 and args.draft_model_path is not None:
        # Reset speculation options to defaults for the draft model.
        draft_neuron_config = copy.deepcopy(config.neuron_config)
        # eagle requires the draft model to have speculation enabled for the last draft run
        if not neuron_config.enable_eagle_speculation:
            draft_neuron_config.speculation_length = 0
        draft_neuron_config.trace_tokengen_model = True
        draft_neuron_config.enable_fused_speculation = False
        # Set eagle specific config changes
        if neuron_config.enable_eagle_speculation:
            draft_neuron_config.is_eagle_draft = True
            draft_neuron_config.sequence_parallel_enabled = False

        if args.draft_model_tp_degree is not None:
            draft_neuron_config.tp_degree = args.draft_model_tp_degree

        draft_config = model_cls.get_config_cls()(
            draft_neuron_config, load_config=load_pretrained_config(args.draft_model_path)
        )
        if neuron_config.enable_fused_speculation:
            fused_spec_config = FusedSpecNeuronConfig(
                model_cls._model_cls,
                draft_config=draft_config,
                draft_model_path=args.draft_model_path,
            )
            config.fused_spec_config = fused_spec_config

        else:
            draft_model = model_cls(args.draft_model_path, draft_config)

    model = model_cls(args.model_path, config)

    # Quantize model.
    if neuron_config.quantized:
        model_cls.save_quantized_state_dict(args.model_path, config)

    # Compile and save model.
    compiling_start_time = time.monotonic()
    if not args.skip_compile:
        print("\nCompiling and saving model...")
        model.compile(args.compiled_model_path, debug=args.hlo_debug)
        if draft_model is not None and neuron_config.enable_fused_speculation is False:
            print("\nCompiling and saving draft model...")
            draft_model.compile(args.compiled_draft_model_path)

    if args.enable_torch_dist:
        torch.distributed.barrier()

    if args.compile_only:
        return
    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")
    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    if draft_model is not None and neuron_config.enable_fused_speculation is False:
        print("\nLoading draft model to Neuron...")
        draft_model.load(args.compiled_draft_model_path)

    if args.enable_torch_dist:
        torch.distributed.barrier()

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
        args.prompts[0],
        args.check_accuracy_mode,
        args.divergence_difference_tol,
        args.tol_map,
        num_tokens_to_check=args.num_tokens_to_check,
        draft_model=draft_model,
        expected_outputs_path=args.expected_outputs_path,
    )

    # Generate outputs.
    run_generation(
        model,
        tokenizer,
        args.prompts,
        generation_config,
        draft_model=draft_model,
        adapter_ids=adapter_ids,
    )

    # Benchmarking.
    if args.benchmark:
        benchmark_sampling(model, draft_model, generation_config)


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def run_generation(
    model,
    tokenizer,
    prompts,
    generation_config,
    draft_model=None,
    adapter_ids=None,
):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        draft_model=draft_model,
        generation_config=generation_config,
        adapter_ids=adapter_ids,
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(
    model,
    tokenizer,
    generation_config,
    prompt,
    check_accuracy_mode,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check=None,
    draft_model=None,
    expected_outputs_path=None,
):
    if model.neuron_config.is_medusa:
        # Medusa doesn't use greedy sampling, so check accuracy doesn't work.
        assert (
            check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK
        ), "Accuracy checking not supported for Medusa"

    if check_accuracy_mode == CheckAccuracyMode.SKIP_ACCURACY_CHECK:
        print("\nSkipping accuracy check")
        return

    expected_outputs = None
    if expected_outputs_path is not None:
        expected_outputs = torch.load(expected_outputs_path)

    if check_accuracy_mode == CheckAccuracyMode.TOKEN_MATCHING:
        print("\nChecking accuracy by token matching")
        check_accuracy(
            model,
            tokenizer,
            generation_config,
            prompt=prompt,
            draft_model=draft_model,
            expected_token_ids=expected_outputs,
        )
    elif check_accuracy_mode == CheckAccuracyMode.LOGIT_MATCHING:
        assert draft_model is None, "Logit matching not supported for speculation"
        print("\nChecking accuracy by logit matching")

        expected_logits = None
        if expected_outputs is not None:
            expected_logits = torch.stack(expected_outputs.scores)

        if tol_map:
            tol_map = ast.literal_eval(tol_map)

        check_accuracy_logits(
            model,
            tokenizer,
            generation_config,
            prompt=prompt,
            expected_logits=expected_logits,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
        )
    else:
        raise ValueError(f"Unsupported check accuracy mode: {check_accuracy_mode}")


def main():
    args = parse_args()
    assert (
        args.task_type in MODEL_TYPES[args.model_type]
    ), f"Unsupported task: {args.model_type}/{args.task_type}"

    if args.enable_torch_dist:
        torch.distributed.init_process_group(
            backend="gloo",
            world_size=get_init_world_size(),
            rank=get_init_rank(),
        )
        node_rank = torch.distributed.get_rank()
        args.start_rank_id = node_rank * args.local_ranks_size
        torch.distributed.barrier()

    model_cls = MODEL_TYPES[args.model_type][args.task_type]
    run_inference(model_cls, args)


if __name__ == "__main__":
    main()
