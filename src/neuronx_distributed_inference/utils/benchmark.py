# flake8: noqa
import copy
import json
import time
from functools import partial

import numpy as np
import torch
from transformers import GenerationConfig

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.constants import *
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"


def benchmark_sampling(
    model: NeuronApplicationBase,
    draft_model: NeuronApplicationBase = None,
    generation_config: GenerationConfig = None,
    target: str = None,
    image=None,
):
    neuron_config = model.neuron_config

    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=generation_config.top_k
        if isinstance(generation_config.top_k, list)
        else [generation_config.top_k],
        top_p=generation_config.top_p
        if isinstance(generation_config.top_p, list)
        else [generation_config.top_p],
        temperature=generation_config.temperature
        if isinstance(generation_config.temperature, list)
        else [generation_config.temperature],
    )

    target = target if target is not None else "all"

    report = {}

    # on_device_sampling flow does not support min_new_tokens
    # to override eos_tokens so we remove EOS tokens to ensure
    # token generation happens.
    modified_generation_config = copy.deepcopy(generation_config)
    if model.on_device_sampling:
        modified_generation_config.eos_token_id = []
    # Benchmark E2E model
    if target in ["all", "e2e"]:
        # FIXME: fix pixel values generation
        input_ids, attention_mask, pixel_values, sampling_params = get_sample_inputs(
            END_TO_END_MODEL, neuron_config, image=image, sampling_params=sampling_params
        )
        input_param = {
            "input_ids": input_ids,
            "generation_config": modified_generation_config,
            "attention_mask": attention_mask,
            "max_new_tokens": neuron_config.max_new_tokens,
            "top_k": 1,
            "do_sample": draft_model is None and not neuron_config.enable_fused_speculation,
            "sampling_params": sampling_params,
        }

        if draft_model is not None:
            hf_draft_model = HuggingFaceGenerationAdapter(draft_model)
            hf_draft_model.generation_config.update(
                num_assistant_tokens=model.neuron_config.speculation_length
            )
            input_param["assistant_model"] = hf_draft_model

        if model.neuron_config.enable_fused_speculation:
            input_param["prompt_lookup_num_tokens"] = model.neuron_config.speculation_length

        if pixel_values is not None:
            input_param["pixel_values"] = pixel_values

        if target == "all":
            latency_collectors = create_submodule_latency_collectors(model)

        def post_warmup_func():
            if target == "all":
                register_latency_collectors(latency_collectors, model)

        # Register latency collectors after warm-up to avoid recording warm-up metrics.
        generation_model = HuggingFaceGenerationAdapter(model)
        e2e_benchmark = Benchmark(
            generation_model.generate,
            input_param,
            preprocess_func=model.reset,
            post_warmup_func=post_warmup_func,
        )
        e2e_benchmark.run()
        report[END_TO_END_MODEL] = generate_report(
            e2e_benchmark.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=e2e_benchmark.num_runs,
        )

        if target == "all":
            report.update(
                generate_submodule_reports(
                    latency_collectors, neuron_config, e2e_benchmark.num_runs
                )
            )

    # Benchmark context encoding model only
    if target == "context_encode":
        input_param = get_sample_inputs(
            CONTEXT_ENCODING_MODEL, neuron_config, sampling_params=sampling_params
        )
        ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, neuron_config)
        ctx_enc_benchmark.run()
        report[CONTEXT_ENCODING_MODEL] = generate_report(
            ctx_enc_benchmark.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=ctx_enc_benchmark.num_runs,
        )

    # Benchmark token generation model only
    if hasattr(model, "token_generation_model") and target == "token_gen":
        input_param = get_sample_inputs(
            TOKEN_GENERATION_MODEL, neuron_config, sampling_params=sampling_params
        )
        tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param)
        tkn_gen_benchmark.run()
        report[TOKEN_GENERATION_MODEL] = generate_report(
            tkn_gen_benchmark.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=tkn_gen_benchmark.num_runs,
        )

    # Benchmark speculation model only
    if hasattr(model, "speculation_model") and target == "speculation":
        input_param = get_sample_inputs(
            SPECULATION_MODEL, neuron_config, sampling_params=sampling_params
        )
        spec_benchmark = Benchmark(model.speculation_model, input_param)
        spec_benchmark.run()
        report[SPECULATION_MODEL] = generate_report(
            spec_benchmark.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=spec_benchmark.num_runs,
        )

    # Benchmark Medusa speculation model
    if hasattr(model, "medusa_speculation_model") and target == "speculation":
        input_param = get_sample_inputs(
            MEDUSA_MODEL, neuron_config, sampling_params=sampling_params
        )
        spec_benchmark = Benchmark(model.medusa_speculation_model, input_param)
        spec_benchmark.run()
        report[MEDUSA_MODEL] = generate_report(
            spec_benchmark.latency_list,
            neuron_config.max_length,
            neuron_config.max_batch_size,
            n_runs=spec_benchmark.num_runs,
        )

    model.reset()
    if draft_model is not None:
        draft_model.reset()

    print("Benchmark completed and its result is as following")
    print(json.dumps(report, indent=4))
    with open(BENCHMARK_REPORT_FILENAME, "w") as f:
        json.dump(report, f)
    print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

    return report


def get_sample_inputs(model_type, neuron_config: NeuronConfig, sampling_params, image=None):
    max_context_length = neuron_config.max_context_length
    max_len = neuron_config.max_length
    batch_size = neuron_config.batch_size
    num_medusa_heads = neuron_config.num_medusa_heads if neuron_config.num_medusa_heads else 4
    medusa_speculation_length = (
        neuron_config.medusa_speculation_length if neuron_config.medusa_speculation_length else 64
    )

    sample_inputs = None
    if model_type == END_TO_END_MODEL:
        input_ids = torch.randint(0, 100, (batch_size, max_context_length))
        attention_mask = torch.zeros((batch_size, max_context_length), dtype=torch.int32)
        assert (
            image is None
        ), "image is not supported currently for benchmarking for END_TO_END_MODEL"

        sample_inputs = (input_ids, attention_mask, None, sampling_params)

    elif model_type == CONTEXT_ENCODING_MODEL:
        input_ids = torch.zeros((batch_size, max_context_length), dtype=torch.int32)
        attention_mask = torch.zeros((batch_size, max_context_length), dtype=torch.int32)
        position_ids = torch.zeros((batch_size, max_context_length), dtype=torch.int32)
        seq_ids = torch.zeros((batch_size), dtype=torch.int32)

        if neuron_config.is_medusa:
            accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int32)
            current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int32)
            medusa_mask = torch.zeros(
                (batch_size, medusa_speculation_length, medusa_speculation_length),
                dtype=torch.int32,
            )
            scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int32)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            )
        elif image:
            pixel_values = torch.zeros(
                (
                    batch_size,
                    3,
                    neuron_config.hf_config.vision_config.image_size,
                    neuron_config.hf_config.vision_config.image_size,
                ),
                dtype=neuron_config.hf_config.torch_dtype,
            )
            text_embedding_indices = torch.zeros(
                (batch_size, max_context_length), dtype=torch.int32
            )
            image_embedding_indices = torch.zeros(
                (batch_size, max_context_length), dtype=torch.int32
            )

            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                pixel_values,
                text_embedding_indices,
                image_embedding_indices,
            )
        else:
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )
    elif model_type == TOKEN_GENERATION_MODEL:
        input_ids = torch.zeros((batch_size, 1), dtype=torch.int32)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int32)
        position_ids = torch.zeros((batch_size, 1), dtype=torch.int32)
        seq_ids = torch.zeros((batch_size), dtype=torch.int32)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
        )
    elif model_type == SPECULATION_MODEL:
        spec_len = neuron_config.speculation_length
        input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int32)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int32)
        position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int32)
        seq_ids = torch.zeros((batch_size), dtype=torch.int32)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
        )

    elif model_type == MEDUSA_MODEL:
        spec_len = neuron_config.medusa_speculation_length
        input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int32)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int32)
        position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int32)
        seq_ids = torch.zeros((batch_size), dtype=torch.int32)
        accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int32)
        current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int32)
        medusa_mask = torch.zeros(
            (batch_size, medusa_speculation_length, medusa_speculation_length), dtype=torch.int32
        )
        scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int32)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            accepted_indices,
            current_length,
            medusa_mask,
            scatter_index,
        )

    return sample_inputs


def create_submodule_latency_collectors(model):
    collectors = {}
    collectors[CONTEXT_ENCODING_MODEL] = LatencyCollector()
    if hasattr(model, "token_generation_model"):
        collectors[TOKEN_GENERATION_MODEL] = LatencyCollector()
    if hasattr(model, "speculation_model"):
        collectors[SPECULATION_MODEL] = LatencyCollector()
    return collectors


def register_latency_collectors(latency_collectors, model):
    register_forward_latency_collector(
        latency_collectors[CONTEXT_ENCODING_MODEL], model.context_encoding_model
    )
    if TOKEN_GENERATION_MODEL in latency_collectors:
        register_forward_latency_collector(
            latency_collectors[TOKEN_GENERATION_MODEL], model.token_generation_model
        )
    if SPECULATION_MODEL in latency_collectors:
        register_forward_latency_collector(
            latency_collectors[SPECULATION_MODEL], model.speculation_model
        )


def register_forward_latency_collector(latency_collector, model):
    model.register_forward_pre_hook(latency_collector.pre_hook)
    model.register_forward_hook(latency_collector.hook)


def generate_submodule_reports(latency_collectors, neuron_config, num_runs):
    reports = {}
    for key, collector in latency_collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = neuron_config.max_context_length
        elif key == "token_generation_model":
            tokens_len = neuron_config.max_new_tokens
        reports[key] = generate_report(
            collector.latency_list, tokens_len, neuron_config.max_batch_size, num_runs
        )
    return reports


class Benchmark:
    def __init__(
        self, benchmark_func, input_param, num_runs=20, preprocess_func=None, post_warmup_func=None
    ) -> None:
        if isinstance(input_param, (tuple, list)):
            self.benchmark_func = partial(benchmark_func, *input_param)
        elif isinstance(input_param, dict):
            self.benchmark_func = partial(benchmark_func, **input_param)
        else:
            self.benchmark_func = partial(benchmark_func, input_param)

        self.num_runs = num_runs
        self.preprocess_func = preprocess_func
        self.post_warmup_func = post_warmup_func
        self.latency_list = None

    def run(self):
        # Warm up
        if self.preprocess_func:
            self.preprocess_func()
        self.benchmark_func()

        if self.post_warmup_func:
            self.post_warmup_func()

        latency_collector = LatencyCollector()
        for _ in range(self.num_runs):
            latency_collector.pre_hook()
            if self.preprocess_func:
                self.preprocess_func()
            self.benchmark_func()
            latency_collector.hook()
        self.latency_list = latency_collector.latency_list


class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)


def generate_report(latency_list, max_length, max_batch_size, n_runs=20):
    latency_array = np.array(latency_list)

    total_time = np.sum(latency_array)
    throughput = (n_runs * max_length * max_batch_size) / total_time

    return {
        "latency_ms_p50": np.percentile(latency_array, 50) * 1000,
        "latency_ms_p90": np.percentile(latency_array, 90) * 1000,
        "latency_ms_p95": np.percentile(latency_array, 95) * 1000,
        "latency_ms_p99": np.percentile(latency_array, 99) * 1000,
        "latency_ms_p100": np.percentile(latency_array, 100) * 1000,
        "latency_ms_avg": np.average(latency_array) * 1000,
        "throughput": throughput,
    }
