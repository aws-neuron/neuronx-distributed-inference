# flake8: noqa
import json
import time
from functools import partial

import numpy as np
import torch
from transformers import PreTrainedModel

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.constants import *

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"


def benchmark_sampling(
    model: PreTrainedModel,
    draft_model: PreTrainedModel = None,
    target: str = None,
    image=None,
):
    neuron_config = model.neuron_config

    target = target if target is not None else "all"

    report = {}

    # Benchmark E2E model
    if target in ["all", "e2e"]:
        # FIXME: fix pixel values generation
        input_ids, attention_mask, pixel_values = get_sample_inputs(
            END_TO_END_MODEL, neuron_config, image=image
        )
        input_param = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": neuron_config.max_new_tokens,
            "top_k": 1,
            "do_sample": draft_model is None,
            "assistant_model": draft_model,
        }

        if pixel_values is not None:
            input_param["pixel_values"] = pixel_values

        if target == "all":
            latency_collectors = create_submodule_latency_collectors(model)

        def post_warmup_func():
            if target == "all":
                register_latency_collectors(latency_collectors, model)

        # Register latency collectors after warm-up to avoid recording warm-up metrics.
        e2e_benchmark = Benchmark(
            model.generate,
            input_param,
            preprocess_func=model.reset,
            post_warmup_func=post_warmup_func,
        )
        e2e_benchmark.run()
        report[END_TO_END_MODEL] = generate_report(
            e2e_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size
        )

        if target == "all":
            report.update(
                generate_submodule_reports(
                    latency_collectors, neuron_config.max_length, neuron_config.max_batch_size
                )
            )

    # Benchmark context encoding model only
    if target == "context_encode":
        input_param = get_sample_inputs(CONTEXT_ENCODING_MODEL, neuron_config)
        ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, neuron_config)
        ctx_enc_benchmark.run()
        report[CONTEXT_ENCODING_MODEL] = generate_report(
            ctx_enc_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size
        )

    # Benchmark token generation model only
    if hasattr(model, "token_generation_model") and target == "token_gen":
        input_param = get_sample_inputs(TOKEN_GENERATION_MODEL, neuron_config)
        tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param)
        tkn_gen_benchmark.run()
        report[TOKEN_GENERATION_MODEL] = generate_report(
            tkn_gen_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size
        )

    # Benchmark speculation model only
    if hasattr(model, "speculation_model") and target == "speculation":
        input_param = get_sample_inputs(SPECULATION_MODEL, neuron_config)
        spec_benchmark = Benchmark(model.speculation_model, input_param)
        spec_benchmark.run()
        report[SPECULATION_MODEL] = generate_report(
            spec_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size
        )

    # Benchmark Medusa speculation model
    if hasattr(model, "medusa_speculation_model") and target == "speculation":
        input_param = get_sample_inputs(MEDUSA_MODEL, neuron_config)
        spec_benchmark = Benchmark(model.medusa_speculation_model, input_param)
        spec_benchmark.run()
        report[MEDUSA_MODEL] = generate_report(
            spec_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size
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


def get_sample_inputs(model_type, neuron_config: NeuronConfig, image=None):
    max_context_length = neuron_config.max_length
    max_len = neuron_config.max_length + neuron_config.max_new_tokens
    batch_size = neuron_config.batch_size
    num_medusa_heads = neuron_config.num_medusa_heads if neuron_config.num_medusa_heads else 4
    medusa_speculation_length = (
        neuron_config.medusa_speculation_length if neuron_config.medusa_speculation_length else 64
    )

    sample_inputs = None
    if model_type == END_TO_END_MODEL:
        input_ids = torch.randint(0, 100, (batch_size, max_context_length))
        attention_mask = torch.zeros((batch_size, max_context_length), dtype=torch.int64)
        assert (
            image is None
        ), "image is not supported currently for benchmarking for END_TO_END_MODEL"

        sample_inputs = (input_ids, attention_mask, None)

    elif model_type == CONTEXT_ENCODING_MODEL:
        input_ids = torch.zeros((batch_size, max_context_length), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_context_length), dtype=torch.int64)
        position_ids = torch.zeros((batch_size, max_context_length), dtype=torch.int64)
        seq_ids = torch.zeros((batch_size), dtype=torch.int64)

        if neuron_config.is_medusa:
            accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
            current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
            medusa_mask = torch.zeros(
                (batch_size, medusa_speculation_length, medusa_speculation_length),
                dtype=torch.int64,
            )
            scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int64)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
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
                (batch_size, max_context_length), dtype=torch.int64
            )
            image_embedding_indices = torch.zeros(
                (batch_size, max_context_length), dtype=torch.int64
            )

            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
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
            )
    elif model_type == TOKEN_GENERATION_MODEL:
        input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int64)
        position_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
        seq_ids = torch.zeros((batch_size), dtype=torch.int64)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
        )
    elif model_type == SPECULATION_MODEL:
        spec_len = neuron_config.speculation_length
        input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int64)
        position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
        seq_ids = torch.zeros((batch_size), dtype=torch.int64)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
        )

    elif model_type == MEDUSA_MODEL:
        spec_len = neuron_config.medusa_speculation_length
        input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int64)
        position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
        seq_ids = torch.zeros((batch_size), dtype=torch.int64)
        accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
        current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
        medusa_mask = torch.zeros(
            (batch_size, medusa_speculation_length, medusa_speculation_length), dtype=torch.int64
        )
        scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int64)
        sample_inputs = (
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
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


def generate_submodule_reports(latency_collectors, max_length, max_batch_size):
    return {
        key: generate_report(collector.latency_list, max_length, max_batch_size)
        for key, collector in latency_collectors.items()
    }


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


def generate_report(latency_list, max_length, max_batch_size):
    latency_array = np.array(latency_list)

    n_runs = len(latency_list)
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
