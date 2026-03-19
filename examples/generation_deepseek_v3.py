"""
DeepSeek V3 (671B) inference demo on AWS Trainium.

Hardware requirements:
  - Full model (671B): trn2.48xlarge (tp_degree=32 or tp_degree=64)
  - Mini model (testing): trn2.3xlarge (tp_degree=2)

Usage:
  # Full model (requires trn2.48xlarge and DeepSeek-V3 weights):
  python generation_deepseek_v3.py --model-path /path/to/DeepSeek-V3

  # With benchmarking parameters:
  python generation_deepseek_v3.py --model-path /path/to/DeepSeek-V3 \
      --tp-degree 32 --seq-len 512 --batch-size 1 --max-new-tokens 128

  # Load from compiled checkpoint:
  python generation_deepseek_v3.py --traced-model-path /path/to/traced_model --skip-compile
"""

import argparse
import os
import threading
import time

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.deepseek.modeling_deepseek import (
    DeepseekV3InferenceConfig,
    NeuronDeepseekV3ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config


torch.manual_seed(0)


def _mem_gb():
    """Return (RSS_GB, available_GB, swap_used_GB) from /proc."""
    rss = 0
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / (1024 * 1024)
                break
    avail = swap = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / (1024 * 1024)
            elif line.startswith("SwapTotal:"):
                swap_total = int(line.split()[1]) / (1024 * 1024)
            elif line.startswith("SwapFree:"):
                swap = swap_total - int(line.split()[1]) / (1024 * 1024)
    return rss, avail, swap


def _start_mem_monitor(interval=30):
    """Background thread that logs memory every `interval` seconds."""
    def _monitor():
        while not _stop_monitor.is_set():
            rss, avail, swap = _mem_gb()
            print(f"  [MEM] RSS={rss:.1f}GB  Available={avail:.1f}GB  Swap={swap:.1f}GB")
            _stop_monitor.wait(interval)
    global _stop_monitor
    _stop_monitor = threading.Event()
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return _stop_monitor


def get_neuron_config(tp_degree=32, batch_size=1, seq_len=4096):
    # trn2 default is lnc=2: pairs of physical NCs share 24GB HBM banks.
    # With lnc=2: tp=32 -> 32 logical cores, tp=64 -> 64 logical cores.
    # lnc=1 causes HBM OOM because two ranks share one 24GB bank (~42GB needed).
    lnc = 2
    return MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        ctx_batch_size=1,
        tkg_batch_size=batch_size,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=lnc,
        save_sharded_checkpoint=True,
    )


def generate(model_path, traced_model_path, skip_compile=False,
             tp_degree=32, seq_len=4096, batch_size=1, max_new_tokens=128):

    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        generation_config = GenerationConfig()
    generation_config.update(do_sample=True, top_k=1, eos_token_id=1, pad_token_id=1)

    stop_monitor = _start_mem_monitor(interval=30)

    if not skip_compile:
        neuron_config = get_neuron_config(tp_degree=tp_degree, batch_size=batch_size, seq_len=seq_len)
        config = DeepseekV3InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print(f"\n{'='*60}")
        print(f"  DeepSeek V3 Benchmark")
        print(f"  tp_degree={tp_degree}, seq_len={seq_len}, batch_size={batch_size}")
        print(f"{'='*60}")

        rss, avail, swap = _mem_gb()
        print(f"  [MEM] Before compile: RSS={rss:.1f}GB  Available={avail:.1f}GB")

        print("\n[1/3] Compiling model...")
        t0 = time.time()
        model = NeuronDeepseekV3ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        compile_time = time.time() - t0
        print(f"  Compilation time: {compile_time:.1f}s ({compile_time/60:.1f}m)")
        tokenizer.save_pretrained(traced_model_path)

    print("\n[2/3] Loading compiled model...")
    t0 = time.time()
    model = NeuronDeepseekV3ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    print(f"\n[3/3] Generating (max_new_tokens={max_new_tokens})...")
    prompts = ["The capital of France is"]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]

    generation_model = HuggingFaceGenerationAdapter(model)

    # Warmup run
    print("  Warmup run...")
    _ = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=4,
    )

    # Timed run
    print("  Timed run...")
    t0 = time.time()
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    total_time = time.time() - t0
    output_len = outputs.shape[1]
    new_tokens = output_len - input_len

    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Report metrics
    tpot = total_time / new_tokens if new_tokens > 1 else 0
    throughput = new_tokens / total_time if total_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Input tokens:    {input_len}")
    print(f"  Output tokens:   {new_tokens}")
    print(f"  Total time:      {total_time:.3f}s")
    print(f"  TPOT:            {tpot*1000:.1f}ms")
    print(f"  Throughput:      {throughput:.1f} tok/s")
    if not skip_compile:
        print(f"  Compile time:    {compile_time:.1f}s")
    print(f"  Load time:       {load_time:.1f}s")
    print(f"{'='*60}")

    for i, text in enumerate(output_tokens):
        print(f"\nOutput {i}: {text}")

    stop_monitor.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V3 inference on Neuron")
    parser.add_argument("--model-path", type=str, required=True, help="Path to HF model weights")
    parser.add_argument("--traced-model-path", type=str, default="/tmp/deepseek_v3_traced", help="Path for compiled model")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation, load from traced path")
    parser.add_argument("--tp-degree", type=int, default=32, help="Tensor parallelism degree")
    parser.add_argument("--seq-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    generate(
        model_path=args.model_path,
        traced_model_path=args.traced_model_path,
        skip_compile=args.skip_compile,
        tp_degree=args.tp_degree,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
