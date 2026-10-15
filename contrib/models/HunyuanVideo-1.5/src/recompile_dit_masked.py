"""
Recompile DiT with text_mask KV zeroing fix.
Uses same dimensions as dit_tp4_480p (IMG=3180, TXT=320).
"""

import os, sys, time, torch, json

TP_DEGREE = 4
os.environ["NEURON_RT_NUM_CORES"] = str(TP_DEGREE)
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "1"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed import ModelBuilder
from neuronx_distributed.trace.parallel_context import NxDParallelState
from neuronx_distributed.trace.functions import shard_checkpoint
from safetensors.torch import load_file

from dit_tp_wrapper import HunyuanDiTTPWrapper, create_sample_inputs, build_weight_map

MODEL_DIR = os.path.join(os.environ.get("HUNYUAN_MODELS_DIR", "./models"), "HunyuanVideo-1.5/transformer/480p_t2v")
SAVE_DIR = os.path.join(os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled"), "dit_tp4_480p")
COMPILER_WORKDIR = os.path.join(os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled"), "compiler_workdir_masked")

IMG_SEQ_LEN = 3180
TXT_SEQ_LEN = 320


def main():
    sep = "=" * 60
    print(sep)
    print("Recompile DiT with text_mask KV zeroing fix")
    print(sep)

    # Load checkpoint
    ckpt_path = os.path.join(MODEL_DIR, "diffusion_pytorch_model.safetensors")
    print(f"Loading checkpoint...")
    t0 = time.time()
    original_sd = load_file(ckpt_path)
    print(f"  Loaded {len(original_sd)} tensors in {time.time() - t0:.1f}s")

    # Remap weights
    weight_map = build_weight_map(54)
    remapped_sd = {}
    for orig_key, tp_key in weight_map.items():
        if orig_key in original_sd:
            w = original_sd[orig_key]
            if "q_norm" in orig_key or "k_norm" in orig_key:
                remapped_sd[tp_key] = w.to(torch.float32)
            else:
                remapped_sd[tp_key] = w.to(torch.bfloat16)
    print(f"Remapped {len(remapped_sd)} weights")
    del original_sd

    sample_inputs = create_sample_inputs(
        batch_size=1,
        hidden_size=2048,
        img_seq_len=IMG_SEQ_LEN,
        txt_seq_len=TXT_SEQ_LEN,
    )
    print(f"Sample inputs: img={sample_inputs[0].shape}, txt={sample_inputs[1].shape}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(COMPILER_WORKDIR, exist_ok=True)

    with NxDParallelState(world_size=TP_DEGREE, tensor_model_parallel_size=TP_DEGREE):
        model = HunyuanDiTTPWrapper(
            hidden_size=2048,
            heads_num=16,
            num_blocks=54,
            mlp_width_ratio=4.0,
            patch_size=[1, 1, 1],
            out_channels=32,
            qkv_bias=True,
            dtype=torch.bfloat16,
        )
        model.eval()

        print("\nTracing...")
        t0 = time.time()
        builder = ModelBuilder(model)
        builder.trace(args=sample_inputs, tag="dit_480p_masked")
        print(f"  Trace: {time.time() - t0:.1f}s")

        compiler_args = "--model-type=transformer -O1 --auto-cast=none"
        print(f"Compiling: {compiler_args}")
        t0 = time.time()
        nxd_model = builder.compile(
            compiler_workdir=COMPILER_WORKDIR,
            compiler_args=compiler_args,
        )
        compile_time = time.time() - t0
        print(f"  Compile: {compile_time:.1f}s")

        # Save
        nxd_model.save(os.path.join(SAVE_DIR, "nxd_model.pt"))
        weights_path = os.path.join(SAVE_DIR, "weights")
        os.makedirs(weights_path, exist_ok=True)
        shard_checkpoint(
            checkpoint=remapped_sd, model=model, serialize_path=weights_path
        )

    # Load and benchmark
    print("\nLoading to Neuron...")
    t0 = time.time()
    sharded_weights = []
    for rank in range(TP_DEGREE):
        sharded_weights.append(
            load_file(f"{weights_path}/tp{rank}_sharded_checkpoint.safetensors")
        )
    nxd_model.set_weights(sharded_weights)
    nxd_model.to_neuron()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Warmup + benchmark
    for i in range(3):
        t0 = time.time()
        _ = nxd_model(*sample_inputs)
        print(f"  Warmup {i}: {(time.time() - t0) * 1000:.0f}ms")

    times = []
    for _ in range(10):
        t0 = time.time()
        _ = nxd_model(*sample_inputs)
        times.append(time.time() - t0)

    avg = sum(times) / len(times) * 1000
    print(sep)
    print(f"RESULTS: DiT TP=4 with text_mask KV zeroing")
    print(f"  Avg latency: {avg:.1f}ms/step ({len(times)} runs)")
    print(f"  Compile time: {compile_time:.1f}s")

    results = {
        "img_seq_len": IMG_SEQ_LEN,
        "txt_seq_len": TXT_SEQ_LEN,
        "compile_time_s": compile_time,
        "avg_ms": avg,
        "min_ms": min(times) * 1000,
        "fix": "text_mask KV zeroing",
    }
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
