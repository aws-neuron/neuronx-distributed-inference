"""
Trace byT5 text encoder with torch_neuronx.trace().

byT5 architecture: T5Stack encoder-only
  - d_model=1472, d_ff=3584, d_kv=64, 6 heads, 12 layers
  - vocab=384 (byte-level + special tokens for color/font)
  - Input: (B, 256) int token IDs + (B, 256) float attention mask
  - Output: (B, 256, 1472) hidden states

The byT5 model + Glyph-SDXL-v2 fine-tuned weights are loaded, then the
T5Stack encoder is traced with torch_neuronx.trace() on a single NeuronCore.

After tracing, we also trace the ByT5Mapper (projects 1472 -> 2048 for DiT).

Launch:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python ./trace_byt5.py

Environment:
    NEURON_RT_VIRTUAL_CORE_SIZE=2  (default on LNC=2 instance)
"""

import sys
import os
import time
import json
import torch
import torch.nn.functional as F
import torch_neuronx

sys.path.insert(0, os.environ.get("HUNYUAN_REPO_DIR", "./HunyuanVideo-1.5"))
os.environ["HUNYUAN_ATTN_MODE"] = "torch"

MODELS_DIR = os.environ.get("HUNYUAN_MODELS_DIR", "./models")
COMPILED_DIR = os.environ.get("HUNYUAN_COMPILED_DIR", "./compiled")
os.makedirs(COMPILED_DIR, exist_ok=True)


def cosine_similarity(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


class ByT5EncoderWrapper(torch.nn.Module):
    """Wrapper for tracing: takes input_ids and attention_mask, returns hidden states."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        # T5Stack returns BaseModelOutput; extract last_hidden_state
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class ByT5MapperWrapper(torch.nn.Module):
    """Wrapper for tracing the ByT5Mapper (1472 -> 2048 projection)."""

    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper

    def forward(self, x):
        return self.mapper(x)


def main():
    dtype = torch.bfloat16

    print("=" * 60)
    print("Trace byT5 Encoder + ByT5Mapper")
    print("=" * 60)

    # ---- Load byT5 encoder ----
    print("\n--- Loading byT5 encoder ---")
    t0 = time.time()

    from hyvideo.models.text_encoders.byT5 import load_glyph_byT5_v2

    byt5_args = dict(
        byT5_google_path=f"{MODELS_DIR}/byt5-small",
        byT5_ckpt_path=f"{MODELS_DIR}/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
        multilingual_prompt_format_color_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/color_idx.json",
        multilingual_prompt_format_font_path=f"{MODELS_DIR}/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
        byt5_max_length=256,
    )
    byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=torch.device("cpu"))
    byt5_model = byt5_kwargs["byt5_model"]  # T5Stack
    byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]

    total_params = sum(p.numel() for p in byt5_model.parameters())
    print(f"Loaded in {time.time() - t0:.1f}s, {total_params / 1e6:.1f}M params")
    print(f"Vocab size: {len(byt5_tokenizer)}")

    # ---- Load ByT5Mapper (from DiT checkpoint, without loading full DiT) ----
    print("\n--- Loading ByT5Mapper ---")

    # Import ByT5Mapper directly (no diffusers dependency)
    from hyvideo.models.text_encoders.byT5 import ByT5Mapper

    t0 = time.time()
    # ByT5Mapper config for 480p_t2v: in_dim=1472, out_dim=1472, hidden=8192, out_dim1=2048
    # Actually: in_dim=1472, out_dim=1472, hidden_dim=1472*4=5888, out_dim1=2048
    # Let's find the exact config from the checkpoint keys
    import safetensors.torch as st

    dit_path = f"{MODELS_DIR}/HunyuanVideo-1.5/transformer/480p_t2v"
    # Load just the byt5_in weights from the safetensors file
    import glob

    st_files = sorted(glob.glob(f"{dit_path}/*.safetensors"))

    byt5_in_state = {}
    for sf in st_files:
        with st.safe_open(sf, framework="pt") as f:
            for key in f.keys():
                if key.startswith("byt5_in."):
                    byt5_in_state[key.replace("byt5_in.", "")] = f.get_tensor(key)

    if not byt5_in_state:
        print("ERROR: No byt5_in weights found in checkpoint!")
        sys.exit(1)

    # Determine dims from weight shapes
    fc1_weight = byt5_in_state["fc1.weight"]  # (hidden_dim, in_dim)
    fc2_weight = byt5_in_state["fc2.weight"]  # (out_dim, hidden_dim)
    fc3_weight = byt5_in_state["fc3.weight"]  # (out_dim1, out_dim)
    in_dim = fc1_weight.shape[1]
    hidden_dim = fc1_weight.shape[0]
    out_dim = fc2_weight.shape[0]
    out_dim1 = fc3_weight.shape[0]
    print(
        f"ByT5Mapper dims: in={in_dim}, hidden={hidden_dim}, out={out_dim}, out1={out_dim1}"
    )

    byt5_mapper = ByT5Mapper(
        in_dim, out_dim, hidden_dim, out_dim1, use_residual=(in_dim == out_dim)
    )
    byt5_mapper.load_state_dict(byt5_in_state)
    byt5_mapper = byt5_mapper.to(dtype)
    mapper_params = sum(p.numel() for p in byt5_mapper.parameters())
    print(
        f"ByT5Mapper loaded in {time.time() - t0:.1f}s, {mapper_params / 1e6:.1f}M params"
    )

    # ---- CPU reference ----
    print("\n--- CPU reference ---")
    byt5_model.eval()
    byt5_mapper.eval()

    test_prompt = 'A beautiful sunset over the ocean with text "Hello World".'
    tokens = byt5_tokenizer(
        test_prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids  # (1, 256) int64
    attention_mask = tokens.attention_mask.float()  # (1, 256) float32

    with torch.no_grad():
        cpu_encoder_out = byt5_model(input_ids, attention_mask=attention_mask)[0]
        cpu_mapper_out = byt5_mapper(cpu_encoder_out.to(dtype))

    print(f"CPU encoder output: {cpu_encoder_out.shape}, dtype={cpu_encoder_out.dtype}")
    print(f"CPU mapper output:  {cpu_mapper_out.shape}, dtype={cpu_mapper_out.dtype}")

    # ---- Trace byT5 encoder ----
    print("\n--- Tracing byT5 encoder ---")
    wrapper = ByT5EncoderWrapper(byt5_model)
    wrapper.eval()

    # Example inputs for tracing
    example_ids = torch.zeros(1, 256, dtype=torch.long)
    example_mask = torch.ones(1, 256, dtype=torch.float32)

    t_trace = time.time()
    traced_encoder = torch_neuronx.trace(
        wrapper,
        (example_ids, example_mask),
        compiler_args=[
            "--auto-cast",
            "matmult",
            "--model-type=transformer",
        ],
    )
    trace_time = time.time() - t_trace
    print(f"Encoder traced in {trace_time:.1f}s")

    # Save traced encoder
    encoder_path = f"{COMPILED_DIR}/byt5_encoder.pt"
    torch_neuronx.async_load(traced_encoder)  # Enable async loading
    traced_encoder.save(encoder_path)
    print(f"Saved traced encoder to {encoder_path}")

    # ---- Validate encoder ----
    print("\n--- Validating traced encoder ---")
    neuron_encoder_out = traced_encoder(input_ids, attention_mask)
    cos_sim = cosine_similarity(cpu_encoder_out, neuron_encoder_out)
    mae = (cpu_encoder_out.float() - neuron_encoder_out.float()).abs().mean().item()
    max_err = (cpu_encoder_out.float() - neuron_encoder_out.float()).abs().max().item()
    print(f"Encoder: cos_sim={cos_sim:.6f}, mae={mae:.6f}, max_err={max_err:.6f}")

    # Benchmark encoder
    print("\n--- Benchmarking traced encoder ---")
    # Warmup
    for _ in range(5):
        _ = traced_encoder(input_ids, attention_mask)

    times = []
    for _ in range(50):
        t0 = time.time()
        _ = traced_encoder(input_ids, attention_mask)
        times.append(time.time() - t0)

    avg_ms = sum(times) * 1000 / len(times)
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(
        f"Encoder latency: avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms (50 runs)"
    )

    # ---- Trace ByT5Mapper ----
    print("\n--- Tracing ByT5Mapper ---")
    mapper_wrapper = ByT5MapperWrapper(byt5_mapper)
    mapper_wrapper.eval()

    # Example: (1, 256, 1472) bf16 input
    example_mapper_input = torch.randn(1, 256, 1472, dtype=dtype)

    t_trace2 = time.time()
    traced_mapper = torch_neuronx.trace(
        mapper_wrapper,
        (example_mapper_input,),
        compiler_args=[
            "--auto-cast",
            "matmult",
        ],
    )
    trace_time2 = time.time() - t_trace2
    print(f"Mapper traced in {trace_time2:.1f}s")

    # Save traced mapper
    mapper_path = f"{COMPILED_DIR}/byt5_mapper.pt"
    traced_mapper.save(mapper_path)
    print(f"Saved traced mapper to {mapper_path}")

    # ---- Validate mapper ----
    print("\n--- Validating traced mapper ---")
    # Feed encoder output through mapper
    mapper_input = neuron_encoder_out.to(dtype)
    neuron_mapper_out = traced_mapper(mapper_input)
    cos_sim_mapper = cosine_similarity(cpu_mapper_out, neuron_mapper_out)
    mae_mapper = (
        (cpu_mapper_out.float() - neuron_mapper_out.float()).abs().mean().item()
    )
    max_err_mapper = (
        (cpu_mapper_out.float() - neuron_mapper_out.float()).abs().max().item()
    )
    print(
        f"Mapper: cos_sim={cos_sim_mapper:.6f}, mae={mae_mapper:.6f}, max_err={max_err_mapper:.6f}"
    )

    # Benchmark mapper
    print("\n--- Benchmarking traced mapper ---")
    for _ in range(5):
        _ = traced_mapper(mapper_input)

    mapper_times = []
    for _ in range(50):
        t0 = time.time()
        _ = traced_mapper(mapper_input)
        mapper_times.append(time.time() - t0)

    mapper_avg_ms = sum(mapper_times) * 1000 / len(mapper_times)
    mapper_min_ms = min(mapper_times) * 1000
    print(
        f"Mapper latency: avg={mapper_avg_ms:.2f}ms, min={mapper_min_ms:.2f}ms (50 runs)"
    )

    # ---- E2E: encoder + mapper ----
    print("\n--- E2E: encoder -> cast bf16 -> mapper ---")
    e2e_times = []
    for _ in range(50):
        t0 = time.time()
        enc_out = traced_encoder(input_ids, attention_mask)
        enc_bf16 = enc_out.to(dtype)
        map_out = traced_mapper(enc_bf16)
        e2e_times.append(time.time() - t0)

    e2e_avg = sum(e2e_times) * 1000 / len(e2e_times)
    e2e_min = min(e2e_times) * 1000
    print(f"E2E latency: avg={e2e_avg:.2f}ms, min={e2e_min:.2f}ms (50 runs)")
    print(f"E2E output shape: {map_out.shape}, dtype={map_out.dtype}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"byT5 Encoder:")
    print(f"  Params:      {total_params / 1e6:.1f}M")
    print(f"  Trace time:  {trace_time:.1f}s")
    print(f"  Latency:     {avg_ms:.2f}ms (avg, 50 runs)")
    print(f"  Accuracy:    cos_sim={cos_sim:.6f}")
    print(f"  Saved to:    {encoder_path}")
    print(f"")
    print(f"ByT5Mapper:")
    print(f"  Params:      {mapper_params / 1e6:.1f}M")
    print(f"  Trace time:  {trace_time2:.1f}s")
    print(f"  Latency:     {mapper_avg_ms:.2f}ms (avg, 50 runs)")
    print(f"  Accuracy:    cos_sim={cos_sim_mapper:.6f}")
    print(f"  Saved to:    {mapper_path}")
    print(f"")
    print(f"E2E (encoder + mapper):")
    print(f"  Latency:     {e2e_avg:.2f}ms (avg, 50 runs)")


    # Save results
    results = {
        "task": "017-byt5-trace",
        "encoder": {
            "params_M": round(total_params / 1e6, 1),
            "trace_time_s": round(trace_time, 1),
            "latency_avg_ms": round(avg_ms, 2),
            "latency_min_ms": round(min_ms, 2),
            "cosine_similarity": round(cos_sim, 6),
            "mae": round(mae, 6),
            "max_err": round(max_err, 6),
            "saved_path": encoder_path,
        },
        "mapper": {
            "params_M": round(mapper_params / 1e6, 1),
            "trace_time_s": round(trace_time2, 1),
            "latency_avg_ms": round(mapper_avg_ms, 2),
            "latency_min_ms": round(mapper_min_ms, 2),
            "cosine_similarity": round(cos_sim_mapper, 6),
            "mae": round(mae_mapper, 6),
            "max_err": round(max_err_mapper, 6),
            "saved_path": mapper_path,
        },
        "e2e": {
            "latency_avg_ms": round(e2e_avg, 2),
            "latency_min_ms": round(e2e_min, 2),
        },
    }
    results_path = f"{COMPILED_DIR}/byt5_trace_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
