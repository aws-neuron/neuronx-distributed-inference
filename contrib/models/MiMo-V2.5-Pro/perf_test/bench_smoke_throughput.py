#!/usr/bin/env python3
"""Throughput benchmark for MiMo-V2.5-Pro via the NxDI direct (smoke) path.

Loads the pre-compiled seq512 NEFF, then runs several batched generate()
calls and reports aggregate throughput (BS * new_tokens / wall_time) so it
can be compared apples-to-apples with the vLLM `output token throughput`.

Env:
  MIMO_V25_PRO_MODEL_PATH, MIMO_V25_PRO_COMPILED_PATH  (paths)
  BATCH_SIZE=48  MAX_NEW_TOKENS=120  INPUT_LEN=360  N_ITERS=3
Set NEURON_RT_INSPECT_DEVICE_PROFILE=<dir> before running to also capture a
device profile (dumps *.ntff under that dir).
"""
import os
import sys
import time

os.environ.setdefault("XLA_HANDLE_SPECIAL_SCALAR", "1")
os.environ.setdefault("UNSAFE_FP8FNCAST", "1")

MODEL_PATH = os.environ.get("MIMO_V25_PRO_MODEL_PATH",
                            "/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8")
COMPILED_PATH = os.environ.get(
    "MIMO_V25_PRO_COMPILED_PATH",
    "/opt/dlami/nvme/models/compiled/mimo_v2_5_pro_bs48_moetp1_ep64_fp8moe_bf16attn_seq512/")
TP_DEGREE = int(os.environ.get("TP_DEGREE", "64"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))
MOE_TP = int(os.environ.get("MOE_TP", "1"))
MOE_EP = int(os.environ.get("MOE_EP", "64"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "120"))
INPUT_LEN = int(os.environ.get("INPUT_LEN", "360"))
N_ITERS = int(os.environ.get("N_ITERS", "3"))

os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))))


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter, load_pretrained_config)

    contrib_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
    # allow running from /tmp too
    for cand in (contrib_src,
                 "/home/ubuntu/ndi-pr150-MiMo/contrib/models/MiMo-V2.5-Pro/src"):
        if os.path.isdir(cand):
            sys.path.insert(0, os.path.abspath(cand))
            break
    from modeling_mimo_v2 import MiMoV2InferenceConfig, NeuronMiMoV2ForCausalLM

    print(f"[bench] MODEL={MODEL_PATH}")
    print(f"[bench] COMPILED={COMPILED_PATH}")
    print(f"[bench] BS={BATCH_SIZE} SEQ={SEQ_LEN} INPUT_LEN={INPUT_LEN} "
          f"MAX_NEW={MAX_NEW_TOKENS} N_ITERS={N_ITERS}")

    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE, ep_degree=1, logical_nc_config=2,
        batch_size=BATCH_SIZE, max_batch_size=BATCH_SIZE, ctx_batch_size=1,
        tkg_batch_size=BATCH_SIZE, seq_len=SEQ_LEN, n_active_tokens=128,
        torch_dtype="bfloat16", capacity_factor=1.0, glu_mlp=True,
        moe_ep_degree=MOE_EP, moe_tp_degree=MOE_TP,
        context_encoding_buckets=[SEQ_LEN],
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config={"use_shard_on_block_dynamic_while": True,
                                 "block_sharding_strategy": "PING_PONG"},
        save_sharded_checkpoint=True, quantized=True,
        quantized_checkpoints_path=MODEL_PATH, quantization_dtype="f8e4m3",
        quantization_type="blockwise_symmetric",
        quantization_block_axis=[1, 2], quantization_block_size=[128, 128],
        modules_to_not_convert=["embed_tokens", "lm_head", "norm", "router",
                                "o_proj", "q_proj", "k_proj", "v_proj"])

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_config))

    t0 = time.time()
    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    print(f"[bench] instantiated in {time.time()-t0:.1f}s")
    t0 = time.time()
    model.load(COMPILED_PATH, skip_warmup=False)
    print(f"[bench] loaded in {time.time()-t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    adapter = HuggingFaceGenerationAdapter(model)

    # Build a prompt of ~INPUT_LEN tokens, replicated to fill the batch.
    base = "Please write a detailed explanation about large language models. "
    prompt = (base * 40)
    enc = tok([prompt], return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0][:INPUT_LEN]
    prompt = tok.decode(ids)
    inputs = tok([prompt] * BATCH_SIZE, return_tensors="pt", padding="max_length",
                 max_length=INPUT_LEN, truncation=True)
    gen = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS,
                           do_sample=False,
                           pad_token_id=getattr(tok, "pad_token_id", None) or tok.eos_token_id)
    print(f"[bench] input_ids.shape={tuple(inputs['input_ids'].shape)}")

    # Separate prefill (context-encoding, CTE) from decode (token-generation,
    # TKG). A max_new_tokens=1 call is prefill + one decode step ~= TTFT; the
    # full call is prefill + (MAX_NEW_TOKENS-1) decode steps. Subtracting the
    # two isolates the steady-state per-step decode cost, from which we derive
    # decode throughput. Prefill throughput = (BS * INPUT_LEN) / prefill_time.
    gen1 = GenerationConfig(max_new_tokens=1, min_new_tokens=1, do_sample=False,
                            pad_token_id=getattr(tok, "pad_token_id", None) or tok.eos_token_id)

    def _run(gcfg):
        t0 = time.time()
        adapter.generate(input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"],
                         generation_config=gcfg)
        return time.time() - t0

    print("[bench] warmup...")
    _run(gen)  # warmup, not counted

    prefill_times, full_times = [], []
    for it in range(1, N_ITERS + 1):
        t_prefill = _run(gen1)         # prefill + 1 decode step ~= TTFT
        t_full = _run(gen)             # prefill + (MAX_NEW_TOKENS-1) decode steps
        prefill_times.append(t_prefill)
        full_times.append(t_full)
        # decode-only time for this iter = full - prefill, over (MAX_NEW_TOKENS-1) steps
        decode_time = t_full - t_prefill
        decode_steps = MAX_NEW_TOKENS - 1
        prefill_in = BATCH_SIZE * INPUT_LEN
        decode_out = BATCH_SIZE * decode_steps
        print(f"[bench] iter{it}: "
              f"prefill(TTFT)={t_prefill:.3f}s ({prefill_in/t_prefill:,.0f} in-tok/s) | "
              f"full={t_full:.2f}s | "
              f"decode={decode_time:.2f}s ({decode_out/decode_time:.1f} out-tok/s, "
              f"per_stream={decode_steps/decode_time:.2f} tok/s) | "
              f"end2end_out={BATCH_SIZE*MAX_NEW_TOKENS/t_full:.1f} tok/s")

    n = len(full_times)
    avg_prefill = sum(prefill_times) / n
    avg_full = sum(full_times) / n
    avg_decode = avg_full - avg_prefill
    prefill_in = BATCH_SIZE * INPUT_LEN
    decode_out = BATCH_SIZE * (MAX_NEW_TOKENS - 1)
    print(f"[bench] === AVG over {n} iters (BS={BATCH_SIZE}, in={INPUT_LEN}, "
          f"out={MAX_NEW_TOKENS}) ===")
    print(f"[bench]   PREFILL: {avg_prefill:.3f}s  "
          f"throughput={prefill_in/avg_prefill:,.0f} input-tok/s  "
          f"(TTFT per request ~= {avg_prefill*1000:.0f} ms)")
    print(f"[bench]   DECODE : {avg_decode:.2f}s for {MAX_NEW_TOKENS-1} steps  "
          f"throughput={decode_out/avg_decode:.1f} output-tok/s  "
          f"per_stream={(MAX_NEW_TOKENS-1)/avg_decode:.2f} tok/s")
    print(f"[bench]   END2END: {avg_full:.2f}s  "
          f"output_throughput={BATCH_SIZE*MAX_NEW_TOKENS/avg_full:.1f} tok/s")
    print("[bench] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
