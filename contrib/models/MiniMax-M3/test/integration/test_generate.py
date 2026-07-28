"""Correctly drive M3 generate using NxDI's HF adapter (auto KV cache + decode dispatch)."""
import os, sys, time, json
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/neuronx-distributed-inference/contrib/models/MiniMax-M3/src")

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.configuration_utils import PretrainedConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from modeling_minimax_m3 import (
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
    MiniMaxM3NeuronConfig,
)

MODEL_PATH = "/mnt/nvme/models/MiniMax-M3"
COMPILED_PATH = "/mnt/scratch/neuron_models/MiniMax-M3-full-v6-stride2"
BATCH = 32
SEQ_LEN = 512

os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    "/mnt/scratch/nxd_compile/MiniMax-M3-full-v6-stride2",
)

neuron_config = MiniMaxM3NeuronConfig(
    tp_degree=64, ep_degree=1, logical_nc_config=2,
    batch_size=BATCH, max_batch_size=BATCH, ctx_batch_size=1,
    tkg_batch_size=BATCH, seq_len=SEQ_LEN, n_active_tokens=128,
    torch_dtype=torch.bfloat16, capacity_factor=1.0, glu_mlp=True,
    moe_ep_degree=64, moe_tp_degree=1,
    context_encoding_buckets=[SEQ_LEN],
    fused_qkv=True, save_sharded_checkpoint=True,
    router_config={"act_fn": "sigmoid", "dtype": "float32"},
    blockwise_matmul_config={
        "use_shard_on_block_dynamic_while": True,
        "block_sharding_strategy": "PING_PONG",
    },
)

with open(f"{MODEL_PATH}/config.json") as f:
    raw = json.load(f)
text_cfg = dict(raw["text_config"])
text_cfg["model_type"] = "minimax_m3"
text_cfg["architectures"] = ["MiniMaxM3SparseForCausalLM"]
text_cfg["tie_word_embeddings"] = text_cfg.get("tie_word_embeddings", False)
hf_cfg = PretrainedConfig(**text_cfg)
hf_cfg._name_or_path = MODEL_PATH

config = MiniMaxM3InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(hf_config=hf_cfg),
)
config._name_or_path = MODEL_PATH

print(f"[load] loading cached NEFF + pre-sharded weights...")
t0 = time.perf_counter()
model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH, skip_warmup=False)
print(f"[load] done in {time.perf_counter()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Use a short prompt with padding to seq_len. Important: pad with attention_mask=0
# at left, so the real prompt is at the right, and decode starts from there.
prompt = "The capital of France is"
# Pad LEFT so prompt is at the right edge — then decode picks up at position seq_len
inputs = tok([prompt] * BATCH, return_tensors="pt", padding="max_length", max_length=64)
print(f"input_ids shape={tuple(inputs.input_ids.shape)} attn_mask shape={tuple(inputs.attention_mask.shape)}")

# Try via HF GenerationAdapter which does the right dispatch
print("\n[generate] using HF generation adapter...")
adapter = HuggingFaceGenerationAdapter(model)
gen_config = GenerationConfig(
    max_new_tokens=20,
    do_sample=False,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)

t0 = time.perf_counter()
with torch.no_grad():
    output_ids = adapter.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=gen_config,
    )
elapsed = (time.perf_counter() - t0) * 1000
print(f"[generate] total={elapsed:.1f}ms for {output_ids.shape[1] - inputs.input_ids.shape[1]} new tokens")

# Decode first 3 outputs
for i in range(min(3, output_ids.shape[0])):
    print(f"\n--- sample {i} ---")
    print(tok.decode(output_ids[i], skip_special_tokens=True))

print("\nDONE.")
