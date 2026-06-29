"""Full MiniMax-M3 (60 layers) with M2-style hybrid sharding.

Key config (matches the M2 contrib PR pattern):
  - TP=64, EP=1 (outer), moe_tp=1, moe_ep=64
  - LNC=2 → 64 logical cores
  - batch_size>=32 (required by NxDI EP path: num_experts/top_k = 128/4 = 32)
  - blockwise_matmul: use_shard_on_block_dynamic_while, PING_PONG
  - fused_qkv=True (avoids per-rank activation blowup)
  - save_sharded_checkpoint=True (cache 854GB shards)
"""
import os, sys, time, json
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/neuronx-distributed-inference/contrib/models/MiniMax-M3/src")

import torch
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_minimax_m3 import (
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
    MiniMaxM3NeuronConfig,
)

MODEL_PATH = os.environ.get("M3_MODEL_PATH", "/mnt/nvme/models/MiniMax-M3")
COMPILED_PATH = os.environ.get("M3_COMPILED_PATH", "/mnt/scratch/neuron_models/MiniMax-M3-full-v6-stride2")
TP_DEGREE = int(os.environ.get("M3_TP_DEGREE", "64"))
MOE_EP = int(os.environ.get("M3_MOE_EP", "64"))
MOE_TP = int(os.environ.get("M3_MOE_TP", "1"))
LNC = 2
BATCH = int(os.environ.get("M3_BATCH", "32"))   # >= 32 required for EP
SEQ_LEN = int(os.environ.get("M3_SEQ_LEN", "512"))

print(f"MODEL={MODEL_PATH}")
print(f"COMPILED={COMPILED_PATH}")
print(f"TP={TP_DEGREE} EP_outer=1 moe_tp={MOE_TP} moe_ep={MOE_EP} batch={BATCH} seq_len={SEQ_LEN} LNC={LNC}")
os.makedirs(COMPILED_PATH, exist_ok=True)
os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(LNC)
# Isolate compile workdir for parallel-safety
os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    os.path.join("/mnt/scratch/nxd_compile", os.path.basename(COMPILED_PATH.rstrip("/"))),
)

neuron_config = MiniMaxM3NeuronConfig(
    tp_degree=TP_DEGREE,
    ep_degree=1,                    # outer EP=1 (M2 pattern)
    logical_nc_config=LNC,
    batch_size=BATCH,
    max_batch_size=BATCH,
    ctx_batch_size=1,
    tkg_batch_size=BATCH,
    seq_len=SEQ_LEN,
    n_active_tokens=128,
    torch_dtype=torch.bfloat16,
    capacity_factor=1.0,
    glu_mlp=True,
    moe_ep_degree=MOE_EP,
    moe_tp_degree=MOE_TP,
    context_encoding_buckets=[SEQ_LEN],
    fused_qkv=False,                # try non-fused like MiMo-V2 contrib
    save_sharded_checkpoint=True,
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

print(f"\nConfig:")
print(f"  num_hidden_layers={config.num_hidden_layers}")
print(f"  num_local_experts={config.num_local_experts}  experts_per_tok={config.num_experts_per_tok}")
print(f"  hidden_size={config.hidden_size}")
print(f"  moe_intermediate_size={config.moe_intermediate_size}  intermediate_size={config.intermediate_size}")
print(f"  moe_intermediate_pad_size={config.moe_intermediate_pad_size}")
print(f"  fused_qkv={config.neuron_config.fused_qkv}")
print(f"  moe_tp/ep={config.neuron_config.moe_tp_degree}/{config.neuron_config.moe_ep_degree}")

# Compile
if not (Path(COMPILED_PATH) / "model.pt").exists():
    print("\n[compile] starting (full 60-layer M3)...")
    t0 = time.perf_counter()
    model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
    model.compile(COMPILED_PATH)
    print(f"[compile] done in {time.perf_counter()-t0:.1f}s")
else:
    print(f"[compile] reusing {COMPILED_PATH}")

# Load
print("\n[load] loading weights to Neuron...")
t0 = time.perf_counter()
model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH, skip_warmup=False)
print(f"[load] done in {time.perf_counter()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Batch input — with batch_size=32 we replicate one prompt
prompt = "The capital of France is"
inputs = tok([prompt] * BATCH, return_tensors="pt", padding="max_length", max_length=SEQ_LEN)
input_ids = inputs.input_ids
print(f"\nprompt={prompt!r}  shape={tuple(input_ids.shape)}")

def _forward(ids):
    pos = torch.arange(ids.shape[1]).unsqueeze(0).expand(ids.shape[0], -1)
    with torch.no_grad():
        return model(ids, position_ids=pos)

# Warmup
print("\n[warmup] 3 prefill calls...")
for _ in range(3):
    _forward(input_ids)

# TTFT (per batch)
print("[ttft] measuring prefill latency (5 runs)...")
ttft = []
for _ in range(5):
    t0 = time.perf_counter()
    _forward(input_ids)
    ttft.append((time.perf_counter()-t0)*1000)
print(f"[ttft] mean={sum(ttft)/len(ttft):.2f} ms  vals={[f'{x:.2f}' for x in ttft]}")

# ITL (decode latency per token)
print("\n[itl] measuring decode latency (20 tokens)...")
gen = input_ids.clone()
_forward(gen)

itl = []
for step in range(20):
    t0 = time.perf_counter()
    out = _forward(gen)
    itl.append((time.perf_counter()-t0)*1000)
    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, tuple) else out)
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    gen = torch.cat([gen, nxt], dim=-1)

avg_itl = sum(itl)/len(itl)
tps = (BATCH * 1000.0) / avg_itl if avg_itl > 0 else 0.0
print(f"[itl]  mean={avg_itl:.2f} ms/step  throughput={tps:.2f} tok/s (across batch={BATCH})")

# First generation
print(f"\ngenerated (item 0): {tok.decode(gen[0], skip_special_tokens=True)[:300]!r}")
