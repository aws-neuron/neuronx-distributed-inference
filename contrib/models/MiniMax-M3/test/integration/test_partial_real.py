"""Partial MiniMax-M3 (first N layers, real weights) compile + TTFT/ITL on Trn2.

Loads only the first N layers from the real MiniMax-M3 checkpoint and runs
end-to-end on Trn2 hardware. Validates the modeling code against real
weights while staying within HBM limits.
"""
import os, sys, time, json
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TEST_DIR.parent.parent / "src"))

import torch
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_minimax_m3 import (
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
    MiniMaxM3NeuronConfig,
)

MODEL_PATH = "/mnt/data/models/MiniMax-M3"
COMPILED_PATH = "/mnt/scratch/neuron_models/MiniMax-M3-partial"
NUM_LAYERS = int(os.environ.get("M3_NUM_LAYERS", "6"))  # 3 dense + 3 MoE
TP_DEGREE = int(os.environ.get("M3_TP_DEGREE", "32"))
LNC = 2
SEQ_LEN = int(os.environ.get("M3_SEQ_LEN", "512"))

print(f"MODEL={MODEL_PATH}")
print(f"COMPILED={COMPILED_PATH}")
print(f"NUM_LAYERS={NUM_LAYERS}  TP={TP_DEGREE}  LNC={LNC}  SEQ_LEN={SEQ_LEN}")
os.makedirs(COMPILED_PATH, exist_ok=True)
os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(LNC)

neuron_config = MiniMaxM3NeuronConfig(
    tp_degree=TP_DEGREE,
    logical_nc_config=LNC,
    batch_size=1,
    seq_len=SEQ_LEN,
    max_context_length=SEQ_LEN,
    torch_dtype=torch.bfloat16,
)

with open(f"{MODEL_PATH}/config.json") as f:
    raw = json.load(f)
text_cfg = dict(raw["text_config"])
text_cfg["model_type"] = "minimax_m3"
text_cfg["architectures"] = ["MiniMaxM3SparseForCausalLM"]
text_cfg["tie_word_embeddings"] = text_cfg.get("tie_word_embeddings", False)
# Truncate to first NUM_LAYERS layers
text_cfg["num_hidden_layers"] = NUM_LAYERS
text_cfg["moe_layer_freq"] = text_cfg["moe_layer_freq"][:NUM_LAYERS]
hf_cfg = PretrainedConfig(**text_cfg)
hf_cfg._name_or_path = MODEL_PATH

config = MiniMaxM3InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(hf_config=hf_cfg),
)
config._name_or_path = MODEL_PATH

print(f"\nConfig:")
print(f"  num_hidden_layers={config.num_hidden_layers}")
print(f"  num_local_experts={config.num_local_experts}")
print(f"  hidden_size={config.hidden_size}")
print(f"  moe_layer_freq={config.moe_layer_freq}")

if not (Path(COMPILED_PATH) / "model.pt").exists():
    print("\n[compile] starting...")
    t0 = time.perf_counter()
    model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
    model.compile(COMPILED_PATH)
    print(f"[compile] done in {time.perf_counter()-t0:.1f}s")
else:
    print(f"[compile] reusing {COMPILED_PATH}")

print("\n[load] loading weights to Neuron devices...")
t0 = time.perf_counter()
model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH)
print(f"[load] done in {time.perf_counter()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
print(f"\nprompt={prompt!r}  tokens={input_ids.shape[1]}")

def _forward(ids):
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        return model(ids, position_ids=pos)

print("\n[warmup] 3 prefill calls...")
for _ in range(3):
    _forward(input_ids)

print("[ttft] measuring prefill latency (5 runs)...")
ttft = []
for _ in range(5):
    t0 = time.perf_counter()
    _forward(input_ids)
    ttft.append((time.perf_counter()-t0)*1000)
print(f"[ttft] mean={sum(ttft)/len(ttft):.2f} ms  vals={[f'{x:.2f}' for x in ttft]}")

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
tps = 1000.0/avg_itl if avg_itl > 0 else 0.0
print(f"[itl]  mean={avg_itl:.2f} ms/token  throughput={tps:.2f} tok/s")

print(f"\ngenerated: {tok.decode(gen[0], skip_special_tokens=True)!r}")
