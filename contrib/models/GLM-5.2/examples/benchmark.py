import os, sys, time, json

os.environ["UNSAFE_FP8FNCAST"] = "1"
_orig_makedirs = os.makedirs
def _safe_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(name, mode=mode, exist_ok=True)
os.makedirs = _safe_makedirs
import shutil
_orig_rmtree = shutil.rmtree
def _safe_rmtree(path, ignore_errors=False, onerror=None, **kw):
    return _orig_rmtree(path, ignore_errors=True, **kw)
shutil.rmtree = _safe_rmtree

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from modeling_glm5 import NeuronGLM5ForCausalLM, GLM5InferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from transformers import PreTrainedTokenizerFast

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/GLM-5.2-FP8"))
COMPILED = os.environ.get("COMPILED_MODEL_PATH", os.path.expanduser("~/glm52_compiled"))
SEQ = int(os.environ.get("SEQ", "2048"))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

with open(f"{MODEL_PATH}/config.json") as f:
    hf_config = json.load(f)

neuron_config = MoENeuronConfig(
    tp_degree=64, batch_size=1, seq_len=SEQ, n_active_tokens=SEQ,
    torch_dtype=torch.bfloat16, fused_qkv=False,
    qkv_kernel_enabled=False, qkv_nki_kernel_enabled=False,
    moe_fused_nki_kernel_enabled=True, expert_mlp_nki_kernel_enabled=False,
    mlp_kernel_enabled=True, quantized=True, quantization_dtype="f8e4m3",
    quantized_checkpoints_path=MODEL_PATH,
    modules_to_not_convert=["lm_head","self_attn","shared_expert","layers.0.mlp","layers.1.mlp","layers.2.mlp"],
    layer_boundary_markers=True, weights_to_skip_layout_optimization=[".*"],
    logical_nc_config=2, save_sharded_checkpoint=True, local_ranks_size=64,
    flash_decoding_enabled=False, on_cpu=False,
)
def load_config(c):
    for k, v in hf_config.items(): setattr(c, k, v)
    c.dsa_enabled = False  # full MLA (indexer is no-op for seq_len <= 2048)

log("building config + model")
config = GLM5InferenceConfig(neuron_config=neuron_config, load_config=load_config)
model = NeuronGLM5ForCausalLM(MODEL_PATH, config)
log("loading weights onto 64 NeuronCores...")
t0 = time.time()
model.load(COMPILED)
log(f"load done in {time.time()-t0:.0f}s")

tok = PreTrainedTokenizerFast(tokenizer_file=f"{MODEL_PATH}/tokenizer.json",
                              eos_token="<|endoftext|>", pad_token="<|endoftext|>")
wrapped = HuggingFaceGenerationAdapter(model)

def gen(prompt, max_new_tokens):
    pad_len = SEQ - max_new_tokens
    inputs = tok(prompt, return_tensors="pt", padding="max_length", max_length=pad_len)
    with torch.no_grad():
        out = wrapped.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                               max_new_tokens=max_new_tokens, do_sample=False)
    return out

PROMPT = "The capital of France is"
log("warmup (1 tok)")
gen(PROMPT, 1)

# TTFT: time to first token (prefill)
ttfts = []
for _ in range(3):
    t0 = time.time(); gen(PROMPT, 1); ttfts.append((time.time()-t0)*1000)
ttft = sum(ttfts)/len(ttfts)
log(f"TTFT avg = {ttft:.1f} ms  (runs: {[round(x) for x in ttfts]})")

# ITL/TPOT + throughput over N tokens
N = 64
t0 = time.time()
out = gen(PROMPT, N)
elapsed = (time.time()-t0)*1000
itl = (elapsed - ttft)/(N-1)
log(f"E2E {elapsed:.0f} ms for {N} new tok | ITL/TPOT ~= {itl:.1f} ms/tok | throughput {1000*N/elapsed:.2f} tok/s")
text = tok.decode(out[0], skip_special_tokens=True)
log("SAMPLE: " + text[-300:].replace(chr(10), " "))
log("BENCH_DONE")
