import os, sys, time, json, gc

# --- env / SDK 2.29 workarounds (from reference README) ---
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

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/GLM-5.2-FP8"))
COMPILED_MODEL_PATH = os.environ.get("COMPILED_MODEL_PATH", os.path.expanduser("~/glm52_compiled"))
SEQ = int(os.environ.get("SEQ", "2048"))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

with open(f"{MODEL_PATH}/config.json") as f:
    hf_config = json.load(f)

neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=SEQ,
    n_active_tokens=SEQ,
    torch_dtype=torch.bfloat16,
    fused_qkv=False,
    qkv_kernel_enabled=False,
    qkv_nki_kernel_enabled=False,
    moe_fused_nki_kernel_enabled=True,
    expert_mlp_nki_kernel_enabled=False,
    mlp_kernel_enabled=True,
    quantized=True,
    quantization_dtype="f8e4m3",
    quantized_checkpoints_path=MODEL_PATH,
    modules_to_not_convert=[
        "lm_head", "self_attn", "shared_expert",
        "layers.0.mlp", "layers.1.mlp", "layers.2.mlp",
    ],
    layer_boundary_markers=True,
    weights_to_skip_layout_optimization=[".*"],
    logical_nc_config=2,
    save_sharded_checkpoint=True,
    local_ranks_size=64,
    flash_decoding_enabled=False,
    on_cpu=False,
)

def load_config(c):
    for k, v in hf_config.items():
        setattr(c, k, v)
    # GLM-5.2 has heterogeneous per-layer indexers (indexer_types: full/shared);
    # some layers (e.g. layer 3) lack their own indexer weights, which breaks the
    # reference's per-layer indexer assumption. For seq_len <= index_topk (2048),
    # the DSA indexer is a mathematical no-op (top-2048 over <=2048 keys = full
    # attention), so disable DSA and run standard full MLA (cache dim 576).
    c.dsa_enabled = False

log(f"MODEL_PATH={MODEL_PATH} SEQ={SEQ}")
log(f"rope_theta(nested)={hf_config.get('rope_parameters')}  head_dim(hf)={hf_config.get('head_dim')}")
config = GLM5InferenceConfig(neuron_config=neuron_config, load_config=load_config)
log(f"config built: layers={config.num_hidden_layers} experts={getattr(config,'n_routed_experts','?')} "
    f"head_dim(cache)={config.head_dim} rope_theta={getattr(config,'rope_theta','?')}")

model = NeuronGLM5ForCausalLM(MODEL_PATH, config)

if not os.path.exists(os.path.join(COMPILED_MODEL_PATH, "model.pt")):
    log("compiling...")
    t0 = time.time()
    model.compile(COMPILED_MODEL_PATH)
    log(f"compile done in {time.time()-t0:.0f}s")
else:
    log("compiled artifacts already present, skipping compile")

log("DONE_COMPILE")
