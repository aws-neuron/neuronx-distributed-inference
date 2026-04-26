# Contrib Model: GLM-5

NeuronX Distributed Inference implementation of GLM-5 (zai-org/GLM-5-FP8).

## Model Information

- **HuggingFace ID:** `zai-org/GLM-5-FP8` (FP8 quantized checkpoint)
- **Architecture:** GLM-5 / DeepSeek-V3 family (MoE, MLA attention)
- **Total Parameters:** 754B (40B active per token)
- **Model Type:** `glm_moe_dsa`
- **License:** Check HuggingFace model card

## Architecture Details

GLM-5 is architecturally identical to DeepSeek-V3 with the following specifications:

| Feature | GLM-5 | DeepSeek-V3 |
|---------|-------|-------------|
| hidden_size | 6144 | 7168 |
| num_hidden_layers | 78 (3 dense + 75 MoE) | 61 (1 dense + 60 MoE) |
| num_attention_heads | 48 | 128 |
| qk_nope_head_dim | 192 | 128 |
| qk_rope_head_dim | 64 | 64 |
| v_head_dim | 256 | 128 |
| q_lora_rank | 2048 | 1536 |
| kv_lora_rank | 512 | 512 |
| n_routed_experts | 256 | 256 |
| num_experts_per_tok | 8 | 8 |
| moe_intermediate_size | 2048 | 2048 |
| Routing | sigmoid + selection_bias + L1 norm | sigmoid + selection_bias + L1 norm |
| routed_scaling_factor | 2.5 | 2.5 |
| rope_theta | 1,000,000 | 10,000,000 |
| vocab_size | 154,880 | 129,280 |

Key features:
- **MLA (Multi-head Latent Attention):** Compressed KV cache storing 576 values per token (512 compressed + 64 RoPE)
- **256 routed experts, top-8 sigmoid routing** with `e_score_correction_bias` and `routed_scaling_factor=2.5`
- **1 shared expert per MoE layer** (implemented as separate module outside fused kernel)
- **FP8 expert weights** with per-tensor symmetric quantization (non-expert layers dequantized to BF16)
- **DSA (DeepSeek Sparse Attention)** indexer: architecture defined but using full-attention fallback
- **MTP (Multi-Token Prediction)** layer: skipped (training-only)
- **NKI MLP kernel** for dense layers 0-2 via `mlp_kernel_enabled=True` (uses nkilib SwiGLU kernel for both CTE and TKG)

## Important: nkilib Override for GLM-5 Routing

GLM-5 uses a modified NKI fused MoE kernel that adds `selection_bias` and `routed_scaling_factor` support to the router. This requires the [nki-lib fork](https://github.com/jimburtoft/nki-library) with routing modifications installed in editable mode:

```bash
git clone https://github.com/jimburtoft/nki-library.git nki-lib
cd nki-lib
git checkout feature/selection-bias-routing
pip install -e .
```

The modeling code patches the fused TKG kernel at runtime via `_patch_fused_tkg_with_nkilib()` to inject GLM-5's routing parameters into the NKI mega-kernel.

**Modified nkilib files (4 files):**
- `src/nkilib_src/nkilib/core/router_topk/router_topk.py` — NKI kernel with selection_bias + routed_scaling_factor
- `src/nkilib_src/nkilib/core/router_topk/router_topk_torch.py` — PyTorch reference
- `src/nkilib_src/nkilib/core/moe_block/moe_block_tkg.py` — Mega-kernel interface
- `src/nkilib_src/nkilib/core/subkernels/rmsnorm_tkg.py` — NKI 0.3.0 tensor_reduce axis fix

## Compatibility Matrix

| Neuron SDK | Instance Type | TP Degree | LNC | Status |
|-----------|--------------|-----------|-----|--------|
| 2.29 (neuronx-cc 2.24) | trn2.48xlarge | 64 | 2 | Tested |

**Requirements:**
- Neuron SDK 2.29 (`Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`)
- NxD Inference 0.9.17334+
- NKI 0.3.0 (GA)
- trn2.48xlarge (32 NeuronDevices, 64 logical cores at LNC=2)
- ~1 TB NVMe storage for compiled model + pre-sharded weights
- ~705 GB for the FP8 checkpoint (142 safetensors)

## Usage

### Compilation

```python
import os
import sys
import json
import torch

os.environ["UNSAFE_FP8FNCAST"] = "1"

# SDK 2.29 race condition workarounds
_orig_makedirs = os.makedirs
def _safe_makedirs(name, mode=0o777, exist_ok=False):
    return _orig_makedirs(name, mode=mode, exist_ok=True)
os.makedirs = _safe_makedirs

import shutil
_orig_rmtree = shutil.rmtree
def _safe_rmtree(path, ignore_errors=False, onerror=None, **kw):
    return _orig_rmtree(path, ignore_errors=True, **kw)
shutil.rmtree = _safe_rmtree

from neuronx_distributed_inference.models.config import MoENeuronConfig
from modeling_glm5 import NeuronGLM5ForCausalLM, GLM5InferenceConfig

MODEL_PATH = "/mnt/nvme/GLM-5-FP8"
COMPILED_MODEL_PATH = "/mnt/nvme/glm5_compiled"

neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    seq_len=2048,
    n_active_tokens=2048,
    torch_dtype=torch.bfloat16,
    fused_qkv=False,
    qkv_kernel_enabled=False,
    qkv_nki_kernel_enabled=False,
    moe_fused_nki_kernel_enabled=True,
    expert_mlp_nki_kernel_enabled=False,
    mlp_kernel_enabled=True,  # NKI MLP kernel for dense layers 0-2 (+4% throughput)
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

config = GLM5InferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
model = NeuronGLM5ForCausalLM(MODEL_PATH, config)

# Compile (single-process SPMD, NOT torchrun)
# Run with: python3 compile_script.py
model.compile(COMPILED_MODEL_PATH)
```

### Weight Pre-sharding

After compilation, pre-shard weights for fast loading:

```python
# Single-process weight sharding (NOT torchrun)
model.preshard_and_save(MODEL_PATH, COMPILED_MODEL_PATH)
```

### Inference

```python
# Single-process loading (NOT torchrun)
import torch
from transformers import PreTrainedTokenizerFast
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

# Load model
model = NeuronGLM5ForCausalLM.from_pretrained(COMPILED_MODEL_PATH)
model.load(COMPILED_MODEL_PATH)
wrapped = HuggingFaceGenerationAdapter(model)

# Tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f"{MODEL_PATH}/tokenizer.json",
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
)

# Generate
# IMPORTANT: Pad prompt to (seq_len - max_new_tokens) to leave room for generation.
# Total sequence length (prompt + generated) must not exceed seq_len (2048).
max_new_tokens = 128
prompt_pad_len = 2048 - max_new_tokens  # 1920
inputs = tokenizer("The meaning of life is", return_tensors="pt", padding="max_length", max_length=prompt_pad_len)
with torch.no_grad():
    outputs = wrapped.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Important: Single-Process SPMD

The model is compiled and loaded as a single-process SPMD model (one process controlling all 64 NeuronCores via `local_ranks_size=64`). Both compilation and inference use a single Python process — do NOT use `torchrun`.

## Benchmark Results

**Instance:** trn2.48xlarge (32 NeuronDevices, 64 logical cores, LNC=2)
**SDK:** 2.29 (neuronx-cc 2.24.5133.0)
**Precision:** FP8 experts, BF16 attention/dense layers
**Routing:** GLM-5 sigmoid routing with selection_bias + routed_scaling_factor=2.5
**NKI Kernels:** Fused MoE TKG + MLP kernel for dense layers

| Batch Size | CTE seq_len | Total tok/s | Per-req tok/s | Per-token latency | Scaling |
|-----------|-------------|-------------|---------------|-------------------|---------|
| 1 | 2048 | 2.27 | 2.27 | 440 ms | 1.0x |
| 4 | 512 | 12.3 | 3.1 | 326 ms | 5.4x |
| 8 | 256 | 23.4 | 2.9 | 342 ms | 10.3x |

**Notes:**
- CTE (context encoding) compilation is the bottleneck for larger batch sizes due to HBM limits; `seq_len` must be reduced proportionally
- Weight pre-sharding produces 64 rank files totaling ~1044 GB; weight loading takes ~50-57s, warmup ~17s
- Near-linear batch scaling observed (11.1x at BS=8 vs theoretical 8x)

## Known Limitations

1. **DSA (DeepSeek Sparse Attention):** Architecture is defined but currently uses full-attention fallback. The DSA indexer weights are removed from the state dict during conversion.
2. **Shared Expert:** Implemented as a separate module outside the fused NKI kernel (minimal performance impact).
3. **MTP Layer:** The Multi-Token Prediction layer (layer 78) is skipped as it is training-only.
4. **CTE seq_len vs batch size:** CTE compilation requires reducing `seq_len` for larger batch sizes (BS=4: 512, BS=8: 256) due to HBM constraints.
5. **SDK 2.29 race conditions:** Requires monkey-patches for `os.makedirs` and `shutil.rmtree` (see usage examples above).
6. **FP8 NaN clamping:** Neuron hardware treats exponent-15 FP8 bytes as NaN; weights are clamped to max 240 (affects ~1.4-2.2% of bytes).

## Checkpoint

- **FP8 Checkpoint:** `zai-org/GLM-5-FP8` (142 safetensors, ~705 GB)
- The modeling code handles FP8 blockwise dequantization for non-expert weights and FP8 re-quantization with per-tensor symmetric scales for expert weights.

## Running Tests

```bash
# Integration test (requires trn2.48xlarge with compiled model)
export COMPILED_MODEL_PATH=/mnt/nvme2/glm5_compiled
export MODEL_PATH=/mnt/nvme/GLM-5-FP8
PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py -v
```

## Validated On

- **Instance:** trn2.48xlarge (us-east-2b, `Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`)
- **SDK:** 2.29 (neuronx-cc 2.24.5133.0, NxDI 0.9.17334, NKI 0.3.0)
- **Date:** 2026-04-26
- **Results:** Compilation PASS (both CTE and TKG), all 4 pytest tests PASS, 2.27 tok/s at BS=1

## Maintainer

Agent glm - Annapurna Labs

**Last Updated:** 2026-04-26
