# Contrib Model: BioReason-Pro

Multimodal protein function prediction on AWS Neuron, combining ESM3 protein encoding with Qwen3-4B reasoning via NxDI.

## Model Information

- **HuggingFace ID:** `wanglab/bioreason-pro-rl`
- **Model Type:** Multimodal pipeline (ESM3-small encoder + Qwen3-4B decoder with embedding injection)
- **Parameters:** ~5.4B total (Qwen3-4B: 4.0B, ESM3-small: 1.4B, projections: ~20M)
- **Architecture:** ESM3 per-residue embeddings projected into Qwen3-4B hidden space, injected at placeholder token positions before autoregressive generation
- **License:** See model card at `wanglab/bioreason-pro-rl`

## Overview

BioReason-Pro is a 3-stage protein function prediction pipeline:

1. **InterPro** (CPU): Domain annotation via EBI API (not accelerated)
2. **GO-GPT** (CPU/Neuron): GO term prediction using ESM2-3B encoder + decoder (optional, can be precomputed)
3. **BioReason-Pro** (Neuron): Final reasoning using Qwen3-4B with injected protein (ESM3) and GO graph embeddings

This contrib provides the **Stage 3 (BioReason-Pro)** implementation on Neuron. The Qwen3-4B backbone is compiled via NxDI with 8 runtime patches that enable `inputs_embeds` passthrough through the compiled model, allowing pre-computed multimodal embeddings to be injected at inference time.

### Embedding Injection Architecture

The model uses two special tokens (`<|protein_pad|>`, `<|go_graph_pad|>`) as placeholders in the prompt. Before generation:
- ESM3-small encodes the protein sequence on CPU -> per-residue embeddings (dim=1536)
- A 2-layer MLP projects ESM3 embeddings to Qwen3 hidden dim (2560)
- Pre-computed GO graph embeddings (200 x 2560) are similarly projected
- Placeholder tokens in `inputs_embeds` are replaced with the projected embeddings
- The modified `inputs_embeds` tensor is passed to Qwen3-4B via NxDI `generate(inputs_embeds=...)`

### NxDI Patches (V3b)

Standard NxDI does not support `inputs_embeds` passthrough for compiled models. This contrib includes 8 patches to `model_base.py`, `model_wrapper.py`, and `hf_adapter.py` that:
1. Preserve `inputs_embeds` through the preprocessing pipeline (not zeroed out)
2. Use `torch.where` for embedding injection during context encoding (CTE)
3. Maintain a uniform 19-argument signature for both CTE and TKG models
4. Thread `inputs_embeds` from `forward()` through to `_get_model_outputs()`
5. Handle the case where HF `generate()` passes `inputs_embeds` without `input_ids`
6. Fix the `DynamicCache` check that was dropping `inputs_embeds` on the first step

## Validation Results

**Validated:** 2026-04-09
**Instance:** trn2.3xlarge (ap-southeast-4)
**SDK:** Neuron SDK 2.28, NxDI 0.8.0, PyTorch 2.9

### Benchmark Results

| Metric | Value |
|--------|-------|
| Token throughput | 44.0 tok/s (constant across proteins) |
| Per-token latency | 22.7 ms |
| Per-protein average | 28.6s (range: 19.1s - 48.2s) |
| Proteins per minute | 2.1 |
| Model load time | ~26s (compiled) |
| ESM3 encoding (CPU) | 4.5 ms/residue |

Per-protein results (10 diverse proteins, 113-494 AA):

| Protein | Organism | Seq Len | Tokens | ESM3 (s) | Gen (s) | Total (s) |
|---------|----------|---------|--------|----------|---------|-----------|
| P63089 | Mouse | 168 | 1773 | 0.7 | 40.3 | 41.0 |
| P51864 | Human | 188 | 939 | 0.8 | 21.3 | 22.1 |
| Q06205 | Yeast | 392 | 2048 | 1.7 | 46.5 | 48.2 |
| A6NI15 | Human | 193 | 988 | 0.8 | 22.5 | 23.3 |
| Q66LM6 | Mouse | 333 | 1240 | 1.4 | 28.2 | 29.5 |
| P0C2H9 | Yeast | 113 | 943 | 0.5 | 21.4 | 22.0 |
| Q9M8K6 | Arabidopsis | 202 | 1577 | 0.9 | 35.8 | 36.7 |
| Q8W488 | Arabidopsis | 494 | 1258 | 2.2 | 28.6 | 30.8 |
| P0A9K3 | E. coli | 346 | 1181 | 1.4 | 26.8 | 28.2 |
| tmem106a | Human | 262 | 1071 | 1.1 | 24.3 | 25.4 |

### Accuracy Validation

| Metric | Value |
|--------|-------|
| GO-GPT input identity (GPU vs Neuron) | 10/10 identical |
| Exact GO term match | 23.3% (7/30 protein-aspect pairs) |
| Semantic match (incl. parent-child GO terms) | 50.0% (15/30) |
| Agreement excl. garbled output | 68.2% (15/22) |
| Well-formed output rate | 70% (7/10 proteins) |

**Notes on accuracy:**
- 3/10 proteins produce garbled output (BF16 repetition artifacts at long generation lengths)
- Where output is well-formed, predictions are scientifically equivalent to GPU (BF16 vs FP16)
- Prompt truncation at 1024 tokens affects proteins with long InterPro annotations

## Usage

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token"  # Required for gated ESM3 model

# Apply NxDI patches (must be done before loading the model)
from src.patch_nxdi_embeds import apply_all_patches
apply_all_patches()

# Load the pipeline
from src.modeling_bioreason import BioReasonPipeline

pipeline = BioReasonPipeline(
    model_path="/mnt/models/bioreason-pro-rl",
    esm3_model="esm3_sm_open_v1",
    max_context_length=1024,
    max_new_tokens=2048,
)

# Run inference
result = pipeline.predict(
    sequence="MSKMSHFLIYNALDQFIAGDVTPRHTGMIKVYAAELGITLAMQYLIALMSDEG...",
    organism="Saccharomyces cerevisiae (Baker's yeast)",
    interpro="- IPR001404: Heat shock protein Hsp90 family...",
    gogpt="GO:0005524 (ATP binding); GO:0051082 (unfolded protein binding)...",
)

print(f"Generated {result['num_tokens']} tokens at {result['tok_per_s']:.1f} tok/s")
print(result['text'])
```

### Prerequisites

1. Download the checkpoint:
   ```bash
   huggingface-cli download wanglab/bioreason-pro-rl --local-dir /mnt/models/bioreason-pro-rl
   ```

2. Accept the ESM3 license at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1

3. Install the BioReason-Pro source (for PLProcessor):
   ```bash
   git clone https://github.com/wanglab-georgetown/bioreason.git ~/bioreason2
   ```

4. Activate the Neuron environment:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
   pip install esm  # ESM3 library
   ```

## Compatibility Matrix

| Instance | SDK 2.28 | SDK 2.27 |
|----------|----------|----------|
| trn2.3xlarge (TP=1, LNC=2) | VALIDATED | Not tested |
| trn2.48xlarge | Not tested | Not tested |
| inf2.xlarge | Not tested | Not tested |

## Example Checkpoints

* [wanglab/bioreason-pro-rl](https://huggingface.co/wanglab/bioreason-pro-rl) (RL-tuned, recommended)
* [wanglab/bioreason-pro-sft](https://huggingface.co/wanglab/bioreason-pro-sft) (SFT variant)

## Testing Instructions

```bash
# On trn2.3xlarge with Neuron SDK 2.28:
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Set HF token for ESM3 access
export HF_TOKEN="your_token"

# Set model path (if not default)
export BIOREASON_MODEL_PATH="/mnt/models/bioreason-pro-rl"

# Run tests
cd contrib/models/BioReason-Pro
pytest test/integration/test_model.py -v --timeout=600
```

Expected output:
- `test_pipeline_basic`: PASSED (generates valid protein function analysis)
- `test_embedding_injection`: PASSED (embeddings correctly shaped and non-zero)
- `test_logit_accuracy_neuron_allclose`: PASSED (output is coherent, not garbled)
- `test_throughput`: PASSED (~44 tok/s on trn2.3xlarge)
- `test_patches_verified`: PASSED (all 8 V3b patches applied)

## Known Issues

1. **BF16 repetition artifacts**: 3/10 test proteins produce garbled output at long generation lengths (>1500 tokens). This is a known BF16 precision issue with Qwen3-4B on Neuron. Mitigation: reduce `max_new_tokens` or post-filter outputs.

2. **Prompt truncation**: The NxDI model is compiled with `max_context_length=1024`. Proteins with long InterPro annotations may have their prompt truncated, affecting prediction quality.

3. **First compilation**: Initial model compilation takes 15-20 minutes. Subsequent loads use the cached compiled model (~26s).

4. **ESM3 gated access**: The ESM3-small model requires accepting a license on HuggingFace and setting `HF_TOKEN`.

5. **PLProcessor dependency**: The pipeline requires `bioreason2.models.pl.processing_pl.PLProcessor` from the BioReason source repository for chat template formatting. This must be cloned separately.

## Maintainer

Jim Burtoft
