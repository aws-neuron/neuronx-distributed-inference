# MoLFormer on AWS Inferentia2

Compile and run [IBM MoLFormer-XL-both-10pct](https://huggingface.co/ibm/MoLFormer-XL-both-10pct) on AWS Inferentia2 using `torch_neuronx.trace()`.

## Model

| Property | Value |
|----------|-------|
| **Model** | ibm/MoLFormer-XL-both-10pct |
| **Architecture** | Encoder-only transformer (custom, trust_remote_code) |
| **Parameters** | ~47M (FP32) |
| **Input** | Tokenized SMILES strings (max_length=202) |
| **Output** | 768-dim molecular embeddings (pooler_output) |

## Instance

Validated on **inf2.xlarge** (2 NeuronCores), SDK 2.28, PyTorch 2.9.

## Results

### Accuracy (matmult bf16 vs CPU FP32)

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.999995 |
| Max absolute diff | 0.006311 |
| Mean absolute diff | 0.001544 |

### Benchmark (inf2.xlarge, matmult bf16)

| Config | Throughput (inf/s) | P50 Latency (ms) |
|--------|-------------------:|------------------:|
| BS=1, 1 core | 661 | 1.50 |
| BS=1, DP=2 | 1,302 | 1.52 |
| BS=1, DP=2, 4 workers | 1,503 | 2.65 |
| **BS=4, DP=2** | **1,660** | **4.79** |

### Model Size

| Auto-Cast | File Size |
|-----------|-----------|
| FP32 (none) | 160 MB |
| matmult bf16 | 74 MB (54% smaller) |

## Key Findings

1. **`--auto-cast matmult` is critical**: 65% throughput gain for this FP32 model (400 to 661 inf/s single-core)
2. **DataParallel scales near-perfectly**: 1.97x with 2 NeuronCores
3. **Cosine similarity 0.999995**: embeddings are functionally identical with matmult bf16
4. **BS=1 optimal for single-core**: larger batches add overhead without benefit on one NeuronCore
5. **BS=4 optimal for DP**: best whole-instance throughput at 1,660 inf/s

## Usage

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# Run tests
python -m pytest contrib/models/MoLFormer/test/ -v

# Or run the notebook
jupyter lab contrib/models/MoLFormer/molformer_neuron_inf2.ipynb
```

## Dependencies

All pre-installed in the DLAMI PyTorch inference venv:
- torch-neuronx
- transformers
- numpy

## Notes

- MoLFormer requires `trust_remote_code=True` (custom modeling code from HuggingFace hub)
- The `deterministic_eval=True` flag is MoLFormer-specific
- SMILES tokenizer has max_length=202
