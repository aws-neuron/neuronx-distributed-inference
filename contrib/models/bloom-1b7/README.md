# Bloom-1b7 NeuronX Port

NeuronX Distributed Inference port of [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7).

## Architecture

- **Parameters**: 1.7B
- **Hidden size**: 2048
- **Attention heads**: 16 (MHA, not GQA)
- **Layers**: 24
- **Vocab size**: 250,880
- **Position encoding**: ALiBi (Attention with Linear Biases) — no position embeddings
- **Activation**: GELU (tanh approximation)
- **Normalization**: LayerNorm (not RMSNorm), with embedding LayerNorm
- **QKV format**: Fused interleaved [num_heads, 3, head_dim, hidden_size]
- **Weight tying**: LM head tied to token embeddings

## Key Implementation Details

- ALiBi bias is computed and added in custom `perform_prefill` and `compute_for_token_gen` overrides
- Fused QKV weights are split into separate Q, K, V during weight conversion
- Embedding LayerNorm applied after token embeddings via `get_model_output` override
- Uses standard `nn.LayerNorm` (not RMSNorm) with `layer_norm_epsilon`

## Compilation

```python
from modeling_bloom import NeuronBloomForCausalLM, BloomInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig

neuron_config = NeuronConfig(
    tp_degree=1, batch_size=1, seq_len=128,
    max_context_length=128, torch_dtype=torch.bfloat16,
)
config = BloomInferenceConfig.from_pretrained(model_path, neuron_config=neuron_config)
model = NeuronBloomForCausalLM(model_path, config)
model.compile(output_path)
```

## Validation Results

- **Greedy match**: 76.9% (vs HF bfloat16 reference)
- **Model**: bloom-1b7, tp=1, seq_len=128, bfloat16

## Performance

Measured on trn1.32xlarge, batch_size=1, seq_len=128, bfloat16. Utilization is per-NeuronCore (TP=1).

| Metric | Value |
|--------|-------|
| Throughput | 63.3 tok/s |
| Context Encoding MBU | 18.6% |
| Context Encoding MFU | 7.4% |
| Token Generation MBU | 16.1% |
| Token Generation MFU | 0.1% |
