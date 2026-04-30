# Contrib Model: Qwen3.5-4B

NeuronX Distributed Inference implementation of Qwen3.5-4B, a dense hybrid DeltaNet + GQA decoder from Alibaba Cloud.

This variant reuses the proven PR 140/141 architecture:

- Standard GQA layers use NxDI `KVCacheManager`.
- DeltaNet layers return dummy KV tensors to satisfy NxDI cache plumbing.
- Real DeltaNet state is carried through layer-local `recurrent_state_buffer` and `conv_state_buffer` side-channel aliases.

## Model Information

| Feature | Value |
| --- | --- |
| HuggingFace ID | `Qwen/Qwen3.5-4B` |
| Model type | `qwen3_5_text` under top-level `qwen3_5` |
| Layers | 32: 24 DeltaNet + 8 GQA |
| Layer pattern | `[3 DeltaNet + 1 GQA] x 8` |
| Hidden size | 2560 |
| MLP | Dense SwiGLU, intermediate size 9216 |
| GQA attention | 16 Q heads, 4 KV heads, head_dim 256 |
| DeltaNet | 32 value heads, 16 key heads, k_dim=v_dim=128 |
| Conv kernel | 4, state stores last 3 pre-conv QKV tokens |
| RoPE | Partial RoPE, 25% of head_dim = 64 dims |
| Vocabulary | 248,320 |
| Tied embeddings | Yes |

Derived DeltaNet shapes:

| Tensor | Shape |
| --- | --- |
| `in_proj_qkv.weight` | `[8192, 2560]` |
| `in_proj_z.weight` | `[4096, 2560]` |
| `in_proj_a.weight` | `[32, 2560]` |
| `in_proj_b.weight` | `[32, 2560]` |
| `conv1d.weight` | `[8192, 1, 4]` |
| `recurrent_state_buffer` | `[max_batch, 32, 128, 128]` |
| `conv_state_buffer` | `[max_batch, 8192, 3]` |

## Status

Prepared for Trn2 bring-up. Validate TP=4, batch=1, seq_len=128 first, then increase context or reduce TP only after the baseline generates correctly.

## Testing

CPU unit tests:

```bash
cd contrib/models/Qwen3.5-4B
pytest test/unit -v
```

Trainium integration:

```bash
cd contrib/models/Qwen3.5-4B
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
QWEN35_MODEL_PATH=/home/ubuntu/models/Qwen3.5-4B \
QWEN35_COMPILED_PATH=/home/ubuntu/models/qwen35_4b_traced_trn2 \
QWEN35_TP_DEGREE=4 \
QWEN35_SEQ_LEN=128 \
pytest test/integration/test_model.py --capture=tee-sys -v
```

## Known Limitations

- DeltaNet weights are replicated across TP ranks in v1.
- DeltaNet layers still allocate dummy KV cache through NxDI's normal cache manager.
- MoE, VL, quantization, speculation, and custom hybrid cache cleanup are out of scope.
