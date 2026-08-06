# Running Isaac-0.2-2B with vLLM on AWS Neuron

## Setup

### 1. Download Model Weights

```bash
huggingface-cli download PerceptronAI/Isaac-0.2-2B-Preview --local-dir /mnt/models/Isaac-0.2-2B-Preview
```

### 2. Activate vLLM Environment

Use the DLAMI venv that includes vLLM 0.16.0 + vllm-neuron 0.5.0:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
```

### 3. Apply vLLM Patches

Isaac is a contrib model and requires patching vllm-neuron to register the model:

```bash
NXDI_ROOT="/mnt/models/neuronx-distributed-inference"
PYTHONPATH="${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src:${NXDI_ROOT}/src:$PYTHONPATH" \
    python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/vllm/patch_vllm_isaac.py
```

This patches 3 files in the installed vllm-neuron package:
1. `constants.py` — Registers `IsaacForConditionalGeneration` as a multimodal model
2. `neuronx_distributed_model_loader.py` — Adds Isaac wrapper class with `load_weights()` and custom `execute_model()` override
3. `neuronx_distributed_model_runner.py` — Adds multimodal data routing for `"isaac"` model type

### 3.5. Patch modular_isaac.py (Required)

Isaac's HuggingFace `modular_isaac.py` imports the proprietary `perceptron.tensorstream` package, which
is unavailable on Neuron instances. This must be patched before vLLM can load the model config:

```bash
NXDI_ROOT="/mnt/models/neuronx-distributed-inference"
python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/gpu_benchmark/nuke_perceptron_import.py \
    /mnt/models/Isaac-0.2-2B-Preview/modular_isaac.py
```

**Important**: If HuggingFace has already cached the model code, also patch the cached copy:

```bash
python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/gpu_benchmark/nuke_perceptron_import.py \
    ~/.cache/huggingface/modules/transformers_modules/Isaac_hyphen_0_dot_2_hyphen_2B_hyphen_Preview/modular_isaac.py
```

### 4. Compile Model (if not already compiled)

The model must be compiled via NxDI before vLLM can serve it:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
PYTHONPATH="${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src:${NXDI_ROOT}/src:$PYTHONPATH" \
    python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/test/integration/run_isaac.py compile
```

## Running

### Offline Inference

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
NXDI_ROOT="/mnt/models/neuronx-distributed-inference"
PYTHONPATH="${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src:${NXDI_ROOT}/src:$PYTHONPATH" \
    python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/vllm/run_offline_inference.py
```

### Online Serving

1. Start the server:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
NXDI_ROOT="/mnt/models/neuronx-distributed-inference"
PYTHONPATH="${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/src:${NXDI_ROOT}/src:$PYTHONPATH" \
    bash ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/vllm/start-vllm-server.sh
```

2. Query the server:

```bash
python ${NXDI_ROOT}/contrib/models/Isaac-0.2-2B/vllm/run_online_inference.py --base-url http://localhost:8080
```

Or use curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Isaac-0.2-2B-Preview",
    "messages": [{"role": "user", "content": "What is quantum computing?"}],
    "max_tokens": 100,
    "temperature": 0
  }'
```

## Configuration

Key vLLM parameters for Isaac:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `tensor-parallel-size` | 1 | 2B model fits on single core |
| `max-model-len` | 1024 | Adjust based on compiled buckets |
| `max-num-seqs` | 1 | VLM framework limitation |
| `trust-remote-code` | Required | Isaac uses custom model code |
| `attn_kernel_enabled` | true | CTE flash attention (+2%) |

## Tested Results

| Mode | Status | Throughput | Notes |
|------|--------|------------|-------|
| Text-only (offline) | **Working** | ~78 tok/s | Correct output verified |
| Image+text (offline) | Not working | N/A | pixel_values format mismatch |
| Online API server | Not tested | N/A | Text-only expected to work |

**Example output** (text-only):
```
Prompt: "What is the capital of France?"
Output: "<think>\n\n</think>\n\nThe capital of France is Paris."
```

## Known Limitations

1. **Image+text is not supported via vLLM**: vLLM-neuron delivers `pixel_values` in pre-flattened
   patch format `[num_patches, patch_dim]`, but Isaac's NxDI model expects raw image tensors
   `[B, 3, 256, 256]`. Fixing this requires adapting vLLM's multimodal preprocessing or adding
   a reshape layer in the wrapper.

2. **On-device sampling mismatch**: Isaac's NxDI model returns logits (not on-device sampled tokens).
   The `execute_model()` override in the wrapper handles this by extracting
   `output.logits[:, -1, :]` and applying `torch.argmax()`. This means sampling parameters
   like `temperature` and `top_p` are NOT respected — generation is always greedy.

3. **`modular_isaac.py` must be patched**: The proprietary `perceptron.tensorstream` import must be
   removed before vLLM can load the model. See step 3.5 above.

4. **Single sequence only**: `max-num-seqs=1` is required due to the NxDI VLM framework limitation
   (shared with all VLM contrib models).

## Architecture

The vLLM integration uses a 3-file patch approach:

```
vllm-neuron (installed package)
├── worker/constants.py               + "IsaacForConditionalGeneration" in NEURON_MULTI_MODAL_MODELS
├── worker/neuronx_distributed_model_loader.py  + NeuronIsaacForConditionalGeneration class
│                                                  + get_neuron_model() dispatch
└── worker/neuronx_distributed_model_runner.py  + "isaac" multimodal routing
```

The `NeuronIsaacForConditionalGeneration` wrapper:
- Loads the compiled NxDI Isaac model via `load_weights()`
- Overrides `execute_model()` to handle the logits→token ID conversion
- Uses `vision_token_id = 151655` (`<|image_pad|>`) for vision mask construction
