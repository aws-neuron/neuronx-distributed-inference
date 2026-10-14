# Running Qwen3-ASR-1.7B with vLLM on AWS Neuron

## Prerequisites

- trn2.3xlarge instance with SDK 2.29 DLAMI (`Deep Learning AMI Neuron (Ubuntu 24.04) 20260410`)
- Pre-compiled encoder NEFFs and text decoder (see parent `README.md` for compilation steps)
- Model weights: `Qwen/Qwen3-ASR-1.7B`

## Setup

### 1. Install vLLM-neuron

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
git clone https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

### 2. Configure Qwen3-ASR Support

Apply the following patches to vllm-neuron:

#### 2.1 Register Qwen3-ASR in `NEURON_MULTI_MODAL_MODELS`

Modify `vllm_neuron/worker/constants.py`:

```diff
--- a/vllm_neuron/worker/constants.py
+++ b/vllm_neuron/worker/constants.py
@@ -5,6 +5,7 @@ NEURON_MULTI_MODAL_MODELS = [
     "MllamaForConditionalGeneration",
     "LlavaForConditionalGeneration",
     "Llama4ForConditionalGeneration",
+    "Qwen3ASRForConditionalGeneration",
 ]
```

#### 2.2 Add `NeuronQwen3ASRForCausalLM` class to `vllm_neuron/worker/neuronx_distributed_model_loader.py`

Add the following class (see `neuron_qwen3_asr_vllm.py` for the full implementation):

```diff
+class NeuronQwen3ASRForCausalLM(NeuronMultiModalCausalLM):
+    """Qwen3-ASR multimodal model using decomposed pipeline:
+    - Traced encoder NEFFs (bucketed: 500, 1000, 3000 mel frames)
+    - NxDI Qwen3-VL text decoder with vision_embeddings scatter
+    """
+    ...  # See neuron_qwen3_asr_vllm.py for full 460-line implementation
```

The class handles:
- Loading pre-compiled encoder NEFFs from `NEURON_ENCODER_PATH`
- Loading the NxDI text decoder from `NEURON_COMPILED_ARTIFACTS`
- Bucket selection for encoder (5s/10s/30s audio)
- Mel feature extraction → encoder → audio embeddings
- Scatter audio embeddings into token positions via `vision_mask`
- mRoPE position computation (all 3 axes identical for ASR)
- Routing to NxDI text decoder (CTE prefill / TKG decode)

#### 2.3 Add dispatch case in `get_neuron_model()`

Modify `vllm_neuron/worker/neuronx_distributed_model_runner.py`:

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -775,6 +775,8 @@ def get_neuron_model(
     elif architecture == "Llama4ForConditionalGeneration":
         model = NeuronLlama4ForCausalLM(model_config.hf_config)
+    elif architecture == "Qwen3ASRForConditionalGeneration":
+        model = NeuronQwen3ASRForCausalLM(model_config.hf_config)
     else:
         model = NeuronCausalLM(model_config.hf_config)
```

#### 2.4 Add Qwen3-ASR to audio pass-through in `_process_multi_modal_data_neuron`

Modify `vllm_neuron/worker/neuronx_distributed_model_runner.py`:

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -1139,7 +1139,7 @@
-        if self.model.architecture == "ShukaModel":
-            pass  # Shuka-1 audio data (input_features) passes through directly
+        if self.model.architecture == "ShukaModel" or self.model.architecture == "Qwen3ASRForConditionalGeneration":
+            pass  # Audio data passes through directly (Shuka-1: input_features, Qwen3-ASR: input_audio_features)
         elif self.model.model.config.model_type == "llava":
```

#### 2.5 Fix `_get_model_configs` for nested `thinker_config`

Modify `vllm_neuron/worker/neuronx_distributed_model_loader.py` in the `_get_model_configs` function:

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_loader.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_loader.py
@@ -1535,7 +1535,10 @@ def _get_model_configs(config: PretrainedConfig) -> str:
     # For multimodal models like Llava/Mllama, use text_config
-    text_config = getattr(config, "text_config", None)
+    text_config = getattr(config, "text_config", None) or getattr(
+        getattr(config, "thinker_config", None), "text_config", None
+    )
     if text_config is not None:
         config = text_config
```

#### 2.6 Fix `get_num_layers_from_hf_config` for `thinker_config`

Modify `vllm_neuron/worker/utils.py`:

```diff
--- a/vllm_neuron/worker/utils.py
+++ b/vllm_neuron/worker/utils.py
@@ -XX,6 +XX,10 @@ def get_num_layers_from_hf_config(config):
+    # Handle thinker_config nesting (Qwen3-ASR)
+    thinker_config = getattr(config, "thinker_config", None)
+    if thinker_config is not None:
+        text_config = getattr(thinker_config, "text_config", None)
+        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
+            return text_config.num_hidden_layers
```

#### 2.7 Fix infinite recursion in `platform.py`

Modify `vllm_neuron/platform.py` in `_register_neuron_multimodal_models()`:

```diff
--- a/vllm_neuron/platform.py
+++ b/vllm_neuron/platform.py
@@ -131,7 +131,7 @@ def _register_neuron_multimodal_models():
     for arch in NEURON_MULTI_MODAL_MODELS:
-        if ModelRegistry._try_inspect_model_cls(arch) is not None:
+        if arch in ModelRegistry.models:
             ModelRegistry.register_model(arch, NeuronMultiModalCausalLM)
```

### 3. Set Environment Variables

```bash
# Path to pre-compiled text decoder NEFFs
export NEURON_COMPILED_ARTIFACTS='/path/to/compiled/qwen3_asr_vl_text_tp4'

# Path to pre-compiled encoder NEFFs (encoder_T500.pt, encoder_T1000.pt, encoder_T3000.pt)
export NEURON_ENCODER_PATH='/path/to/compiled/qwen3_asr_encoder'

# Visible cores (4 cores for TP=4)
export NEURON_RT_VISIBLE_CORES='0-3'
```

### 4. Run Inference

#### 4.1 Start vLLM Server

```bash
bash start-vllm-server.sh
```

Or manually:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model 'Qwen/Qwen3-ASR-1.7B' \
    --tensor-parallel-size 4 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --port 8000 \
    --trust-remote-code \
    --additional-config '{"override_neuron_config": {"text_neuron_config": {"tp_degree": 4, "batch_size": 1, "n_positions": 1024, "seq_len": 1024}}}'
```

#### 4.2 Test Transcription

```bash
python3 test_transcription.py
```

## Performance

| Metric | Value |
|--------|-------|
| E2E latency (2.9s audio, 17 tokens) | 140ms |
| TPOT (raw, from benchmark) | 4.9ms |
| TPOT (via vLLM API, includes overhead) | ~8.2ms |
| Audio throughput | ~49.7 audio-sec/wall-sec |

## Known Limitations

- **Batch size**: Currently limited to `max-num-seqs=1` due to NxDI `scatter_by_index_put()` assuming BS=1 for multimodal prefill
- **Prefix caching**: Must be disabled (`--no-enable-prefix-caching`)
- **Block size**: Maximum 256 (use 128)
- **Transcription API**: Only chat completions endpoint tested; `/v1/audio/transcriptions` not yet validated
