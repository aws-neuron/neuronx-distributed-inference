# Running InternVL3-8B-Instruct with vLLM on AWS Neuron

## Prerequisites

- Model downloaded to `/mnt/models/InternVL3-8B-Instruct/`
- trn2.3xlarge instance with SDK 2.29 (DLAMI 20260410)
- vLLM venv: `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate`

## Setup

### 1. Patch vLLM-neuron

vLLM-neuron 0.5.0 does not natively support InternVL3. The following patches are required:

#### 1.1 Register InternVLChatModel as multimodal

Edit `/vllm/vllm_neuron/worker/constants.py`:

```diff
 NEURON_MULTI_MODAL_MODELS = [
     "MllamaForConditionalGeneration",
     "LlavaForConditionalGeneration",
     "Llama4ForConditionalGeneration",
     "Qwen2VLForConditionalGeneration",
     "Qwen3VLForConditionalGeneration",
+    "InternVLChatModel",
 ]
```

#### 1.2 Handle InternVL architecture in model config extraction

Edit `/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py`:

In `_get_model_configs()`, InternVL3 uses `llm_config` instead of `text_config`:

```diff
 def _get_model_configs(config: PretrainedConfig) -> str:
     archs = getattr(config, "architectures", [])
     if not archs:
         raise ValueError("No architectures specified in the pretrained config.")
     architecture = archs[0]
     if architecture in NEURON_MULTI_MODAL_MODELS:
-        config = getattr(config, "text_config", None)
+        config = getattr(config, "text_config", None) or getattr(config, "llm_config", None)
     num_key_value_heads = getattr(config, "num_key_value_heads", None)
```

#### 1.3 Add NeuronInternVL3ForCausalLM class

Add to `/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py` (after the NeuronLlama4ForCausalLM class):

```python
class NeuronInternVL3ForCausalLM(NeuronMultiModalCausalLM):
    """InternVL3 multimodal model using dynamically loaded contrib model."""

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        import importlib

        neuronx_module = importlib.import_module("modeling_internvl3")
        neuronx_model_cls = getattr(neuronx_module, "NeuronInternVL3ForCausalLM")

        default_neuron_config = kwargs["neuron_config"]
        override_neuron_config = _validate_image_to_text_override_neuron_config(
            kwargs["override_neuron_config"]
        )

        vision_neuron_config = copy.deepcopy(default_neuron_config)
        vision_neuron_config.update(
            override_neuron_config.get("vision_neuron_config", {})
        )
        # InternVL3 vision encoder has fused QKV weights
        vision_neuron_config["fused_qkv"] = True
        vision_neuron_config["buckets"] = [1]
        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **vision_neuron_config
        )

        text_neuron_config = copy.deepcopy(default_neuron_config)
        text_neuron_config.update(override_neuron_config.get("text_neuron_config", {}))
        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **text_neuron_config
        )

        config = neuronx_model_cls.get_config_cls().from_pretrained(
            model_name_or_path,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

        success, compiled_model_path, _ = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, config=config, **kwargs
        )

        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(model_name_or_path)

            self._compile_and_load_model(
                model_name_or_path, neuronx_model_cls, config, compiled_model_path
            )

        # Set vision token ID
        self.vision_token_id = 151667  # <IMG_CONTEXT>
        return success, compiled_model_path
```

#### 1.4 Map InternVLChatModel in get_neuron_model()

Edit `/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py`:

```diff
     elif architecture == "Llama4ForConditionalGeneration":
         model = NeuronLlama4ForCausalLM(model_config.hf_config)
+    elif architecture == "InternVLChatModel":
+        from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronInternVL3ForCausalLM
+        model = NeuronInternVL3ForCausalLM(model_config.hf_config)
     else:
         model = NeuronCausalLM(model_config.hf_config)
```

#### 1.5 Handle InternVL3 multimodal data processing

Edit `/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py`, in `_process_multi_modal_data_neuron()`:

```diff
         elif self.model.model.config.model_type == "llama4":
             pass  # llama4 doesn't require special processing
+        elif self.model.model.config.model_type == "internvl_chat":
+            pass  # InternVL3 processes pixel_values directly
         else:
             raise NotImplementedError(
```

#### 1.6 Handle InternVLChatModel in _get_neuron_model_cls()

The `_get_neuron_model_cls()` function splits architecture on "For" which doesn't work for `InternVLChatModel`. Add a special case:

```diff
 def _get_neuron_model_cls(architecture: str):
+    # Special case: InternVLChatModel doesn't follow *For* naming convention
+    if architecture == "InternVLChatModel":
+        import importlib
+        mod = importlib.import_module("modeling_internvl3")
+        return getattr(mod, "NeuronInternVL3ForCausalLM")
+
     if architecture.startswith("Neuron") and "For" in architecture:
```

### 2. Run Inference

#### 2.1 Offline Inference

```bash
PYTHONPATH="$PWD/contrib/models/InternVL3-8B-Instruct/src:$PYTHONPATH" \
python contrib/models/InternVL3-8B-Instruct/vllm/run_offline_inference.py
```

#### 2.2 Online Inference

```bash
# Start server
PYTHONPATH="$PWD/contrib/models/InternVL3-8B-Instruct/src:$PYTHONPATH" \
bash contrib/models/InternVL3-8B-Instruct/vllm/start-vllm-server.sh

# Query server
python contrib/models/InternVL3-8B-Instruct/vllm/run_online_inference.py
```

## Status

**Tested and working (2026-04-28).** Offline inference validated on trn2.3xlarge with SDK 2.29.

- Compilation: ~4.5 min (6 CTE + 6 TKG buckets)
- Text loading: 16.6s, Vision loading: 2.5s
- Throughput: ~42.64 tok/s output
- Correct text generation verified

### vLLM LLM() Parameter Notes

- Do NOT pass `device="neuron"` — Neuron is auto-detected via platform plugin in vLLM 0.16.0
- `override_neuron_config` must be inside `additional_config`, not a top-level parameter
- `enable_prefix_caching=False` is required (Neuron doesn't support block-based prefix caching)

## Known Issues

1. InternVL3's HF config uses `llm_config` instead of `text_config` (non-standard)
2. Architecture name `InternVLChatModel` doesn't follow `*ForConditionalGeneration` convention
3. Batch_size>1 not supported (same limitation as direct NxDI inference)
4. `trust_remote_code=True` required for tokenizer
