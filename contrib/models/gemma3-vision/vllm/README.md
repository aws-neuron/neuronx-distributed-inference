# Running Gemma3 Vision Models with vLLM on AWS Neuron

## Setup
*Note*: In the following, we assume that the HuggingFace model weights are available on the host. If not,
download them using the following commands:

```bash
hf auth login --token <YOUR_HF_TOKEN>
hf download google/gemma-3-27b-it --local-dir </path/to/hf/weights>
```

The `</path/to/hf/weights>` path will need to be provided to vLLM as `--model`/`model` argument.
If the HuggingFace CLI is not installed, run:

```bash
python3 -m venv hf_env
source hf_env/bin/activate
pip install -U "huggingface_hub[cli]"
```

### 1. Install vLLM
```bash
git clone --branch "0.3.0" https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

### 2. Configure Gemma3 Support
Modify `vllm-neuron/vllm-neuron/worker/constants.py`:
Modify `vllm-neuron/vllm-neuron/worker/neuronx_distributed_model_loader.py`:
Modify `vllm-neuron/vllm-neuron/worker/neuronx_distributed_model_runner.py`:

#### 2.1 Register Gemma3 HuggingFace model class in supported `NEURON_MULTI_MODAL_MODELS`

```diff
--- a/vllm_neuron/worker/constants.py
+++ b/vllm_neuron/worker/constants.py
@@ -5,6 +5,7 @@ NEURON_MULTI_MODAL_MODELS = [
     "MllamaForConditionalGeneration",
     "LlavaForConditionalGeneration",
     "Llama4ForConditionalGeneration",
+    "Gemma3ForConditionalGeneration"
 ]

 TORCH_DTYPE_TO_NEURON_AMP = {
```

#### 2.2 Fix wrong import in `vllm_neuron/worker/neuronx_distributed_model_loader.py`

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_loader.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_loader.py
@@ -51,7 +51,7 @@ from vllm.config import (
 )
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.v1.outputs import SamplerOutput
-from vllm.v1.sample import sampler as Sampler
+from vllm.v1.sample.sampler import Sampler

 from vllm_neuron.worker.constants import (
     NEURON_MULTI_MODAL_MODELS,
```

#### 2.3 Add `NeuronGemma3ForConditionalGeneration` class to `vllm_neuron/worker/neuronx_distributed_model_loader.py`

```diff
@@ -704,6 +704,61 @@ class NeuronLlama4ForCausalLM(NeuronMultiModalCausalLM):
             **kwargs,
         )

+class NeuronGemma3ForConditionalGeneration(NeuronLlama4ForCausalLM):
+    """Gemma3 multimodal model using dynamically loaded NeuronGemma3ForConditionalGeneration from contrib."""
+
+    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
+        import importlib
+
+        neuronx_module = importlib.import_module("gemma3_vision.modeling_gemma3")
+        neuronx_model_cls = getattr(neuronx_module, "NeuronGemma3ForConditionalGeneration")
+
+        default_neuron_config = kwargs["neuron_config"]
+        override_neuron_config = _validate_image_to_text_override_neuron_config(
+            kwargs["override_neuron_config"]
+        )
+
+        vision_neuron_config = copy.deepcopy(default_neuron_config)
+        vision_neuron_config.update(
+            override_neuron_config.get("vision_neuron_config", {})
+        )
+        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
+            **vision_neuron_config
+        )
+
+        text_neuron_config = copy.deepcopy(default_neuron_config)
+        text_neuron_config.update(override_neuron_config.get("text_neuron_config", {}))
+        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
+            **text_neuron_config
+        )
+
+        config = neuronx_model_cls.get_config_cls()(
+            text_neuron_config=text_neuron_config,
+            vision_neuron_config=vision_neuron_config,
+            load_config=load_pretrained_config(model_name_or_path),
+        )
+
+        success, compiled_model_path, _ = self._load_weights_common(
+            model_name_or_path, neuronx_model_cls, config=config, **kwargs
+        )
+
+        if not success:
+            if not os.path.exists(model_name_or_path):
+                model_name_or_path = self._save_pretrained_model(model_name_or_path)
+
+            self._compile_and_load_model(
+                model_name_or_path, neuronx_model_cls, config, compiled_model_path
+            )
+
+        # Load tokenizer to get vision token ID
+        from transformers import AutoTokenizer
+
+        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
+        self.vision_token_id = tokenizer(
+            "<|image|>", add_special_tokens=False
+        ).input_ids[0]
+        return success, compiled_model_path
+

 def _get_model_configs(config: PretrainedConfig) -> str:
     logger.debug("PretrainedConfig: %s", config)
```

#### 2.4 Map `NeuronGemma3ForConditionalGeneration` to corresponding HuggingFace model class in `vllm_neuron/worker/neuronx_distributed_model_runner.py`


```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -775,6 +830,8 @@ def get_neuron_model(
         model = NeuronPixtralForCausalLM(model_config.hf_config)
     elif architecture == "Llama4ForConditionalGeneration":
         model = NeuronLlama4ForCausalLM(model_config.hf_config)
+    elif architecture == "Gemma3ForConditionalGeneration":
+        model = NeuronGemma3ForConditionalGeneration(model_config.hf_config)
     else:
         model = NeuronCausalLM(model_config.hf_config)
```


#### 2.5 Add Gemma3 to the list of models that use the Llama4 multi-modal data processor

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -1067,7 +1067,7 @@ class NeuronxDistributedModelRunner(LoRAModelRunnerMixin):

         if self.model.model.config.model_type == "llava":
             mm_kwargs = self._process_multi_modal_data_neuron_llava(mm_kwargs)
-        elif self.model.model.config.model_type == "llama4":
+        elif self.model.model.config.model_type in ['llama4', 'gemma3']:
             pass  # llama4 doesn't require special processing
         else:
             raise NotImplementedError(
```

### 3. Run inference

#### 3.1 Offline Inference

```bash
PYTHONPATH="$PWD/contrib/models/gemma3-vision/src:src:$PYTHONPATH" run python contrib/models/gemma3-vision/vllm/run_offline_inference.py
```

#### 3.2 Online Inference

1. Start the vLLM server:

```bash
PYTHONPATH="$PWD/contrib/models/gemma3-vision/src:src:$PYTHONPATH" bash contrib/models/gemma3-vision/vllm/start-vllm-server.sh
```

2. Query the running server:

```bash
PYTHONPATH="$PWD/contrib/models/gemma3-vision/src:src:$PYTHONPATH" run python contrib/models/gemma3-vision/vllm/run_online_inference.py
```
