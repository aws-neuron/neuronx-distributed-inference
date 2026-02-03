TODO: 
* Refactor/simplify NeuronGemma3ForCausalLM.load_weights?
* Add download weights from HF
* Add online inference

**Warning**: `vllm-neuron` shipped with Neuron 2.27.1 requires Torch 2.8. Make sure to use the appropriate Neuron Python virtual environment:

```bash
source ~/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
```

## Setup

### 1. Install vLLM
```bash
git clone --branch "0.2.2+lts" https://github.com/vllm-project/vllm-neuron.git
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
@@ -3,7 +3,7 @@ import torch
 
 NEURON_MULTI_MODAL_MODELS = [
     "MllamaForConditionalGeneration", "LlavaForConditionalGeneration",
-    "Llama4ForConditionalGeneration"
+    "Llama4ForConditionalGeneration", "Gemma3ForConditionalGeneration"
 ]
 
 TORCH_DTYPE_TO_NEURON_AMP = {
```

#### 2.2 Fix broken import in `vllm_neuron/worker/neuronx_distributed_model_loader.py`

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_loader.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_loader.py
@@ -44,7 +44,8 @@ from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                          SchedulerConfig, SpeculativeConfig)
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.v1.outputs import SamplerOutput
-from vllm.v1.sample import sampler as Sampler
+from vllm.v1.sample.sampler import Sampler
 
 from vllm_neuron.worker.constants import (NEURON_MULTI_MODAL_MODELS,
                                           TORCH_DTYPE_TO_NEURON_AMP)
```

#### 2.3 Add `NeuronGemma3ForCausalLM` class to `vllm_neuron/worker/neuronx_distributed_model_loader.py`

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_loader.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_loader.py
@@ -616,6 +617,62 @@ class NeuronLlama4ForCausalLM(NeuronMultiModalCausalLM):
                                **kwargs)
 
 
+class NeuronGemma3ForCausalLM(NeuronLlama4ForCausalLM):
+    
+    def load_weights(self, model_name_or_path: str, architecture: str,
+                     **kwargs):
+
+        import importlib
+        neuronx_module = importlib.import_module("gemma3_vision.modeling_gemma3")
+        neuronx_model_cls = getattr(neuronx_module, "NeuronGemma3ForCausalLM")
+        
+        default_neuron_config = kwargs["neuron_config"]
+        override_neuron_config = _validate_image_to_text_override_neuron_config(
+            kwargs["override_neuron_config"])
+
+        vision_neuron_config = copy.deepcopy(default_neuron_config)
+        vision_neuron_config.update(
+            override_neuron_config.get("vision_neuron_config", {}))
+        vision_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
+            **vision_neuron_config)
+
+        text_neuron_config = copy.deepcopy(default_neuron_config)
+        text_neuron_config.update(
+            override_neuron_config.get("text_neuron_config", {}))
+        text_neuron_config = neuronx_model_cls.get_neuron_config_cls()(
+            **text_neuron_config)
+
+        config = neuronx_model_cls.get_config_cls()(
+            text_neuron_config=text_neuron_config,
+            vision_neuron_config=vision_neuron_config,
+            load_config=load_pretrained_config(model_name_or_path))
+
+        # Pixtral model could hit OOB error when BS > 4
+        if architecture == "LlavaForConditionalGeneration":
+            if text_neuron_config.batch_size > 4 or text_neuron_config.tkg_batch_size > 4:
+                raise ValueError(
+                    "Neuron Pixtral model does not support batch size > 4 in vLLM v1 yet. This limitation will be addressed in future release."
+                )
+
+        success, compiled_model_path, _ = self._load_weights_common(
+            model_name_or_path, neuronx_model_cls, config=config, **kwargs)
+
+        if not success:
+            if not os.path.exists(model_name_or_path):
+                model_name_or_path = self._save_pretrained_model(
+                    model_name_or_path)
+
+            self._compile_and_load_model(model_name_or_path, neuronx_model_cls,
+                                         config, compiled_model_path)
+
+        # Load tokenizer to get vision token ID
+        from transformers import AutoTokenizer
+        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
+        self.vision_token_id = tokenizer("<|image|>",
+                                         add_special_tokens=False).input_ids[0]
+        return success, compiled_model_path
+
+
 def _get_model_configs(config: PretrainedConfig) -> str:
     logger.debug(f"PretrainedConfig: {config}")
```

#### 2.4 Map `NeuronGemma3ForCausalLM` to corresponding HuggingFace model class in `vllm_neuron/worker/neuronx_distributed_model_runner.py`

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_loader.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_loader.py
@@ -680,6 +737,8 @@ def get_neuron_model(model_config: ModelConfig,
         model = NeuronPixtralForCausalLM(model_config.hf_config)
     elif architecture == "Llama4ForConditionalGeneration":
         model = NeuronLlama4ForCausalLM(model_config.hf_config)
+    elif architecture == "Gemma3ForConditionalGeneration":
+        model = NeuronGemma3ForCausalLM(model_config.hf_config)
     else:
         model = NeuronCausalLM(model_config.hf_config)
```

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -702,7 +702,7 @@ class NeuronxDistributedModelRunner(LoRAModelRunnerMixin):
         if self.model.model.config.model_type == 'llava':
             mm_data_neuron = self._process_multi_modal_data_neuron_llava(
                 mm_data)
-        elif self.model.model.config.model_type == 'llama4':
+        elif self.model.model.config.model_type in ['llama4', 'gemma3']:
             mm_data_neuron = self._process_multi_modal_data_neuron_llama4(
                 mm_data)
         else:
```

#### 2.5 Add Gemma3 to the list of models that use the Llama4 multi-modal data processor

```diff
--- a/vllm_neuron/worker/neuronx_distributed_model_runner.py
+++ b/vllm_neuron/worker/neuronx_distributed_model_runner.py
@@ -702,7 +702,7 @@ class NeuronxDistributedModelRunner(LoRAModelRunnerMixin):
         if self.model.model.config.model_type == 'llava':
             mm_data_neuron = self._process_multi_modal_data_neuron_llava(
                 mm_data)
-        elif self.model.model.config.model_type == 'llama4':
+        elif self.model.model.config.model_type in ['llama4', 'gemma3']:
             mm_data_neuron = self._process_multi_modal_data_neuron_llama4(
                 mm_data)
         else:
```

### 3. Run inference

#### 3.1 Offline Inference

```bash
PYTHONPATH="/home/ubuntu/nxdi-gemma3-contribution/contrib/models/gemma3-vision/src:src:$PYTHONPATH" uv run python contrib/models/gemma3-vision/vllm/run_offline_inference.py 
```
