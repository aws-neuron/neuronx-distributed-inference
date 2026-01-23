Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

# vLLM Inference with Gemma3 on AWS Neuron

## Prerequisites
- AWS Neuron DLAMI or Neuron SDK installed
- Precompiled Gemma3 model artifacts

## Setup

### 1. Install vLLM
```bash
git clone -b 2.26.1 https://github.com/aws-neuron/upstreaming-to-vllm.git
cd upstreaming-to-vllm
# Skip if using Neuron DLAMI: pip install -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

### 2. Configure Gemma3 Support

Modify `upstreaming-to-vllm/vllm/model_executor/model_loader/neuronx_distributed.py`:

#### 2.1 Register Gemma3 class in `_NEURON_SUPPORTED_MODELS` 
```python
_NEURON_SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    ...
    "Gemma3ForConditionalGeneration":
    ("models.gemma3.modeling_gemma3",
     "NeuronGemma3ForCausalLM")
}
```

#### 2.2 Add `Gemma3ForConditionalGeneration` in the if-else clause of `get_neuron_model()` function
```python
    elif model_arch == "Gemma3ForConditionalGeneration":
        model = NeuronGemma3ForCausalLM(model_config.hf_config)
```

#### 2.3 Add `NeuronGemma3ForCausalLM` class
```python
class NeuronGemma3ForCausalLM(NeuronMllamaForCausalLM):

    def __init__(self, config: PretrainedConfig) -> None:
        """
        Compared to NeuronMllamaForCausalLM,
        1. Set self.on_device_sampling_disabled based on the env variable
        2. Add vocab_size to HF config's top-level
        """
        super().__init__(config)
        # has_image is the only multimodal input that is used in
        # token-generation
        # This is a cache (on CPU) that saves has_image data per sequence id
        # The number of entries in this cache is <= Batch-Size
        self.has_image_cache: dict[int, torch.Tensor] = {}
        self.config = config
        self.config.vocab_size = config.get_text_config().vocab_size
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, logits_as_input=True)

        self.on_device_sampling_disabled = bool(
            int(os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0")))
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()
            
        # Lazy initialized
        self.model: nn.Module
        self.is_reorder_needed: bool = True
    
    def sample(self, hidden_states, sampling_metadata):
        """
        Compared to NeuronMllamaForCausalLM,
        1. Remove the first input (None) from self.sampler.forward()
        """
        if not self.on_device_sampling_disabled:
            with torch.profiler.record_function("sample"):
                hidden_states = hidden_states.flatten()
                res = []
                sample_idx = 0
                for seq_group in sampling_metadata.seq_groups:
                    seq_ids = seq_group.seq_ids
                    samples = []
                    for seq_id in seq_ids:
                        token_id = hidden_states[sample_idx].item()
                        samples.append(
                            SequenceOutput(
                                parent_seq_id=seq_id,
                                output_token=token_id,
                                logprobs={token_id: Logprob(token_id)}))
                        sample_idx += 1
                    res.append(
                        CompletionSequenceGroupOutput(samples=samples,
                                                      prompt_logprobs=None))
                next_tokens = SamplerOutput(outputs=res)
        else:
            next_tokens = self.sampler(hidden_states, sampling_metadata)
        return next_tokens
       
    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                input_block_ids: torch.Tensor,
                sampling_params,
                images_flattened: torch.Tensor = None,
                vision_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        Copy of NeuronLlama4ForCausalLM.forward, with minor changes in logger messages.
        """
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            logger.info(f"pixel_values.shape = {pixel_values.shape}")
            # pixel_values = pixel_values.permute((1, 0, 2, 3, 4))
            bsz, n_chunks, n_channels, h, w = pixel_values.shape  # (1, 5, 3, 336, 336)
            pixel_values = pixel_values.reshape(bsz * n_chunks, n_channels, h, w)  # (5, 3, 336, 336)
            pixel_values = pixel_values.to(torch.bfloat16)
            if vision_mask is None:
                vision_mask = (
                    input_ids == self.config.image_token_index).unsqueeze(-1)

        if vision_mask is not None:
            vision_mask = vision_mask.to(torch.bool)

        origin_input_block_ids = input_block_ids
        if self.is_reorder_needed:
            # sort block ids sequentially for perf/neuron support reasons
            input_block_ids, sorted_indices = torch.sort(input_block_ids)
            input_ids = torch.index_select(input_ids, 0, sorted_indices)
            positions = torch.index_select(positions, 0, sorted_indices)
            sampling_params = torch.index_select(sampling_params, 0,
                                                 sorted_indices)

        if input_ids.shape[0] != sampling_params.shape[0]:
            sampling_params = sampling_params[:input_ids.shape[0]]

        output = self.model(
            input_ids.to(torch.int32),
            attention_mask=None,
            position_ids=positions.to(torch.int32),
            seq_ids=input_block_ids.flatten().to(torch.int32),
            pixel_values=pixel_values,
            vision_mask=vision_mask,
            sampling_params=sampling_params,
        )
        if self.config.neuron_config.on_device_sampling_config:
            output = output.hidden_states
        else:
            output = output.logits[:, -1, :]

        if self.is_reorder_needed and origin_input_block_ids.shape[0] != 1:
            restored_indices = torch.argsort(sorted_indices)
            output = torch.index_select(output, 0, restored_indices)

        return output

    def load_weights(self, model_name_or_path: str, **kwargs):
        """
        Copy of NeuronLlama4ForCausalLM.forward, with minor changes in logger messages.
        """
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)
        
        if os.getenv("NEURON_COMPILED_ARTIFACTS") is not None:
            compiled_model_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        else:
            raise RuntimeError(
                "Gemma3 only supports loading a precompiled model at the moment. Please specify the compiled model path using the environment variable 'NEURON_COMPILED_ARTIFACTS'."
            )

        try:
            self.model = neuronx_model_cls(compiled_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.vision_token_id = tokenizer(
                "<|image|>", add_special_tokens=False).input_ids[0]
            self.model.load(compiled_model_path)
            self.config.neuron_config = self.model.config.neuron_config
            logger.info(
                "Successfully loaded precompiled model artifacts from %s",
                compiled_model_path)
            return
        except (FileNotFoundError, ValueError):
            logger.warning("Failed to load the model from %s.", compiled_model_path)
        raise RuntimeError(
            "Gemma3 only supports loading a precompiled model at the moment")
```

### 3. Update Model Runner

Modify `upstreaming-to-vllm/vllm/worker/neuronx_distributed_model_runner.py`:

Add `gemma3` model support in `process_multi_modal_data_neuron` function:
```python
    elif self.model.config.model_type in ['llama4', 'gemma3']:
        return mm_data
```

## Usage

### Offline Inference
```bash
python scripts/vllm_offline_inference.py
```

### Online Inference
1. Start the server:
```bash
chmod +x scripts/vllm_online_inference.sh
./scripts/vllm_online_inference.sh
```

2. Run client (in another terminal):
```bash
python scripts/vllm_online_inference.py
```

## Troubleshooting
- Ensure `NEURON_COMPILED_ARTIFACTS` points to valid compiled model
- Check port configuration matches between server and client
- For local images, use `allowed_local_media_path` parameter