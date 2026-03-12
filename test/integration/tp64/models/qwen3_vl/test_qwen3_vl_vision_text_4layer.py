import os
import logging
import yaml
import pytest
import torch

from transformers import AutoProcessor, GenerationConfig

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits_v2
from neuronx_distributed_inference.modules.checkpoint import load_state_dict

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLInferenceConfig, Qwen3VLNeuronConfig, NeuronQwen3VLForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


model_path = "/shared/cache/checkpoints/Qwen3-VL/Qwen3-VL-8B-Thinking-4layer/"
traced_path = "/tmp/qwen3_vl_test/"

# Load neuron_configs from yaml
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURR_DIR, "model_configs/bs1_bf16_baseline.yaml"), "r") as f:
    MODEL_CONFIG = yaml.safe_load(f)
TEXT_NEURON_CONFIG = Qwen3VLNeuronConfig(**MODEL_CONFIG["inference_config"]["neuron_config"]["text_neuron_config"])
VISION_NEURON_CONFIG = Qwen3VLNeuronConfig(**MODEL_CONFIG["inference_config"]["neuron_config"]["vision_neuron_config"])

# Set env vars
ENV_VARS = MODEL_CONFIG["env_vars"]
for k, v in ENV_VARS.items():
    os.environ[k] = str(v)


@pytest.mark.skip("Skip in NxDTest. Will use NIT test templates to add logit matching test.")
def test_original_vs_neuron():
    # Input processing
    processor = AutoProcessor.from_pretrained(model_path)
    text_prompt = "How many images do you see? What do you see in these images?"
    image_path = ["dog.jpg", "car.jpg"]
    role='user'

    input_ids, attention_mask, vision_inputs = NeuronQwen3VLForCausalLM.prepare_input_args(text_prompt, image_path, processor, role)

    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id = 151643,
        eos_token_id = [151645],
        pad_token_id=151645, 
        output_logits=True,
        max_new_tokens=16,
    )

    """Get golden from running HuggingFace Model on CPU"""
    hf_model = NeuronQwen3VLForCausalLM.load_hf_model(model_path)
    sd = load_state_dict(model_path)
    hf_model.load_state_dict(sd, strict=False)
    hf_model.eval()

    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=vision_inputs["pixel_values"],
            image_grid_thw=vision_inputs["image_grid_thw"],
            max_new_tokens=generation_config.max_new_tokens,
            min_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=generation_config,
        )

    expected_logits = torch.stack(outputs.scores) # [seq_len, batch_size, vocab_size]
    print(f"Expected logits: {expected_logits.shape}")
    expected_logits = expected_logits[:generation_config.max_new_tokens, :, :]
    print(f"Expected logits: {expected_logits.shape}")

    expected_token_ids = expected_logits.argmax(dim=2).T
    expected_tokens = processor.batch_decode(
        expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"Expected Output: {expected_tokens} {expected_token_ids.shape} {expected_token_ids}")

    """Get Neuron Model"""
    config = Qwen3VLInferenceConfig(
        text_neuron_config=TEXT_NEURON_CONFIG,
        vision_neuron_config=VISION_NEURON_CONFIG,
        load_config=load_pretrained_config(model_path),
    )
    neuron_model = NeuronQwen3VLForCausalLM(model_path, config)

    neuron_model.compile(traced_path)
    neuron_model.load(traced_path)

    # Validations
    check_accuracy_logits_v2(
        neuron_model,
        expected_logits,
        inputs_input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=generation_config.max_new_tokens,
        additional_input_args=vision_inputs,
    )


if __name__ == "__main__":
    test_original_vs_neuron()