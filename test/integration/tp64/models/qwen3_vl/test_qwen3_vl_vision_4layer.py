import os
import logging
import yaml
import pytest

from transformers import AutoProcessor

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.modules.checkpoint import load_state_dict

from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_vision import NeuronQwen3VLForImageEncoding
from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLInferenceConfig, Qwen3VLNeuronConfig

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


@pytest.mark.skip("Skip in NxDTest. Will use NIT test templates to add logit matching test.")
def test_original_vs_neuron():
    # Input processing
    processor = AutoProcessor.from_pretrained(model_path)
    text_prompt="How many images do you see? What do you see in these images?"
    image_path=["car.jpg", "car.jpg"]
    role='user'

    input_ids, attention_mask, vision_inputs = NeuronQwen3VLForImageEncoding.prepare_input_args(text_prompt, image_path, processor, role)


    """Neuron Vision Encoder Model"""
    config = Qwen3VLInferenceConfig(
        text_neuron_config=TEXT_NEURON_CONFIG,
        vision_neuron_config=VISION_NEURON_CONFIG,
        load_config=load_pretrained_config(model_path),
    )
    neuron_model = NeuronQwen3VLForImageEncoding(model_path, config)

    neuron_model.compile(traced_path)
    neuron_model.load(traced_path)

    neuron_hidden_states, neuron_ds_feat_list = neuron_model(**vision_inputs)

    """HuggingFace Vision Encoder Model"""
    hf_model = NeuronQwen3VLForImageEncoding.load_hf_model(model_path)
    sd = load_state_dict(model_path)
    hf_model.load_state_dict(sd, strict=False)
    hf_hidden_states, hf_ds_feat_list = hf_model(**vision_inputs)
    
    # Validate hidden_states
    passed, max_error = check_accuracy_embeddings(
        neuron_hidden_states.float(),
        hf_hidden_states.float(),
        plot_outputs=True,
        rtol=1e-2,
        atol=1e-5,
    )
    print(f"Golden and Neuron hidden_states match: {passed}, max relative error: {max_error}\n")
    assert passed

    # Validate deepstack features
    for i, (neuron_ds_feat, hf_ds_feat) in enumerate(zip(neuron_ds_feat_list, hf_ds_feat_list)):
        passed, max_error = check_accuracy_embeddings(
            neuron_ds_feat.float(),
            hf_ds_feat.float(),
            plot_outputs=False,
            rtol=1e-2,
            atol=1e-5,
        )
        print(f"Golden and Neuron {i}th deepstack feature match: {passed}, max relative error: {max_error}\n")
    assert passed


if __name__ == "__main__":
    test_original_vs_neuron()