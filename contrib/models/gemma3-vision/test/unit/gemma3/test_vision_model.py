import os
import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp

from gemma3_vision.modeling_gemma3_vision import NeuronGemma3VisionModel
from test.unit.gemma3.test_config import get_gemma3_config
from test.utils import assert_tensor_all_close, mark_step, cpu_setup, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_vision_model(monkeypatch, base_compiler_flags, tolerances, compiler_flags) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    # --- Input and Configurations ---
    batch_size, num_channels, image_size = 2, 3, 896
    inputs_dtype = model_dtype = torch.float32
    
    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    config = get_gemma3_config(
        tkg_batch_size=2,
        text_tp_degree=1,
        vision_tp_degree=1,
        text_seq_length=64,
        vision_seq_len=64
    )
    config.vision_config.image_size = image_size
    config.vision_config.num_hidden_layers = 5  # test with smaller network

    # --- CPU Reference Execution ---
    # Note: We explicitly set 'NXD_CPU_MODE' to force a CPU-only environment.
    #       This is critical because the module's initialization logic (in
    #       get_rmsnorm_cls) checks this variable to choose between the
    #       CPU and Neuron-specific RMSNorm implementations.
    cpu_setup(model_dtype)
    cpu_vision_model = NeuronGemma3VisionModel(config).to(dtype=model_dtype)
    cpu_vision_model.eval()

    with torch.no_grad():
        cpu_output = cpu_vision_model(pixel_values) 

    # --- Neuron Device Execution ---
    # Note: Tear down CPU environment and switch to NeuronCore mode
    destroy_mp()
    os.environ.setdefault("NXD_CPU_MODE", "0")
    set_random_seed(0)

    nrn_vision_model = NeuronGemma3VisionModel(config).to(dtype=model_dtype)
    nrn_vision_model.eval()

    with torch.no_grad():
        nrn_vision_model = nrn_vision_model.to(device=xm.xla_device())
        mark_step()
        nrn_output = nrn_vision_model(pixel_values.to(device=xm.xla_device()))
        mark_step()
        nrn_output = nrn_output.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Gemma3 vision model outputs", computed_value=nrn_output, reference_value=cpu_output, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_vision_model_vs_transformers_implementation(random_seed) -> None:
    batch_size, num_channels, image_size = 2, 3, 896
    inputs_dtype = model_dtype = torch.float32
    
    pixel_values = torch.randn(batch_size, num_channels, image_size, image_size).to(dtype=inputs_dtype)

    # --- Set NxDI Model ---
    config = get_gemma3_config(
        tkg_batch_size=2,
        text_tp_degree=1,
        vision_tp_degree=1,
        text_seq_length=64,
        vision_seq_len=64
    )
    config.vision_config.image_size = image_size
    config.vision_config.num_hidden_layers = 5  # test with smaller network

    vision_model = NeuronGemma3VisionModel(config=config).to(dtype=model_dtype)
    vision_model.eval()
    vision_model.to(device=xm.xla_device()) 
    
    # --- Set Transformers Model ---
    hf_config = AutoConfig.from_pretrained("google/gemma-3-27b-it")  # nosec B615
    hf_config.vision_config.image_size = image_size
    hf_config.vision_config.num_hidden_layers = 5  # test with smaller network

    reference_model = Gemma3ForConditionalGeneration(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(vision_model.state_dict(), strict=False)
    reference_model.eval()
    
    with torch.no_grad():
        # reference model Gemma3ForConditionalGeneration includes a language model (LM)
        # use get_image_features() to pass the input pixel through vision_tower and multi_modal_projector only (exclude LM)
        ref_output = reference_model.get_image_features(pixel_values)
        output = vision_model(pixel_values.to(device=xm.xla_device()))
        output = output.cpu()

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Gemma3 vision model outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
