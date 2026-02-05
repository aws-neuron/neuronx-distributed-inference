import os
import pytest
import torch
import torch_xla.core.xla_model as xm
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import destroy_mp, init_cpu_env

from gemma3_vision.modeling_gemma3_vision import NeuronGemma3MultiModalProjector
from test.utils import assert_tensor_all_close, mark_step, create_neuron_config, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES


def _cpu_setup(dtype):
    set_random_seed(0)
    os.environ.setdefault("NXD_CPU_MODE", "1")
    init_cpu_env()
    torch.set_default_dtype(dtype)
    torch.set_default_device("cpu")


@pytest.mark.parametrize("tolerances, compiler_flags", [
    (FP32_TOLERANCES, ["--model-type=transformer", "--auto-cast=none"]),
    (FP16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=fp16"]),
    (BF16_TOLERANCES, ["--model-type=transformer", "--auto-cast=matmult", "--enable-mixed-precision-accumulation", "--auto-cast-type=bf16"]),
    ])
def test_multimodal_projector(monkeypatch, base_compiler_flags, tolerances, compiler_flags, hf_config) -> None:
    monkeypatch.setenv("NEURON_CC_FLAGS", " ".join(base_compiler_flags + compiler_flags))

    image_size, patch_size = 448, 28
    num_patches = int((image_size/patch_size)**2)
    batch_size, max_seq_len, hidden_size = 2, 64, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    hf_config.vision_config.image_size = image_size
    hf_config.vision_config.patch_size = patch_size

    vision_outputs = torch.randn(batch_size, num_patches, hidden_size).to(dtype=inputs_dtype)

    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        torch_dtype=model_dtype,
        tp_degree=2,
        hf_config=hf_config
    )

    # --- CPU Reference Execution ---
    # Note: We explicitly set 'NXD_CPU_MODE' to force a CPU-only environment.
    #       This is critical because the module's initialization logic (in
    #       get_rmsnorm_cls) checks this variable to choose between the
    #       CPU and Neuron-specific RMSNorm implementations.
    _cpu_setup(model_dtype)
    mm_projector = NeuronGemma3MultiModalProjector(config=nrn_config).to(dtype=model_dtype)
    mm_projector.eval()

    with torch.no_grad():
        cpu_output = mm_projector(vision_outputs)

    # --- Neuron Device Execution ---
    # Note: Tear down CPU environment and switch to NeuronCore mode
    destroy_mp()
    os.environ.setdefault("NXD_CPU_MODE", "0")
    set_random_seed(0)

    with torch.no_grad():
        mm_projector_nrn = mm_projector.to(device=xm.xla_device())
        mark_step()
        nrn_output = mm_projector_nrn(vision_outputs.to(device=xm.xla_device()))
        mark_step()
        nrn_output = nrn_output.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="Multi modal projector outputs", computed_value=nrn_output, reference_value=cpu_output, rtol=rtol, atol=atol, equal_nan=True)


def test_nxdi_mm_projector_vs_transformers_implementation(random_seed, hf_config) -> None:
    image_size, patch_size = 448, 28
    num_patches = int((image_size/patch_size)**2)
    batch_size, max_seq_len, hidden_size = 2, 64, hf_config.vision_config.hidden_size
    inputs_dtype = model_dtype = torch.float32

    hf_config.vision_config.image_size = image_size
    hf_config.vision_config.patch_size = patch_size

    vision_outputs = torch.randn(batch_size, num_patches, hidden_size).to(dtype=inputs_dtype)

    # --- Set NxDI Model ---
    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        torch_dtype=model_dtype,
        tp_degree=2,
        hf_config=hf_config
    )

    mm_projector = NeuronGemma3MultiModalProjector(config=nrn_config).to(dtype=model_dtype)
    mm_projector.eval()
    mm_projector.to(device=xm.xla_device())

    # --- Set Transformers Model ---
    hf_config.vision_config.image_size = image_size
    hf_config.vision_config.patch_size = patch_size

    reference_model = Gemma3MultiModalProjector(config=hf_config).to(dtype=model_dtype)
    reference_model.load_state_dict(mm_projector.state_dict(), strict=True)
    reference_model.eval()

    with torch.no_grad():
        ref_output = reference_model(vision_outputs=vision_outputs)
        output = mm_projector(vision_outputs=vision_outputs.to(device=xm.xla_device()))
        output = output.cpu()

    rtol, atol = FP32_TOLERANCES.rtol, FP32_TOLERANCES.atol
    assert_tensor_all_close(test_objective="Multi modal projector outputs", computed_value=output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
