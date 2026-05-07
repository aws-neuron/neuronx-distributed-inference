import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

from gemma3_vision.modeling_gemma3_text import NeuronGemma3RotaryEmbedding
from test.utils import assert_tensor_all_close, mark_step, cpu_setup, create_neuron_config, FP32_TOLERANCES, FP16_TOLERANCES, BF16_TOLERANCES


@pytest.mark.parametrize("inputs_dtype, tolerances", [
    (torch.float32, FP32_TOLERANCES),
    (torch.bfloat16, BF16_TOLERANCES),
    ])
@pytest.mark.parametrize("position", [128, 1024, 2048, 4096, 6144, 8192])
def test_rope_global_vs_transformers_implementation(inputs_dtype, tolerances, position, hf_config) -> None:
    # --- Set NxDI Model ---
    batch_size, max_seq_len = 2, 64
    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        torch_dtype=inputs_dtype,
        tp_degree=1,
        hf_config=hf_config
    )

    partial_rotary_factor = getattr(nrn_config.text_config, "partial_rotary_factor", 1.0)
    dim = int(nrn_config.text_config.head_dim * partial_rotary_factor)
    max_position_embeddings = nrn_config.text_config.max_position_embeddings

    nrn_rope = NeuronGemma3RotaryEmbedding(
        dim=dim,
        max_position_embeddings=max_position_embeddings,
        base=nrn_config.text_config.rope_theta,
        scaling_type=nrn_config.text_config.rope_scaling["rope_type"],
        scaling_factor=nrn_config.text_config.rope_scaling["factor"],
    )

    # --- Set Transformers Model ---
    hf_text_config = hf_config.text_config
    reference_rope = Gemma3RotaryEmbedding(config=hf_text_config)

    # --- Inputs ---
    batch_size, sequence_length, num_heads, head_dim = 2, 1, 1, 128
    x = torch.randn(batch_size, num_heads, sequence_length, head_dim).to(dtype=inputs_dtype)
    position_ids = torch.full((batch_size, sequence_length), position, dtype=torch.int32)

    # --- Run Rope ---
    ref_cos, ref_sin = reference_rope(x, position_ids)
    cos, sin = nrn_rope(x, position_ids)

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="cos", computed_value=cos, reference_value=ref_cos, rtol=rtol, atol=atol, equal_nan=True)
    assert_tensor_all_close(test_objective="sin", computed_value=sin, reference_value=ref_sin, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("inputs_dtype, tolerances", [
    (torch.float32, FP32_TOLERANCES),
    (torch.bfloat16, BF16_TOLERANCES),
    ])
@pytest.mark.parametrize("position", [128, 1024, 2048, 4096, 6144, 8192])
def test_rope_local_vs_transformers_implementation(inputs_dtype, tolerances, position, hf_config) -> None:
    # --- Set NxDI Model ---
    batch_size, max_seq_len = 2, 64
    nrn_config = create_neuron_config(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        torch_dtype=inputs_dtype,
        tp_degree=1,
        hf_config=hf_config
    )

    partial_rotary_factor = getattr(nrn_config.text_config, "partial_rotary_factor", 1.0)
    dim = int(nrn_config.text_config.head_dim * partial_rotary_factor)
    max_position_embeddings = nrn_config.text_config.max_position_embeddings

    nrn_rope = NeuronGemma3RotaryEmbedding(
        dim=dim,
        max_position_embeddings=max_position_embeddings,
        base=nrn_config.text_config.rope_local_base_freq,
    )

    # --- Set Transformers Model ---
    hf_text_config = hf_config.text_config  # nosec B615
    hf_text_config.rope_theta = hf_text_config.rope_local_base_freq
    hf_text_config.rope_scaling = {"rope_type": "default"}

    reference_rope = Gemma3RotaryEmbedding(config=hf_text_config)

    # --- Inputs ---
    batch_size, sequence_length, num_heads, head_dim = 2, 1, 1, 128
    x = torch.randn(batch_size, num_heads, sequence_length, head_dim).to(dtype=inputs_dtype)
    position_ids = torch.full((batch_size, sequence_length), position, dtype=torch.int32)

    # --- Run Rope ---
    ref_cos, ref_sin = reference_rope(x, position_ids)
    cos, sin = nrn_rope(x, position_ids)

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="cos", computed_value=cos, reference_value=ref_cos, rtol=rtol, atol=atol, equal_nan=True)
    assert_tensor_all_close(test_objective="sin", computed_value=sin, reference_value=ref_sin, rtol=rtol, atol=atol, equal_nan=True)
