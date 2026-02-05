import pytest
import torch
import torch_xla

from gemma3_vision.modeling_gemma3_text import NeuronGemma3RMSNorm, Gemma3RMSNorm
from test.utils import assert_tensor_all_close, mark_step, BF16_TOLERANCES


@pytest.mark.parametrize("inputs_dtype, tolerances", [
    (torch.bfloat16, BF16_TOLERANCES),
    ])
def test_custom_vs_hf_rms_norm_implementation(random_seed, inputs_dtype, tolerances, hf_config) -> None:
    device = torch_xla.device()
    batch_size, sequence_length = 2, 16
    hidden_size, eps = hf_config.text_config.hidden_size, hf_config.text_config.rms_norm_eps

    x = torch.rand((batch_size, sequence_length, hidden_size), dtype=inputs_dtype)
    nrn_norm = NeuronGemma3RMSNorm(hidden_size=hidden_size, eps=eps)
    nrn_norm.eval()
    ref_norm = Gemma3RMSNorm(dim=hidden_size, eps=eps)
    ref_norm.load_state_dict(nrn_norm.state_dict(), strict=True)
    ref_norm.eval()

    x = x.to(device=device)
    ref_norm = ref_norm.to(device=device)
    nrn_norm = nrn_norm.to(device=device)

    with torch.no_grad():
        mark_step()
        ref_output = ref_norm(x)
        mark_step()
        nrn_output = nrn_norm(x)
        mark_step()

        ref_output = ref_output.cpu()
        nrn_output = nrn_output.cpu()

    rtol, atol = tolerances.rtol, tolerances.atol
    assert_tensor_all_close(test_objective="RMS Norm", computed_value=nrn_output, reference_value=ref_output, rtol=rtol, atol=atol, equal_nan=True)
