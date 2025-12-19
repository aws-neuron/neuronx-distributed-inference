import os
import pytest
import torch

from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import CustomRMSNormV2Padded
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, _get_shared_checkpoint_path, _get_rand_weights
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings

# Set random seed for reproducibility
set_random_seed(0)

HIDDEN_SIZE = 3072
HIDDEN_SIZE_ACTUAL = 2880
EPS = 1e-5

TRACE_DIR="/tmp/nxd_inference/"

def checkpoint_loader_fn(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")

    sd["weight_unpadded"] = sd["weight"].clone()
    sd["weight"] = torch.nn.functional.pad(sd["weight"], (0, HIDDEN_SIZE - HIDDEN_SIZE_ACTUAL))

    return sd

@pytest.mark.parametrize("seq_len", [160, 640])
def test_gpt_oss_rms_norm(seq_len):
    """
    Test the CustomRMSNormV2Padded model against a reference implementation on CPU.
    """
    inputs = (torch.randn((
            1, # bs
            seq_len,
            HIDDEN_SIZE,
            )) * 0.05).to(torch.float32)

    cpu_model = GptOssRMSNorm(HIDDEN_SIZE_ACTUAL, EPS)
    ckpt_path = _get_shared_checkpoint_path(os.path.join(TRACE_DIR, "ckpt"))
    cpu_model = _get_rand_weights(cpu_model, ckpt_path)
    cpu_output = cpu_model(inputs[..., :HIDDEN_SIZE_ACTUAL])

    example_inputs = [(torch.randn(1, seq_len, HIDDEN_SIZE),)]

    neuron_model = build_module(
        module_cls=CustomRMSNormV2Padded,
        example_inputs=example_inputs,
        module_init_kwargs={"hidden_size": HIDDEN_SIZE,
                            "hidden_size_actual": HIDDEN_SIZE_ACTUAL,
                            "eps": EPS},
        compiler_workdir=os.path.join(TRACE_DIR, "compiler"),
        logical_nc_config=2,
        checkpoint_path=ckpt_path,
        checkpoint_loader_fn=checkpoint_loader_fn,
    )

    neuron_output = neuron_model(inputs)[..., :HIDDEN_SIZE_ACTUAL]

    passed, _ = check_accuracy_embeddings(neuron_output, cpu_output)
    assert passed, "Accuracy check failed"
