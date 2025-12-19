import copy
import os
import pytest

import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    _reduce_scatter_along_dim,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    destroy_model_parallel,
)
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import (
    GptOssNeuronConfig,
    NeuronGptOssAttention,
)
from .test_utils import get_rtol


DEFAULT_CONFIG = {
    "experts_per_token": 4,
    "head_dim": 64,
    "hidden_size": 3072,
    "initial_context_length": 4096,
    "intermediate_size": 3072,
    "model_type": "gpt_oss",
    "num_attention_heads": 64,
    "num_experts": 128,
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 32.0,
    "rope_scaling_factor": 32.0,
    "rope_theta": 150000.0,
    "vocab_size": 201088,
    "attention_bias": True,
    "rms_norm_eps": 1e-05,
    "use_polar_compatible_rope": True,
}

SEQ_LEN = 2048
TORCH_DTYPE = torch.float32
TP_DEGREE = 64

CKPT_DIR = "/tmp/nxdi_test/test_attn_module_cte/"
os.makedirs(CKPT_DIR, exist_ok=True)


def create_windowed_attention_mask(
    batch_size: int, seq_len: int, window_size: int
) -> torch.Tensor:
    """create a causal, window attention mask"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
    for i in range(seq_len):
        if i >= window_size:
            mask[i, : i - window_size + 1] = False
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask


def get_cte_config(
    tp_degree,
    cp_degree,
    attention_dp_degree,
    sp_enabled=False,
    attn_kernel_enabled=False,
    qkv_kernel_enabled=False,
    o_proj_kernel_enabled=False,
    seq_len=SEQ_LEN,
    torch_dtype=TORCH_DTYPE,
    **update_kwargs,
):
    neuron_config = GptOssNeuronConfig(
        tp_degree=tp_degree,
        batch_size=8,
        ctx_batch_size=1,
        tkg_batch_size=8,
        is_continuous_batching=True,
        max_context_length=seq_len,
        seq_len=seq_len,
        torch_dtype=torch_dtype,
        cp_degree=cp_degree,
        attention_dp_degree=attention_dp_degree,
        sequence_parallel_enabled=sp_enabled,
        logical_neuron_cores=2,
        attn_kernel_enabled=attn_kernel_enabled,  # Kernel disabled to use native compiler prefill attention
        fused_qkv=True,
        qkv_kernel_enabled=qkv_kernel_enabled,
        attn_block_tkg_nki_kernel_enabled=o_proj_kernel_enabled, # Enable this kernel to also trigger the o_proj kernel.
    )
    neuron_config.is_prefill_stage = True

    config_dict = copy.deepcopy(DEFAULT_CONFIG)
    config_dict.update(**update_kwargs)
    config = InferenceConfig(neuron_config, **config_dict)
    config.torch_dtype = torch_dtype

    return config


class AttentionBlock(torch.nn.Module):
    def __init__(self, inference_config: InferenceConfig):
        super().__init__()
        self.attention = NeuronGptOssAttention(inference_config)

    def forward(self, hidden_states, attention_mask, position_ids, rotary_freqs):
        if self.attention.sequence_parallel_enabled:
            hidden_states = _reduce_scatter_along_dim(
                hidden_states,
                1,
                xm.REDUCE_MAX,
                process_group=get_tensor_model_parallel_group(),
            )
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_freqs=rotary_freqs,
        )

        if self.attention.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                attention_output.hidden_states,
                1,
                process_group=get_tensor_model_parallel_group(),
            )
        else:
            hidden_states = attention_output.hidden_states

        return hidden_states


# Kernels don't respect the fp32 dtype, a small amount of increased error in those cases are acceptable.
@pytest.mark.parametrize(
    "tp_degree, cp_degree, sliding_window, sp_enabled, attn_kernel_enabled, qkv_kernel_enabled, o_proj_kernel_enabled, tolerances",
    [
        (64, 1, None, False, False, False, False, None),
        (64, 16, None, False, False, False, False, None),
        (64, 1, 128, False, False, False, False, None),
        (64, 16, 128, False, False, False, False, None),
        (64, 1, None, True, False, False, False, None),
        (64, 16, None, True, False, False, False, None),
        (64, 1, 128, True, False, False, False, None),
        (64, 16, 128, True, False, False, False, None),
        (64, 16, None, True, True, False, False, (0.008, 0.005)),
        (64, 16, None, True, True, True, False, (0.009, 0.005)),
        (64, 16, None, True, True, True, True, (0.009, 0.005)),
    ],
)
def test_attn_module(
    tp_degree, cp_degree, sliding_window, sp_enabled, attn_kernel_enabled, qkv_kernel_enabled, o_proj_kernel_enabled, tolerances
):
    attention_dp_degree = 8 if not sliding_window and cp_degree > 1 else 1
    device_config = get_cte_config(
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        attention_dp_degree=attention_dp_degree,
        sp_enabled=sp_enabled,
        sliding_window=sliding_window,
        attn_kernel_enabled=attn_kernel_enabled,
        qkv_kernel_enabled=qkv_kernel_enabled,
        o_proj_kernel_enabled=o_proj_kernel_enabled,
    )
    cpu_config = get_cte_config(
        tp_degree=1, cp_degree=1, attention_dp_degree=1, sliding_window=sliding_window
    )

    torch.manual_seed(0)
    cpu_model = AttentionBlock(cpu_config)

    state_dict = cpu_model.state_dict()
    state_dict["attention.rank_util.rank"] = torch.arange(
        0, TP_DEGREE, dtype=torch.int32
    )
    state_dict["attention.tkg_learned_sinks.sink"] = (
        state_dict["attention.learned_sinks.sink"].clone().contiguous()
    )
    for prefix in ["cte", "tkg"]:
        for suffix in ["weight", "bias"]:
            state_dict[f"attention.{prefix}_qkv_proj.Wqkv.{suffix}"] = (
                state_dict[f"attention.qkv_proj.Wqkv.{suffix}"].clone().contiguous()
            )
            state_dict[f"attention.{prefix}_o_proj.o_proj.{suffix}"] = (
                state_dict[f"attention.o_proj.o_proj.{suffix}"].clone().contiguous()
            )

    checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    torch.save(state_dict, checkpoint_path)

    hidden = torch.rand((1, SEQ_LEN, cpu_config.hidden_size))
    if sliding_window is None:
        attention_mask = (
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).view(1, 1, SEQ_LEN, SEQ_LEN).bool()
        )
    else:
        attention_mask = create_windowed_attention_mask(
            batch_size=1, seq_len=SEQ_LEN, window_size=sliding_window
        )
    position_ids = torch.arange(0, SEQ_LEN, dtype=torch.int32).unsqueeze(0)
    rotary_freqs = torch.rand((1, SEQ_LEN, 64))

    inputs = [(hidden, attention_mask, position_ids, rotary_freqs)]

    example_inputs = [
        (
            torch.ones((1, SEQ_LEN, cpu_config.hidden_size)),
            torch.ones((1, 1, SEQ_LEN, SEQ_LEN)).bool(),
            torch.arange(0, SEQ_LEN, dtype=torch.int32).unsqueeze(0),
            torch.ones((1, SEQ_LEN, 64)),
        )
    ]

    torch.manual_seed(0)
    module_neuron = build_module(
        AttentionBlock,
        example_inputs,
        tp_degree=TP_DEGREE,
        module_init_kwargs={"inference_config": device_config},
        checkpoint_path=checkpoint_path,
        compiler_args="--auto-cast=none",
        logical_nc_config=2,
    )

    if tolerances:
        rtol, atol = tolerances
    else:
        rtol = get_rtol(TORCH_DTYPE)
        atol = 1e-5

    validate_accuracy(
        neuron_model=module_neuron,
        cpu_callable=cpu_model,
        inputs=inputs,
        assert_close_kwargs={"rtol": rtol, "atol": atol},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--no-cov"])
