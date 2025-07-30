import copy
import logging
import os

import pytest
import torch
from transformers.models.llama4.modeling_llama4 import Llama4VisionEncoderLayer

from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import (
    PackingIndex,
    VisionEncoder,
    _TransformerBlock,
)
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.testing import build_module

from .test_config import get_llama4_config
from .test_utils import (
    cleanup_tmp_workdir,
    get_compiler_args,
    get_rand_weights,
    get_rtol,
    get_tmp_workdir,
    setup_debug_env,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()

# configs
NUM_LAYERS = 1


def get_freq_cis(config):
    # TODO: refactor VisionEncoder to make this a class method
    # This is in modeling_llama4_vision,py::VisionEncoder.__init__
    image_h, image_w = config.vision_config.image_size, config.vision_config.image_size
    patch_h, patch_w = config.vision_config.patch_size, config.vision_config.patch_size
    idx_h, idx_w = image_h // patch_h, image_w // patch_w
    img_idx = torch.arange(image_h * image_w // (patch_h * patch_w), dtype=torch.int32)
    img_idx = img_idx.reshape(idx_h * idx_w, 1)
    img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
    img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

    packed_img_idx = torch.empty(
        img_idx.shape[0],
        img_idx.shape[1],
        PackingIndex.NUM_METADATA - 1,
        dtype=torch.int32,
    )
    packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx_w
    packed_img_idx[:, :, PackingIndex.X] = img_idx % idx_w
    packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx_h)
    packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx_w)
    packed_img_idx[:, :, PackingIndex.IDX] = img_idx
    packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)

    # compute rope freqs
    rope_freq = VisionEncoder.get_rope_freqs(
        config.vision_config.hidden_size // config.vision_config.num_attention_heads // 2
    )
    freqs_x = VisionEncoder.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.X] + 1)
    freqs_y = VisionEncoder.compute_rope_freqs(rope_freq, packed_img_idx[:, :, PackingIndex.Y] + 1)
    freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
    # disable RoPE for padding and cls tokens
    neuron_freqs = freqs.masked_fill(packed_img_idx[:, :, PackingIndex.IDX, None] < 0, 0)

    # compute complex freqs
    original_freq_cis = torch.view_as_complex(
        torch.stack([torch.cos(neuron_freqs), torch.sin(neuron_freqs)], dim=-1)
    )

    return original_freq_cis, neuron_freqs


def convert_to_neuron_state_dict(config, tmp_workdir):
    new_state_dict = torch.load(os.path.join(tmp_workdir, "checkpoint.pt"), map_location="cpu")
    prefix = ""
    new_prefix = ""
    # copied from neuronx_distributed_inference.models.llama4.modeling_llama4_vision::NeuronLlama4ForImageEncoding.convert_hf_to_neuron_state_dict()
    if config.neuron_config.fused_qkv:
        q_weight = new_state_dict.pop(f"{prefix}self_attn.q_proj.weight")
        k_weight = new_state_dict.pop(f"{prefix}self_attn.k_proj.weight")
        v_weight = new_state_dict.pop(f"{prefix}self_attn.v_proj.weight")
        # the shape of weight matrix is 90-degree rotated, so here torch.cat along dim 0 instead of dim 1
        new_state_dict[f"{new_prefix}attn.qkv_proj.Wqkv.weight"] = torch.cat(
            [q_weight, k_weight, v_weight], dim=0
        )
        q_bias = new_state_dict.pop(f"{prefix}self_attn.q_proj.bias")
        k_bias = new_state_dict.pop(f"{prefix}self_attn.k_proj.bias")
        v_bias = new_state_dict.pop(f"{prefix}self_attn.v_proj.bias")
        new_state_dict[f"{new_prefix}attn.qkv_proj.Wqkv.bias"] = torch.cat(
            [q_bias, k_bias, v_bias], dim=0
        )
    else:
        new_state_dict[f"{new_prefix}attn.qkv_proj.q_proj.weight"] = new_state_dict.pop(
            f"{prefix}self_attn.q_proj.weight"
        )
        new_state_dict[f"{new_prefix}attn.qkv_proj.k_proj.weight"] = new_state_dict.pop(
            f"{prefix}self_attn.k_proj.weight"
        )
        new_state_dict[f"{new_prefix}attn.qkv_proj.v_proj.weight"] = new_state_dict.pop(
            f"{prefix}self_attn.v_proj.weight"
        )
        new_state_dict[f"{new_prefix}attn.qkv_proj.q_proj.bias"] = new_state_dict.pop(
            f"{prefix}self_attn.q_proj.bias"
        )
        new_state_dict[f"{new_prefix}attn.qkv_proj.k_proj.bias"] = new_state_dict.pop(
            f"{prefix}self_attn.k_proj.bias"
        )
        new_state_dict[f"{new_prefix}attn.qkv_proj.v_proj.bias"] = new_state_dict.pop(
            f"{prefix}self_attn.v_proj.bias"
        )

    # they will be renamed to {prefix}attn.o_proj.o_proj.* in GroupQueryAttention_O#preshard_hook
    new_state_dict[f"{new_prefix}attn.o_proj.weight"] = new_state_dict.pop(
        f"{prefix}self_attn.o_proj.weight"
    )
    new_state_dict[f"{new_prefix}attn.o_proj.bias"] = new_state_dict.pop(
        f"{prefix}self_attn.o_proj.bias"
    )

    # ln_1 and ln_2 mapping
    if f"{prefix}input_layernorm.weight" in new_state_dict:
        new_state_dict[f"{new_prefix}ln_1.weight"] = new_state_dict.pop(
            f"{prefix}input_layernorm.weight"
        )
    if f"{prefix}input_layernorm.bias" in new_state_dict:
        new_state_dict[f"{new_prefix}ln_1.bias"] = new_state_dict.pop(
            f"{prefix}input_layernorm.bias"
        )
    if f"{prefix}post_attention_layernorm.weight" in new_state_dict:
        new_state_dict[f"{new_prefix}ln_2.weight"] = new_state_dict.pop(
            f"{prefix}post_attention_layernorm.weight"
        )
    if f"{prefix}post_attention_layernorm.bias" in new_state_dict:
        new_state_dict[f"{new_prefix}ln_2.bias"] = new_state_dict.pop(
            f"{prefix}post_attention_layernorm.bias"
        )

    # mlp mapping: fc1 -> c_fc, fc2 -> c_proj
    if f"{prefix}mlp.fc1.weight" in new_state_dict:
        new_state_dict[f"{new_prefix}mlp.c_fc.weight"] = new_state_dict.pop(
            f"{prefix}mlp.fc1.weight"
        )
    if f"{prefix}mlp.fc1.bias" in new_state_dict:
        new_state_dict[f"{new_prefix}mlp.c_fc.bias"] = new_state_dict.pop(f"{prefix}mlp.fc1.bias")
    if f"{prefix}mlp.fc2.weight" in new_state_dict:
        new_state_dict[f"{new_prefix}mlp.c_proj.weight"] = new_state_dict.pop(
            f"{prefix}mlp.fc2.weight"
        )
    if f"{prefix}mlp.fc2.bias" in new_state_dict:
        new_state_dict[f"{new_prefix}mlp.c_proj.bias"] = new_state_dict.pop(f"{prefix}mlp.fc2.bias")

    torch.save(new_state_dict, os.path.join(tmp_workdir, "checkpoint.pt"))
    return new_state_dict


@pytest.mark.parametrize(
    "dtype, num_chunks",
    [
        pytest.param(
            dtype,
            num_chunks,
            id=f"dtype_{str(dtype).split('.')[-1]}_num_chunks_{num_chunks}",
        )
        for dtype in [torch.float32, torch.float16]
        for num_chunks in [8, 16, 88]
    ],
)
def test_original_cpu_vs_nxdi_neuron(dtype, num_chunks):
    # config
    config = get_llama4_config(dtype)
    config.vision_config.num_hidden_layers = NUM_LAYERS
    config.vision_config._attn_implementation = "eager"  # only used in HF ref impl
    # Make sure the vision module gets the correct neuron_config
    config.neuron_config = copy.deepcopy(config.vision_config.neuron_config)
    print(f"config.vision_config {vars(config.vision_config)}")
    print(f"config.vision_config.neuron_config {vars(config.vision_config.neuron_config)}")
    print(f"config.neuron_config {vars(config.neuron_config)}")

    # inputs
    seq_len = (config.vision_config.image_size // config.vision_config.patch_size) ** 2 + 1
    input_hidden = torch.randn([num_chunks, seq_len, config.vision_config.hidden_size], dtype=dtype)
    original_freq_cis, neuron_freqs = get_freq_cis(config)

    # use different example_inputs to trace, to rule out hardcoded tensor in HLO
    example_inputs = [
        (torch.ones_like(input_hidden), torch.ones_like(neuron_freqs)),
    ]

    # Get expected output by running HF code on CPU
    module_cpu = Llama4VisionEncoderLayer(config.vision_config)
    tmp_workdir = get_tmp_workdir()
    module_cpu = get_rand_weights(
        module_cpu, os.path.join(tmp_workdir, "checkpoint.pt"), dtype=dtype
    )
    expected_output = module_cpu(input_hidden, original_freq_cis, None)[
        0
    ]  # HF ref impl return outputs = (hidden_state,)
    logger.info(f"Got expected output {expected_output.shape}, {expected_output}")

    # Compile neuron module
    _ = convert_to_neuron_state_dict(config, tmp_workdir)
    module_neuron = build_module(
        _TransformerBlock,
        example_inputs,
        tp_degree=config.vision_config.neuron_config.tp_degree,
        module_init_kwargs={
            "config": config,
        },
        checkpoint_path=os.path.join(tmp_workdir, "checkpoint.pt"),
        compiler_args=get_compiler_args(),
    )
    neuron_output = module_neuron(input_hidden, neuron_freqs)

    logger.info(f"Got neuron output {neuron_output.shape}, {neuron_output}")

    logger.info(f"\nValidating accuracy for num_chunks {num_chunks}")
    passed, max_error = check_accuracy_embeddings(
        neuron_output, expected_output, plot_outputs=True, rtol=get_rtol(dtype), atol=1e-5
    )
    logger.info(f"Golden and Neuron outputs match: {passed}, max relative error: {max_error}")
    assert passed

    cleanup_tmp_workdir(tmp_workdir)
    return


if __name__ == "__main__":
    test_original_cpu_vs_nxdi_neuron(dtype=torch.float16, num_chunks=88)
