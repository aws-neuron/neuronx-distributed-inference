import torch
import torch.nn as nn

from neuronx_distributed_inference.models.mllama.encoder_utils import (
    build_encoder_attention_mask,
    expand_num_tokens_to_mult8,
)
from neuronx_distributed_inference.models.mllama.utils import get_negative_inf_value

from .test_utils import load_checkpoint, logger, save_checkpoint, setup_debug_env, trace_nxd_model

VISION_SEQ_LEN = 1601
VISION_HIDDEN_DIM = 1280
MAX_NUM_CHUNKS = 4
TORCH_DTYPE = torch.float32


def build_encoder_attention_mask_meta(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Meta's implementation.
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


class EncoderAttentionMaskMeta(nn.Module):
    def forward(self, x: torch.Tensor, ar: torch.Tensor, ntok: int, num_chunks: int, n_heads: int):
        return build_encoder_attention_mask_meta(x, ar, ntok, num_chunks, n_heads)


class EncoderAttentionMaskNeuron(nn.Module):
    def forward(self, x: torch.Tensor, ar: torch.Tensor, ntok: int, num_chunks: int, n_heads: int):
        return build_encoder_attention_mask(x, ar, ntok, num_chunks, n_heads)


def get_example_inputs():
    vision_seq_len = torch.tensor(VISION_SEQ_LEN, dtype=torch.int32)
    vision_hidden_dim = torch.tensor(VISION_HIDDEN_DIM, dtype=torch.int32)
    num_chunks = torch.tensor(MAX_NUM_CHUNKS, dtype=torch.int32)
    n_heads = torch.tensor(1, dtype=torch.int32)
    ar = torch.tensor([1, 1], dtype=torch.int32).view(1, 2)
    x = torch.randn(1, num_chunks, vision_seq_len, vision_hidden_dim, dtype=TORCH_DTYPE)
    x, pad = expand_num_tokens_to_mult8(x)
    return x, ar, vision_seq_len, num_chunks, n_heads


def test_ve_attention_mask():
    setup_debug_env()

    cpu_model_meta = EncoderAttentionMaskMeta()
    save_checkpoint(cpu_model_meta)
    cpu_model = EncoderAttentionMaskNeuron()
    cpu_model.load_state_dict(load_checkpoint())

    # Trace to get neuron model
    example_inputs = get_example_inputs()
    x, ar, vision_seq_len, num_chunks, n_heads = example_inputs
    neuron_model = trace_nxd_model(EncoderAttentionMaskNeuron, example_inputs, tp_degree=1)

    # Test all possible aspect ratios (with max_num_chunks=4)
    aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
    for aspect_ratio in aspect_ratios:
        logger.info(f"Testing aspect ratio: {str(aspect_ratio)}")
        ar = torch.tensor(aspect_ratio, dtype=torch.int32).view(1, 2)

        # Compare Meta vs our implementation on CPU
        masks_meta = cpu_model_meta(x, ar, vision_seq_len, num_chunks, n_heads)
        masks_neuron_cpu = cpu_model(x, ar, vision_seq_len, num_chunks, n_heads)
        assert torch.allclose(masks_meta, masks_neuron_cpu)
        logger.info("Correctness test passing on CPU")

        masks_neuron_xla = neuron_model(x, ar, vision_seq_len, num_chunks, n_heads)
        assert torch.allclose(masks_meta, masks_neuron_xla)
        logger.info(
            f"{masks_meta.shape}, {(masks_meta == 0).sum()}, {(masks_neuron_cpu == 0).sum()}, {(masks_neuron_xla == 0).sum()}"
        )
        logger.info("Correctness test passing on device.\n")

    logger.info("ALL TESTS PASSING")


if __name__ == "__main__":
    test_ve_attention_mask()
