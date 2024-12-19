import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed_inference.models.mllama.encoder_utils import (
    build_attention_mask_gen_vectors,
    expand_num_tokens_to_mult8,
)
from neuronx_distributed_inference.models.mllama.modeling_mllama_vision import NeuronImageAttention
from neuronx_distributed_inference.models.mllama.utils import get_negative_inf_value

from .test_utils import load_checkpoint, logger, save_checkpoint, setup_debug_env, trace_nxd_model

NUM_ITERATIONS = 10
VISION_SEQ_LEN = 1601

VISION_HIDDEN_DIM = 1280
NUM_CHUNKS = 4
HEAD_DIM = 80
TORCH_DTYPE = torch.bfloat16
ATOL = 1e-2

BATCH_SIZE = 1
NUM_HEADS = 1


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


class SDPA_Meta(nn.Module):
    def forward(self, x, ar, Q, K, V):
        ve_attn_mask = build_encoder_attention_mask_meta(
            x, ar, VISION_SEQ_LEN, NUM_CHUNKS, NUM_HEADS
        )
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=ve_attn_mask)


class SDPA_Maskless(nn.Module):
    def __init__(self, use_flash_attention):
        self.use_flash_attention = use_flash_attention
        super().__init__()

    def forward(self, x, ar, Q, K, V):
        """
        x: (B, NUM_CHUNKS, VISION_SEQ_LEN, VISION_HIDDEN_DIM)
        ar: (B, 2)
        Q: (B, N, S, H)
        K: (B, N, S, H)
        V: (B, N, S, H)
        """
        bs, num_chunks, _, _ = x.shape
        _, num_heads, _, _ = Q.shape
        assert Q.shape[0] == bs and Q.shape[2] == (x.shape[2] * num_chunks)
        mask_gen_vectors = build_attention_mask_gen_vectors(
            x, ar, VISION_SEQ_LEN, num_chunks, num_heads
        )
        attn_output = NeuronImageAttention.perform_maskless_sdpa(
            Q,
            K,
            V,
            mask_gen_vectors=mask_gen_vectors,
            dtype=TORCH_DTYPE,
            use_flash_attention=self.use_flash_attention,
        )
        if self.use_flash_attention:
            # Output is BHDS from the kernel, convert to BHSD
            attn_output = attn_output.transpose(-1, -2)
        return attn_output


def get_example_inputs():
    x = torch.randn(BATCH_SIZE, NUM_CHUNKS, VISION_SEQ_LEN, VISION_HIDDEN_DIM, dtype=TORCH_DTYPE)
    x, _ = expand_num_tokens_to_mult8(x)
    ar = torch.tensor([1, 1], dtype=torch.int32).view(1, 2).expand(BATCH_SIZE, -1)
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, NUM_CHUNKS * x.shape[2], HEAD_DIM, dtype=TORCH_DTYPE)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, NUM_CHUNKS * x.shape[2], HEAD_DIM, dtype=TORCH_DTYPE)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, NUM_CHUNKS * x.shape[2], HEAD_DIM, dtype=TORCH_DTYPE)
    return x, ar, Q, K, V


def check_allclose(it, t1, t2, atol=None, log_debug=False):
    diff = t1 - t2
    atol_ = torch.max(torch.abs(diff)).item()
    rtol_ = torch.max(torch.abs(diff) / torch.abs(t1)).item()
    if log_debug:
        logger.info(f"Iteration {it}: atol={atol_}, rtol={rtol_}")
    tol_dict = {"atol": atol} if atol is not None else {}
    debug_str = f"Iteration {it}: atol={atol_}, rtol={rtol_}\n{diff}"
    assert torch.allclose(t1, t2, **tol_dict), debug_str


def test_maskless_sdpa():
    setup_debug_env()

    cpu_model_meta = SDPA_Meta()
    save_checkpoint(cpu_model_meta)
    cpu_model_maskless = SDPA_Maskless(use_flash_attention=False)
    cpu_model_maskless.load_state_dict(load_checkpoint())

    example_inputs = get_example_inputs()
    neuron_model_maskless = trace_nxd_model(
        SDPA_Maskless, example_inputs, tp_degree=1, use_flash_attention=True
    )

    aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
    for aspect_ratio in aspect_ratios:
        logger.info(f"Testing aspect ratio: {str(aspect_ratio)}")
        ar = torch.tensor(aspect_ratio, dtype=torch.int32).view(1, 2).expand(BATCH_SIZE, -1)
        for it in range(NUM_ITERATIONS):
            x, _, Q, K, V = get_example_inputs()

            output_meta = cpu_model_meta(x, ar, Q, K, V)
            output_maskless = cpu_model_maskless(x, ar, Q, K, V)
            check_allclose(it, output_meta, output_maskless)

            output_neuron_maskless = neuron_model_maskless(x, ar, Q, K, V)
            check_allclose(it, output_meta, output_neuron_maskless, atol=ATOL, log_debug=True)

        logger.info(f"Passed CPU and Device correctness tests ({NUM_ITERATIONS} iterations)\n")


if __name__ == "__main__":
    test_maskless_sdpa()
