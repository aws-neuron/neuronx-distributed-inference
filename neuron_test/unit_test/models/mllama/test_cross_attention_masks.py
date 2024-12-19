import itertools
from typing import List

import torch

from neuronx_distributed_inference.models.mllama.modeling_mllama import NeuronLlamaCrossAttention

from .test_utils import load_checkpoint, logger, save_checkpoint, setup_debug_env, trace_nxd_model

VISION_SEQ_LEN = 1601
HIDDEN_DIM = 4096
TORCH_DTYPE = torch.float32


class Meta_XAttenMaskCalculator(torch.nn.Module):
    def __init__(self, max_num_chunks, torch_dtype) -> None:
        torch.nn.Module.__init__(self)
        self.max_num_chunks = max_num_chunks
        self.torch_dtype = torch_dtype

    def _get_full_row_masked_out_mask(
        self,
        attn_bias,
        negative_inf_value,
    ):
        """
        attn_bias should be a 4D tensor of shape [B, H, S1, S2]
        where B is the batch size, H is the number of heads,
        and S1/S2 are the sequence lengths. This returns
        a 4D tensor of shape [B, H, S1, 1] which stores boolean
        values which are 0 if the a full row in the last dimension
        contains negative infinity values, otherwise it's 1.
        """
        full_row_mask = (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]
        return full_row_mask

    def _pad_masks(
        self,
        all_masks: List[List[List[int]]],
        all_num_chunks: List[List[int]],
        total_len: int,
        max_num_chunks: int,
    ) -> torch.Tensor:
        dtype = torch.float32
        inf_value = torch.finfo(dtype).min

        bsz = len(all_masks)
        max_num_media = max([len(m) for m in all_masks])

        out_masks = torch.full(
            (bsz, total_len, max_num_media, max_num_chunks),
            inf_value,
            dtype=dtype,
        )
        for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
            for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
                if len(mask_elem) == 2:
                    mask_elem[1] = min(mask_elem[1], total_len)
                    if mask_elem[1] == -1:
                        mask_elem[1] = total_len
                    out_masks[idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks].fill_(
                        0.0
                    )
        return out_masks

    def _get_xattn_mask(self, total_len, vision_tokens, vision_masks, num_chunks, has_image):
        cross_attention_masks = self._pad_masks(
            vision_masks,
            num_chunks,
            total_len,
            self.max_num_chunks,
        )
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=3)
            .view(bsz, ntext, -1)
            .unsqueeze(1)
        )

        negative_inf_value = torch.finfo(vision_tokens.dtype).min
        full_text_row_masked_out_mask = self._get_full_row_masked_out_mask(
            cross_attention_masks,
            negative_inf_value,
        )
        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(device=vision_tokens.device, dtype=vision_tokens.dtype),
            full_text_row_masked_out_mask,
        )

    def forward(self, total_len, vision_tokens, vision_masks, num_chunks, has_image):
        return self._get_xattn_mask(total_len, vision_tokens, vision_masks, num_chunks, has_image)


class NXD_XAttenMaskCalculator(NeuronLlamaCrossAttention):
    def __init__(self, max_num_chunks, torch_dtype) -> None:
        torch.nn.Module.__init__(self)
        self.max_num_chunks = max_num_chunks
        self.torch_dtype = torch_dtype

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        # skip calling the parent model's preshard_hook
        return model_state_dict

    def forward(self, total_len, vision_tokens, vision_masks, num_chunks, has_image):
        return self._get_xattn_mask(total_len, vision_tokens, vision_masks, num_chunks, has_image)


def get_example_inputs():
    total_len = torch.tensor([512])
    vision_tokens = torch.rand([1, 1, 4, VISION_SEQ_LEN, HIDDEN_DIM], dtype=TORCH_DTYPE)
    vision_masks = torch.tensor([[[5, -1]]])
    num_chunks = torch.tensor([[4]])
    has_image = torch.tensor([1])
    return total_len, vision_tokens, vision_masks, num_chunks, has_image


def test_llama_mm_cross_attention_mask():
    setup_debug_env()

    init_args = dict(
        max_num_chunks=4,
        torch_dtype=TORCH_DTYPE,
    )

    cpu_model_meta = Meta_XAttenMaskCalculator(**init_args)
    save_checkpoint(cpu_model_meta)
    cpu_model = NXD_XAttenMaskCalculator(**init_args)
    cpu_model.load_state_dict(load_checkpoint())

    example_inputs = get_example_inputs()
    total_len, vision_tokens, vision_masks, num_chunks, has_image = example_inputs
    nxd_xatten_calculator_model_cls = NXD_XAttenMaskCalculator
    neuron_model = trace_nxd_model(
        nxd_xatten_calculator_model_cls, example_inputs, tp_degree=1, **init_args
    )

    # Test across multiple configurations
    total_len_list = [512]
    vision_token_index_list = [8, 12, 100]
    has_image_list = [0, 1]
    num_chunks_list = [1, 4]
    for tl, vt, hi, nc in itertools.product(
        total_len_list, vision_token_index_list, has_image_list, num_chunks_list
    ):
        logger.info(
            f"Testing total_len: {total_len}, vision_masks: {vt}, num_chunks: {nc}, has_image: {has_image}"
        )
        total_len = torch.tensor([tl])
        vision_masks = torch.tensor([[[vt, -1]]])
        num_chunks = torch.tensor([[nc]])
        has_image = torch.tensor([hi])

        meta_test_inputs = (total_len, vision_tokens, vision_masks, num_chunks, has_image)
        if not has_image:
            # in Meta's implementation, and empty vision_tokens
            # and empty vision_masks are used if input has no image
            vision_tokens_meta = torch.rand([1, 0, 4, 1601, 4096])
            vision_masks_meta = torch.tensor([[]])
            meta_test_inputs = (
                total_len,
                vision_tokens_meta,
                vision_masks_meta,
                num_chunks,
                has_image,
            )

        # Get Meta's output on CPU
        mask1_cpu, mask2_cpu = cpu_model(*meta_test_inputs)

        # Get NxD's output on CPU
        mask1_neuron_cpu, mask2_neuron_cpu = cpu_model(*meta_test_inputs)

        assert torch.allclose(mask1_cpu, mask1_neuron_cpu)
        assert torch.allclose(mask2_neuron_cpu, mask2_neuron_cpu)
        logger.info("Correctness test passing on CPU.")

        # Get NXD's output on TRN
        vision_masks = torch.tensor([[[vt, -1]]])
        nxd_test_inputs = (total_len, vision_tokens, vision_masks, num_chunks, has_image)
        mask1_neuron, mask2_neuron = neuron_model(*nxd_test_inputs)

        logger.info(f"{mask1_cpu.shape}, {mask1_neuron_cpu.shape}, {mask1_neuron.shape}")
        logger.info(f"{mask1_cpu.sum()}, {mask1_neuron_cpu.sum()}, {mask1_neuron.sum()}")
        logger.info(f"{mask2_cpu.shape}, {mask2_neuron_cpu.shape}, {mask2_neuron.shape}")
        logger.info(f"{mask2_cpu.sum()}, {mask2_neuron_cpu.sum()}, {mask2_neuron.sum()}")

        # Check accuracy
        if has_image:
            assert torch.allclose(
                mask1_neuron, mask1_cpu
            ), f"Failed torch.allclose for case total_len: {total_len}, vision_masks: {vt}, num_chunks: {nc},  has_image: {has_image}, \n CPU: {mask1_cpu}, \n Neuron{mask1_neuron}"
            assert torch.allclose(
                mask2_neuron, mask2_cpu
            ), f"Failed torch.allclose for case total_len: {total_len}, vision_masks: {vt}, num_chunks: {nc},  has_image: {has_image}, \n CPU: {mask2_cpu}, \n Neuron{mask2_neuron}"
        else:
            # If no image, returned masks should be of all zeros
            assert torch.allclose(mask2_neuron, mask2_cpu)
            assert torch.allclose(mask1_neuron, torch.zeros_like(mask1_neuron))
            assert mask1_cpu.nelement() == 0
        logger.info("Correctness test passing on device.\n")

    logger.info("ALL TESTS PASSING")


if __name__ == "__main__":
    test_llama_mm_cross_attention_mask()
