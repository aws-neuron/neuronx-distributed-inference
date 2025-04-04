import math

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.mllama.modeling_mllama_vision import TilePositionEmbedding

from .test_utils import load_checkpoint, logger, save_checkpoint, setup_debug_env, trace_nxd_model

VISION_SEQ_LEN = 1601
VISION_HIDDEN_DIM = 1280
MAX_NUM_CHUNKS = 4
TORCH_DTYPE = torch.float32


class TilePositionEmbeddingMeta(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width, dtype=TORCH_DTYPE) / math.sqrt(width),
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=TORCH_DTYPE))

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(
            x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype
        )
        for idx, arx in enumerate(ar):
            w, h = arx
            out_pos_embed[idx, : w * h] = embed[:w, :h].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed
        return x


def get_example_inputs():
    x = torch.randn(1, MAX_NUM_CHUNKS, VISION_SEQ_LEN, VISION_HIDDEN_DIM, dtype=TORCH_DTYPE)
    ar = torch.tensor([1, 1], dtype=torch.int32).view(1, 2)
    return x, ar


def test_tile_pos_embed():
    setup_debug_env()

    init_args = dict(num_tiles=MAX_NUM_CHUNKS, width=VISION_HIDDEN_DIM, gated=True)
    cpu_model_meta = TilePositionEmbeddingMeta(**init_args)
    save_checkpoint(cpu_model_meta)
    cpu_model = TilePositionEmbedding(**init_args)
    cpu_model.load_state_dict(load_checkpoint())

    # Trace to get neuron model
    example_inputs = get_example_inputs()
    x, ar = example_inputs
    neuron_model = trace_nxd_model(TilePositionEmbedding, example_inputs, tp_degree=1, **init_args)

    # Test all possible aspect ratios (with max_num_chunks=4)
    aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
    for aspect_ratio in aspect_ratios:
        logger.info(f"Testing aspect ratio: {str(aspect_ratio)}")
        ar = torch.tensor(aspect_ratio, dtype=torch.int32).view(1, 2)

        # Compare Meta vs our implementation on CPU
        x_out_meta = cpu_model_meta(x, ar)
        x_out_cpu = cpu_model(x, ar)
        assert torch.allclose(x_out_meta, x_out_cpu)
        logger.info("Correctness test passing on CPU.")

        x_out_xla = neuron_model(x, ar)
        logger.info(
            f"{x_out_meta.shape}, {x.sum()}, {x_out_meta.sum()}, {x_out_cpu.sum()}, {x_out_xla.sum()}"
        )
        assert torch.allclose(x_out_meta, x_out_xla)
        logger.info("Correctness test passing on device.\n")

    logger.info("ALL TESTS PASSING")


if __name__ == "__main__":
    test_tile_pos_embed()
