import torch
import torch.nn as nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.mllama.modeling_mllama_vision import VisionEncoder
from neuronx_distributed_inference.models.mllama.utils import META_CHECKPOINT, to_2tuple

from .test_utils import load_checkpoint, logger, save_checkpoint, setup_debug_env, trace_nxd_model

VISION_SEQ_LEN = 1601
VISION_HIDDEN_DIM = 1280
MAX_NUM_CHUNKS = 4
TORCH_DTYPE = torch.float32


class VisionEncoderPosEmbedOnly(VisionEncoder):
    def __init__(self, max_num_tiles, image_size, patch_size, width):
        nn.Module.__init__(self)
        self.config = InferenceConfig(neuron_config=None)
        self.config.checkpoint = META_CHECKPOINT
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width, dtype=TORCH_DTYPE)
        )
        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.grid_size[0] * self.grid_size[1] + 1,
                width,
                dtype=TORCH_DTYPE,
            )
        )
        # Don't initialize to zero, otherwise the gated_positional_embedding has no effect on output
        self.gated_positional_embedding_gate = nn.Parameter(torch.randn(1, dtype=TORCH_DTYPE))

    def forward(self, x, ar):
        return self.apply_positional_embedding(x, ar, ar_ids=None)


class VisionEncoderMeta(VisionEncoderPosEmbedOnly):
    def apply_positional_embedding(self, x, ar, ar_ids=None):
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (1 - self.gated_positional_embedding_gate.tanh())
        x = x.view(bsz, num_chunks, num_tokens, dim)
        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += _pos_embed * self.gated_positional_embedding_gate.tanh()
        return x


def get_example_inputs():
    x = torch.randn(1, MAX_NUM_CHUNKS, VISION_SEQ_LEN, VISION_HIDDEN_DIM, dtype=TORCH_DTYPE)
    ar = torch.tensor([1, 1], dtype=torch.int32).view(1, 2)
    return x, ar


def test_apply_pos_embed():
    setup_debug_env()

    init_args = dict(
        max_num_tiles=MAX_NUM_CHUNKS,
        image_size=560,
        patch_size=14,
        width=VISION_HIDDEN_DIM,
    )

    cpu_model_meta = VisionEncoderMeta(**init_args)
    save_checkpoint(cpu_model_meta)
    cpu_model = VisionEncoderPosEmbedOnly(**init_args)
    cpu_model.load_state_dict(load_checkpoint())

    # Trace to get neuron model
    example_inputs = get_example_inputs()
    x, ar = example_inputs
    neuron_model = trace_nxd_model(
        VisionEncoderPosEmbedOnly, example_inputs, tp_degree=1, **init_args
    )

    # Test all possible aspect ratios (with max_num_chunks=4)
    aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
    for aspect_ratio in aspect_ratios:
        print("Testing aspect ratio:", tuple(aspect_ratio))
        ar = torch.tensor(aspect_ratio, dtype=torch.int32).view(1, 2)

        # Compare Meta vs our implementation on CPU
        x_out_meta = cpu_model_meta(x, ar)
        x_out_cpu = cpu_model(x, ar)
        assert torch.allclose(x_out_meta, x_out_cpu)
        logger.info("Correctness test passing on CPU.")

        x_out_xla = neuron_model(x, ar)
        assert torch.allclose(x_out_meta, x_out_xla)
        logger.info(
            f"{x_out_meta.shape}, {x.sum()}, {x_out_meta.sum()}, {x_out_cpu.sum()}, {x_out_xla.sum()}"
        )
        logger.info("Correctness test passing on device.\n")

    logger.info("ALL TESTS PASSING")


if __name__ == "__main__":
    test_apply_pos_embed()
