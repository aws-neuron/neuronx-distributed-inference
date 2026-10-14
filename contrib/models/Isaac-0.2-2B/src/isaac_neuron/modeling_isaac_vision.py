# Copyright 2025 © Amazon.com and Affiliates
"""Isaac vision model for NxDI: SigLIP2 encoder + pixel shuffle + 2-layer MLP projector.

Isaac's vision pipeline:
  pixel_values -> SigLIP2 encoder -> pixel_shuffle (2x2, 1152->4608) -> MLP projector (4608->2048)

The MLP projector is a 2-layer network: Linear(4608->18432) -> SiLU -> Linear(18432->2048).
No bias terms, ~122M parameters.

Pixel shuffle is a deterministic CPU-side operation (channel concatenation of 2x2 patch groups).
"""

import logging
from typing import List, Tuple

import torch
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import (
    Llama4VisionModelWrapper,
)
from neuronx_distributed_inference.modules.async_execution import is_ranked_io

from isaac_neuron.siglip.modeling_siglip import NeuronSiglipVisionModel
from isaac_neuron.utils import pixel_shuffle_varlen

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NeuronIsaacMultiModalProjector(nn.Module):
    """Isaac's 2-layer MLP projector: Linear -> SiLU -> Linear.

    Maps pixel-shuffled vision features (4608-dim) to text hidden size (2048-dim).
    No bias terms on either linear layer.

    HF weight keys:
        model.vision_embedding.1.weight  -> projector_fc1.weight  (4608, 18432)
        model.vision_embedding.2          -> SiLU (no weights)
        model.vision_embedding.3.weight  -> projector_fc2.weight  (18432, 2048)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        vision_hidden = config.vision_config.hidden_size  # 1152
        pixel_shuffle_scale = getattr(config, "pixel_shuffle_scale", 2)
        projector_input_dim = vision_hidden * (pixel_shuffle_scale**2)  # 4608

        # Isaac uses intermediate_size from vision config for the projector
        # The HF model has: Linear(4608, 18432) -> SiLU -> Linear(18432, 2048)
        projector_intermediate = getattr(
            config,
            "projector_intermediate_size",
            projector_input_dim * 4,  # 18432
        )
        text_hidden = config.text_config.hidden_size  # 2048

        self.fc1 = nn.Linear(projector_input_dim, projector_intermediate, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(projector_intermediate, text_hidden, bias=False)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        """Forward pass: project vision features to text embedding space.

        Args:
            vision_outputs: (batch, num_patches, 4608) pixel-shuffled features

        Returns:
            (batch, num_patches, 2048) projected embeddings
        """
        hidden = self.fc1(vision_outputs)
        hidden = self.act(hidden)
        hidden = self.fc2(hidden)
        return hidden


class NeuronIsaacVisionModel(nn.Module):
    """Isaac vision model: SigLIP2 encoder + pixel shuffle + MLP projector.

    Full pipeline:
        pixel_values -> SigLIP2 -> pixel_shuffle(scale=2) -> MLP projector -> vision_embeddings
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.pixel_shuffle_scale = getattr(config, "pixel_shuffle_scale", 2)

        logger.info(f"NeuronIsaacVisionModel: vision_config={vars(self.vision_config)}")

        # SigLIP2 vision encoder (reused from Gemma3-vision contrib)
        self.vision_encoder = NeuronSiglipVisionModel(self.vision_config)

        # MLP projector (2-layer with SiLU)
        self.multi_modal_projector = NeuronIsaacMultiModalProjector(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Generate vision embeddings from pixel values.

        Args:
            pixel_values: (batch, num_channels, image_size, image_size)

        Returns:
            vision_embeddings: (batch, num_vision_tokens, text_hidden_size)
                where num_vision_tokens = (image_size / patch_size)^2 / pixel_shuffle_scale^2
        """
        # SigLIP2 encoder
        encoder_output = self.vision_encoder(pixel_values).last_hidden_state
        logger.info(f"encoder_output.shape={encoder_output.shape}")

        # Pixel shuffle: merge 2x2 patches by channel concatenation
        # (batch, num_patches, 1152) -> (batch, num_patches/4, 4608)
        shuffled = pixel_shuffle_varlen(encoder_output, scale=self.pixel_shuffle_scale)
        logger.info(f"pixel_shuffle output.shape={shuffled.shape}")

        # MLP projector: (batch, num_patches/4, 4608) -> (batch, num_patches/4, 2048)
        projected = self.multi_modal_projector(shuffled)
        logger.info(f"projected_embedding.shape={projected.shape}")

        return projected


class IsaacVisionModelWrapper(Llama4VisionModelWrapper):
    """Neuron ModelWrapper for Isaac's vision model.

    Inherits from Llama4VisionModelWrapper (same as Gemma3).
    Generates input shapes for trace and compilation.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate example inputs for vision encoder tracing.

        Returns:
            List of (pixel_values,) tuples for each bucket.
        """
        inputs = []
        for bucket in self.neuron_config.buckets:
            pixel_values = torch.ones(
                [
                    self.neuron_config.batch_size,
                    self.config.vision_config.num_channels,
                    self.config.vision_config.image_size,
                    self.config.vision_config.image_size,
                ],
                dtype=self.config.neuron_config.torch_dtype,
            )
            inputs.append((pixel_values,))
        return inputs

    def forward(self, *args):
        """Forward pass for vision encoder wrapper.

        Handles batch size padding when input batch < compiled batch.
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() first."
            )

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        pixel_values = args[0]
        input_batch_size = pixel_values.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            return self._forward(*args)

        cur_batch = 0
        outputs = []

        logging.debug(
            f"input_batch_size={input_batch_size}, compiled_batch_size={self.neuron_config.batch_size}"
        )

        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                batch_args = [
                    arg[cur_batch : cur_batch + self.neuron_config.batch_size]
                    for arg in args
                ]
                batch_args = self.vllm_cte_repadding(batch_args)
                output = self._forward(*batch_args)
            else:
                output = self._forward_with_pad(
                    *[
                        arg[cur_batch:input_batch_size]
                        if not is_ranked_io(arg)
                        else arg
                        for arg in args
                    ]
                )
            outputs.append(output)
            cur_batch += self.neuron_config.batch_size

        return output

    def _forward_with_pad(self, *args):
        """Forward with batch padding for undersized inputs."""

        def pad_helper(tensor, pad_type="fill_0", batch_sort_indices=None):
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size

            def repeat_first_batchline(tensor, padded_shape):
                return tensor[0].repeat(padded_shape[0], 1, 1, 1).to(tensor.dtype)

            def fill_value_tensor(value):
                return lambda tensor, padded_shape: torch.full(
                    padded_shape, fill_value=value, dtype=tensor.dtype
                )

            PAD_TYPES = {
                "repeat_first_batchline": repeat_first_batchline,
                "fill_0": fill_value_tensor(0),
                "fill_1": fill_value_tensor(1),
                "fill_-1": fill_value_tensor(-1),
            }

            padded_tensor = PAD_TYPES[pad_type](tensor, padded_shape)
            padded_tensor[: tensor.shape[0]] = tensor

            if batch_sort_indices is not None:
                padded_tensor = torch.index_select(padded_tensor, 0, batch_sort_indices)

            return padded_tensor

        pixel_values = args[0]
        orig_batch_size = pixel_values.shape[0]

        padded_args = []
        for arg in args:
            if is_ranked_io(arg):
                padded_args.append(arg)
            else:
                padded_arg = pad_helper(
                    arg,
                    pad_type="repeat_first_batchline",
                    batch_sort_indices=None,
                )
                padded_args.append(padded_arg)

        outputs = self._forward(*padded_args)
        return outputs[:orig_batch_size]
