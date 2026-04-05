import logging
from typing import List, Tuple

import torch
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama4.modeling_llama4_vision import Llama4VisionModelWrapper
from neuronx_distributed_inference.modules.async_execution import is_ranked_io

from gemma3_vision.siglip.modeling_siglip import NeuronSiglipVisionModel
from gemma3_vision.modeling_gemma3_text import get_rmsnorm_cls

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NeuronGemma3MultiModalProjector(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = get_rmsnorm_cls()(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


class NeuronGemma3VisionModel(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        logger.info(f"in NeuronGemma3VisionModel self.vision_config {vars(self.vision_config)}")

        # TODO: data parallel optimization
        # self.global_rank = SPMDRank(world_size=self.neuron_config.world_size)
        # assert (
        #     self.neuron_config.world_size % self.neuron_config.tp_degree == 0
        # ), "Invalid parallel config. world_size should be a multiple of tp_degree"
        # self.dp_degree = self.neuron_config.world_size // self.neuron_config.tp_degree
        # self.data_parallel_enabled = self.dp_degree > 1
        # self.data_parallel_group = get_data_parallel_group()

        self.vision_encoder = NeuronSiglipVisionModel(self.vision_config)
        # multi_modal_projector need to read text model hidden_size, so we pass in the entire config to it
        self.multi_modal_projector = NeuronGemma3MultiModalProjector(self.config)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate vision embeddings from flattened pixel values.

        This function handles dynamic image shapes as well as multiple images by splitting each image
        into a number of fixed-size chunks. Afterwards, all chunks are stacked together on the batch dimension (dim=0)

        Args:
        pixel_values (Tensor): Vision pixel values of shape [num_chunks, 1(constant), num_chunnels, image_size, image_size]

        Returns:
        vision embeddings (Tensor): Vision embeddings (after projection) padded to the nearest bucket size.

        """
        # TODO: data parallel optimization
        # if self.data_parallel_enabled:
        #     dp_rank = get_dp_rank_spmd(self.global_rank.get_rank(), self.neuron_config.tp_degree)
        #     # split inputs along batch dim
        #     pixel_values = scatter_to_process_group_spmd(
        #         pixel_values,
        #         partition_dim=0,
        #         rank=dp_rank,
        #         process_group=self.data_parallel_group,
        #     )

        embedding = self.vision_encoder(pixel_values).last_hidden_state
        logger.info(f"embedding.shape {embedding.shape}")

        projected_embedding = self.multi_modal_projector(embedding)
        logger.info(f"projected_embedding.shape {projected_embedding.shape}")

        # TODO: data parallel optimization
        # if self.data_parallel_enabled:
        #     h_image_proj = gather_from_tensor_model_parallel_region_with_dim(
        #         h_image_proj, gather_dim=0, process_group=self.data_parallel_group
        #     )
        return projected_embedding


class Gemma3VisionModelWrapper(Llama4VisionModelWrapper):
    """
    Neuron ModelWrapper class for Gemma3's vision model (NeuronSiglipVisionModel).
    Inherits from Llama4VisionModelWrapper.
    Generates input shapes for trace and compilation. Disables bucketing.
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
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override Llama4VisionModelWrapper.input_generator().

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
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
                dtype=self.config.neuron_config.torch_dtype
            )
            inputs.append((pixel_values,))

        return inputs

    def forward(self, *args):
        """
        Override ModelWrapper.forward() to adapt for vision encoder.
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        # convert int64 to int32 to improve compatibility with compiler; does not apply to cpu case
        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(*args)

        pixel_values = args[0]
        input_batch_size = pixel_values.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            output = self._forward(*args)
            return output

        cur_batch = 0
        outputs = []

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}"
        )

        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(
                    f"running foward on batch {cur_batch}:{cur_batch + self.neuron_config.batch_size}"
                )

                # pad to next bucket for context encoding with bs > 1
                # batch_arg represent single prompt in batch of prompts
                batch_args = [
                    arg[cur_batch : cur_batch + self.neuron_config.batch_size] for arg in args
                ]
                batch_args = self.vllm_cte_repadding(batch_args)

                output = self._forward(*batch_args)

            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.neuron_config.batch_size}"
                )
                output = self._forward_with_pad(
                    *[
                        arg[cur_batch:input_batch_size] if not is_ranked_io(arg) else arg
                        for arg in args
                    ]
                )

            outputs.append(output)
            cur_batch += self.neuron_config.batch_size

        return output

    def _forward_with_pad(self, *args):
        """
        Override ModelWrapper._forward_with_pad
        as vision encoder's args only includes pixel values (i.e. len(args) = 1)
        """
        # Note: NxD's tracing flow (Model Builder) does not yet support kwargs, because of which we cannot support
        # optional parameters. Kwargs support is being added as a part of the new Model Builder API. Until then we
        # maintain a specific set of inputs that the ModelWrapper can support.
        # This is not the best way to maintain code. But soon kwargs suport will render this irrelevant.

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor, pad_type="fill_0", batch_sort_indices=None):
            """
            As part of continuous batching:
            * If users provide us input batch size less than compiled batch size, NxDI
              need to pad the inputs to the compiled batch size.
            * seq_ids are used to indicate which kv cache line is used for each input batch line.
              NxDI expects the seq_ids to always be [0, 1, 2, ..., compiled_batch_size) by default.
            * To fulfill these requirements, NxDI pads the seq_ids with the missing slots and sorts
              it in ascending order. Every other input args are reordered accordingly and
              missing slots are padded with `repeat_first_batchline`. While returning back response,
              we use index selct to pick the outputs corresponding to user provided seq_ids.
            Eg:
            Input [[10],[20]] and seq_ids [[3], [2]] with compiled batch size as 4.
            seq_ids [[3], [2]] -> [[3], [2], [0], [1]] (filled missing slots)    -> [[0], [1], [2], [3]] (sort)
            Input  [[10],[20]] -> [[10],[20],[10],[10]] (repeat_first_batchline) -> [[10],[10],[20],[10]](reorder)

            As part of continuous batching with prefix caching, the second restriction no longer holds true,
            so sorting of seq_ids and reordering of input args is no longer needed. Padding is required which is added
            towards the end using `repeat_first_batchline` with the exception of slot_mapping (set to -1 instead)
            as this is used to update the block kv cache. While returning back response, we just drop off the
            padded outputs lines at the end of the batch.
            Eg:
            Input [[10],[20]] ; seq_ids [[3], [2]] and slot mapping [[50],[100]] with compiled batch size as 4.
            seq_ids [[3], [2]]        -> [[3], [2], [0], [1]]  (filled missing slots)
            Input [[10],[20]]         -> [[10],[20],[10],[10]] (repeat_first_batchline)
            slot mapping [[50],[100]] -> [[50],[100],[-1], [-1]] (padded with -1)
            """
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size

            def repeat_first_batchline(tensor, padded_shape):
                return tensor[0].repeat(padded_shape[0], 1, 1, 1).to(tensor.dtype)

            def fill_value_tensor(value):
                return lambda tensor, padded_shape: torch.full(padded_shape, fill_value=value, dtype=tensor.dtype)

            PAD_TYPES = {
                "repeat_first_batchline": repeat_first_batchline,
                "fill_0": fill_value_tensor(0),
                "fill_1": fill_value_tensor(1),
                "fill_-1": fill_value_tensor(-1),
            }

            if pad_type not in PAD_TYPES:
                raise ValueError(f"Unknown pad_type '{pad_type}'. Available: {list(PAD_TYPES.keys())}")

            padded_tensor = PAD_TYPES[pad_type](tensor, padded_shape)
            padded_tensor[: tensor.shape[0]] = tensor

            if batch_sort_indices is not None:
                padded_tensor = torch.index_select(padded_tensor, 0, batch_sort_indices)

            return padded_tensor

        reorder_seq_ids = False
        pixel_values = args[0]
        orig_batch_size = pixel_values.shape[0]
        seq_ids_list = list(range(orig_batch_size))
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32)

        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_seq_ids, indices = torch.sort(padded_seq_ids) if reorder_seq_ids else (padded_seq_ids, None)

        padded_args = []
        # pad pixel_values
        for arg in args:
            if is_ranked_io(arg):  # async output
                # ===========READ THIS=============
                # args[0] can be either input_ids
                # or an async_output. If the output
                # is async, it means that the sorting
                # and padding has already been done
                # properly, so we simply append the
                # result. This is true because the
                # results from async are fed directly
                # to the next iteration without data
                # modification, and the model was
                # executed with padded & sorted inputs.
                # =================================
                padded_args.append(arg)
            else:
                padded_arg = pad_helper(
                    arg,
                    pad_type="repeat_first_batchline",
                    batch_sort_indices=indices,
                )
                padded_args.append(padded_arg)

        outputs = self._forward(*padded_args)

        return outputs[:orig_batch_size]
