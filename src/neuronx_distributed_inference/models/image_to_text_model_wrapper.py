import torch

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    ModelWrapper,
)
from neuronx_distributed_inference.modules.async_execution import is_ranked_io
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params


IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids", "seq_ids", "sampling_params", "prev_hidden",
                                          "adapter_ids", "accepted_indices", "current_length", "medusa_mask", "scatter_index", "slot_mapping",
                                          "active_block_table", "num_queries", "computed_context_lens", "tile_q_indices", "tile_block_tables",
                                          "tile_masks", "inputs_embeds", "kv_cache", "active_mask", "rotary_position_id", "vision_embeddings", "vision_mask")


class ImageToTextModelWrapper(ModelWrapper):
    """
    A class that wraps the image understanding Multimodal model for vision encoding, context encoding and token generation tasks.
    This class overrides input_generator() to provide additional pixel_values, vision_mask in the sample inputs for tracing.
    It removes inputs related to medusa and Lora since ImageToText models do not support them.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        # This wrapper is for image understanding models which include a vision and text model. The model conditionally contains
        # vision model inputs. NXD does not support kwargs in trace yet. This
        # forces all inputs to be ordered. Conditional inputs can only be
        # supported if we have kwargs. Kwargs support is planned to be added
        # soon. Here we choose to disable all flows except the text and
        # vision text flow.

        # To add support add appropriate forward and input generators.
        # TODO: Once kwargs is added consolidate implementations
        super().__init__(
            config=config,
            model_cls=model_cls,
            tag=tag,
            compiler_args=compiler_args,
            priority_model_idx=priority_model_idx,
            model_init_kwargs=model_init_kwargs,
        )

    # pad the inputs up to the compiled batch size in the end
    def pad_helper(self, tensor, pad_type="fill_0", batch_sort_indices=None):
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
            return tensor[0].unsqueeze(0).repeat(padded_shape[0], 1).to(tensor.dtype)

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

    def _forward_with_pad(self, *args):
        """
        Note: NxD's tracing flow (Model Builder) does not yet support kwargs, because of which we cannot support
        optional parameters. Kwargs support is being added as a part of the new Model Builder API. Until then we
        maintain a specific set of inputs that the ModelWrapper can support.
        This is not the best way to maintain code. But soon kwargs suport will render this irrelevant.

        Supported parameter order for ImageToText is,
        ┌───────┬───────────────────────┐
        │ index │     parameter         │
        ├───────┼───────────────────────┤
        │     0 │ input_id              │
        │     1 │ attn_mask             │
        │     2 │ position_ids          │
        │     3 │ seq_ids               │
        │     4 │ sampling_params       │
        │     5 │ prev_hidden           │
        │     6 │ adapter_ids           │
        │     7 │ accepted_indices      │
        │     8 │ current_length        │
        │     9 │ medusa_mask           │
        │    10 │ scatter_index         │
        │    11 │ slot_mapping          │
        │    12 │ active_block_table    │
        │    13 │ num_queries           │
        │    14 │ computed_context_lens │
        │    15 │ tile_q_indices        │
        │    16 │ tile_block_tables     │
        │    17 │ tile_masks            │
        │    18 │ inputs_embeds         │
        │    19 │ kv_cache              │
        │    20 │ active_mask           │
        │    21 │ rotary_position_id    │
        │    22 │ vision_embeddings     │
        │    23 │ vision_mask           │
        └───────┴───────────────────────┘
        """

        seq_ids = args[3]
        sampling_params = args[4]

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead
        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list
            + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_seq_ids, indices = torch.sort(padded_seq_ids)
        padded_args = []
        # pad input_ids, attn_mask and position_ids
        for i, arg in enumerate(args[0:3]):
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
                padded_arg = self.pad_helper(
                    arg,
                    pad_type="repeat_first_batchline",
                    batch_sort_indices=indices if not self.is_prefix_caching else None,
                )
                padded_args.append(padded_arg)

        # for block kv layout the seq_ids may lies outside of range(self.neuron_config.max_batch_size)
        # therefore, we need to remove potential extra paddings to seq_ids
        padded_seq_ids = padded_seq_ids[: self.neuron_config.max_batch_size]
        padded_args.append(padded_seq_ids)

        # pad sampling params by repeating first batchline
        padded_sampling_params = self.pad_helper(
            sampling_params,
            pad_type="repeat_first_batchline",
            batch_sort_indices=indices if not self.is_prefix_caching else None,
        )
        padded_args.append(padded_sampling_params)
        # add args prev_hidden --> vision_mask without padding
        for i, arg in enumerate(args[5:24]):
            padded_args.append(arg)

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            if self.async_mode:
                return logits
            return torch.index_select(logits, 0, seq_ids)
        else:
            logits, *kv_cache = outputs
            return [torch.index_select(logits, 0, seq_ids), *kv_cache]

    @staticmethod
    def get_dummy_vision_inputs(config, input_ids, n_active_tokens, fill_value):
        input_batch_size, input_sequence_len = input_ids.shape[0], input_ids.shape[-1]
        if input_sequence_len > 1:
            vision_embeddings = torch.zeros(
                input_batch_size,
                n_active_tokens,
                config.hidden_size,
                dtype=config.neuron_config.torch_dtype,
            )
            # we trace not actual vision mask, but positions, for best performance.
            vision_mask = torch.full(
                size=(
                    input_batch_size,
                    n_active_tokens,
                    1),
                fill_value=fill_value,
                dtype=torch.int32
            )
        else:
            vision_embeddings = torch.zeros((0), dtype=config.neuron_config.torch_dtype)
            vision_mask = torch.zeros((0), dtype=torch.bool)
        return vision_embeddings, vision_mask

    def input_generator(
        self,
    ):
        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )

            input_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros(
                (self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32
            )
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32
            )
            # During model tracing, we fill vision embeddings and vision_mask with zeros
            vision_embeddings, vision_mask = self.get_dummy_vision_inputs(
                config=self.config,
                input_ids=input_ids,
                n_active_tokens=n_active_tokens,
                fill_value=0
            )

            if self.tag == CONTEXT_ENCODING_MODEL_TAG or self.tag == TOKEN_GENERATION_MODEL_TAG:
                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        torch.empty(0),  # prev_hidden
                        torch.empty(0),  # adapter_ids
                        torch.empty(0),  # accepted_indices
                        torch.empty(0),  # current_length
                        torch.empty(0),  # medusa_mask
                        torch.empty(0),  # scatter_index
                        torch.empty(0),  # slot_mapping=None,
                        torch.empty(0),  # active_block_table=None,
                        torch.empty(0),  # num_queries=None,
                        torch.empty(0),  # computed_context_lens=None,
                        torch.empty(0),  # tile_q_indices=None,
                        torch.empty(0),  # tile_block_tables=None,
                        torch.empty(0),  # tile_masks=None,
                        torch.empty(0),  # inputs_embeds: Optional[torch.FloatTensor] = None,
                        torch.empty(0),  # kv_cache: Optional[torch.Tensor] = None,
                        torch.empty(0),  # active_mask=None,
                        torch.empty(0),  # rotary_position_id=None,
                        vision_embeddings,
                        vision_mask,
                    )
                )
            else:
                raise ValueError(f"Unsupported model tag '{self.tag}' for ImageToText models")

        return inputs
