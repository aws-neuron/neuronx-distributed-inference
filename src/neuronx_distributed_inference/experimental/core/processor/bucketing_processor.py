import bisect
import torch

from neuronx_distributed_inference.experimental.core.pad import pad_to_shape


##################################################
# Modules for bucketing support
#
# Not putting them inside a class, so that they
# can be easily integrated into other processors
# if needed.
##################################################


def collect_buckets(reserved_example_inputs: dict):
    """
    Collect all buckets from reserved_example_inputs.

    This function assumes the 3rd element in the input tuple is attention_mask,
    and uses its shape (2nd dim) to determine the sequence length for bucketing.

    It also assumes model tags start with either "prefill" or "decode" to
    categorize the inputs.

    .. note::
       This func requires the 3rd arg in the reserved_example_inputs is an
       attention mask of shape (batch_size, kv_len)

    .. todo::
       Use model type tag to determine the model type instead of hardcoding.

    :param reserved_example_inputs: A dictionary where keys are model tags
        (e.g., "prefill_512", "decode_1") and values are tuples of example
        inputs for the model.
    :type reserved_example_inputs: dict

    :returns:
        - **buckets** (*dict*) -- A dictionary with two keys: "prefill" and
          "decode". Each key maps to another dictionary where keys are sequence
          lengths and values are the corresponding example inputs.
        - **bucket_table** (*dict*) -- A dictionary with two keys: "prefill"
          and "decode". Each key maps to a sorted list of available bucket
          lengths.
    :rtype: tuple[dict, dict]
    """
    example_inputs = reserved_example_inputs

    buckets = {"prefill": {}, "decode": {}}
    for model_tag, inputs in example_inputs.items():
        if model_tag.startswith("prefill"):
            model_type = "prefill"
        elif model_tag.startswith("decode"):
            model_type = "decode"
        else:
            raise ValueError(f"Found a model tag {model_tag} that doesn't "
                             "start with 'prefill' or 'decode'. Please "
                             "update your model tag name, or update the "
                             "bucket selection logics used by the input "
                             "output procesor.")

        # Assuming inputs[2] is attention_mask
        kv_len = inputs[2].shape[1]
        buckets[model_type][kv_len] = inputs

    bucket_table = {
        "prefill": sorted(buckets["prefill"].keys()),
        "decode": sorted(buckets["decode"].keys()),
    }

    return buckets, bucket_table


def get_buckets_by_model_type(
    tokens: torch.Tensor,
    buckets: dict,
    bucket_table: dict
):
    """
    Get the bucket choices based on the input tokens shape.

    :param tokens: The input tokens tensor of shape (batch_size, seq_len).
    :type tokens: torch.Tensor
    :param buckets: A dictionary with two keys: "prefill" and "decode". Each key
        maps to a list of available bucket lengths.
    :type buckets: dict

    :returns: A list of available bucket lengths for the current model type.
    :rtype: list
    """
    model_type = "prefill" if tokens.shape[1] > 1 else "decode"
    return buckets[model_type], bucket_table[model_type]


def select_smallest_bucket(bucket_choices, bucket_table, cur_len):
    """
    Select the smallest bucket that can fit cur_len.
    """
    if cur_len > bucket_table[-1]:
        raise ValueError(f"No valid bucket found for length {cur_len}."
                         f" Available buckets: {bucket_table}")

    # Need to use bisect_left to find the bucket that is >= cur_len
    bucket_idx = bisect.bisect_left(bucket_table, cur_len)
    selected_bucket = bucket_table[bucket_idx]
    return bucket_choices[selected_bucket]


##################################################
# BucketingProcessor class
##################################################


class BucketingProcessor(torch.nn.Module):
    """
    This processor applies bucketing techniques to optimize memory usage
    and computation by routing requests to the smallest suitable bucket
    and padding them to predetermined bucket sizes.

    :param model: The underlying neural network model to wrap with bucketing functionality
    :type model: torch.nn.Module
    :param pad_token_id: The token ID to use for padding input_tokens to bucket sizes
    :type pad_token_id: int

    :ivar model: The wrapped neural network model
    :vartype model: torch.nn.Module
    :ivar pad_token_id: Token ID used for padding
    :vartype pad_token_id: int
    :ivar buckets: Collection of bucket configurations derived from model's reserved inputs
    :vartype buckets: Any
    :ivar bucket_table: Lookup table for bucket selection
    :vartype bucket_table: Any

    .. note::
       The bucketing strategy can be customized by replacing the ``collect_buckets``
       function with alternative implementations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pad_token_id: int,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        # Can swap collect_buckets_llm_1d with other modules if needed
        self.buckets, self.bucket_table = collect_buckets(
            self.model.reserved_example_inputs
        )

    def forward(self, input_tokens, last_pos, attention_mask, **kwargs):
        """
        Forward pass through the bucketing processor.

        Applies preprocessing to pad inputs to appropriate bucket sizes before
        passing them through the underlying model.

        :param tokens: Input token tensor
        :type tokens: torch.Tensor
        :param last_pos: Position indices for the last token in each sequence
        :type last_pos: torch.Tensor
        :param attention_mask: Attention mask tensor indicating valid positions
        :type attention_mask: torch.Tensor

        :returns: generated token ids
        :rtype: torch.Tensor
        """
        input_tokens, last_pos, attention_mask = self.pre_process(
            input_tokens, last_pos, attention_mask
        )
        output = self.model(input_tokens, last_pos, attention_mask)
        return output

    def pre_process(self, input_tokens, last_pos, attention_mask):
        """
        Preprocess inputs by padding them to the smallest suitable bucket size.

        Determines the appropriate bucket based on the maximum sequence length
        and pads all inputs to match the bucket dimensions for efficient
        batching.

        :param tokens: Input token tensor to be padded
        :type tokens: torch.Tensor
        :param last_pos: Position indices (assumed to have correct shape)
        :type last_pos: torch.Tensor
        :param attention_mask: Attention mask to be padded
        :type attention_mask: torch.Tensor

        :returns: Tuple of preprocessed tensors (tokens, last_pos, attention_mask)
                  padded to bucket dimensions
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        .. note::
           The ``last_pos`` tensor is expected to already have the correct shape
           and is not modified during preprocessing.
        """
        next_len = last_pos.max().item() + 1
        buckets, bucket_table = get_buckets_by_model_type(
            input_tokens, self.buckets, self.bucket_table
        )
        expected_inputs = select_smallest_bucket(
            buckets, bucket_table, next_len
        )

        # Assuming last_pos already has the correct shape
        input_tokens = pad_to_shape(
            input_tokens, expected_inputs[0].shape, value=self.pad_token_id
        )
        attention_mask = pad_to_shape(
            attention_mask, expected_inputs[2].shape, value=0
        )

        return (input_tokens, last_pos, attention_mask)
