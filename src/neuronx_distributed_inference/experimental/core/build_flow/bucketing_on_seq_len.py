import logging
from typing import List

import torch
from neuronx_distributed import ModelBuilder
from neuronx_distributed_inference.modules.autobucketing import generate_buckets

logger = logging.getLogger("Neuron")


def build_for_bucketing_on_seq_len(
    model: torch.nn.Module,
    batch_size: int,
    max_seq_len: int,
    prefill_buckets: List[int] = None,
    decode_buckets: List[int] = None,
):
    """
    Traces and compiles the model to run in NXDI, with bucketing and weight
    layout optimization (WLO).

    .. todo::
    1. Better documentation
    2. Use config parsed from YAML/JSON

    This build function is opinionated in its inputs. It expects the models
    being traced to have the required inputs. You can always write your own
    build function that does not have this expectation.

    Why do you want to make your model compatible with this build function?
    The common build function can enable you to plug into any features built
    on top of this common API.

    Do you have a conditional input?
    Feel free to add that as a separate build function!

    :param model: The model to be built.
    :type model: torch.nn.Module
    :param batch_size: The batch size.
    :type batch_size: int
    :param sequence_length: The sequence length.
    :type sequence_length: int | List[int]

    :returns: The built model.
    :rtype: torch.nn.Module
    """

    # If sequence_length is an integer, convert it to a list
    assert isinstance(max_seq_len, int), "Only integer sequence length is supported"

    # Initialize ModelBuilderV2, this model already has a limit of max kv cache length
    builder = ModelBuilder(model=model)

    # Add prefill trace
    if not prefill_buckets or len(prefill_buckets) == 0:
        prefill_buckets = generate_buckets(min_length=128, max_length=max_seq_len)
    logger.info(f"There are {len(prefill_buckets)} prefill buckets: {prefill_buckets}")

    for seq_len_bucket in prefill_buckets:
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, seq_len_bucket), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len_bucket), dtype=torch.int32),
            },
            tag=f"prefill_{seq_len_bucket}",
        )

    # Add decode trace
    if not decode_buckets or len(decode_buckets) == 0:
        decode_buckets = generate_buckets(min_length=128, max_length=max_seq_len)
    logger.info(f"There are {len(decode_buckets)} decode buckets: {decode_buckets}")

    for seq_len_bucket in decode_buckets:
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, 1), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len_bucket), dtype=torch.int32),
            },
            tag=f"decode_{seq_len_bucket}",
        )

    # Compile the model
    traced_model = builder.compile(
        priority_model_key=f"decode_{decode_buckets[0]}",
    )

    return traced_model
