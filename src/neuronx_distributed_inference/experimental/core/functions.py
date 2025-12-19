from typing import Dict, List

import torch
from neuronx_distributed.trace.model_builder import ModelBuilderV2

# TODO: migrate away from this file


def build(
    model: torch.nn.Module,
    world_size: int = -1,
    batch_size: int = 1,
    sequence_length: int = 1024,
    sequence_length_bucketing=True,
    sequence_length_buckets: Dict[str, List[int]] = None,
):
    """
    Traces and compiled the model to run in NXDI.

    TODO: Explain better.

    This build function is opinionated in its inputs. It expects the models being traced
    to have the required inputs. You can always write your own build function that
    does not have this expectation.

    Why do you want to make your model compatible with this build function?
    The common build function can enable you to plug into any features built
    on top of this common API.

    Do you have a conditional input ?
    Feel free to add that as a separate build function!

    Args:
        model (torch.nn.Module): The model to be built.
        world_size (int): The number of cores to use for inference.
        batch_size (int): The batch size.
        sequence_length (int | List[int]): The sequence length.
        sequence_length_bucketing (bool, optional): Whether to use sequence length bucketing. Defaults to True.

    Returns:
        torch.nn.Module: The built model.
    """

    # If sequence_length is an integer, convert it to a list
    if isinstance(sequence_length, List):
        raise NotImplementedError

    # TODO: Add default bucketing.

    # Initialize ModelBuilderV2
    builder = ModelBuilderV2(
        model=model,
    )

    # Add prefill trace
    builder.trace(
        kwargs={
            "tokens": torch.ones((batch_size, sequence_length), dtype=torch.int32),
            "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
            "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.int32),
        },
        tag="prefill",
    )

    # Add decode trace
    builder.trace(
        kwargs={
            "tokens": torch.ones((batch_size, 1), dtype=torch.int32),
            "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
            "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.int32),
        },
        tag="decode",
    )

    # Compile the model
    traced_model = builder.compile(
        compiler_args="--auto-cast=none",
    )

    return traced_model
