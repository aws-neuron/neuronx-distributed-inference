import logging
import os
import time

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from neuronx_distributed_inference.models.llama4.utils.encoder_utils import scatter_by_index_put
from neuronx_distributed_inference.utils.testing import build_function

from .test_utils import get_compiler_args, get_rtol, setup_debug_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
setup_debug_env()

# Global Constants
OUTPUT_DIM = 5120
TP_DEGREE = 1
TARGET_CHUNKS = 88
SEQ_LEN = 3695
CHUNK_SIZE = 144
MAX_POSITIONS = 8192


def original_meta_scatter_embeddings(image_batch, image_mask, h_image, encoded_patches_proj):
    """
    Meta's original scatter embedding solution, does not work on neuron.
    Include here as logit matching test, because new put together should
    match the output from meta's scatter embedding.
    """
    assert not torch.isnan(encoded_patches_proj).any()

    num_images_per_sequence = [encoded_patches_proj.size(0)]  # Assuming single batch for simplicity

    encoded_patches_list = encoded_patches_proj.split(num_images_per_sequence, dim=0)
    for index in range(h_image.size(0)):
        encoded_patches_per_sample = encoded_patches_list[index]
        sample_image_mask = image_mask[index]

        if encoded_patches_per_sample.numel() == 0:
            continue
        encoded_patches_per_sample = encoded_patches_per_sample.contiguous().view(
            -1, encoded_patches_per_sample.size(-1)
        )

        n_tokens_to_fill = sample_image_mask.sum()
        assert n_tokens_to_fill <= encoded_patches_per_sample.size(0)

        h_image[index].masked_scatter_(
            sample_image_mask.expand(-1, h_image.size(-1)),
            encoded_patches_per_sample[:n_tokens_to_fill],
        )

    return h_image


def get_false_indices(valid_mask):
    """
    Extract the indices where the mask is False.

    Args:
    valid_mask (torch.Tensor): A 2D boolean tensor of shape (batch_size, seq_len)

    Returns:
    torch.Tensor: A 1D tensor containing the indices where the mask is False
    """
    image_mask_1d = valid_mask.reshape(-1)
    image_mask_1d = image_mask_1d.bool()
    false_indices = torch.nonzero(~image_mask_1d).squeeze()
    return false_indices


def generate_fake_image_mask(seq_len, num_vision_chunks):
    """
    Generate a fake image mask with a specific pattern of True and False values.

    Args:
    seq_len (int): The total length of the sequence
    num_vision_chunks (int): The number of vision chunks to generate

    Returns:
    torch.Tensor: A 1D boolean tensor representing the fake image mask
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)

    # First two positions are always False
    current_pos = 2

    # Generate N-1 chunks of [144 True, 1 False]
    for _ in range(num_vision_chunks - 1):
        if current_pos + 145 > seq_len:
            break
        mask[current_pos : current_pos + CHUNK_SIZE] = True
        current_pos += CHUNK_SIZE + 1

    # Add the last chunk: [False, 144 True, False]
    if current_pos + CHUNK_SIZE + 2 <= seq_len:
        mask[current_pos + 1 : current_pos + CHUNK_SIZE + 1] = True

    return mask


def generate_positions_from_mask(mask):
    """
    Generate position indices from a boolean mask.

    Args:
    mask (torch.Tensor): A 1D boolean tensor

    Returns:
    torch.Tensor: A 1D tensor containing the indices where the mask is True
    """
    return torch.nonzero(mask).squeeze()


def generate_fake_data(num_vision_chunks, dtype):
    """
    Generate fake data for testing the scatter_by_index_put function.

    Args:
    num_vision_chunks (int): The number of vision chunks to generate

    Returns:
    dict: A dictionary containing 'cpu' and 'neuron' keys, each with a tuple of tensors
    """
    batch_size = 1

    # Generate fake image mask
    valid_mask_cpu = generate_fake_image_mask(SEQ_LEN, num_vision_chunks).unsqueeze(0).unsqueeze(-1)

    # Generate positions from the mask
    positions_cpu = generate_positions_from_mask(valid_mask_cpu.squeeze())

    # CPU inputs
    h_image_cpu = torch.zeros(batch_size, SEQ_LEN, OUTPUT_DIM, dtype=dtype)
    encoded_patches_proj_cpu = torch.randn(num_vision_chunks, CHUNK_SIZE, OUTPUT_DIM, dtype=dtype)

    # Neuron inputs (padded)
    h_image_padded = torch.zeros((batch_size, MAX_POSITIONS, OUTPUT_DIM), dtype=dtype)
    h_image_padded[:, :SEQ_LEN, :] = h_image_cpu

    # Calculate the number of positions to pad
    N = len(positions_cpu)
    padding_size = TARGET_CHUNKS * CHUNK_SIZE - N

    # Pad positions with MAX_POSITIONS - 1
    padding = torch.full((padding_size,), MAX_POSITIONS - 1, dtype=torch.long)
    positions_padded = torch.cat([positions_cpu, padding])
    positions_padded = positions_padded.unsqueeze(0).unsqueeze(-1)  # Add batch dimension

    # Pad encoded_patches_proj to TARGET_CHUNKS with zeros
    encoded_patches_proj_padded = torch.zeros((TARGET_CHUNKS, CHUNK_SIZE, OUTPUT_DIM), dtype=dtype)
    encoded_patches_proj_padded[:num_vision_chunks] = encoded_patches_proj_cpu

    return {
        "cpu": (h_image_cpu, valid_mask_cpu, encoded_patches_proj_cpu, positions_cpu),
        "neuron": (h_image_padded, encoded_patches_proj_padded, positions_padded),
    }


def depad_output(padded_output, original_seq_len):
    """
    Remove padding from the output tensor.

    Args:
    padded_output (torch.Tensor): The padded output tensor
    original_seq_len (int): The original sequence length before padding

    Returns:
    torch.Tensor: The depadded output tensor
    """
    return padded_output[:, :original_seq_len, :]


def run_test_for_chunks(neuron_func, num_chunks, dtype):
    """
    Run a test for the scatter_by_index_put function with a specific number of chunks.

    Args:
    neuron_func (callable): The compiled neuron function to test
    num_chunks (int): The number of chunks to use in the test

    Returns:
    float: The execution time of the neuron function
    """
    fake_data = generate_fake_data(num_chunks, dtype=dtype)

    # Unpack CPU data
    h_image_cpu, valid_mask_cpu, encoded_patches_proj_cpu, _ = fake_data["cpu"]

    # Run original scatter_embeddings on CPU
    expected_output = original_meta_scatter_embeddings(
        [[h_image_cpu]], valid_mask_cpu, h_image_cpu, encoded_patches_proj_cpu
    )

    # Unpack Neuron data
    h_image_padded, encoded_patches_proj_padded, positions_padded = fake_data["neuron"]

    # Prepare inputs for neuron function
    inputs = (h_image_padded, encoded_patches_proj_padded, positions_padded)

    # Run the neuron function
    start_time = time.time()
    padded_outputs = neuron_func(*inputs)
    end_time = time.time()

    outputs = depad_output(padded_outputs, h_image_cpu.size(1))

    # Compare results
    torch.testing.assert_close(outputs, expected_output, rtol=get_rtol(dtype), atol=1e-5)

    return end_time - start_time


def profile_function(func, *args, name):
    """
    Profile a function using torch.profiler.

    Args:
    func (callable): The function to profile
    *args: Arguments to pass to the function
    name (str): A name for the profiling session

    Returns:
    None
    """
    with profile(
        activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, with_stack=True
    ) as prof:
        for _ in range(10):
            func(*args)

    prof.export_chrome_trace(f"torch_profile_{name}.json")
    print(f"Profiling results for {name}:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            id=f"dtype_{str(dtype).split('.')[-1]}",
        )
        for dtype in [torch.float32, torch.float16]
    ],
)
def test_neuron_put_index(dtype):
    """
    Test the neuron_put_index function with various numbers of chunks and profile its performance.

    Args:
    None

    Returns:
    None
    """
    # Prepare example inputs for compilation
    fake_data = generate_fake_data(TARGET_CHUNKS, dtype=dtype)
    example_inputs = [tuple(torch.ones_like(t) for t in fake_data["neuron"])]

    # Build neuron function
    neuron_func = build_function(
        scatter_by_index_put,
        example_inputs,
        tp_degree=TP_DEGREE,
        compiler_args=get_compiler_args(),
    )

    # Test with different chunk numbers
    chunk_numbers = range(1, TARGET_CHUNKS + 1)
    for chunks in chunk_numbers:
        print(f"Testing with {chunks} chunks:")
        execution_time = run_test_for_chunks(neuron_func, chunks, dtype=dtype)
        print(f"Execution time: {execution_time:.4f} seconds")
        print("Test passed successfully!")
        print()

    # Performance test for compiled neuron function
    print("Profiling Neuron-compiled function:")
    profile_function(neuron_func, *fake_data["neuron"], name="neuron_compiled_new_25chunk")
