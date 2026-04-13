# Code copied from aws-neuron/transformers-neuronx

import itertools
import math

import torch
import torch.nn.functional as F


def round_up_to_divisor(value, divisor):
    return math.ceil(value / divisor) * divisor


def pad_sizes(shape, dims, sizes, left=False):
    if isinstance(dims, int):
        dims = (dims,)
    if isinstance(sizes, int):
        sizes = (sizes,) * len(dims)
    lhs = [0] * len(shape)
    rhs = [0] * len(shape)
    side = lhs if left else rhs
    for dim, size in zip(dims, sizes):
        # Don't truncate tensor if the current size exceeds the "padded size"
        side[dim] = max(0, size - shape[dim])
    # This creates a tuple alternating between left and right padding for each dimension.
    # Example: If lhs=0,2,0 and rhs=1,0,3, result would be (0,1,2,0,0,3)
    sizes = tuple(itertools.chain(*zip(reversed(lhs), reversed(rhs))))
    if sum(sizes) == 0:
        return None
    return sizes


def pad(tensor, dims, sizes, left=False):
    if tensor is None:
        return tensor
    padding = pad_sizes(tensor.shape, dims, sizes, left=left)
    if padding is not None:
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.detach()
        return F.pad(tensor, padding)
    return tensor


def pad_interleaved(tensor, dim, size, source_len_per_group, pad_len_per_group):
    """
    Pad the selected `dim` of `tensor`, up to `size`. Put zeros interleavedly base on `source_len_per_group`
    and `pad_len_per_group`

        [
            # group i # ? ... x source_len_per_group ... ?, 0, ... x pad_len_per_group ... , 0]
        ]

    For example:
        pad_interleaved([1,2,3]], dim=0, size=9, source_len_per_group=1, pad_len_per_group=2)
        each group becomes [?, 0, 0], with number of group as 3, we have
        result: [1, 0, 0, 2, 0, 0, 3, 0, 0]
    """
    assert isinstance(dim, int), "pad_interleaved now only supports single dim"

    assert isinstance(size, int), "pad_interleaved now only supports single dim of size"

    padded_shape = list(tensor.shape)

    padded_shape[dim] = size

    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)

    num_groups = size // (source_len_per_group + pad_len_per_group)

    src_indices = [slice(0, None) for _ in range(len(padded_shape))]
    target_indices = [slice(0, None, None) for _ in range(len(padded_shape))]
    s_src = 0
    s_target = 0
    for _ in range(num_groups):
        target_indices[dim] = slice(s_target, s_target + source_len_per_group)
        src_indices[dim] = slice(s_src, s_src + source_len_per_group)
        padded_tensor[target_indices] = tensor[src_indices]
        s_src += source_len_per_group
        s_target += source_len_per_group + pad_len_per_group

    return padded_tensor


class MaybePadder:
    def __init__(self, size, padding="end", split_size=None, interleaved_factor=None) -> None:
        self.split_size = split_size
        self.size = size
        self.padding = padding
        self.interleaved_factor = interleaved_factor

    def __call__(self, weight, dim):
        if self.padding == "end":
            return pad(weight, dim, self.size, left=False)
        else:
            if weight is None:
                return weight
            assert self.padding == "interleaved", f"Invalid padding mode {self.padding}"
            assert self.interleaved_factor, "interleaved_factor is not provided"
            # when split_size is set, we first split the target weight at dim
            # into (split_size x ?), for example, to do interleaved padding on of KV weight
            # we first need to reshape it into (hidden, num_kv_head, d_head)
            # and then apply interleaved padding on num_kv_head
            weight_shapes = list(weight.shape)

            padded_shape = weight_shapes.copy()
            padded_shape[dim] = self.size

            new_size = self.size
            if self.split_size:
                assert (
                    weight_shapes[dim] % self.split_size == 0
                ), f"shape on dim_{dim} {weight_shapes[dim]} cannot be evenly divisible by provided split_size {self.split_size}"
                new_shape = (
                    weight_shapes[:dim]
                    + [self.split_size]
                    + [weight_shapes[dim] // self.split_size]
                    + weight_shapes[dim + 1 :]
                )
                weight = weight.view(new_shape)
                new_size = self.size // (weight_shapes[dim] // self.split_size)
            res = pad_interleaved(
                weight,
                dim,
                new_size,
                weight.shape[dim] // self.interleaved_factor,
                (new_size - weight.shape[dim]) // self.interleaved_factor,
            )
            return res.view(padded_shape)
