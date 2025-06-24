import neuronxcc.nki.language as nl

# Partition size for a tile computation, and its upper bound is the number
# of PE columns in TensorEngine (128)
B_P_SIZE = nl.tile_size.pmax

# Free dimension size for a tile computation, and its upper bound is the
# max num of fp32 elements a PSUM partition can hold (512)
B_F_SIZE = nl.tile_size.psum_fmax

# Sometimes B_D_SIZE is also used for a tile computation, and it means the
# dim size. This corresponds to the head_dim for an attention module.
# B_D_SIZE is typcially used as the non reduction dimension, so its
# upper-bound from hardware is the 512, like B_F_SIZE.

# Check more details on neuron hardware spec from
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html#tensor-engine


def ceil_div(a, b):
    return (a + b - 1) // b


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0
