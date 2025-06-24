from torch_neuronx.xla_impl.ops import xla_hlo_call


def slice_along(tensor, dim, limit, start=0, stride=1):
    """
    Slice along a dimension.
    """
    dimensions = [
        dict(start=0, limit=size, stride=1) for size in tensor.sizes
    ]
    dimensions[dim] = dict(start=start, limit=limit, stride=stride)

    sizes = list(tensor.sizes)
    sizes[dim] = (limit - start + stride - 1) // stride

    return tensor.dtype[sizes].Slice(
        tensor,
        slice_dimensions=dimensions
    )


def concatenate(operands, dimension):
    # Concatenates a sequence of arrays along dimension.
    dtype = operands[0].dtype
    sizes = list(operands[0].sizes)
    for op_idx in range(1, len(operands)):
        for dim_idx in range(len(sizes)):
            if dim_idx != dimension:
                assert sizes[dim_idx] == operands[op_idx].sizes[dim_idx], \
                    "All tensors must have the same shape (except in the concatenating dimension)."
        sizes[dimension] = sizes[dimension] + operands[op_idx].sizes[dimension]
    output = dtype[sizes].Concatenate(*operands, dimensions=[dimension])
    return output


def full(value, dtype, sizes):
    result = dtype.Constant(constant_value=value)
    result = dtype[sizes].Broadcast(result, dimensions=[])
    return result


def all_gather(tensor, dim, tp_degree, replica_groups=None):
    shape = list(tensor.sizes)
    dtype = tensor.dtype

    if replica_groups is None:
        replica_groups = [list(range(tp_degree))]
        shape[dim] *= tp_degree
    else:
        shape[dim] *= len(replica_groups[0])

    return dtype[shape].AllGather(
        tensor,
        dimensions=[dim],
        replica_groups=replica_groups,
    )


def permute(tensor, dimensions):
    size = list(tensor.sizes)
    permuted_size = [size[dim] for dim in dimensions]
    return tensor.dtype[permuted_size].Transpose(tensor, dimensions=dimensions)


@xla_hlo_call
def looped_einsum(lhs, rhs, tp_degree):
    """
    Performs a tiled matrix multiplication using AwsNeuronCollectiveMatmul Kernel with tensor parallelism.

    This function implements a memory-efficient matrix multiplication by processing the input
    in tiles, using AWS Neuron's custom call for collective matrix multiplication. It reshapes
    and permutes the inputs appropriately to handle larger matrices that might not fit in memory
    for the collective matmul call. The collective matmul does dot(all-gather(w), x).

    Args:
        lhs (2D Tensor) [m , k_i]: Left-hand side input tensor that is sharded across TP degree
        rhs (3D Tensor) [b, s, k]: Right-hand side input tensor k = tp_degree*k_i
        tp_degree (int): Tensor parallelism degree, determines the distribution of computation
                        across devices

        Assumes the last dimension as the contraction dim for both rhs and lhs

    Returns:
        Tensor: Result of the matrix multiplication reshaped to match the expected output
               dimensions [b, s, m]

    Notes:
        - Uses a tile size of 4096 for processing larger matrices
        - Employs AWS Neuron's custom collective matmul operation
        - The function performs the following steps:
          1. Reshapes the right-hand side input
          2. Processes the multiplication in tiles
          3. Accumulates results
          4. Reshapes the output to the desired dimensions
    """
    dtype = lhs.dtype
    rhs_shape = list(rhs.sizes)
    rhs_reshaped = rhs.dtype[(rhs_shape[0] * rhs_shape[1], rhs_shape[-1])].Reshape(rhs)
    rhs_new_shape = list(rhs_reshaped.sizes)
    lhs_shape = list(lhs.sizes)
    tile_size = 4096
    n_tiles = rhs_shape[-1] // tile_size

    results = full(0.0, dtype=dtype, sizes=(lhs_shape[0] * tp_degree, rhs_new_shape[0]))
    for i in range(n_tiles):
        start = i * tile_size
        end = (i + 1) * tile_size
        rhs_t = permute(rhs_reshaped, [1, 0])
        rhs_slice = slice_along(rhs_t, dim=0, start=start, limit=end)
        lhs_slice = slice_along(lhs, dim=1, start=start, limit=end)
        # lhs_gather = all_gather(lhs_slice, dim=0, tp_degree=tp_degree)
        result_shape = [lhs_shape[0] * tp_degree] + [rhs_new_shape[0]]
        rhs_contracting_dim = 0
        num_groups = 1
        config = f"rhs_contracting_dim={rhs_contracting_dim},tp_degree={tp_degree},num_groups={num_groups},use_sb_to_sb=0"
        backend_config = str(config).encode()
        output_tiled = dtype[result_shape].CustomCall(lhs_slice, rhs_slice,
                                                      custom_call_target="AwsNeuronCollectiveMatmul",
                                                      backend_config=backend_config)
        results = results.dtype[result_shape].Add(results, output_tiled)
    results = permute(results, [1, 0])
    out_shape = rhs_shape[:2] + [result_shape[0]]

    return dtype[out_shape].Reshape(results)


def dot10(lhs, rhs):
    """
    Performs matrix multiplication by contracting dimension 1 of lhs with dimension 0 of rhs.

    The name 'dot10' comes from the contraction pattern: dimension 1 of the left matrix
    with dimension 0 of the right matrix. This is standard matrix multiplication where:
    - For lhs matrix of shape (M, K)
    - For rhs matrix of shape (K, N)
    - Results in matrix of shape (M, N)

    Args:
        lhs (Tensor): Left-hand side matrix of shape (M, K)
        rhs (Tensor): Right-hand side matrix of shape (K, N)

    Returns:
        Tensor: Result of matrix multiplication with shape (M, N)

    Notes:
        - The '10' in the name represents: 1 (from lhs) contracts with 0 (from rhs)
        - This is equivalent to a standard matrix multiplication operation
        - Uses explicit dimension numbers for contraction through dot_dimension_numbers
    """
    dtype = lhs.dtype
    lhs_size, _ = lhs.sizes
    _, rhs_size = rhs.sizes
    dot_dims = dict(lhs_contracting_dimensions=[1], rhs_contracting_dimensions=[0])
    return dtype[lhs_size, rhs_size].Dot(lhs, rhs, dot_dimension_numbers=dot_dims)


@xla_hlo_call
def tiled_all_gather_matmul(lhs, rhs, tp_degree, tile_size=4096):
    """
    Performs tiled matrix multiplication with all-gather operation for distributed computing.

    This function implements a memory-efficient matrix multiplication by processing the input
    in tiles and utilizing all-gather operations for distributed processing. The computation
    is split across multiple devices using tensor parallelism.

    Args:
        lhs (2D Tensor) [m , k_i]: Left-hand side input tensor that is sharded across TP degree
        rhs (3D Tensor) [b, s, k]: Right-hand side input tensor k = tp_degree*k_i
        tp_degree (int): Tensor parallelism degree, specifies number of parallel processes
        tile_size (int, optional): Size of tiles for processing. Defaults to 4096
       Assumes the last dimension as the contraction dim for both rhs and lhs
    Returns:
        Tensor: Result of the matrix multiplication reshaped to  [b, s, m]

    Notes:
        - Implements tiled processing to handle large matrices efficiently
        - Uses all_gather operation to collect distributed data
        - Process:
          1. Reshapes input tensors
          2. Processes in tiles of specified size
          3. Performs all-gather operation on left-hand side
          4. Accumulates results using dot product
          5. Reshapes to final output dimensions
    """
    dtype = lhs.dtype
    rhs_shape = list(rhs.sizes)
    rhs_reshaped = rhs.dtype[(rhs_shape[0] * rhs_shape[1], rhs_shape[-1])].Reshape(rhs)
    rhs_new_shape = list(rhs_reshaped.sizes)
    lhs_shape = list(lhs.sizes)
    n_tiles = rhs_shape[-1] // tile_size

    results = full(0.0, dtype=dtype, sizes=(lhs_shape[0] * tp_degree, rhs_new_shape[0]))
    for i in range(n_tiles):
        start = i * tile_size
        end = (i + 1) * tile_size
        rhs_t = permute(rhs_reshaped, [1, 0])
        rhs_slice = slice_along(rhs_t, dim=0, start=start, limit=end)
        lhs_slice = slice_along(lhs, dim=1, start=start, limit=end)
        lhs_gather = all_gather(lhs_slice, dim=0, tp_degree=tp_degree)
        result_shape = [lhs_shape[0] * tp_degree] + [rhs_new_shape[0]]
        output_tiled = dot10(lhs_gather, rhs_slice)
        results = results.dtype[result_shape].Add(results, output_tiled)
    results = permute(results, [1, 0])
    out_shape = rhs_shape[:2] + [result_shape[0]]

    return dtype[out_shape].Reshape(results)
