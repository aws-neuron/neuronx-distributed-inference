from math import log2, ceil
from neuronx_distributed_inference.models.config import InferenceConfig


BUCKET_SELECTION_STRATEGIES = {"max", "first_fit", "second_fit"}


def generate_buckets(min_length: int, max_length: int):
    if min_length == max_length:
        return [max_length]

    min_bound = int(log2(min_length))
    max_bound = round(log2(max_length))  # we use round because it creates optimal bucket spacing

    # NOTE: because range operates on [a,b), and we rounded the log2 result
    # we won't get 2**i results close to the max_length.
    # ex. we won't see bucket spacing of [128,256,512,513] or [128,256,510,512]
    buckets = [2**i for i in range(min_bound, max_bound)] + [max_length]
    return buckets


def generate_2d_buckets_for_prefix_caching(
    min_vertical_len: int, max_vertical_len: int, min_horizontal_len: int, max_horizontal_len: int, is_context_encode=False
):
    """
    This uses 2 dimensional bucketing over vertical and horizontal dimensions.
    Vertical dimension corresponds to the number of active tokens.
    Horizontal dimension corresponds to the size of the prefix.

    TODO: Add UT for this func
    """
    vertical_ranges = generate_buckets(min_vertical_len, max_vertical_len)
    horizontal_ranges = generate_buckets(min_horizontal_len, max_horizontal_len)
    # Add a special case for no prefix
    if is_context_encode:
        horizontal_ranges = [0] + horizontal_ranges

    buckets = []
    for vertical_range in vertical_ranges:
        for horizontal_range in horizontal_ranges:
            buckets.append([vertical_range, horizontal_range])
    return buckets


def generate_2d_buckets_for_prefix_caching_from_config(
    vertical_ranges, horizontal_ranges, is_context_encode=False
):
    """
    This uses 2 dimentional bucketing over vertical and horizontal dimentions.
    Vertical dimention corresponds to the number of active tokens.
    Horizontal dimension corresponds to the size of the prefix.

    TODO: Add UT for this func
    """
    if is_context_encode:
        horizontal_ranges = [0] + horizontal_ranges

    buckets = []
    for vertical_range in vertical_ranges:
        for horizontal_range in horizontal_ranges:
            buckets.append([vertical_range, horizontal_range])
    return buckets


def generate_buckets_on_chunk_size(
    q_tile_size: int,
    max_context_len: int,
):
    """
    Generate buckets on the chunk size, for chunked prefill only.

    This will at most generate 3 buckets, i.e., [smallest_bucket,
    median_bucket, max_bucket], and any of the bucket options will be a
    multiple of q_tile_size. It generates limited number of buckets, to avoid
    resulting in too many buckets when combining with other bucketing
    dimensions.

    This design on the bucketing is based on experience, and it is
    subject to changes in future.
    """
    if max_context_len < q_tile_size:
        return [q_tile_size]

    num_q_tiles = ceil(max_context_len / q_tile_size)

    all_buckets = list(range(1, num_q_tiles + 1))
    all_buckets = [b * q_tile_size for b in all_buckets]

    left_index = 0
    right_index = len(all_buckets) - 1
    median_index = right_index // 2

    chunk_size_buckets = [all_buckets[left_index]]
    if median_index > left_index:
        chunk_size_buckets.append(all_buckets[median_index])
    if right_index > median_index:
        chunk_size_buckets.append(all_buckets[right_index])
    return chunk_size_buckets


def generate_buckets_for_chunked_prefill_cte(inference_config: InferenceConfig):
    """
    Generate buckets for chunked prefill context encoding model

    It has two bucketing dimensions:
        - chunk size: how many new tokens to process per iteration
        - number of tiles: how many tiles to compute for attention per iteration

    Chunked prefill in NxDI splits attention computation with a tile grid, and
    the tile size is defined by `kernel_q_tile_size` and `kernel_kv_tile_size`.

    For example, if the chunks size=256, max_len=1024, max_num_seqs=8,
    kernel_q_tile_size=128, kernel_kv_tile_size=512, the max number of tiles
    needed will be (256/128) * (8*1024/512) = 2 * 16 = 32. Since not all the
    tiles are needed by the final result, it has a bucketing dimension on the
    number of tiles.
    """
    neuron_config = inference_config.neuron_config
    chunked_prefill_config = neuron_config.chunked_prefill_config
    max_context_length = neuron_config.max_context_length
    assert chunked_prefill_config is not None

    q_tile_size = chunked_prefill_config.kernel_q_tile_size
    kv_tile_size = chunked_prefill_config.kernel_kv_tile_size

    max_q_len = max_context_length
    max_kv_cache_len = neuron_config.max_length * chunked_prefill_config.max_num_seqs
    max_num_tiles = ceil(max_q_len / q_tile_size) * ceil(max_kv_cache_len / kv_tile_size)

    if not neuron_config.enable_bucketing:
        return [[max_context_length, max_num_tiles]]
    elif neuron_config.context_encoding_buckets is not None:
        return neuron_config.context_encoding_buckets

    # Generate buckets on chunk size dimension
    chunk_size_buckets = generate_buckets_on_chunk_size(q_tile_size, max_context_length)

    # Generate buckets on num of tiles dimension
    if max_num_tiles >= 4:
        tile_buckets = [1, max_num_tiles // 2, max_num_tiles]
    else:
        tile_buckets = [max_num_tiles]

    # return buckets in sorted order
    buckets = [[q, tile] for q in chunk_size_buckets for tile in tile_buckets]
    return buckets


def generate_buckets_for_cte(inference_config: InferenceConfig):
    """
    Generate buckets for context encoding model

    If bucketing is enable, for basic cases it will generate buckets
    for different input seq len, ranging from 128 to the max_context_length.

    TODO: Extract prefix caching bucketing logics out for a cleaner
    code base.
    """
    if inference_config.neuron_config.is_chunked_prefill:
        return generate_buckets_for_chunked_prefill_cte(inference_config)

    if not inference_config.neuron_config.enable_bucketing:
        if inference_config.neuron_config.is_prefix_caching:
            buckets = generate_2d_buckets_for_prefix_caching(
                inference_config.neuron_config.max_context_length,
                inference_config.neuron_config.max_context_length,
                inference_config.neuron_config.max_context_length,
                inference_config.neuron_config.max_context_length,
                True,
            )
        else:
            buckets = generate_buckets(
                inference_config.neuron_config.max_context_length,
                inference_config.neuron_config.max_context_length,
            )
    else:
        if inference_config.neuron_config.context_encoding_buckets is not None:
            if inference_config.neuron_config.is_prefix_caching:
                buckets = generate_2d_buckets_for_prefix_caching_from_config(
                    inference_config.neuron_config.context_encoding_buckets,
                    inference_config.neuron_config.prefix_buckets,
                    True,
                )
            else:
                buckets = inference_config.neuron_config.context_encoding_buckets
        else:
            if inference_config.neuron_config.is_prefix_caching:
                # currently we have a requirement of minimal 512 to use the FA kernel
                buckets = generate_2d_buckets_for_prefix_caching(
                    512,
                    inference_config.neuron_config.max_context_length,
                    512,
                    inference_config.neuron_config.max_context_length,
                    True,
                )
            else:
                buckets = generate_buckets(
                    128, inference_config.neuron_config.max_context_length
                )
    return buckets


def generate_buckets_for_tkg(inference_config: InferenceConfig):
    """
    Generate buckets for token generation model

    If bucketing is enable, for basic cases it will generate buckets
    for different KV cache len, ranging from 128 to the max_length.

    TODO: Extract prefix caching bucketing logics out for a cleaner
    code base.
    """
    if not inference_config.neuron_config.enable_bucketing:
        if inference_config.neuron_config.is_prefix_caching:
            buckets = generate_2d_buckets_for_prefix_caching(
                1,
                1,
                inference_config.neuron_config.max_length,
                inference_config.neuron_config.max_length
            )
        else:
            buckets = generate_buckets(
                inference_config.neuron_config.max_length,
                inference_config.neuron_config.max_length
            )
    else:
        if inference_config.neuron_config.token_generation_buckets is not None:
            if inference_config.neuron_config.is_prefix_caching:
                buckets = inference_config.neuron_config.token_generation_buckets
                new_buckets = []
                for i in buckets:
                    new_buckets.append([1, i])
                buckets = new_buckets
            else:
                buckets = inference_config.neuron_config.token_generation_buckets
        else:
            if inference_config.neuron_config.is_prefix_caching:
                # currently we have a requirement of minimal 256 to use the block attn nki kernel
                buckets = generate_2d_buckets_for_prefix_caching(
                    1, 1, 256, inference_config.neuron_config.max_length
                )
            else:
                buckets = generate_buckets(
                    128, inference_config.neuron_config.max_length
                )
    return buckets


def generate_buckets_for_fused_spec(inference_config: InferenceConfig):
    """
    Generate buckets for fused speculation model

    TODO: Extract prefix caching bucketing logics out for a cleaner
    code base.
    """
    if not inference_config.neuron_config.enable_bucketing:
        if inference_config.neuron_config.is_prefix_caching:
            buckets = generate_2d_buckets_for_prefix_caching(
                1,
                1,
                inference_config.neuron_config.max_length,
                inference_config.neuron_config.max_length
            )
        else:
            buckets = generate_buckets(
                inference_config.neuron_config.max_length,
                inference_config.neuron_config.max_length
            )
    else:
        if inference_config.neuron_config.token_generation_buckets is not None:
            if inference_config.neuron_config.is_prefix_caching:
                buckets = inference_config.neuron_config.token_generation_buckets
                new_buckets = []
                for i in buckets:
                    new_buckets.append([1, i])
                buckets = new_buckets
            else:
                buckets = inference_config.neuron_config.token_generation_buckets
        else:
            if inference_config.neuron_config.is_prefix_caching:
                # currently we have a requirement of minimal 256 to use the block attn nki kernel
                buckets = generate_2d_buckets_for_prefix_caching(
                    1, 1, 256, inference_config.neuron_config.max_length
                )
            else:
                buckets = generate_buckets(
                    128, inference_config.neuron_config.max_length
                )
    return buckets


def generate_buckets_for_speculation(inference_config: InferenceConfig):
    """
    Generate buckets for vanilla speculation model
    """
    if not inference_config.neuron_config.enable_bucketing:
        buckets = generate_buckets(
            inference_config.neuron_config.max_length,
            inference_config.neuron_config.max_length
        )
    else:
        if inference_config.neuron_config.token_generation_buckets is not None:
            buckets = inference_config.neuron_config.token_generation_buckets
        else:
            buckets = generate_buckets(
                128, inference_config.neuron_config.max_length
            )
    return buckets
