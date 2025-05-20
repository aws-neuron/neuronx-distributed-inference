from math import log2


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


def generate_2d_buckets_for_prefix_cahcing(
    min_vertical_len: int, max_vertical_len: int, min_horizontal_len: int, max_horizontal_len: int
):
    """
    This uses 2 dimentional bucketing over vertical and horizontal dimentions.
    Vertical dimention corresponds to the number of active tokens.
    Horizontal dimension corresponds to the size of the prefix.
    """
    vertical_ranges = generate_buckets(min_vertical_len, max_vertical_len)
    horizontal_ranges = generate_buckets(min_horizontal_len, max_horizontal_len)

    buckets = []
    for vertical_range in vertical_ranges:
        for horizontal_range in horizontal_ranges:
            buckets.append([vertical_range, horizontal_range])
    return buckets
