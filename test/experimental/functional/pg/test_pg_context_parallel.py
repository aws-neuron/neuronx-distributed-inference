import pytest
import re
import torch

from neuronx_distributed.trace.mock_torchdist import mock_distributed
import neuronx_distributed_inference.experimental.functional.pg.context_parallel as cp


TRN2_8_BY_8_CP_MESH = [
    [0, 4, 16, 20, 32, 36, 48, 52],
    [1, 5, 17, 21, 33, 37, 49, 53],
    [2, 6, 18, 22, 34, 38, 50, 54],
    [3, 7, 19, 23, 35, 39, 51, 55],
    [12, 8, 28, 24, 44, 40, 60, 56],
    [13, 9, 29, 25, 45, 41, 61, 57],
    [14, 10, 30, 26, 46, 42, 62, 58],
    [15, 11, 31, 27, 47, 43, 63, 59]
]


@pytest.fixture(scope="function", autouse=True)
def reset_process_groups():
    """
    Fixture that resets the global state maintained in the context parallel class
    """
    yield
    cp._ATTENTION_TP_CP_GROUP = None
    cp._ATTENTION_CP_GROUP = None


@pytest.mark.parametrize("world_size, cp_degree, expected_tp_mesh",[
    (16, 8, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]),
    (32, 8, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]),
    (64, 8, cp._TRN2_8_BY_8_TP_MESH),
])
def test_get_context_parallel_tp_mesh(world_size, cp_degree, expected_tp_mesh):
    actual = cp.get_context_parallel_tp_mesh(world_size, cp_degree)

    assert actual == expected_tp_mesh


@pytest.mark.parametrize("world_size, cp_degree, expected_cp_mesh",[
    (16, 8, [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]),
    (32, 8, [[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]),
    (64, 8, TRN2_8_BY_8_CP_MESH),
])
def test_get_context_parallel_cp_mesh(world_size, cp_degree, expected_cp_mesh):
    actual = cp.get_context_parallel_cp_mesh(world_size, cp_degree)

    assert actual == expected_cp_mesh


@pytest.mark.parametrize("world_size, cp_degree, expected_tp_mesh",[
    (16, 8, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]),
    (32, 8, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]),
    (64, 8, cp._TRN2_8_BY_8_TP_MESH),
])
def test_initialize_context_parallel_tp_group(world_size, cp_degree, expected_tp_mesh):
    with mock_distributed(world_size=world_size):
        cp.initialize_context_parallel_tp_group(world_size, cp_degree)

        actual = cp.get_context_parallel_tp_group()._mesh
        assert actual == expected_tp_mesh


def test_initialize_context_parallel_tp_group_invalid_cp_degree():
    with pytest.raises(AssertionError, match=re.escape("World size (16) must be evenly divisble by CP degree (5)")):
        cp.initialize_context_parallel_tp_group(world_size=16, cp_degree=5)


def test_initialize_context_parallel_tp_group_already_initialized():
    cp._ATTENTION_TP_CP_GROUP = 1

    cp.initialize_context_parallel_tp_group(world_size=16, cp_degree=2)

    actual = cp.get_context_parallel_tp_group()
    assert actual == 1


@pytest.mark.parametrize("world_size, cp_degree, expected_cp_mesh",[
    (16, 8, [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]),
    (32, 8, [[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]),
    (64, 8, TRN2_8_BY_8_CP_MESH),
])
def test_initialize_context_parallel_cp_group(world_size, cp_degree, expected_cp_mesh):
    with mock_distributed(world_size=world_size):
        cp.initialize_context_parallel_cp_group(world_size, cp_degree)

        actual = cp.get_context_parallel_cp_group()._mesh
        assert actual == expected_cp_mesh


def test_initialize_context_parallel_cp_group_invalid_cp_degree():
    with pytest.raises(AssertionError, match=re.escape("World size (16) must be evenly divisble by CP degree (5)")):
        cp.initialize_context_parallel_cp_group(world_size=16, cp_degree=5)


def test_initialize_context_parallel_cp_group_already_initialized():
    cp._ATTENTION_CP_GROUP = 1

    cp.initialize_context_parallel_cp_group(world_size=16, cp_degree=2)

    actual = cp.get_context_parallel_cp_group()
    assert actual == 1


@pytest.mark.parametrize("world_size, cp_degree, expected_tp_mesh, expected_cp_mesh",[
    (16, 8, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]], [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]),
    (32, 8, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]],
     [[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]),
    (64, 8, cp._TRN2_8_BY_8_TP_MESH, TRN2_8_BY_8_CP_MESH),
])
def test_initialize_context_parallel_process_groups(world_size, cp_degree, expected_tp_mesh, expected_cp_mesh):
    with mock_distributed(world_size=world_size):
        cp.initialize_context_parallel_process_groups(world_size, cp_degree)

        actual_tp_mesh = cp.get_context_parallel_tp_group()._mesh
        actual_cp_mesh = cp.get_context_parallel_cp_group()._mesh
        assert actual_tp_mesh == expected_tp_mesh
        assert actual_cp_mesh == expected_cp_mesh


def test_get_context_parallel_tp_group_uninitialized():
    with pytest.raises(AssertionError, match="_ATTENTION_TP_CP_GROUP is not initialized"):
        cp.get_context_parallel_tp_group()


def test_get_context_parallel_cp_group_uninitialized():
    with pytest.raises(AssertionError, match="_ATTENTION_CP_GROUP is not initialized"):
        cp.get_context_parallel_cp_group()


@pytest.mark.parametrize("rank,world_size,cp_degree,expected_result", [
    (torch.tensor(1), 16, 4, torch.tensor(0)),
    (torch.tensor(11), 16, 4, torch.tensor(2)),
    (torch.tensor(0), 16, 8, torch.tensor(0)),
    (torch.tensor(15), 16, 8, torch.tensor(7)),
    (torch.tensor(1), 64, 8, torch.tensor(0)),
    (torch.tensor(31), 64, 8, torch.tensor(2)),
])
def test_get_cp_rank(rank, world_size, cp_degree, expected_result):
    actual = cp.get_cp_rank(rank=rank, world_size=world_size, cp_degree=cp_degree)

    assert actual.item() == expected_result.item()
    assert actual.dtype == torch.int32


def test_get_cp_rank_invalid_cp_degree():
    rank = torch.tensor(1)
    world_size = 8
    cp_degree = 9

    with pytest.raises(AssertionError, match="Cp degree size should be <= world_size"):
        cp.get_cp_rank(rank=rank, world_size=world_size, cp_degree=cp_degree)


def test_get_cp_rank_invalid_rank():
    rank = torch.tensor(10)
    world_size = 8
    cp_degree = 2

    with pytest.raises(AssertionError, match="Rank should be between 0 and"):
        cp.get_cp_rank(rank=rank, world_size=world_size, cp_degree=cp_degree)


def test_get_rank_8_by_8():
    for rank in range(64):
        rank = torch.tensor(rank)

        actual = cp.get_rank_8_by_8(rank)

        assert rank.item() in cp._TRN2_8_BY_8_TP_MESH[actual.item()]


def test_get_rank_8_by_8_invalid_rank():
    rank = torch.tensor(64)

    with pytest.raises(AssertionError, match="Rank must be between 0 and 63"):
        cp.get_rank_8_by_8(rank)
