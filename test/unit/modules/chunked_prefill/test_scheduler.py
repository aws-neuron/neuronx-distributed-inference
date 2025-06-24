# flake8: noqa
import numpy as np
import pytest
import torch

from neuronx_distributed_inference.modules.chunked_prefill.scheduler import (
    FlashPASchedule,
    GridTileScheduler,
    ceil_div,
)


class TestFlashPASchedule:

    def _prepare_schedule(self):
        # Assuming
        # num_tiles = 6
        # tile_size_q = 6
        # tile_size_kv = 8

        block_size = 4
        tile_q_indices = torch.tensor(
            [0, 1, 1, 2, 1, 2], dtype=torch.int32
        )
        tile_block_table_offsets = torch.tensor(
            [0, 0, 2, 2, 4, 4], dtype=torch.int32
        )
        tile_q_seq_ids = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2],
                [0, 0, 0, 0, 1, 2],
                [2, 2, 3, 3, 3, 3],
                [0, 0, 0, 0, 1, 2],
                [2, 2, 3, 3, 3, 3],
            ]
        ).to(torch.int32)
        tile_kv_seq_ids = torch.tensor(
            [
                [0, 0, 0, 4, 1, 1, 1, 1],
                [0, 0, 0, 4, 1, 1, 1, 1],
                [1, 1, 4, 4, 2, 2, 2, 2],
                [1, 1, 4, 4, 2, 2, 2, 2],
                [2, 4, 4, 4, 4, 4, 4, 4],
                [2, 4, 4, 4, 4, 4, 4, 4],
            ]
        ).to(torch.int32)
        tile_kv_skip_indices = torch.tensor([1, 3, 5], dtype=torch.int32)

        schedule = FlashPASchedule(
            tile_q_indices,
            tile_block_table_offsets,
            tile_q_seq_ids,
            tile_kv_seq_ids,
            tile_kv_skip_indices,
            block_size,
        )
        return schedule

    def test_build_tile_masks(self):
        schedule = self._prepare_schedule()
        actual = schedule.build_tile_masks(b_p_size=4)

        expected = torch.tensor(
            [
               [[ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False]],

               [[ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [ True,  True, False, False,  True, False, False, False],
                [False, False,  True,  True, False, False,  True,  True],
                [False, False, False, False, False, False, False, False]],

               [[False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [ True, False, False, False,  True, False, False, False],
                [False, False,  True,  True, False, False,  True,  True]],

               [[False, False,  True,  True, False, False,  True,  True],
                [False, False,  True,  True, False, False,  True,  True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False]],

               [[False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False]],

               [[ True, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False]]
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_build_tile_block_tables(self):
        schedule = self._prepare_schedule()
        block_table = torch.arange(100)
        actual = schedule.build_tile_block_tables(
            block_tables=block_table,
            skip_value=2937,
        )

        # Skip loading KV cache for tiles 1, 3, 5
        expected = torch.tensor(
            [
                [   0,    1],
                [2937, 2937],
                [   2,    3],
                [2937, 2937],
                [   4,    5],
                [2937, 2937]
            ]
        )
        torch.testing.assert_close(actual, expected)


class TestPagedAttentionScheduler:
    def _compute_truth(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        prompt_lens,
        context_lens,
        column_order,
        pad_num_tile_to=None,
    ):
        prompt_ends = np.cumsum(prompt_lens)
        prompt_starts = np.concatenate([[0], prompt_ends[:-1]])

        padded_context_lens = ceil_div(context_lens, block_size) * block_size
        context_starts = np.concatenate([[0], np.cumsum(padded_context_lens)[:-1]])
        context_ends = context_starts + context_lens

        total_prompt = prompt_ends[-1]
        total_context = context_ends[-1]
        num_q_tile = ceil_div(total_prompt, tile_size_q)
        num_kv_tile = ceil_div(total_context, tile_size_kv)

        def _is_tile_needed(q_tile_idx, kv_tile_idx):
            q_start = q_tile_idx * tile_size_q
            q_end = q_start + tile_size_q
            kv_start = kv_tile_idx * tile_size_kv
            kv_end = kv_start + tile_size_kv
            for seq_id in range(num_seqs):
                seq_q_start = prompt_starts[seq_id]
                seq_q_end = prompt_ends[seq_id]
                seq_kv_start = context_starts[seq_id]
                seq_kv_end = context_ends[seq_id]
                # check if seq has overlap with tile
                if q_start >= seq_q_end or q_end <= seq_q_start:
                    continue
                if kv_start >= seq_kv_end or kv_end <= seq_kv_start:
                    continue
                # has overlap, this tile is needed
                return True
            return False

        # prepare seq ids
        def _gen_seq_ids(size, default_value, seq_starts, seq_ends):
            seq_ids = np.full((size,), default_value, dtype=np.int32)
            for seq_id in range(num_seqs):
                seq_ids[seq_starts[seq_id] : seq_ends[seq_id]] = seq_id
            return seq_ids

        q_seq_ids = _gen_seq_ids(num_q_tile * tile_size_q, num_seqs, prompt_starts, prompt_ends)
        kv_seq_ids = _gen_seq_ids(
            num_kv_tile * tile_size_kv, num_seqs + 1, context_starts, context_ends
        )

        tile_q_indices = []
        tile_block_table_offsets = []
        tile_q_seq_ids = []
        tile_kv_seq_ids = []
        tile_kv_skip_indices = []
        last_tile_idx = -1
        for tile_idx in range(num_kv_tile * num_q_tile):
            if column_order:
                q_tile_idx = tile_idx % num_q_tile
                kv_tile_idx = tile_idx // num_q_tile
            else:
                q_tile_idx = tile_idx // num_kv_tile
                kv_tile_idx = tile_idx % num_kv_tile
            if _is_tile_needed(q_tile_idx, kv_tile_idx):
                if kv_tile_idx == last_tile_idx:
                    tile_kv_skip_indices.append(len(tile_q_indices))
                last_tile_idx = kv_tile_idx
                tile_q_indices.append(q_tile_idx)
                tile_block_table_offsets.append(kv_tile_idx * tile_size_kv // block_size)
                tile_q_seq_ids.append(
                    q_seq_ids[q_tile_idx * tile_size_q : (q_tile_idx + 1) * tile_size_q]
                )
                tile_kv_seq_ids.append(
                    kv_seq_ids[kv_tile_idx * tile_size_kv : (kv_tile_idx + 1) * tile_size_kv]
                )

        if pad_num_tile_to is not None:
            while len(tile_q_indices) < pad_num_tile_to:
                tile_block_table_offsets.append(0)
                tile_q_seq_ids.append(np.zeros((tile_size_q,), dtype=np.int32))
                tile_kv_seq_ids.append(np.ones((tile_size_kv,), dtype=np.int32))
                tile_kv_skip_indices.append(len(tile_q_indices))
                tile_q_indices.append(0)

        return FlashPASchedule(
            tile_q_indices=torch.tensor(tile_q_indices, dtype=torch.int32),
            tile_block_table_offsets=torch.tensor(tile_block_table_offsets, dtype=torch.int32),
            tile_q_seq_ids=torch.tensor(tile_q_seq_ids, dtype=torch.int32),
            tile_kv_seq_ids=torch.tensor(tile_kv_seq_ids, dtype=torch.int32),
            tile_kv_skip_indices=torch.tensor(tile_kv_skip_indices, dtype=torch.int32),
            block_size=block_size,
        )

    def _validate_schedule(self, pa_schedule, truth):
        if pa_schedule.num_tiles == 0:
            assert truth.num_tiles == 0
        else:
            assert np.all(pa_schedule.tile_q_indices == truth.tile_q_indices)
            assert np.all(pa_schedule.tile_block_table_offsets == truth.tile_block_table_offsets)
            assert np.all(pa_schedule.tile_q_seq_ids == truth.tile_q_seq_ids)
            assert np.all(pa_schedule.tile_kv_seq_ids == truth.tile_kv_seq_ids)
            assert np.all(pa_schedule.tile_kv_skip_indices == truth.tile_kv_skip_indices)

    def _run_test(
        self,
        prompt_lens,
        context_lens,
        block_size,
        tile_size_q,
        tile_size_kv,
        column_order,
        num_tile_to_pad=None,
    ):
        assert len(prompt_lens) == len(context_lens)
        num_seqs = len(prompt_lens)
        scheduler = GridTileScheduler(
            prompt_lens=torch.tensor(prompt_lens),
            context_lens=torch.tensor(context_lens),
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            block_size=block_size,
            column_order=column_order,
        )
        pa_schedule = scheduler.compute_schedule()
        num_tile_after_pad = None
        if num_tile_to_pad is not None:
            num_tile_after_pad = pa_schedule.num_tiles + num_tile_to_pad
            pa_schedule = pa_schedule.pad_schedule(num_tile_after_pad)
        truth = self._compute_truth(
            num_seqs=num_seqs,
            block_size=block_size,
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            prompt_lens=prompt_lens,
            context_lens=context_lens,
            column_order=column_order,
            pad_num_tile_to=num_tile_after_pad,
        )
        self._validate_schedule(pa_schedule, truth)
        return pa_schedule

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    @pytest.mark.parametrize("column_order", [False, True])
    @pytest.mark.parametrize("num_tile_to_pad", [0, 4, -3])
    def test_random_size(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        column_order,
        num_tile_to_pad,
    ):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = 4096
        prompt_lens = np.random.randint(1, MAX_QUERY_LEN_PER_SEQ, size=(num_seqs,))
        context_lens = np.random.randint(1, MAX_KV_LEN_PER_SEQ, size=(num_seqs,))
        self._run_test(
            prompt_lens,
            context_lens,
            block_size,
            tile_size_q,
            tile_size_kv,
            column_order,
            num_tile_to_pad=num_tile_to_pad,
        )

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    @pytest.mark.parametrize("offset", [0, 1, -1])
    @pytest.mark.parametrize("column_order", [False, True])
    def test_aligned_size_plus_offset(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        offset,
        column_order,
    ):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 2 * tile_size_q
        MAX_KV_LEN_PER_SEQ = 2 * tile_size_kv
        prompt_lens = np.random.randint(1, MAX_QUERY_LEN_PER_SEQ, size=(num_seqs,))
        prompt_lens = ceil_div(prompt_lens, tile_size_q) * tile_size_q
        context_lens = np.random.randint(1, MAX_KV_LEN_PER_SEQ, size=(num_seqs,))
        context_lens = ceil_div(context_lens, block_size) * block_size
        # shift the first element by offset so that all elements are aligned to tile boundary + offset
        prompt_lens[0] += offset
        context_lens[0] += offset
        self._run_test(
            prompt_lens, context_lens, block_size, tile_size_q, tile_size_kv, column_order
        )

    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 128, 128, 2048],
            [16, 128, 128, 4096],
            [16, 128, 512, 4096],
        ],
    )
    @pytest.mark.parametrize("column_order", [False, True])
    def test_size1_q_and_empty_kv(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        column_order,
    ):
        assert tile_size_kv % block_size == 0
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = 4096
        prompt_lens = np.random.randint(1, MAX_QUERY_LEN_PER_SEQ, size=(num_seqs,))
        context_lens = np.random.randint(1, MAX_KV_LEN_PER_SEQ, size=(num_seqs,))
        # randomly pick 50% of sequences to have zero context tokens
        zero_len_seq = np.random.choice(num_seqs, size=(num_seqs // 2,))
        context_lens[zero_len_seq] = 0
        self._run_test(
            prompt_lens, context_lens, block_size, tile_size_q, tile_size_kv, column_order
        )

    # test the case with a large amount of padding and empty tiles
    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [8, 512, 8, 16],
            [16, 512, 8, 16],
            [32, 1024, 16, 16],
        ],
    )
    @pytest.mark.parametrize("column_order", [False, True])
    def test_small_tile_large_padding(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        column_order,
    ):
        MAX_QUERY_LEN_PER_SEQ = 128
        MAX_KV_LEN_PER_SEQ = block_size // 2
        prompt_lens = np.random.randint(1, MAX_QUERY_LEN_PER_SEQ, size=(num_seqs,))
        context_lens = np.random.randint(1, MAX_KV_LEN_PER_SEQ, size=(num_seqs,))
        self._run_test(
            prompt_lens, context_lens, block_size, tile_size_q, tile_size_kv, column_order
        )

    # test no prior, only active, i.e. full prefill
    @pytest.mark.parametrize(
        "num_seqs, block_size, tile_size_q, tile_size_kv",
        [
            [3, 16, 128, 2048],
            [11, 128, 512, 4096],
        ],
    )
    @pytest.mark.parametrize("column_order", [False, True])
    def test_prefill_no_ctx(
        self,
        num_seqs,
        block_size,
        tile_size_q,
        tile_size_kv,
        column_order,
    ):
        MAX_QUERY_LEN_PER_SEQ = 128
        prompt_lens = np.random.randint(1, MAX_QUERY_LEN_PER_SEQ, size=(num_seqs,))
        context_lens = np.zeros_like(prompt_lens)
        pa_schedule = self._run_test(
            prompt_lens, context_lens, block_size, tile_size_q, tile_size_kv, column_order
        )
        assert pa_schedule.num_tiles == 0
