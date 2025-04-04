import json
import logging
import os
import re

import pytest
import torch

from neuronx_distributed_inference.modules.eagle.token_tree import TokenTree

logger = logging.getLogger(__name__)


@pytest.fixture(
    params=[
        "token_tree_test_configs/other_tree/config_1.json",
        "token_tree_test_configs/other_tree/config_2.json",
        "token_tree_test_configs/other_tree/config_3.json",
        "token_tree_test_configs/other_tree/config_4.json",
        "token_tree_test_configs/other_tree/config_5.json",
        # Add more configuration files as needed
    ]
)
def tree_config_path(request):
    config_path = os.path.join(os.path.dirname(__file__), request.param)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, config_path


@pytest.fixture
def error_logger():
    def _error_logger(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__} with args: {args}, kwargs: {kwargs}")
                raise e

        return wrapper

    return _error_logger


class TestTokenTree:
    @pytest.fixture(autouse=True)
    def setup(self, tree_config_path, error_logger):
        self.config, self.config_path = tree_config_path
        self.error_logger = error_logger
        self.token_tree = TokenTree(self.config)
        self.depth = self.token_tree.depth

    def test_tree_drafted_nums(self):
        @self.error_logger
        def _test_tree_drafted_nums():
            drafted_nums = [0]
            for i in range(self.depth):
                drafted_nums.append(drafted_nums[-1] + self.token_tree.level_node_count[i])
            assert drafted_nums == self.token_tree.drafted_nums

        _test_tree_drafted_nums()

    def test_tree_path(self):
        @self.error_logger
        def _test_tree_path():
            for path in self.token_tree.path:
                i = 0
                prev_node = 0
                path = path.tolist()
                for node in path:
                    if node > prev_node:
                        assert node in self.token_tree.level_node_ids[i]
                        assert node in self.token_tree.tree_config[prev_node]
                    elif node == 0:
                        assert node in self.token_tree.level_node_ids[i]
                    else:
                        break
                    prev_node = node
                    i += 1

        _test_tree_path()

    def test_tree_path_permute_mask(self):
        @self.error_logger
        def _test_tree_path_permute_mask():
            for idx, path in enumerate(self.token_tree.path):
                assert torch.equal(
                    self.token_tree.path_permute_mask[idx, : self.token_tree.depth], path
                )

        _test_tree_path_permute_mask()

    def test_tree_position_id_offset(self):
        @self.error_logger
        def _test_tree_position_id_offset():
            for i in range(self.depth - 1):
                position_id_offset = self.token_tree.drafted_nums[i] + torch.arange(
                    0, self.token_tree.width
                )

                assert position_id_offset.tolist() == self.token_tree.position_id_offset[i]

        _test_tree_position_id_offset()

    def test_tree_rotary_position_id_offset(self):
        @self.error_logger
        def _test_tree_rotary_position_id_offset():
            rotary_position_id_offset = []
            for i in range(self.depth):
                for _ in self.token_tree.level_node_ids[i]:
                    rotary_position_id_offset.append(i)
            assert rotary_position_id_offset == self.token_tree.rotary_position_id_offset.tolist()

        _test_tree_rotary_position_id_offset()

    def test_tree_cache_scatter_indices(self):
        @self.error_logger
        def _test_tree_cache_scatter_indices():
            for idx, path in enumerate(self.token_tree.path):
                prev_node = 0
                for i, node in enumerate(path):
                    if prev_node > node:
                        break
                    assert self.token_tree.cache_scatter_indices[idx][node] == i
                    prev_node = node
                assert len(set(self.token_tree.cache_scatter_indices[idx])) == len(
                    self.token_tree.cache_scatter_indices[idx]
                )

        _test_tree_cache_scatter_indices()

    def test_tree_full_attention_mask(self):
        @self.error_logger
        def _test_tree_full_attention_mask():

            def process_tree(tree):
                all_children_dict = {node: set([node]) for node in tree.keys()}

                def dfs(node):
                    for child in tree.get(node, []):
                        # Add child to all_children_dict for the current node and all its ancestors
                        current = node
                        while current is not None:
                            all_children_dict[current].add(child)
                            all_children_dict[current].update(all_children_dict[child])
                            current = next((k for k, v in tree.items() if current in v), None)

                        # Recursive call for the child
                        dfs(child)

                # Start DFS from each root node (nodes without parents)
                for root in set(tree.keys()) - set(sum(tree.values(), [])):
                    dfs(root)

                # Convert sets to sorted lists in all_children_dict
                all_children_dict = {k: sorted(list(v)) for k, v in all_children_dict.items()}

                return all_children_dict

            all_children_dict = process_tree(self.config)

            count = 0
            full_tree_attn_mask = self.token_tree.full_tree_attn_mask.tolist()
            for k, v in all_children_dict.items():
                for node in v:
                    count += 1
                    assert full_tree_attn_mask[int(node)][int(k)] == 1
            assert count == sum([sum(sublist) for sublist in full_tree_attn_mask])

        _test_tree_full_attention_mask()

    def test_tree_token_matching(self):
        @self.error_logger
        def _test_tree_token_matching():
            # Mock data
            candidate_input_ids = torch.randint(
                0, 32000, (1, self.token_tree.node_nums), dtype=torch.int32
            )
            target_tokens = torch.randint(
                0, 32000, (1, self.token_tree.node_nums), dtype=torch.int32
            )

            # Perform the token matching process
            paths = self.token_tree.path.to(device=candidate_input_ids.device, dtype=torch.int32)
            parent_paths = self.token_tree.parent_path.to(
                device=candidate_input_ids.device, dtype=torch.int32
            )

            path_idx = torch.randint(0, self.token_tree.path.shape[0], (), dtype=torch.int32)

            path = paths[path_idx, :]
            i = 0
            prev_node = 0
            path = path.tolist()
            for node in path:
                if node > prev_node:
                    assert node in self.token_tree.level_node_ids[i]
                    assert node in self.token_tree.tree_config[prev_node]
                elif node == 0:
                    assert node in self.token_tree.level_node_ids[i]
                else:
                    break
                prev_node = node
                i += 1

            path_len = min(i, len(path) - 1)

            for i in range(path_len):
                nxt_node = path[i + 1]
                cur_node = path[i]
                temp = torch.randint(0, 32000, (), dtype=torch.int32)
                candidate_input_ids[:, nxt_node] = temp
                target_tokens[:, cur_node] = temp

            candidate_input_ids_comp = candidate_input_ids[:, paths]
            target_tokens_comp = target_tokens[:, parent_paths]

            index = (
                (~(candidate_input_ids_comp[:, :, 1:] == target_tokens_comp[:, :, :-1])).cumsum(
                    dim=-1
                )
                < 1
            ).sum(dim=-1)

            dest_idx = index.argmax(dim=1)
            dest_idx = dest_idx.unsqueeze(0)
            dest_len = torch.gather(index, dim=1, index=dest_idx)

            # Assertions
            assert dest_idx.shape == (
                1,
                1,
            ), f"Expected dest_idx shape (1, 2), but got {dest_idx.shape}"
            assert dest_len.shape == (
                1,
                1,
            ), f"Expected dest_len shape (1, 2), but got {dest_len.shape}"

            # Check if dest_idx contains the correct indices
            expected_dest_idx = torch.tensor(path_idx).expand_as(dest_idx)
            assert torch.all(
                dest_idx == expected_dest_idx
            ), f"Expected dest_idx {expected_dest_idx}, but got {dest_idx}"

            # Check if dest_len contains the correct lengths
            expected_dest_len = torch.tensor(path_len).expand_as(dest_len)
            assert torch.all(
                dest_len == expected_dest_len
            ), f"Expected dest_len {expected_dest_len}, but got {dest_len}"

        _test_tree_token_matching()
