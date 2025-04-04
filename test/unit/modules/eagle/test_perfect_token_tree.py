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
        "token_tree_test_configs/perfect_tree/2-branch.json",
        "token_tree_test_configs/perfect_tree/3-branch.json",
        "token_tree_test_configs/perfect_tree/4-branch.json",
        "token_tree_test_configs/perfect_tree/5-branch.json",
        "token_tree_test_configs/perfect_tree/6-branch.json",
        "token_tree_test_configs/perfect_tree/7-branch.json",
        "token_tree_test_configs/perfect_tree/8-branch.json",
        "token_tree_test_configs/perfect_tree/9-branch.json",
        "token_tree_test_configs/perfect_tree/10-branch.json",
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


TREE_DEPTH = 3


class TestTokenTree:
    @pytest.fixture(autouse=True)
    def setup(self, tree_config_path, error_logger):
        self.config, self.config_path = tree_config_path
        self.error_logger = error_logger
        self.token_tree = TokenTree(self.config)
        self.branch_num = self.find_branch_num()
        self.depth = TREE_DEPTH
        self.path = None
        self.drafted_nums = None

    def find_branch_num(self):
        match = re.search(r"/(\d+)-branch\.json$", self.config_path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Could not extract branch number from tree config filename")

    def test_basic_tree_stats(self):
        @self.error_logger
        def _test_tree_depth():
            expected_depth = self.depth
            assert self.token_tree.depth == expected_depth

        @self.error_logger
        def _test_tree_node_nums():
            expected_node_nums = 0
            for i in range(self.depth):
                expected_node_nums += self.branch_num**i
            assert self.token_tree.node_nums == expected_node_nums

        @self.error_logger
        def _test_tree_width():
            assert self.token_tree.width == self.branch_num ** (self.depth - 1)

        @self.error_logger
        def _test_tree_width_wo_leaf():
            assert self.token_tree.width_wo_leaf == self.branch_num ** (self.depth - 2)

        _test_tree_depth()
        _test_tree_node_nums()
        _test_tree_width()
        _test_tree_width_wo_leaf()

    def test_tree_drafted_nums(self):
        @self.error_logger
        def _test_tree_drafted_nums():
            drafted_nums = [0]
            for i in range(self.depth):
                drafted_nums.append(drafted_nums[-1] + self.branch_num**i)
            assert drafted_nums == self.token_tree.drafted_nums
            self.drafted_nums = drafted_nums

        _test_tree_drafted_nums()

    def test_tree_path(self):
        @self.error_logger
        def _test_tree_path(depth, branch_number):
            def generate_paths_recursive(current_depth, current_path):
                if current_depth == depth:
                    all_paths.append(current_path)
                    return

                start_node = current_path[-1] * branch_number + 1
                for i in range(branch_number):
                    new_path = current_path + [start_node + i]
                    generate_paths_recursive(current_depth + 1, new_path)

            all_paths = []
            generate_paths_recursive(1, [0])
            assert all_paths == (self.token_tree.path).tolist()

        _test_tree_path(self.depth, self.branch_num)

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
                logging.warning(position_id_offset)
                logging.warning(self.token_tree.position_id_offset[i])
                assert position_id_offset.tolist() == self.token_tree.position_id_offset[i]

        _test_tree_position_id_offset()

    def test_tree_rotary_position_id_offset(self):
        @self.error_logger
        def _test_tree_rotary_position_id_offset():
            rotary_position_id_offset = []
            for i in range(self.depth):
                for _ in range(self.branch_num**i):
                    rotary_position_id_offset.append(i)
            assert rotary_position_id_offset == self.token_tree.rotary_position_id_offset.tolist()

        _test_tree_rotary_position_id_offset()

    def test_tree_cache_scatter_indices(self):
        @self.error_logger
        def _test_tree_cache_scatter_indices():
            for idx, path in enumerate(self.token_tree.path):
                for i, node in enumerate(path):
                    assert self.token_tree.cache_scatter_indices[idx][node] == i
                assert len(set(self.token_tree.cache_scatter_indices[idx])) == len(
                    self.token_tree.cache_scatter_indices[idx]
                )

        _test_tree_cache_scatter_indices()

    def test_tree_full_attention_mask(self):
        @self.error_logger
        def _test_tree_full_attention_mask():
            def validate_attention_mask(attention_mask, depth, branch_number):
                # Calculate the total number of nodes in the tree
                num_nodes = (branch_number**depth - 1) // (branch_number - 1)

                # Check the shape of the attention mask
                assert attention_mask.shape == (
                    num_nodes,
                    num_nodes,
                ), f"Attention mask shape should be ({num_nodes}, {num_nodes})"

                # Check that the mask is binary (0 or 1)
                assert torch.all(
                    (attention_mask == 0) | (attention_mask == 1)
                ), "Attention mask should only contain 0 or 1"

                # Check that the diagonal is all 1s (each node can attend to itself)
                assert torch.all(
                    torch.diag(attention_mask) == 1
                ), "Diagonal of attention mask should be all 1s"

                # Check that the upper triangle (excluding diagonal) is all 0s
                upper_triangle = torch.triu(attention_mask, diagonal=1)
                assert torch.all(
                    upper_triangle == 0
                ), "Upper triangle of attention mask (excluding diagonal) should be all 0s"

                # Check the attention pattern for each node
                for node in range(num_nodes):
                    # Get the level of the current node
                    level = (node + 1).bit_length() - 1

                    # Check that each node can attend to its ancestors
                    parent = (node - 1) // branch_number
                    while parent >= 0:
                        assert (
                            attention_mask[node, parent] == 1
                        ), f"Node {node} should be able to attend to its ancestor {parent}"
                        parent = (parent - 1) // branch_number

                    # Check that each node can be attended by its descendants
                    if level < depth - 1:
                        start_child = node * branch_number + 1
                        end_child = start_child + branch_number
                        for child in range(start_child, end_child):
                            assert (
                                attention_mask[child, node] == 1
                            ), f"Node {child} should be able to attend to its parent {node}"

                    # Check that nodes cannot attend to or be attended by nodes in other branches
                    # (unless they are ancestors or descendants)
                    for other_node in range(num_nodes):
                        if other_node != node:
                            if (
                                attention_mask[node, other_node] == 1
                                or attention_mask[other_node, node] == 1
                            ):
                                assert is_ancestor_or_descendant(
                                    node, other_node, branch_number
                                ), f"Nodes {node} and {other_node} should not have attention connection (they are unrelated)"

                print("Attention mask validation passed!")

            def is_ancestor_or_descendant(node1, node2, branch_number):
                while node1 > node2:
                    node1 = (node1 - 1) // branch_number
                while node2 > node1:
                    node2 = (node2 - 1) // branch_number
                return node1 == node2

            validate_attention_mask(
                self.token_tree.full_tree_attn_mask, self.depth, self.branch_num
            )

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
