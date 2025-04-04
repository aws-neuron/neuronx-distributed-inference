import copy
import logging
from collections import deque

import torch


class TokenTree:
    def __init__(self, tree_config):
        """Initialize TokenTree with a tree configuration.

        Args:
            tree_config (dict[int, list[int]]): Adjacency list representation of the tree
        """
        try:
            self.tree_config = self.parse_tree_config(tree_config)
            logging.warning(f"tree_config: {self.tree_config}")
            self.tree = {}
            self.depth = 0
            self.node_nums = 0
            self.max_width = 0
            self.level = []
            self.level_child = []

            # Attention masks
            self.full_tree_attn_mask = None
            self.level_attn_mask = None
            self.prior_tt_mask = None

            # Initialize tree structure
            if self.tree_config:
                self.init_token_tree()

            self.width = max(self.level)
            self.width_wo_leaf = max(self.level[:-1])

            # Compute tensors for token tree based Eagle SD
            self.level_node_count = self.get_level_node_count()
            self.level_internal_node_count = self.get_level_internal_node_count()
            self.path, self.parent_path, self.cache_scatter_indices = self.get_all_paths_tensor()
            self.path_permute_mask, self.path = self.get_all_permute_mask(self.path)
            self.parent_path_permute_mask, self.parent_path = self.get_all_permute_mask(
                self.parent_path
            )
            self.rotary_position_id_offset = self.get_rotary_position_id()
            self.drafted_nums = self.get_drafted_num()
            self.position_id_offset = self.get_position_id_offset()
            self.topk_permute_index = self.get_topk_permute_index()
            self.draft_hidden_gather_index = self.get_draft_hidden_gather_index()

            # assert tensors values are non negative
            assert self.width >= 0
            assert all(x >= 0 for x in self.level_node_count)
            for path in self.path:
                assert all(x >= 0 for x in path)
            assert torch.all(self.path_permute_mask >= 0)
            assert torch.all(self.rotary_position_id_offset >= 0)
            for topk_permute_index in self.topk_permute_index:
                assert all(x >= 0 for x in topk_permute_index)
            for draft_hidden_gather_index in self.draft_hidden_gather_index:
                assert all(x >= 0 for x in draft_hidden_gather_index)
            for position_id_offset in self.position_id_offset:
                assert all(x >= 0 for x in position_id_offset)

        except Exception as e:
            raise ValueError(f"Error initializing TokenTree: {str(e)}")

    def parse_tree_config(self, tree_config):
        """Parse tree config and complete the adjacency dict for simplified config file."""
        parsed_config = {}
        for k, v in tree_config.items():
            node_id = int(k)
            if v:  # If the node has children
                parsed_config[node_id] = [int(x) for x in v]
            else:  # If the node is a leaf (empty list) or not specified
                parsed_config[node_id] = []

        # Add implicit leaf nodes
        all_nodes = set(parsed_config.keys())
        for children in parsed_config.values():
            all_nodes.update(children)

        for node in all_nodes:
            if node not in parsed_config:
                parsed_config[node] = []

        return parsed_config

    def init_token_tree(self):
        """Initialize tree structure using BFS traversal."""
        try:
            if not self.tree_config:
                return

            q = deque([0])
            visited = set()
            level_nodes = {}
            current_level = 0

            # Initialize list to store node IDs for each level
            self.level_node_ids = []
            self.level_node_ids_with_child = []
            current_level_nodes = []

            while q:
                level_size = len(q)
                level_nodes[current_level] = level_size
                level_width = []
                current_level_nodes = []
                current_level_nodes_with_child = []

                # Process all nodes at current level
                for _ in range(level_size):
                    node_id = q.popleft()
                    visited.add(node_id)
                    current_level_nodes.append(node_id)

                    # Get children from tree_config
                    children = self.tree_config[node_id]
                    level_width.append(len(children))
                    if len(children) != 0:
                        current_level_nodes_with_child.append(node_id)

                    # Add unvisited children to queue
                    for child_id in children:
                        if child_id not in visited:
                            q.append(child_id)
                        else:
                            raise ValueError(f"Cycle detected at node {child_id}")
                self.level_child.append(level_width)

                # Update tree metrics for this level
                self.max_width = max(self.max_width, max(level_width))
                self.level_node_ids_with_child.append(current_level_nodes_with_child)
                self.level_node_ids.append(current_level_nodes)
                current_level += 1

            # Set final tree metrics
            self.level = [level_nodes[i] for i in range(current_level)]
            self.depth = current_level
            self.node_nums = len(visited)

            # Verify level_node_ids structure
            assert (
                len(self.level_node_ids) == self.depth
            ), "Mismatch between depth and level_node_ids"
            assert (
                sum(len(level) for level in self.level_node_ids) == self.node_nums
            ), "Mismatch in total nodes"

            # Generate attention mask
            self.generate_full_attention_mask()
            # self._generate_all_attention_masks()

        except Exception as e:
            raise ValueError(f"Error in tree initialization: {str(e)}")

    def generate_full_attention_mask(self):
        """Generate full attention mask for the tree using DFS.

        Returns:
            torch.Tensor: Attention mask of shape [max_node, max_node] where mask[i][j] = 1
                        means node i can attend to node j.
        """
        try:
            # Initialize mask and visited set
            self.full_tree_attn_mask = torch.zeros(
                self.node_nums, self.node_nums, dtype=torch.float32
            )
            visited = set()

            def dfs(node_id):
                """DFS helper function to build attention mask.

                Args:
                    node_id (int): Current node being processed
                """
                if node_id not in self.tree_config:
                    return
                visited.add(node_id)

                # Recursively process children
                for child_id in self.tree_config[node_id]:
                    if child_id not in visited:  # Prevent cycles
                        dfs(child_id)

                # Set attention mask: all visited nodes can attend to current node
                for id in visited:
                    self.full_tree_attn_mask[id][node_id] = 1

                # Remove current node from visited set (backtracking)
                visited.remove(node_id)

            dfs(0)

            draft_tree_attn_mask = []
            self.full_tree_attn_mask = self.full_tree_attn_mask.transpose(0, 1)

            for i in range(self.depth - 1):
                temp = torch.zeros((len(self.level_node_ids[i]), self.node_nums))
                temp[: len(self.level_node_ids[i]), :] = self.full_tree_attn_mask[
                    self.level_node_ids[i], :
                ]
                draft_tree_attn_mask.append(temp)

            for i in range(self.depth - 1):
                for node_id in self.level_node_ids[i]:
                    draft_tree_attn_mask[i][node_id - self.level_node_ids[i][0], node_id] = 0

            self.draft_tree_attn_mask = draft_tree_attn_mask

            return self.full_tree_attn_mask

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            raise

    def _validate_tree_structure(self):
        """Validate the tree structure after initialization."""

        # Get all nodes from tree_config (both parents and children)
        all_nodes = set()
        for parent, children in self.tree_config.items():
            all_nodes.add(parent)
            all_nodes.update(children)

        # Verify all nodes are consecutive starting from 0
        expected_nodes = set(range(self.node_nums))
        if expected_nodes != all_nodes:
            raise ValueError(
                f"Non-consecutive node IDs detected. "
                f"Missing nodes: {expected_nodes - all_nodes}"
            )

        # Verify sum of nodes at each level matches total nodes
        if sum(self.level) != self.node_nums:
            raise ValueError(f"Level sum ({sum(self.level)}) " f"!= total nodes ({self.node_nums})")

    def get_all_paths_tensor(self):
        """
        Find all paths from root to leaves and return as tensor

        Args:
            self.tree_config: dict where key is node idx and value is list of children indices

        Returns:
            torch.Tensor: tensor where each row represents a path from root to leaf
            Shape will be (num_leaves, max_path_length)
        """

        def is_leaf(node_idx):
            """Determine if the given node index represents a leaf node in the tree."""
            return node_idx not in self.tree_config or not self.tree_config[node_idx]

        def get_max_path_length(node_idx, memo={}):
            """Calculate the maximum path length from the given node to a leaf node."""

            if is_leaf(node_idx):
                return 1

            if node_idx in memo:
                return memo[node_idx]

            max_len = 0
            if node_idx in self.tree_config:
                for child in self.tree_config[node_idx]:
                    child_len = get_max_path_length(child, memo)
                    max_len = max(max_len, child_len)

            memo[node_idx] = max_len + 1
            return memo[node_idx]

        def count_leaves(node_idx):
            """Count the number of leaf nodes in the subtree rooted at the given node."""
            if is_leaf(node_idx):
                return 1

            count = 0
            if node_idx in self.tree_config:
                for child in self.tree_config[node_idx]:
                    count += count_leaves(child)
            return count

        def dfs(node_idx, current_path, paths):
            current_path.append(node_idx)

            # If leaf node, add padded path to paths
            if is_leaf(node_idx):

                # Pad path to max_path_length
                padded_path = current_path + [-1] * (max_path_length - len(current_path))
                paths.append(padded_path)

            # Recurse on children
            if node_idx in self.tree_config:
                for child in self.tree_config[node_idx]:
                    dfs(child, current_path.copy(), paths)

        # Get maximum path length and number of leaves
        max_path_length = get_max_path_length(0)

        # Collect all paths
        all_paths = []
        dfs(0, [], all_paths)

        scatter_idxs = []
        for path in all_paths:
            scatter_idxs.append(self.create_scatter_indices(path, self.node_nums))

        all_parent_paths = []
        all_parent_paths = copy.deepcopy(all_paths)

        paths_tensor = torch.tensor(all_paths, dtype=torch.int64)
        parent_paths_tensor = torch.tensor(all_parent_paths, dtype=torch.int64)
        return paths_tensor, parent_paths_tensor, scatter_idxs

    def create_scatter_indices(self, path, total_length):
        """
        Creates scatter indices where path indices maintain their relative positions
        and remaining positions are filled sequentially.

        Args:
            path: List of indices that should maintain their positions
            total_length: Total length of the output scatter indices

        Returns:
            scatter_indices: List where path indices are preserved and other positions
                            are filled sequentially with remaining numbers
        """
        scatter_indices = [-1] * total_length

        # First, place the path indices in their positions
        for i, pos in enumerate(path):
            scatter_indices[pos] = i

        # Create a counter for filling remaining positions
        next_value = len(path)

        # Fill remaining positions
        for i in range(total_length):
            if scatter_indices[i] == -1:
                scatter_indices[i] = next_value
                next_value += 1

        return scatter_indices

    def get_all_permute_mask(self, path):
        """
        Generate permutation masks for each path in the input.

        Args:
        path (torch.Tensor): Input tensor containing paths through the tree.

        Returns:
        tuple: (output_permute_masks, new_path)
            - output_permute_masks: Tensor containing permutation masks for each path.
            - new_path: Tensor containing paths with unused nodes appended.
        """
        output_permute_masks = torch.zeros(
            (path.shape[0], self.node_nums), dtype=torch.int64, device=path.device
        )
        new_path = torch.zeros_like(path)
        for idx in range(path.shape[0]):

            # Get current path
            current_path = path[idx]

            # Create a mask for current row
            current_mask = torch.zeros(self.node_nums, dtype=torch.int64, device=path.device)

            # Fill in nodes from path first
            current_path = torch.tensor([x for x in current_path if x != -1], dtype=path.dtype)

            path_length = len(current_path)
            current_mask[:path_length] = current_path

            # Find unused nodes
            used_nodes = set(current_path.cpu().tolist())
            all_nodes = set(range(self.node_nums))
            unused_nodes = list(all_nodes - used_nodes)

            # Fill remaining positions with unused nodes
            if len(unused_nodes) > 0:
                current_mask[path_length:] = torch.tensor(
                    unused_nodes, dtype=torch.int64, device=path.device
                )
            new_path[idx] = current_mask[: self.depth]
            output_permute_masks[idx] = current_mask

        return output_permute_masks, new_path

    def get_rotary_position_id(self):
        """
        Generate rotary position IDs based on the level of each node in the tree.

        Returns:
        torch.Tensor: Tensor containing rotary position IDs for each node.
        """
        rotary_position_id = []
        for i in range(len(self.level_node_ids)):
            for _ in range(len(self.level_node_ids[i])):
                rotary_position_id.append(i)

        return torch.tensor(rotary_position_id, dtype=torch.int64)

    def get_hidden_scatter_index(self):
        """
        Generate indices for scattering hidden states in the tree structure.

        Returns:
        torch.Tensor: Tensor containing scatter indices for each level and node.
        """
        output_scatter_indice = torch.zeros(1, self.depth - 1, self.width)
        for level in range(self.depth - 1):
            if level == 0:
                output_scatter_indice[0, level, :] = torch.zeros(self.width)
            else:
                level_scatter_indice = torch.zeros(self.width)
                for i, node_idx in enumerate(self.level_node_ids[level]):
                    for child_id in self.tree_config[node_idx]:
                        level_scatter_indice[child_id - self.level_node_ids[level + 1][0]] = i
                output_scatter_indice[0, level, :] = level_scatter_indice
        return output_scatter_indice.to(dtype=torch.int64)

    def get_draft_hidden_gather_index(self):
        """
        Generate indices for gathering hidden states in the draft process.

        Returns:
        list: List of lists containing gather indices for each level.
        """
        hidden_gather_indice = []

        for i in range(len(self.level_child) - 1):
            hidden_gather_index = []
            count = 0

            for j in self.level_child[i]:
                for _ in range(j):
                    hidden_gather_index.append(count)
                count += 1

            hidden_gather_indice.append(hidden_gather_index)

        return hidden_gather_indice

    def get_level_child_without_leaf(self):
        """
        Count the number of non-leaf children for each node at each level.

        Returns:
        list: List of lists containing counts of non-leaf children for each node at each level.
        """
        level_child_without_leaf = []
        for i, lst in enumerate(self.level_node_ids):
            temp = []
            for node in lst:
                count = 0
                for children in self.tree_config[node]:
                    if len(self.tree_config[children]) != 0:
                        count += 1
                temp.append(count)
            level_child_without_leaf.append(temp)
        return level_child_without_leaf

    def get_position_id_offset(self):
        """
        Generate position ID offsets for each level in the tree.

        Returns:
        list: List of lists containing position ID offsets for each node at each level.
        """
        position_id_offsets = []
        for i in range(self.depth - 1):
            position_id_offset = []

            for j in range(self.width):
                position_id_offset.append(j + self.drafted_nums[i])

            position_id_offsets.append(position_id_offset)

        return position_id_offsets

    def get_drafted_num(self):
        """
        Calculate the cumulative sum of nodes at each level of the tree.

        Returns:
        list: A list where each element represents the total number of nodes
            up to and including that level. The first element is always 0.
        """
        drafted_nums = [0]
        for i in range(len(self.level_node_ids)):
            drafted_nums.append(drafted_nums[-1] + len(self.level_node_ids[i]))
        return drafted_nums

    def get_level_node_count(self):
        """
        Count the number of nodes at each level of the tree.

        Returns:
        list: A list where each element represents the number of nodes at that level.
        """
        level_node_count = []
        for i in range(self.depth):
            level_node_count.append(len(self.level_node_ids[i]))
        return level_node_count

    def get_level_internal_node_count(self):
        """
        Count the number of internal nodes (nodes with children) at each level of the tree.

        Returns:
        list: A list where each element represents the number of internal nodes at that level.
        """
        internal_node_count = []
        for i in range(self.depth):

            internal_node_count.append(len(self.level_node_ids_with_child[i]))
        return internal_node_count

    def get_topk_permute_index(self):
        """
        Generate permutation indices for top-k selection at each level of the tree.

        This function creates a list of indices for each level, which can be used
        to select and arrange the top-k elements from each node's children.

        Returns:
        list: A list of lists, where each inner list contains permutation indices for a level.
        """
        permute_index = []

        for i in range(self.depth - 1):
            max_topk = max(self.level_child[i])
            permute_index_per_level = []
            count = 0

            for num_child in self.level_child[i]:
                for j in range(num_child):
                    permute_index_per_level.append(count + j)

                if num_child != 0:
                    count += max_topk

            permute_index.append(permute_index_per_level)

        return permute_index

    def debug_info(self, logger):
        """#print debug information about the token tree structure.

        Args:
            logger: logging.Logger object for output
        """
        logger.info(f"\n{'='*50}")
        logger.info("TOKEN TREE DEBUG INFORMATION")
        logger.info(f"{'='*50}")

        # Basic tree metrics
        logger.info("\n1. Basic Metrics:")
        logger.info(f"  - Tree Depth: {self.depth}")
        logger.info(f"  - Total Nodes: {self.node_nums}")
        logger.info(f"  - Maximum Width: {self.max_width}")

        # Level information
        logger.info("\n2. Level Information:")
        for level_idx in range(len(self.level)):
            logger.info(f"  Level {level_idx}:")
            logger.info(f"    - Nodes: {self.level[level_idx]}")

        # Tree configuration
        logger.info("\n3. Tree Configuration (Adjacency List):")
        for node, children in self.tree_config.items():
            logger.info(f"  Node {node} → Children: {children}")

        # Attention Masks
        logger.info("\n4. Attention Masks:")
        if self.full_tree_attn_mask is not None:
            logger.info("  Full Tree Attention Mask:")
            logger.info(f"    Shape: {self.full_tree_attn_mask.shape}")
            mask_preview = self.full_tree_attn_mask.cpu().numpy()
            logger.info("    Preview  :")
            for row in mask_preview:
                logger.info(f"    {row}")

        # Validate tree structure
        logger.info("\n5. Tree Validation:")
        try:
            self._validate_tree_structure(logger)
        except Exception as e:
            logger.error(f"Tree validation failed: {str(e)}")

        logger.info(f"\n{'='*50}\n")

        logger.info("\n6. Attention Masks:")

        # Full Tree Attention Mask
        if self.full_tree_attn_mask is not None:
            logger.info("\n  4.1 Full Tree Attention Mask:")
            logger.info(f"    Shape: {self.full_tree_attn_mask.shape}")
            logger.info("    Preview (5x5):")
            mask_preview = self.full_tree_attn_mask[:5, :5].cpu().numpy()
            for row in mask_preview:
                logger.info(f"    {row}")

        # Level Attention Masks
        if hasattr(self, "level_attention_masks"):
            logger.info("\n  4.2 Level Attention Masks:")
            logger.info(f"    Shape: {self.level_attention_masks.shape}")
            for level in range(min(3, self.depth)):  # Show first 3 levels
                logger.info(f"\n    Level {level} mask preview (3x3):")
                level_preview = self.level_attention_masks[level, :3, :3].cpu().numpy()
                for row in level_preview:
                    logger.info(f"    {row}")

        # Prior TT Masks
        if hasattr(self, "prior_tt_masks"):
            logger.info("\n  4.3 Prior Token Tree Masks:")
            logger.info(f"    Shape: {self.prior_tt_masks.shape}")
            for level in range(1, min(3, self.depth)):
                logger.info(f"\n    Level {level} prior mask preview (3x3):")
                prior_preview = self.prior_tt_masks[level, :3, :3].cpu().numpy()
                for row in prior_preview:
                    logger.info(f"    {row}")

        # Level Statistics
        logger.info("\n5. Level Statistics:")
        for level_idx in range(self.depth):
            logger.info(f"\n  Level {level_idx}:")
            logger.info(f"    - Nodes in level: {self.level[level_idx]}")
            if hasattr(self, "level_attention_masks"):
                active_connections = torch.sum(self.level_attention_masks[level_idx]).item()
                logger.info(f"    - Active attention self.tree_config: {active_connections}")
            if level_idx > 0 and hasattr(self, "prior_tt_masks"):
                prior_connections = torch.sum(self.prior_tt_masks[level_idx]).item()
                logger.info(f"    - Prior level self.tree_config: {prior_connections}")

        # Mask Generation Status
        logger.info("\n6. Mask Generation Status:")
        logger.info(
            f"  - Level attention masks generated: {hasattr(self, 'level_attention_masks')}"
        )
        logger.info(f"  - Prior TT masks generated: {hasattr(self, 'prior_tt_masks')}")

        logger.info(f"\n{'='*50}\n")
