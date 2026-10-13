# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid prefix-boundary checkpoint cache primitives.

Attention KV can remain in the normal vLLM/NxDI block cache. GDN recurrent and
conv state should not be cached as ordinary per-block data; it is a checkpoint
for the cumulative prefix at a reusable boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Mapping


@dataclass(frozen=True)
class HybridPrefixCheckpointKey:
    cumulative_prefix_hash: Hashable
    prefix_len: int
    cache_salt: Hashable | None = None
    model_revision: Hashable | None = None
    layout_version: int = 1


@dataclass
class HybridPrefixCheckpoint:
    key: HybridPrefixCheckpointKey
    recurrent_states: Mapping[int, Any]
    conv_states: Mapping[int, Any]
    ref_count: int = 0
    last_access_tick: int = 0

    def has_all_required_state(self, required_gdn_layers: tuple[int, ...]) -> bool:
        return all(
            layer_id in self.recurrent_states and layer_id in self.conv_states
            for layer_id in required_gdn_layers
        )


@dataclass(frozen=True)
class HybridPrefixReusePlan:
    attention_hit_len: int
    gdn_checkpoint_hit_len: int
    restore_checkpoint_prefix_len: int
    residual_replay_len: int
    suffix_len: int
    checkpoint_key: HybridPrefixCheckpointKey | None


class HybridPrefixCheckpointCache:
    """LRU/refcount cache for cumulative-prefix GDN checkpoints."""

    def __init__(
        self,
        *,
        required_gdn_layers: list[int] | tuple[int, ...],
        checkpoint_interval: int,
        max_checkpoints: int | None = None,
        layout_version: int = 1,
    ):
        if not required_gdn_layers:
            raise ValueError("required_gdn_layers must not be empty")
        self.required_gdn_layers = tuple(int(layer) for layer in required_gdn_layers)
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {checkpoint_interval}"
            )
        self.max_checkpoints = max_checkpoints
        if self.max_checkpoints is not None and self.max_checkpoints <= 0:
            raise ValueError(f"max_checkpoints must be positive, got {max_checkpoints}")
        self.layout_version = int(layout_version)
        self._checkpoints: OrderedDict[
            HybridPrefixCheckpointKey, HybridPrefixCheckpoint
        ] = OrderedDict()
        self._tick = 0

    def __len__(self) -> int:
        return len(self._checkpoints)

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    def make_key(
        self,
        *,
        cumulative_prefix_hash: Hashable,
        prefix_len: int,
        cache_salt: Hashable | None = None,
        model_revision: Hashable | None = None,
        layout_version: int | None = None,
    ) -> HybridPrefixCheckpointKey:
        prefix_len = int(prefix_len)
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        if prefix_len % self.checkpoint_interval != 0:
            raise ValueError(
                "prefix_len must align to checkpoint_interval "
                f"{self.checkpoint_interval}, got {prefix_len}"
            )
        return HybridPrefixCheckpointKey(
            cumulative_prefix_hash=cumulative_prefix_hash,
            prefix_len=prefix_len,
            cache_salt=cache_salt,
            model_revision=model_revision,
            layout_version=self.layout_version
            if layout_version is None
            else int(layout_version),
        )

    def put_checkpoint(
        self,
        *,
        cumulative_prefix_hash: Hashable,
        prefix_len: int,
        recurrent_states: Mapping[int, Any],
        conv_states: Mapping[int, Any],
        cache_salt: Hashable | None = None,
        model_revision: Hashable | None = None,
        layout_version: int | None = None,
    ) -> HybridPrefixCheckpointKey:
        key = self.make_key(
            cumulative_prefix_hash=cumulative_prefix_hash,
            prefix_len=prefix_len,
            cache_salt=cache_salt,
            model_revision=model_revision,
            layout_version=layout_version,
        )
        checkpoint = HybridPrefixCheckpoint(
            key=key,
            recurrent_states=dict(recurrent_states),
            conv_states=dict(conv_states),
            last_access_tick=self._next_tick(),
        )
        if not checkpoint.has_all_required_state(self.required_gdn_layers):
            raise ValueError(
                "checkpoint must include recurrent and conv state for every "
                f"required GDN layer: {self.required_gdn_layers}"
            )
        self._checkpoints[key] = checkpoint
        self._checkpoints.move_to_end(key)
        self.evict_to_capacity()
        return key

    def get_checkpoint(
        self,
        key: HybridPrefixCheckpointKey,
    ) -> HybridPrefixCheckpoint | None:
        checkpoint = self._checkpoints.get(key)
        if checkpoint is None:
            return None
        checkpoint.last_access_tick = self._next_tick()
        self._checkpoints.move_to_end(key)
        return checkpoint

    def inc_ref(self, key: HybridPrefixCheckpointKey) -> int:
        checkpoint = self.get_checkpoint(key)
        if checkpoint is None:
            raise KeyError(key)
        checkpoint.ref_count += 1
        return checkpoint.ref_count

    def dec_ref(self, key: HybridPrefixCheckpointKey) -> int:
        checkpoint = self.get_checkpoint(key)
        if checkpoint is None:
            raise KeyError(key)
        checkpoint.ref_count = max(0, checkpoint.ref_count - 1)
        return checkpoint.ref_count

    def evict_to_capacity(self) -> list[HybridPrefixCheckpointKey]:
        if self.max_checkpoints is None:
            return []
        evicted: list[HybridPrefixCheckpointKey] = []
        for key, checkpoint in list(self._checkpoints.items()):
            if len(self._checkpoints) <= self.max_checkpoints:
                break
            if checkpoint.ref_count > 0:
                continue
            del self._checkpoints[key]
            evicted.append(key)
        return evicted

    def compute_reuse_plan(
        self,
        *,
        cumulative_hashes_by_prefix_len: Mapping[int, Hashable],
        attention_hit_len: int,
        request_prefix_len: int,
        cache_salt: Hashable | None = None,
        model_revision: Hashable | None = None,
        layout_version: int | None = None,
    ) -> HybridPrefixReusePlan:
        attention_hit_len = max(0, int(attention_hit_len))
        request_prefix_len = max(0, int(request_prefix_len))
        target_suffix_start = min(attention_hit_len, request_prefix_len)

        candidate_prefix_lens = sorted(
            (
                int(prefix_len)
                for prefix_len in cumulative_hashes_by_prefix_len
                if int(prefix_len) <= target_suffix_start
                and int(prefix_len) % self.checkpoint_interval == 0
            ),
            reverse=True,
        )
        for prefix_len in candidate_prefix_lens:
            key = self.make_key(
                cumulative_prefix_hash=cumulative_hashes_by_prefix_len[prefix_len],
                prefix_len=prefix_len,
                cache_salt=cache_salt,
                model_revision=model_revision,
                layout_version=layout_version,
            )
            checkpoint = self.get_checkpoint(key)
            if checkpoint is None:
                continue
            if not checkpoint.has_all_required_state(self.required_gdn_layers):
                continue
            return HybridPrefixReusePlan(
                attention_hit_len=attention_hit_len,
                gdn_checkpoint_hit_len=prefix_len,
                restore_checkpoint_prefix_len=prefix_len,
                residual_replay_len=target_suffix_start - prefix_len,
                suffix_len=request_prefix_len - target_suffix_start,
                checkpoint_key=key,
            )

        return HybridPrefixReusePlan(
            attention_hit_len=attention_hit_len,
            gdn_checkpoint_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=target_suffix_start,
            suffix_len=request_prefix_len - target_suffix_start,
            checkpoint_key=None,
        )
