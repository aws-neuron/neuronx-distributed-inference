# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# ==========================================================================
#
# Gemma-4-26B-A4B-it configuration shim.
#
# Round 2 simplification: the actual config classes
# (`Gemma4InferenceConfig`, `Gemma4NeuronConfig`) live in
# `modeling_gemma4_neuron.py` (PR #106 pattern). This file used to host
# factory functions that delayed NxDI imports for laptop-side parsing; that
# was removed because every consumer of these classes already imports from
# `modeling_gemma4_neuron` (which itself requires NxDI). Keeping the
# factories was dead code that just hid bugs.
#
# What stays here: a small HF-side compatibility shim
# (`Gemma4TextConfig`) that lets static tools parse this package without the
# upstream gemma4 transformers branch installed. It is unused at runtime
# when the real `transformers.models.gemma4` is importable.

from __future__ import annotations

from typing import Any, Optional


try:
    from transformers.models.gemma4.configuration_gemma4 import (  # type: ignore[import-not-found]
        Gemma4TextConfig as _HFGemma4TextConfig,
    )
except Exception:  # pragma: no cover - shim path
    _HFGemma4TextConfig = None  # noqa: N816


class _Gemma4TextConfigShim:
    """Mirror the fields of HF `Gemma4TextConfig` we need for static parsing."""

    model_type = "gemma4_text"

    # 26B-A4B values from the actual HF config.json (see traces/round2_diff.md
    # for the corrections vs round-1 guesses).
    vocab_size: int = 262_144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 262_144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = 0
    eos_token_id: Any = 1
    bos_token_id: Optional[int] = 2
    tie_word_embeddings: bool = True
    rope_parameters: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 1024
    layer_types: Optional[list] = None
    final_logit_softcapping: Optional[float] = 30.0
    use_bidirectional_attention: Optional[str] = None
    # 26B-A4B: PLE disabled (hidden_size_per_layer_input == 0).
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 0
    num_global_key_value_heads: Optional[int] = 2
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    num_kv_shared_layers: int = 0
    # 26B-A4B has MoE enabled.
    enable_moe_block: bool = True
    use_double_wide_mlp: bool = False
    num_experts: Optional[int] = 128
    top_k_experts: Optional[int] = 8
    moe_intermediate_size: Optional[int] = 704

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._post_init()

    def _post_init(self) -> None:
        if self.layer_types is None:
            # 5 sliding : 1 full pattern, last layer == full (per HF post-init).
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention" if (i + 1) % sliding_window_pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
            self.layer_types[-1] = "full_attention"

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }


Gemma4TextConfig = _HFGemma4TextConfig if _HFGemma4TextConfig is not None else _Gemma4TextConfigShim


__all__ = ["Gemma4TextConfig"]
