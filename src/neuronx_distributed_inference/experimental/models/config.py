from dataclasses import dataclass
from typing import Optional


import torch


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    dtype: torch.dtype = torch.float32
    rms_norm_eps: int = 1e-05
    rope_theta: int = 500000.0
    pad_token: int = 0
    intermediate_size_mlp: Optional[int] = None
    attention_chunk_size: Optional[int] = None
    interleave_moe_layer_step: Optional[int] = None
    nope_layer_interval: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    num_local_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None


# LLama3 1B
# TODO: move to llama3 sub-dir
Llama3_2_1B = Config(
    vocab_size=128256,
    hidden_size=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    intermediate_size=8192,
    dtype=torch.bfloat16,
    rms_norm_eps=1e-05,
    rope_theta=500000.0,
    pad_token=128004,
)


# LLama4 Scout
# TODO: move to llama4 sub-dir
Llama4_Scout = Config(
    vocab_size=202048,
    hidden_size=5120,
    n_layers=48,
    n_heads=40,
    n_kv_heads=8,
    head_dim=128,
    intermediate_size=8192,
    intermediate_size_mlp=16384,
    dtype=torch.bfloat16,
    rms_norm_eps=1e-05,
    rope_theta=500000.0,
    pad_token=200018,
    attention_chunk_size=8192,
    interleave_moe_layer_step=1,
    nope_layer_interval=4,
    use_qk_norm=True,
    num_local_experts=16,
    num_experts_per_tok=1,
)
