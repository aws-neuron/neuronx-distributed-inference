import copy
import logging
import os
import pytest

import torch
from torch_neuronx.testing import neuron_allclose

from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_process_groups import tp_mesh_8_by_8
from neuronx_distributed_inference.modules.attention.gqa import replicate_kv
from neuronx_distributed_inference.utils.distributed import split_along_dim
from neuronx_distributed_inference.utils.testing import build_module

from neuronx_distributed_inference.models.gpt_oss.modeling_gpt_oss import GptOssNeuronConfig, NeuronGptOssAttention
from neuronx_distributed_inference.modules.kvcache.gpt_oss_kv_cache_manager import GptOssKVCacheManager


CP16_TO_TP64_HEAD_MAPPING = (0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6,
                             0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7,
                             1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7, 1, 3, 5, 7)

CP16_TO_TP8DP8_HEAD_MAPPING = (0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7,
                               0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7,
                               0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)

DEFAULT_CONFIG = {
    "experts_per_token": 4,
    "head_dim": 64,
    "hidden_size": 3072,
    "initial_context_length": 4096,
    "intermediate_size": 3072,
    "model_type": "gpt_oss",
    "num_attention_heads": 64,
    "num_experts": 128,
    "num_hidden_layers": 2,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 32.0,
    "rope_scaling_factor": 32.0,
    "rope_theta": 150000.0,
    "vocab_size": 201088,
    "attention_bias": True,
    "rms_norm_eps": 1e-05,
    "use_polar_compatible_rope": True,
}

BATCH_SIZE = 8
PROMPT_LEN = 50
SEQ_LEN = 256
SLIDING_WINDOW = 128
TKG_SEQ_LEN = 1
TORCH_DTYPE = torch.float32
TP_DEGREE = 64

CKPT_DIR = "/tmp/nxdi_test/test_attn_module_tkg/"
os.makedirs(CKPT_DIR, exist_ok=True)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_simple_attn_mask(seq_len, position_ids):
    pos = position_ids[:, 0]
    idx = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
    base_mask = idx < pos.unsqueeze(1)
    return base_mask[:, None, None, :]


# ModelBase._create_windowed_attn_mask_tkg()
def create_windowed_attn_mask_tkg(
        attention_mask, window_size, position_ids
    ):
        # Create tkg mask for sliding window. E.g.:
        # position = 3, window_size = 4 -> mask = [1,1,1,0]
        # position = 5, window_size = 4 -> mask = [1,0,1,1]
        batch_size, _ = attention_mask.shape
        pos = position_ids[:, 0]
        idx = torch.arange(window_size, device=attention_mask.device).unsqueeze(0)
        base_mask = idx < pos.unsqueeze(1)  # for input_len <= window_size

        full_mask = torch.ones((batch_size, window_size), dtype=torch.bool, device=attention_mask.device)
        zero_pos = pos % window_size
        zero_mask = idx == zero_pos.unsqueeze(1)
        full_mask = torch.where(zero_mask, False, full_mask)  # for input_len > window_size

        seq_less_than_window = pos < window_size
        final_mask = torch.where(seq_less_than_window.unsqueeze(1), base_mask, full_mask)
        return final_mask[:, None, None, :]


def generate_prefill_kv_cache_from_config(config, prompt_len):
    temp_kv_mgr = GptOssKVCacheManager(config, 
                                      num_kv_head=config.num_key_value_heads, 
                                      sliding_window=config.sliding_window)
    key_values = generate_prefill_kv_cache_from_shapes(k_shapes=temp_kv_mgr.k_shapes,
                                                       v_shapes=temp_kv_mgr.v_shapes,
                                                       prompt_len=prompt_len)
    return key_values


def generate_prefill_kv_cache_from_shapes(k_shapes, v_shapes, prompt_len):
    kv_cache = []
    for k_shape, v_shape in zip(k_shapes, v_shapes):
        k = torch.rand(k_shape)
        k = k[:, :, :prompt_len, :]
        v = torch.rand(v_shape)
        v = v[:, :, :prompt_len, :]
        kv_cache.append((k, v))
    return kv_cache    


def get_tkg_config(tp_degree,
                   cp_degree,
                   sp_enabled=False,
                   seq_len=SEQ_LEN,
                   torch_dtype=TORCH_DTYPE,
                   batch_size=1,
                   attention_dp_degree=None,
                   tkg_kernels_enabled=False,
                   **update_kwargs): 
    neuron_config = GptOssNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_batch_size=batch_size,
        max_context_length=seq_len,
        seq_len=seq_len,
        max_length=seq_len,
        torch_dtype=torch_dtype,
        cp_degree=cp_degree,
        dp_degree=cp_degree,
        sequence_parallel_enabled=sp_enabled,
        logical_neuron_cores=2,
        attn_kernel_enabled=False,  # Disable kernel to use native compiler prefill attention
        fused_qkv=True,
        qkv_kernel_enabled=tkg_kernels_enabled,
        attn_block_tkg_nki_kernel_enabled=tkg_kernels_enabled,
        attn_block_tkg_nki_kernel_cascaded_attention=tkg_kernels_enabled,
    )
    neuron_config.is_prefill_stage = False
    if attention_dp_degree is not None:
        neuron_config.attention_dp_degree = attention_dp_degree

    config_dict = copy.deepcopy(DEFAULT_CONFIG)
    config_dict.update(**update_kwargs)
    config = InferenceConfig(
        neuron_config, **config_dict
    )
    config.torch_dtype = torch_dtype

    return config


def form_ranked_kv_buffer_tp64cp1dp1(buffer):
    buffer_repl, _ = replicate_kv(buffer, source_heads=8, repeats=8, head_dim=1)
    b, _, s, h = buffer.shape
    buffer_ranked = torch.zeros((TP_DEGREE, b, 1, s, h), dtype=buffer.dtype, device=buffer.device)
    for rank in range(TP_DEGREE):
        buffer_single_rank = split_along_dim(buffer_repl, dim=1, rank=rank, num_partitions=64)
        buffer_ranked[rank] = buffer_single_rank.contiguous()
    return buffer_ranked


def form_ranked_kv_buffer_tp64cp16dp1(buffer):
    b, _, s, h = buffer.shape
    buffer_ranked = torch.zeros((TP_DEGREE, b, 1, s, h), dtype=buffer.dtype, device=buffer.device)
    for rank in range(TP_DEGREE):
        head_rank = CP16_TO_TP64_HEAD_MAPPING[rank]
        buffer_single_rank = split_along_dim(buffer, dim=1, rank=head_rank, num_partitions=8)
        buffer_ranked[rank] = buffer_single_rank.contiguous()
    return buffer_ranked


def form_ranked_kv_buffer_tp8cp8dp8(buffer, mesh):
    dp_degree, tp_degree = 8, 8
    b, _, s, h = buffer.shape
    buffer_ranked = torch.zeros((TP_DEGREE, (b // dp_degree) + 1, 1, s, h), dtype=buffer.dtype, device=buffer.device)

    for i in range(len(mesh)):
        for j in range(len(mesh[0])):
            # Handle DP degree - Split along batch dim
            buffer_dp = split_along_dim(buffer, dim=0, rank=i, num_partitions=dp_degree)
            bsz_dp = buffer_dp.shape[0]

            # Handle TP degree - Split along head dim
            buffer_tp = split_along_dim(buffer_dp, dim=1, rank=j, num_partitions=tp_degree)

            output_rank = mesh[i][j]
            buffer_ranked[output_rank][:bsz_dp] = buffer_tp 

    return buffer_ranked


def form_ranked_kv_buffer_tp8cp16dp8(buffer, mesh):
    dp_degree, tp_degree = 8, 8
    b, _, s, h = buffer.shape
    buffer_ranked = torch.zeros((TP_DEGREE, (b // dp_degree) + 1, 1, s, h), dtype=buffer.dtype, device=buffer.device)

    for i in range(len(mesh)):
        for j in range(len(mesh[0])):
            
            # Handle DP degree - Split along batch dim
            buffer_dp = split_along_dim(buffer, dim=0, rank=i, num_partitions=dp_degree)
            bsz_dp = buffer_dp.shape[0]

            # Handle TP degree - Split along head dim
            input_rank = len(mesh) * i + j
            output_rank = mesh[i][j]
            head_rank = CP16_TO_TP8DP8_HEAD_MAPPING[input_rank]
            buffer_tp = split_along_dim(buffer_dp, dim=1, rank=head_rank, num_partitions=tp_degree)

            buffer_ranked[output_rank][:bsz_dp] = buffer_tp 

    return buffer_ranked


def form_ranked_buffer_tp64_helper(past_key_values, form_ranked_kv_buffer_fn):
    ranked_past_key_values = []
    for i in range(len(past_key_values)):
        buffer = past_key_values[i]
        buffer_ranked = form_ranked_kv_buffer_fn(buffer)
        ranked_past_key_values.append(buffer_ranked)
    return ranked_past_key_values


def form_ranked_buffer_tp64cp1dp1(past_key_values):
    logger.info("form_ranked_buffer_tp64cp1dp1")
    ranked_past_key_values = form_ranked_buffer_tp64_helper(past_key_values, 
                                                            form_ranked_kv_buffer_tp64cp1dp1)
    return ranked_past_key_values


def form_ranked_buffer_tp64cp16dp1(past_key_values):
    logger.info("form_ranked_buffer_tp64cp16dp1")
    ranked_past_key_values = form_ranked_buffer_tp64_helper(past_key_values, 
                                                            form_ranked_kv_buffer_tp64cp16dp1)
    return ranked_past_key_values


def form_ranked_buffer_tp8dp8_helper(past_key_values, form_ranked_kv_buffer_fn):
    mesh = tp_mesh_8_by_8()

    ranked_past_key_values = []
    for i in range(len(past_key_values)):
        buffer = past_key_values[i]
        is_swa_layer = (((i // 2) % 2) == 0)
        if is_swa_layer:
            buffer_ranked = form_ranked_kv_buffer_tp64cp1dp1(buffer)
        else:
            buffer_ranked = form_ranked_kv_buffer_fn(buffer, mesh)

        ranked_past_key_values.append(buffer_ranked)

    return ranked_past_key_values


def form_ranked_buffer_tp8cp8dp8(past_key_values):
    logger.info("form_ranked_buffer_tp8cp8dp8")
    ranked_past_key_values = form_ranked_buffer_tp8dp8_helper(past_key_values,
                                                              form_ranked_kv_buffer_tp8cp8dp8)
    return ranked_past_key_values


def form_ranked_buffer_tp8cp16dp8(past_key_values):
    logger.info("form_ranked_buffer_tp8cp16dp8")
    ranked_past_key_values = form_ranked_buffer_tp8dp8_helper(past_key_values,
                                                              form_ranked_kv_buffer_tp8cp16dp8)
    return ranked_past_key_values


def form_ranked_buffer(config, past_key_values):
    # Generate ranked KV cache buffers
    cp_degree = config.neuron_config.cp_degree
    dp_degree = config.neuron_config.attention_dp_degree

    if cp_degree == 8 and dp_degree == 8:
        ranked_prev_key_values = form_ranked_buffer_tp8cp8dp8(past_key_values)
    elif cp_degree == 16 and dp_degree == 8:
        ranked_prev_key_values = form_ranked_buffer_tp8cp16dp8(past_key_values)
    elif cp_degree == 16 and dp_degree == 1:
        ranked_prev_key_values = form_ranked_buffer_tp64cp16dp1(past_key_values)
    elif cp_degree == 1 and dp_degree == 1:
        ranked_prev_key_values = form_ranked_buffer_tp64cp1dp1(past_key_values)
    else:
        raise NotImplementedError(f"KV cache ranking not implemented for " \
                                  f"TP {config.neuron_config.tp_degree} " \
                                  f"DP {config.neuron_config.attention_dp_degree} " \
                                  f"CP {config.neuron_config.cp_degree}")

    return ranked_prev_key_values


def construct_device_state_dict_from_cpu_model(cpu_model, config, torch_dtype):
    state_dict = cpu_model.state_dict()
    state_dict["attention.rank_util.rank"] = torch.arange(0, TP_DEGREE, dtype=torch.int32)

    state_dict["attention.tkg_learned_sinks.sink"] = state_dict["attention.learned_sinks.sink"].clone().contiguous()
    for prefix in ["cte", "tkg"]:
        for suffix in ["weight", "bias"]:
            state_dict[f"attention.{prefix}_qkv_proj.Wqkv.{suffix}"] = state_dict[f"attention.qkv_proj.Wqkv.{suffix}"].clone().contiguous()
            state_dict[f"attention.{prefix}_o_proj.o_proj.{suffix}"] = state_dict[f"attention.o_proj.o_proj.{suffix}"].clone().contiguous()

    # Save correct shape of KV cache weights for local device rank
    if config.neuron_config.attention_dp_degree == 8:
        for i in range(len(cpu_model.kv_mgr.past_key_values)):
            b, _, s, d = state_dict[f"kv_mgr.past_key_values.{i}"].shape
            del state_dict[f"kv_mgr.past_key_values.{i}"]
            dp_cache_bs = (b // config.neuron_config.attention_dp_degree) + 1
            if (i // 2) % 2 == 0:  # Non-SWA layer
                state_dict[f"kv_mgr.past_key_values.{i}"] = torch.zeros((b, 1, s, d), dtype=torch_dtype)
            else:  # SWA layer
                state_dict[f"kv_mgr.past_key_values.{i}"] = torch.zeros((dp_cache_bs, 1, s, d), dtype=torch_dtype)
    elif config.neuron_config.tp_degree == 64:
        for i in range(len(cpu_model.kv_mgr.past_key_values)):
            b, _, s, d = state_dict[f"kv_mgr.past_key_values.{i}"].shape
            del state_dict[f"kv_mgr.past_key_values.{i}"]
            state_dict[f"kv_mgr.past_key_values.{i}"] = torch.zeros((b, 1, s, d), dtype=torch_dtype)
    else:
        raise NotImplementedError("Invalid configuration for KV cache checkpoint.")

    return state_dict


class AttentionBlockWithKVCache(torch.nn.Module):
    def __init__(self, config: InferenceConfig, prompt_len: int, layer_index: int, sliding_window: int):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.is_cpu = (self.config.neuron_config.tp_degree == 1)
        self.prompt_len = prompt_len
        self.layer_index = layer_index
        self.sliding_window = sliding_window
        self.attention = NeuronGptOssAttention(self.config)
        self.rank_util = None if self.is_cpu else SPMDRank(TP_DEGREE)

        self.kv_mgr = GptOssKVCacheManager(config, 
                                          num_kv_head=config.num_key_value_heads, 
                                          global_rank=self.rank_util,
                                          sliding_window=self.sliding_window)

        logger.info("AttentionBlockWithKVCache Parameters")
        logger.info(f"Layer Index: {layer_index} SWA enabled: {config.sliding_window is not None}")
        logger.info(f"TP degree: {self.attention.tp_degree}")
        logger.info(f"CP degree: {self.attention.cp_degree}")
        logger.info(f"DP degree: {self.attention.dp_degree}")
        logger.info(f"Learned sink size: {self.attention.learned_sinks_size}")
        logger.info(f"Biases enabled: QKV: {self.attention.qkv_bias} | O: {self.attention.o_bias}")
        logger.info(f"Kernels enabled: {self.attention.attn_block_tkg_nki_kernel_enabled}")
        logger.info(f"Fused QKV: {self.attention.fused_qkv}")
        logger.info(f"Torch dtype: {self.attention.torch_dtype}")

    def load_kvcache_cpu(self, kvcache_buffers):
        assert len(kvcache_buffers) == len(self.kv_mgr.past_key_values)

        for i in range(len(self.kv_mgr.past_key_values)):
            self.kv_mgr.past_key_values[i].copy_(kvcache_buffers[i])

    def forward(self, hidden_states, attention_mask, position_ids):
        past_k, past_v = self.kv_mgr.get_kv_by_layer_id(idx=self.layer_index, 
                                                        seq_len=self.kv_mgr.v_shapes[self.layer_index][2])

        attn_out = self.attention(hidden_states=hidden_states,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_value=(past_k, past_v),
                                  kv_mgr=self.kv_mgr,
                                  idx=self.layer_index,
                                  kvcache_buffer=self.kv_mgr.past_key_values,
                                  active_block_table=None)

        return attn_out.hidden_states


@pytest.mark.parametrize(
    "layer_idx, cp_degree, dp_degree, kernels_enabled, torch_dtype",
    [
        (0, 1, 1, False, torch.float32),  # SWA layer, TP64 CP1 DP1 - Float32 - Kernels disabled
        (1, 1, 1, False, torch.float32),  # Causal layer, TP64 CP1 DP1 - Float32 - Kernels disabled
        (1, 16, 1, False, torch.float32),  # Causal layer, TP64 CP16 DP1 - Float32 - Kernels disabled
        (1, 8, 8, False, torch.float32),  # Causal layer, TP8 CP8 DP8 - Float32 - Kernels disabled
        (1, 16, 8, False, torch.float32),  # Causal layer, TP8 CP16 DP8 - Float32 - Kernels disabled
        (0, 1, 1, False, torch.bfloat16),  # SWA layer, TP64 CP1 DP1 - Bfloat16 - Kernels disabled
        (1, 1, 1, False, torch.bfloat16),  # Causal layer, TP64 CP1 DP1 - Bfloat16 - Kernels disabled
        (1, 16, 1, False, torch.bfloat16),  # Causal layer, TP64 CP16 DP1 - Bfloat16 - Kernels disabled
        (1, 8, 8, False, torch.bfloat16),  # Causal layer, TP8 CP8 DP8 - Bfloat16 - Kernels disabled
        (1, 16, 8, False, torch.bfloat16),  # Causal layer, TP8 CP16 DP8 - Bfloat16 - Kernels disabled

        # (0, 1, 1, True, torch.float32),  # SWA layer, TP64 CP1 DP1 - Float32 - Kernels enabled
        # (1, 1, 1, True, torch.float32),  # Causal layer, TP64 CP1 DP1 - Float32 - Kernels enabled
        # (1, 16, 1, True, torch.float32),  # Causal layer, TP64 CP16 DP1 - Float32 - Kernels enabled
        # (1, 8, 8, True, torch.float32),  # Causal layer, TP8 CP8 DP8 - Float32 - Kernels enabled
        # (1, 16, 8, True, torch.float32),  # Causal layer, TP8 CP16 DP8 - Float32 - Kernels enabled
        (0, 1, 1, True, torch.bfloat16),  # SWA layer, TP64 CP1 DP1 - Bfloat16 - Kernels enabled
        (1, 1, 1, True, torch.bfloat16),  # Causal layer, TP64 CP1 DP1 - Bfloat16 - Kernels enabled
        (1, 16, 1, True, torch.bfloat16),  # Causal layer, TP64 CP16 DP1 - Bfloat16 - Kernels enabled
        (1, 8, 8, True, torch.bfloat16),  # Causal layer, TP8 CP8 DP8 - Bfloat16 - Kernels enabled
        (1, 16, 8, True, torch.bfloat16),  # Causal layer, TP8 CP16 DP8 - Bfloat16 - Kernels enabled
    ]
)
def test_attn_module_tkg(layer_idx, cp_degree, dp_degree, kernels_enabled, torch_dtype):
    # Fixed test parameters
    batch_size = BATCH_SIZE
    tkg_seq_len = TKG_SEQ_LEN
    prompt_len = PROMPT_LEN
    seq_len = SEQ_LEN

    is_swa_layer = ((layer_idx % 2) == 0)

    cpu_config = get_tkg_config(tp_degree=1,
                                cp_degree=1,
                                sp_enabled=False,
                                sliding_window=SLIDING_WINDOW,
                                torch_dtype=torch_dtype,
                                batch_size=batch_size)

    device_config = get_tkg_config(tp_degree=TP_DEGREE,
                                   cp_degree=cp_degree,
                                   sp_enabled=False,
                                   sliding_window=SLIDING_WINDOW,
                                   torch_dtype=torch_dtype,
                                   batch_size=batch_size,
                                   attention_dp_degree=dp_degree,
                                   tkg_kernels_enabled=kernels_enabled)

    # Apply layer-specific checks and overrides
    if is_swa_layer:  # SWA enabled
        assert device_config.neuron_config.attention_dp_degree == 1
        assert device_config.neuron_config.cp_degree == 1
    else:  # SWA disabled
        device_config.sliding_window = None

    # Create test inputs
    torch.manual_seed(0)
    hidden_states = torch.rand((batch_size, tkg_seq_len, cpu_config.hidden_size), dtype=torch_dtype)
    position_ids = torch.ones((batch_size, 1), dtype=torch.int32) * prompt_len
    
    if is_swa_layer:
        orig_attn_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        attention_mask = create_windowed_attn_mask_tkg(orig_attn_mask,
                                                        window_size=SLIDING_WINDOW,
                                                        position_ids=position_ids)
    else:
        attention_mask = create_simple_attn_mask(seq_len, position_ids)

    # Create CPU KV cache
    prefill_kv_cache = generate_prefill_kv_cache_from_config(cpu_config, prompt_len)
    kv_mgr = GptOssKVCacheManager(cpu_config, 
                                 num_kv_head=cpu_config.num_key_value_heads, 
                                 sliding_window=cpu_config.sliding_window)
    os.environ["NXD_CPU_MODE"] = "1"
    kv_mgr.update_cache(is_for_context_encoding=True,
                        seq_ids=torch.arange(batch_size, dtype=torch.int32).unsqueeze(dim=1),
                        position_ids=position_ids,
                        new_key_values=prefill_kv_cache,
                        seq_len=seq_len,
                        scatter_index=None,
                        active_mask=None)
    os.environ["NXD_CPU_MODE"] = "0"

    # Build CPU model
    cpu_model = AttentionBlockWithKVCache(cpu_config, prompt_len, layer_idx, SLIDING_WINDOW)
    cpu_model.to(torch_dtype)
    cpu_model.load_kvcache_cpu(kv_mgr.past_key_values)

    # Construct and save local checkpoint
    state_dict = construct_device_state_dict_from_cpu_model(cpu_model, device_config, torch_dtype)
    checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    torch.save(state_dict, checkpoint_path)

    # Build device model
    example_inputs = [(
        torch.ones((batch_size, tkg_seq_len, device_config.hidden_size), dtype=torch_dtype),
        torch.ones_like(attention_mask),
        torch.ones_like(position_ids)
    )]
    torch.manual_seed(0)
    module_neuron = build_module(
        AttentionBlockWithKVCache,
        example_inputs,
        tp_degree=TP_DEGREE,
        module_init_kwargs={"config": device_config,
                            "prompt_len": prompt_len,
                            "layer_index": layer_idx,
                            "sliding_window": SLIDING_WINDOW},
        checkpoint_path=checkpoint_path,
        compiler_args="--auto-cast=none",
        logical_nc_config=2,
        priority_model_idx=None,
    )

    # Generate ranked KV cache buffers
    ranked_prev_key_values = form_ranked_buffer(device_config, kv_mgr.past_key_values)

    # Load ranked kv cache to device model HBM 
    for idx in range(len(kv_mgr.past_key_values)):
        for rank in range(TP_DEGREE):
            past_key_value = ranked_prev_key_values[idx][rank]
            module_neuron.nxd_model.weights[rank][f"kv_mgr.past_key_values.{idx}"].copy_(past_key_value)

    # Conduct test
    hidden_cpu = cpu_model(hidden_states=hidden_states,
                           attention_mask=attention_mask,
                           position_ids=position_ids)

    hidden_device = module_neuron(hidden_states,
                                  attention_mask,
                                  position_ids)

    hidden_allclose = neuron_allclose(
        hidden_device,
        hidden_cpu,
        rtol = 1.6e-2 if kernels_enabled else None,
        atol = 5e-3 if kernels_enabled else None)
    logger.info(f"Allclose report: {hidden_allclose} {hidden_cpu.shape}")
    assert hidden_allclose.allclose


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--no-cov"])
