
import os
from dataclasses import dataclass
import logging

from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import init_cpu_env
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3RotaryEmbedding

torch.set_printoptions(precision=5)


logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)06d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class NumericalTolerances:
    rtol: float
    atol: float

# Default tolerances from torch.testing.assert_close
FP32_TOLERANCES = NumericalTolerances(rtol=1.3e-6, atol=1e-5)
FP16_TOLERANCES = NumericalTolerances(rtol=1e-3, atol=1e-5)
BF16_TOLERANCES = NumericalTolerances(rtol=1.6e-2, atol=1e-5)


def cpu_setup(dtype):
    set_random_seed(0)
    os.environ.setdefault("NXD_CPU_MODE", "1")
    init_cpu_env()
    torch.set_default_dtype(dtype)
    torch.set_default_device("cpu")


def mark_step() -> None:
    torch_xla.sync()
    xm.wait_device_ops()


def assert_tensor_all_close(
        test_objective: str,
        computed_value: torch.FloatTensor,
        reference_value: torch.FloatTensor,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = True,
        ) -> None:
    assert computed_value.dtype == reference_value.dtype, "dtypes are not matching"
    try:
        assert torch.allclose(computed_value, reference_value, rtol, atol, equal_nan), f"{test_objective} are not matching!"
        logger.info(f"{test_objective} ({reference_value.numel()} value(s)) are matching (atol={atol:.1e} - rtol={rtol:.1e})!")
    except AssertionError as e:
        logger.error(e)

        logger.info("------ TOTAL ERROR ANALYSIS ------")
        abs_difference = torch.abs(computed_value - reference_value)
        rel_difference = abs_difference / torch.abs(reference_value)
        threshold = atol + torch.abs(reference_value) * rtol
        mask = abs_difference > threshold
        num_non_matching_values, total_values = mask.sum().item(), mask.numel()
        percentage = (num_non_matching_values / total_values) * 100
        logger.info(f"{num_non_matching_values}/{total_values} value(s) ({percentage:.2f}%) are not within tolerances (atol={atol:.1e} - rtol={rtol:.1e})")
        logger.info(f"Reference values: {reference_value[mask]}")
        logger.info(f"Computed  values: {computed_value[mask]}")
        logger.info(f"Abs. diff.: {abs_difference[mask]}")
        logger.info(f"Threshold:  {threshold[mask]}")

        logger.info("------ ABSOLUTE ERROR ANALYSIS ------")
        logger.info(f"Absolute error tolerance (atol):  {atol:.1e}")
        atol_dominates = atol > 10.0 * torch.abs(reference_value) * rtol
        atol_dominated_values = atol_dominates.sum().item()
        if atol_dominated_values:
            percentage = (atol_dominated_values / total_values) * 100
            logger.info(f"Absolute error dominates (atol > 10*rtol) for {atol_dominated_values}/{total_values} value(s) ({percentage:.2f}%)")
            a_mask = (abs_difference > atol) & atol_dominates
            num_non_matching_values = a_mask.sum().item()
            percentage = (num_non_matching_values / total_values) * 100
            logger.info(f"{num_non_matching_values}/{total_values} value(s) ({percentage:.2f}%) are not within absolute tolerances (atol={atol:.1e})")
            logger.info(f"Mean abs. diff.: {abs_difference[a_mask].mean():.3e} - Max abs. diff.: {abs_difference[a_mask].max():.3e}")
            logger.info(f"Reference values: {reference_value[a_mask]}")
            logger.info(f"Computed  values: {computed_value[a_mask]}")
            logger.info(f"Abs. diff.: {abs_difference[a_mask]}")
        else:
            logger.info(f"There are no values (0/{total_values} value(s) - 0.00%) for which the absolute error dominates (atol > 10*rtol)")

        logger.info("------ RELATIVE ERROR ANALYSIS ------")
        logger.info(f"Relative error tolerance (rtol):  {rtol:.1e}")
        rtol_dominates = torch.abs(reference_value) * rtol > 10.0 * atol
        rtol_dominated_values = rtol_dominates.sum().item()
        if rtol_dominated_values:
            percentage = (rtol_dominated_values / total_values) * 100
            logger.info(f"Relative error dominates (rtol > 10*atol) for {rtol_dominated_values}/{total_values} value(s) ({percentage:.2f}%)")
            r_mask = (rel_difference > rtol) & rtol_dominates
            num_non_matching_values = r_mask.sum().item()
            percentage = (num_non_matching_values / total_values) * 100
            logger.info(f"{num_non_matching_values}/{total_values} value(s) ({percentage:.2f}%) are not within relative tolerances (rtol={rtol:.1e})")
            logger.info(f"Mean rel. diff.: {rel_difference[r_mask].mean():.3e} - Max rel. diff.: {rel_difference[r_mask].max():.3e}")
            logger.info(f"Reference values: {reference_value[r_mask]}")
            logger.info(f"Computed  values: {computed_value[r_mask]}")
            logger.info(f"Rel. diff.: {rel_difference[r_mask]}")
        else:
            logger.info(f"There are no values (0/{total_values} value(s) - 0.00%) for which the relative error dominates (rtol > 10*atol)")
        raise e


# This mock KV cache manager is used to test model on CPU as NxDI implementation of KV Cache Manager requires XLA tensors.
class MockKVCacheManager(KVCacheManager):
    def update_cache(
        self,
        is_for_context_encoding,
        seq_ids,
        position_ids,
        new_key_values,
        seq_len: int,
        scatter_index=None,
        active_mask=None,
        kvcache_buffer=None,
        **kwargs
    ):
        return new_key_values



def create_position_ids_for_context_processing(attention_mask_2d: torch.LongTensor) -> torch.LongTensor:
    position_ids = attention_mask_2d.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask_2d == 0, 1)
    return position_ids


def create_position_ids_for_token_generation(attention_mask_2d: torch.LongTensor) -> torch.LongTensor:
    full_position_ids = create_position_ids_for_context_processing(attention_mask_2d=attention_mask_2d)
    return torch.amax(full_position_ids, dim=1, keepdim=True) + 1


def create_position_ids(attention_mask_2d: torch.LongTensor, is_for_context_encoding: bool) -> torch.LongTensor:
    if is_for_context_encoding:
        return create_position_ids_for_context_processing(attention_mask_2d=attention_mask_2d)
    else: 
        return create_position_ids_for_token_generation(attention_mask_2d=attention_mask_2d)


def create_cache_position(attention_mask_2d: torch.LongTensor, is_for_context_encoding: bool) -> torch.LongTensor:
    # From tranformers.utils.GenerationMixin._get_initial_cache_position
    cache_position = torch.ones_like(attention_mask_2d[0, :], dtype=torch.int64).cumsum(0) - 1
    if is_for_context_encoding:
        return cache_position
    else:
        return cache_position[-1:]
    

def update_2d_attention_mask(attention_mask_2d: torch.LongTensor, padding_side: str) -> torch.LongTensor:
    batch_size, _ = attention_mask_2d.shape
    if padding_side == "left":
        attention_mask_2d = torch.cat([attention_mask_2d, attention_mask_2d.new_ones((batch_size, 1))], dim=1)
        #attention_mask_2d = attention_mask_2d[:, 1:]
    else:
        attention_mask_2d = torch.cat([attention_mask_2d.new_ones((batch_size, 1)), attention_mask_2d], dim=1)
    return attention_mask_2d


def create_rope(position_ids: torch.LongTensor, hf_config: PretrainedConfig) -> torch.FloatTensor:
    batch_size, sequence_length = position_ids.shape
    x = torch.randn(batch_size, hf_config.num_attention_heads, sequence_length, hf_config.head_dim).to(dtype=torch.float32)
    rope = Gemma3RotaryEmbedding(config=hf_config)
    cos, sin = rope(x, position_ids) 
    return cos, sin


def create_hidden_states(attention_mask_2d: torch.LongTensor, hf_config: PretrainedConfig, is_for_context_encoding: bool) -> torch.FloatTensor:
    batch_size, max_input_length = attention_mask_2d.shape
    sequence_length = max_input_length if is_for_context_encoding else 1
    return torch.randn(batch_size, sequence_length, hf_config.hidden_size, requires_grad=False).to(dtype=torch.float32)


def create_hf_attention_mask_4d(
        attention_mask_2d: torch.LongTensor,
        cache_position: torch.LongTensor,
        is_for_context_encoding: bool,
        is_swa_layer: bool,
        sliding_window_size: int,
        dtype: torch.dtype = torch.float32,
        ) -> torch.FloatTensor:
    batch_size, sequence_length = attention_mask_2d.shape
    target_length = sequence_length
    if not is_for_context_encoding:
        sequence_length = 1
    print("attention mask 2D")
    print(attention_mask_2d)
    attention_mask_4d = Gemma3TextModel._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=attention_mask_2d,
        sequence_length=sequence_length, # len_q
        target_length=target_length, # len_k
        dtype=dtype,
        device=attention_mask_2d.device,
        cache_position=cache_position,
        batch_size=batch_size,
    )
    # Adapted from transformers.models.cohere2.modeling_cohere2.Cohere2DecoderLayer.forward
    if not is_swa_layer:
        return attention_mask_4d
    else:
        print("attention mask 4D")
        print(attention_mask_4d[0])
        last_cache_position = cache_position[-1] + 1 # Current total seq length, fixed from HF
        effective_seq_len = max(cache_position.shape[0], sliding_window_size)
        min_dtype = torch.finfo(dtype).min
        sliding_window_mask = torch.tril(
            torch.ones_like(attention_mask_4d, dtype=torch.bool), diagonal=-sliding_window_size
        )
        attention_mask_4d = torch.where(sliding_window_mask, min_dtype, attention_mask_4d)
        offset = max(0, last_cache_position - effective_seq_len)
        return attention_mask_4d[:, :, :, offset : offset + effective_seq_len]


def left_to_right_padding(x: torch.FloatTensor, attention_mask_2d: torch.LongTensor) -> torch.FloatTensor:
    # x is a 4D tensor of shape (batch_size, num_kv_heads, seq_length, head_dim)
    # attention_mask_2d is a 2D tensor of shape (batch_size, seq_length)
    _, bucket_size = attention_mask_2d.shape
    seq_lengths = attention_mask_2d.sum(dim=1).view(-1, 1)
    max_seq_lengths = seq_lengths.max().item()
    offset = max_seq_lengths - seq_lengths
    roll_index = torch.remainder(torch.arange(0, bucket_size)[None, :] + offset, bucket_size)\
        .view(-1, 1, bucket_size, 1)\
        .expand_as(x)
    return torch.gather(x, dim=2, index=roll_index)


def apply_sliding_window(x: torch.FloatTensor,
                         position_ids: torch.LongTensor,
                         sliding_window_size: int,
                         padding_side: str) -> torch.FloatTensor:
    # x is a 4D tensor of shape (batch_size, num_kv_heads, seq_length, head_dim)
    # position_ids is a 2D tensor of shape (batch_size, seq_length)
    batch_size, num_kv_heads, _, head_dim = x.shape
    if padding_side == "left":
        max_position_ids = torch.max(position_ids)[None, None].expand(batch_size, -1)
    else:
        max_position_ids = torch.amax(position_ids, dim=1, keepdim=True)
    offset = torch.clamp(max_position_ids - sliding_window_size + 1, min=0)
    index = torch.arange(sliding_window_size)[None, :] + offset
    index = index[:, None, :, None].expand(-1, num_kv_heads, -1, head_dim)
    return torch.gather(x, dim=2, index=index)
