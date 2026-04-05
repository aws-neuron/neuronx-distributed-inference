
import os
from dataclasses import dataclass
import logging

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import init_cpu_env
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from transformers import Gemma3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

from gemma3_vision.modeling_gemma3 import Gemma3InferenceConfig

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


def create_neuron_config(
    batch_size: int,
    max_seq_len: int,
    tp_degree: int,
    torch_dtype: torch.dtype,
    hf_config: Gemma3Config
    ) -> Gemma3InferenceConfig:
    return Gemma3InferenceConfig(
        text_neuron_config=NeuronConfig(
            tp_degree=tp_degree,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            attn_kernel_enabled=False,
            seq_len=max_seq_len
        ),
        vision_neuron_config=NeuronConfig(
            tp_degree=tp_degree,
            batch_size=1,
            torch_dtype=torch_dtype,
            attn_kernel_enabled=False,
            seq_len=max_seq_len
        ),
        load_config=load_pretrained_config(hf_config=hf_config),
    )


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


def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


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

    attention_mask_4d = _prepare_4d_causal_attention_mask_with_cache_position(
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
        last_cache_position = cache_position[-1] + 1 # Current total seq length, fixed from HF
        effective_seq_len = max(cache_position.shape[0], sliding_window_size)
        min_dtype = torch.finfo(dtype).min
        sliding_window_mask = torch.tril(
            torch.ones_like(attention_mask_4d, dtype=torch.bool), diagonal=-sliding_window_size
        )
        attention_mask_4d = torch.where(sliding_window_mask, min_dtype, attention_mask_4d)
        offset = max(0, last_cache_position - effective_seq_len)
        return attention_mask_4d[:, :, :, offset : offset + effective_seq_len]


def causal_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask


def window_mask(batch_size: int, seq_len: int, window_size: int):
    """create a causal, window attention mask"""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)
    for i in range(seq_len):
        if i >= window_size:
            mask[i, : i - window_size + 1] = False
    mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    return mask
