from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

from gemma3_vision.modeling_gemma3 import Gemma3InferenceConfig


def get_test_name_suffix(
    tp_degree: int,
    torch_dtype: torch.dtype,
    batch_size: int,
    num_images_per_sample: int,
    max_seq_len: int,
) -> str:
    dtype_str = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(torch_dtype, str(torch_dtype).split(".")[-1])
    vision_batch_size = batch_size * num_images_per_sample
    return f"_{tp_degree}_{dtype_str}_tbs{batch_size}_vbs{vision_batch_size}_s{max_seq_len}"


def get_hf_config(
    hf_model_path: Path,
    torch_dtype: Optional[torch.dtype] = None,
    num_hidden_layers: Optional[int] = None,
) -> Gemma3Config:
    hf_config = Gemma3Config.from_pretrained(hf_model_path)

    if torch_dtype is not None:
        hf_config.torch_dtype = torch_dtype

    if num_hidden_layers is not None:
        hf_config.num_hidden_layers = num_hidden_layers
        if getattr(hf_config, "text_config", None) is not None:
            hf_config.text_config.num_hidden_layers = num_hidden_layers
        if getattr(hf_config, "vision_config", None) is not None:
            hf_config.vision_config.num_hidden_layers = num_hidden_layers

    return hf_config


def save_hf_checkpoint(
    output_dir_path: Path,
    config_file_path: Path,
    torch_dtype: torch.dtype,
) -> None:
    hf_config = Gemma3Config.from_pretrained(config_file_path, torch_dtype=torch_dtype)
    hf_model = Gemma3ForConditionalGeneration(config=hf_config)  # random weights
    hf_model.save_pretrained(output_dir_path)


def create_neuron_config(
    hf_config_path: Path,
    text_batch_size: int = 1,
    vision_batch_size: int = 1,
    total_max_seq_len: int = 1024,
    torch_dtype: torch.dtype = torch.float16,
    lnc: int = 1,
    tp_degree: int = 8,
) -> Gemma3InferenceConfig:
    text_config = NeuronConfig(
        batch_size=text_batch_size,
        seq_len=total_max_seq_len,
        torch_dtype=torch_dtype,
        rpl_reduce_dtype=torch.float32,
        cast_type="as-declared",
        logical_nc_config=lnc,
        tp_degree=tp_degree,
        world_size=tp_degree,
        skip_sharding=False,
        save_sharded_checkpoint=True,
        enable_bucketing=True,
        context_encoding_buckets=[total_max_seq_len],
        token_generation_buckets=[total_max_seq_len],
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=False,
            do_sample=False,
            deterministic=True,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=False,
        ),
        output_logits=True,
    )

    vision_config = NeuronConfig(
        batch_size=vision_batch_size,
        seq_len=total_max_seq_len,  # Does not matter
        torch_dtype=torch_dtype,
        rpl_reduce_dtype=torch.float32,
        logical_nc_config=lnc,
        tp_degree=tp_degree,
        world_size=tp_degree,
        skip_sharding=False,
        save_sharded_checkpoint=True,
        enable_bucketing=True,
        buckets=[vision_batch_size],
    )

    nrn_config = Gemma3InferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config_path),
    )
    return nrn_config


def create_generation_config(nrn_config: Gemma3InferenceConfig) -> GenerationConfig:
    return GenerationConfig(
        do_sample=False,
        pad_token_id=nrn_config.text_config.pad_token_id,
        output_scores=True,  # Processed & warped logits
        output_logits=False,  # Raw logits -> not needed
        return_dict_in_generate=True,
    )


def prepare_inputs(
    nrn_config: Gemma3InferenceConfig, torch_dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    batch_size = nrn_config.text_config.neuron_config.batch_size
    text_tokens_length = 16
    text_input_ids = (
        torch.rand((batch_size, text_tokens_length)) * nrn_config.text_config.vocab_size
    )

    image_per_sample = (
        nrn_config.vision_config.neuron_config.batch_size // batch_size
    )
    vision_tokens_length = nrn_config.mm_tokens_per_image
    vision_input_ids = torch.full(
        [batch_size, image_per_sample * vision_tokens_length],
        fill_value=nrn_config.image_token_index,
    )

    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1).to(
        dtype=torch.int32
    )

    total_length = text_tokens_length + vision_tokens_length
    attention_mask_2d = torch.ones((batch_size, total_length), dtype=torch.int32)

    pixel_values = torch.rand(
        (
            batch_size * image_per_sample,
            nrn_config.vision_config.num_channels,
            nrn_config.vision_config.image_size,
            nrn_config.vision_config.image_size,
        ),
        dtype=torch.float32,
    )
    pixel_values = (2.0 * pixel_values - 1.0).to(dtype=torch_dtype)

    vision_mask = (input_ids == nrn_config.image_token_index).unsqueeze(-1)
    vision_mask = vision_mask.to(torch.bool)

    return input_ids, attention_mask_2d, pixel_values, vision_mask
