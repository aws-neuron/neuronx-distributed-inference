from gemma3_vision.ndxi_patch import apply_patch
apply_patch()

from pathlib import Path
from typing import Dict, Optional, Tuple

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.accuracy import (
    generate_expected_logits,
    check_accuracy_logits_v2,
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
import torch
from transformers import Gemma3ForConditionalGeneration, Gemma3Config, GenerationConfig

from gemma3_vision.modeling_gemma3 import NeuronGemma3ForCausalLM, Gemma3InferenceConfig


torch.manual_seed(0)


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
    hf_model = Gemma3ForConditionalGeneration(config=hf_config) # random weights
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
        seq_len=total_max_seq_len, # Does not matter
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
        output_logits=False, # Raw logits -> not needed
        return_dict_in_generate=True)
    

def prepare_inputs(nrn_config: Gemma3InferenceConfig, torch_dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    batch_size = nrn_config.text_config.neuron_config.batch_size
    text_tokens_length = 16
    text_input_ids = torch.rand((batch_size, text_tokens_length)) * nrn_config.text_config.vocab_size

    image_per_sample = nrn_config.vision_config.neuron_config.batch_size // batch_size
    vision_tokens_length = nrn_config.mm_tokens_per_image
    vision_input_ids = torch.full([batch_size, image_per_sample * vision_tokens_length], fill_value=nrn_config.image_token_index)

    input_ids = torch.cat((text_input_ids, vision_input_ids), dim=1).to(dtype=torch.int32)

    total_length = text_tokens_length + vision_tokens_length
    attention_mask_2d = torch.ones((batch_size, total_length), dtype=torch.int32)

    pixel_values = torch.rand((
            batch_size * image_per_sample,
            nrn_config.vision_config.num_channels,
            nrn_config.vision_config.image_size,
            nrn_config.vision_config.image_size,
        ),
        dtype=torch.float32
    )
    pixel_values = (2.0 * pixel_values - 1.0).to(dtype=torch_dtype)

    vision_mask = (input_ids == nrn_config.image_token_index).unsqueeze(-1)
    vision_mask = vision_mask.to(torch.bool)

    return input_ids, attention_mask_2d, pixel_values, vision_mask


def test_original_cpu_vs_nxdi_neuron(
    config_file_path: Path,
    tmp_dir_path: Path,
    torch_dtype: torch.dtype, 
    token_divergence_atol: float,
    perf_thresholds: Dict[str, float],
    batch_size: int = 1,
    num_images_per_sample: int = 1,
    total_max_seq_len: int = 1024,
    lnc: int = 1,
    tp_degree: int = 8,
    num_tokens_to_check: int = 16
    ) -> None:
    nrn_config = create_neuron_config(
        hf_config_path=config_file_path,
        text_batch_size=batch_size,
        vision_batch_size=(num_images_per_sample * batch_size),
        total_max_seq_len=total_max_seq_len,
        torch_dtype=torch_dtype,
        lnc=lnc,
        tp_degree=tp_degree
    )

    input_ids, attention_mask, pixel_values, vision_mask = prepare_inputs(
        nrn_config=nrn_config,
        torch_dtype=torch_dtype
    )
    
    generation_config = create_generation_config(nrn_config=nrn_config)

    save_hf_checkpoint(
        output_dir_path=tmp_dir_path, 
        config_file_path=config_file_path,
        torch_dtype=torch_dtype,
        )

    nrn_config._name_or_path = tmp_dir_path.as_posix()
    nrn_model = NeuronGemma3ForCausalLM(model_path=tmp_dir_path, config=nrn_config)

    traced_model_path = tmp_dir_path / "traced_model"
    traced_model_path.mkdir(exist_ok=True)
    
    nrn_model.compile(traced_model_path.as_posix())

    nrn_model.load(traced_model_path.as_posix())

    benchmark_report = benchmark_sampling(
        model=nrn_model, 
        generation_config=generation_config,
        image=False # image=True currently broken (Neuron 2.27.1)
        )
    
    assert benchmark_report["context_encoding_model"]["latency_ms_p50"] < perf_thresholds["text_cte_p50_latency"] * 1.1
    assert benchmark_report["context_encoding_model"]["throughput"] > perf_thresholds["text_cte_throughput"] * 0.9
    assert benchmark_report["token_generation_model"]["latency_ms_p50"] < perf_thresholds["tkg_p50_latency"] * 1.1
    assert benchmark_report["token_generation_model"]["throughput"] > perf_thresholds["tkg_throughput"] * 0.9

    expected_logits = generate_expected_logits(
        neuron_model=nrn_model,
        input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens=num_tokens_to_check,
        additional_input_args={
            "pixel_values": pixel_values,
        },
    )

    additional_input_args = {
        "pixel_values": pixel_values,
        "vision_mask": vision_mask,
    }

    check_accuracy_logits_v2(
        neuron_model=nrn_model,
        expected_logits=expected_logits,
        inputs_input_ids=input_ids,
        inputs_attention_mask=attention_mask,
        generation_config=generation_config,
        num_tokens_to_check=num_tokens_to_check,
        additional_input_args=additional_input_args,
        divergence_difference_tol=token_divergence_atol,
    )


if __name__ == "__main__":
    import tempfile
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmp_dir.name)
    torch_dtype = torch.float16
    token_divergence_atol = 0.02
    config_file_path = Path(__file__).resolve().parent / "config_gemma3_4layers.json"
    perf_thresholds = {
        "text_cte_p50_latency": 20.55,
        "text_cte_throughput": 49807.3,
        "tkg_p50_latency": 4.42,
        "tkg_throughput": 226.4,
    }
    tp_degree = 8

    test_original_cpu_vs_nxdi_neuron(
        config_file_path=config_file_path,
        tmp_dir_path=tmp_dir_path,
        torch_dtype=torch_dtype,
        token_divergence_atol=token_divergence_atol,
        perf_thresholds=perf_thresholds,
        tp_degree=tp_degree,
        )
    tmp_dir.cleanup()