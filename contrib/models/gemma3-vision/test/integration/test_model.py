from gemma3_vision.ndxi_patch import apply_patch
apply_patch()

import os # noqa: E402
from pathlib import Path # noqa: E402
from typing import Dict # noqa: E402

import pytest
import torch

from neuronx_distributed_inference.utils.accuracy import (
    generate_expected_logits,
    check_accuracy_logits_v2,
)
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

from gemma3_vision.modeling_gemma3 import NeuronGemma3ForConditionalGeneration
from .utils import (
    get_test_name_suffix,
    save_hf_checkpoint,
    create_neuron_config,
    create_generation_config,
    prepare_inputs,
)


NUM_TOKENS_TO_CHECK = 16
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))


@pytest.mark.parametrize(
    "config_file_path,tp_degree,torch_dtype,batch_size,num_images_per_sample,total_max_seq_len,token_divergence_atol,perf_thresholds",
    [
        (
            Path(__file__).resolve().parent / "config_gemma3_4layers.json",
            8,
            torch.float16,
            1,
            1,
            1024,
            0.02,
            {
                "text_cte_p50_latency": 20.55,
                "text_cte_throughput": 49807.3,
                "tkg_p50_latency": 4.42,
                "tkg_throughput": 226.4,
            },
        ),
    ]
)
def test_original_cpu_vs_nxdi_neuron(
    tmp_path: Path,
    config_file_path: Path,
    tp_degree: int,
    torch_dtype: torch.dtype,
    batch_size: int,
    num_images_per_sample: int,
    total_max_seq_len: int,
    token_divergence_atol: float,
    perf_thresholds: Dict[str, float],
    ) -> None:
    suffix = get_test_name_suffix(
        tp_degree=tp_degree,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        num_images_per_sample=num_images_per_sample,
        max_seq_len=total_max_seq_len
    )

    nrn_config = create_neuron_config(
        hf_config_path=config_file_path,
        text_batch_size=batch_size,
        vision_batch_size=(num_images_per_sample * batch_size),
        total_max_seq_len=total_max_seq_len,
        torch_dtype=torch_dtype,
        lnc=LNC,
        tp_degree=tp_degree
    )

    input_ids, attention_mask, pixel_values, vision_mask = prepare_inputs(
        nrn_config=nrn_config,
        torch_dtype=torch_dtype
    )

    generation_config = create_generation_config(nrn_config=nrn_config)

    save_hf_checkpoint(
        output_dir_path=tmp_path,
        config_file_path=config_file_path,
        torch_dtype=torch_dtype,
        )

    nrn_config._name_or_path = tmp_path.as_posix()
    nrn_model = NeuronGemma3ForConditionalGeneration(model_path=tmp_path, config=nrn_config)

    traced_model_path = tmp_path / ("traced_model" + suffix)
    traced_model_path.mkdir(exist_ok=True)

    nrn_model.compile(traced_model_path.as_posix())

    nrn_model.load(traced_model_path.as_posix())

    benchmark_report = benchmark_sampling(
        model=nrn_model,
        generation_config=generation_config,
        image=False, # image=True currently broken (Neuron 2.27.1)
        benchmark_report_path=f"./benchmark_report{suffix}.json"
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
        num_tokens=NUM_TOKENS_TO_CHECK,
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
        num_tokens_to_check=NUM_TOKENS_TO_CHECK,
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
    batch_size = num_images_per_sample = 1
    total_max_seq_len = 1024

    test_original_cpu_vs_nxdi_neuron(
        config_file_path=config_file_path,
        tmp_path=tmp_dir_path,
        torch_dtype=torch_dtype,
        token_divergence_atol=token_divergence_atol,
        perf_thresholds=perf_thresholds,
        tp_degree=tp_degree,
        batch_size=batch_size,
        num_images_per_sample=num_images_per_sample,
        total_max_seq_len=total_max_seq_len,
        )

    tmp_dir.cleanup()
