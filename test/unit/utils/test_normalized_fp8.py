import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from neuronx_distributed_inference.modules.checkpoint import (
    inspect_safetensors_sharded,
)
from neuronx_distributed_inference.utils.normalized_fp8 import (
    _expected_names,
    validate_gpt_oss_120b_normalized_fp8_checkpoint,
)


def _config(**neuron_overrides):
    neuron_config = {
        "quantized": True,
        "quantization_dtype": "f8e4m3",
        "fused_qkv": False,
        "vocab_parallel": False,
        "attention_dp_degree": 1,
        "cp_degree": 1,
    }
    neuron_config.update(neuron_overrides)
    return SimpleNamespace(
        neuron_config=SimpleNamespace(**neuron_config),
        num_hidden_layers=36,
        num_local_experts=128,
    )


def _write_checkpoint(
    root: Path,
    *,
    dtype_overrides=None,
    shape_overrides=None,
    sidecar_names=None,
    rename=None,
    model_filename="model-00001-of-00001.safetensors",
):
    expected, fp8_weights, scalar_scales, expert_scales = _expected_names(36)
    dtype_overrides = dtype_overrides or {}
    shape_overrides = shape_overrides or {}
    sidecar_names = sidecar_names or expert_scales
    rename = rename or {}

    tensors = {}
    owners = {}
    for original_name in sorted(expected):
        name = rename.get(original_name, original_name)
        if original_name in fp8_weights:
            dtype = dtype_overrides.get(original_name, torch.float8_e4m3fn)
            shape = shape_overrides.get(original_name, (1, 1))
        elif original_name in scalar_scales:
            dtype = dtype_overrides.get(original_name, torch.float32)
            shape = shape_overrides.get(original_name, (1,))
        elif original_name in expert_scales:
            dtype = dtype_overrides.get(original_name, torch.float32)
            shape = shape_overrides.get(
                original_name,
                (128, 1, 1)
                if original_name.endswith("down_proj.scale")
                else (128, 2, 1),
            )
        elif original_name.endswith(".rank"):
            dtype = dtype_overrides.get(original_name, torch.int32)
            shape = shape_overrides.get(original_name, (1,))
        else:
            dtype = dtype_overrides.get(original_name, torch.bfloat16)
            shape = shape_overrides.get(original_name, (1,))
        tensors[name] = torch.zeros(shape, dtype=dtype)
        owners[name] = (
            "expert-scales.safetensors"
            if original_name in sidecar_names
            else model_filename
        )

    root.mkdir()
    model_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if owners[name] == model_filename
    }
    expert_tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if owners[name] == "expert-scales.safetensors"
    }
    save_file(model_tensors, str(root / model_filename))
    save_file(expert_tensors, str(root / "expert-scales.safetensors"))
    index = {"weight_map": owners}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return root


def test_valid_normalized_fp8_checkpoint_passes(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")

    validate_gpt_oss_120b_normalized_fp8_checkpoint(str(checkpoint), _config())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "rename": {
                    "layers.0.self_attn.learned_sinks.sink": "layers.0.self_attn.sinks"
                }
            },
            "namespace mismatch",
        ),
        (
            {"dtype_overrides": {"layers.0.self_attn.q_proj.weight": torch.bfloat16}},
            "not F8_E4M3",
        ),
        (
            {"shape_overrides": {"layers.0.self_attn.q_proj.scale": (2,)}},
            "shape/dtype mismatch",
        ),
        (
            {
                "shape_overrides": {
                    "layers.0.feed_forward.moe.expert_mlps.mlp_op.down_proj.scale": (
                        128,
                        2,
                        1,
                    )
                }
            },
            "shape/dtype mismatch",
        ),
        (
            {
                "sidecar_names": {
                    "layers.0.feed_forward.moe.expert_mlps.mlp_op.down_proj.scale"
                }
                ^ _expected_names(36)[3]
            },
            "not in expert sidecar",
        ),
    ],
)
def test_invalid_normalized_fp8_checkpoint_fails_closed(tmp_path, kwargs, message):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", **kwargs)

    with pytest.raises(ValueError, match=message):
        validate_gpt_oss_120b_normalized_fp8_checkpoint(str(checkpoint), _config())


@pytest.mark.parametrize(
    "neuron_overrides",
    [
        {"quantized": False},
        {"quantization_dtype": "bf16"},
        {"fused_qkv": True},
        {"vocab_parallel": True},
        {"attention_dp_degree": 2},
    ],
)
def test_invalid_normalized_fp8_config_fails_closed(tmp_path, neuron_overrides):
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")

    with pytest.raises(ValueError):
        validate_gpt_oss_120b_normalized_fp8_checkpoint(
            str(checkpoint), _config(**neuron_overrides)
        )


def test_cross_file_tensor_collision_fails_before_payload_load(tmp_path):
    checkpoint = tmp_path / "collision"
    checkpoint.mkdir()
    save_file({"duplicate": torch.ones(1)}, str(checkpoint / "one.safetensors"))
    save_file(
        {"duplicate": torch.ones(1), "other": torch.ones(1)},
        str(checkpoint / "two.safetensors"),
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "duplicate": "one.safetensors",
                    "other": "two.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate SafeTensors tensor name"):
        inspect_safetensors_sharded(str(checkpoint))


def test_canonical_model_file_with_sidecar_fails_closed(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint", model_filename="model.safetensors"
    )

    with pytest.raises(ValueError, match="indexed model shards"):
        validate_gpt_oss_120b_normalized_fp8_checkpoint(str(checkpoint), _config())


def test_default_hf_conversion_contract_remains_present():
    application_base = (
        Path(__file__).parents[3]
        / "src"
        / "neuronx_distributed_inference"
        / "models"
        / "application_base.py"
    )
    source = application_base.read_text(encoding="utf-8")
    assert 'os.environ.get("CHECKPOINT_FORMAT", "hf")' in source
    assert "if not normalized_fp8:" in source
    assert "model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)" in source

    gpt_oss_source = (
        application_base.parent / "gpt_oss" / "modeling_gpt_oss.py"
    ).read_text(encoding="utf-8")
    assert 'if checkpoint_format == "hf":' in gpt_oss_source
    assert "_convert_hf_format_state_dict" in gpt_oss_source
    assert "_convert_neuron_format_state_dict" in gpt_oss_source
