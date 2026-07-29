"""Minimal Hugging Face config registration for Qwen3.5/Qwen3.6 vLLM smoke.

The Neuron vLLM environment can lag upstream Transformers. vLLM validates the
HF config before the NxDI model registry gets a chance to instantiate the
contrib model, so register a permissive config class for the new model_type.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Qwen35TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen35Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen35TextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen35TextConfig(**text_config)
        self.text_config = text_config
        if text_config is not None:
            for name, value in text_config.to_dict().items():
                if name not in {"architectures", "model_type"}:
                    kwargs.setdefault(name, value)
            rope_parameters = getattr(text_config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                kwargs.setdefault("rope_theta", rope_parameters.get("rope_theta"))
        super().__init__(**kwargs)


def _is_registered(model_type: str) -> bool:
    try:
        AutoConfig.for_model(model_type)
    except ValueError:
        return False
    return True


def register_qwen35_hf_config() -> None:
    if not _is_registered(Qwen35TextConfig.model_type):
        AutoConfig.register(Qwen35TextConfig.model_type, Qwen35TextConfig)
    if not _is_registered(Qwen35Config.model_type):
        AutoConfig.register(Qwen35Config.model_type, Qwen35Config)


def register_qwen35_vllm_architecture() -> None:
    try:
        from vllm.model_executor.models import ModelRegistry
    except Exception:
        return

    supported_archs = ModelRegistry.get_supported_archs()
    qwen3_impl = "vllm.model_executor.models.qwen3:Qwen3ForCausalLM"
    for arch in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"):
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, qwen3_impl)


def register_qwen35_config() -> None:
    register_qwen35_hf_config()
    register_qwen35_vllm_architecture()
