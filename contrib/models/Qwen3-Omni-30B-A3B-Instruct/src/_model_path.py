import os

_DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def resolve_model_path() -> str:
    return os.environ.get("QWEN3_OMNI_MODEL_PATH", _DEFAULT_MODEL_ID)
