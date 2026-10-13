import importlib.util
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = (
    _REPO_ROOT
    / "contrib"
    / "models"
    / "Qwen3.6-27B"
    / "scripts"
    / "openai_compat_server.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_openai_compat_server",
    _SCRIPT_PATH,
)
_SERVER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SERVER)


def test_stop_string_is_treated_as_one_sequence():
    assert _SERVER._normalize_stop_sequences("END") == ["END"]


def test_stop_list_preserves_string_sequences():
    assert _SERVER._normalize_stop_sequences(["END", "DONE"]) == ["END", "DONE"]


def test_completion_prompt_preserves_token_id_prompt():
    assert _SERVER._completion_prompt([101, 202, 303]) == [101, 202, 303]


def test_completion_prompt_uses_first_token_id_prompt_for_batched_input():
    assert _SERVER._completion_prompt([[101, 202], [303, 404]]) == [101, 202]


def test_completion_prompt_uses_first_text_prompt_for_batched_input():
    assert _SERVER._completion_prompt(["first", "second"]) == "first"


def test_completion_prompt_rejects_mixed_token_id_prompt():
    with pytest.raises(ValueError, match="token-id prompt lists"):
        _SERVER._completion_prompt([101, "bad"])
