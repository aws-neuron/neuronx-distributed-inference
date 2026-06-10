import importlib.util
import urllib.error
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHAT_APC = _load_script(
    "qwen36_openai_chat_apc_validation",
    "validation_scripts/qwen36_openai_chat_apc_validation.py",
)
_BOUNDARY_APC = _load_script(
    "qwen36_openai_boundary_apc_probe",
    "validation_scripts/qwen36_openai_boundary_apc_probe.py",
)


def test_chat_apc_gate_fails_without_exact_repeats():
    summary = {
        "all_status_ok": True,
        "warm_full_exact_text": False,
        "partial_repeat_exact_text": True,
        "multi_turn_repeat_exact_text": True,
        "semantic_smoke_passed": True,
        "warm_full_speedup_passed": True,
        "partial_reference_speedup_passed": True,
    }

    assert _CHAT_APC._apc_gate_failures(summary) == ["warm_full_exact_text"]


def test_chat_apc_speedup_gate_requires_threshold():
    assert _CHAT_APC._speedup_passes(2.0, 1.5)
    assert not _CHAT_APC._speedup_passes(1.0, 1.5)
    assert not _CHAT_APC._speedup_passes(None, 1.5)
    assert _CHAT_APC._speedup_passes(None, 0.0)


def test_boundary_metric_snapshot_is_optional(monkeypatch):
    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("metrics disabled")

    monkeypatch.setattr(_BOUNDARY_APC.urllib.request, "urlopen", raise_url_error)

    assert _BOUNDARY_APC._metric_snapshot("http://127.0.0.1:8000", 0.1) == {}
