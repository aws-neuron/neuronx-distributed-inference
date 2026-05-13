"""Auto-register Qwen3.5/Qwen3.6 HF config when this folder is on PYTHONPATH.

Do not import vLLM here. Neuron helper commands such as libneuronpjrt-path run
inside Python subprocesses and expect clean stdout.
"""

from hf_qwen35_config import register_qwen35_hf_config

register_qwen35_hf_config()
