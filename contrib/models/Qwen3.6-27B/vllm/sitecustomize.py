"""Auto-register Qwen3.5/Qwen3.6 HF config when this folder is on PYTHONPATH.

Do not import vLLM here unless explicitly requested through an environment
flag. Neuron helper commands such as libneuronpjrt-path run inside Python
subprocesses and expect clean stdout.
"""

import os

from hf_qwen35_config import register_qwen35_hf_config

register_qwen35_hf_config()

if any(
    os.environ.get(name)
    for name in (
        "QWEN36_HYBRID_APC_INSTALL_PATCH",
        "QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS",
    )
):
    from qwen36_hybrid_apc_scheduler_patch import install_import_hook

    install_import_hook()
