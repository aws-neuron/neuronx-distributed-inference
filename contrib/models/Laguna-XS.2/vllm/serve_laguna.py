#!/usr/bin/env python3
"""Register Laguna-XS.2 with vLLM-neuron and start the API server.

This script:
1. Patches AutoConfig.from_pretrained to bypass the incompatible @strict
   decorator in Laguna's custom config class (SDK 2.29 huggingface_hub issue)
2. Registers NeuronLagunaForCausalLM in the NxDI MODEL_TYPES registry
3. Launches the vLLM OpenAI-compatible API server

Prerequisites:
    - Pre-compiled model with sharded weights at NEURON_COMPILED_ARTIFACTS path:
        <compiled_path>/model.pt              (compiled NEFFs)
        <compiled_path>/neuron_config.json    (with save_sharded_checkpoint=true)
        <compiled_path>/weights/tp{0..N}_sharded_checkpoint.safetensors
    - Without pre-sharded weights, model loading will OOM on trn2.3xlarge (128GB RAM).
      Use the compile_and_shard.py script to prepare artifacts first.

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
    cd /path/to/neuronx-distributed-inference
    export PYTHONPATH=src:contrib/models/Laguna-XS.2:$PYTHONPATH
    export NEURON_COMPILED_ARTIFACTS=/path/to/laguna-compiled
    export LAGUNA_CONTRIB_PATH=/path/to/contrib/models/Laguna-XS.2

    python contrib/models/Laguna-XS.2/vllm/serve_laguna.py \
        --model /path/to/Laguna-XS.2 \
        --tensor-parallel-size 4 \
        --max-model-len 4096 \
        --max-num-seqs 4 \
        --block-size 128 \
        --no-enable-prefix-caching

    # Or with all defaults:
    python contrib/models/Laguna-XS.2/vllm/serve_laguna.py
"""

import json
import os
import sys


def patch_autoconfig_for_laguna():
    """Patch AutoConfig.from_pretrained to handle Laguna's config.

    Laguna's config.json has auto_map pointing to a custom LagunaConfig class
    that uses a @strict decorator from huggingface_hub. This decorator is
    incompatible with the huggingface_hub version in SDK 2.29, causing:
        StrictDataclassDefinitionError: Class 'LagunaConfig' must be a
        dataclass before applying @strict.

    We intercept AutoConfig.from_pretrained to return a generic
    PretrainedConfig for laguna model_type configs, bypassing the custom class.
    """
    from transformers import AutoConfig, PretrainedConfig

    _orig_from_pretrained = AutoConfig.from_pretrained.__func__

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_path = os.path.join(str(pretrained_model_name_or_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            if config_dict.get("model_type") == "laguna":
                # vLLM validates rope_parameters expecting a top-level rope_type.
                # Laguna uses nested per-attention-type RoPE params which our NxDI
                # model handles directly. Add a top-level rope_type for vLLM compat.
                rope_params = config_dict.get("rope_parameters", {})
                if "rope_type" not in rope_params:
                    rope_params["rope_type"] = "yarn"
                    config_dict["rope_parameters"] = rope_params
                return PretrainedConfig(**config_dict)
        return _orig_from_pretrained(cls, pretrained_model_name_or_path, **kwargs)

    AutoConfig.from_pretrained = patched_from_pretrained
    print("[Laguna] Patched AutoConfig.from_pretrained")


def patch_nxdi_load_config():
    """Patch NxDI's hf_adapter.py on disk to handle Laguna's trust_remote_code issue.

    The engine core subprocess also needs this patch, so we write to disk.
    """
    import importlib

    spec = importlib.util.find_spec("neuronx_distributed_inference.utils.hf_adapter")
    adapter_path = spec.origin

    with open(adapter_path, "r") as f:
        content = f.read()

    if "laguna" in content:
        print("[Laguna] hf_adapter.py already patched")
        return

    # Patch the load_pretrained_config function to handle laguna
    # Find the AutoConfig.from_pretrained call and add a laguna check before it
    old_line = (
        "config: PretrainedConfig = AutoConfig.from_pretrained(model_path_or_name)"
    )
    new_lines = (
        "# Laguna: bypass custom config class (incompatible @strict decorator)\n"
        "            import json as _json\n"
        "            _cfg_path = os.path.join(str(model_path_or_name), 'config.json')\n"
        "            if os.path.exists(_cfg_path):\n"
        "                with open(_cfg_path) as _f:\n"
        "                    _cd = _json.load(_f)\n"
        "                if _cd.get('model_type') == 'laguna':\n"
        "                    # Add top-level rope_type for vLLM compat\n"
        "                    _rp = _cd.get('rope_parameters', {})\n"
        "                    if 'rope_type' not in _rp:\n"
        "                        _rp['rope_type'] = 'yarn'\n"
        "                        _cd['rope_parameters'] = _rp\n"
        "                    config = PretrainedConfig(**_cd)\n"
        "                else:\n"
        "                    config = AutoConfig.from_pretrained(model_path_or_name)\n"
        "            else:\n"
        "                config = AutoConfig.from_pretrained(model_path_or_name)"
    )

    if old_line in content:
        content = content.replace(old_line, new_lines, 1)
        with open(adapter_path, "w") as f:
            f.write(content)
        print(f"[Laguna] Patched hf_adapter.py at {adapter_path}")
    else:
        print(
            "[Laguna] WARNING: Could not find AutoConfig.from_pretrained in hf_adapter.py"
        )


def register_laguna_model():
    """Register Laguna in NxDI MODEL_TYPES (on disk) and vLLM ModelRegistry.

    Because vLLM 0.16.0 spawns engine core as a subprocess, in-memory patches
    to MODEL_TYPES don't propagate. We must patch the NxDI constants.py file
    on disk so the subprocess can also find 'laguna'.
    """
    import importlib

    # Patch NxDI constants.py on disk
    spec = importlib.util.find_spec("neuronx_distributed_inference.utils.constants")
    constants_path = spec.origin

    with open(constants_path, "r") as f:
        content = f.read()

    if "laguna" in content:
        print("[Laguna] NxDI constants.py already patched")
    else:
        # Add import and registration after the last existing MODEL_TYPES entry
        # The NxDI constants.py looks like:
        #   MODEL_TYPES = {
        #       "gpt_oss": ...,
        #       ...
        #       "qwen3_vl": ...,
        #   }
        # We add our import at the top and entry in the dict.

        # Add import at the module level (after last existing import)
        import_line = (
            "\n# Laguna contrib model (auto-patched by serve_laguna.py)\n"
            "import sys as _sys\n"
            "import os as _os\n"
            "_laguna_contrib = _os.environ.get('LAGUNA_CONTRIB_PATH', '')\n"
            "if _laguna_contrib and _laguna_contrib not in _sys.path:\n"
            "    _sys.path.insert(0, _laguna_contrib)\n"
        )

        # Instead of complex file parsing, just add to MODEL_TYPES at runtime
        # by appending code that modifies the dict after it's defined
        patch_code = (
            "\n# Laguna contrib registration (auto-patched)\n"
            "try:\n"
            "    from src.modeling_laguna import NeuronLagunaForCausalLM\n"
            '    MODEL_TYPES["laguna"] = {"causal-lm": NeuronLagunaForCausalLM}\n'
            "except ImportError:\n"
            "    pass  # Laguna contrib not in PYTHONPATH\n"
        )

        content += patch_code

        with open(constants_path, "w") as f:
            f.write(content)
        print(f"[Laguna] Patched NxDI constants.py at {constants_path}")

    # Also register in-memory for this process
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES

    if "laguna" not in MODEL_TYPES:
        from src.modeling_laguna import NeuronLagunaForCausalLM

        MODEL_TYPES["laguna"] = {"causal-lm": NeuronLagunaForCausalLM}

    # Register in vLLM's ModelRegistry to pass architecture validation.
    # On Neuron, actual model loading is handled by NxDI (not vLLM's model classes),
    # so we register LlamaForCausalLM as a placeholder.
    from vllm.model_executor.models.registry import ModelRegistry

    try:
        ModelRegistry.register_model(
            "LagunaForCausalLM",
            "vllm.model_executor.models.llama:LlamaForCausalLM",
        )
        print("[Laguna] Registered LagunaForCausalLM in vLLM ModelRegistry")
    except Exception as e:
        print(f"[Laguna] ModelRegistry registration note: {e}")


def main():
    # Apply patches BEFORE any vLLM or transformers imports that might trigger
    # AutoConfig.from_pretrained for the Laguna model
    patch_autoconfig_for_laguna()
    patch_nxdi_load_config()
    register_laguna_model()

    # Set framework env var
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")

    # Default compiled artifacts path (must contain model.pt + weights/ directory)
    os.environ.setdefault("NEURON_COMPILED_ARTIFACTS", "/path/to/laguna-compiled")

    # Build argv for vLLM if not already provided
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "--model",
            os.environ.get("LAGUNA_MODEL_PATH", "/path/to/Laguna-XS.2"),
            "--tensor-parallel-size",
            os.environ.get("LAGUNA_TP_DEGREE", "4"),
            "--max-model-len",
            os.environ.get("LAGUNA_MAX_MODEL_LEN", "4096"),
            "--max-num-seqs",
            os.environ.get("LAGUNA_MAX_NUM_SEQS", "4"),
            "--block-size",
            "128",
            "--no-enable-prefix-caching",
            "--trust-remote-code",
        ]

    # Ensure required args are present
    if "--trust-remote-code" not in sys.argv:
        sys.argv.append("--trust-remote-code")
    if "--block-size" not in sys.argv:
        sys.argv.extend(["--block-size", "128"])
    if "--no-enable-prefix-caching" not in sys.argv:
        sys.argv.append("--no-enable-prefix-caching")

    print(f"[Laguna] Starting vLLM with args: {sys.argv[1:]}")
    print()

    # Launch vLLM server
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import FlexibleArgumentParser, make_arg_parser

    parser = FlexibleArgumentParser(description="vLLM OpenAI-compatible server")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    import asyncio

    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
