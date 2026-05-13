# Copyright 2025 © Amazon.com and Affiliates
"""Isaac-0.2-2B NxDI integration test script.

Compiles and runs the Isaac VLM model on Neuron.
Supports both text-only and image+text generation.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    export PYTHONPATH=/mnt/models/neuronx-distributed-inference/contrib/models/Isaac-0.2-2B/src:$PYTHONPATH
    python run_isaac.py
"""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import logging  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, AutoProcessor  # noqa: E402

from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)  # noqa: E402
from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: E402
    load_pretrained_config,
    HuggingFaceGenerationAdapter,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)  # noqa: E402

from isaac_neuron.modeling_isaac import (  # noqa: E402
    NeuronIsaacForConditionalGeneration,
    IsaacInferenceConfig,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Model configuration
DATA_PATH = os.getenv("DATA_HOME", "/mnt/models")

CONFIG = {
    "TEXT_TP_DEGREE": 1,  # TP=1 for 2B model on trn2.3xlarge
    "VISION_TP_DEGREE": 1,
    "WORLD_SIZE": 1,
    "BATCH_SIZE": 1,
    "SEQ_LENGTH": 1024,  # Start small for initial compilation test
    "CTX_BUCKETS": [1024],
    "TKG_BUCKETS": [1024],
    "DTYPE": torch.bfloat16,
    "MODEL_PATH": f"{DATA_PATH}/Isaac-0.2-2B-Preview",
    "TRACED_MODEL_PATH": f"{DATA_PATH}/traced_model/Isaac-0.2-2B",
    "MAX_NEW_TOKENS": 50,
    # Optimizations
    "FUSED_QKV": False,  # Start without QKV fusion
    "VISION_FUSED_QKV": False,
    "ASYNC_MODE": False,  # Disable async for debugging
    "OUTPUT_LOGITS": True,
    "ON_DEVICE_SAMPLING": OnDeviceSamplingConfig(
        dynamic=True,
        do_sample=True,
        deterministic=True,
        temperature=1.0,
        top_p=1.0,
        top_k=1,  # Greedy for validation
        global_topk=256,
        top_k_kernel_enabled=True,
    ),
}

# Environment setup
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
torch.manual_seed(42)


def create_neuron_configs():
    """Create text and vision neuron configurations."""
    text_config = NeuronConfig(
        batch_size=CONFIG["BATCH_SIZE"],
        seq_len=CONFIG["SEQ_LENGTH"],
        torch_dtype=CONFIG["DTYPE"],
        # Distributed
        tp_degree=CONFIG["TEXT_TP_DEGREE"],
        cp_degree=1,
        save_sharded_checkpoint=True,
        skip_sharding=False,
        # Continuous batching
        is_continuous_batching=True,
        ctx_batch_size=1,
        # Bucketing
        enable_bucketing=True,
        context_encoding_buckets=CONFIG["CTX_BUCKETS"],
        token_generation_buckets=CONFIG["TKG_BUCKETS"],
        # Optimizations
        async_mode=CONFIG["ASYNC_MODE"],
        on_device_sampling_config=CONFIG["ON_DEVICE_SAMPLING"],
        output_logits=CONFIG["OUTPUT_LOGITS"],
        fused_qkv=CONFIG["FUSED_QKV"],
        sequence_parallel_enabled=False,
        # Kernels — conservative for initial test
        # ISA limit: text MLP intermediate=6144 > 4096 at TP=1
        attn_kernel_enabled=False,
        attn_tkg_nki_kernel_enabled=False,
        attn_tkg_builtin_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    vision_config = NeuronConfig(
        batch_size=CONFIG["BATCH_SIZE"],
        seq_len=CONFIG["SEQ_LENGTH"],
        torch_dtype=CONFIG["DTYPE"],
        # Distributed
        tp_degree=CONFIG["VISION_TP_DEGREE"],
        world_size=CONFIG["WORLD_SIZE"],
        save_sharded_checkpoint=True,
        # Continuous batching
        is_continuous_batching=True,
        ctx_batch_size=1,
        # Bucketing
        enable_bucketing=True,
        buckets=[1],
        # Optimizations
        fused_qkv=CONFIG["VISION_FUSED_QKV"],
        # Kernels — all disabled for vision encoder
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )

    return text_config, vision_config


def setup_model():
    """Initialize model configuration and compile/load."""
    text_config, vision_config = create_neuron_configs()

    # Isaac uses trust_remote_code; load HF config directly
    hf_config = AutoConfig.from_pretrained(CONFIG["MODEL_PATH"], trust_remote_code=True)

    config = IsaacInferenceConfig(
        text_neuron_config=text_config,
        vision_neuron_config=vision_config,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    print(
        f"Text config: {config.text_config.num_hidden_layers} layers, "
        f"hidden={config.text_config.hidden_size}"
    )
    print(
        f"Vision config: {config.vision_config.num_hidden_layers} layers, "
        f"hidden={config.vision_config.hidden_size}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["MODEL_PATH"], padding_side="right", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return config, tokenizer


def compile_model(config, tokenizer):
    """Compile model (text + vision) and save traced artifacts."""
    print("\nCompiling Isaac model (text + vision)...")
    model = NeuronIsaacForConditionalGeneration(CONFIG["MODEL_PATH"], config)
    # debug=False to avoid profiler's CUDA introspection issue on Neuron instances
    model.compile(CONFIG["TRACED_MODEL_PATH"], debug=False)
    tokenizer.save_pretrained(CONFIG["TRACED_MODEL_PATH"])
    print(f"Model compiled and saved to {CONFIG['TRACED_MODEL_PATH']}")
    # Load compiled model for inference
    model.load(CONFIG["TRACED_MODEL_PATH"], skip_warmup=True)
    return model


def load_model():
    """Load pre-compiled model from traced checkpoint."""
    print(f"\nLoading model from {CONFIG['TRACED_MODEL_PATH']}...")
    model = NeuronIsaacForConditionalGeneration(CONFIG["TRACED_MODEL_PATH"])
    model.load(CONFIG["TRACED_MODEL_PATH"], skip_warmup=True)
    return model


def run_text_only(model, tokenizer):
    """Run text-only generation test."""
    print("\n=== Text-only Generation ===")
    prompt = "The capital of France is"

    messages = [{"role": "user", "content": prompt}]
    # Use tokenizer directly (Isaac's processor requires tensor_stream for images)
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    attention_mask = torch.ones_like(input_ids)

    print(f"Input: '{prompt}'")
    print(f"Input IDs shape: {input_ids.shape}")

    generation_model = HuggingFaceGenerationAdapter(model)
    sampling_params = prepare_sampling_params(
        batch_size=CONFIG["BATCH_SIZE"],
        top_k=[1],
        top_p=[1.0],
        temperature=[0.0],
    )

    outputs = generation_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
        max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
    )

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, text in enumerate(output_text):
        print(f"Output {i}: {text}")


def main():
    import sys

    config, tokenizer = setup_model()

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode == "compile":
        # Force recompilation
        import shutil

        if os.path.exists(CONFIG["TRACED_MODEL_PATH"]):
            print(f"Removing old traced model at {CONFIG['TRACED_MODEL_PATH']}...")
            shutil.rmtree(CONFIG["TRACED_MODEL_PATH"])
        model = compile_model(config, tokenizer)
    elif mode == "load":
        # Load only
        model = load_model()
    else:
        # Auto: compile if not found, else load
        if not os.path.exists(CONFIG["TRACED_MODEL_PATH"]):
            model = compile_model(config, tokenizer)
        else:
            model = load_model()

    run_text_only(model, tokenizer)


if __name__ == "__main__":
    main()
