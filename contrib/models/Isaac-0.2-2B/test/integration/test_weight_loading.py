"""Test weight loading: HF -> NxDI state dict conversion for Isaac."""

from isaac_neuron.ndxi_patch import apply_patch

apply_patch()

import torch
from collections import OrderedDict
from transformers import AutoConfig, AutoModelForCausalLM
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed.utils import cpu_mode
from isaac_neuron.modeling_isaac import (
    IsaacInferenceConfig,
    NeuronIsaacForConditionalGeneration,
)

MODEL_PATH = "/mnt/models/Isaac-0.2-2B-Preview"


def main():
    # 1) Load HF model and get state dict
    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    hf_state_dict = OrderedDict(hf_model.state_dict())
    print(f"HF state dict keys: {len(hf_state_dict)}")
    for k in sorted(hf_state_dict.keys())[:15]:
        print(f"  {k}: {hf_state_dict[k].shape}")
    print("  ...")
    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2) Create NxDI config
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text_nc = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        cp_degree=1,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        context_encoding_buckets=[1024],
        token_generation_buckets=[1024],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
        on_device_sampling_config=OnDeviceSamplingConfig(
            dynamic=True,
            do_sample=True,
            deterministic=True,
            top_k=1,
            global_topk=256,
            top_k_kernel_enabled=True,
        ),
        output_logits=True,
    )
    vision_nc = NeuronConfig(
        batch_size=1,
        seq_len=1024,
        torch_dtype=torch.bfloat16,
        tp_degree=1,
        world_size=1,
        is_continuous_batching=True,
        ctx_batch_size=1,
        enable_bucketing=True,
        buckets=[1],
        fused_qkv=False,
        attn_kernel_enabled=False,
        qkv_kernel_enabled=False,
        mlp_kernel_enabled=False,
    )
    config = IsaacInferenceConfig(
        text_neuron_config=text_nc,
        vision_neuron_config=vision_nc,
        load_config=load_pretrained_config(hf_config=hf_config),
    )

    # 3) Run state dict conversion
    print("\nRunning convert_hf_to_neuron_state_dict...")
    neuron_sd = NeuronIsaacForConditionalGeneration.convert_hf_to_neuron_state_dict(
        hf_state_dict, config
    )
    print(f"Neuron state dict keys: {len(neuron_sd)}")

    # 4) Compute expected NxDI parameter names analytically
    print("\nComputing expected NxDI parameter names...")

    # Text model expected keys (28 decoder layers, Qwen3 architecture)
    num_text_layers = config.text_config.num_hidden_layers  # 28
    expected_text = set()
    expected_text.add("embed_tokens.weight")
    expected_text.add("lm_head.weight")
    expected_text.add("norm.weight")
    for i in range(num_text_layers):
        pfx = f"layers.{i}"
        expected_text.add(f"{pfx}.input_layernorm.weight")
        expected_text.add(f"{pfx}.post_attention_layernorm.weight")
        expected_text.add(f"{pfx}.mlp.gate_proj.weight")
        expected_text.add(f"{pfx}.mlp.up_proj.weight")
        expected_text.add(f"{pfx}.mlp.down_proj.weight")
        # NxDI attention: qkv_proj.{q,k,v}_proj.weight, o_proj.o_proj.weight
        expected_text.add(f"{pfx}.self_attn.qkv_proj.q_proj.weight")
        expected_text.add(f"{pfx}.self_attn.qkv_proj.k_proj.weight")
        expected_text.add(f"{pfx}.self_attn.qkv_proj.v_proj.weight")
        expected_text.add(f"{pfx}.self_attn.o_proj.o_proj.weight")
        expected_text.add(f"{pfx}.self_attn.q_layernorm.weight")
        expected_text.add(f"{pfx}.self_attn.k_layernorm.weight")

    # Vision encoder expected keys (SigLIP2, 27 layers)
    num_vision_layers = config.vision_config.num_hidden_layers  # 27
    expected_vision = set()
    # SigLIP patch embedding
    expected_vision.add(
        "vision_encoder.vision_encoder.vision_model.embeddings.patch_embedding.weight"
    )
    expected_vision.add(
        "vision_encoder.vision_encoder.vision_model.embeddings.patch_embedding.bias"
    )
    expected_vision.add(
        "vision_encoder.vision_encoder.vision_model.embeddings.position_embedding.weight"
    )
    # SigLIP encoder layers
    for i in range(num_vision_layers):
        vpfx = f"vision_encoder.vision_encoder.vision_model.encoder.layers.{i}"
        expected_vision.add(f"{vpfx}.layer_norm1.weight")
        expected_vision.add(f"{vpfx}.layer_norm1.bias")
        expected_vision.add(f"{vpfx}.layer_norm2.weight")
        expected_vision.add(f"{vpfx}.layer_norm2.bias")
        # NxDI vision attention: qkv_proj.{q,k,v}_proj.{weight,bias}, o_proj.o_proj.{weight,bias}
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.q_proj.weight")
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.q_proj.bias")
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.k_proj.weight")
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.k_proj.bias")
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.v_proj.weight")
        expected_vision.add(f"{vpfx}.self_attn.qkv_proj.v_proj.bias")
        expected_vision.add(f"{vpfx}.self_attn.o_proj.o_proj.weight")
        expected_vision.add(f"{vpfx}.self_attn.o_proj.o_proj.bias")
        # MLP
        expected_vision.add(f"{vpfx}.mlp.fc1.weight")
        expected_vision.add(f"{vpfx}.mlp.fc1.bias")
        expected_vision.add(f"{vpfx}.mlp.fc2.weight")
        expected_vision.add(f"{vpfx}.mlp.fc2.bias")
    # SigLIP post layer norm
    expected_vision.add(
        "vision_encoder.vision_encoder.vision_model.post_layernorm.weight"
    )
    expected_vision.add(
        "vision_encoder.vision_encoder.vision_model.post_layernorm.bias"
    )
    # MLP projector
    expected_vision.add("vision_encoder.multi_modal_projector.fc1.weight")
    expected_vision.add("vision_encoder.multi_modal_projector.fc2.weight")

    expected_keys = expected_text | expected_vision
    neuron_keys = set(neuron_sd.keys())

    # Filter runtime keys
    skip_patterns = ("rank_util", "sampler", "lm_head.bias")
    neuron_filtered = {k for k in neuron_keys if not any(p in k for p in skip_patterns)}

    missing = expected_keys - neuron_filtered
    unexpected = neuron_filtered - expected_keys

    print(f"\n=== RESULTS ===")
    print(f"Expected keys: {len(expected_keys)}")
    print(f"Neuron state dict keys (filtered): {len(neuron_filtered)}")
    print(f"Missing (in model, not in weights): {len(missing)}")
    print(f"Unexpected (in weights, not in model): {len(unexpected)}")

    if missing:
        print("\nMISSING keys:")
        for k in sorted(missing):
            print(f"  {k}")

    if unexpected:
        print("\nUNEXPECTED keys:")
        for k in sorted(unexpected):
            print(f"  {k}")

    if not missing and not unexpected:
        print("\n*** ALL WEIGHTS MATCH PERFECTLY ***")


if __name__ == "__main__":
    main()
