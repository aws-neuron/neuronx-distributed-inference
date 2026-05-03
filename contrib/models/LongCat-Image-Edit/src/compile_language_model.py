"""
Language Model Compilation using ModelBuilder API for Compiled Compatibility.

Compiles the Qwen2.5-VL Language Model (shared between Qwen-Image-Edit and
LongCat-Image-Edit) using ModelBuilder API with tp_degree=4 and world_size=8.

Key features:
- TP=4 is perfect for Qwen2.5-VL GQA: 28Q/4=7 heads/rank, 4KV/4=1 head/rank
- world_size=8 for compatibility with Compiled transformer
- Monkey-patches F.scaled_dot_product_attention with BMM-based implementation
  for Neuron tracing compatibility

Usage:
    neuron_parallel_compile python compile_language_model.py --max_sequence_length 512
"""

import os
import json
import gc
import math

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"
os.environ["XLA_DISABLE_FUNCTIONALIZATION"] = "0"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"

compiler_flags = """ --target=trn2 --lnc=2 --model-type=transformer --enable-fast-loading-neuron-binaries """
os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

import torch
import torch.nn as nn
import argparse

from diffusers import LongCatImageEditPipeline

from neuronx_distributed import ModelBuilder, NxDParallelState, shard_checkpoint
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state

from neuron_parallel_utils import shard_qwen2_attention, shard_qwen2_mlp, get_sharded_data

CACHE_DIR = "/opt/dlami/nvme/longcat_hf_cache"
MODEL_ID = "meituan-longcat/LongCat-Image-Edit"


def load_pipeline(dtype=torch.bfloat16):
    load_kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return LongCatImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)


class f32Wrapper(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
    def forward(self, x, *args, **kwargs):
        t = x.dtype
        output = self.original(x.to(torch.float32), *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


# ============================================================
# Custom SDPA replacement for Neuron tracing compatibility
# (from Qwen reference neuron_commons.py)
# ============================================================
def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=None, is_causal=None, scale=None,
                                         enable_gqa=False, **kwargs):
    """Custom scaled dot product attention using BMM for Neuron compatibility."""
    orig_shape = None
    q_len = query.shape[-2]
    kv_len = key.shape[-2]

    if len(query.shape) == 4:
        orig_shape = query.shape
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, _, _ = key.shape

        if num_kv_heads != num_q_heads:
            num_groups = num_q_heads // num_kv_heads
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])

    if scale is None:
        scale = 1 / math.sqrt(query.size(-1))

    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * scale

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=attention_scores.device), diagonal=1)
        causal_mask = torch.where(
            causal_mask == 1,
            torch.tensor(float('-inf'), dtype=attention_scores.dtype, device=attention_scores.device),
            torch.tensor(0.0, dtype=attention_scores.dtype, device=attention_scores.device))
        attention_scores = attention_scores + causal_mask

    if attn_mask is not None:
        if attn_mask.dim() == 4:
            attn_mask = attn_mask.reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        elif attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        attention_scores = attention_scores + attn_mask.to(attention_scores.dtype)

    attention_probs = attention_scores.softmax(dim=-1)
    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2])
    return attn_out


class NeuronLanguageModel(nn.Module):
    """Neuron-optimized Qwen2.5-VL Language Model with TP=4."""

    def __init__(self, original_language_model, tp_degree):
        super().__init__()
        self.tp_degree = tp_degree
        self.language_model = original_language_model
        self.config = original_language_model.config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers

        print(f"  Language model: hidden_size={self.hidden_size}, layers={self.num_hidden_layers}")
        print(f"    Q heads: {self.config.num_attention_heads}, KV heads: {self.config.num_key_value_heads}")

        for i, layer in enumerate(self.language_model.layers):
            layer.self_attn = shard_qwen2_attention(tp_degree, layer.self_attn)
            layer.mlp = shard_qwen2_mlp(layer.mlp)
            if i == 0:
                print(f"  Sharded layer 0")
        print(f"  Sharded all {len(self.language_model.layers)} layers")

        upcast_norms_to_f32(self.language_model)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.last_hidden_state


class TracingWrapper(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
    def forward(self, inputs_embeds, attention_mask, position_ids):
        return self.language_model(inputs_embeds, attention_mask, position_ids)


def compile_language_model(args):
    tp_degree = 4
    world_size = 8
    batch_size = args.batch_size
    sequence_length = args.max_sequence_length
    hidden_size = 3584

    print("=" * 60)
    print("Compiling Language Model (TP=4, BMM attention)")
    print("=" * 60)
    print(f"  Batch={batch_size}, SeqLen={sequence_length}, TP={tp_degree}, World={world_size}")

    # ============================================================
    # CRITICAL: Monkey-patch F.scaled_dot_product_attention BEFORE
    # loading the model, so the traced graph uses BMM-based attention
    # that Neuron can compile correctly.
    # ============================================================
    sdpa_original = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = neuron_scaled_dot_product_attention
    print("  Patched F.scaled_dot_product_attention -> neuron BMM attention")

    sample_inputs_embeds = torch.randn(batch_size, sequence_length, hidden_size, dtype=torch.bfloat16)
    # CRITICAL: Use realistic attention_mask with padding (not all-ones)
    # Real inputs have ~334/842 valid tokens + padding to max_seq_len
    # Tracing with all-ones mask may cause compiler to optimize away mask handling
    real_len = min(sequence_length * 2 // 3, sequence_length)  # ~2/3 real tokens
    sample_attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.int64)
    sample_attention_mask[:, :real_len] = 1
    # Use realistic position_ids (M-RoPE style, non-sequential, up to ~600)
    sample_position_ids = torch.zeros(3, batch_size, sequence_length, dtype=torch.int64)
    for d in range(3):
        sample_position_ids[d, :, :real_len] = torch.arange(real_len).unsqueeze(0)
        # Padding positions get continuing positions
        sample_position_ids[d, :, real_len:] = real_len + torch.arange(sequence_length - real_len).unsqueeze(0)

    with NxDParallelState(world_size=world_size, tensor_model_parallel_size=tp_degree):
        print("Loading model...")
        pipe = load_pipeline(torch.bfloat16)

        original_language_model = pipe.text_encoder.model.language_model
        unsharded_state = original_language_model.state_dict()

        print(f"\nCreating Neuron language model (TP={tp_degree})...")
        neuron_lm = NeuronLanguageModel(original_language_model, tp_degree)
        neuron_lm = neuron_lm.to(torch.bfloat16)
        neuron_lm.eval()

        del pipe
        gc.collect()

        model = TracingWrapper(neuron_lm)

        builder = ModelBuilder(model=model)
        print("Tracing...")
        builder.trace(
            kwargs={
                "inputs_embeds": sample_inputs_embeds,
                "attention_mask": sample_attention_mask,
                "position_ids": sample_position_ids,
            },
            tag="inference",
        )

        print("Compiling...")
        traced_model = builder.compile(
            compiler_args="--model-type=transformer -O1 --auto-cast=none",
            compiler_workdir=args.compiler_workdir,
        )

        output_path = f"{args.compiled_models_dir}/language_model"
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))

        weights_path = os.path.join(output_path, "weights")
        os.makedirs(weights_path, exist_ok=True)

        checkpoint = {}
        for key, value in model.state_dict().items():
            orig_key = key.replace("language_model.language_model.", "", 1)
            if orig_key in unsharded_state:
                checkpoint[key] = unsharded_state[orig_key].clone()
            else:
                checkpoint[key] = value.clone()

        shard_checkpoint(checkpoint=checkpoint, model=model, serialize_path=weights_path)

        # Post-process
        from safetensors.torch import load_file, save_file
        inv_freq_buffers = {}
        for name, buf in neuron_lm.language_model.named_buffers():
            if 'inv_freq' in name:
                inv_freq_buffers[f"language_model.language_model.{name}"] = buf.to(torch.bfloat16).clone()
        print(f"  Collected {len(inv_freq_buffers)} inv_freq buffers")

        for rank in range(tp_degree):
            shard_file = os.path.join(weights_path, f"tp{rank}_sharded_checkpoint.safetensors")
            if not os.path.exists(shard_file):
                continue
            data = dict(load_file(shard_file))
            cleaned = {k: v for k, v in data.items() if 'master_weight' not in k}
            cleaned.update(inv_freq_buffers)
            save_file(cleaned, shard_file)
            print(f"  tp{rank}: {len(data)} -> {len(cleaned)} tensors")

        config = {
            "max_sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "tp_degree": tp_degree,
            "world_size": world_size,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nLanguage Model compiled: {output_path}")

    # Restore original SDPA
    torch.nn.functional.scaled_dot_product_attention = sdpa_original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compiled_models_dir", type=str, default="/opt/dlami/nvme/compiled_models")
    parser.add_argument("--compiler_workdir", type=str, default="/opt/dlami/nvme/compiler_workdir")
    args = parser.parse_args()

    if args.model_path:
        MODEL_ID = args.model_path
        CACHE_DIR = None

    compile_language_model(args)
