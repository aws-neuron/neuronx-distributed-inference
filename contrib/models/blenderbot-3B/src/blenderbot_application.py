"""
Blenderbot NeuronX Port - Application Layer (Part 2: Encoder, Decoder, Wrappers, Applications)

Follows the Whisper pattern for encoder-decoder models on NeuronX:
  - Separate encoder and decoder NeuronApplicationBase subclasses
  - Separate ModelWrapper subclasses for encoder, decoder-prefill, decoder-decode
  - Top-level NeuronApplicationBlenderbot orchestrates both
"""

import math
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.modules.checkpoint import load_state_dict

from .configuration_blenderbot_neuron import BlenderbotInferenceConfig
from .modeling_blenderbot_neuron import (
    LayerNorm, BlenderbotEncoderLayer, BlenderbotDecoderLayer, BlenderbotMLP,
    BlenderbotSelfAttention, BlenderbotCrossAttention,
)


# ─── Top-level Encoder Module ───

class NeuronBlenderbotEncoder(nn.Module):
    """
    Full encoder: embed_tokens + embed_positions + encoder_layers + layer_norm.
    Ref: BlenderbotEncoder in modeling_blenderbot.py
    """
    def __init__(self, vocab_size: int, d_model: int, max_pos: int, n_head: int,
                 ffn_dim: int, n_layer: int, batch_size: int, pad_token_id: int = 0,
                 scale_embedding: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_positions = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([
            BlenderbotEncoderLayer(d_model, n_head, ffn_dim, batch_size, max_pos, dtype=dtype)
            for _ in range(n_layer)
        ])
        self.layer_norm = LayerNorm(d_model)
        self.model_dtype = dtype

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        bsz, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        # Embedding weights stay float32; cast output (see learnings §8)
        hidden = (self.embed_tokens(input_ids) * self.embed_scale + self.embed_positions(position_ids)).to(self.model_dtype)
        # Expand attention_mask [bsz, seq_len] -> [bsz, 1, seq_len, seq_len] for self-attention
        # Each query position can attend to all non-padded key positions
        enc_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(torch.bool)
        for layer in self.layers:
            hidden = layer(hidden, mask=enc_mask)
        return self.layer_norm(hidden)


# ─── Top-level Decoder Module ───

class NeuronBlenderbotDecoder(nn.Module):
    """
    Full decoder: embed_tokens + embed_positions + decoder_layers + layer_norm + lm_head.
    Ref: BlenderbotDecoder + BlenderbotForConditionalGeneration in modeling_blenderbot.py
    """
    def __init__(self, vocab_size: int, d_model: int, max_pos: int, n_head: int,
                 ffn_dim: int, n_layer: int, batch_size: int, enc_seq_len: int,
                 pad_token_id: int = 0, scale_embedding: bool = True,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = max_pos
        self.vocab_size = vocab_size
        self.embed_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_positions = nn.Embedding(max_pos, d_model)
        self.layers = nn.ModuleList([
            BlenderbotDecoderLayer(d_model, n_head, ffn_dim, batch_size, max_pos, enc_seq_len, dtype=dtype)
            for _ in range(n_layer)
        ])
        self.layer_norm = LayerNorm(d_model)
        self.model_dtype = dtype

    def forward(self, x: Tensor, xa: Tensor, last_pos: Tensor, pad_mask: Tensor, encoder_mask: Tensor):
        is_prefill = x.shape[1] > 1
        if is_prefill:
            position_ids = torch.arange(self.seq_len, dtype=torch.long, device=x.device)
            pe = self.embed_positions(position_ids)
        else:
            pe = self.embed_positions(last_pos)

        hidden = (self.embed_tokens(x) * self.embed_scale + pe).to(self.model_dtype)

        # Build causal mask for self-attention
        if is_prefill:
            mask = torch.full((self.seq_len, self.seq_len), True, device=pad_mask.device).tril(diagonal=0)
            input_mask = pad_mask[:, None, None, :].expand(self.batch_size, 1, self.seq_len, self.seq_len).to(torch.bool)
            mask = torch.logical_and(mask, input_mask)
        else:
            mask = pad_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.seq_len).to(torch.bool)

        # Cross-attention mask: [bsz, 1, 1, enc_seq_len] - mask out padding in encoder output
        cross_attn_mask = encoder_mask[:, None, None, :].to(torch.bool)

        self_attn_k_caches, self_attn_v_caches = [], []
        cross_attn_k_caches, cross_attn_v_caches = [], []

        for layer in self.layers:
            hidden, sk, sv, ck, cv = layer(hidden, xa, last_pos=last_pos, mask=mask, cross_attn_mask=cross_attn_mask)
            self_attn_k_caches.append(sk)
            self_attn_v_caches.append(sv)
            cross_attn_k_caches.append(ck)
            cross_attn_v_caches.append(cv)

        hidden = self.layer_norm(hidden)
        # Tied embeddings: project back using embed_tokens weight
        logits = (hidden @ torch.transpose(self.embed_tokens.weight.to(hidden.dtype), 0, 1)).float()

        return logits, *self_attn_k_caches, *self_attn_v_caches, *cross_attn_k_caches, *cross_attn_v_caches


# ─── Model Instances (for ModelBuilder) ───

class BlenderbotEncoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        self.module = NeuronBlenderbotEncoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            max_pos=self.config.max_position_embeddings,
            n_head=self.config.encoder_attention_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            n_layer=self.config.encoder_layers,
            batch_size=self.neuron_config.batch_size,
            pad_token_id=self.config.pad_token_id,
            scale_embedding=getattr(self.config, 'scale_embedding', True),
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        return self.module, {}


class BlenderbotDecoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        self.module = NeuronBlenderbotDecoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            max_pos=self.config.max_position_embeddings,
            n_head=self.config.decoder_attention_heads,
            ffn_dim=self.config.decoder_ffn_dim,
            n_layer=self.config.decoder_layers,
            batch_size=self.neuron_config.batch_size,
            enc_seq_len=self.config.max_position_embeddings,
            pad_token_id=self.config.pad_token_id,
            scale_embedding=getattr(self.config, 'scale_embedding', True),
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        aliases = {}
        output_index = 1
        for layer in self.module.layers:
            aliases[layer.self_attn.cache_k] = output_index
            output_index += 1
        for layer in self.module.layers:
            aliases[layer.self_attn.cache_v] = output_index
            output_index += 1
        for layer in self.module.layers:
            aliases[layer.encoder_attn.cache_k] = output_index
            output_index += 1
        for layer in self.module.layers:
            aliases[layer.encoder_attn.cache_v] = output_index
            output_index += 1
        return self.module, aliases


# ─── Model Wrappers ───

class ModelWrapperBlenderbotEncoder(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        input_ids = torch.zeros((self.neuron_config.batch_size, self.config.max_position_embeddings), dtype=torch.int32)
        attention_mask = torch.ones((self.neuron_config.batch_size, self.config.max_position_embeddings), dtype=torch.int32)
        return [(input_ids, attention_mask)]

    def get_model_instance(self):
        return BlenderbotEncoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperBlenderbotDecoderPrefill(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        bs = self.neuron_config.batch_size
        dec_seq = self.config.max_position_embeddings
        enc_seq = self.config.max_position_embeddings
        tokens = torch.zeros((bs, dec_seq), dtype=torch.int32)
        encoder_out = torch.randn(bs, enc_seq, self.config.d_model, dtype=self.neuron_config.torch_dtype)
        last_pos = torch.zeros(1, dtype=torch.int32)
        pad_mask = torch.zeros((bs, dec_seq), dtype=torch.int32)
        encoder_mask = torch.ones((bs, enc_seq), dtype=torch.int32)
        return [(tokens, encoder_out, last_pos, pad_mask, encoder_mask)]

    def get_model_instance(self):
        return BlenderbotDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperBlenderbotDecoderDecode(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        bs = self.neuron_config.batch_size
        dec_seq = self.config.max_position_embeddings
        enc_seq = self.config.max_position_embeddings
        tokens = torch.zeros((bs, 1), dtype=torch.int32)
        encoder_out = torch.randn(bs, enc_seq, self.config.d_model, dtype=self.neuron_config.torch_dtype)
        last_pos = torch.zeros(1, dtype=torch.int32)
        pad_mask = torch.zeros((bs, dec_seq), dtype=torch.int32)
        encoder_mask = torch.ones((bs, enc_seq), dtype=torch.int32)
        return [(tokens, encoder_out, last_pos, pad_mask, encoder_mask)]

    def get_model_instance(self):
        return BlenderbotDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ─── Application Classes ───

class NeuronApplicationBlenderbotEncoder(NeuronApplicationBase):
    _model_cls = NeuronBlenderbotEncoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.encoder_model = ModelWrapperBlenderbotEncoder(
            config=self.config, model_cls=self._model_cls, tag="Encoder",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.encoder_model)

    def get_compiler_args(self):
        args = "--model-type=transformer"
        args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            args += " --auto-cast=none"
        return args

    @staticmethod
    def load_hf_model(model_path):
        from transformers import BlenderbotForConditionalGeneration
        return BlenderbotForConditionalGeneration.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return _convert_encoder_state_dict(state_dict)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.traced_model(input_ids.to(torch.int32), attention_mask.to(torch.int32)).to(self.config.neuron_config.torch_dtype)


class NeuronApplicationBlenderbotDecoder(NeuronApplicationBase):
    _model_cls = NeuronBlenderbotDecoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.decoder_prefill_model = ModelWrapperBlenderbotDecoderPrefill(
            config=self.config, model_cls=self._model_cls, tag="DecoderPrefill",
            compiler_args=self.get_compiler_args(),
        )
        self.decoder_decode_model = ModelWrapperBlenderbotDecoderDecode(
            config=self.config, model_cls=self._model_cls, tag="DecoderDecode",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.decoder_prefill_model)
        self.models.append(self.decoder_decode_model)

    def get_compiler_args(self):
        args = "--model-type=transformer"
        args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            args += " --auto-cast=none"
        return args

    @staticmethod
    def load_hf_model(model_path):
        from transformers import BlenderbotForConditionalGeneration
        return BlenderbotForConditionalGeneration.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        return _convert_decoder_state_dict(state_dict)

    def forward(self, text: torch.Tensor, encoder_out: torch.Tensor, last_pos: torch.Tensor,
                pad_mask: torch.Tensor, encoder_mask: torch.Tensor):
        """Dispatch to prefill or decode wrapper based on input shape."""
        if text.shape[1] > 1:
            return self.decoder_prefill_model(text, encoder_out, last_pos, pad_mask, encoder_mask)
        else:
            return self.decoder_decode_model(text, encoder_out, last_pos, pad_mask, encoder_mask)


class NeuronApplicationBlenderbot(nn.Module):
    """Top-level orchestrator: compiles/loads encoder and decoder separately.

    Both encoder and decoder model_paths point to the FULL HF model directory,
    so load_hf_model can load the complete model and convert_hf_to_neuron_state_dict
    extracts the relevant portion.
    """
    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__()
        self.config = config
        # model_path should contain encoder/ and decoder/ subdirectories
        # with pre-converted weights in new torch format
        self.encoder = NeuronApplicationBlenderbotEncoder(
            model_path=os.path.join(model_path, "encoder"), config=config, *args, **kwargs
        )
        self.decoder = NeuronApplicationBlenderbotDecoder(
            model_path=os.path.join(model_path, "decoder"), config=config, *args, **kwargs
        )

    def compile(self, compiled_model_path, *args, **kwargs):
        self.encoder.compile(os.path.join(compiled_model_path, "encoder"), *args, **kwargs)
        self.decoder.compile(os.path.join(compiled_model_path, "decoder"), *args, **kwargs)

    def load(self, compiled_model_path, *args, **kwargs):
        self.encoder.load(os.path.join(compiled_model_path, "encoder"), *args, **kwargs)
        self.decoder.load(os.path.join(compiled_model_path, "decoder"), *args, **kwargs)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 decoder_start_token_id: int = 1, eos_token_id: int = 2, pad_token_id: int = 0):
        """Autoregressive generation for seq2seq."""
        input_ids = input_ids.to(torch.int32)
        bs = input_ids.shape[0]
        enc_seq_len = self.config.max_position_embeddings

        # Create attention mask before padding (1 for real tokens, 0 for padding)
        input_len = input_ids.shape[1]
        attention_mask = torch.ones((bs, enc_seq_len), dtype=torch.int32)
        if input_len < enc_seq_len:
            attention_mask[:, input_len:] = 0
            input_ids = F.pad(input_ids, (0, enc_seq_len - input_len), value=pad_token_id)

        encoder_output = self.encoder(input_ids, attention_mask)

        dec_seq_len = self.config.max_position_embeddings

        # Start with decoder_start_token_id
        generated = [decoder_start_token_id]
        for step in range(max_new_tokens):
            if step == 0:
                # Prefill: pad tokens to full seq_len
                tokens = torch.full((bs, dec_seq_len), pad_token_id, dtype=torch.int32)
                tokens[0, 0] = decoder_start_token_id
                last_pos = torch.tensor([0], dtype=torch.int32)
                pad_mask = torch.zeros((bs, dec_seq_len), dtype=torch.int32)
                pad_mask[0, 0] = 1
            else:
                tokens = torch.tensor([[generated[-1]]], dtype=torch.int32)
                last_pos = torch.tensor([step], dtype=torch.int32)
                pad_mask = torch.zeros((bs, dec_seq_len), dtype=torch.int32)
                pad_mask[0, :step + 1] = 1

            outputs = self.decoder(tokens, encoder_output, last_pos, pad_mask, attention_mask)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
            if logits.dim() == 2:
                next_token_logits = logits
            else:
                next_token_logits = logits[:, 0, :] if step == 0 else logits[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            if next_token == eos_token_id:
                break

        return torch.tensor([generated], dtype=torch.long)


# ─── State Dict Conversion ───

def _convert_encoder_state_dict(state_dict: dict) -> dict:
    """Passthrough - weights are pre-converted during split step."""
    return state_dict


def _convert_decoder_state_dict(state_dict: dict) -> dict:
    """Passthrough - weights are pre-converted during split step."""
    return state_dict


def split_hf_weights(hf_model_path: str, output_path: str):
    """Split HF BlenderbotForConditionalGeneration weights into encoder/decoder.

    Creates output_path/encoder/ and output_path/decoder/ with:
    - config.json (copy from HF)
    - model.safetensors (converted and renamed weights)

    Key mapping:
    - model.encoder.* -> encoder (strip prefix, fc1/fc2 -> mlp.fc1/fc2)
    - model.decoder.* -> decoder (strip prefix, fc1/fc2 -> mlp.fc1/fc2)
    - model.shared.weight -> embed_tokens.weight (both)
    """
    import shutil
    from safetensors.torch import save_file

    from transformers import BlenderbotForConditionalGeneration
    print("Loading HF model for weight splitting...")
    hf_model = BlenderbotForConditionalGeneration.from_pretrained(hf_model_path)
    sd = hf_model.state_dict()
    del hf_model

    enc_dir = os.path.join(output_path, "encoder")
    dec_dir = os.path.join(output_path, "decoder")
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)

    # Copy config.json
    config_src = os.path.join(hf_model_path, "config.json")
    shutil.copy2(config_src, os.path.join(enc_dir, "config.json"))
    shutil.copy2(config_src, os.path.join(dec_dir, "config.json"))

    encoder_sd = {}
    decoder_sd = {}

    for key, value in sd.items():
        if key == "model.shared.weight":
            encoder_sd["embed_tokens.weight"] = value
            decoder_sd["embed_tokens.weight"] = value
        elif key.startswith("model.encoder."):
            new_key = key[len("model.encoder."):]
            parts = new_key.split(".")
            if len(parts) >= 3 and parts[0] == "layers" and parts[2] in ("fc1", "fc2"):
                new_key = f"layers.{parts[1]}.mlp.{parts[2]}.{'.'.join(parts[3:])}"
            encoder_sd[new_key] = value
        elif key.startswith("model.decoder."):
            new_key = key[len("model.decoder."):]
            parts = new_key.split(".")
            if len(parts) >= 3 and parts[0] == "layers" and parts[2] in ("fc1", "fc2"):
                new_key = f"layers.{parts[1]}.mlp.{parts[2]}.{'.'.join(parts[3:])}"
            decoder_sd[new_key] = value
        elif key in ("lm_head.weight", "final_logits_bias"):
            pass  # tied to shared embedding, skip
        else:
            print(f"  Unhandled key: {key}")

    print(f"  Encoder: {len(encoder_sd)} keys")
    print(f"  Decoder: {len(decoder_sd)} keys")

    save_file(encoder_sd, os.path.join(enc_dir, "model.safetensors"))
    save_file(decoder_sd, os.path.join(dec_dir, "model.safetensors"))
    print("Weight splitting done!")
