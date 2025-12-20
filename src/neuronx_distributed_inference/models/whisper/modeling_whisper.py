# Modified from https://github.com/openai/whisper/blob/main/whisper/model.py

import math
import os
from typing import Optional, Iterable, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import BaseModelInstance, ModelWrapper
from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.whisper.utils.config import get_dims_from_config
from neuronx_distributed_inference.models.whisper.utils.decoding import decode as decode_function
from neuronx_distributed_inference.models.whisper.utils.state_dict import (
    convert_hf_state_dict_to_neuron,
    expand_state_dict,
)

from transformers import WhisperModel
from transformers.models.whisper.modeling_whisper import sinusoids
from whisper import Whisper


def ceil_div(a: int, b: int) -> int:
    """Integer division with ceiling."""
    return -(-a // b)


class WhisperInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = get_dims_from_config(self)


class LayerNorm(nn.LayerNorm):
    """
    Converts input to float32 before applying LayerNorm to avoid precision issues.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class NeuronMLP(torch.nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=True, gather_output=False, dtype=dtype)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=True, input_is_parallel=True, dtype=dtype
        )

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class NeuronAttention(nn.Module):

    def __init__(
        self, n_state: int, n_head: int, batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32, kvcache=True
    ):
        super().__init__()

        assert n_state % n_head == 0, f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        self.head_dim = n_state // n_head

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"
        tp_degree = parallel_state.get_tensor_model_parallel_group().size()

        # head per core
        self.n_heads = ceil_div(n_head, tp_degree)
        self.n_kv_heads = self.n_heads  # Whisper doesn't use GQA

        self.query = ColumnParallelLinear(
            n_state,
            self.n_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.key = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=False,  # No bias for key projection
            gather_output=False,
            dtype=dtype,
        )
        self.value = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.out = RowParallelLinear(
            self.n_heads * tp_degree * self.head_dim,
            n_state,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.cache_k = (
            nn.Parameter(
                torch.zeros((batch_size, self.n_kv_heads, seq_len, self.head_dim), dtype=dtype), requires_grad=False
            )
            if kvcache
            else None
        )
        self.cache_v = (
            nn.Parameter(
                torch.zeros((batch_size, self.n_kv_heads, seq_len, self.head_dim), dtype=dtype), requires_grad=False
            )
            if kvcache
            else None
        )

    def forward(
        self,
        x: Tensor,
        last_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        bsz, seq_len, hidden_dim = x.shape

        # bs, head, seqlen, head_dim
        q = self.query(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.cache_k is not None and self.cache_v is not None:
            if seq_len > 1:  # prefill: save all to cache
                indices = torch.arange(start=0, end=seq_len, dtype=torch.int64, device=q.device)
                indices = indices.view(1, 1, seq_len, 1)
                indices = indices.expand(bsz, self.n_kv_heads, seq_len, self.head_dim)
            else:  # decode: save only the last token [last_pos] to cache
                indices = last_pos.view(bsz, 1, 1, 1).expand_as(k).to(torch.int64)

            updated_kcache = torch.scatter(self.cache_k, 2, indices, k)
            updated_vcache = torch.scatter(self.cache_v, 2, indices, v)

            k = updated_kcache
            v = updated_vcache

        # Q.K^T/√d
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = torch.where(mask, scores, torch.finfo(scores.dtype).min)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        if self.cache_k is not None and self.cache_v is not None:
            return self.out(output), updated_kcache, updated_vcache
        else:
            return self.out(output)


class NeuronCrossAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int, batch_size: int, kv_seq_len: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        assert n_state % n_head == 0, f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        self.head_dim = n_state // n_head

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"
        tp_degree = parallel_state.get_tensor_model_parallel_group().size()

        # head per core
        self.n_heads = ceil_div(n_head, tp_degree)
        self.n_kv_heads = self.n_heads  # Whisper doesn't use GQA

        self.query = ColumnParallelLinear(
            n_state,
            self.n_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.key = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=False,  # No bias for key projection
            gather_output=False,
            dtype=dtype,
        )
        self.value = ColumnParallelLinear(
            n_state,
            self.n_kv_heads * tp_degree * self.head_dim,
            bias=True,
            gather_output=False,
            dtype=dtype,
        )
        self.out = RowParallelLinear(
            self.n_heads * tp_degree * self.head_dim,
            n_state,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
        )

        self.cache_k = nn.Parameter(
            torch.zeros((batch_size, self.n_kv_heads, kv_seq_len, self.head_dim), dtype=dtype), requires_grad=False
        )
        self.cache_v = nn.Parameter(
            torch.zeros((batch_size, self.n_kv_heads, kv_seq_len, self.head_dim), dtype=dtype), requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        is_prefill: bool = True,
    ):
        bsz, seq_len, hidden_dim = x.shape
        kv_seq_len = xa.shape[1]

        # bs, head, seqlen, head_dim
        q = self.query(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(xa).view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(xa).view(bsz, kv_seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Save KV Cache
        if is_prefill:
            indices = torch.arange(start=0, end=kv_seq_len, dtype=torch.int64, device=q.device)
            indices = indices.view(1, 1, kv_seq_len, 1)
            indices = indices.expand(bsz, self.n_kv_heads, kv_seq_len, self.head_dim)

            updated_kcache = torch.scatter(self.cache_k, 2, indices, k)
            updated_vcache = torch.scatter(self.cache_v, 2, indices, v)
        else:
            updated_kcache = self.cache_k
            updated_vcache = self.cache_v

        k = updated_kcache
        v = updated_vcache

        # Q.K^T/√d
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out(output), updated_kcache, updated_vcache


class NeuronResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        batch_size: int,
        seq_len: int,
        cross_attention: bool = False,
        cross_attn_seq_len: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.attn = NeuronAttention(n_state, n_head, batch_size, seq_len, dtype=dtype, kvcache=cross_attention)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            NeuronCrossAttention(n_state, n_head, batch_size, cross_attn_seq_len, dtype=dtype)
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = NeuronMLP(n_state, n_mlp, dtype=dtype)
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,  # "a" for audio
        last_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        if self.cross_attn:
            h, self_attn_cache_k, self_attn_cache_v = self.attn(self.attn_ln(x), last_pos=last_pos, mask=mask)
        else:
            h = self.attn(self.attn_ln(x), last_pos=last_pos, mask=mask)
        x = x + h
        if self.cross_attn:
            h, cross_attn_cache_k, cross_attn_cache_v = self.cross_attn(
                self.cross_attn_ln(x), xa, is_prefill=x.shape[1] > 1
            )
            x = x + h
        x = x + self.mlp(self.mlp_ln(x))

        if self.cross_attn:
            return x, self_attn_cache_k, self_attn_cache_v, cross_attn_cache_k, cross_attn_cache_v
        else:
            return x


class NeuronAudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        seq_len = n_ctx
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1, dtype=dtype)
        self.positional_embedding = nn.Parameter(sinusoids(n_ctx, n_state), requires_grad=False)

        self.blocks: Iterable[NeuronResidualAttentionBlock] = nn.ModuleList(
            [NeuronResidualAttentionBlock(n_state, n_head, batch_size, seq_len, dtype=dtype) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class NeuronTextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_text_ctx: int,
        n_audio_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = n_text_ctx
        self.vocab_size = n_vocab

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Embedding(n_text_ctx, n_state)

        self.blocks: Iterable[NeuronResidualAttentionBlock] = nn.ModuleList(
            [
                NeuronResidualAttentionBlock(
                    n_state,
                    n_head,
                    self.batch_size,
                    self.seq_len,
                    cross_attention=True,
                    cross_attn_seq_len=n_audio_ctx,
                    dtype=dtype,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Tensor, last_pos: torch.Tensor, pad_mask: torch.Tensor):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        last_pos : torch.Tensor, shape = (batch_size,)
            indices of the last valid token position for each sequence in the batch
        pad_mask : torch.Tensor, shape = (batch_size, n_ctx)
            boolean mask indicating valid positions (True) vs padded positions (False)
        """
        assert (
            x.shape[1] == 1 or x.shape[1] == self.seq_len
        ), f"Input sequence length {x.shape[1]} must be 1 (decode) or {self.seq_len} (prefill)"

        is_prefill = x.shape[1] > 1
        if is_prefill:
            pe = self.positional_embedding.weight
        else:
            pe = self.positional_embedding(last_pos)  # TODO: check if it's correct when batch_size > 1
        x = self.token_embedding(x) + pe
        x = x.to(xa.dtype)

        mask = None
        if is_prefill:
            mask = torch.full((self.seq_len, self.seq_len), True, device=pad_mask.device).tril(diagonal=0)
            input_mask = (
                pad_mask[:, None, None, :].expand(self.batch_size, 1, self.seq_len, self.seq_len).to(torch.bool)
            )
            mask = torch.logical_and(mask, input_mask)
        else:
            mask = pad_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.seq_len).to(torch.bool)

        self_attn_k_caches = []
        self_attn_v_caches = []
        cross_attn_k_caches = []
        cross_attn_v_caches = []

        for block in self.blocks:
            x, sk, sv, ck, cv = block(x, xa, last_pos=last_pos, mask=mask)
            self_attn_k_caches.append(sk)
            self_attn_v_caches.append(sv)
            cross_attn_k_caches.append(ck)
            cross_attn_v_caches.append(cv)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, *self_attn_k_caches, *self_attn_v_caches, *cross_attn_k_caches, *cross_attn_v_caches


class WhisperModelEncoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        dims = self.config.dims
        self.module = NeuronAudioEncoder(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
            batch_size=self.neuron_config.batch_size,
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        aliases = {}
        return self.module, aliases


class WhisperModelDecoderInstance(BaseModelInstance):
    def __init__(self, config):
        self.module = None
        self.config = config
        self.neuron_config = config.neuron_config

    def load_module(self):
        dims = self.config.dims
        self.module = NeuronTextDecoder(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_audio_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer,
            batch_size=self.neuron_config.batch_size,
            dtype=self.neuron_config.torch_dtype,
        )

    def get(self, bucket_rank, **kwargs):
        aliases = {}
        output_index = 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.attn.cache_k] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.attn.cache_v] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.cross_attn.cache_k] = output_index
            output_index = output_index + 1
        for i, layer in enumerate(self.module.blocks):
            aliases[layer.cross_attn.cache_v] = output_index
            output_index = output_index + 1
        return self.module, aliases


class ModelWrapperWhisperEncoder(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None  # Set to None if no bucketing needed

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing
        audio = torch.randn(
            self.neuron_config.batch_size,
            self.config.dims.n_mels,
            self.config.dims.n_audio_ctx * 2,
            dtype=self.neuron_config.torch_dtype,
        )
        inputs = [(audio,)]
        return inputs

    def get_model_instance(self):
        return WhisperModelEncoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperWhisperDecoderPrefill(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None  # Set to None if no bucketing needed

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing
        audio_embed = torch.randn(
            self.neuron_config.batch_size,
            self.config.dims.n_audio_ctx,
            self.config.dims.n_audio_state,
            dtype=self.neuron_config.torch_dtype,
        )
        padded_tokens = torch.zeros((self.neuron_config.batch_size, self.config.dims.n_text_ctx), dtype=torch.int32)
        last_pos = torch.zeros(1, dtype=torch.int32)
        pad_mask = torch.zeros((self.neuron_config.batch_size, self.config.dims.n_text_ctx), dtype=torch.int32)
        inputs = [
            (padded_tokens, audio_embed, last_pos, pad_mask),
        ]
        return inputs

    def get_model_instance(self):
        return WhisperModelDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class ModelWrapperWhisperDecoderDecode(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None, model_init_kwargs={}):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs)
        self.bucket_config = None  # Set to None if no bucketing needed

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        # Generate example inputs for tracing
        audio_embed = torch.randn(
            self.neuron_config.batch_size,
            self.config.dims.n_audio_ctx,
            self.config.dims.n_audio_state,
            dtype=self.neuron_config.torch_dtype,
        )
        padded_tokens = torch.zeros((self.neuron_config.batch_size, 1), dtype=torch.int32)
        last_pos = torch.zeros(1, dtype=torch.int32)
        pad_mask = torch.zeros((self.neuron_config.batch_size, self.config.dims.n_text_ctx), dtype=torch.int32)
        inputs = [
            (padded_tokens, audio_embed, last_pos, pad_mask),
        ]
        return inputs

    def get_model_instance(self):
        return WhisperModelDecoderInstance(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NeuronApplicationWhisperEncoder(NeuronApplicationBase):
    _model_cls = NeuronAudioEncoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.dims = config.dims
        self.encoder_model = ModelWrapperWhisperEncoder(
            config=self.config,
            model_cls=self._model_cls,
            tag="Encoder",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.encoder_model)

        # workaround for whisper PyTorchInference init, dummy blocks
        self.blocks = []

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        return compiler_args

    @staticmethod
    def load_hf_model(model_path):
        return WhisperModel.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: WhisperInferenceConfig) -> dict:
        state_dict = convert_hf_state_dict_to_neuron(state_dict, type="encoder")
        state_dict = expand_state_dict(state_dict, config.dims, config.neuron_config.tp_degree)
        return state_dict

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Whisper encoder.
        :param audio: Tensor of shape (batch_size, n_mels, n_audio_ctx)
        :return: Encoded audio features
        """
        # Ensure audio is in the correct dtype, return in the original dtype
        return self.traced_model(audio.to(self.config.neuron_config.torch_dtype)).to(audio.dtype)


class NeuronApplicationWhisperDecoder(NeuronApplicationBase):
    _model_cls = NeuronTextDecoder

    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(model_path, config, *args, **kwargs)
        self.dims = config.dims
        self.decoder_prefill_model = ModelWrapperWhisperDecoderPrefill(
            config=self.config,
            model_cls=self._model_cls,
            tag="DecoderPrefill",
            compiler_args=self.get_compiler_args(),
        )
        self.decoder_decode_model = ModelWrapperWhisperDecoderDecode(
            config=self.config,
            model_cls=self._model_cls,
            tag="DecoderDecode",
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.decoder_prefill_model)
        self.models.append(self.decoder_decode_model)

        # workaround for whisper PyTorchInference init, dummy blocks
        self.blocks = []

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        if self.config.neuron_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        return compiler_args

    @staticmethod
    def load_hf_model(model_path):
        return WhisperModel.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: WhisperInferenceConfig) -> dict:
        state_dict = convert_hf_state_dict_to_neuron(state_dict, type="decoder")
        state_dict = expand_state_dict(state_dict, config.dims, config.neuron_config.tp_degree)
        return state_dict

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor, last_pos: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Whisper decoder.
        :param text: Tensor of shape (batch_size, <= n_text_ctx)
        :param audio: Encoded audio features of shape (batch_size, n_audio_ctx, n_audio_state)
        :param last_pos: Tensor of shape (1,) indicating the last valid token position
        :param pad_mask: Tensor of shape (batch_size, n_text_ctx) indicating valid positions
        :return: Logits for the next token prediction
        """
        return self.traced_model(text, audio, last_pos, pad_mask)


class NeuronApplicationWhisper(Whisper):
    def __init__(self, model_path, config, *args, **kwargs):
        super().__init__(config.dims)
        self.config = config
        self.dims = config.dims
        self.encoder_path_suffix = "encoder"
        self.decoder_path_suffix = "decoder"
        self.encoder = NeuronApplicationWhisperEncoder(
            model_path=os.path.join(model_path, self.encoder_path_suffix), config=config, *args, **kwargs
        )
        self.decoder = NeuronApplicationWhisperDecoder(
            model_path=os.path.join(model_path, self.decoder_path_suffix), config=config, *args, **kwargs
        )

    def compile(self, compiled_model_path, *args, **kwargs):
        self.encoder.compile(os.path.join(compiled_model_path, self.encoder_path_suffix), *args, **kwargs)
        self.decoder.compile(os.path.join(compiled_model_path, self.decoder_path_suffix), *args, **kwargs)

    def load(self, compiled_model_path, *args, **kwargs):
        self.encoder.load(os.path.join(compiled_model_path, self.encoder_path_suffix), *args, **kwargs)
        self.decoder.load(os.path.join(compiled_model_path, self.decoder_path_suffix), *args, **kwargs)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        tokens = tokens.to(torch.int32)
        padded_tokens, last_pos, pad_mask = self._prepare_decoder_inputs(tokens)
        return self.decoder(
            padded_tokens, audio_features.to(self.config.neuron_config.torch_dtype), last_pos, pad_mask
        )[:, : last_pos + 1]

    def _prepare_decoder_inputs(self, tokens: torch.Tensor):
        pad_token = -1
        last_pos = torch.tensor([len(prompt) - 1 for prompt in tokens], dtype=torch.int32)
        padded_tokens = F.pad(tokens, (0, self.dims.n_text_ctx - tokens.shape[1]), value=pad_token)
        pad_mask = torch.where(padded_tokens != pad_token, 1, 0).to(torch.int32)
        padded_tokens = torch.where(padded_tokens == pad_token, 0, padded_tokens)
        return padded_tokens, last_pos, pad_mask

    @property
    def device(self):
        return torch.device("cpu")

    decode = decode_function
