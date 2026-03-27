# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Voxtral Mini 3B for NxD Inference on AWS Neuron (Trainium2 / Inferentia2).

Architecture: Audio encoder (Whisper-like) + Projector + LLM backbone (Llama).

The audio encoder is traced with torch_neuronx.trace() and the projector runs
on CPU.  The LLM backbone reuses NxDI's existing NeuronLlamaModel /
ImageToTextModelWrapper with scatter-based embedding injection
(encode_vision_to_input).

Key classes
-----------
VoxtralTextModel       -- NeuronLlamaModel with audio scatter injection
VoxtralForCausalLM     -- NeuronBaseForImageToText wrapper (compile / load)
VoxtralInferenceConfig -- PixtralInferenceConfig builder (load_config helper)
NeuronApplicationVoxtral -- Top-level orchestrator: compile, load, generate
"""

import gc
import json
import logging
import os
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Audio constants
AUDIO_INTERMEDIATE_SIZE = 5120  # pack 4 * 1280 hidden states
AUDIO_TOKEN_ID = 24  # from Voxtral config.json


# ---------------------------------------------------------------------------
# NxDI model classes
# ---------------------------------------------------------------------------
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
)
from neuronx_distributed_inference.models.image_to_text_model_base import (
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.models.pixtral.modeling_pixtral import (
    PixtralInferenceConfig,
)
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
    scatter_by_index_put,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
)


class VoxtralTextModel(NeuronLlamaModel):
    """NeuronLlamaModel with audio embedding injection via scatter."""

    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


class VoxtralForCausalLM(NeuronBaseForImageToText):
    """Voxtral conditional generation model for Neuron.

    Uses the existing ImageToTextModelWrapper for the text decoder.
    The audio encoder and projector are handled externally by
    NeuronApplicationVoxtral.
    """

    text_model_cls = VoxtralTextModel
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_cls = None
    vision_model_wrapper = None

    # Set dynamically before construction based on model config.
    _seq_len: int = 2048
    _n_positions: int = 4096
    _text_hidden_size: int = 3072
    _batch_size: int = 1

    # ------------------------------------------------------------------
    # State dict conversion
    # ------------------------------------------------------------------
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Delegate to NeuronLlamaForCausalLM with text_config."""
        return NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, config.text_config
        )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, model_path, inference_config, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            model_path,
            inference_config,
            *args,
            **kwargs,
        )

    def enable_vision_encoder(self, **kwargs):
        """No-op: audio encoder is traced separately."""
        pass

    # ------------------------------------------------------------------
    # Compile / Load (text decoder only)
    # ------------------------------------------------------------------
    def compile(self, compiled_model_path, debug=False, dry_run=False):
        self.config.save(compiled_model_path)
        text_path = os.path.join(compiled_model_path, "text_model") + "/"
        os.makedirs(text_path, exist_ok=True)
        text_traced_model = self.get_text_builder(debug).trace(
            initialize_model_weights=False, dry_run=dry_run
        )
        if not dry_run:
            torch.jit.save(text_traced_model, text_path + "model.pt")
            del text_traced_model
            logger.info("Finished compiling text model!")
        self._save_configs_to_compiler_workdir()
        if dry_run:
            return
        self.shard_text_weights(text_path, debug)
        logger.info("Finished sharding text weights!")
        self.is_compiled = True

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        text_path = os.path.join(compiled_model_path, "text_model") + "/"
        self.text_traced_model = torch.jit.load(text_path + "model.pt")
        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        logger.info("Sharding weights on load...")
        text_weights = self.get_text_builder().shard_checkpoint()
        start_rank_tensor = torch.tensor(
            [start_rank_id], dtype=torch.int32, device="cpu"
        )
        self.text_traced_model.nxd_model.initialize(text_weights, start_rank_tensor)
        logger.info("Finished text weights loading")
        for model_wrapper in self.text_models:
            model_wrapper.model = self.text_traced_model
        self.is_loaded_to_neuron = True
        if not self.neuron_config.skip_warmup and not skip_warmup:
            self.warmup()
        else:
            logger.info("Skipping model warmup")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    @classmethod
    def get_config_cls(cls):
        return PixtralInferenceConfig

    # ------------------------------------------------------------------
    # Compiler args
    # ------------------------------------------------------------------
    def get_compiler_args(self):
        """Return Neuron compiler flags. Auto-detects trn2 for --lnc=2."""
        try:
            result = subprocess.run(
                ["neuron-ls", "--json-output"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "trn2" in result.stdout.lower():
                lnc_flag = "--lnc=2 "
            else:
                lnc_flag = ""
        except Exception:
            lnc_flag = ""

        return (
            "--auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma ' "
            f"{lnc_flag}-O1 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------
    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden,
        adapter_ids,
        vision_embeddings,
        vision_mask,
        deepstack_vision_embeds,
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
        rotary_position_ids=None,
    ):
        if rotary_position_ids is None:
            rotary_position_ids = torch.empty(0)

        # When called without audio (text-only), provide zero tensors.
        if vision_embeddings is None:
            vision_embeddings = torch.zeros(
                (self._batch_size, self._seq_len, self._text_hidden_size),
                dtype=torch.bfloat16,
            )
        if vision_mask is None:
            vision_mask = torch.full(
                (self._batch_size, self._seq_len, 1),
                fill_value=self._seq_len - 1,
                dtype=torch.int32,
            )

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                *[torch.empty(0) for _ in range(16)],
                rotary_position_ids,
                vision_embeddings,
                vision_mask,
            )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                *[torch.empty(0) for _ in range(16)],
                rotary_position_ids,
                torch.empty(0, dtype=torch.bfloat16),
                torch.empty(0, dtype=torch.bool),
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def get_required_kwargs(self):
        return ["vision_embeddings", "vision_mask"]


# ---------------------------------------------------------------------------
# Inference config builder
# ---------------------------------------------------------------------------
class VoxtralInferenceConfig:
    """Helper to build a PixtralInferenceConfig for Voxtral."""

    def __init__(
        self,
        text_model_path: str,
        full_config: dict,
        tp_degree: int = 1,
        batch_size: int = 1,
        seq_len: int = 2048,
        n_positions: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.text_model_path = text_model_path
        self.full_config = full_config
        self.tp_degree = tp_degree
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_positions = n_positions
        self.dtype = dtype

    def build(self) -> PixtralInferenceConfig:
        """Build and return the PixtralInferenceConfig."""
        text_cfg = self.full_config
        text_hidden_size = text_cfg["hidden_size"]
        audio_token_id = self.full_config.get("audio_token_id", AUDIO_TOKEN_ID)

        text_neuron_config = NeuronConfig(
            tp_degree=self.tp_degree,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            n_positions=self.n_positions,
            torch_dtype=self.dtype,
            on_device_sampling_config=None,
            enable_bucketing=False,
            flash_decoding_enabled=False,
            fused_qkv=True,
        )
        vision_neuron_config = NeuronConfig(
            tp_degree=self.tp_degree,
            batch_size=self.batch_size,
            seq_len=1500,
            torch_dtype=self.dtype,
            enable_bucketing=False,
            on_device_sampling_config=None,
        )

        text_model_path = self.text_model_path

        def load_config(config_obj):
            config_obj.text_config = SimpleNamespace(
                hidden_size=text_cfg["hidden_size"],
                num_attention_heads=text_cfg["num_attention_heads"],
                num_hidden_layers=text_cfg["num_hidden_layers"],
                num_key_value_heads=text_cfg["num_key_value_heads"],
                vocab_size=text_cfg["vocab_size"],
                max_position_embeddings=text_cfg.get("max_position_embeddings", 131072),
                rope_theta=text_cfg.get("rope_theta", 1e8),
                rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-5),
                hidden_act=text_cfg.get("hidden_act", "silu"),
                intermediate_size=text_cfg.get("intermediate_size", 8192),
                head_dim=text_cfg.get("head_dim", 128),
                sliding_window=text_cfg.get("sliding_window", None),
                tie_word_embeddings=text_cfg.get("tie_word_embeddings", False),
                pad_token_id=0,
                bos_token_id=text_cfg.get("bos_token_id", 1),
                eos_token_id=text_cfg.get("eos_token_id", 2),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            config_obj.vision_config = SimpleNamespace(
                hidden_size=text_hidden_size,
                image_size=1024,
                patch_size=16,
                num_hidden_layers=1,
                num_channels=3,
                num_attention_heads=8,
                rope_theta=10000.0,
                head_dim=64,
                intermediate_size=4096,
                hidden_act="silu",
            )
            config_obj._name_or_path = text_model_path
            config_obj.multimodal_projector_bias = False
            config_obj.projector_hidden_act = "gelu"
            config_obj.vision_feature_layer = -1
            config_obj.output_attentions = False
            config_obj.output_hidden_states = False
            config_obj.return_dict = True
            config_obj.tie_word_embeddings = False
            config_obj.image_token_index = audio_token_id

        return PixtralInferenceConfig(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            load_config=load_config,
        )


# ---------------------------------------------------------------------------
# Top-level application class
# ---------------------------------------------------------------------------
class NeuronApplicationVoxtral:
    """Top-level orchestrator for Voxtral on Neuron.

    Manages three components:
    1. Audio encoder  -- torch_neuronx traced NEFF
    2. Projector      -- CPU nn.Module
    3. Text decoder   -- NxDI VoxtralForCausalLM (ImageToTextModelWrapper)

    Usage::

        app = NeuronApplicationVoxtral(
            model_path="/path/to/voxtral-mini-3B",
            tp_degree=1,
            seq_len=2048,
            n_positions=4096,
        )
        app.compile("/path/to/compiled")
        app.load("/path/to/compiled")

        # Text-only
        text = app.generate_text("What is the capital of France?")

        # Audio + text
        text = app.generate_with_audio(input_ids, attention_mask, input_features)
    """

    def __init__(
        self,
        model_path: str,
        tp_degree: int = 1,
        batch_size: int = 1,
        seq_len: int = 2048,
        n_positions: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.tp_degree = tp_degree
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_positions = n_positions
        self.dtype = dtype

        # Read full model config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            self.full_config = json.load(f)

        self.text_hidden_size = self.full_config["text_config"]["hidden_size"]
        self.audio_token_id = self.full_config.get("audio_token_id", AUDIO_TOKEN_ID)

        # Component references (populated by load)
        self.audio_encoder = None
        self.projector = None
        self.vl_model = None
        self.adapter = None
        self.tokenizer = None
        self.processor = None

        # Paths
        self._text_model_path = None

    # ------------------------------------------------------------------
    # Text weight extraction
    # ------------------------------------------------------------------
    def _ensure_text_weights(self) -> str:
        """Extract text-only backbone weights if needed. Returns text_model_path."""
        text_model_path = os.path.join(self.model_path, "text_only")
        if os.path.exists(text_model_path):
            self._text_model_path = text_model_path
            return text_model_path

        logger.info("Extracting text-only backbone weights...")
        from transformers import VoxtralForConditionalGeneration

        t0 = time.time()
        full_model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_path, dtype=self.dtype
        )
        os.makedirs(text_model_path, exist_ok=True)
        full_model.language_model.save_pretrained(text_model_path)

        # Copy tokenizer / processor files
        for fname in [
            "tekken.json",
            "params.json",
            "preprocessor_config.json",
            "generation_config.json",
        ]:
            src = os.path.join(self.model_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, text_model_path)

        del full_model
        gc.collect()
        logger.info(f"Text extraction done in {time.time() - t0:.1f}s")

        self._text_model_path = text_model_path
        return text_model_path

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(self, compiled_path: str):
        """Compile all components (audio encoder + text decoder).

        Parameters
        ----------
        compiled_path : str
            Directory where compiled NEFFs will be stored.
        """
        import torch_neuronx

        text_model_path = self._ensure_text_weights()
        audio_encoder_path = os.path.join(compiled_path, "audio_encoder.pt")
        text_decoder_path = os.path.join(compiled_path, "text_decoder")

        # --- Audio encoder ---
        if not os.path.exists(audio_encoder_path):
            logger.info("Compiling audio encoder...")
            from transformers import VoxtralForConditionalGeneration

            t0 = time.time()
            full_model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_path, dtype=self.dtype
            )
            enc = full_model.audio_tower
            enc.eval()
            del full_model.language_model
            del full_model.multi_modal_projector
            del full_model
            gc.collect()

            example_input = torch.randn(1, 128, 3000, dtype=self.dtype)
            traced = torch_neuronx.trace(
                enc,
                example_input,
                compiler_args=[
                    "--auto-cast",
                    "matmult",
                    "--model-type=transformer",
                ],
                inline_weights_to_neff=False,
            )
            os.makedirs(os.path.dirname(audio_encoder_path), exist_ok=True)
            torch_neuronx.async_load(traced)
            torch.jit.save(traced, audio_encoder_path)
            logger.info(f"Audio encoder compiled in {time.time() - t0:.1f}s")
            del enc, traced
            gc.collect()
        else:
            logger.info(f"Audio encoder already compiled at {audio_encoder_path}")

        # --- Text decoder ---
        if not os.path.exists(
            os.path.join(text_decoder_path, "text_model", "model.pt")
        ):
            logger.info("Compiling text decoder...")
            t0 = time.time()
            vl_model = self._build_vl_model(text_model_path)
            os.makedirs(text_decoder_path, exist_ok=True)
            vl_model.compile(text_decoder_path)
            logger.info(f"Text decoder compiled in {time.time() - t0:.1f}s")
            del vl_model
            gc.collect()
        else:
            logger.info(f"Text decoder already compiled at {text_decoder_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, compiled_path: str):
        """Load all compiled components.

        Parameters
        ----------
        compiled_path : str
            Directory with compiled NEFFs (same path passed to compile()).
        """
        import torch_neuronx

        text_model_path = self._ensure_text_weights()
        audio_encoder_path = os.path.join(compiled_path, "audio_encoder.pt")
        text_decoder_path = os.path.join(compiled_path, "text_decoder")

        # Audio encoder
        logger.info(f"Loading audio encoder from {audio_encoder_path}")
        self.audio_encoder = torch.jit.load(audio_encoder_path)
        torch_neuronx.async_load(self.audio_encoder)

        # Projector (CPU)
        logger.info("Loading projector (CPU)...")
        from transformers import VoxtralForConditionalGeneration

        t0 = time.time()
        full_model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_path, dtype=self.dtype
        )
        self.projector = full_model.multi_modal_projector
        self.projector.eval()
        del full_model.audio_tower
        del full_model.language_model
        del full_model
        gc.collect()
        logger.info(f"Projector loaded in {time.time() - t0:.1f}s")

        # Text decoder
        logger.info(f"Loading text decoder from {text_decoder_path}")
        self.vl_model = self._build_vl_model(text_model_path)
        self.vl_model.load(text_decoder_path)

        # Adapter, tokenizer, processor
        self.adapter = HuggingFaceGenerationAdapter(self.vl_model)

        from transformers import AutoProcessor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        logger.info("All model components loaded successfully!")

    # ------------------------------------------------------------------
    # Audio pipeline
    # ------------------------------------------------------------------
    def run_audio_pipeline(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run audio through encoder -> reshape -> projector.

        Parameters
        ----------
        input_features : torch.Tensor
            Mel spectrogram of shape ``[1, 128, 3000]``.

        Returns
        -------
        torch.Tensor
            Projected embeddings of shape ``[1, 375, hidden_size]``.
        """
        with torch.no_grad():
            enc_output = self.audio_encoder(input_features.to(self.dtype))

        if isinstance(enc_output, dict):
            audio_hidden = enc_output.get(
                "last_hidden_state", list(enc_output.values())[0]
            )
        elif isinstance(enc_output, tuple):
            audio_hidden = enc_output[0]
        else:
            audio_hidden = enc_output

        # Reshape: [1, 1500, 1280] -> [375, 5120]
        audio_hidden_flat = audio_hidden.reshape(-1, AUDIO_INTERMEDIATE_SIZE)

        # Projector on CPU
        with torch.no_grad():
            audio_embeds = self.projector(audio_hidden_flat)

        return audio_embeds

    # ------------------------------------------------------------------
    # Generation: audio + text
    # ------------------------------------------------------------------
    def generate_with_audio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        max_new_tokens: int = 200,
    ) -> torch.Tensor:
        """Full pipeline: audio encode + scatter + text decode.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs from ``processor.apply_chat_template()``.
        attention_mask : torch.Tensor
            Attention mask matching input_ids.
        input_features : torch.Tensor
            Mel spectrogram ``[1, 128, 3000]``.
        max_new_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        torch.Tensor
            Full output token IDs (prompt + generated).
        """
        audio_embeds = self.run_audio_pipeline(input_features)

        # Prepare scatter args
        audio_embeds_3d = audio_embeds.unsqueeze(0).to(self.dtype)
        audio_positions = (input_ids == self.audio_token_id).squeeze(0)
        vision_mask = generate_positions_from_mask(audio_positions)
        vision_mask_padded = pad_positions(vision_mask, self.seq_len, self.seq_len - 1)
        audio_embeds_padded = pad_vision_embeddings(audio_embeds_3d, self.seq_len)

        output_ids = self.adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            vision_embeddings=audio_embeds_padded,
            vision_mask=vision_mask_padded,
        )
        return output_ids

    # ------------------------------------------------------------------
    # Generation: text only
    # ------------------------------------------------------------------
    def generate_text_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 200,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Text-only generation (no audio).

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs.
        attention_mask : torch.Tensor
            Attention mask.
        max_new_tokens : int
            Maximum tokens to generate.
        do_sample : bool
            Whether to sample.

        Returns
        -------
        torch.Tensor
            Full output token IDs (prompt + generated).
        """
        output_ids = self.adapter.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return output_ids

    # ------------------------------------------------------------------
    # Convenience: generate from text string
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        do_sample: bool = False,
    ) -> str:
        """Generate text from a plain text prompt.

        Parameters
        ----------
        prompt : str
            User text prompt.
        max_new_tokens : int
            Maximum tokens to generate.
        do_sample : bool
            Whether to sample.

        Returns
        -------
        str
            Generated text.
        """
        conversation = [{"role": "user", "content": prompt}]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_tensors="pt"
        )
        if hasattr(inputs, "input_ids"):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = inputs
            attention_mask = torch.ones_like(input_ids)

        seq_len_actual = input_ids.shape[1]
        output_ids = self.generate_text_only(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        new_tokens = output_ids[0, seq_len_actual:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Convenience: transcribe audio file
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_url: str,
        prompt: str = "Transcribe this audio.",
        max_new_tokens: int = 500,
    ) -> str:
        """Transcribe an audio file.

        Parameters
        ----------
        audio_url : str
            Path or URL to an audio file.
        prompt : str
            Transcription instruction.
        max_new_tokens : int
            Maximum tokens to generate.

        Returns
        -------
        str
            Transcription text.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": audio_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_features = inputs["input_features"]
        seq_len_actual = input_ids.shape[1]

        output_ids = self.generate_with_audio(
            input_ids, attention_mask, input_features, max_new_tokens=max_new_tokens
        )
        new_tokens = output_ids[0, seq_len_actual:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vl_model(self, text_model_path: str) -> VoxtralForCausalLM:
        """Build VoxtralForCausalLM with the correct config."""
        # Read text config
        with open(os.path.join(text_model_path, "config.json")) as f:
            text_cfg = json.load(f)

        # Set class-level attributes
        VoxtralForCausalLM._seq_len = self.seq_len
        VoxtralForCausalLM._n_positions = self.n_positions
        VoxtralForCausalLM._text_hidden_size = self.text_hidden_size
        VoxtralForCausalLM._batch_size = self.batch_size

        config_builder = VoxtralInferenceConfig(
            text_model_path=text_model_path,
            full_config=text_cfg,
            tp_degree=self.tp_degree,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            n_positions=self.n_positions,
            dtype=self.dtype,
        )
        pixtral_config = config_builder.build()
        return VoxtralForCausalLM(text_model_path, pixtral_config)
