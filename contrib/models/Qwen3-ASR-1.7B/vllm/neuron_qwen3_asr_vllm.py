class NeuronQwen3ASRForCausalLM(NeuronMultiModalCausalLM):
    """Qwen3-ASR-1.7B audio-language model for Neuron.

    Uses the NxDI NeuronQwen3VLForCausalLM text decoder (same Qwen3 architecture
    with mRoPE and QK-norm) with separately traced Whisper encoder NEFFs.

    Audio flow:
    - execute_model() extracts mel features from multi_modal_kwargs
    - Selects appropriate encoder bucket (5s/10s/30s)
    - Runs traced encoder NEFF to get audio embeddings
    - Constructs vision_embeddings/vision_mask for CTE scatter
    - forward() passes to NxDI CTE (prefill) or TKG (decode)
    """

    AUDIO_TOKEN_ID = 151676  # <|audio_pad|> placeholder token

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.encoders = {}  # bucket_T -> traced NEFF
        self._rope_deltas = None

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        """Load pre-compiled Qwen3-ASR text decoder + traced encoder NEFFs.

        Expects:
        - NEURON_COMPILED_ARTIFACTS env var pointing to compiled text decoder
        - NEURON_ENCODER_PATH env var pointing to directory with encoder_T{500,1000,3000}.pt
        """
        import torch_neuronx
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
            NeuronQwen3VLForCausalLM as NxDIQwen3VL,
            Qwen3VLInferenceConfig,
            Qwen3VLNeuronConfig,
        )
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLTextForCausalLM,
        )
        from neuronx_distributed_inference.models.application_base import (
            load_state_dict as nxdi_load_sd,
        )
        from neuronx_distributed_inference.models.image_to_text_model_base import (
            normalize_path,
        )

        neuron_config_dict = kwargs.get("neuron_config", {})
        tp_degree = neuron_config_dict.get("tp_degree", 4)
        batch_size = neuron_config_dict.get("batch_size", 1)
        n_positions = neuron_config_dict.get("n_positions", 1024)
        seq_len = neuron_config_dict.get("seq_len", 1024)

        compiled_path = os.getenv("NEURON_COMPILED_ARTIFACTS")
        encoder_path = os.getenv(
            "NEURON_ENCODER_PATH", "/mnt/models/compiled/qwen3_asr_encoder"
        )

        if not compiled_path:
            raise ValueError(
                "NEURON_COMPILED_ARTIFACTS must be set for Qwen3-ASR "
                "(e.g., /mnt/models/compiled/qwen3_asr_vl_text_tp4)"
            )

        logger.info("Loading Qwen3-ASR text decoder from %s", compiled_path)
        logger.info("Loading Qwen3-ASR encoders from %s", encoder_path)

        # --- Load HF config for Qwen3-ASR ---
        # Use vLLM's registered Qwen3ASRConfig (avoids trust_remote_code issues in subprocesses)
        from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig
        import json
        config_path = os.path.join(model_name_or_path, "config.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        hf_config = Qwen3ASRConfig(**raw_config)
        text_config = hf_config.thinker_config.text_config

        # --- Build NxDI config ---
        text_neuron_config = Qwen3VLNeuronConfig(
            tp_degree=tp_degree,
            batch_size=batch_size,
            n_positions=n_positions,
            seq_len=seq_len,
            max_context_length=n_positions,
            on_device_sampling_config=None,
            torch_dtype=torch.bfloat16,
            is_continuous_batching=batch_size > 1,
        )
        vision_neuron_config = Qwen3VLNeuronConfig(
            tp_degree=tp_degree,
            batch_size=1,
            seq_len=512,
            n_positions=512,
            on_device_sampling_config=None,
            torch_dtype=torch.bfloat16,
        )

        text_config_dict = {
            "hidden_size": text_config.hidden_size,
            "num_hidden_layers": text_config.num_hidden_layers,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
            "intermediate_size": text_config.intermediate_size,
            "vocab_size": text_config.vocab_size,
            "max_position_embeddings": text_config.max_position_embeddings,
            "rope_theta": text_config.rope_theta,
            "rms_norm_eps": text_config.rms_norm_eps,
            "tie_word_embeddings": text_config.tie_word_embeddings,
            "attention_bias": getattr(text_config, "attention_bias", False),
            "hidden_act": "silu",
            "rope_scaling": {
                "type": "mrope",
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            },
            "pad_token_id": 151643,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "initializer_range": 0.02,
        }
        vision_config_dict = {
            "hidden_size": 1024,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "intermediate_size": 4096,
            "image_size": 224,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [],
            "vocab_size": text_config.vocab_size,
            "max_position_embeddings": 512,
            "depth": 1,
            "hidden_act": "gelu",
            "in_channels": 3,
            "initializer_range": 0.02,
            "num_heads": 16,
            "num_position_embeddings": 256,
            "out_hidden_size": text_config.hidden_size,
            "temporal_patch_size": 2,
        }

        config = Qwen3VLInferenceConfig(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            text_config=text_config_dict,
            vision_config=vision_config_dict,
            _name_or_path=model_name_or_path,
            image_token_id=self.AUDIO_TOKEN_ID,
        )

        # --- Instantiate NxDI model (subclass with no vision encoder) ---
        class _Qwen3ASRNxDI(NxDIQwen3VL):
            vision_model_cls = None
            vision_model_wrapper = None

            def enable_vision_encoder(self, **kwargs):
                pass

            def load(self, compiled_model_path, start_rank_id=0, debug=False, **kwargs):
                text_path = normalize_path(compiled_model_path) + "text_model/"
                self.text_traced_model = torch.jit.load(text_path + "model.pt")
                text_weights = self.get_text_builder(debug).shard_checkpoint()
                start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32)
                self.text_traced_model.nxd_model.initialize(
                    text_weights, start_rank_tensor
                )
                for model_wrapper in self.text_models:
                    model_wrapper.model = self.text_traced_model
                self.is_loaded_to_neuron = True

            def compile(self, compiled_model_path, debug=False, **kwargs):
                pass

            @classmethod
            def get_state_dict(cls, model_name_or_path, config):
                raw_sd = nxdi_load_sd(model_name_or_path)
                converted_sd = {}
                for key, value in raw_sd.items():
                    if key.startswith("thinker.audio_tower."):
                        continue
                    if key.startswith("thinker.model."):
                        new_key = "language_model." + key[len("thinker.model.") :]
                        converted_sd[new_key] = value
                    elif key.startswith("thinker.lm_head."):
                        new_key = key[len("thinker.") :]
                        converted_sd[new_key] = value
                    else:
                        converted_sd[key] = value
                model_sd = NeuronQwen3VLTextForCausalLM.convert_hf_to_neuron_state_dict(
                    converted_sd, config.text_config
                )
                if getattr(config.text_config, "tie_word_embeddings", False):
                    if (
                        "embed_tokens.weight" in model_sd
                        and "lm_head.weight" not in model_sd
                    ):
                        model_sd["lm_head.weight"] = model_sd["embed_tokens.weight"]
                return model_sd

        nxdi_model = _Qwen3ASRNxDI(compiled_path, config)
        nxdi_model.load(compiled_path)

        # Store the NxDI model
        # Add missing HF config attributes needed by NeuronBaseForImageToText.forward()
        for attr in ("output_attentions", "output_hidden_states", "use_return_dict"):
            if not hasattr(nxdi_model.text_config, attr):
                setattr(nxdi_model.text_config, attr, False)
        self.model = nxdi_model
        self._dtype = torch.bfloat16
        self._n_positions = n_positions

        # --- Load traced encoder NEFFs ---
        encoder_buckets = [500, 1000, 3000]
        for T in encoder_buckets:
            neff_path = os.path.join(encoder_path, f"encoder_T{T}.pt")
            if os.path.exists(neff_path):
                enc = torch.jit.load(neff_path)
                torch_neuronx.move_trace_to_device(enc, 0)
                # Warmup
                _ = enc(torch.randn(128, T))
                self.encoders[T] = enc
                logger.info("Loaded encoder bucket T=%d from %s", T, neff_path)
            else:
                logger.warning("Encoder NEFF not found: %s", neff_path)

        if not self.encoders:
            raise FileNotFoundError(
                f"No encoder NEFFs found in {encoder_path}. "
                "Expected files like encoder_T500.pt, encoder_T1000.pt, encoder_T3000.pt"
            )

        logger.info(
            "Qwen3-ASR model loaded: %d encoder buckets, text decoder TP=%d",
            len(self.encoders),
            tp_degree,
        )
        return True, compiled_path

    def _get_encoder_output_length(self, T_mel: int) -> int:
        """Calculate encoder output token count for given mel length."""
        input_lengths_leave = T_mel % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (T_mel // 100) * 13
        )
        return output_lengths

    def _select_bucket(self, T_mel: int) -> int:
        """Select the smallest encoder bucket that fits the mel length."""
        buckets = sorted(self.encoders.keys())
        for b in buckets:
            if T_mel <= b:
                return b
        return buckets[-1]

    def _run_encoder(
        self, mel_features: torch.Tensor, actual_mel_len: int
    ) -> torch.Tensor:
        """Run audio through the appropriate encoder bucket NEFF.

        Args:
            mel_features: Mel spectrogram [128, T] (128 mel bins for Qwen3-ASR)
            actual_mel_len: Actual number of valid mel frames

        Returns:
            Audio embeddings [N_tokens, hidden_size]
        """
        bucket_T = self._select_bucket(actual_mel_len)
        N_tokens = self._get_encoder_output_length(actual_mel_len)

        # Pad/trim mel to bucket size
        mel_input = mel_features[:, :bucket_T]
        if mel_input.shape[1] < bucket_T:
            mel_input = torch.nn.functional.pad(
                mel_input, (0, bucket_T - mel_input.shape[1])
            )

        with torch.no_grad():
            mel_input = mel_input.float()  # Encoder NEFF expects float32
            output = self.encoders[bucket_T](mel_input)

        # Trim to actual output length
        return output[:N_tokens]  # [N_tokens, hidden_size]

    def execute_model(self, model_input):
        """Extract audio features and run the encoder during prefill."""
        from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
            generate_positions_from_mask,
            pad_positions,
            pad_vision_embeddings,
        )

        input_audio_features = None
        feature_attention_mask = None

        if model_input.multi_modal_kwargs is not None:
            input_audio_features = model_input.multi_modal_kwargs.get(
                "input_audio_features"
            )
            feature_attention_mask = model_input.multi_modal_kwargs.get(
                "feature_attention_mask"
            )

        is_prefill = model_input.input_tokens.shape[-1] > 1
        vision_embeddings = None
        vision_mask = None

        if input_audio_features is not None and is_prefill:
            # Extract mel features
            if isinstance(input_audio_features, list):
                mel = input_audio_features[0]
            else:
                mel = input_audio_features
            if mel.dim() == 3:
                mel = mel.squeeze(0)  # [128, T]

            # Get actual mel length from attention mask
            if feature_attention_mask is not None:
                if isinstance(feature_attention_mask, list):
                    feature_attention_mask = feature_attention_mask[0]
                if feature_attention_mask.dim() > 1:
                    feature_attention_mask = feature_attention_mask.squeeze(0)
                actual_mel_len = int(feature_attention_mask.sum().item())
            else:
                actual_mel_len = mel.shape[-1]

            # Run encoder
            audio_embeddings = self._run_encoder(mel, actual_mel_len)
            audio_embeddings = audio_embeddings.to(self._dtype)
            N_tokens = audio_embeddings.shape[0]

            # Count placeholders in input_ids
            num_placeholder = (
                (model_input.input_tokens == self.AUDIO_TOKEN_ID).sum().item()
            )
            actual_scatter_count = min(N_tokens, num_placeholder)
            audio_embeddings = audio_embeddings[:actual_scatter_count]

            # Build vision_mask from audio token positions
            audio_positions = (model_input.input_tokens == self.AUDIO_TOKEN_ID).squeeze(
                0
            )
            position_indices = torch.where(audio_positions)[0][:actual_scatter_count]
            trimmed_mask = torch.zeros(
                model_input.input_tokens.shape[-1], dtype=torch.bool
            )
            trimmed_mask[position_indices] = True
            vision_mask = generate_positions_from_mask(trimmed_mask)

            # Pad to bucket size
            bucket_size = self._n_positions
            vision_mask = pad_positions(vision_mask, bucket_size, bucket_size - 1)

            # Reshape embeddings for scatter: [1, N, hidden] -> padded
            embedding_dim = audio_embeddings.shape[-1]
            vision_embeddings = audio_embeddings.unsqueeze(0)  # [1, N, hidden]
            vision_embeddings = pad_vision_embeddings(vision_embeddings, bucket_size)

        hidden_states = self.forward(
            input_ids=model_input.input_tokens,
            positions=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            sampling_params=model_input.sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
        sampling_params: torch.Tensor,
        vision_embeddings: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass routing to NxDI's CTE (prefill) or TKG (decode).

        Handles mRoPE position computation for Qwen3-ASR (all 3 axes identical
        since audio is 1D, unlike Qwen3-VL which has spatial positions).
        """
        import copy as _copy

        is_prefill = input_ids.shape[-1] > 1
        batch_size = input_ids.shape[0]

        # Compute mRoPE position IDs (all 3 axes same for ASR)
        if is_prefill:
            seq_len = input_ids.shape[1]
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            rotary_position_ids = pos.unsqueeze(0).expand(3, -1, -1)
            self._rope_deltas = torch.zeros(1, 1, dtype=torch.long)
        else:
            if self._rope_deltas is not None:
                delta = self._rope_deltas.to(input_ids.device)
                delta = delta.repeat_interleave(
                    batch_size // max(delta.shape[0], 1), dim=0
                )
            else:
                delta = 0
            rotary_position_ids = _copy.deepcopy(positions)
            rotary_position_ids = rotary_position_ids.view(1, -1).expand(batch_size, -1)
            rotary_position_ids = rotary_position_ids.add(delta)
            rotary_position_ids = rotary_position_ids.unsqueeze(0).expand(3, -1, -1)

        # Get dummy vision inputs for decode or text-only prefill
        if vision_embeddings is None:
            from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
                NeuronQwen3VLTextModelWrapper,
            )

            pad_limit = self._n_positions if is_prefill else 1
            # For decode, we need bucket_size = n_positions for the mask
            vision_embeddings, vision_mask, _ = (
                NeuronQwen3VLTextModelWrapper.get_dummy_vision_inputs(
                    config=self.model.text_config,
                    input_ids=input_ids,
                    n_active_tokens=self._n_positions,
                    fill_value=(self._n_positions - 1),
                )
            )

        deepstack_vision_embeds = torch.zeros(0)

        with self._reordered(
            input_block_ids,
            input_ids=input_ids,
            positions=positions,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        ) as (sorted_ids, inputs, restore):
            # Call grandparent forward (NeuronBaseForImageToText) which accepts vision_embeddings
            # NeuronQwen3VLForCausalLM.forward() only accepts pixel_values, not vision_embeddings
            from neuronx_distributed_inference.models.image_to_text_model_base import NeuronBaseForImageToText
            output = NeuronBaseForImageToText.forward(
                self.model,
                inputs["input_ids"].to(torch.int32),
                attention_mask=None,
                position_ids=inputs["positions"].to(torch.int32),
                seq_ids=sorted_ids.flatten().to(torch.int32),
                sampling_params=inputs["sampling_params"],
                vision_embeddings=inputs.get("vision_embeddings"),
                vision_mask=inputs.get("vision_mask"),
                rotary_position_ids=rotary_position_ids,
                deepstack_vision_embeds=deepstack_vision_embeds,
            )

            if self.model.config.neuron_config.on_device_sampling_config:
                output = output.hidden_states
            else:
                output = output.logits[:, -1, :]

            return restore(output)


