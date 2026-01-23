class NeuronPixtralTextModel(NeuronLlamaModel):
    """
    The neuron version of the Pixtral Text Model
    """
    def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask) -> torch.Tensor:
        # Concat vision and text embeddings during context encoding
        # Both inputs_embeds and vision_embeddings should be of the same shape: [BS, Total tokens (image + text), Hidden]
        # And vision_mask should of the shape [BS, Total tokens (image + text), 1]
        # Entries in vision_mask with value `True` represent vision tokens and with value `False` represent text tokens
        # For text-only inputs, vision_mask should be all `False`
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


class NeuronPixtralForCausalLM(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronPixtralTextModel
    vision_model_cls = NeuronPixtralVisionModel

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = PixtralVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls):
        return PixtralInferenceConfig

    def get_vision_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.vision_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return f"--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor={cc_pipeline_tiling_factor} --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024 \
                --internal-hlo2tensorizer-options='--verify-hlo=true'"

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        if new_config.vision_config.neuron_config.enable_bucketing:
            # neuron_config.buckets default to neuron_config.seq_len is not given. For vision we want to do auto-bucketing here
            if new_config.vision_config.neuron_config.buckets == [new_config.vision_config.neuron_config.seq_len] or \
                    new_config.vision_config.neuron_config.buckets is None:
                # 1024 vision seq len corresponds to a single 512x512 image. Smaller bucket size does not make sense in real life.
                if new_config.vision_config.neuron_config.seq_len > 1024:
                    new_config.vision_config.neuron_config.buckets = autobucketing.generate_buckets(
                        1024, new_config.vision_config.neuron_config.seq_len
                    )
                else:
                    new_config.vision_config.neuron_config.buckets = [new_config.vision_config.neuron_config.seq_len]
        # This should not be needed as in vision modeling code we should always use vision_config.neuron_config as vision model's neuron config
        # added this line just to add insurance to avoid mix-up
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # text model state dict convertion
        attention_keys = {
            ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
            ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
            ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
            ".self_attn.o_proj.": ".self_attn.o_proj.o_proj.",
        }
        new_state_dict = {}
        for dict_key in state_dict:
            if 'language_model.model.' in dict_key:
                new_key = dict_key.replace('language_model.model.', "")
                if not inference_config.neuron_config.fused_qkv:
                    for atten_key in attention_keys:
                        if atten_key in new_key:
                            replacement_atten_key = attention_keys[atten_key]
                            new_key = new_key.replace(atten_key, replacement_atten_key)
                new_state_dict[new_key] = state_dict[dict_key]
            elif 'language_model.' in dict_key:
                new_key = dict_key.replace('language_model.', "")
                new_state_dict[new_key] = state_dict[dict_key]
            else:
                new_state_dict[dict_key] = state_dict[dict_key]
        state_dict = NeuronLlamaForCausalLM.convert_hf_to_neuron_state_dict(
            new_state_dict, inference_config.text_config
        )

        # vision model state dict convertion
        state_dict = NeuronPixtralForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config
        )

        return state_dict

    def get_padding_length(self, input_ids):
        # vision inputs should be padded to context encoding model bucket
        buckets = self.context_encoding_model.config.neuron_config.buckets

        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise Exception("No bucket found for provided input_ids!")

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
            "image_sizes",
        ]

    def concat_causal_lm_outputs(self, outputs_list):
        concatenated_logits = []
        concatenated_hidden_states = []
        concatenated_tokens = []
        for output in outputs_list:
            if isinstance(output.logits, torch.Tensor):
                concatenated_logits.append(output.logits)
            if isinstance(output.hidden_states, torch.Tensor):
                concatenated_hidden_states.append(output.hidden_states)
            elif isinstance(output.hidden_states, list):
                concatenated_hidden_states.extend(output.hidden_states)
            if hasattr(output, 'tokens') and isinstance(output.tokens, torch.Tensor):
                concatenated_tokens.append(output.tokens)
        concatenated_logits = torch.cat(concatenated_logits, dim=0) if len(concatenated_logits) > 0 else None
        concatenated_tokens = torch.cat(concatenated_tokens, dim=0) if len(concatenated_tokens) else None

        concatentated_output = CausalLMOutputWithPast(
            logits=concatenated_logits,
            hidden_states=concatenated_hidden_states,
        )
        if concatenated_tokens is not None:
            concatentated_output.tokens = concatenated_tokens
        return concatentated_output

    def forward_atomic_prefill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None
    ):
        if image_sizes is None:
            assert len(pixel_values.shape) == 4, "Pixel value shape is expected to be [batch_size, num_channels, img_height, img_width]"
            img_hight = pixel_values.shape[2]
            img_width = pixel_values.shape[3]
            image_sizes = torch.tensor([[img_hight, img_width]], dtype=torch.int32)

        if vision_mask is None:
            vision_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
        # Convert vision mask from bool to indices
        assert (
            vision_mask.dtype == torch.bool
        ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"
        vision_mask = generate_positions_from_mask(vision_mask.squeeze())

        vision_embeddings = self.vision_encoder_model(
            pixel_values.to(self.vision_config.neuron_config.torch_dtype), image_sizes
        ).to(self.text_config.neuron_config.torch_dtype)

        # Pad vision embeddings and vision mask to corresponding text bucket
        pad_limit = self.get_padding_length(input_ids)
        print(f"vision_mask shape: {vision_mask.shape}, pad_limit: {pad_limit}, input_ids shape: {input_ids.shape}")
        vision_mask = pad_positions(
            vision_mask, pad_limit, (pad_limit - 1)
        )
        vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        tensor_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (
            (pixel_values is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):  # call vision encoder
            outputs = []
            for i in range(input_ids.shape[0]):
                outputs.append(
                    self.forward_atomic_prefill(
                        input_ids[i].unsqueeze(0),
                        attention_mask[i].unsqueeze(0) if (attention_mask is not None) else attention_mask,
                        position_ids[i].unsqueeze(0) if (position_ids is not None) else position_ids,
                        seq_ids[i].unsqueeze(0) if (seq_ids is not None) else seq_ids,
                        sampling_params[i].unsqueeze(0) if (sampling_params is not None) else sampling_params,
                        pixel_values[i].unsqueeze(0) if (pixel_values is not None) else pixel_values,
                        vision_mask[i].unsqueeze(0) if (vision_mask is not None) else vision_mask,
                        image_sizes[i].unsqueeze(0) if (image_sizes is not None) else image_sizes,
                    )
                )
            return self.concat_causal_lm_outputs(outputs)
        else:
            pad_limit = self.get_padding_length(input_ids)
            vision_embeddings, vision_mask = self.context_encoding_model.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1)
            )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
            )

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)

    def to_cpu(self):
        raise NotImplementedError("to_cpu() is not implemented")


class NeuronLlama4ForCausalLM(NeuronBaseForImageToText):
    # model cls
    text_model_cls = NeuronLlama4TextModel
    vision_model_cls = NeuronLlama4VisionEmbeddings

    # model wrappers
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Llama4VisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        self.compile_tag = VISION_ENCODER_MODEL_TAG

        new_config = copy.deepcopy(self.config)
        new_config.neuron_config = copy.deepcopy(self.vision_config.neuron_config)
        if new_config.neuron_config.enable_bucketing:
            if new_config.neuron_config.buckets is None:
                new_config.neuron_config.buckets = generate_llama4_vision_encoder_buckets(
                    self.neuron_config.dp_degree, VISION_MAX_NUM_CHUNKS
                )
        else:
            new_config.neuron_config.buckets = generate_buckets(
                VISION_MAX_NUM_CHUNKS, VISION_MAX_NUM_CHUNKS
            )
        self.vision_config.neuron_config.buckets = new_config.neuron_config.buckets
        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            # to turn on weight layout optimization
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=False,
            return_ranked_to_cpu=True
        )
        self.vision_models.append(self.vision_encoder_model)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        # text model state dict convertion
        state_dict = NeuronLlama4TextForCausalLM.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.text_config
        )

        # vision model state dict convertion
        state_dict = NeuronLlama4ForImageEncoding.convert_hf_to_neuron_state_dict(
            state_dict, inference_config.vision_config
        )

        return state_dict

    def _convert_input_dict_to_ordered_tuple(self, input_dict: Dict[str, Any]):
        """
        Utility function to convert input dictionary to ordered tuple
        based on outputs of _get_model_outputs
        """
        args = []

        for key in IMAGE_TO_TEXT_MODEL_WRAPPER_INPUT_KEYS:
            if key in input_dict and input_dict[key] is not None:
                arg = input_dict[key]
            else:
                arg = torch.empty(0)
            args.append(arg)

        return tuple(args)

    def _select_buckets_for_padding_length(self, position_ids):
        neuron_config = self.config.neuron_config
        context_encoding_buckets = neuron_config.context_encoding_buckets if neuron_config.context_encoding_buckets is not None \
            else neuron_config.buckets
        token_generation_buckets = neuron_config.token_generation_buckets if neuron_config.token_generation_buckets is not None \
            else neuron_config.buckets

        selected_buckets = token_generation_buckets
        if self._is_prefill(position_ids):
            selected_buckets = context_encoding_buckets

        return selected_buckets

    def get_padding_length(self, buckets, position_ids):
        max_position_id = torch.max(position_ids).item()
        for val in buckets:
            if val > max_position_id:
                return val
        raise ValueError("No bucket found for provided input_ids!")

    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "pixel_values",
            "vision_mask",
        ]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        medusa_args=None,
        input_capture_hook: Optional[Callable] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        batch_size, _ = input_ids.shape
        buckets = self._select_buckets_for_padding_length(position_ids)
        pad_limit = self.get_padding_length(buckets, position_ids)
        if (
            (pixel_values is not None)
            and (vision_mask is not None)
            and input_ids.shape[-1] > 1
            and pixel_values.sum() != 0
        ):  # call vision encoder
            assert (
                vision_mask.dtype == torch.bool
            ), f"Parameter `vision_mask` must be of type bool, recieved {vision_mask.dtype}"
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())
            vision_mask = pad_positions(
                vision_mask, pad_limit, (pad_limit - 1)
            )

            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
            ).to(self.text_config.neuron_config.torch_dtype)

            # flatten vision embeddings
            embedding_dim = vision_embeddings.shape[-1]
            vision_embeddings = vision_embeddings.view(-1, embedding_dim).unsqueeze(0)

            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)
        else:
            vision_embeddings, vision_mask = self.context_encoding_model.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1)
            )

        output_token = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )
        return output_token

    @classmethod
    def get_config_cls(cls):
        return Llama4InferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Llama4ForConditionalGeneration

        return Llama4ForConditionalGeneration.from_pretrained(model_path, **kwargs)

    def to_cpu(self):
        """
        Initialize CPU versions of both text and vision models with different parallelism configurations,
        shard and load their weights, and assign to respective model wrappers.
        This function as of now only supports TP DEGREE of 1 in vision and text.
        """
        os.environ["NXD_CPU_MODE"] = "1"

        # Validation checks
        if self.neuron_config.torch_dtype == torch.bfloat16 and (
            self.neuron_config.tp_degree > 1 or self.neuron_config.ve_tp_degree > 1
        ):
            raise NotImplementedError(
                "The gloo backend does not natively support bfloat16, please proceed with float32 dtype instead."
            )
        if self.neuron_config.speculation_length > 0:
            raise NotImplementedError("Speculation is not yet supported for CPU inference.")

        # destroy distributed process if already started
        if model_parallel_is_initialized():
            destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Initialize distributed processing
        if "WORLD_SIZE" in os.environ:
            assert (
                int(os.environ["WORLD_SIZE"]) == self.neuron_config.world_size
            ), "Total number of processes does not match implied world size from NeuronConfig inputs."
            torch.distributed.init_process_group("gloo")
        if not torch.distributed.is_initialized():
            if self.neuron_config.world_size == 1:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                torch.distributed.init_process_group(
                    backend="gloo",
                    world_size=1,
                    rank=0,
                )
            else:
                raise RuntimeError("Please initialize parallel processing via 'torchrun'.")

        # Initialize model parallel for vision and text model. We only support TP Degree 1 at this point.
        initialize_model_parallel(
            tensor_model_parallel_size=self.neuron_config.tp_degree,
            pipeline_model_parallel_size=1,  # No pipeline parallelism for vision encoder
            expert_model_parallel_size=1,  # No expert parallelism for vision encoder
            skip_collective_init=True,
        )

        # Initialize and load vision model with vision-specific config
        vision_base_model = self.vision_model_cls(self.config)
        vision_base_model = vision_base_model.to(
            self.vision_config.neuron_config.torch_dtype
        )

        vision_model_sd = (
            self.checkpoint_loader_fn()
        )  # You might need a separate loader for vision weights
        if self.vision_config.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                vision_model_sd,
                vision_base_model,
                torch.distributed.get_rank(),
                self.vision_config.neuron_config.tp_degree,
            )

        vision_base_model.load_state_dict(vision_model_sd, strict=False)

        # Initialize and load text model with text-specific config
        text_base_model = self.text_model_cls(self.config.text_config)
        text_base_model = text_base_model.to(self.config.text_config.neuron_config.torch_dtype)

        text_model_sd = self.checkpoint_loader_fn()
        if self.neuron_config.tp_degree > 1:
            get_sharded_checkpoint(
                text_model_sd,
                text_base_model,
                torch.distributed.get_rank(),
                self.neuron_config.tp_degree,
            )
        text_base_model.load_state_dict(text_model_sd, strict=False)

        # Assign models to their respective wrappers
        for model_wrapper in self.text_models:
            model_wrapper.model = text_base_model

        for model_wrapper in self.vision_models:
            model_wrapper.model = vision_base_model

        self.eval()

    # Wraps NeuronBaseForCausalLM.enable_context_encoding() to add compile_tag.
    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    # Wraps NeuronBaseForCausalLM.enable_token_generation() to add compile_tag.
    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self) -> str:
        logical_nc_config = self.text_config.neuron_config.logical_nc_config

        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O2"
        elif self.compile_tag == VISION_ENCODER_MODEL_TAG:
            return f"-O1 --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap' " \
                   f"--auto-cast=none --lnc={logical_nc_config}"
        else:
            raise ValueError(f"get_compiler_args() Invalid compile tag encountered: {self.compile_tag}")

        args = f"--auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap " \
               f"--cc-pipeline-tiling-factor=1 --vectorize-strided-dma --enable-scalar-dge-vectorization' " \
               f"--lnc={logical_nc_config} {optimization_level} "
        return args
