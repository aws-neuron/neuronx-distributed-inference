import torch
import logging
import copy
import time
import os
import json
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union, Dict

from safetensors.torch import load_file
from transformers.modeling_outputs import CausalLMOutputWithPast

import neuronx_distributed.trace.hlo_utils as hlo_utils
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM
from neuronx_distributed_inference.models.config import InferenceConfig, to_dict
from neuronx_distributed.trace.model_builder import ModelBuilder
from neuronx_distributed_inference.models.application_base import (
    COMPILED_MODEL_FILE_NAME,
    normalize_path,
)
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG
from neuronx_distributed_inference.utils.snapshot import (
    ScriptModuleWrapper,
    SnapshotOutputFormat,
    SnapshotCaptureConfig,
    get_snapshot_hook,
    register_nxd_model_hook,
    unregister_nxd_model_hooks,
)

logger = logging.getLogger("Neuron")


class ImageToTextInferenceConfig(InferenceConfig):
    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            # text_config will be the parent neuron's config
            neuron_config=copy.deepcopy(text_neuron_config),
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )
        assert hasattr(self, "text_config"), "Passed config doesn't contain text_config"
        assert hasattr(self, "vision_config"), "Passed config doesn't contain vision_config"
        if isinstance(self.text_config, SimpleNamespace):
            self.text_config = vars(self.text_config)
        # Llama4 needs to disable chunked attention
        # since kernels are not yet supported with chunked attention
        # TODO: Remove below pop once chunked attention is supported with kernels
        self.text_config = InferenceConfig(text_neuron_config, **self.text_config)

        # We need to save the model's _name_or_path in text_config to be able to bring up the HF text only model
        self.text_config._name_or_path = self._name_or_path

        if isinstance(self.vision_config, SimpleNamespace):
            self.vision_config = vars(self.vision_config)
        self.vision_config = InferenceConfig(vision_neuron_config, **self.vision_config)

    def get_required_attributes(self) -> List[str]:
        raise NotImplementedError("get_required_attributes is not implemented")

    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """

        def hasattr_nested(obj, attr_chain):
            attrs = attr_chain.split(".")
            for attr in attrs:
                if isinstance(obj, dict):
                    if attr not in obj:
                        return False
                    obj = obj[attr]
                else:
                    if not hasattr(obj, attr):
                        return False
                    obj = getattr(obj, attr)
            return True

        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr_nested(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> "InferenceConfig":
        merged_kwargs = json.loads(json_string)
        merged_kwargs.update(kwargs)

        # Initialize NeuronConfig from dict.
        text_neuron_config = cls.get_neuron_config_cls()(**merged_kwargs["text_config"]["neuron_config"])
        vision_neuron_config = cls.get_neuron_config_cls()(**merged_kwargs["vision_config"]["neuron_config"])
        merged_kwargs.pop("neuron_config")
        merged_kwargs["text_config"].pop("neuron_config")
        merged_kwargs["vision_config"].pop("neuron_config")

        return cls(text_neuron_config, vision_neuron_config, **merged_kwargs)

    @classmethod
    def get_neuron_config_cls(cls):
        raise NotImplementedError("get_neuron_config_cls is not implemented")


class NeuronBaseForImageToText(NeuronBaseForCausalLM):
    def __init__(
        self,
        text_model_cls,
        vision_model_cls,
        text_model_wrapper,
        vision_model_wrapper,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # model cls
        text_model_cls = text_model_cls
        vision_model_cls = vision_model_cls

        # model wrappers
        text_model_wrapper = text_model_wrapper
        vision_model_wrapper = vision_model_wrapper

        self.text_config = self.config.text_config
        self.vision_config = self.config.vision_config
        self.vocab_size = self.text_config.vocab_size
        self.padding_side = self.neuron_config.padding_side
        self.kv_cache_populated = False

        self.model_wrapper = self.text_model_wrapper
        self._model_cls = self.text_model_cls

        # TODO: refactor application_base and model_base to handle nested configs
        self.full_config = self.config
        self.config = self.text_config

        # self.models is initialized in super().__init__() with CTE and TKG
        # ModelWrapper objects. To ensure only ImageToTextModelWrappers are present in self.models,
        # we reset this list prior to enable_context_encoding() and enable_token_generation().
        self.models = []

        self.enable_context_encoding()
        self.enable_token_generation()

        self.text_models = self.models

        # switch config back to full config
        self.config = self.full_config
        self.vision_models = []
        self.enable_vision_encoder()

        self.text_builder = None
        self.vision_builder = None

        if self.neuron_config.is_prefix_caching:
            raise ValueError("NeuronBaseForImageToText does not yet support prefix caching")
        if self.neuron_config.enable_fused_speculation:
            raise ValueError("NeuronBaseForImageToText does not yet support fused speculation")
        if self.neuron_config.is_chunked_prefill:
            raise ValueError("NeuronBaseForImageToText does not yet support chunked prefill")
        if self.neuron_config.is_medusa:
            raise ValueError("NeuronBaseForImageToText does not yet support medusa")

    def get_text_builder(self, debug=False):
        if self.text_builder is None:
            base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
            base_compile_work_dir = normalize_path(base_compile_work_dir) + "text_model/"
            # Use this function to initialize non-standard TP/PP/DP distributed
            # process groups.
            self.text_builder = ModelBuilder(
                router=None,
                tp_degree=self.text_config.neuron_config.tp_degree,
                pp_degree=self.text_config.neuron_config.pp_degree,
                ep_degree=self.text_config.neuron_config.ep_degree,
                world_size=self.text_config.neuron_config.world_size,
                start_rank_id=self.text_config.neuron_config.start_rank_id,
                local_ranks_size=self.text_config.neuron_config.local_ranks_size,
                checkpoint_loader=self.checkpoint_loader_fn,
                compiler_workdir=base_compile_work_dir,
                debug=debug,
                num_cores_per_group=self.text_config.num_cores_per_group,
                init_custom_process_group_fn=None,
                logical_nc_config=self.text_config.neuron_config.logical_nc_config,
                weights_to_skip_layout_optimization=self.text_config.neuron_config.weights_to_skip_layout_optimization,
            )
            for model in self.text_models:
                self.text_builder.add(
                    key=model.tag,
                    model_instance=model.get_model_instance(),
                    example_inputs=model.input_generator(),
                    compiler_args=model.compiler_args,
                    bucket_config=model.bucket_config,
                    priority_model_idx=model.priority_model_idx,
                )
        return self.text_builder

    def get_vision_builder(self, debug=False):
        if self.vision_builder is None:
            base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
            base_compile_work_dir = normalize_path(base_compile_work_dir) + "vision_model/"
            # Use this function to initialize non-standard TP/PP/DP distributed
            # process groups.
            self.vision_builder = ModelBuilder(
                router=None,
                tp_degree=self.vision_config.neuron_config.tp_degree,
                pp_degree=self.vision_config.neuron_config.pp_degree,
                ep_degree=self.vision_config.neuron_config.ep_degree,
                world_size=self.vision_config.neuron_config.world_size,
                start_rank_id=self.vision_config.neuron_config.start_rank_id,
                local_ranks_size=self.vision_config.neuron_config.local_ranks_size,
                checkpoint_loader=self.checkpoint_loader_fn,
                compiler_workdir=base_compile_work_dir,
                debug=debug,
                num_cores_per_group=self.vision_config.num_cores_per_group,
                init_custom_process_group_fn=None,
                logical_nc_config=self.vision_config.neuron_config.logical_nc_config,
                weights_to_skip_layout_optimization=self.vision_config.neuron_config.weights_to_skip_layout_optimization,
            )
            for model in self.vision_models:
                self.vision_builder.add(
                    key=model.tag,
                    model_instance=model.get_model_instance(),
                    example_inputs=model.input_generator(),
                    compiler_args=model.compiler_args,
                    bucket_config=model.bucket_config,
                    priority_model_idx=model.priority_model_idx,
                )
        return self.vision_builder

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        """logic to enable_vision_encoder for this model."""
        raise NotImplementedError("enable_vision_encoder is not implemented")

    def shard_text_weights(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        sharded_checkpoint_dir = os.path.join(compiled_model_path, "weights/")
        if pre_shard_weights_hook:
            pre_shard_weights_hook(self)

        if self.text_config.neuron_config.skip_sharding:
            logger.info("Pre-sharding the checkpoints is forced to be SKIPPED with skip_sharding.")
        elif not self.text_config.neuron_config.save_sharded_checkpoint:
            logger.info(
                "SKIPPING pre-sharding the checkpoints. The checkpoints will be sharded during load time."
            )
        else:
            self.get_text_builder(debug).shard_checkpoint(serialize_path=sharded_checkpoint_dir)

            if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
                self.get_text_builder(debug).transform_weight_layout_with_overriden_option(
                    sharded_checkpoint_dir=sharded_checkpoint_dir
                )

    def shard_vision_weights(self, compiled_model_path, debug=False, pre_shard_weights_hook=None):
        sharded_checkpoint_dir = os.path.join(compiled_model_path, "weights/")
        if pre_shard_weights_hook:
            pre_shard_weights_hook(self)

        if self.vision_config.neuron_config.skip_sharding:
            logger.info("Pre-sharding the checkpoints is forced to be SKIPPED with skip_sharding.")
        elif not self.vision_config.neuron_config.save_sharded_checkpoint:
            logger.info(
                "SKIPPING pre-sharding the checkpoints. The checkpoints will be sharded during load time."
            )
        else:
            self.get_vision_builder(debug).shard_checkpoint(serialize_path=sharded_checkpoint_dir)

            if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
                self.get_vision_builder(debug).transform_weight_layout_with_overriden_option(
                    sharded_checkpoint_dir=sharded_checkpoint_dir
                )

    def compile(self, compiled_model_path, debug=False, pre_shard_weights_hook=None, dry_run=False):
        # save config
        self.config.save(compiled_model_path)

        # Create text and vision model paths
        text_compiled_model_path = normalize_path(compiled_model_path) + "text_model/"
        vision_compiled_model_path = normalize_path(compiled_model_path) + "vision_model/"
        os.makedirs(text_compiled_model_path, exist_ok=True)
        os.makedirs(vision_compiled_model_path, exist_ok=True)

        # Trace text and vision models
        text_traced_model = self.get_text_builder(debug).trace(initialize_model_weights=False, dry_run=dry_run)
        if not dry_run:
            torch.jit.save(text_traced_model, text_compiled_model_path + COMPILED_MODEL_FILE_NAME)
            del text_traced_model
            logger.info("Finished compiling text model!")

        vision_traced_model = self.get_vision_builder(debug).trace(initialize_model_weights=False, dry_run=dry_run)
        if not dry_run:
            torch.jit.save(vision_traced_model, vision_compiled_model_path + COMPILED_MODEL_FILE_NAME)
            del vision_traced_model
            logger.info("Finished compiling vision model!")

        self._save_configs_to_compiler_workdir()

        if dry_run:
            return

        # Shard the weights
        self.shard_text_weights(text_compiled_model_path, debug, pre_shard_weights_hook)
        logger.info("Finished sharding weights for text model!")

        self.shard_vision_weights(vision_compiled_model_path, debug, pre_shard_weights_hook)
        logger.info("Finished sharding weights for vision model!")

        self.is_compiled = True
        logger.info("Compilation complete for E2E model!")

    def _save_configs_to_compiler_workdir(self):
        # save full model neuron config
        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self.config.save(base_compile_work_dir)

        # save sub-appmodel configs
        text_model_compile_work_dir = os.path.join(base_compile_work_dir, "text_model")
        vision_model_compile_work_dir = os.path.join(base_compile_work_dir, "vision_model")
        self.config.text_config.save(text_model_compile_work_dir)
        self.config.vision_config.save(vision_model_compile_work_dir)

        # generate a new config for each submodel and bucket size
        for submodel in self.text_models:
            for bucket_rank, bucket_size in enumerate(submodel.config.neuron_config.buckets):
                specific_config = copy.deepcopy(submodel.config)
                specific_config.neuron_config.buckets = [bucket_size]

                if submodel.tag == CONTEXT_ENCODING_MODEL_TAG:
                    specific_config.neuron_config.context_encoding_buckets = specific_config.neuron_config.buckets
                else:
                    specific_config.neuron_config.token_generation_buckets = specific_config.neuron_config.buckets

                submodel_path = os.path.join(text_model_compile_work_dir, submodel.tag, f"_tp0_bk{bucket_rank}")
                specific_config.save(submodel_path)

        for submodel in self.vision_models:
            for bucket_rank, bucket_size in enumerate(submodel.config.neuron_config.buckets):
                specific_config = copy.deepcopy(submodel.config)
                specific_config.neuron_config.buckets = [bucket_size]
                submodel_path = os.path.join(vision_model_compile_work_dir, submodel.tag, f"_tp0_bk{bucket_rank}")
                specific_config.save(submodel_path)

    def load(
        self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False
    ):
        compiled_model_path = normalize_path(compiled_model_path)
        text_compiled_model_path = normalize_path(compiled_model_path) + "./text_model/"
        vision_compiled_model_path = normalize_path(compiled_model_path) + "./vision_model/"

        """Loads the compiled model checkpoint to the Neuron device."""
        self.text_traced_model = torch.jit.load(text_compiled_model_path + COMPILED_MODEL_FILE_NAME)
        self.vision_traced_model = torch.jit.load(
            vision_compiled_model_path + COMPILED_MODEL_FILE_NAME
        )

        self.load_weights(
            text_compiled_model_path,
            vision_compiled_model_path,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
        )

        for model_wrapper in self.text_models:
            model_wrapper.model = self.text_traced_model

        for model_wrapper in self.vision_models:
            model_wrapper.model = self.vision_traced_model

        self.is_loaded_to_neuron = True

        if not self.neuron_config.skip_warmup and not skip_warmup:
            self.warmup()  # warmup will be executed only if both flags are false
        else:
            logger.info("Skipping model warmup")

    def load_weights(
        self,
        text_compiled_model_path,
        vision_compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
    ):
        """Loads the model weights to the Neuron device."""
        if self.text_traced_model is None or self.vision_traced_model is None:
            raise ValueError("Model is not loaded")

        if start_rank_id is None:
            start_rank_id = self.neuron_config.start_rank_id
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        text_weights = []
        start_time = time.monotonic()
        if self.neuron_config.save_sharded_checkpoint:
            logging.info(
                f"Loading presharded checkpoints for {start_rank_id}...{start_rank_id + local_ranks_size - 1}"
            )
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(
                    os.path.join(
                        text_compiled_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"
                    )
                )
                text_weights.append(ckpt)
        else:
            logger.info("Sharding weights on load...")
            text_weights = self.get_text_builder().shard_checkpoint()

        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.text_traced_model.nxd_model.initialize(text_weights, start_rank_tensor)
        logger.info(f"Finished text weights loading in {time.monotonic() - start_time} seconds")

        vision_weights = []
        start_time = time.monotonic()
        if self.neuron_config.save_sharded_checkpoint:
            logging.info(
                f"Loading presharded checkpoints for {start_rank_id}...{start_rank_id + local_ranks_size - 1}"
            )
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(
                    os.path.join(
                        vision_compiled_model_path,
                        f"weights/tp{rank}_sharded_checkpoint.safetensors",
                    )
                )
                vision_weights.append(ckpt)
        else:
            logger.info("Sharding weights on load...")
            vision_weights = self.get_vision_builder().shard_checkpoint()

        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.vision_traced_model.nxd_model.initialize(vision_weights, start_rank_tensor)
        logger.info(f"Finished vision weights loading in {time.monotonic() - start_time} seconds")

    def register_snapshot_hooks(
        self,
        output_path: str,
        output_format: SnapshotOutputFormat,
        capture_at_requests: List[int],
        *_,  # capture unused positional args to not crash overriden method
        ranks: Optional[List[int]] = None,
        **kwargs,  # make sure new kwargs don't crash overriden method
    ):
        """
        Registers snapshot hooks to capture input snapshots for all submodels and bucket.

        This overrides the NeuronApplicationBase implementation to register snapshots for the independent
        text and vision models. See the NeuronApplicationBase docstrings for details.
        """
        assert self.is_loaded_to_neuron, "Must load model before you register snapshot hooks"
        if ranks is None:
            ranks = [0]

        def _register_submodel(submodel, output_path, model_builder):
            snapshot_config = SnapshotCaptureConfig().capture_at_request(
                capture_at_requests
            )
            submodel.model = ScriptModuleWrapper(submodel.model)
            snapshot_hook = get_snapshot_hook(
                output_path,
                output_format,
                snapshot_config,
                model_builder,
                ranks,
                is_input_ranked=submodel.async_mode or submodel.pipeline_execution,
            )
            submodel.model.register_forward_hook(snapshot_hook)
            register_nxd_model_hook(submodel.model, "forward_async", snapshot_hook)
            register_nxd_model_hook(submodel.model, "forward_ranked", snapshot_hook)
            logger.info(f"Registered snapshot hooks for {submodel.tag=}")

        text_output_path = os.path.join(output_path, "text_model")
        for submodel in self.text_models:
            _register_submodel(submodel, text_output_path, self.get_text_builder())

        vision_output_path = os.path.join(output_path, "vision_model")
        for submodel in self.vision_models:
            _register_submodel(submodel, vision_output_path, self.get_vision_builder())

    def unregister_snapshot_hooks(self):
        """
        Unregisters snapshot hooks for this model.

        This overrides the NeuronApplicationBase implementation to unregister snapshots for the independent
        text and vision models. See the NeuronApplicationBase docstrings for details.
        """

        def _unregister_submodel(submodel):
            if isinstance(submodel.model, ScriptModuleWrapper):
                submodel.model = submodel.model.wrapped_module
                unregister_nxd_model_hooks(submodel.model, "forward_async")
                unregister_nxd_model_hooks(submodel.model, "forward_ranked")
                logger.info(f"Unregistered snapshot hooks for {submodel.tag=}")

        for submodel in self.text_models:
            _unregister_submodel(submodel)

        for submodel in self.vision_models:
            _unregister_submodel(submodel)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        prev_hidden: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.LongTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        llava_args: Optional[List] = [],
        input_capture_hook: Optional[Callable] = None,
        slot_mapping: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.LongTensor] = None,
        full_context_lens: Optional[torch.LongTensor] = None,
        computed_context_lens: Optional[torch.LongTensor] = None,
        vision_embeddings: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        self.preprocess_inputs(
            input_ids=input_ids,
            seq_ids=seq_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            sampling_params=sampling_params,
            prev_hidden=prev_hidden,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            adapter_ids=adapter_ids,
            medusa_args=medusa_args,
            return_dict=return_dict,
            llava_args=llava_args,
            input_capture_hook=input_capture_hook,
            slot_mapping=slot_mapping,
            block_table=block_table,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
        )

        if self.async_mode:
            outputs, is_run_on_neuron = self._get_model_outputs_async(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                prev_hidden=prev_hidden,
                adapter_ids=adapter_ids,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
                medusa_args=medusa_args,
                llava_args=llava_args,
            )
        else:
            outputs, is_run_on_neuron = self._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                vision_embeddings,
                vision_mask,
                medusa_args,
                llava_args,
            )

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        # Process outputs
        constructed_outputs = self._get_constructed_outputs(outputs, is_run_on_neuron)

        # Apply tensor_capture_hook if provided and tensors are captured
        if tensor_capture_hook and constructed_outputs.captured_tensors:
            # Apply the hook if captured tensors are found
            tensor_capture_hook(self, constructed_outputs.captured_tensors)

        return constructed_outputs

    def _get_captured_tensors_offset(self):
        """
        Returns the number of tensors that were captured based on tensor_capture_config.
        This is used to determine the offset in the output tensors.

        Returns:
            int: The total number of tensors to be captured
        """
        if self.text_config.neuron_config.tensor_capture_config:
            return self.text_config.neuron_config.tensor_capture_config.get_offset()
        return 0

    def _get_constructed_outputs(self, outputs, is_run_on_neuron):
        """
        Process model outputs and handle tensor capture.

        Args:
            outputs: Raw outputs from the model
            is_run_on_neuron: Whether the model was run on Neuron device

        Returns:
            CausalLMOutputWithPast: Processed outputs with captured tensors if available
        """
        # Process outputs
        if self.on_device_sampling and self.text_config.neuron_config.output_logits and not \
                (self.text_config.neuron_config.enable_fused_speculation or self.text_config.neuron_config.is_medusa):
            logits_or_next_tokens = outputs[:2]
            constructed_outputs = self._construct_output_with_tokens_and_logits(next_tokens=logits_or_next_tokens[0], logits=logits_or_next_tokens[1])
        else:
            if is_run_on_neuron:
                # When run on neuron, KV cache remains on device
                logits_or_next_tokens = outputs
            else:
                # When run on cpu, KV cache is returned which has to be ignored
                logits_or_next_tokens, *_ = outputs
            constructed_outputs = self._construct_output(logits_or_next_tokens)

        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug("---output---")
            logging.debug(
                f"{'tokens' if self.on_device_sampling else 'logits'} = %s, ",
                logits_or_next_tokens,
            )

        return constructed_outputs

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
        medusa_args,
        llava_args,
        slot_mapping=None,
        block_table=None,
        full_context_lens=None,
        computed_context_lens=None,
    ):

        if self._is_prefill(position_ids):
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping=None,
                torch.empty(0),  # active_block_table=None,
                torch.empty(0),  # num_queries=None,
                torch.empty(0),  # computed_context_lens=None,
                torch.empty(0),  # tile_q_indices=None,
                torch.empty(0),  # tile_block_tables=None,
                torch.empty(0),  # tile_masks=None,
                torch.empty(0),  # inputs_embeds: Optional[torch.FloatTensor] = None,
                torch.empty(0),  # kv_cache: Optional[torch.Tensor] = None,
                torch.empty(0),  # active_mask=None,
                torch.empty(0),  # rotary_position_id=None,
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
                torch.empty(0),  # prev_hidden
                torch.empty(0),  # adapter_ids
                torch.empty(0),  # accepted_indices
                torch.empty(0),  # current_length
                torch.empty(0),  # medusa_mask
                torch.empty(0),  # scatter_index
                torch.empty(0),  # slot_mapping=None,
                torch.empty(0),  # active_block_table=None,
                torch.empty(0),  # num_queries=None,
                torch.empty(0),  # computed_context_lens=None,
                torch.empty(0),  # tile_q_indices=None,
                torch.empty(0),  # tile_block_tables=None,
                torch.empty(0),  # tile_masks=None,
                torch.empty(0),  # inputs_embeds: Optional[torch.FloatTensor] = None,
                torch.empty(0),  # kv_cache: Optional[torch.Tensor] = None,
                torch.empty(0),  # active_mask=None,
                torch.empty(0),  # rotary_position_id=None,
                vision_embeddings,
                vision_mask,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron
