"""
Ministral-3-14B-Instruct-2512 (Leanstral) on AWS Neuron via NxDI.

A vision-language model combining a Pixtral vision encoder with a Llama-compatible
text decoder and a Mistral3-specific PatchMerger projector. Architecture: 40 layers,
hidden=5120, heads=32/8kv, vocab=131072, intermediate=16384, head_dim=128.

This contrib model reuses NxDI's Pixtral VL pipeline (NeuronPixtralVisionModel for the
vision encoder, NeuronLlamaModel for the text decoder) with three key adaptations:

1. CPU projector: The Mistral3 PatchMerger (spatial 2x2 merge via F.unfold + 2-layer MLP)
   runs on CPU since it has no NxDI equivalent.
2. SHARD_OVER_HEADS GQA: Avoids replicating KV heads when kv_heads >= tp_degree. With 8 KV
   heads at TP=4, each rank gets kv_heads_per_rank=2 instead of stock NxDI's replication to 8.
3. Multi-KV-head TKG kernel: A modified nki-library attention_block_tkg kernel that supports
   kv_heads_per_rank > 1 via a virtual-batch approach.

Requires:
- SDK 2.28 (neuronx-cc >= 2.23, neuronx-distributed-inference >= 0.8)
- trn2.3xlarge (TP=4, LNC=2)
- Model checkpoint: mistralai/Ministral-3-14B-Instruct-2512 (HuggingFace, gated)

Known limitations:
- TKG kernel uses grid=1 (NCC_IXLV002 workaround), ~4% text throughput cost
- FP8 checkpoint weights are dequantized to bf16 during state_dict conversion
"""

import copy
import json
import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors import safe_open

logger = logging.getLogger(__name__)

# Mistral3 uses image_token_id=10 for [IMG] tokens in the vocabulary
IMAGE_TOKEN_ID = 10
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2


# ---------------------------------------------------------------------------
# SHARD_OVER_HEADS GQA patch
# ---------------------------------------------------------------------------
# NxDI 0.8 only supports CONVERT_TO_MHA (replicates KV heads to match Q heads)
# and REPLICATE_TO_TP_DEGREE. For models where kv_heads >= tp_degree and
# kv_heads % tp_degree == 0, we can shard KV heads across ranks instead.
# This avoids inflating KV cache memory by 4x and enables the multi-KV-head
# TKG kernel path.
#
# This patch should be applied BEFORE any NxDI model classes are imported.
# It is a candidate for upstream NxDI inclusion (see fork branch
# feature/shard-over-heads-gqa on github.com/jimburtoft/neuronx-distributed-inference).


_shard_over_heads_applied = False


def apply_shard_over_heads_patch():
    """Patch NxDI's GQA sharding to support kv_heads >= tp_degree without replication."""
    global _shard_over_heads_applied
    if _shard_over_heads_applied:
        return

    import neuronx_distributed_inference.modules.attention.gqa as gqa_module

    _orig_determine = gqa_module.determine_sharding_strategy
    _orig_get_shardable = gqa_module.get_shardable_head_counts

    def _patched_determine(
        tp_degree, source_key_value_heads, desired_sharding_strategy=None
    ):
        if (
            source_key_value_heads >= tp_degree
            and source_key_value_heads % tp_degree == 0
        ):
            return gqa_module.GQA.CONVERT_TO_MHA
        return _orig_determine(
            tp_degree, source_key_value_heads, desired_sharding_strategy
        )

    def _patched_get_shardable(
        tp_degree, num_attention_heads, num_key_value_heads, sharding_strategy
    ):
        if (
            sharding_strategy == gqa_module.GQA.CONVERT_TO_MHA
            and num_key_value_heads >= tp_degree
            and num_key_value_heads % tp_degree == 0
        ):
            from neuronx_distributed_inference.modules.attention.gqa import (
                get_number_of_extra_heads,
            )

            updated = num_attention_heads + get_number_of_extra_heads(
                num_attention_heads, tp_degree
            )
            return updated, num_key_value_heads
        return _orig_get_shardable(
            tp_degree, num_attention_heads, num_key_value_heads, sharding_strategy
        )

    # Patch in all modules that import these functions
    for module_path in [
        "neuronx_distributed_inference.modules.attention.gqa",
        "neuronx_distributed_inference.modules.kvcache.kv_cache_manager",
    ]:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            mod.determine_sharding_strategy = _patched_determine
            mod.get_shardable_head_counts = _patched_get_shardable
        except (ImportError, AttributeError):
            pass

    # Also patch gpt_kv_cache_manager if present
    try:
        import neuronx_distributed_inference.modules.kvcache.gpt_kv_cache_manager as gpt_kv

        gpt_kv.determine_sharding_strategy = _patched_determine
        gpt_kv.get_shardable_head_counts = _patched_get_shardable
    except (ImportError, AttributeError):
        pass

    _shard_over_heads_applied = True
    logger.info("SHARD_OVER_HEADS GQA patch applied")


# ---------------------------------------------------------------------------
# Multi-KV-head TKG kernel adapter patch
# ---------------------------------------------------------------------------
# The stock NxDI TKG kernel hardcodes kv_heads=1 per rank. This adapter
# replaces the dispatch method to call our modified nki-library kernel
# (attention_block_tkg_multi_kv) which supports kv_heads_per_rank > 1.
# For kv_heads_per_rank == 1, the patch is a no-op passthrough.

_multi_kv_patch_applied = False


def apply_multi_kv_tkg_patch():
    """Patch NxDI's TKG kernel dispatch for multi-KV-head support."""
    global _multi_kv_patch_applied
    if _multi_kv_patch_applied:
        return

    from . import patch_native_multi_kv

    patch_native_multi_kv.apply_patch()
    _multi_kv_patch_applied = True
    logger.info("Multi-KV-head TKG kernel adapter patch applied")


# ---------------------------------------------------------------------------
# CPU Projector (Mistral3 PatchMerger + MLP)
# ---------------------------------------------------------------------------
# The Mistral3 projector does spatial 2x2 merging of vision patches followed
# by a 2-layer MLP. It runs on CPU because NxDI's Pixtral pipeline does not
# include this specific projector variant.


class Mistral3PatchMerger(nn.Module):
    """Spatial 2x2 patch merger using F.unfold for correct spatial ordering."""

    def __init__(self, hidden_size, spatial_merge_size=SPATIAL_MERGE_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.merging_layer = nn.Linear(
            hidden_size * spatial_merge_size * spatial_merge_size,
            hidden_size,
            bias=False,
        )

    def forward(self, features, ph, pw):
        merge = self.spatial_merge_size
        ph_m = (ph // merge) * merge
        pw_m = (pw // merge) * merge
        feats = features.view(ph, pw, self.hidden_size)[:ph_m, :pw_m, :]
        image_grid = feats.permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.unfold(image_grid, kernel_size=merge, stride=merge)
        grid = grid.view(self.hidden_size * merge * merge, -1).t()
        return self.merging_layer(grid)


class Mistral3CPUProjector(nn.Module):
    """CPU-side vision-to-text projector: RMSNorm -> PatchMerger -> 2-layer MLP."""

    def __init__(
        self,
        vision_hidden_size,
        text_hidden_size,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(vision_hidden_size, eps=1e-5)
        self.patch_merger = Mistral3PatchMerger(vision_hidden_size, spatial_merge_size)
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=False)

    def forward(self, vision_features, image_h, image_w):
        ph = image_h // PATCH_SIZE
        pw = image_w // PATCH_SIZE
        feats = vision_features.squeeze(0)
        feats = self.norm(feats.float()).to(feats.dtype)
        feats = self.patch_merger(feats, ph, pw)
        feats = self.linear_1(feats)
        feats = self.act(feats)
        feats = self.linear_2(feats)
        return feats


def load_cpu_projector(model_path, vision_hidden_size, text_hidden_size):
    """Load Mistral3CPUProjector weights from safetensors checkpoint."""
    projector = Mistral3CPUProjector(vision_hidden_size, text_hidden_size)
    weight_mapping = {
        "multi_modal_projector.norm.weight": "norm.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight": "patch_merger.merging_layer.weight",
        "multi_modal_projector.linear_1.weight": "linear_1.weight",
        "multi_modal_projector.linear_2.weight": "linear_2.weight",
    }
    safetensors_files = sorted(
        f
        for f in os.listdir(model_path)
        if f.endswith(".safetensors") and "consolidated" not in f
    )
    for fname in safetensors_files:
        with safe_open(os.path.join(model_path, fname), framework="pt") as f:
            for key in f.keys():
                if key in weight_mapping:
                    target_key = weight_mapping[key]
                    parts = target_key.split(".")
                    module = projector
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    existing = getattr(module, parts[-1])
                    with torch.no_grad():
                        existing.copy_(f.get_tensor(key).to(existing.dtype))
    projector.eval().to(torch.bfloat16)
    return projector


# ---------------------------------------------------------------------------
# NxDI imports (deferred to avoid import before patches are applied)
# ---------------------------------------------------------------------------
# These are imported at function/class scope to ensure patches are applied
# before the NxDI module import chain runs.


def _get_nxdi_imports():
    """Lazy import of NxDI classes. Call after patches are applied."""
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        NeuronLlamaModel,
    )
    from neuronx_distributed_inference.models.image_to_text_model_base import (
        NeuronBaseForImageToText,
    )
    from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
        ImageToTextModelWrapper,
    )
    from neuronx_distributed_inference.models.model_wrapper import (
        VISION_ENCODER_MODEL_TAG,
    )
    from neuronx_distributed_inference.models.pixtral.modeling_pixtral import (
        PixtralInferenceConfig,
    )
    from neuronx_distributed_inference.models.pixtral.modeling_pixtral_vision import (
        NeuronPixtralVisionModel,
        PixtralVisionModelWrapper,
    )
    from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
        generate_positions_from_mask,
        pad_positions,
        pad_vision_embeddings,
        scatter_by_index_put,
    )
    import neuronx_distributed_inference.modules.autobucketing as autobucketing

    return {
        "NeuronConfig": NeuronConfig,
        "NeuronLlamaModel": NeuronLlamaModel,
        "NeuronBaseForImageToText": NeuronBaseForImageToText,
        "ImageToTextModelWrapper": ImageToTextModelWrapper,
        "VISION_ENCODER_MODEL_TAG": VISION_ENCODER_MODEL_TAG,
        "PixtralInferenceConfig": PixtralInferenceConfig,
        "NeuronPixtralVisionModel": NeuronPixtralVisionModel,
        "PixtralVisionModelWrapper": PixtralVisionModelWrapper,
        "generate_positions_from_mask": generate_positions_from_mask,
        "pad_positions": pad_positions,
        "pad_vision_embeddings": pad_vision_embeddings,
        "scatter_by_index_put": scatter_by_index_put,
        "autobucketing": autobucketing,
    }


# ---------------------------------------------------------------------------
# Inference config builder
# ---------------------------------------------------------------------------


def build_inference_config(
    model_path,
    tp_degree=4,
    batch_size=1,
    seq_len=2048,
    n_positions=4096,
    vision_seq_len=4096,
    tkg_buckets=None,
    enable_tkg_kernel=True,
):
    """Build a PixtralInferenceConfig for Ministral-3-14B.

    Args:
        model_path: Path to HuggingFace checkpoint directory.
        tp_degree: Tensor parallelism degree. Default 4 for trn2.3xlarge.
        batch_size: Batch size. Default 1.
        seq_len: Maximum text sequence length. Default 2048.
        n_positions: Maximum position embeddings. Default 4096.
        vision_seq_len: Maximum vision sequence length. Default 4096.
        tkg_buckets: Token generation buckets. Default [256, 512, 1024, seq_len].
        enable_tkg_kernel: Enable the fused NKI TKG attention kernel. Default True.

    Returns:
        PixtralInferenceConfig instance.
    """
    nxdi = _get_nxdi_imports()
    NeuronConfig = nxdi["NeuronConfig"]
    PixtralInferenceConfig = nxdi["PixtralInferenceConfig"]

    if tkg_buckets is None:
        tkg_buckets = [256, 512, 1024, seq_len]

    with open(os.path.join(model_path, "config.json")) as f:
        full_config = json.load(f)
    text_cfg = full_config["text_config"]
    vision_cfg = full_config["vision_config"]
    rope_params = text_cfg.get("rope_parameters", {})

    text_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        n_positions=n_positions,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        enable_bucketing=True,
        token_generation_buckets=tkg_buckets,
        fused_qkv=True,
        qkv_nki_kernel_enabled=True,
        attn_block_tkg_nki_kernel_enabled=enable_tkg_kernel,
        attn_block_tkg_nki_kernel_cache_update=enable_tkg_kernel,
    )
    vision_neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=vision_seq_len,
        torch_dtype=torch.bfloat16,
        enable_bucketing=True,
        on_device_sampling_config=None,
    )

    def custom_load_config(config_obj):
        """Populate PixtralInferenceConfig from Ministral3 config.json.

        Ministral3 is not registered in HuggingFace AutoConfig, so we build
        the text_config and vision_config SimpleNamespace objects manually.
        """
        rope_scaling_dict = {
            "rope_type": rope_params.get("rope_type", "yarn"),
            "type": rope_params.get("type", "yarn"),
            "factor": rope_params.get("factor", 16.0),
            "beta_fast": rope_params.get("beta_fast", 32.0),
            "beta_slow": rope_params.get("beta_slow", 1.0),
            "original_max_position_embeddings": rope_params.get(
                "original_max_position_embeddings", 16384
            ),
            "mscale": rope_params.get("mscale", 1.0),
            "mscale_all_dim": rope_params.get("mscale_all_dim", 1.0),
        }
        tc = SimpleNamespace(
            hidden_size=text_cfg["hidden_size"],
            num_attention_heads=text_cfg["num_attention_heads"],
            num_hidden_layers=text_cfg["num_hidden_layers"],
            num_key_value_heads=text_cfg["num_key_value_heads"],
            vocab_size=text_cfg["vocab_size"],
            max_position_embeddings=text_cfg["max_position_embeddings"],
            rope_theta=rope_params.get("rope_theta", 1e9),
            rope_scaling=rope_scaling_dict,
            rms_norm_eps=text_cfg["rms_norm_eps"],
            hidden_act=text_cfg["hidden_act"],
            intermediate_size=text_cfg["intermediate_size"],
            head_dim=text_cfg.get("head_dim", 128),
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=full_config.get("tie_word_embeddings", False),
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        vc = SimpleNamespace(
            hidden_size=vision_cfg["hidden_size"],
            num_attention_heads=vision_cfg["num_attention_heads"],
            num_hidden_layers=vision_cfg["num_hidden_layers"],
            num_channels=vision_cfg["num_channels"],
            patch_size=vision_cfg["patch_size"],
            image_size=vision_cfg["image_size"],
            rope_theta=vision_cfg.get("rope_parameters", {}).get("rope_theta", 10000.0),
            head_dim=vision_cfg.get("head_dim", 64),
            intermediate_size=vision_cfg.get("intermediate_size", 4096),
            hidden_act=vision_cfg.get("hidden_act", "silu"),
        )
        config_obj.text_config = tc
        config_obj.vision_config = vc
        config_obj.multimodal_projector_bias = False
        config_obj.projector_hidden_act = "gelu"
        config_obj.vision_feature_layer = -1
        config_obj.spatial_merge_size = full_config.get(
            "spatial_merge_size", SPATIAL_MERGE_SIZE
        )
        config_obj.image_token_index = IMAGE_TOKEN_ID
        config_obj._name_or_path = model_path
        config_obj.output_attentions = False
        config_obj.output_hidden_states = False
        config_obj.return_dict = True

    return PixtralInferenceConfig(
        text_neuron_config=text_neuron_config,
        vision_neuron_config=vision_neuron_config,
        load_config=custom_load_config,
    )


# ---------------------------------------------------------------------------
# Model classes (built lazily via _build_model_classes)
# ---------------------------------------------------------------------------


def _build_model_classes():
    """Build the actual model classes with NxDI base classes.

    Must be called after apply_shard_over_heads_patch() and apply_multi_kv_tkg_patch().
    Returns a dict of class objects.
    """
    nxdi = _get_nxdi_imports()

    NeuronLlamaModel = nxdi["NeuronLlamaModel"]
    NeuronBaseForImageToText = nxdi["NeuronBaseForImageToText"]
    ImageToTextModelWrapper = nxdi["ImageToTextModelWrapper"]
    VISION_ENCODER_MODEL_TAG = nxdi["VISION_ENCODER_MODEL_TAG"]
    PixtralInferenceConfig = nxdi["PixtralInferenceConfig"]
    NeuronPixtralVisionModel = nxdi["NeuronPixtralVisionModel"]
    PixtralVisionModelWrapper = nxdi["PixtralVisionModelWrapper"]
    generate_positions_from_mask = nxdi["generate_positions_from_mask"]
    pad_positions = nxdi["pad_positions"]
    pad_vision_embeddings = nxdi["pad_vision_embeddings"]
    scatter_by_index_put = nxdi["scatter_by_index_put"]
    autobucketing = nxdi["autobucketing"]

    class _NeuronLeanstralTextModel(NeuronLlamaModel):
        """Llama text model with vision embedding injection."""

        def encode_vision_to_input(self, inputs_embeds, vision_embeddings, vision_mask):
            return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)

    class _NeuronLeanstralVisionModel(NeuronPixtralVisionModel):
        """Pixtral ViT without built-in projector."""

        def __init__(self, config):
            super().__init__(config)
            if hasattr(self, "multi_modal_projector"):
                del self.multi_modal_projector

        def forward(self, patch_embeds, attention_mask, position_ids):
            patch_embeds = self.vision_patch_conv_linear(patch_embeds)
            patch_embeds = self.vision_ln_pre(patch_embeds)
            return self.vision_transformer(
                patch_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

    class _LeanstralVisionWrapper(PixtralVisionModelWrapper):
        """Fix unpad slice for no-projector output."""

        def pad_inputs(self, patch_embeds, attention_mask, position_ids):
            result = super().pad_inputs(patch_embeds, attention_mask, position_ids)
            if self.original_patch_embed_slices is not None:
                self.original_patch_embed_slices[-1][-1] = (
                    self.config.vision_config.hidden_size
                )
            return result

    class _NeuronLeanstralForCausalLM(NeuronBaseForImageToText):
        """Full VL model: Pixtral vision + Llama text + CPU projector."""

        text_model_cls = _NeuronLeanstralTextModel
        vision_model_cls = _NeuronLeanstralVisionModel
        text_model_wrapper = ImageToTextModelWrapper
        vision_model_wrapper = _LeanstralVisionWrapper

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
            self.cpu_projector = load_cpu_projector(
                model_path,
                vision_hidden_size=self.config.vision_config.hidden_size,
                text_hidden_size=self.config.text_config.hidden_size,
            )

        @classmethod
        def get_config_cls(cls):
            return PixtralInferenceConfig

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
            """Override to drop deepstack_vision_embeds (25th arg).

            ImageToTextModelWrapper traces the model with 24 positional args.
            The base class passes 25 (including deepstack_vision_embeds), which
            causes an arg-count mismatch at runtime. We drop it here.
            """
            if rotary_position_ids is None:
                rotary_position_ids = torch.empty(0)

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
                    torch.empty(0, dtype=self.text_config.neuron_config.torch_dtype),
                    torch.empty(0, dtype=torch.bool),
                )
                is_run_on_neuron = self.token_generation_model.is_neuron()

            return outputs, is_run_on_neuron

        def get_vision_compiler_args(self):
            return (
                "--enable-saturate-infinity --auto-cast=none --model-type=transformer "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2 ' -O1 "
                "--internal-hlo2tensorizer-options='--verify-hlo=true'"
            )

        def get_compiler_args(self):
            return (
                "--enable-saturate-infinity --auto-cast=none --model-type=transformer "
                "--tensorizer-options='--enable-ccop-compute-overlap "
                "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma ' -O1 "
                "--internal-hlo2tensorizer-options='--verify-hlo=true'"
            )

        def enable_vision_encoder(
            self, enable_wlt_optimization=True, **model_init_kwargs
        ):
            new_config = copy.deepcopy(self.config)
            if new_config.vision_config.neuron_config.enable_bucketing:
                vc_nc = new_config.vision_config.neuron_config
                if vc_nc.buckets == [vc_nc.seq_len] or vc_nc.buckets is None:
                    if vc_nc.seq_len > 1024:
                        vc_nc.buckets = autobucketing.generate_buckets(
                            1024, vc_nc.seq_len
                        )
                    else:
                        vc_nc.buckets = [vc_nc.seq_len]
            new_config.neuron_config = copy.deepcopy(
                new_config.vision_config.neuron_config
            )
            self.vision_encoder_model = self.vision_model_wrapper(
                config=new_config,
                model_cls=self.vision_model_cls,
                tag=VISION_ENCODER_MODEL_TAG,
                compiler_args=self.get_vision_compiler_args(),
                model_init_kwargs=model_init_kwargs,
                priority_model_idx=(0 if enable_wlt_optimization else None),
                pipeline_execution=True,
                return_ranked_to_cpu=True,
            )
            self.vision_models.append(self.vision_encoder_model)

        def get_required_kwargs(self):
            return ["pixel_values", "vision_mask", "image_sizes"]

        def get_padding_length(self, input_ids):
            buckets = self.context_encoding_model.config.neuron_config.buckets
            for val in buckets:
                if val >= input_ids.shape[1]:
                    return val
            raise ValueError(
                f"No bucket found for input_ids length {input_ids.shape[1]}. "
                f"Available buckets: {buckets}"
            )

        def forward_atomic_prefill(
            self,
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
            pixel_values,
            vision_mask,
            image_sizes,
        ):
            """Run vision encoder + CPU projector + text prefill for one batch item."""
            if image_sizes is None:
                img_h, img_w = pixel_values.shape[2], pixel_values.shape[3]
                image_sizes = torch.tensor([[img_h, img_w]], dtype=torch.int32)

            if position_ids is None:
                position_ids = torch.arange(
                    input_ids.shape[1], dtype=torch.int32
                ).unsqueeze(0)

            if seq_ids is None:
                seq_ids = torch.zeros(input_ids.shape[0], dtype=torch.int32)

            if sampling_params is None:
                sampling_params = torch.zeros(
                    input_ids.shape[0], 3, dtype=torch.float32
                )

            if vision_mask is None:
                vision_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).to(torch.bool)
            vision_mask = generate_positions_from_mask(vision_mask.squeeze())

            # 1. Vision encoder (on Neuron)
            vision_embeddings = self.vision_encoder_model(
                pixel_values.to(self.vision_config.neuron_config.torch_dtype),
                image_sizes,
            )

            # 2. CPU projector: RMSNorm -> PatchMerger -> MLP
            img_h = image_sizes[0, 0].item()
            img_w = image_sizes[0, 1].item()
            with torch.no_grad():
                projected = self.cpu_projector(vision_embeddings, img_h, img_w)
            vision_embeddings = projected.unsqueeze(0).to(
                self.text_config.neuron_config.torch_dtype
            )

            # 3. Pad to text bucket
            pad_limit = self.get_padding_length(input_ids)
            vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))
            vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask,
                deepstack_vision_embeds=None,
            )

        def check_empty_pixel_values(self, pixel_values):
            if pixel_values is None:
                return True
            if isinstance(pixel_values, torch.Tensor):
                return pixel_values.sum() == 0
            if isinstance(pixel_values, list):
                return all(pv.sum() == 0 for pv in pixel_values)
            return True

        def get_batch_line_mm_input(self, mm_input, index):
            if mm_input is None:
                return None
            if isinstance(mm_input, list):
                return mm_input[index]
            if isinstance(mm_input, torch.Tensor):
                return mm_input[index].unsqueeze(0)
            return None

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            seq_ids=None,
            sampling_params=None,
            pixel_values=None,
            vision_mask=None,
            image_sizes=None,
            **kwargs,
        ):
            """Forward pass supporting both VL prefill and text-only decode."""
            if input_ids.shape[-1] > 1 and not self.check_empty_pixel_values(
                pixel_values
            ):
                # VL prefill: process each batch item through vision pipeline
                outputs = []
                for index in range(input_ids.shape[0]):
                    outputs.append(
                        self.forward_atomic_prefill(
                            input_ids[index].unsqueeze(0),
                            attention_mask[index].unsqueeze(0)
                            if attention_mask is not None
                            else None,
                            position_ids[index].unsqueeze(0)
                            if position_ids is not None
                            else None,
                            seq_ids[index].unsqueeze(0)
                            if seq_ids is not None
                            else None,
                            sampling_params[index].unsqueeze(0)
                            if sampling_params is not None
                            else None,
                            self.get_batch_line_mm_input(pixel_values, index),
                            self.get_batch_line_mm_input(vision_mask, index),
                            self.get_batch_line_mm_input(image_sizes, index),
                        )
                    )
                from transformers.modeling_outputs import CausalLMOutputWithPast

                logits = (
                    torch.cat([o.logits for o in outputs], dim=0)
                    if outputs[0].logits is not None
                    else None
                )
                tokens_list = [
                    o.tokens
                    for o in outputs
                    if hasattr(o, "tokens") and o.tokens is not None
                ]
                tokens = torch.cat(tokens_list, dim=0) if tokens_list else None
                out = CausalLMOutputWithPast(logits=logits, hidden_states=[])
                if tokens is not None:
                    out.tokens = tokens
                return out
            else:
                # Text-only prefill or TKG decode
                pad_limit = (
                    self.get_padding_length(input_ids) if input_ids.shape[-1] > 1 else 1
                )
                vision_embeddings_dummy, vision_mask_dummy = (
                    ImageToTextModelWrapper.get_dummy_vision_inputs(
                        config=self.text_config,
                        input_ids=input_ids,
                        n_active_tokens=pad_limit,
                        fill_value=(pad_limit - 1),
                    )
                )
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    vision_embeddings=vision_embeddings_dummy,
                    vision_mask=vision_mask_dummy,
                )

        @staticmethod
        def load_hf_model(model_path, **kwargs):
            """Load HuggingFace checkpoint as a state_dict wrapper."""
            state_dict = {}
            safetensors_files = sorted(
                f
                for f in os.listdir(model_path)
                if f.endswith(".safetensors") and "consolidated" not in f
            )
            for fname in safetensors_files:
                with safe_open(os.path.join(model_path, fname), framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

            class _StateDict:
                def __init__(self, sd):
                    self._sd = sd

                def state_dict(self):
                    return self._sd

            return _StateDict(state_dict)

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict, inference_config):
            """Convert HuggingFace Ministral3 weights to NxDI format.

            Handles: FP8 dequantization, text/vision split, QKV fusion,
            attention key remapping, rank utilities, vision key remapping.
            """
            # Phase 1: FP8 dequantize
            clean = {}
            fp8_count = 0
            for key, val in state_dict.items():
                if ".activation_scale" in key or ".weight_scale_inv" in key:
                    continue
                if val.dtype == torch.float8_e4m3fn:
                    scale_key = key.replace(".weight", ".weight_scale_inv")
                    scale = (
                        state_dict[scale_key].float()
                        if scale_key in state_dict
                        else torch.tensor(1.0)
                    )
                    clean[key] = (val.float() * scale).to(torch.bfloat16)
                    fp8_count += 1
                else:
                    clean[key] = val
            logger.info("Dequantized %d FP8 tensors", fp8_count)

            # Phase 2: Split text vs vision
            text_dict = {}
            for key, val in clean.items():
                if key.startswith("language_model.model."):
                    text_dict[key.replace("language_model.model.", "")] = val
                elif key.startswith("language_model."):
                    text_dict[key.replace("language_model.", "")] = val

            # Phase 3: Remap attention keys
            if inference_config.text_config.neuron_config.fused_qkv:
                num_layers = inference_config.text_config.num_hidden_layers
                for i in range(num_layers):
                    q_key = f"layers.{i}.self_attn.q_proj.weight"
                    k_key = f"layers.{i}.self_attn.k_proj.weight"
                    v_key = f"layers.{i}.self_attn.v_proj.weight"
                    if q_key in text_dict and k_key in text_dict and v_key in text_dict:
                        fused = torch.cat(
                            [text_dict[q_key], text_dict[k_key], text_dict[v_key]],
                            dim=0,
                        )
                        text_dict[f"layers.{i}.self_attn.Wqkv.weight"] = fused
                        del text_dict[q_key], text_dict[k_key], text_dict[v_key]
            else:
                remap = {
                    ".self_attn.q_proj.": ".self_attn.qkv_proj.q_proj.",
                    ".self_attn.k_proj.": ".self_attn.qkv_proj.k_proj.",
                    ".self_attn.v_proj.": ".self_attn.qkv_proj.v_proj.",
                }
                remapped = {}
                for key, val in text_dict.items():
                    new_key = key
                    for pat, rep in remap.items():
                        if pat in new_key:
                            new_key = new_key.replace(pat, rep)
                            break
                    remapped[new_key] = val
                text_dict = remapped

            # Phase 4: Add rank utilities
            tp = inference_config.text_config.neuron_config.tp_degree
            for i in range(inference_config.text_config.num_hidden_layers):
                text_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                    0, tp, dtype=torch.int32
                )
            text_dict["rank_util.rank"] = torch.arange(0, tp, dtype=torch.int32)

            # Phase 5: Vision keys
            vision_dict = {}
            for key, val in clean.items():
                if key.startswith("vision_tower."):
                    new_key = key.replace("vision_tower.", "vision_")
                    vision_dict[new_key] = val.to(
                        inference_config.vision_config.neuron_config.torch_dtype
                    )
            # Reshape patch conv weight: Conv2d -> Linear equivalent
            patch_key = "vision_patch_conv.weight"
            if patch_key in vision_dict:
                vision_dict["vision_patch_conv_linear.weight"] = vision_dict.pop(
                    patch_key
                ).reshape(
                    -1,
                    inference_config.vision_config.num_channels
                    * inference_config.vision_config.patch_size**2,
                )

            merged = {**text_dict, **vision_dict}
            logger.info(
                "State dict converted: %d text keys, %d vision keys",
                len(text_dict),
                len(vision_dict),
            )
            return merged

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            pass

    return {
        "NeuronLeanstralTextModel": _NeuronLeanstralTextModel,
        "NeuronLeanstralVisionModel": _NeuronLeanstralVisionModel,
        "LeanstralVisionWrapper": _LeanstralVisionWrapper,
        "NeuronLeanstralForCausalLM": _NeuronLeanstralForCausalLM,
    }


# ---------------------------------------------------------------------------
# Public API: NeuronLeanstralForCausalLM
# ---------------------------------------------------------------------------
# The class is built lazily to allow patches to be applied first.

_model_classes = None


def get_model_cls():
    """Get the NeuronLeanstralForCausalLM class, applying patches if needed.

    Returns the model class ready for instantiation:
        model = get_model_cls()(model_path, inference_config)
    """
    global _model_classes
    if _model_classes is None:
        # Ensure patches are applied before building classes
        apply_shard_over_heads_patch()
        apply_multi_kv_tkg_patch()
        _model_classes = _build_model_classes()
    return _model_classes["NeuronLeanstralForCausalLM"]
