import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from safetensors.torch import save_file
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_vision import (
    Qwen2VLVisionRotaryEmbedding,
)
from neuronx_distributed_inference.models.qwen3_vl.utils.slicing import slice_by_image_hw
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
from neuronx_distributed_inference.modules.padding import (
    pad_tensor,
    unpad_tensor,
)
from neuronx_distributed_inference.modules.checkpoint import load_state_dict
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NeuronQwen3VLVisionAttention(NeuronAttentionBase):
    """The same as Neuron3VLAttention other than config param name"""

    def __init__(self, config):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            head_dim=config.hidden_size // config.num_heads,
            sequence_parallel_enabled=False,
            rotary_emb=Qwen2VLVisionRotaryEmbedding(),
            qkv_bias=True,
            o_bias=True,
        )

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        self._position_embeddings = position_embeddings
        try:
            return super().forward(hidden_states, attention_mask=attention_mask, **kwargs)
        finally:
            self._position_embeddings = None

    def apply_rotary_embedding(
        self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
    ):
        # Expected shape: Q.shape = K.shape = (bs=1, num_head_per_tp_rank, vision_seq_len, head_dim)
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, self._position_embeddings)
            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        return Q, K, cos_cache, sin_cache


class NeuronQwen3VLVisionMLP(nn.Module):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            gather_output=False,
            dtype=config.neuron_config.torch_dtype,
        )
        self.linear_fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            dtype=config.neuron_config.torch_dtype,
            reduce_dtype=config.neuron_config.rpl_reduce_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class NeuronQwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = NeuronQwen3VLVisionAttention(config=config)
        self.mlp = NeuronQwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states
            + self.attn(
                self.norm1(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            ).hidden_states
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NeuronQwen3VLVisionModel(nn.Module):

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        logger.info(f"in NeuronQwen3VLVisionModel self.vision_config {vars(self.vision_config)}")

        self.patch_size = self.vision_config.patch_size

        self.blocks = nn.ModuleList(
            [NeuronQwen3VLVisionBlock(self.vision_config) for _ in range(self.vision_config.depth)]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config=self.vision_config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = self.vision_config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=self.vision_config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.vision_config.deepstack_visual_indexes))
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        vision_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            rotary_pos_emb (`torch.Tensor` of shape `(seq_len, head_dim//2)`):
                Precomputed rotary position embeddings (cos/sin frequencies) for encoding relative positional information in attention.
            vision_attention_mask (`torch.Tensor` of shape `(seq_len, seq_len)`):
                Square mask for different images for each image to attend to itself.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        # Reshape to add batch dim to match with hidden_states reshape below
        emb = emb.unsqueeze(0)
        position_embeddings = (emb.cos(), emb.sin())

        # Insert batch dim = 1; multiple images are merged into a single flattened sequence
        hidden_states = hidden_states.unsqueeze(0)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            logger.debug(f"\ncurrent layer {layer_num}")
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=vision_attention_mask,
            )
            if layer_num in self.deepstack_visual_indexes:
                logger.debug(f"Use layer{layer_num}'s hidden_states for deepstack")
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                logger.debug(f"deepstack_feature shape {deepstack_feature.shape}")
                deepstack_feature_lists.append(deepstack_feature)
                logger.debug(f"deepstack_feature_lists len {len(deepstack_feature_lists)}")
        hidden_states = self.merger(hidden_states)
        logger.debug(
            f"\nNeuronQwen3VLVisionModel returning: hidden_states.shape {hidden_states.shape}, deepstack_feature_lists len {len(deepstack_feature_lists)} shape {deepstack_feature_lists[0].shape}"
        )

        return hidden_states, deepstack_feature_lists


class NeuronQwen3VLVisionModelWrapper(ModelWrapper):
    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        # FIXME: when setting to True only return the first output hidden_states, no deepstack_feature_lists
        # but in vision+text mode we want to eliminate Neuron-CPU data transfer time
        # by setting pipeline_execution = True and return_ranked_to_cpu = False
        pipeline_execution: bool = False,
        return_ranked_to_cpu: bool = False,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            pipeline_execution,
            return_ranked_to_cpu,
            model_init_kwargs,
        )
        self.vision_config = config.vision_config
        self.num_grid_per_side = int(config.vision_config.num_position_embeddings**0.5)
        self.spatial_merge_size = config.vision_config.spatial_merge_size

        head_dim = self.vision_config.hidden_size // self.vision_config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        sd = load_state_dict(config._name_or_path)

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=self.vision_config,
        )
        self.patch_embed.proj.weight.data = sd["model.visual.patch_embed.proj.weight"].to(config.vision_config.neuron_config.torch_dtype)
        self.patch_embed.proj.bias.data = sd["model.visual.patch_embed.proj.bias"].to(config.vision_config.neuron_config.torch_dtype)

        self.pos_embed = nn.Embedding(
            self.vision_config.num_position_embeddings,
            self.vision_config.hidden_size,
            dtype=self.vision_config.neuron_config.torch_dtype,
        )
        self.pos_embed.weight.data = sd["model.visual.pos_embed.weight"].to(config.vision_config.neuron_config.torch_dtype)

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(
            idx_list, dtype=torch.long, device=self.pos_embed.weight.device
        )  # shape (4, num_image * seq_len_per_image)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )  # shape (4, num_image * seq_len_per_image)
        pos_embeds = (
            self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        )  # shape (4, num_image * seq_len_per_image, hidden_size=1152)
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )  # shape (num_image * seq_len_per_image, hidden_size=1152)

        # Huggingface reference impl hit compilation error due to dynamicity
        # F xla/hlo/ir/hlo_instruction.cc:2285] Check failed: operand->shape().is_unbounded_dynamic()
        # || ShapeUtil::StaticExtentProduct(shape) == ShapeUtil::StaticExtentProduct(operand->shape()) shape: f32[4,1] operand: f32[4,40000,1152]
        # patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_list = slice_by_image_hw(patch_pos_embeds, grid_hs, grid_ws)

        patch_pos_embeds_permute = []
        merge_size = self.config.vision_config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds_list, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        inputs = []
        # bucket is num of images
        for vision_seq_len in self.config.vision_config.neuron_config.buckets:
            pixel_values = torch.ones(
                [
                    vision_seq_len,
                    self.config.vision_config.hidden_size,
                ],
                dtype=self.config.vision_config.neuron_config.torch_dtype,
            )
            rotary_pos_emb = torch.ones(
                [
                    vision_seq_len,
                    # The frequency table allocates head_dim // 4 values per spatial dimension (row & col),
                    # resulting in rotary_dim = head_dim // 2 after concatenation
                    (self.config.vision_config.hidden_size // self.config.vision_config.num_heads // 2),
                ],
                dtype=torch.float32
            )
            vision_attention_mask = torch.ones(
                [
                    vision_seq_len,
                    vision_seq_len,
                ],
                dtype=torch.int32
            )

            inputs.append((pixel_values, rotary_pos_emb, vision_attention_mask))
        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def get_target_bucket(
        self,
        patch_embeds,
    ) -> int:
        """
        Override ModelWrapper.get_target_bucket().
        Get the closest bucket size.

        Returns:
            int: target bucket size
        """

        patch_seq_len = patch_embeds.shape[0]
        # InferenceConfig would use seq_len if buckets are not specified. Validation is done there.
        largest_bucket = self.config.vision_config.neuron_config.buckets[-1]

        # validate the input patch_seq_len does not exceed largest bucket
        assert patch_seq_len <= largest_bucket, \
            f"Total number of image patches {patch_seq_len} exceeds largest bucket ({largest_bucket})"

        # return closest bucket
        for i, bucket in enumerate(self.config.vision_config.neuron_config.buckets):
            if patch_seq_len <= bucket:
                logger.info(f"Routing patch_seq_len {patch_seq_len} to bucket size {bucket}")
                return bucket

    @staticmethod
    def create_vision_attention_mask(image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Create a block diagonal attention mask where each image attends only to itself.

        Args:
            image_grid_thw: Tensor of shape (num_images, 3) where each row is [T, H, W]
                            representing temporal, height, width dimensions

        Returns:
            attention_mask: Boolean tensor of shape (total_tokens, total_tokens)
                            True where attention is allowed, False where blocked
        """
        # Calculate number of tokens per image
        tokens_per_image = image_grid_thw.prod(dim=1)  # T * H * W for each image
        total_tokens = tokens_per_image.sum().item()

        # Initialize mask with zeros (no attention)
        attention_mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)

        # Fill in block diagonal by setting each image's token range to attend to itself
        start_idx = 0
        for num_tokens in tokens_per_image:
            num_tokens = num_tokens.item()
            end_idx = start_idx + num_tokens
            attention_mask[start_idx:end_idx, start_idx:end_idx] = True
            start_idx = end_idx

        return attention_mask.to(torch.int32)

    def forward(self, pixel_values, grid_thw):
        """
        Override ModelWrapper.forward().
        """

        pixel_values = pixel_values.to(self.config.vision_config.neuron_config.torch_dtype)
        grid_thw = grid_thw.to(torch.int32)
        vision_attention_mask = NeuronQwen3VLVisionModelWrapper.create_vision_attention_mask(grid_thw)

        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )

        # To support dynamic image shapes, we project image patches and compute rotary positional embeddings on CPU
        # in model wrapper, before calling VisionEncoder's model forward. This is avoid bucketing on grid_thw's num_images since its shape is [num_images, 3]. Hence, we only bucket on sequence length of flattened images.
        pixel_values = self.patch_embed(pixel_values)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        pixel_values = pixel_values + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(torch.float)  # shape [seq_len, 36]

        # pad inputs
        target_vision_bucket = self.get_target_bucket(pixel_values)
        padded_pixel_values, original_idx_slices = pad_tensor(
            pixel_values, (target_vision_bucket, pixel_values.shape[1])
        )
        padded_rotary_pos_emb, _ = pad_tensor(rotary_pos_emb, (target_vision_bucket, rotary_pos_emb.shape[1]))
        vision_attention_mask, _ = pad_tensor(vision_attention_mask, (target_vision_bucket, target_vision_bucket), pad_value=0)
        # forward
        hidden_states, deepstack_feature_list = self._forward(padded_pixel_values, padded_rotary_pos_emb, vision_attention_mask)
        # Unpad outputs to match HF Qwen3VLVisionModel output
        # Qwen3VLVisionPatchMerger downscale seqlen and upscale hidden size
        divisor = self.vision_config.spatial_merge_size**2
        original_idx_slices[0] = [x // divisor for x in original_idx_slices[0]]
        original_idx_slices[1][1] = self.vision_config.out_hidden_size
        unpadded_hidden_states = unpad_tensor(hidden_states, original_idx_slices)
        unpadded_deepstack_feature_list = []
        for feat in deepstack_feature_list:
            unpadded_feat = unpad_tensor(feat, original_idx_slices)
            unpadded_deepstack_feature_list.append(unpadded_feat)
        return unpadded_hidden_states, unpadded_deepstack_feature_list


class NeuronQwen3VLForImageEncoding(NeuronApplicationBase):
    """
    Neuron Application class for Qwen3VL image encoding case.
    Wraps NeuronQwen3VLVisionModel with Neuron specific functionalities such as compile and load.
    """

    _model_cls = NeuronQwen3VLVisionModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        # will only have one model one tag
        # after compilation, in /tmp/nxd_model,
        # you should only see one folder called f"self._model_cls.__name__"
        self.models.append(self.model)

    def get_model_wrapper_cls(self):
        return NeuronQwen3VLVisionModelWrapper

    def forward(self, pixel_values, image_grid_thw):
        return self.models[0](pixel_values, image_grid_thw)

    def get_compiler_args(self):
        compiler_args = "--auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 ' -O1 \
                --internal-max-instruction-limit=15000000"
        logger.info(f"Compiling {self._model_cls.__name__} vision model with args: {compiler_args}")
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        class hf_vision_model(torch.nn.Module):
            def __init__(self, model_path, **kwargs):
                super().__init__()
                # config
                self.hf_config = Qwen3VLConfig.from_pretrained(model_path, **kwargs)
                hf_vision_config = Qwen3VLVisionConfig(**vars(self.hf_config.vision_config))
                self.visual = Qwen3VLVisionModel._from_config(hf_vision_config)

            def forward(self, pixel_values, image_grid_thw):
                return self.visual(pixel_values, image_grid_thw)

            def save_pretrained(self, save_model_path):
                self.hf_config.save_pretrained(save_model_path)
                save_file(self.state_dict(), os.path.join(save_model_path, "model.safetensors"))

            def load_state_dict(self, state_dict, **kwargs):
                # for loading HF checkpoint
                new_state_dict = {}
                for key, value in state_dict.items():
                    if "model." in key:
                        key = key.replace("model.", "")
                    new_state_dict[key] = value.clone().detach().contiguous()

                del state_dict
                super().load_state_dict(new_state_dict, **kwargs)

        hf_model = hf_vision_model(model_path, **kwargs)

        return hf_model

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                key = key.replace("visual.", "")
                if ".attn.qkv." in key:
                    key = key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
                elif ".attn.proj." in key:
                    key = key.replace(".attn.proj.", ".attn.o_proj.")
            new_state_dict[key] = value.clone().detach().contiguous()

        del state_dict
        return new_state_dict

    @classmethod
    def get_config_cls(cls):
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLInferenceConfig,
        )

        return Qwen3VLInferenceConfig

    @classmethod
    def prepare_input_args(cls, prompts, images, processor, role="user", config=None):
        """
        Prepare input arguments for Qwen3 VL model.

        Args:
            prompts: str or List[str] - Single prompt or list of prompts
            images: Various formats supported:
                - None: No images for any prompt
                - List[List[Image]]: List of image lists, one per prompt
                - List[Image]: Single list of images (only valid for batch_size=1)
                - Single Image: One image (only valid for batch_size=1)
            processor: The Qwen3 VL processor
            role: Role for the message (default: "user")
            config: Optional config

        Returns:
            input_ids, attention_mask, vision_inputs
        """

        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_size = len(prompts)

        # Normalize images to List[List[Image]] or List[None]
        images = _normalize_images(images, batch_size)

        # Build messages for each batch item
        batch_messages = []
        for prompt, batch_images in zip(prompts, images):
            content = _build_content(prompt, batch_images)
            messages = [{"role": role, "content": content}]
            batch_messages.append(messages)

        # Apply chat template for batch
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # Required for batching
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Handle vision inputs (may not exist for text-only batches)
        vision_inputs = {}
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            vision_inputs["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            vision_inputs["image_grid_thw"] = inputs.image_grid_thw

        return input_ids, attention_mask, vision_inputs


def _normalize_images(images, batch_size):
    """Normalize images input to List[List[Image] | None]."""
    from PIL import Image

    if images is None:
        # No images for any prompt
        return [None] * batch_size

    if not isinstance(images, list):
        # Single image provided
        if batch_size != 1:
            raise ValueError(
                f"Single image provided but batch_size={batch_size}. "
                "Provide a list of image lists for batch processing."
            )
        return [[images]]

    if len(images) == 0:
        return [None] * batch_size

    # Check if it's a flat list of images (for batch_size=1)
    # or a list of image lists
    first_item = images[0]
    is_flat_list = isinstance(first_item, str) or isinstance(first_item, Image.Image)

    if is_flat_list:
        # Flat list of images for single prompt
        if batch_size != 1:
            raise ValueError(
                f"Flat list of images provided but batch_size={batch_size}. "
                "Provide a list of image lists for batch processing."
            )
        return [images]

    # List of image lists (or None entries)
    if len(images) != batch_size:
        raise ValueError(
            f"Number of image sets ({len(images)}) must match " f"number of prompts ({batch_size})"
        )
    return images


def _build_content(prompt, batch_images):
    """Build content list for a single message."""
    import base64
    from io import BytesIO

    from PIL import Image

    content = []

    if batch_images is not None:
        if not isinstance(batch_images, list):
            batch_images = [batch_images]

        for image_or_path in batch_images:
            if isinstance(image_or_path, str):
                # File path
                with open(image_or_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            elif isinstance(image_or_path, Image.Image):
                # PIL Image
                buffer = BytesIO()
                image_or_path.save(buffer, format="JPEG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            else:
                raise TypeError(
                    f"Invalid image_data type: {type(image_or_path)}. "
                    "Expected str (file path) or PIL.Image."
                )
            content.append({"type": "image", "url": f"data:image/jpeg;base64,{base64_image}"})

    content.append({"type": "text", "text": prompt})
    return content
