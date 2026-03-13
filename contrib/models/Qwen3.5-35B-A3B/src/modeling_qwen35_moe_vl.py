"""
Qwen3.5-35B-A3B Vision-Language Model Orchestrator for NeuronX Distributed Inference.

This is the top-level VL model that wires together:
- The vision encoder (modeling_qwen35_moe_vision.py)
- The text decoder (modeling_qwen35_moe.py, modified for vision injection)

It handles:
- Multimodal RoPE (mRoPE) with interleaved layout
- Vision embedding injection via scatter_by_index_put
- Separate compilation and loading of vision and text models
- The CTE+TKG generation loop with vision inputs

Architecture follows the NxDI NeuronBaseForImageToText pattern established
by Qwen3-VL in SDK 2.28, adapted for Qwen3.5 MoE's unique features:
- No deepstack (Qwen3.5 does not use intermediate vision feature injection)
- DeltaNet linear attention layers in the text decoder
- MoE FFN layers in the text decoder
- Interleaved mRoPE (THWTHW... layout) instead of Qwen3-VL's section-based layout
"""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# NxDI imports
try:
    from neuronx_distributed_inference.models.image_to_text_model_base import (
        ImageToTextInferenceConfig,
        NeuronBaseForImageToText,
    )
    from neuronx_distributed_inference.models.config import NeuronConfig

    HAS_NXDI_VL = True
except ImportError:
    HAS_NXDI_VL = False
    logger.warning("NxDI VL base classes not available -- VL model requires SDK 2.28+")

# Local imports
try:
    from .modeling_qwen35_moe import (
        NeuronQwen35MoeForCausalLM,
        NeuronQwen35MoeModel,
        Qwen35MoeInferenceConfig,
        Qwen35ModelWrapper,
    )
    from .modeling_qwen35_moe_vision import (
        NeuronQwen35VisionModel,
        NeuronQwen35VisionModelWrapper,
    )
except ImportError:
    from modeling_qwen35_moe import (
        NeuronQwen35MoeForCausalLM,
        NeuronQwen35MoeModel,
        Qwen35MoeInferenceConfig,
        Qwen35ModelWrapper,
    )
    from modeling_qwen35_moe_vision import (
        NeuronQwen35VisionModel,
        NeuronQwen35VisionModelWrapper,
    )


def get_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    image_token_id=248056,
    video_token_id=248057,
    vision_start_token_id=248053,
    spatial_merge_size=2,
):
    """Compute 3D multimodal RoPE position IDs for Qwen3.5.

    Returns position_ids of shape (3, batch_size, seq_len) where:
    - Axis 0: temporal position
    - Axis 1: height position
    - Axis 2: width position

    For text tokens, all 3 axes have the same sequential position.
    For vision tokens, each axis encodes the spatial/temporal grid position.

    Also returns rope_deltas for use during TKG decoding.

    Adapted from HuggingFace Qwen3_5MoeModel.get_rope_index().
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0
        )
        video_grid_thw[:, 0] = 1

    image_grid_thw_list = (
        image_grid_thw.tolist() if image_grid_thw is not None else None
    )
    video_grid_thw_list = (
        video_grid_thw.tolist() if video_grid_thw is not None else None
    )

    mrope_position_deltas = []
    total_input_ids = input_ids

    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)

    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    image_index, video_index = 0, 0
    attention_mask = attention_mask.to(total_input_ids.device)

    for i, ids in enumerate(total_input_ids):
        ids = ids[attention_mask[i] == 1]
        image_nums, video_nums = 0, 0

        vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
        if len(vision_start_indices) > 0:
            vision_tokens = ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

        input_tokens = ids.tolist()
        llm_pos_ids_list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums

        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid_thw_list[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw_list[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            text_len = ed - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
            position_ids.device
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


class Qwen35MoeVLInferenceConfig:
    """Configuration for the full VL model (text + vision).

    Wraps the existing Qwen35MoeInferenceConfig for text and adds
    vision-specific settings.
    """

    def __init__(
        self,
        text_config,
        vision_config,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        spatial_merge_size=2,
        vision_seq_len_buckets=None,
        **kwargs,
    ):
        """
        Args:
            text_config: Qwen35MoeInferenceConfig instance for the text decoder
            vision_config: dict with vision encoder hyperparams (depth, hidden_size, etc.)
            image_token_id: Token ID for image placeholder tokens
            video_token_id: Token ID for video placeholder tokens
            vision_start_token_id: Token ID for <|vision_start|>
            vision_end_token_id: Token ID for <|vision_end|>
            spatial_merge_size: How many patches are merged (2 = 2x2 = 4 patches merged)
            vision_seq_len_buckets: List of vision sequence length buckets for compilation
        """
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.spatial_merge_size = spatial_merge_size
        self.vision_seq_len_buckets = vision_seq_len_buckets or [1024, 4096, 16384]


class NeuronQwen35MoeVLForCausalLM:
    """Top-level VL model for Qwen3.5-35B-A3B on Neuron.

    This class manages:
    - Separate compilation/loading of vision encoder and text decoder
    - CPU-side mRoPE computation
    - Vision embedding injection into text decoder
    - The CTE+TKG generation loop

    Note: This is NOT an NeuronBaseForImageToText subclass because the
    text decoder (NeuronQwen35MoeForCausalLM) has extensive custom overrides
    (DeltaNet state management, custom forward, custom ModelWrapper) that
    don't fit the base class pattern. Instead, this class composes the two
    models and handles the VL orchestration directly.
    """

    def __init__(self, model_path, text_config, vision_config=None, processor=None):
        """
        Args:
            model_path: Path to HF model directory
            text_config: Qwen35MoeInferenceConfig for text decoder
            vision_config: Qwen35MoeVLInferenceConfig (or None for text-only)
            processor: HF AutoProcessor for image preprocessing
        """
        self.model_path = model_path
        self.text_config = text_config
        self.vl_config = vision_config
        self.processor = processor

        # Text decoder (existing implementation)
        self.text_model = NeuronQwen35MoeForCausalLM(
            model_path=model_path, config=text_config
        )

        # Vision encoder (lazy init -- only built if vl_config provided)
        self.vision_model_wrapper = None
        if vision_config is not None:
            self._init_vision_model(vision_config)

        # mRoPE state
        self.rope_deltas = None

    def _init_vision_model(self, vl_config):
        """Initialize the vision encoder wrapper."""
        from types import SimpleNamespace

        vision_cfg = SimpleNamespace(**vl_config.vision_config)
        self.vision_model_wrapper = NeuronQwen35VisionModelWrapper(
            config=vision_cfg,
            model_cls=None,  # Standalone mode (no NxDI parallel layers)
            vision_seq_len_buckets=vl_config.vision_seq_len_buckets,
        )
        self._vl_config = vl_config

    def compile(self, compiled_model_path):
        """Compile both text and vision models.

        For the vision encoder, use compile_vision_encoder.py separately
        (standalone torch_neuronx.trace compilation). Then use load() to
        load the pre-compiled vision encoder.
        """
        # Compile text decoder
        text_path = os.path.join(compiled_model_path, "text_model")
        os.makedirs(text_path, exist_ok=True)
        self.text_model.compile(text_path)

        # Vision encoder is compiled separately via compile_vision_encoder.py
        if self.vision_model_wrapper is not None:
            logger.info(
                "Vision encoder must be compiled separately using "
                "compile_vision_encoder.py. Use load() to load the "
                "pre-compiled vision encoder."
            )

    def load(self, compiled_model_path, vision_compiled_path=None):
        """Load both compiled models.

        Args:
            compiled_model_path: Path to compiled text model (or parent dir)
            vision_compiled_path: Path to compiled vision encoder .pt file.
                If None, looks for 'vision_encoder.pt' in compiled_model_path.
        """
        text_path = os.path.join(compiled_model_path, "text_model")
        if os.path.exists(text_path):
            self.text_model.load(text_path)
        else:
            # Backward compatibility: text model compiled at root
            self.text_model.load(compiled_model_path)

        # Load vision encoder
        if self.vision_model_wrapper is not None:
            if vision_compiled_path is None:
                vision_compiled_path = os.path.join(
                    compiled_model_path, "vision_encoder.pt"
                )
            if os.path.exists(vision_compiled_path):
                self.vision_model_wrapper.load_compiled(vision_compiled_path)
                # Also load CPU-side weights (patch_embed, pos_embed)
                self.vision_model_wrapper.load_vision_weights_from_hf(self.model_path)
                logger.info("Vision encoder loaded from pre-compiled model")
            else:
                logger.warning(
                    f"No compiled vision encoder found at {vision_compiled_path}. "
                    "Vision encoding will not be available."
                )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        video_grid_thw=None,
        max_new_tokens=32,
        **kwargs,
    ):
        """Generate text from text and/or vision inputs.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            pixel_values: Vision pixel values from HF processor (or None for text-only)
            image_grid_thw: (num_images, 3) grid dimensions
            video_grid_thw: (num_videos, 3) grid dimensions
            max_new_tokens: Maximum new tokens to generate

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens) token IDs
        """
        has_vision = pixel_values is not None and pixel_values.numel() > 0

        # Step 1: Compute 3D mRoPE position IDs
        if has_vision and self._vl_config is not None:
            position_ids, self.rope_deltas = get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                image_token_id=self._vl_config.image_token_id,
                video_token_id=self._vl_config.video_token_id,
                vision_start_token_id=self._vl_config.vision_start_token_id,
                spatial_merge_size=self._vl_config.spatial_merge_size,
            )
        else:
            # Text-only: use standard sequential position IDs
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0)
            self.rope_deltas = None

        # Step 2: Run vision encoder and prepare injection args
        llava_args = []
        if has_vision and self.vision_model_wrapper is not None:
            vision_embeddings = self.vision_model_wrapper(pixel_values, image_grid_thw)
            # vision_embeddings: (total_merged_tokens, out_hidden_size)

            # Build vision_mask: boolean mask of image token positions
            image_token_id = self._vl_config.image_token_id
            vision_bool_mask = input_ids == image_token_id  # (BS, seq_len)

            # Convert bool mask to position indices (1D tensor of int positions)
            positions = (
                vision_bool_mask[0].nonzero(as_tuple=False).squeeze(-1)
            )  # (n_vision_tokens,)

            # Reshape vision_embeddings to (1, n_vision_tokens, hidden_size)
            n_vis = positions.shape[0]
            hidden_size = vision_embeddings.shape[-1]
            vis_emb = vision_embeddings[:n_vis].unsqueeze(0)  # (1, n_vis, hidden)

            # Pad to match input sequence length for compiled graph compatibility
            seq_len = input_ids.shape[1]
            pad_limit = seq_len  # Must match the bucket size

            # Pad vision_embeddings to (1, pad_limit, hidden_size)
            if n_vis < pad_limit:
                pad_emb = torch.zeros(
                    (1, pad_limit - n_vis, hidden_size),
                    dtype=vis_emb.dtype,
                )
                vis_emb_padded = torch.cat([vis_emb, pad_emb], dim=1)
            else:
                vis_emb_padded = vis_emb[:, :pad_limit]

            # Pad positions to (1, pad_limit, 1) with a SAFE fill value.
            # CRITICAL: fill_value must NOT be a real token position, otherwise
            # index_put_ will scatter zero embeddings over a real token's embedding.
            # Use a large sentinel (2**30) that pad_inputs() will clamp to the
            # last padded (bucket) position, which is always a padding slot.
            _SAFE_SCATTER_SENTINEL = 2**30
            positions_padded = torch.full(
                (1, pad_limit, 1),
                fill_value=_SAFE_SCATTER_SENTINEL,
                dtype=torch.int32,
            )
            positions_padded[0, :n_vis, 0] = positions[:pad_limit].to(torch.int32)

            llava_args = [vis_emb_padded, positions_padded]

            # Append 3D mRoPE position IDs for the text model.
            # position_ids shape: (3, batch_size, seq_len) from get_rope_index.
            # _get_model_outputs receives this at slot 21 and pre-computes
            # mRoPE cos/sin in get_model_output() for all decoder layers.
            if position_ids.ndim == 3:
                mrope_pos = position_ids[:, :, :seq_len].to(torch.int32).contiguous()
                llava_args.append(mrope_pos)
        else:
            vision_embeddings = None

        # Step 3: Context encoding (prefill)
        generated_ids = input_ids.clone()

        # CRITICAL: Always pass an explicit attention_mask for CTE.
        # The base class _infer_attention_mask() assumes sequential position_ids
        # (position_ids[i] >= i). When position_ids come from mRoPE temporal
        # axis (non-sequential, e.g., all vision tokens share position 4),
        # the inferred mask incorrectly masks out most of the sequence.
        # Fix: provide a real all-ones mask for the actual token positions.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # For slot 2 (position_ids): use SEQUENTIAL positions regardless of mRoPE.
        # Slot 2 is only used for: (1) logit position selection via torch.max(),
        # (2) attention mask inference (which we bypass with explicit mask above).
        # The actual RoPE computation uses slot 21 (rotary_position_ids) from
        # _get_model_outputs, NOT slot 2. Using sequential slot 2 ensures
        # correct logit selection and avoids any position_ids-related issues.
        seq_len = input_ids.shape[1]
        cte_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            output = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=cte_position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                llava_args=llava_args,
            )

        logits = output[0] if isinstance(output, tuple) else output.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

        # Step 4: Token generation (TKG) loop
        for _ in range(max_new_tokens - 1):
            pos_ids = torch.tensor([[generated_ids.shape[1] - 1]])
            if self.rope_deltas is not None:
                pos_ids = pos_ids + self.rope_deltas

            last_token = generated_ids[:, -1:]
            with torch.no_grad():
                output = self.text_model(
                    input_ids=last_token,
                    position_ids=pos_ids,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                )
            logits = output[0] if isinstance(output, tuple) else output.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

            # Stop on EOS
            if next_token.item() in (151643, 151645):  # Qwen EOS tokens
                break

        return generated_ids

    @staticmethod
    def prepare_input_args(text_prompt, image_path, processor, role="user"):
        """Prepare inputs for vision+text generation.

        Args:
            text_prompt: Text prompt string
            image_path: Path to image file (or None for text-only)
            processor: HF AutoProcessor
            role: Message role (default "user")

        Returns:
            input_ids, attention_mask, vision_inputs dict
        """
        content = []
        if image_path is not None:
            import base64
            from pathlib import Path

            image_data = Path(image_path).read_bytes()
            b64 = base64.b64encode(image_data).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "url": f"data:image/jpeg;base64,{b64}",
                }
            )
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": role, "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        vision_inputs = {}
        if "pixel_values" in inputs:
            vision_inputs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            vision_inputs["image_grid_thw"] = inputs["image_grid_thw"]
        if "video_grid_thw" in inputs:
            vision_inputs["video_grid_thw"] = inputs["video_grid_thw"]

        return input_ids, attention_mask, vision_inputs
