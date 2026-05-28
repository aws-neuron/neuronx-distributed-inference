"""
Shared wrappers and utilities for LongCat-Image-Edit Neuron adaptation.

Provides:
- NeuronTextEncoderWrapper: Combines compiled vision encoder + language model
- NKI Flash Attention wrappers
- f32Wrapper for normalization stability
- Custom SDPA implementations for Neuron compatibility
"""

import torch
import math
from torch import nn
from typing import Optional, Tuple

# Try to import NKI kernel
try:
    import neuronxcc.nki as nki
    from neuronxcc.nki.language import nc
    try:
        from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    except ImportError:
        from neuronxcc.nki.kernels.attention import attention_isa_kernel
    _flash_fwd_call = nki.jit()(attention_isa_kernel)
    NKI_AVAILABLE = True
    print("NKI Flash Attention kernel loaded successfully")
except ImportError as e:
    _flash_fwd_call = None
    NKI_AVAILABLE = False
    nc = None
    print(f"NKI Flash Attention not available: {e}")


class f32Wrapper(nn.Module):
    """Wrapper to run normalization layers in float32 for numerical stability."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, *args, **kwargs):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y, *args, **kwargs)
        return output.type(t)


def upcast_norms_to_f32(module):
    """Upcast normalization layers to float32 for numerical stability."""
    for name, child in module.named_children():
        if isinstance(child, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        elif 'RMSNorm' in child.__class__.__name__:
            setattr(module, name, f32Wrapper(child.to(torch.float32)))
        else:
            upcast_norms_to_f32(child)


def nki_flash_attention(query, key, value):
    """
    NKI Flash Attention wrapper.

    Args:
        query: [B, H, S, D]
        key: [B, H, S, D]
        value: [B, H, S, D]

    Returns:
        attention output [B, H, S, D]
    """
    import os

    bs, n_head, q_len, d_head = query.shape
    k_len = key.shape[2]
    v_len = value.shape[2]

    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, k_len))
    v = value.clone().reshape((bs * n_head, v_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)
    scale = 1 / math.sqrt(d_head)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    return attn_output.reshape((bs, n_head, q_len, d_head))


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=None, is_causal=None, scale=None,
                                         enable_gqa=False, **kwargs):
    """Custom scaled dot product attention for Neuron (supports GQA and causal masking)."""
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
        attn_out = attn_out.reshape(orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2])
    return attn_out


def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None,
                      scale=None, enable_gqa=False):
    """Attention wrapper for text encoder -- always uses custom implementation."""
    return neuron_scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale)


def attention_wrapper_sharded_without_swap(query, key, value):
    """Sharded attention wrapper using NKI kernel for trn2."""
    import os

    bs, n_head, q_len, d_head = query.shape
    _, _, kv_len, _ = key.shape

    if q_len != kv_len or not NKI_AVAILABLE or _flash_fwd_call is None:
        return neuron_scaled_dot_product_attention(query, key, value)

    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, kv_len))
    v = value.clone().reshape((bs * n_head, kv_len, d_head))
    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    scale = 1.0 / math.sqrt(d_head)
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "2"))

    if vc_size == 2:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    return attn_output.reshape((bs, n_head, q_len, d_head))


class NeuronTextEncoderWrapper(nn.Module):
    """
    Wrapper for compiled Qwen2.5-VL text encoder on Neuron.

    Combines separately compiled vision encoder and language model.
    This wrapper handles the embedding combination logic that normally
    happens inside the original text encoder.

    Both LongCat-Image-Edit and Qwen-Image-Edit use the same Qwen2.5-VL
    text encoder, so this is largely reused from the reference.
    """
    def __init__(self, original_text_encoder,
                 compiled_vision_encoder=None,
                 compiled_language_model=None,
                 cpu_language_model=None,
                 cpu_vision_encoder=None,
                 image_size=448, max_seq_len=512,
                 language_model_batch_size=1):
        super().__init__()
        self.config = original_text_encoder.config
        self.dtype = torch.bfloat16
        self._device = torch.device('cpu')

        # Copy embed_tokens weights
        orig_embed = original_text_encoder.model.language_model.embed_tokens
        self.embed_tokens = nn.Embedding(
            orig_embed.num_embeddings,
            orig_embed.embedding_dim,
            padding_idx=orig_embed.padding_idx,
            dtype=torch.bfloat16,
        )
        self.embed_tokens.weight.data = orig_embed.weight.data.clone().to(torch.bfloat16)
        print(f"  Copied embed_tokens: {orig_embed.num_embeddings} x {orig_embed.embedding_dim}")

        # Use original model's get_rope_index for correct M-RoPE position IDs
        self._original_get_rope_index = original_text_encoder.model.get_rope_index

        # Copy visual_merger if needed (only for CPU vision encoder)
        if compiled_vision_encoder is None and hasattr(original_text_encoder.model.visual, 'merger'):
            import copy
            self.visual_merger = copy.deepcopy(original_text_encoder.model.visual.merger)
            self.visual_merger = self.visual_merger.to(torch.bfloat16)
        else:
            self.visual_merger = None

        # Compiled models
        self.compiled_vision_encoder = compiled_vision_encoder
        self.compiled_language_model = compiled_language_model
        self.cpu_language_model = cpu_language_model
        self.cpu_vision_encoder = cpu_vision_encoder

        self.use_cpu_vision_encoder = cpu_vision_encoder is not None
        self.use_compiled_vision_encoder = compiled_vision_encoder is not None
        self.use_cpu_language_model = cpu_language_model is not None
        self.use_compiled_language_model = compiled_language_model is not None
        self.language_model_batch_size = language_model_batch_size

        # Image processing parameters
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.patch_size = 14
        self.spatial_merge_size = 2
        num_patches_per_side = image_size // self.patch_size
        self.num_image_tokens = (num_patches_per_side // self.spatial_merge_size) ** 2

        # Special token IDs
        self.image_token_id = getattr(self.config, 'image_token_id', 151655)
        self.vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)

    @property
    def device(self):
        """Return device for pipeline compatibility."""
        return self._device

    def _get_rope_index(self, input_ids, image_grid_thw, attention_mask):
        """Calculate 3D position_ids for M-RoPE using original model's method."""
        position_ids, _ = self._original_get_rope_index(
            input_ids, image_grid_thw, None, attention_mask)
        return position_ids

        t = image_grid_thw[0, 0]
        h = image_grid_thw[0, 1]
        w = image_grid_thw[0, 2]
        llm_grid_h = h // self.spatial_merge_size
        llm_grid_w = w // self.spatial_merge_size
        grid_hw = llm_grid_h * llm_grid_w

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        is_image_token = (input_ids == self.image_token_id)
        has_images = is_image_token.any()

        if not has_images:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
            return position_ids

        position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=device)

        for b in range(batch_size):
            valid_mask = attention_mask[b] == 1
            batch_is_image = is_image_token[b] & valid_mask
            num_image_tokens = batch_is_image.sum()

            if num_image_tokens == 0:
                cumsum = valid_mask.long().cumsum(-1) - 1
                cumsum = cumsum * valid_mask.long()
                position_ids[:, b, :] = cumsum.unsqueeze(0).expand(3, -1)
                continue

            image_indices = torch.where(batch_is_image)[0]
            num_imgs = image_indices.shape[0]

            img_local_idx = torch.arange(num_imgs, device=device)
            t_pos = img_local_idx // grid_hw
            remainder = img_local_idx % grid_hw
            h_pos = remainder // llm_grid_w
            w_pos = remainder % llm_grid_w

            is_text = valid_mask & ~batch_is_image
            text_cumsum = is_text.long().cumsum(-1)

            first_image_idx = image_indices[0] if num_imgs > 0 else 0
            if first_image_idx > 0:
                text_offset = text_cumsum[first_image_idx - 1]
            else:
                text_offset = torch.zeros(1, dtype=torch.long, device=device)[0]

            position_ids[0, b, image_indices] = text_offset + t_pos
            position_ids[1, b, image_indices] = text_offset + h_pos
            position_ids[2, b, image_indices] = text_offset + w_pos

            max_img_pos = torch.max(torch.stack([t_pos, h_pos, w_pos]).max(dim=0)[0])
            after_image_offset = text_offset + max_img_pos + 1

            text_before_first_image = torch.arange(seq_len, device=device) < first_image_idx
            text_before_mask = is_text & text_before_first_image
            if text_before_mask.any():
                text_before_pos = text_before_mask.long().cumsum(-1) - 1
                text_before_pos = text_before_pos * text_before_mask.long()
                for d in range(3):
                    position_ids[d, b, :] = torch.where(
                        text_before_mask, text_before_pos, position_ids[d, b, :])

            last_image_idx = image_indices[-1] if num_imgs > 0 else 0
            text_after_last_image = torch.arange(seq_len, device=device) > last_image_idx
            text_after_mask = is_text & text_after_last_image
            if text_after_mask.any():
                text_after_local = text_after_mask.long().cumsum(-1)
                offset_at_last = text_after_local[last_image_idx] if last_image_idx < seq_len else 0
                text_after_pos = after_image_offset + (text_after_local - offset_at_last - 1)
                text_after_pos = text_after_pos * text_after_mask.long()
                for d in range(3):
                    position_ids[d, b, :] = torch.where(
                        text_after_mask, text_after_pos, position_ids[d, b, :])

        return position_ids

    def _merge_embeddings(self, text_embeds, image_embeds, input_ids, image_token_id):
        """Merge text and image embeddings at image token positions."""
        batch_size, seq_len, hidden_size = text_embeds.shape
        if image_embeds is None:
            return text_embeds

        image_mask = (input_ids == image_token_id)
        inputs_embeds = text_embeds.clone()

        if batch_size == 1:
            image_indices = image_mask[0].nonzero(as_tuple=True)[0]
            num_image_positions = image_indices.shape[0]
            if num_image_positions > 0:
                num_to_use = min(num_image_positions, image_embeds.shape[0])
                inputs_embeds[0, image_indices[:num_to_use]] = image_embeds[:num_to_use]
            return inputs_embeds

        for b in range(batch_size):
            image_indices = image_mask[b].nonzero(as_tuple=True)[0]
            num_image_positions = image_indices.shape[0]
            if num_image_positions > 0:
                num_to_use = min(num_image_positions, image_embeds.shape[0])
                inputs_embeds[b, image_indices[:num_to_use]] = image_embeds[:num_to_use]

        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, output_hidden_states=True, return_dict=True, **kwargs):
        """
        Forward pass combining vision encoder and language model.

        For Neuron inference:
        1. Vision encoder on compiled model (or CPU fallback)
        2. Combine image embeds with text embeds
        3. Pad to max_seq_len for compiled model
        4. Language model on compiled model
        5. Remove padding from output
        """
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        # Step 1: Process images through vision encoder
        if pixel_values is not None:
            if self.use_cpu_vision_encoder:
                with torch.no_grad():
                    image_embeds = self.cpu_vision_encoder(pixel_values, image_grid_thw)
            elif self.use_compiled_vision_encoder:
                expected_patches = (self.image_size // self.patch_size) ** 2
                actual_patches = pixel_values.shape[0]
                num_images = image_grid_thw.shape[0]

                pixel_values = pixel_values.to(torch.float32)

                if num_images > 1:
                    all_embeds = []
                    patch_idx = 0
                    for img_idx in range(num_images):
                        t = image_grid_thw[img_idx, 0]
                        h = image_grid_thw[img_idx, 1]
                        w = image_grid_thw[img_idx, 2]
                        img_patches = (t * h * w).item()

                        img_pv = pixel_values[patch_idx:patch_idx + img_patches]
                        patch_idx += img_patches

                        if img_patches < expected_patches:
                            padding = torch.zeros(
                                expected_patches - img_patches, img_pv.shape[1],
                                dtype=img_pv.dtype, device=img_pv.device)
                            img_pv = torch.cat([img_pv, padding], dim=0)
                        elif img_patches > expected_patches:
                            img_pv = img_pv[:expected_patches]

                        grid_size = self.image_size // self.patch_size
                        single_grid = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                        img_embeds = self.compiled_vision_encoder(
                            pixel_values=img_pv, grid_thw=single_grid)

                        merged_h = h // self.spatial_merge_size
                        merged_w = w // self.spatial_merge_size
                        actual_output = (t * merged_h * merged_w).item()
                        img_embeds = img_embeds[:actual_output]
                        all_embeds.append(img_embeds)

                    image_embeds = torch.cat(all_embeds, dim=0)
                else:
                    if actual_patches != expected_patches:
                        if actual_patches < expected_patches:
                            padding = torch.zeros(
                                expected_patches - actual_patches, pixel_values.shape[1],
                                dtype=pixel_values.dtype, device=pixel_values.device)
                            pixel_values = torch.cat([pixel_values, padding], dim=0)
                        else:
                            pixel_values = pixel_values[:expected_patches]
                        grid_size = self.image_size // self.patch_size
                        image_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                    image_embeds = self.compiled_vision_encoder(
                        pixel_values=pixel_values, grid_thw=image_grid_thw)

                image_embeds = image_embeds.to(torch.bfloat16)
            else:
                raise RuntimeError("No vision encoder available!")
        else:
            image_embeds = None

        # Step 2: Get text embeddings
        text_embeds = self.embed_tokens(input_ids)

        # Step 3: Combine embeddings
        if image_embeds is not None:
            inputs_embeds = self._merge_embeddings(
                text_embeds, image_embeds, input_ids, self.image_token_id)
        else:
            inputs_embeds = text_embeds

        # Step 4: Calculate M-RoPE position IDs
        position_ids = self._get_rope_index(input_ids, image_grid_thw, attention_mask)

        # Step 5: Run language model
        if self.use_cpu_language_model:
            with torch.no_grad():
                cpu_outputs = self.cpu_language_model(
                    inputs_embeds=inputs_embeds.to(torch.bfloat16),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = cpu_outputs.last_hidden_state

            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states,
                })()
            return hidden_states

        elif self.use_compiled_language_model:
            original_seq_len = inputs_embeds.shape[1]
            hidden_size = inputs_embeds.shape[2]

            # Pad to compiled sequence length
            if original_seq_len < self.max_seq_len:
                pad_len = self.max_seq_len - original_seq_len
                embed_padding = torch.zeros(
                    batch_size, pad_len, hidden_size,
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = torch.cat([inputs_embeds, embed_padding], dim=1)

                if attention_mask is not None:
                    mask_padding = torch.zeros(
                        batch_size, pad_len,
                        dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

                if position_ids is not None:
                    last_pos = position_ids[:, :, -1:] + 1
                    pad_positions = last_pos + torch.arange(pad_len, device=position_ids.device).view(1, 1, -1)
                    position_ids = torch.cat([position_ids, pad_positions], dim=2)
            elif original_seq_len > self.max_seq_len:
                print(f"  WARNING: Sequence {original_seq_len} > max {self.max_seq_len}, truncating")
                inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_seq_len]
                if position_ids is not None:
                    position_ids = position_ids[:, :, :self.max_seq_len]
                original_seq_len = self.max_seq_len

            # Batch padding
            actual_batch_size = inputs_embeds.shape[0]
            if actual_batch_size < self.language_model_batch_size:
                pad_batch = self.language_model_batch_size - actual_batch_size
                inputs_embeds = torch.cat([
                    inputs_embeds,
                    torch.zeros((pad_batch, inputs_embeds.shape[1], inputs_embeds.shape[2]),
                               dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                ], dim=0)
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros((pad_batch, attention_mask.shape[1]),
                                   dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=0)
                if position_ids is not None:
                    position_ids = torch.cat([
                        position_ids,
                        position_ids[:, :1, :].repeat(1, pad_batch, 1)
                    ], dim=1)

            hidden_states = self.compiled_language_model(
                inputs_embeds.to(torch.bfloat16), attention_mask, position_ids)

            if actual_batch_size < self.language_model_batch_size:
                hidden_states = hidden_states[:actual_batch_size]
            hidden_states = hidden_states[:, :original_seq_len, :]

            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states,
                })()
            return hidden_states

        else:
            raise RuntimeError("No language model available!")
