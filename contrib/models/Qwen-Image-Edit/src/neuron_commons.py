import torch
import math
from torch import nn
from diffusers import QwenImageEditPlusPipeline
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Try to import NKI kernel, but don't fail if not available
try:
    import neuronxcc.nki as nki
    from neuronxcc.nki.language import nc
    try:
        from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
    except ImportError:
        from neuronxcc.nki.kernels.attention import attention_isa_kernel
    _flash_fwd_call = nki.jit()(attention_isa_kernel)
    NKI_AVAILABLE = True
    print(f"NKI Flash Attention kernel loaded successfully")
except ImportError as e:
    _flash_fwd_call = None
    NKI_AVAILABLE = False
    nc = None
    print(f"NKI Flash Attention not available: {e}")


class InferenceTextEncoderWrapper(nn.Module):
    """Wrapper for Qwen2.5-VL text encoder for inference on Neuron."""
    def __init__(self, dtype, text_encoder: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.dtype = dtype
        self.device = text_encoder.device
        self.text_encoder = text_encoder
        self.config = text_encoder.config

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                image_grid_thw=None, **kwargs):
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **kwargs
        )
        return outputs


class NeuronTextEncoderWrapper(nn.Module):
    """
    Wrapper for compiled Qwen2.5-VL text encoder on Neuron.

    Combines separately compiled vision encoder and language model.
    This wrapper handles the embedding combination logic that normally
    happens inside the original text encoder.

    Supports three modes for Language Model:
    1. compiled_language_model: Neuron-compiled model with parallel_model_trace (TP=8)
    2. compiled_language_model_v3: Neuron-compiled model with ModelBuilder API (TP=4, world_size=8)
    3. cpu_language_model: Original model on CPU (slower but avoids GQA issues)

    IMPORTANT: This wrapper COPIES necessary components and does NOT keep
    references to the original model, to avoid memory bloat.
    """
    def __init__(self, original_text_encoder, compiled_vision_encoder=None,
                 compiled_vision_encoder_v3=None,  # V3 vision encoder (TP=4, NxDModel)
                 compiled_language_model=None, compiled_language_model_v3=None,
                 cpu_language_model=None,
                 cpu_vision_encoder=None,  # Option to use CPU vision encoder
                 image_size=448, max_seq_len=512,
                 language_model_batch_size=1):  # Batch size for V3 language model
        super().__init__()
        # Copy config (small object)
        self.config = original_text_encoder.config
        self.dtype = torch.bfloat16

        # IMPORTANT: Copy embed_tokens weights instead of keeping reference!
        # This allows the original model to be garbage collected.
        orig_embed = original_text_encoder.model.language_model.embed_tokens
        self.embed_tokens = nn.Embedding(
            orig_embed.num_embeddings,
            orig_embed.embedding_dim,
            padding_idx=orig_embed.padding_idx,
            dtype=torch.bfloat16
        )
        self.embed_tokens.weight.data = orig_embed.weight.data.clone().to(torch.bfloat16)
        print(f"  Copied embed_tokens: {orig_embed.num_embeddings} x {orig_embed.embedding_dim} "
              f"= {orig_embed.weight.numel() * 2 / 1e9:.2f} GB")

        # Copy visual_merger if it exists (small module)
        # Note: For V3 vision encoder, merger is included in the compiled model
        if compiled_vision_encoder_v3 is None and hasattr(original_text_encoder.model.visual, 'merger'):
            # Deep copy the merger module (only needed for non-V3 or CPU vision encoder)
            import copy
            self.visual_merger = copy.deepcopy(original_text_encoder.model.visual.merger)
            self.visual_merger = self.visual_merger.to(torch.bfloat16)
        else:
            self.visual_merger = None

        # Compiled models
        self.compiled_vision_encoder = compiled_vision_encoder
        self.compiled_vision_encoder_v3 = compiled_vision_encoder_v3  # V3 (NxDModel, TP=4)
        self.compiled_language_model = compiled_language_model
        self.compiled_language_model_v3 = compiled_language_model_v3

        # CPU Vision Encoder (for better accuracy, avoids compilation precision loss)
        self.cpu_vision_encoder = cpu_vision_encoder
        self.use_cpu_vision_encoder = cpu_vision_encoder is not None

        # V3 Vision Encoder (ModelBuilder API, TP=4, world_size=8, float32)
        self.use_v3_vision_encoder = compiled_vision_encoder_v3 is not None

        # CPU Language Model (alternative to compiled, avoids GQA alignment issues)
        self.cpu_language_model = cpu_language_model
        self.use_cpu_language_model = cpu_language_model is not None

        # V3 Language Model (ModelBuilder API, TP=4, world_size=8)
        self.use_v3_language_model = compiled_language_model_v3 is not None
        self.language_model_batch_size = language_model_batch_size  # Compiled batch size

        # DO NOT keep original_text_encoder - it's 16+ GB!
        # self.original_text_encoder = original_text_encoder  # REMOVED!

        # Image processing parameters
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.patch_size = 14
        self.spatial_merge_size = 2

        # Calculate expected dimensions
        num_patches_per_side = image_size // self.patch_size
        self.num_image_tokens = (num_patches_per_side // self.spatial_merge_size) ** 2

        # Special token IDs from config
        self.image_token_id = getattr(self.config, 'image_token_id', 151655)
        self.vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)

    def _get_rope_index(self, input_ids, image_grid_thw, attention_mask):
        """
        Calculate 3D position_ids for M-RoPE (Multimodal RoPE).

        For multimodal input (text + images), position_ids have different patterns:
        - Text tokens: sequential positions (same for t, h, w dimensions)
        - Image tokens: 3D grid positions based on spatial layout

        This replicates the logic from Qwen2_5_VLModel.get_rope_index().

        OPTIMIZED: Uses vectorized tensor operations to avoid CPU synchronization.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # If no images, use simple text-only position_ids
        if image_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
            return position_ids

        # Multimodal case: vectorized computation of 3D positions
        # Get grid dimensions (avoid .tolist() by using tensor indexing)
        t = image_grid_thw[0, 0]
        h = image_grid_thw[0, 1]
        w = image_grid_thw[0, 2]
        llm_grid_h = h // self.spatial_merge_size
        llm_grid_w = w // self.spatial_merge_size
        grid_hw = llm_grid_h * llm_grid_w

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Create image token mask for all batches at once
        is_image_token = (input_ids == self.image_token_id)  # [batch, seq]

        # Check if any batch has image tokens (avoid .item() by checking tensor)
        has_images = is_image_token.any()

        if not has_images:
            # No images in any batch, use simple sequential positions
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)
            return position_ids

        # Initialize position_ids
        position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=device)

        # Process each batch (still need loop for batch, but inner ops are vectorized)
        for b in range(batch_size):
            valid_mask = attention_mask[b] == 1
            valid_len = valid_mask.sum()

            # Get image token mask for valid positions
            batch_is_image = is_image_token[b] & valid_mask
            num_image_tokens = batch_is_image.sum()

            if num_image_tokens == 0:
                # No images, use sequential positions
                pos = torch.arange(seq_len, device=device)
                masked_pos = pos * valid_mask.long()
                # Compute cumsum for valid positions only
                cumsum = valid_mask.long().cumsum(-1) - 1
                cumsum = cumsum * valid_mask.long()
                position_ids[:, b, :] = cumsum.unsqueeze(0).expand(3, -1)
                continue

            # Vectorized computation for multimodal case
            # Create index arrays for image tokens
            image_indices = torch.where(batch_is_image)[0]  # positions of image tokens
            num_imgs = image_indices.shape[0]

            # Compute grid positions for all image tokens at once
            img_local_idx = torch.arange(num_imgs, device=device)
            t_pos = img_local_idx // grid_hw
            remainder = img_local_idx % grid_hw
            h_pos = remainder // llm_grid_w
            w_pos = remainder % llm_grid_w

            # Compute text offset: count non-image tokens before each position
            # First, get cumulative count of non-image tokens
            is_text = valid_mask & ~batch_is_image
            text_cumsum = is_text.long().cumsum(-1)

            # For image tokens, the offset is the text count before the first image token
            first_image_idx = image_indices[0] if num_imgs > 0 else 0
            text_offset = text_cumsum[first_image_idx] - (1 if is_text[first_image_idx] else 0)
            if first_image_idx > 0:
                text_offset = text_cumsum[first_image_idx - 1]
            else:
                text_offset = torch.zeros(1, dtype=torch.long, device=device)[0]

            # Set image token positions
            position_ids[0, b, image_indices] = text_offset + t_pos
            position_ids[1, b, image_indices] = text_offset + h_pos
            position_ids[2, b, image_indices] = text_offset + w_pos

            # Compute max position used by images
            max_img_pos = torch.max(torch.stack([t_pos, h_pos, w_pos]).max(dim=0)[0])
            after_image_offset = text_offset + max_img_pos + 1

            # Set text token positions
            # Text before images: sequential from 0
            text_before_first_image = torch.arange(seq_len, device=device) < first_image_idx
            text_before_mask = is_text & text_before_first_image
            if text_before_mask.any():
                text_before_pos = text_before_mask.long().cumsum(-1) - 1
                text_before_pos = text_before_pos * text_before_mask.long()
                for d in range(3):
                    position_ids[d, b, :] = torch.where(
                        text_before_mask,
                        text_before_pos,
                        position_ids[d, b, :]
                    )

            # Text after images: sequential from after_image_offset
            last_image_idx = image_indices[-1] if num_imgs > 0 else 0
            text_after_last_image = torch.arange(seq_len, device=device) > last_image_idx
            text_after_mask = is_text & text_after_last_image
            if text_after_mask.any():
                # Count text tokens after last image
                text_after_local = text_after_mask.long().cumsum(-1)
                # Subtract count at last_image_idx to get local index
                offset_at_last = text_after_local[last_image_idx] if last_image_idx < seq_len else 0
                text_after_pos = after_image_offset + (text_after_local - offset_at_last - 1)
                text_after_pos = text_after_pos * text_after_mask.long()
                for d in range(3):
                    position_ids[d, b, :] = torch.where(
                        text_after_mask,
                        text_after_pos,
                        position_ids[d, b, :]
                    )

        return position_ids

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                image_grid_thw=None, output_hidden_states=True, return_dict=True, **kwargs):
        """
        Forward pass combining vision encoder and language model.

        For Neuron inference, we run:
        1. Vision encoder on compiled model (or CPU fallback)
        2. Combine image embeds with text embeds
        3. Pad to max_seq_len for compiled model
        4. Language model on compiled model
        5. Remove padding from output
        """
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        # Step 1: Process images through vision encoder
        if pixel_values is not None:
            # Determine dtype for vision encoder
            # - CPU vision encoder: use original dtype (usually float32 from pipeline)
            # - Compiled vision encoder: always float32 (required for accuracy)
            if self.use_cpu_vision_encoder:
                # Keep original dtype for CPU (highest precision)
                pass
            else:
                # Use float32 for compiled vision encoder (required for accuracy)
                pixel_values = pixel_values.to(torch.float32)

            # Option 1: Use CPU Vision Encoder (highest accuracy)
            if self.use_cpu_vision_encoder:
                with torch.no_grad():
                    image_embeds = self.cpu_vision_encoder(pixel_values, image_grid_thw)

            # Option 2: Use V3 Vision Encoder (TP=4, NxDModel, float32, fast)
            elif self.use_v3_vision_encoder:
                # V3 vision encoder expects fixed patch count for single image
                expected_patches_per_image = (self.image_size // self.patch_size) ** 2  # 1024 for 448x448
                actual_patches = pixel_values.shape[0]
                num_images = image_grid_thw.shape[0]

                # For multi-image input, process each image separately
                if num_images > 1:
                    all_embeds = []
                    patch_idx = 0
                    for img_idx in range(num_images):
                        # Use tensor indexing to avoid .tolist() CPU sync
                        t = image_grid_thw[img_idx, 0]
                        h = image_grid_thw[img_idx, 1]
                        w = image_grid_thw[img_idx, 2]
                        img_patches = (t * h * w).item()  # Need scalar for slicing

                        img_pixel_values = pixel_values[patch_idx:patch_idx + img_patches]
                        patch_idx += img_patches

                        # Pad or truncate to expected size
                        if img_patches < expected_patches_per_image:
                            padding = torch.zeros(
                                expected_patches_per_image - img_patches,
                                img_pixel_values.shape[1],
                                dtype=img_pixel_values.dtype,
                                device=img_pixel_values.device
                            )
                            img_pixel_values = torch.cat([img_pixel_values, padding], dim=0)
                        elif img_patches > expected_patches_per_image:
                            img_pixel_values = img_pixel_values[:expected_patches_per_image]

                        # Create grid_thw for single image
                        grid_size = self.image_size // self.patch_size
                        single_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                        # Run V3 vision encoder (NxDModel)
                        img_embeds = self.compiled_vision_encoder_v3(
                            pixel_values=img_pixel_values,
                            grid_thw=single_grid_thw
                        )

                        # Calculate actual output tokens (after spatial merge)
                        merged_h = h // self.spatial_merge_size
                        merged_w = w // self.spatial_merge_size
                        actual_output_tokens = (t * merged_h * merged_w).item()

                        # Truncate to actual output size (remove padding)
                        img_embeds = img_embeds[:actual_output_tokens]
                        all_embeds.append(img_embeds)

                    image_embeds = torch.cat(all_embeds, dim=0)
                else:
                    # Single image processing
                    if actual_patches != expected_patches_per_image:
                        if actual_patches < expected_patches_per_image:
                            padding = torch.zeros(
                                expected_patches_per_image - actual_patches,
                                pixel_values.shape[1],
                                dtype=pixel_values.dtype,
                                device=pixel_values.device
                            )
                            pixel_values = torch.cat([pixel_values, padding], dim=0)
                        else:
                            pixel_values = pixel_values[:expected_patches_per_image]

                        grid_size = self.image_size // self.patch_size
                        image_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                    image_embeds = self.compiled_vision_encoder_v3(
                        pixel_values=pixel_values,
                        grid_thw=image_grid_thw
                    )

                # Convert output to bfloat16 for downstream processing
                image_embeds = image_embeds.to(torch.bfloat16)

            # Option 3: Use single-device compiled Vision Encoder (slower)
            elif self.compiled_vision_encoder is not None:
                # Compiled vision encoder expects fixed patch count for single image
                expected_patches_per_image = (self.image_size // self.patch_size) ** 2  # 1024 for 448x448
                actual_patches = pixel_values.shape[0]
                num_images = image_grid_thw.shape[0]

                # For multi-image input, process each image separately
                if num_images > 1:
                    # Process each image through compiled vision encoder
                    all_embeds = []
                    patch_idx = 0
                    for img_idx in range(num_images):
                        # Use tensor indexing to avoid .tolist() CPU sync
                        t = image_grid_thw[img_idx, 0]
                        h = image_grid_thw[img_idx, 1]
                        w = image_grid_thw[img_idx, 2]
                        img_patches = (t * h * w).item()  # Need scalar for slicing

                        # Extract patches for this image
                        img_pixel_values = pixel_values[patch_idx:patch_idx + img_patches]
                        patch_idx += img_patches

                        # Pad or truncate to expected size
                        if img_patches < expected_patches_per_image:
                            padding = torch.zeros(
                                expected_patches_per_image - img_patches,
                                img_pixel_values.shape[1],
                                dtype=img_pixel_values.dtype,
                                device=img_pixel_values.device
                            )
                            img_pixel_values = torch.cat([img_pixel_values, padding], dim=0)
                        elif img_patches > expected_patches_per_image:
                            img_pixel_values = img_pixel_values[:expected_patches_per_image]

                        # Create grid_thw for single image
                        grid_size = self.image_size // self.patch_size
                        single_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                        # Run vision encoder for this image
                        img_embeds = self.compiled_vision_encoder(img_pixel_values, single_grid_thw)

                        # Calculate actual output tokens (after spatial merge)
                        merged_h = h // self.spatial_merge_size
                        merged_w = w // self.spatial_merge_size
                        actual_output_tokens = (t * merged_h * merged_w).item()

                        # Truncate to actual output size (remove padding)
                        img_embeds = img_embeds[:actual_output_tokens]
                        all_embeds.append(img_embeds)

                    # Concatenate all image embeddings
                    image_embeds = torch.cat(all_embeds, dim=0)
                else:
                    # Single image processing
                    if actual_patches != expected_patches_per_image:
                        if actual_patches < expected_patches_per_image:
                            padding = torch.zeros(
                                expected_patches_per_image - actual_patches,
                                pixel_values.shape[1],
                                dtype=pixel_values.dtype,
                                device=pixel_values.device
                            )
                            pixel_values = torch.cat([pixel_values, padding], dim=0)
                        else:
                            pixel_values = pixel_values[:expected_patches_per_image]

                        grid_size = self.image_size // self.patch_size
                        image_grid_thw = torch.tensor([[1, grid_size, grid_size]], dtype=torch.int64)

                    image_embeds = self.compiled_vision_encoder(pixel_values, image_grid_thw)

                # Convert output to bfloat16 for downstream processing
                image_embeds = image_embeds.to(torch.bfloat16)
                # Note: merger is already included in compiled_vision_encoder
            else:
                # No vision encoder available
                raise RuntimeError(
                    "No vision encoder available! Please either:\n"
                    "  1. Compile: python neuron_qwen_image_edit/compile_text_encoder.py --vision_only\n"
                    "  2. Use --cpu_vision_encoder flag"
                )
        else:
            image_embeds = None

        # Step 2: Get text embeddings
        text_embeds = self.embed_tokens(input_ids)

        # Step 3: Combine embeddings
        # Find image token positions and replace with image embeddings
        if image_embeds is not None:
            # The image token ID in Qwen2.5-VL
            image_token_id = self.config.image_token_id if hasattr(self.config, 'image_token_id') else 151655

            # Create combined embeddings
            inputs_embeds = self._merge_embeddings(
                text_embeds, image_embeds, input_ids, image_token_id
            )
        else:
            inputs_embeds = text_embeds

        # Step 4: Calculate 3D position_ids for M-RoPE (required by Qwen2.5-VL)
        # For multimodal input (text + images), position_ids have special patterns:
        # - Text tokens: sequential positions (same for t, h, w dimensions)
        # - Image tokens: 3D grid positions based on spatial layout
        position_ids = self._get_rope_index(input_ids, image_grid_thw, attention_mask)

        # Step 5: Run language model (CPU, V3, or compiled)
        if self.use_cpu_language_model:
            # CPU Language Model mode - no padding needed, handles dynamic sequence lengths
            # This avoids GQA alignment issues that occur with TP != 4
            with torch.no_grad():
                cpu_outputs = self.cpu_language_model(
                    inputs_embeds=inputs_embeds.to(torch.bfloat16),
                    attention_mask=attention_mask,
                    position_ids=position_ids,  # Pass 3D position_ids for M-RoPE
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = cpu_outputs.last_hidden_state

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states

        elif self.use_v3_language_model:
            # V3 Language Model mode (ModelBuilder API, TP=4, world_size=8)
            # Compatible with V3 CP transformer
            original_seq_len = inputs_embeds.shape[1]
            hidden_size = inputs_embeds.shape[2]

            if original_seq_len < self.max_seq_len:
                # Pad inputs_embeds with zeros
                pad_len = self.max_seq_len - original_seq_len
                embed_padding = torch.zeros(
                    batch_size, pad_len, hidden_size,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                inputs_embeds = torch.cat([inputs_embeds, embed_padding], dim=1)

                # Pad attention_mask with zeros (masked positions)
                if attention_mask is not None:
                    mask_padding = torch.zeros(
                        batch_size, pad_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

                # Pad position_ids with sequential positions
                if position_ids is not None:
                    # position_ids shape: (3, batch, seq_len)
                    last_pos = position_ids[:, :, -1:] + 1
                    pad_positions = last_pos + torch.arange(pad_len, device=position_ids.device).view(1, 1, -1)
                    position_ids = torch.cat([position_ids, pad_positions], dim=2)
            elif original_seq_len > self.max_seq_len:
                # Truncate if too long
                print(f"  WARNING: Sequence length {original_seq_len} > max_seq_len {self.max_seq_len}, truncating")
                inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_seq_len]
                if position_ids is not None:
                    position_ids = position_ids[:, :, :self.max_seq_len]
                original_seq_len = self.max_seq_len

            # Handle batch padding if needed
            actual_batch_size = inputs_embeds.shape[0]
            if actual_batch_size < self.language_model_batch_size:
                pad_batch = self.language_model_batch_size - actual_batch_size
                # Pad inputs_embeds
                inputs_embeds = torch.cat([
                    inputs_embeds,
                    torch.zeros((pad_batch, inputs_embeds.shape[1], inputs_embeds.shape[2]),
                               dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                ], dim=0)
                # Pad attention_mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros((pad_batch, attention_mask.shape[1]),
                                   dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=0)
                # Pad position_ids (shape: 3, batch, seq_len)
                if position_ids is not None:
                    position_ids = torch.cat([
                        position_ids,
                        position_ids[:, :1, :].repeat(1, pad_batch, 1)  # Repeat first sample's positions
                    ], dim=1)

            # Run V3 compiled language model (NxDModel)
            # V3 model expects: inputs_embeds, attention_mask, position_ids
            hidden_states = self.compiled_language_model_v3(
                inputs_embeds.to(torch.bfloat16),
                attention_mask,
                position_ids
            )

            # Remove batch padding from output
            if actual_batch_size < self.language_model_batch_size:
                hidden_states = hidden_states[:actual_batch_size]

            # Remove sequence padding from output
            hidden_states = hidden_states[:, :original_seq_len, :]

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states

        elif self.compiled_language_model is not None:
            # Neuron compiled Language Model mode - requires fixed sequence length
            original_seq_len = inputs_embeds.shape[1]
            hidden_size = inputs_embeds.shape[2]

            if original_seq_len < self.max_seq_len:
                # Pad inputs_embeds with zeros
                pad_len = self.max_seq_len - original_seq_len
                embed_padding = torch.zeros(
                    batch_size, pad_len, hidden_size,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                inputs_embeds = torch.cat([inputs_embeds, embed_padding], dim=1)

                # Pad attention_mask with zeros (masked positions)
                if attention_mask is not None:
                    mask_padding = torch.zeros(
                        batch_size, pad_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

                # Pad position_ids with sequential positions
                if position_ids is not None:
                    # position_ids shape: (3, batch, seq_len)
                    # Pad with sequential positions continuing from the last position
                    last_pos = position_ids[:, :, -1:] + 1  # (3, batch, 1)
                    pad_positions = last_pos + torch.arange(pad_len, device=position_ids.device).view(1, 1, -1)
                    position_ids = torch.cat([position_ids, pad_positions], dim=2)
            elif original_seq_len > self.max_seq_len:
                # Truncate if too long
                print(f"  WARNING: Sequence length {original_seq_len} > max_seq_len {self.max_seq_len}, truncating")
                inputs_embeds = inputs_embeds[:, :self.max_seq_len, :]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_seq_len]
                if position_ids is not None:
                    position_ids = position_ids[:, :, :self.max_seq_len]
                original_seq_len = self.max_seq_len

            # Run compiled language model with position_ids for M-RoPE
            hidden_states = self.compiled_language_model(inputs_embeds, attention_mask, position_ids)

            # Remove padding from output (restore original sequence length)
            hidden_states = hidden_states[:, :original_seq_len, :]

            # Create output similar to original
            if return_dict:
                return type('TextEncoderOutput', (), {
                    'hidden_states': (hidden_states,),
                    'last_hidden_state': hidden_states
                })()
            return hidden_states

        else:
            # No language model available
            raise RuntimeError(
                "No language model available! Please either:\n"
                "1. Compile V3 language model: python neuron_qwen_image_edit/compile_language_model_v3.py\n"
                "2. Compile V1 language model: python neuron_qwen_image_edit/compile_text_encoder.py --language_only\n"
                "3. Use CPU language model by passing cpu_language_model to NeuronTextEncoderWrapper"
            )

    def _merge_embeddings(self, text_embeds, image_embeds, input_ids, image_token_id):
        """
        Merge text and image embeddings at image token positions.

        OPTIMIZED: Uses index-based replacement to minimize CPU synchronization.
        """
        batch_size, seq_len, hidden_size = text_embeds.shape

        if image_embeds is None:
            return text_embeds

        # Find positions of image tokens
        image_mask = (input_ids == image_token_id)  # [batch, seq]

        # Clone to avoid modifying original
        inputs_embeds = text_embeds.clone()

        # For batch_size=1, use optimized path with nonzero
        if batch_size == 1:
            # Get indices of image tokens (returns [N, 2] for 2D input, we need column 1)
            image_indices = image_mask[0].nonzero(as_tuple=True)[0]  # [num_image_tokens]
            num_image_positions = image_indices.shape[0]

            if num_image_positions > 0:
                # Handle case where image_embeds has fewer tokens than positions
                num_to_use = min(num_image_positions, image_embeds.shape[0])

                # Use index_copy_ for efficient in-place replacement
                inputs_embeds[0, image_indices[:num_to_use]] = image_embeds[:num_to_use]

            return inputs_embeds

        # For batch_size > 1, process each batch
        for b in range(batch_size):
            image_indices = image_mask[b].nonzero(as_tuple=True)[0]
            num_image_positions = image_indices.shape[0]

            if num_image_positions > 0:
                num_to_use = min(num_image_positions, image_embeds.shape[0])
                inputs_embeds[b, image_indices[:num_to_use]] = image_embeds[:num_to_use]

        return inputs_embeds


class InferenceTransformerWrapper(nn.Module):
    """Wrapper for QwenImageTransformer2DModel for inference on Neuron."""
    def __init__(self, transformer: QwenImageTransformer2DModel):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.dtype = transformer.dtype
        self.device = transformer.device

    def forward(self, hidden_states, encoder_hidden_states=None,
                timestep=None, encoder_attention_mask=None,
                pooled_projections=None, return_dict=False, **kwargs):
        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            return_dict=return_dict,
        )
        return output


class SimpleWrapper(nn.Module):
    """Simple wrapper for VAE decoder and other modules."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class f32Wrapper(nn.Module):
    """Wrapper to run normalization layers in float32 for numerical stability."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        t = x.dtype
        y = x.to(torch.float32)
        output = self.original(y)
        return output.type(t)


def neuron_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=None, is_causal=None, scale=None,
                                         enable_gqa=False, **kwargs):
    """Custom scaled dot product attention optimized for Neuron.

    Supports:
    - Grouped Query Attention (GQA) where num_kv_heads < num_q_heads
    - Causal masking when is_causal=True
    - Explicit attention masks (attn_mask)
    """
    orig_shape = None
    orig_query_shape = query.shape
    q_len = query.shape[-2]
    kv_len = key.shape[-2]

    if len(query.shape) == 4:
        orig_shape = query.shape
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, _, _ = key.shape

        # Handle GQA: repeat K/V heads to match Q heads
        if num_kv_heads != num_q_heads:
            num_groups = num_q_heads // num_kv_heads
            # Repeat K and V along head dimension
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)

        def to3d(x):
            return x.reshape(-1, x.shape[2], x.shape[3])
        query, key, value = map(to3d, [query, key, value])

    # Use provided scale or default to 1/sqrt(d_k)
    if scale is None:
        scale = 1 / math.sqrt(query.size(-1))

    # Compute attention scores: [batch*heads, q_len, kv_len]
    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * scale

    # Apply causal mask if requested
    if is_causal:
        # Create causal mask: positions above the main diagonal are masked (-inf)
        # Shape: (q_len, kv_len)
        # Use torch.where to avoid NaN from 0 * -inf
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=attention_scores.device),
            diagonal=1
        )
        causal_mask = torch.where(
            causal_mask == 1,
            torch.tensor(float('-inf'), dtype=attention_scores.dtype, device=attention_scores.device),
            torch.tensor(0.0, dtype=attention_scores.dtype, device=attention_scores.device)
        )
        attention_scores = attention_scores + causal_mask

    # Apply explicit attention mask if provided
    if attn_mask is not None:
        # attn_mask can be:
        # - 2D: (q_len, kv_len) - applied to all batches/heads
        # - 3D: (batch*heads, q_len, kv_len) - per-head mask
        # - 4D: (batch, heads, q_len, kv_len) - full mask
        if attn_mask.dim() == 4:
            # Reshape 4D mask to 3D
            attn_mask = attn_mask.reshape(-1, attn_mask.shape[-2], attn_mask.shape[-1])
        elif attn_mask.dim() == 2:
            # Broadcast 2D mask
            attn_mask = attn_mask.unsqueeze(0)

        # Convert boolean mask to additive mask if needed
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        attention_scores = attention_scores + attn_mask.to(attention_scores.dtype)

    attention_probs = attention_scores.softmax(dim=-1)
    attn_out = torch.bmm(attention_probs, value)

    if orig_shape:
        attn_out = attn_out.reshape(
            orig_shape[0], orig_shape[1], attn_out.shape[1], attn_out.shape[2]
        )
    return attn_out


def attention_wrapper_sharded_without_swap(query, key, value):
    """Sharded attention wrapper using NKI kernel for trn2.

    Note: This kernel requires Q, K, V to have the same sequence length.
    For cross-attention with different lengths, fall back to basic attention.
    """
    import os

    bs, n_head, q_len, d_head = query.shape
    _, _, kv_len, _ = key.shape

    # NKI kernel requires same sequence length for Q, K, V and NKI must be available
    if q_len != kv_len or not NKI_AVAILABLE or _flash_fwd_call is None:
        # Fall back to basic attention
        return neuron_scaled_dot_product_attention(query, key, value)

    # Reshape for NKI kernel: expects [bs*n_head, d_head, seq_len] for Q, K
    # and [bs*n_head, seq_len, d_head] for V
    q = query.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs*n_head, d_head, kv_len))
    v = value.clone().reshape((bs*n_head, kv_len, d_head))
    attn_output = torch.zeros((bs*n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    # Compute scale: 1/sqrt(d_head)
    scale = 1.0 / math.sqrt(d_head)

    # Check if using virtual core size 2 (TRN2 default)
    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "2"))
    use_sharded_attention_kernel = (vc_size == 2)

    if use_sharded_attention_kernel:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output,
                              kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output,
                        kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output


# Store original SDPA function
sdpa_original = torch.nn.functional.scaled_dot_product_attention


def attention_wrapper(query, key, value, attn_mask=None, dropout_p=None, is_causal=None,
                      scale=None, enable_gqa=False):
    """Attention wrapper for text encoder.

    Always uses our custom implementation for better Neuron tracing compatibility.
    The custom implementation supports:
    - Causal masking (is_causal=True)
    - Explicit attention masks (attn_mask)
    - GQA (handled by repeat_kv in model's forward, but we handle leftovers)
    """
    # Always use our custom implementation for Neuron compatibility
    return neuron_scaled_dot_product_attention(query, key, value,
                                               attn_mask=attn_mask,
                                               dropout_p=dropout_p,
                                               is_causal=is_causal,
                                               scale=scale)


def attention_wrapper_for_transformer(query, key, value, attn_mask=None,
                                       dropout_p=None, is_causal=None,
                                       scale=None, enable_gqa=False):
    """Attention wrapper for transformer using NKI Flash Attention kernel.

    Uses NKI kernel for optimal performance on Trainium2.
    Falls back to basic attention for incompatible shapes.
    """
    # Check if NKI kernel can be used:
    # 1. NKI must be available
    # 2. Q, K, V must have same sequence length (joint attention)
    # 3. No attention mask (NKI doesn't support masks well)
    # 4. Not causal attention

    bs, n_head, q_len, d_head = query.shape
    _, _, kv_len, _ = key.shape

    use_nki = (
        NKI_AVAILABLE and
        _flash_fwd_call is not None and
        q_len == kv_len and
        attn_mask is None and
        not is_causal
    )

    if use_nki:
        # Use NKI Flash Attention kernel
        return attention_wrapper_sharded_without_swap(query, key, value)
    else:
        # Fall back to basic attention
        return neuron_scaled_dot_product_attention(query, key, value,
                                                   attn_mask=attn_mask,
                                                   dropout_p=dropout_p,
                                                   is_causal=is_causal)
