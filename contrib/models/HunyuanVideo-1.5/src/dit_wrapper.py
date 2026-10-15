"""
HunyuanVideo-1.5 DiT Tracing Wrapper for Standard Neuron SDK.

Creates a traceable nn.Module that wraps the 8.33B DiT transformer.
All dynamic control flow, data-dependent branches, and untraceable ops
are moved to CPU preprocessing.

480p_t2v config:
  - hidden_size=2048, heads_num=16, head_dim=128
  - 54 double-stream blocks, 0 single-stream blocks
  - patch_size=[1,1,1], in_channels=32 (with concat: 32*2+1=65)
  - text_states_dim=3584 (Qwen2.5-VL)
  - glyph_byT5_v2=True, vision_projection="linear"
  - use_cond_type_embedding=True
  - rope_dim_list=[16, 56, 56], rope_theta=256

Tensor Parallel Strategy (TP=4):
  - ColumnParallelLinear: Q/K/V projections, gate projections, up projections
  - RowParallelLinear: output projections, down projections
  - 16 heads / 4 TP = 4 heads per rank

Forward signature for tracing (all tensors, no kwargs branching):
  1. img_tokens      [B, L_img, 2048]  -- already patch-embedded on CPU
  2. vec             [B, 2048]          -- timestep + text_pool + guidance (precomputed)
  3. txt_tokens      [B, L_txt_total, 2048]  -- refined + reordered text (precomputed)
  4. txt_mask        [B, L_txt_total]   -- attention mask for text
  5. freqs_cos       [L_img, 128]       -- RoPE cos (precomputed)
  6. freqs_sin       [L_img, 128]       -- RoPE sin (precomputed)

Returns:
  img_out           [B, L_img, out_channels * prod(patch_size)]
  (unpatchify happens on CPU after)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HunyuanDiTTracingWrapper(nn.Module):
    """
    Traceable wrapper around HunyuanVideo DiT transformer blocks.

    Moves ALL preprocessing (patch embed, timestep embed, token refiner,
    byT5 mapper, vision projection, token reordering, RoPE) to CPU.
    Only the core transformer blocks + final layer run on device.

    This produces a clean forward signature with only tensor inputs
    and no control flow, making it compatible with torch_neuronx.trace()
    and ModelBuilder.trace().
    """

    def __init__(self, dit_model):
        super().__init__()
        # Copy transformer blocks and final layer from the original model
        self.double_blocks = dit_model.double_blocks
        self.final_layer = dit_model.final_layer

        # Store config
        self.hidden_size = dit_model.hidden_size
        self.heads_num = dit_model.heads_num
        self.head_dim = self.hidden_size // self.heads_num

        # Attention mode forced to "torch" (SDPA)
        for block in self.double_blocks:
            if hasattr(block, "attn_mode"):
                block.attn_mode = "torch"

    def forward(
        self,
        img: torch.Tensor,  # [B, L_img, hidden_size]
        txt: torch.Tensor,  # [B, L_txt, hidden_size]
        vec: torch.Tensor,  # [B, hidden_size]
        txt_mask: torch.Tensor,  # [B, L_txt]
        freqs_cos: torch.Tensor,  # [L_img, head_dim]
        freqs_sin: torch.Tensor,  # [L_img, head_dim]
    ) -> torch.Tensor:
        """
        Core transformer forward -- only double blocks + final layer.
        All preprocessing done on CPU.
        """
        freqs_cis = (freqs_cos, freqs_sin)

        # Run 54 double-stream blocks
        # Note: txt_mask is passed for blocks that support it (TP blocks).
        # For original HunyuanVideo blocks, text_mask is a keyword argument.
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, freqs_cis=freqs_cis, text_mask=txt_mask)

        # Final layer: AdaLN + linear projection
        img = self.final_layer(img, vec)

        return img


class HunyuanDiTPreprocessor:
    """
    CPU-side preprocessor that prepares all inputs for the traced DiT wrapper.

    This handles:
    - Patch embedding (Conv3d)
    - Timestep embedding + text pool + guidance
    - Token refiner (text encoding)
    - ByT5 mapper
    - Vision projection
    - Token reordering (data-dependent, cannot trace)
    - RoPE precomputation
    - Cond type embeddings
    """

    def __init__(self, dit_model, device="cpu", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        # Move preprocessing modules to CPU
        self.img_in = dit_model.img_in.to(device).to(dtype)
        self.time_in = dit_model.time_in.to(device).to(dtype)
        self.txt_in = dit_model.txt_in.to(device).to(dtype)
        self.vector_in = (
            dit_model.vector_in.to(device).to(dtype)
            if dit_model.vector_in is not None
            else None
        )
        self.guidance_in = (
            dit_model.guidance_in.to(device).to(dtype)
            if dit_model.guidance_in is not None
            else None
        )
        self.byt5_in = (
            dit_model.byt5_in.to(device).to(dtype)
            if hasattr(dit_model, "byt5_in") and dit_model.byt5_in is not None
            else None
        )
        self.vision_in = (
            dit_model.vision_in.to(device).to(dtype)
            if hasattr(dit_model, "vision_in") and dit_model.vision_in is not None
            else None
        )
        self.cond_type_embedding = (
            dit_model.cond_type_embedding.to(device)
            if hasattr(dit_model, "cond_type_embedding")
            and dit_model.cond_type_embedding is not None
            else None
        )

        # Config
        self.patch_size = dit_model.patch_size
        self.hidden_size = dit_model.hidden_size
        self.heads_num = dit_model.heads_num
        self.head_dim = self.hidden_size // self.heads_num
        self.rope_dim_list = dit_model.rope_dim_list
        self.rope_theta = dit_model.rope_theta
        self.glyph_byT5_v2 = dit_model.glyph_byT5_v2
        self.use_attention_mask = dit_model.use_attention_mask

        # Import RoPE function
        from hyvideo.models.transformers.modules.posemb_layers import (
            get_nd_rotary_pos_embed,
        )

        self.get_nd_rotary_pos_embed = get_nd_rotary_pos_embed

    def compute_rope(self, tt, th, tw):
        """Precompute RoPE frequencies on CPU."""
        freqs_cos, freqs_sin = self.get_nd_rotary_pos_embed(
            tuple(self.rope_dim_list),
            (tt, th, tw),
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos.to(self.dtype), freqs_sin.to(self.dtype)

    @torch.no_grad()
    def preprocess(
        self,
        hidden_states: torch.Tensor,  # [B, C, T, H, W] on CPU
        timestep: torch.Tensor,  # [B]
        text_states: torch.Tensor,  # [B, L_txt, 3584]
        text_states_2: Optional[torch.Tensor],  # [B, 768] or None
        encoder_attention_mask: torch.Tensor,  # [B, L_txt]
        byt5_text_states: Optional[torch.Tensor] = None,  # [B, L_byt5, 1472]
        byt5_text_mask: Optional[torch.Tensor] = None,  # [B, L_byt5]
        vision_states: Optional[torch.Tensor] = None,  # [B, L_vis, 1152]
        guidance: Optional[torch.Tensor] = None,  # [B]
        mask_type: str = "t2v",
    ):
        """
        Run all preprocessing on CPU. Returns tensors ready for the traced wrapper.
        """
        B = hidden_states.shape[0]
        T, H, W = hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4]
        tt = T // self.patch_size[0]
        th = H // self.patch_size[1]
        tw = W // self.patch_size[2]

        # 1. Patch embedding
        img = self.img_in(hidden_states)  # [B, L_img, hidden_size]

        # 2. RoPE
        freqs_cos, freqs_sin = self.compute_rope(tt, th, tw)

        # 3. Timestep + conditioning vector
        t = timestep.to(self.dtype)
        vec = self.time_in(t)  # [B, hidden_size]
        if text_states_2 is not None and self.vector_in is not None:
            vec = vec + self.vector_in(text_states_2.to(self.dtype))
        if self.guidance_in is not None:
            if guidance is None:
                guidance = torch.tensor(
                    [6016.0], device=self.device, dtype=self.dtype
                ).expand(B)
            vec = vec + self.guidance_in(guidance.to(self.dtype))

        # 4. Token refiner for text
        text_mask = encoder_attention_mask.to(self.dtype)
        txt = self.txt_in(
            text_states.to(self.dtype),
            t,
            text_mask if self.use_attention_mask else None,
        )  # [B, L_txt, hidden_size]

        # 5. Cond type embedding for text
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], dtype=torch.long)
            )
            txt = txt + cond_emb

        # 6. ByT5 integration
        if (
            self.glyph_byT5_v2
            and self.byt5_in is not None
            and byt5_text_states is not None
        ):
            byt5_txt = self.byt5_in(byt5_text_states.to(self.dtype))
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb

            # Reorder tokens: valid byT5 first, valid text second, padding last
            # This is data-dependent and CANNOT be traced -- do it here on CPU
            txt, text_mask = self._reorder_tokens(
                byt5_txt, txt, byt5_text_mask, text_mask
            )

        # 7. Vision integration (for T2V: all zeros -> skip)
        if self.vision_in is not None and vision_states is not None:
            if mask_type == "t2v" and torch.all(vision_states == 0):
                # T2V mode: zero out vision features, set mask to zero
                vis_feat = torch.zeros(
                    B,
                    vision_states.shape[1],
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                vis_mask = torch.zeros(
                    B, vision_states.shape[1], device=self.device, dtype=self.dtype
                )
            else:
                vis_feat = self.vision_in(vision_states.to(self.dtype))
                vis_mask = torch.ones(
                    B, vision_states.shape[1], device=self.device, dtype=self.dtype
                )

            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2 * torch.ones_like(vis_feat[:, :, 0], dtype=torch.long)
                )
                vis_feat = vis_feat + cond_emb

            # Reorder with vision tokens
            txt, text_mask = self._reorder_tokens(vis_feat, txt, vis_mask, text_mask)

        return {
            "img": img,
            "txt": txt,
            "vec": vec,
            "txt_mask": text_mask.to(torch.long),
            "freqs_cos": freqs_cos,
            "freqs_sin": freqs_sin,
            "tt": tt,
            "th": th,
            "tw": tw,
        }

    def _reorder_tokens(self, new_tokens, existing_tokens, new_mask, existing_mask):
        """
        Reorder: valid new tokens first, valid existing tokens second, padding last.
        Data-dependent -- must run on CPU, cannot be traced.
        """
        B = existing_tokens.shape[0]
        L_new = new_tokens.shape[1]
        L_old = existing_tokens.shape[1]
        L_total = L_new + L_old
        hidden = existing_tokens.shape[2]

        combined_tokens = torch.zeros(
            B,
            L_total,
            hidden,
            device=existing_tokens.device,
            dtype=existing_tokens.dtype,
        )
        combined_mask = torch.zeros(
            B, L_total, device=existing_mask.device, dtype=existing_mask.dtype
        )

        for i in range(B):
            # Valid new tokens
            new_valid = (
                new_mask[i].bool()
                if new_mask is not None
                else torch.ones(L_new, dtype=torch.bool)
            )
            new_valid_tokens = new_tokens[i][new_valid]
            n_new = new_valid_tokens.shape[0]

            # Valid existing tokens
            old_valid = existing_mask[i].bool()
            old_valid_tokens = existing_tokens[i][old_valid]
            n_old = old_valid_tokens.shape[0]

            # Place: valid new, valid old, rest is padding (zeros)
            combined_tokens[i, :n_new] = new_valid_tokens
            combined_tokens[i, n_new : n_new + n_old] = old_valid_tokens
            combined_mask[i, : n_new + n_old] = 1.0

        return combined_tokens, combined_mask

    def unpatchify(self, img_out, tt, th, tw, out_channels=32):
        """Reshape flat token output back to [B, C, T, H, W] on CPU."""
        p = self.patch_size
        img_out = img_out.reshape(
            img_out.shape[0], tt, th, tw, out_channels, p[0], p[1], p[2]
        )
        # nthwcopq -> nctohpwq
        img_out = torch.einsum("nthwcopq->nctohpwq", img_out)
        img_out = img_out.reshape(
            img_out.shape[0], out_channels, tt * p[0], th * p[1], tw * p[2]
        )
        return img_out


def create_sample_inputs(
    batch_size=1,
    hidden_size=2048,
    img_seq_len=3180,  # 480p: 2*30*53
    txt_seq_len=320,  # 256 byT5 + 64 LLM text (after reorder)
    head_dim=128,
    dtype=torch.bfloat16,
):
    """Create sample inputs for tracing at 480p_5f scale."""
    return {
        "img": torch.randn(batch_size, img_seq_len, hidden_size, dtype=dtype),
        "txt": torch.randn(batch_size, txt_seq_len, hidden_size, dtype=dtype),
        "vec": torch.randn(batch_size, hidden_size, dtype=dtype),
        "txt_mask": torch.ones(batch_size, txt_seq_len, dtype=torch.long),
        "freqs_cos": torch.randn(img_seq_len, head_dim, dtype=dtype),
        "freqs_sin": torch.randn(img_seq_len, head_dim, dtype=dtype),
    }
