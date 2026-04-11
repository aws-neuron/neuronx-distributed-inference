"""CPU-only unit tests for V-JEPA 2.1 encoder."""

import sys
import os
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from modeling_jepa21 import build_vjepa21_encoder, VisionTransformer


class TestEncoderConstruction:
    """Test that encoder models can be constructed with correct shapes."""

    def test_vit_base_construction(self):
        encoder = build_vjepa21_encoder(arch="vit_base", img_size=384, num_frames=16, pretrained=False)
        assert isinstance(encoder, VisionTransformer)
        assert encoder.embed_dim == 768
        assert len(encoder.blocks) == 12

    def test_vit_large_construction(self):
        encoder = build_vjepa21_encoder(arch="vit_large", img_size=384, num_frames=16, pretrained=False)
        assert encoder.embed_dim == 1024
        assert len(encoder.blocks) == 24

    def test_vit_giant_construction(self):
        encoder = build_vjepa21_encoder(arch="vit_giant", img_size=384, num_frames=16, pretrained=False)
        assert encoder.embed_dim == 1408
        assert len(encoder.blocks) == 40

    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError):
            build_vjepa21_encoder(arch="vit_nonexistent")


class TestEncoderForward:
    """Test encoder forward pass on CPU with various input shapes."""

    @pytest.fixture
    def vit_base(self):
        encoder = build_vjepa21_encoder(
            arch="vit_base", img_size=384, num_frames=16, pretrained=False
        )
        encoder.eval()
        return encoder

    def test_video_forward_shape(self, vit_base):
        """Test video input: (B, C, T, H, W) -> (B, N, D)."""
        x = torch.randn(1, 3, 16, 384, 384)
        with torch.no_grad():
            out = vit_base(x)
        # 16 frames / tubelet_size=2 = 8 temporal tokens
        # 384/16 = 24 spatial patches per dim
        # 8 * 24 * 24 = 4608 tokens
        assert out.shape == (1, 4608, 768), f"Expected (1, 4608, 768), got {out.shape}"

    def test_image_forward_shape(self, vit_base):
        """Test single-frame image input via img_temporal_dim_size=1."""
        x = torch.randn(1, 3, 1, 384, 384)
        with torch.no_grad():
            out = vit_base(x)
        # 1 frame / tubelet_size=1 (img path) = 1 temporal token
        # 24 * 24 = 576 spatial tokens
        assert out.shape == (1, 576, 768), f"Expected (1, 576, 768), got {out.shape}"

    def test_batch_forward(self, vit_base):
        """Test batched input."""
        x = torch.randn(2, 3, 16, 384, 384)
        with torch.no_grad():
            out = vit_base(x)
        assert out.shape == (2, 4608, 768)

    def test_hierarchical_output(self, vit_base):
        """Test hierarchical output mode returns concatenated features."""
        vit_base.return_hierarchical = True
        x = torch.randn(1, 3, 16, 384, 384)
        with torch.no_grad():
            out = vit_base(x, training=True)
        # 4 hierarchical layers * embed_dim=768 = 3072
        assert out.shape[0] == 1
        assert out.shape[1] == 4608
        assert out.shape[2] == 768 * 4  # 4 distillation layers
        vit_base.return_hierarchical = False

    def test_output_deterministic(self, vit_base):
        """Test that eval mode produces deterministic output."""
        x = torch.randn(1, 3, 16, 384, 384)
        with torch.no_grad():
            out1 = vit_base(x)
            out2 = vit_base(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_256_resolution(self):
        """Test with 256x256 resolution."""
        encoder = build_vjepa21_encoder(
            arch="vit_base", img_size=256, num_frames=16, pretrained=False
        )
        encoder.eval()
        x = torch.randn(1, 3, 16, 256, 256)
        with torch.no_grad():
            out = encoder(x)
        # 8 * 16 * 16 = 2048 tokens
        assert out.shape == (1, 2048, 768)


class TestEncoderComponents:
    """Test individual components."""

    def test_patch_embed_3d(self):
        from modeling_jepa21 import PatchEmbed3D
        pe = PatchEmbed3D(patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768)
        x = torch.randn(1, 3, 16, 384, 384)
        out = pe(x)
        # (16/2) * (384/16) * (384/16) = 8 * 24 * 24 = 4608
        assert out.shape == (1, 4608, 768)

    def test_patch_embed_3d_image(self):
        from modeling_jepa21 import PatchEmbed3D
        pe = PatchEmbed3D(patch_size=16, tubelet_size=1, in_chans=3, embed_dim=768)
        x = torch.randn(1, 3, 1, 384, 384)
        out = pe(x)
        assert out.shape == (1, 576, 768)

    def test_rope_attention(self):
        from modeling_jepa21 import RoPEAttention
        attn = RoPEAttention(
            dim=768, num_heads=12, qkv_bias=True, use_sdpa=False,
            grid_size=24, interpolate_rope=True, patch_size=16,
        )
        x = torch.randn(1, 576, 768)
        out, _ = attn(x, T=1, H_patches=24, W_patches=24)
        assert out.shape == (1, 576, 768)

    def test_block(self):
        from modeling_jepa21 import Block
        blk = Block(
            dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
            use_rope=True, grid_size=24, interpolate_rope=True, patch_size=16,
            norm_layer=torch.nn.LayerNorm,
        )
        x = torch.randn(1, 576, 768)
        out, _ = blk(x, T=1, H_patches=24, W_patches=24)
        assert out.shape == (1, 576, 768)
