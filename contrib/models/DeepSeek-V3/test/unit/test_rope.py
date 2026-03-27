# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import torch

from src.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    apply_rotary_pos_emb_interleave as hf_apply_rotary_pos_emb_interleave,
)
from .test_helper.reference_model import apply_rotary_emb

TEST_YARN_ROPE_CONFIG = {
    "dim": 6,
    "max_position_embeddings": 5,
    "max_seq_len": 10,
    "beta_fast": 32,
    "beta_slow": 1,
    "rope_theta": 10000.0,
    "factor": 40,
    "mscale": 1,
    "mscale_all_dim": 1,
}

# copy from the model.py (but without the polar as we need apply that separately later.
def reference_freqs_cis_table(yarn_cfg) -> torch.Tensor:
    seqlen = yarn_cfg["max_seq_len"]
    base = yarn_cfg["rope_theta"]

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    dim = yarn_cfg["dim"]
    max_position_embeddings = yarn_cfg["max_position_embeddings"]
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > max_position_embeddings:
        low, high = find_correction_range(yarn_cfg["beta_fast"], yarn_cfg["beta_slow"], dim, base, max_position_embeddings)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / yarn_cfg["factor"] * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return freqs

class TestDeepseekV3Rope(unittest.TestCase):
    def setUp(self):
        self.yarn_config = TEST_YARN_ROPE_CONFIG
        self.dim = TEST_YARN_ROPE_CONFIG["dim"]
        self.max_seq_len = self.yarn_config["max_seq_len"]
        self.reference_freqs = reference_freqs_cis_table(self.yarn_config)

        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            dim=self.dim,
            scaling_factor=self.yarn_config["factor"],
            base=self.yarn_config["rope_theta"],
            original_max_position_embeddings = self.yarn_config["max_position_embeddings"],
            max_position_embeddings = self.max_seq_len,
            mscale=self.yarn_config["mscale"],
            mscale_all_dim=self.yarn_config["mscale_all_dim"],
            beta_fast=self.yarn_config["beta_fast"],
            beta_slow=self.yarn_config["beta_slow"],
        )
        assert self.rotary_emb._mscale == 1.0, ("default test yarn config should produce value of 1 for _mscale,"
                                                " and ref doesnt use _mscale for rope which requires this to be 1")


    def test_freq_table(self):
        """ freq table is [seq_len, dim] represents the angle rotation map """
        test_freqs = self.rotary_emb.get_freqs_table(self.reference_freqs.device, self.max_seq_len)
        assert test_freqs.shape ==  torch.Size([self.max_seq_len, self.dim//2])
        torch.testing.assert_close(self.reference_freqs, test_freqs) # assert on the freq table equivalence


    def test_apply_rope(self):
        """ We compare reference and ours rope. Note they require different input tensor shape: BSHD v.s BHSD """

        SEQ_LEN = self.max_seq_len

        # reference
        for batch in [1, 2, 4]:
            for num_heads in [1, 2, 4]:
                k_pe = torch.rand(batch, SEQ_LEN, num_heads, self.dim) # BSHD

                # Test applying rope with provided precomputed freq table
                # reference method uses torch.polar to make freq table a complex tensor with sin and cos transformed
                # see https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L183
                freqs_cis = torch.polar(torch.ones_like(self.reference_freqs), self.reference_freqs)
                freqs_cis = freqs_cis[:SEQ_LEN]
                reference_rope = apply_rotary_emb(k_pe, freqs_cis)

                # ours
                # alternatively, we compute cos and sin cache
                k_pe = k_pe.transpose(1, 2) # BHSD
                cos, sin = self.rotary_emb(k_pe, self.max_seq_len)
                position_ids = torch.arange(SEQ_LEN).unsqueeze(dim=0)
                test_rope = apply_rotary_pos_emb(k_pe, cos, sin, position_ids).transpose(1,2)

                # result
                torch.testing.assert_close(reference_rope, test_rope)

    def test_matches_hf_interleave_rope(self):
        """Verify NXDI rotate_fn matches HF apply_rotary_pos_emb_interleave.

        Key layout difference:
        - HF transposes interleaved [r0,i0,r1,i1,...] -> split [r0,r1,...,i0,i1,...] BEFORE rotation
        - NXDI rotate_fn operates directly on interleaved layout

        Both produce the SAME rotation (verified by converting NXDI output from
        interleaved to split layout). This ensures the attention dot products
        q_pe @ k_pe^T will be identical since both q_pe and k_pe use the same layout.
        """
        dim = 64
        seq_len = 32

        rotary_emb = DeepseekV3YarnRotaryEmbedding(
            dim=dim,
            scaling_factor=1.0,
            base=10000.0,
            original_max_position_embeddings=4096,
            max_position_embeddings=4096,
            mscale=1.0,
            mscale_all_dim=0,
            beta_fast=32,
            beta_slow=1,
        )

        def interleaved_to_split(x):
            """Convert [r0,i0,r1,i1,...] -> [r0,r1,...,i0,i1,...] (same as HF transpose)."""
            b, h, s, d = x.shape
            return x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        for batch in [1, 2]:
            for num_heads in [1, 4]:
                q = torch.randn(batch, num_heads, seq_len, dim)
                position_ids = torch.arange(seq_len).unsqueeze(0)

                # NXDI path: output is in interleaved layout
                cos_nxdi, sin_nxdi = rotary_emb(q, seq_len)
                nxdi_out = apply_rotary_pos_emb(q, cos_nxdi, sin_nxdi, position_ids)

                # Convert NXDI output from interleaved -> split for comparison
                nxdi_out_split = interleaved_to_split(nxdi_out)

                # HF path: output is in split layout
                cos_half = cos_nxdi[:seq_len, :dim // 2]
                sin_half = sin_nxdi[:seq_len, :dim // 2]
                hf_cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0)
                hf_sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0)
                hf_out, _ = hf_apply_rotary_pos_emb_interleave(q, q, hf_cos, hf_sin)

                torch.testing.assert_close(nxdi_out_split, hf_out, atol=1e-5, rtol=1e-5)
