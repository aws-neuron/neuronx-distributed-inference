import math
import unittest

import torch

from neuronx_distributed_inference.models.deepseek.rope_util import (
    DeepseekV3YarnRotaryEmbedding,
    apply_rotary_pos_emb,
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
