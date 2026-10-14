"""
HunyuanVideo-1.5 Text Processing Components for Neuron

Traced models (torch_neuronx.trace) for:
  - ByT5 text encoder (217M)
  - Token refiner (141M)
  - Token reorder

These use torch_neuronx.trace rather than NxDI ModelWrapper because they are
small models that don't need tensor parallelism.

Usage:
    # Compile all
    compile_byt5_encoder(save_path)
    compile_token_refiner(save_path)
    compile_token_reorder(save_path)

    # Load and use
    byt5 = load_traced_model(byt5_path)
    refiner = load_traced_model(refiner_path)
    reorder = load_traced_model(reorder_path)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx


# ---------------------------------------------------------------------------
# ByT5 Text Encoder
# ---------------------------------------------------------------------------
class ByT5EncoderWrapper(nn.Module):
    """Wraps HF T5EncoderModel for tracing."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def compile_byt5_encoder(save_path: str, model_id: str = "google/byt5-small"):
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()
    wrapper = ByT5EncoderWrapper(model).bfloat16().eval()
    ids = torch.randint(0, 384, [1, 256])
    mask = torch.ones(1, 256, dtype=torch.float32)
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer -O1 --auto-cast=none"
    traced = torch_neuronx.trace(wrapper, (ids, mask),
        compiler_args="--model-type=transformer -O1 --auto-cast=none")
    torch.jit.save(traced, save_path)
    print(f"ByT5 encoder compiled: {save_path}")


# ---------------------------------------------------------------------------
# Token Refiner
# ---------------------------------------------------------------------------
class RefinerAdaNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()

    def forward(self, temb):
        t = self.linear(self.silu(temb))
        g_msa, g_mlp = t.chunk(2, dim=1)
        return g_msa.unsqueeze(1), g_mlp.unsqueeze(1)


class RefinerAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

    def forward(self, hs, attn_mask=None):
        B, S, _ = hs.shape
        q = self.to_q(hs).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(hs).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(hs).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.to_out(out.transpose(1, 2).reshape(B, S, -1))


class RefinerFFN(nn.Module):
    def __init__(self, dim, mult=4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj = nn.Linear(dim, inner)
        self.act = nn.SiLU()
        self.out = nn.Linear(inner, dim)

    def forward(self, x):
        return self.out(self.act(self.proj(x)))


class RefinerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.attn = RefinerAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.ff = RefinerFFN(dim, mult=mlp_ratio)
        self.norm_out = RefinerAdaNorm(dim)

    def forward(self, hs, temb, attn_mask=None):
        attn_out = self.attn(self.norm1(hs), attn_mask)
        g_msa, g_mlp = self.norm_out(temb)
        hs = hs + attn_out * g_msa
        hs = hs + self.ff(self.norm2(hs)) * g_mlp
        return hs


class PlainCombinedTimestepTextProj(nn.Module):
    """Timestep + pooled text projection. Takes pre-computed ts_proj [B,256]."""
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()
        self.ts_l1 = nn.Linear(256, embedding_dim)
        self.ts_silu = nn.SiLU()
        self.ts_l2 = nn.Linear(embedding_dim, embedding_dim)
        self.text_l1 = nn.Linear(pooled_projection_dim, embedding_dim)
        self.text_silu = nn.SiLU()
        self.text_l2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, ts_proj, pooled):
        ts_emb = self.ts_l2(self.ts_silu(self.ts_l1(ts_proj)))
        text_emb = self.text_l2(self.text_silu(self.text_l1(pooled)))
        return ts_emb + text_emb


class TokenRefinerModel(nn.Module):
    """
    Token refiner: refines MLLM embeddings conditioned on timestep.

    Inputs:
        hs: [B, 1000, 3584] — raw MLLM embeddings
        ts_proj: [B, 256] — pre-computed sinusoidal timestep projection
        pooled: [B, 3584] — pooled MLLM embeddings (masked mean)
        refiner_attn_mask: [B, 1, 1000, 1000] — attention mask

    Output:
        [B, 1000, 2048] — refined embeddings
    """
    def __init__(self, in_ch=3584, dim=2048, num_heads=16, head_dim=128,
                 num_layers=2, mlp_ratio=4.0):
        super().__init__()
        self.time_text_embed = PlainCombinedTimestepTextProj(dim, in_ch)
        self.proj_in = nn.Linear(in_ch, dim, bias=True)
        self.refiner_blocks = nn.ModuleList([
            RefinerBlock(dim, num_heads, head_dim, mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, hs, ts_proj, pooled, refiner_attn_mask):
        temb = self.time_text_embed(ts_proj, pooled)
        hs = self.proj_in(hs)
        for block in self.refiner_blocks:
            hs = block(hs, temb, refiner_attn_mask)
        return hs


def _load_refiner_weights(model, hf_model_path):
    """Load token refiner weights from HF HunyuanVideo15Transformer3DModel."""
    from diffusers import HunyuanVideo15Transformer3DModel
    hf = HunyuanVideo15Transformer3DModel.from_pretrained(
        hf_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    sd = {}
    for k, v in hf.state_dict().items():
        if not k.startswith("context_embedder."):
            continue
        nk = k.replace("context_embedder.", "")
        nk = nk.replace("time_text_embed.timestep_embedder.linear_1.", "time_text_embed.ts_l1.")
        nk = nk.replace("time_text_embed.timestep_embedder.linear_2.", "time_text_embed.ts_l2.")
        nk = nk.replace("time_text_embed.text_embedder.linear_1.", "time_text_embed.text_l1.")
        nk = nk.replace("time_text_embed.text_embedder.linear_2.", "time_text_embed.text_l2.")
        nk = nk.replace("token_refiner.", "")
        nk = nk.replace(".attn.to_out.0.", ".attn.to_out.")
        nk = nk.replace(".ff.net.0.proj.", ".ff.proj.")
        nk = nk.replace(".ff.net.2.", ".ff.out.")
        sd[nk] = v
    model.load_state_dict(sd)
    del hf
    return model


def compile_token_refiner(save_path: str,
                          hf_model_path: str = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"):
    model = TokenRefinerModel().bfloat16().eval()
    model = _load_refiner_weights(model, hf_model_path)
    hs = torch.randn(1, 1000, 3584, dtype=torch.bfloat16)
    ts_proj = torch.randn(1, 256, dtype=torch.bfloat16)
    pooled = torch.randn(1, 3584, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 1000, 1000, dtype=torch.bfloat16)
    os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer -O1 --auto-cast=none"
    traced = torch_neuronx.trace(model, (hs, ts_proj, pooled, mask),
        compiler_args="--model-type=transformer -O1 --auto-cast=none")
    torch.jit.save(traced, save_path)
    print(f"Token refiner compiled: {save_path}")


# ---------------------------------------------------------------------------
# Token Reorder
# ---------------------------------------------------------------------------
class TokenReorderModel(nn.Module):
    """
    Computes reorder indices from attention masks using sort (XLA-compatible).

    Inputs:
        mllm_mask: [1000] — 1=valid, 0=padding
        byt5_mask: [256] — 1=valid, 0=padding

    Outputs:
        idx: [1985] — reorder indices
        zero_mask: [1985] — 1=valid, 0=padding
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("im_zeros", torch.zeros(729))
        self.register_buffer("offsets", torch.cat([
            torch.arange(1000), torch.arange(256) + 1000, torch.arange(729) + 1256,
        ]))
        self.register_buffer("positions", torch.arange(1985, dtype=torch.float32))

    def forward(self, mllm_mask, byt5_mask):
        combined = torch.cat([mllm_mask, byt5_mask, self.im_zeros])
        _, sort_idx = torch.sort(-combined.float(), stable=True)
        idx = self.offsets[sort_idx]
        n_valid = combined.sum()
        zero_mask = (self.positions < n_valid).float()
        return idx, zero_mask


def compile_token_reorder(save_path: str):
    model = TokenReorderModel().eval()
    mm = torch.ones(1000)
    bm = torch.ones(256)
    os.environ["NEURON_CC_FLAGS"] = "-O1 --auto-cast=none"
    traced = torch_neuronx.trace(model, (mm, bm))
    torch.jit.save(traced, save_path)
    print(f"Token reorder compiled: {save_path}")


# ---------------------------------------------------------------------------
# Utility: load any traced model
# ---------------------------------------------------------------------------
def load_traced_model(path: str):
    """Load a torch_neuronx traced model. Requires torch_neuronx to be imported."""
    return torch.jit.load(path)
