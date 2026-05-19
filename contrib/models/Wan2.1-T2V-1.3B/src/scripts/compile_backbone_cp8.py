"""CP=8 WAN backbone split into 2 NEFFs of 15 blocks each."""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, time

sys.path.insert(0, "/mnt/work/wan/all_on_neuron_checkpt/contrib/models/wan2.1-t2v-1.3b/src")

import neuronx_distributed
import torch_neuronx
from neuronx_distributed.parallel_layers.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_size, get_tensor_model_parallel_rank,
)
from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="xla")
initialize_model_parallel(tensor_model_parallel_size=8)
rank = get_tensor_model_parallel_rank()
cp = get_tensor_model_parallel_size()

class CPAttn(nn.Module):
    def __init__(self, dim, heads, dim_head, eps=1e-6):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        inner = dim_head * heads
        self.to_q = nn.Linear(dim, inner, bias=True)
        self.to_k = nn.Linear(dim, inner, bias=True)
        self.to_v = nn.Linear(dim, inner, bias=True)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=True), nn.Dropout(0.0))
        self.norm_q = nn.RMSNorm(inner, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(inner, eps=eps, elementwise_affine=True)

    def _gather_seq(self, t):
        return gather_from_tensor_model_parallel_region(t.transpose(1, 2)).transpose(1, 2)

    def forward(self, x, enc=None, rcl=None, rsl=None, rcf=None, rsf=None):
        kv = enc if enc is not None else x
        q = self.norm_q(self.to_q(x))
        k = self.norm_k(self.to_k(kv))
        v = self.to_v(kv)
        if enc is None:
            k = self._gather_seq(k); v = self._gather_seq(v)
        q = q.unflatten(2, (self.heads, self.dim_head)).transpose(1, 2)
        k = k.unflatten(2, (self.heads, self.dim_head)).transpose(1, 2)
        v = v.unflatten(2, (self.heads, self.dim_head)).transpose(1, 2)
        if rcl is not None and enc is None:
            def rope(x, c, s):
                c = c.transpose(1, 2); s = s.transpose(1, 2)
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * c[..., 0::2] - x2 * s[..., 1::2]
                out[..., 1::2] = x1 * s[..., 1::2] + x2 * c[..., 0::2]
                return out.type_as(x)
            q = rope(q, rcl, rsl); k = rope(k, rcf, rsf)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2, 3)
        return self.to_out(out)

class CPBlock(nn.Module):
    def __init__(self, dim=1536, ffn_dim=8960, heads=12, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = CPAttn(dim, heads, dim // heads, eps)
        self.attn2 = CPAttn(dim, heads, dim // heads, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=True)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, enc, temb, rcl, rsl, rcf, rsf):
        s = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
        h = (self.norm1(x.float()) * (1 + s[1]) + s[0]).type_as(x)
        x = (x.float() + self.attn1(h, rcl=rcl, rsl=rsl, rcf=rcf, rsf=rsf) * s[2]).type_as(x)
        h = self.norm2(x.float()).type_as(x)
        x = x + self.attn2(h, enc=enc)
        h = (self.norm3(x.float()) * (1 + s[4]) + s[3]).type_as(x)
        x = (x.float() + self.ffn_down(F.gelu(self.ffn_up(h), approximate="tanh")).float() * s[5]).type_as(x)
        return x

# NEFF 1: patch_embed + condition + blocks[0:15] + pre-gather RoPE
class NEFF1(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(16, 1536, kernel_size=(1,2,2), stride=(1,2,2))
        self.blocks = nn.ModuleList([CPBlock() for _ in range(15)])
        self.condition_embedder = None

    def forward(self, hidden_states, timestep, enc, rope_cos, rope_sin):
        x = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, tp, enc, _ = self.condition_embedder(timestep.expand(x.shape[0]), enc, None, timestep_seq_len=None)
        tp = tp.unflatten(1, (6, -1))
        full_seq = x.shape[1]
        local_seq = full_seq // cp
        x = x[:, rank * local_seq:(rank + 1) * local_seq]
        rcl = rope_cos[:, rank * local_seq:(rank + 1) * local_seq]
        rsl = rope_sin[:, rank * local_seq:(rank + 1) * local_seq]
        rcf = gather_from_tensor_model_parallel_region(rcl.squeeze(2).transpose(1,2)).transpose(1,2).unsqueeze(2)
        rsf = gather_from_tensor_model_parallel_region(rsl.squeeze(2).transpose(1,2)).transpose(1,2).unsqueeze(2)
        for block in self.blocks:
            x = block(x, enc, tp, rcl, rsl, rcf, rsf)
        return x, temb, enc, tp, rcl, rsl, rcf, rsf

# NEFF 2: blocks[15:30] + norm + proj + gather
class NEFF2(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([CPBlock() for _ in range(15)])
        self.norm_out = nn.LayerNorm(1536, elementwise_affine=False)
        self.proj_out = nn.Linear(1536, 64, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, 1536) / 1536**0.5)

    def forward(self, x, temb, enc, tp, rcl, rsl, rcf, rsf):
        for block in self.blocks:
            x = block(x, enc, tp, rcl, rsl, rcf, rsf)
        x = gather_from_tensor_model_parallel_region(x.transpose(1, 2)).transpose(1, 2)
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = (self.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        return self.proj_out(x)

# Load weights
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
hf = WanTransformer3DModel.from_pretrained('/mnt/work/wan/agent_artifacts/data/transformer', torch_dtype=torch.bfloat16)

n1 = NEFF1().to(torch.bfloat16).eval()
n2 = NEFF2().to(torch.bfloat16).eval()
n1.condition_embedder = hf.condition_embedder

hf_sd = hf.state_dict()
for neff, prefix, block_range in [(n1, "", range(15)), (n2, "", range(15, 30))]:
    mapped = {}
    for k, v in hf_sd.items():
        mk = k.replace(".ffn.net.0.proj.", ".ffn_up.").replace(".ffn.net.2.", ".ffn_down.")
        if ".to_out.1." in k: continue
        # Remap block indices for NEFF2
        if neff is n2:
            for bi in block_range:
                mk = mk.replace(f"blocks.{bi}.", f"blocks.{bi-15}.")
        if mk in neff.state_dict(): mapped[mk] = v
    neff.load_state_dict(mapped, strict=False)

del hf, hf_sd

# Trace
h = torch.randn(1, 16, 13, 60, 104, dtype=torch.bfloat16)
t = torch.tensor([999], dtype=torch.int64)
e = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
hf2 = WanTransformer3DModel.from_pretrained('/mnt/work/wan/agent_artifacts/data/transformer', torch_dtype=torch.bfloat16)
rc, rs = hf2.rope(h); del hf2

CC = "--model-type=transformer -O1 --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=false'"

print(f"[Rank {rank}] Tracing NEFF1 (15 blocks)...")
t0 = time.time()
try:
    tr1 = torch_neuronx.trace(n1, (h, t, e, rc, rs), compiler_args=CC)
    print(f"[Rank {rank}] NEFF1 COMPILED in {time.time()-t0:.0f}s!")
except Exception as ex:
    print(f"[Rank {rank}] NEFF1 FAILED: {str(ex)[:150]}")
    sys.exit(1)

# Get NEFF1 output shapes for NEFF2 input
r1 = tr1(h, t, e, rc, rs)
x_mid = r1[0]
print(f"[Rank {rank}] NEFF1 output: x={x_mid.shape}")

print(f"[Rank {rank}] Tracing NEFF2 (15 blocks)...")
t0 = time.time()
try:
    tr2 = torch_neuronx.trace(n2, r1, compiler_args=CC)
    print(f"[Rank {rank}] NEFF2 COMPILED in {time.time()-t0:.0f}s!")
    final = tr2(*r1)
    print(f"[Rank {rank}] Final output: {final.shape}")
    
    out_dir = "/mnt/work/wan/agent_artifacts/data/traced_wan_cp8_49f"
    os.makedirs(out_dir, exist_ok=True)
    torch.jit.save(tr1, f"{out_dir}/neff1_rank{rank}.pt")
    torch.jit.save(tr2, f"{out_dir}/neff2_rank{rank}.pt")
    print(f"[Rank {rank}] Saved!")
except Exception as ex:
    print(f"[Rank {rank}] NEFF2 FAILED: {str(ex)[:150]}")
