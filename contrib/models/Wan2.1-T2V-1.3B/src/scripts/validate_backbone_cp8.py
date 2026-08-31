"""Validate CP=8 backbone: cosine vs CPU, then full denoising loop."""
import torch, torch.nn.functional as F, os, sys, time, numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/work/wan/all_on_neuron_checkpt/contrib/models/wan2.1-t2v-1.3b/src")

import neuronx_distributed
import torch_neuronx
from neuronx_distributed.parallel_layers.parallel_state import (
    initialize_model_parallel, get_tensor_model_parallel_rank, get_tensor_model_parallel_size,
)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="xla")
initialize_model_parallel(tensor_model_parallel_size=8)
rank = get_tensor_model_parallel_rank()

# Load NEFFs
sd = "/mnt/work/wan/agent_artifacts/data/traced_wan_cp8_49f"
tr1 = torch.jit.load(f"{sd}/neff1_rank{rank}.pt")
tr2 = torch.jit.load(f"{sd}/neff2_rank{rank}.pt")
print(f"[Rank {rank}] NEFFs loaded")

# Inputs
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
hf = WanTransformer3DModel.from_pretrained('/mnt/work/wan/agent_artifacts/data/transformer', torch_dtype=torch.bfloat16)

latent = torch.randn(1, 16, 13, 60, 104, dtype=torch.bfloat16)
t = torch.tensor([999], dtype=torch.int64)
rc, rs = hf.rope(latent)

# Load T5 embeddings
embeds = torch.load('/tmp/t5_embeds.pt')
pe = embeds['pe']

# CPU reference (single step)
if rank == 0:
    from modeling_wan import WanBackboneWrapper
    wrapper = WanBackboneWrapper(hf)
    with torch.no_grad():
        cpu_out = wrapper(latent, t, pe, rc, rs)
    print(f"[Rank 0] CPU output: {cpu_out.shape}")

# Neuron forward (2 NEFFs)
r1 = tr1(latent, t, pe, rc, rs)
final_flat = tr2(*r1)

# Unpatchify
b = 1; ppf, pph, ppw = 13, 30, 52
p_t, p_h, p_w = 1, 2, 2
out = final_flat.reshape(b, ppf, pph, ppw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
neuron_out = out.flatten(6, 7).flatten(4, 5).flatten(2, 3)

if rank == 0:
    cos = F.cosine_similarity(cpu_out.flatten().float(), neuron_out.flatten().float(), dim=0)
    print(f"[Rank 0] Cosine vs CPU: {cos.item():.6f}")

# Full denoising loop
if rank == 0:
    print(f"[Rank 0] Starting 20-step denoising...")

from diffusers import UniPCMultistepScheduler
scheduler = UniPCMultistepScheduler.from_pretrained('/mnt/work/wan/agent_artifacts/data/scheduler')

ne = embeds['ne']
gen = torch.Generator("cpu").manual_seed(42)
latents = torch.randn(1, 16, 13, 60, 104, dtype=torch.float32, generator=gen)
scheduler.set_timesteps(20, device="cpu")

t0 = time.time()
for si, tt in enumerate(scheduler.timesteps):
    li = latents.to(torch.bfloat16)
    ts = tt.expand(1)
    
    # CFG: cond
    r1c = tr1(li, ts, pe, rc, rs)
    fc = tr2(*r1c)
    cond = fc.reshape(1, 13, 30, 52, 1, 2, 2, -1).permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6,7).flatten(4,5).flatten(2,3)
    
    # CFG: uncond
    r1u = tr1(li, ts, ne, rc, rs)
    fu = tr2(*r1u)
    uncond = fu.reshape(1, 13, 30, 52, 1, 2, 2, -1).permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6,7).flatten(4,5).flatten(2,3)
    
    noise_pred = uncond + 5.0 * (cond - uncond)
    latents = scheduler.step(noise_pred, tt, latents, return_dict=False)[0]
    
    if rank == 0 and si % 5 == 0:
        print(f"  step {si}/20")

bb_time = time.time() - t0
if rank == 0:
    print(f"[Rank 0] Backbone: {bb_time:.1f}s")
    torch.save(latents, "/tmp/latents_49f.pt")
    print(f"[Rank 0] Latents saved to /tmp/latents_49f.pt")

del hf
