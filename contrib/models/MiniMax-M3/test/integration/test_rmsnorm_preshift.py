"""CPU validation: Gemma-style pre-shift on M3 RMSNorm.

Verifies that loading the M3 checkpoint with pre-shifted +1.0 RMSNorm
weights through a plain `x_norm * w` RMSNorm produces identical output
to the HF reference's `x_norm * (1 + w_orig)`.
"""
import torch
import sys

sys.path.insert(0, "/tmp")  # so we can drop a minimal HF impl


def hf_rmsnorm(x, w_orig, eps=1e-6):
    """HF MiniMax-M3 Gemma-style RMSNorm: scale = (1 + w)."""
    in_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * (1.0 + w_orig.float())
    return x.to(in_dtype)


def neuron_rmsnorm(x, w_shifted, eps=1e-6):
    """NxDI plain RMSNorm `x_norm * w` — but we pass pre-shifted weights."""
    in_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * w_shifted.float()  # w_shifted = 1 + w_orig
    return x.to(in_dtype)


torch.manual_seed(0)

# Load an actual M3 norm weight
from safetensors.torch import load_file
sd = load_file("/mnt/nvme/models/MiniMax-M3/model-00001-of-00059.safetensors")

cases = [
    ("input_layernorm (hidden_size=6144)", sd["language_model.model.layers.0.input_layernorm.weight"]),
    ("post_attention_layernorm (hidden_size=6144)", sd["language_model.model.layers.0.post_attention_layernorm.weight"]),
    ("q_norm (head_dim=128)", sd["language_model.model.layers.0.self_attn.q_norm.weight"]),
    ("k_norm (head_dim=128)", sd["language_model.model.layers.0.self_attn.k_norm.weight"]),
]

for name, w_orig in cases:
    dim = w_orig.shape[0]
    x = torch.randn(2, 16, dim, dtype=torch.bfloat16)

    # Reference: HF (1+w) scaling
    y_hf = hf_rmsnorm(x, w_orig)

    # Neuron path: plain w * pre-shifted weight
    w_shifted = (w_orig.float() + 1.0).to(torch.bfloat16)
    y_neuron = neuron_rmsnorm(x, w_shifted)

    max_diff = (y_hf.float() - y_neuron.float()).abs().max().item()
    rel = max_diff / max(y_hf.float().abs().max().item(), 1e-8)
    print(f"{name}:")
    print(f"  w_orig stats: mean={w_orig.float().mean().item():.4f} min={w_orig.float().min().item():.4f} max={w_orig.float().max().item():.4f}")
    print(f"  w_shifted (1+w) stats: mean={w_shifted.float().mean().item():.4f}")
    print(f"  max-abs-diff (HF vs Neuron-with-pre-shift): {max_diff:.2e}, rel={rel:.2e}")
    print()

# Now show what was wrong BEFORE the fix:
print("=" * 60)
print("Sanity check: what would happen WITHOUT pre-shift?")
print("=" * 60)
w_orig = sd["language_model.model.layers.0.input_layernorm.weight"]
x = torch.randn(2, 16, w_orig.shape[0], dtype=torch.bfloat16)
y_hf = hf_rmsnorm(x, w_orig)
y_buggy = neuron_rmsnorm(x, w_orig)  # would have used plain w (BUGGY)
diff = (y_hf.float() - y_buggy.float()).abs().max().item()
print(f"HF (correct, (1+w)): mean={y_hf.float().mean().item():+.4e} std={y_hf.float().std().item():.4e}")
print(f"Buggy (plain w): mean={y_buggy.float().mean().item():+.4e} std={y_buggy.float().std().item():.4e}")
print(f"max-abs-diff if NOT pre-shifted: {diff:.2e}")
print(f"  (with input_layernorm.weight ≈ -0.94, plain w gives output ≈ -0.94x baseline;")
print(f"   correct (1+w) gives ≈ +0.05x baseline — orders of magnitude difference)")
