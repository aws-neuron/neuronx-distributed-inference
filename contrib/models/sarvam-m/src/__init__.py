# sarvam-m contrib model
# Uses NeuronMixtralForCausalLM (Mistral code path) with patches for head_dim support
from .setup_patches import (
    apply_patches,
    patch_mixtral_head_dim,
    patch_nkilib_eps,
    patch_neuronxcc_eps,
)
