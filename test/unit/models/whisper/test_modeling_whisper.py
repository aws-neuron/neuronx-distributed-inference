import torch

from neuronx_distributed_inference.models.whisper.modeling_whisper import (
    NeuronAttention,
    NeuronCrossAttention,
    NeuronResidualAttentionBlock,
)
from neuronx_distributed_inference.models.whisper.utils.state_dict import expand_state_dict
from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from neuronx_distributed_inference.utils.random import set_random_seed
from whisper import ModelDimensions
from whisper.model import ResidualAttentionBlock, MultiHeadAttention

TP = 2
batch_size = 1
dtype = torch.float32
set_random_seed(0)

# whisper-large-v3-turbo
dims = ModelDimensions(
    n_mels=128,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    n_vocab=51866,
    n_text_ctx=448,
    n_text_state=1280,
    n_text_head=20,
    n_text_layer=4,
)


def rename_mlp_layers(state_dict: dict, prefix=True) -> dict:
    """
    Replace all keys in a state_dict:
    - "mlp.0" → "mlp.up_proj"
    - "mlp.2" → "mlp.down_proj"

    Args:
        state_dict (dict): Original model state dict.
        prefix (bool): If True, replace "mlp." with "mlp." prefix; otherwise, replace "0" and "2" directly.

    Returns:
        dict: New state dict with renamed keys.
    """
    if prefix:
        replacements = {
            "mlp.0": "mlp.up_proj",
            "mlp.2": "mlp.down_proj",
        }
    else:
        replacements = {
            "0": "up_proj",
            "2": "down_proj",
        }

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = value

    del state_dict
    return new_state_dict


def test_self_attention():
    set_random_seed(0)
    input = torch.randn(batch_size, dims.n_audio_ctx, dims.n_audio_state, dtype=dtype)
    print(f"Input shape: {input.shape}")

    # Run inference on CPU model
    model_cpu = MultiHeadAttention(dims.n_audio_state, dims.n_audio_head)
    output_cpu = model_cpu(input)[0]
    print(f"CPU Output: {output_cpu}")
    print(f"CPU Output shape: {output_cpu.shape}")

    state_dict = model_cpu.state_dict()
    state_dict = expand_state_dict(state_dict, dims, TP)

    # Run inference on Neuron model
    class Instance(BaseModelInstance):
        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = NeuronAttention(
                dims.n_audio_state, dims.n_audio_head, batch_size, dims.n_audio_ctx, dtype=dtype
            )

        def get(self, bucket_rank, **kwargs):
            aliases = {
                self.module.cache_k: 1,
                self.module.cache_v: 2,
            }
            return self.module, aliases

    mb = ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=lambda: state_dict)
    mb.add(
        "MultiHeadAttention",
        Instance(),
        example_inputs=[(torch.zeros_like(input),)],
        compiler_args="--auto-cast=none",
    )
    model_nxd = mb.trace(initialize_model_weights=True)
    output_nxd = model_nxd(input)
    print(f"NxD Output: {output_nxd}")
    print(f"NxD Output shape: {output_nxd.shape}")

    # Compare outputs
    passed, max_err = check_accuracy_embeddings(
        output_nxd,
        output_cpu,
    )
    print(f"Passed: {passed}, Max Error: {max_err}")
    assert passed, f"Output validation failed with max error: {max_err}"


def test_cross_attention():
    set_random_seed(0)
    input = torch.randn(batch_size, dims.n_text_ctx, dims.n_text_state, dtype=dtype)
    audio_embed = torch.randn(batch_size, dims.n_audio_ctx, dims.n_audio_state, dtype=dtype)
    print(f"Input shape: {input.shape}")
    print(f"Audio embed shape: {audio_embed.shape}")

    # Run inference on CPU model
    model_cpu = MultiHeadAttention(dims.n_audio_state, dims.n_audio_head)
    output_cpu = model_cpu(input, audio_embed)[0]
    print(f"CPU Output: {output_cpu}")
    print(f"CPU Output shape: {output_cpu.shape}")

    state_dict = model_cpu.state_dict()
    state_dict = expand_state_dict(state_dict, dims, TP)

    # Run inference on Neuron model
    class Instance(BaseModelInstance):
        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = NeuronCrossAttention(
                dims.n_audio_state, dims.n_audio_head, batch_size, dims.n_audio_ctx, dtype=dtype
            )

        def get(self, bucket_rank, **kwargs):
            aliases = {
                self.module.cache_k: 1,
                self.module.cache_v: 2,
            }
            return self.module, aliases

    mb = ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=lambda: state_dict)
    mb.add(
        "CrossAttentionPrefill",
        Instance(),
        example_inputs=[(torch.zeros_like(input), torch.zeros_like(audio_embed))],
        compiler_args="--auto-cast=none",
    )
    model_nxd = mb.trace(initialize_model_weights=True)
    output_nxd = model_nxd(input, audio_embed)
    print(f"NxD Output: {output_nxd}")
    print(f"NxD Output shape: {output_nxd.shape}")

    # Compare outputs
    passed, max_err = check_accuracy_embeddings(
        output_nxd,
        output_cpu,
    )
    print(f"Passed: {passed}, Max Error: {max_err}")
    assert passed, f"Output validation failed with max error: {max_err}"


def test_residual_attention():
    set_random_seed(0)
    input = torch.randn(batch_size, dims.n_audio_ctx, dims.n_audio_state, dtype=dtype)
    print(f"Input shape: {input.shape}")

    # Run inference on CPU model
    model_cpu = ResidualAttentionBlock(dims.n_audio_state, dims.n_audio_head)
    output_cpu = model_cpu(input)
    print(f"CPU Output: {output_cpu}")
    print(f"CPU Output shape: {output_cpu.shape}")

    state_dict = rename_mlp_layers(model_cpu.state_dict())
    state_dict = expand_state_dict(state_dict, dims, TP)

    class Instance(BaseModelInstance):
        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = NeuronResidualAttentionBlock(
                dims.n_audio_state, dims.n_audio_head, batch_size, dims.n_audio_ctx, dtype=dtype
            )

        def get(self, bucket_rank, **kwargs):
            aliases = {}
            return self.module, aliases

    # Run inference on Neuron model
    mb = ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=lambda: state_dict)
    mb.add(
        "ResidualAttentionBlock",
        Instance(),
        example_inputs=[(torch.zeros_like(input),)],
        compiler_args="--auto-cast=none",
    )
    model_nxd = mb.trace(initialize_model_weights=True)
    output_nxd = model_nxd(input)
    print(f"NxD Output: {output_nxd}")
    print(f"NxD Output shape: {output_nxd.shape}")

    # Compare outputs
    passed, max_err = check_accuracy_embeddings(
        output_nxd,
        output_cpu,
    )
    print(f"Passed: {passed}, Max Error: {max_err}")
    assert passed, f"Output validation failed with max error: {max_err}"


if __name__ == "__main__":
    test_self_attention()
    test_cross_attention()
    test_residual_attention()
