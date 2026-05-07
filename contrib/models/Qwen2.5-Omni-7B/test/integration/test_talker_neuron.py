#!/usr/bin/env python3
"""Tests for the Neuron-compiled Talker and Token2Wav implementations.

Tests CPU-level logic: config, state dict conversion, fused embedding,
class structure. Does NOT require Neuron hardware.

Mocks neuronx_distributed and torch_neuronx at sys.modules level so tests
can run on any machine (Mac, Linux, etc.) without Neuron SDK.
"""

# --- Qwen2.5-Omni contrib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401  (applies hf_adapter shim)
# --- end bootstrap ---

import sys
import types
import torch
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Mock setup: neuronx_distributed and torch_neuronx
# ============================================================================

class _MockModuleType(types.ModuleType):
    """Module type that returns MagicMock for any missing attribute."""
    def __getattr__(self, name):
        return MagicMock(name=f"{self.__name__}.{name}")


class _AutoMockFinder:
    """Meta path finder that auto-mocks any package that can't be imported normally.

    Only intercepts packages listed in _MOCK_PREFIXES to avoid breaking stdlib.
    """
    _MOCK_PREFIXES = (
        "neuronx_distributed", "torch_neuronx", "torch_xla",
        "nki", "nkilib", "neuronxcc", "transformers", "huggingface_hub",
        "safetensors", "accelerate", "sentencepiece", "tokenizers",
    )

    def find_module(self, fullname, path=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._MOCK_PREFIXES):
            return self
        return None

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._MOCK_PREFIXES):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _MockModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__spec__ = spec
        return mod

    def exec_module(self, module):
        pass

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _MockModuleType(fullname)
        mod.__path__ = []
        mod.__package__ = fullname
        mod.__loader__ = self
        sys.modules[fullname] = mod
        return mod


def _setup_neuron_mocks():
    """Install auto-mock import hook and set up key attributes.

    Must be called before importing any neuronx_distributed_inference modules.
    """
    # Install the auto-mock finder
    sys.meta_path.insert(0, _AutoMockFinder())

    # Force-import mocked modules so they exist in sys.modules
    import neuronx_distributed
    import neuronx_distributed.utils

    # Set specific attributes that need real values
    neuronx_distributed.utils.cpu_mode = MagicMock(return_value=True)

    import neuronx_distributed.utils.utils
    mock_hardware = MagicMock(name="hardware")
    mock_hardware.TRN1 = "trn1"
    mock_hardware.return_value = "trn2"
    neuronx_distributed.utils.utils.hardware = mock_hardware

    import torch_neuronx.utils
    torch_neuronx.utils.get_platform_target = MagicMock(return_value="trn2")

    # ColumnParallelLinear / RowParallelLinear / ParallelEmbedding need to be
    # real classes so NxDI code can subclass them (e.g. lora_layer.py)
    # ColumnParallelLinear / RowParallelLinear / ParallelEmbedding need to be
    # real classes so NxDI code can subclass them (e.g. lora_layer.py)
    class _MockParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _MockParallelEmbedding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _MockSPMDRank(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    # Set on both .parallel_layers and .parallel_layers.layers
    # (code imports from both paths)
    import neuronx_distributed.parallel_layers as _pl
    import neuronx_distributed.parallel_layers.layers as _pl_layers
    for mod in (_pl, _pl_layers):
        mod.ColumnParallelLinear = _MockParallelLinear
        mod.RowParallelLinear = _MockParallelLinear
        mod.ParallelEmbedding = _MockParallelEmbedding
        mod.SPMDRank = _MockSPMDRank

    # LlamaRMSNorm needs to be a real nn.Module subclass for RMSNorm usage
    class MockLlamaRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

    import transformers.models.llama.modeling_llama
    transformers.models.llama.modeling_llama.LlamaRMSNorm = MockLlamaRMSNorm

    # ACT2FN for MLP activation lookup
    import transformers.activations
    transformers.activations.ACT2FN = {
        "silu": torch.nn.SiLU(),
        "gelu": torch.nn.GELU(),
        "relu": torch.nn.ReLU(),
    }


# Install mocks before any NxDI imports
_MOCK_MODULES = _setup_neuron_mocks()


# ============================================================================
# Helper: create a load_config callable for InferenceConfig
# ============================================================================

def _make_load_config(**attrs):
    """Create a load_config callable that sets attributes on an InferenceConfig."""
    def load_config(self):
        for key, value in attrs.items():
            setattr(self, key, value)
    return load_config


# ============================================================================
# Test 1: Config classes
# ============================================================================

def test_talker_inference_config():
    """Test TalkerInferenceConfig creation and derived attributes."""
    from modeling_qwen25_omni_talker import (
        TalkerInferenceConfig,
        TalkerNeuronConfig,
    )

    neuron_config = TalkerNeuronConfig(
        tp_degree=4,
        batch_size=1,
        seq_len=512,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
    )

    config = TalkerInferenceConfig(
        neuron_config=neuron_config,
        load_config=_make_load_config(
            hidden_size=896,
            num_attention_heads=12,
            num_hidden_layers=24,
            num_key_value_heads=4,
            vocab_size=8448,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            intermediate_size=18944,
            pad_token_id=0,
            max_position_embeddings=32768,
        ),
    )

    # Check derived config
    assert config.head_dim == 128, f"Expected head_dim=128, got {config.head_dim}"
    assert config.qkv_bias == True
    assert config.o_bias == False
    assert config.num_cores_per_group == 1
    assert config.rope_scaling is not None
    assert config.rope_scaling["mrope_section"] == [16, 24, 24]
    assert config.thinker_hidden_size == 3584

    # Check neuron config cls
    assert TalkerInferenceConfig.get_neuron_config_cls() == TalkerNeuronConfig

    # Check required attributes
    required = config.get_required_attributes()
    assert "hidden_size" in required
    assert "num_attention_heads" in required
    assert "head_dim" not in required  # head_dim is derived, not required from HF

    print("PASS: TalkerInferenceConfig")


# ============================================================================
# Test 2: Fused embedding in state dict conversion
# ============================================================================

def test_fused_embedding_conversion():
    """Test that embed_tokens + thinker_to_talker_proj are correctly fused."""
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalkerForCausalLM,
        TalkerInferenceConfig,
        TalkerNeuronConfig,
    )

    # Create fake state dict with talker weights
    vocab_size, embed_dim, hidden_size = 8448, 3584, 896
    embed_weight = torch.randn(vocab_size, embed_dim)
    proj_weight = torch.randn(hidden_size, embed_dim)
    proj_bias = torch.randn(hidden_size)

    state_dict = {
        "talker.model.embed_tokens.weight": embed_weight,
        "talker.thinker_to_talker_proj.weight": proj_weight,
        "talker.thinker_to_talker_proj.bias": proj_bias,
        "talker.codec_head.weight": torch.randn(vocab_size, hidden_size),
        "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(12 * 128, hidden_size),
        "talker.model.layers.0.self_attn.k_proj.weight": torch.randn(4 * 128, hidden_size),
        "talker.model.layers.0.self_attn.v_proj.weight": torch.randn(4 * 128, hidden_size),
        "talker.model.layers.0.self_attn.q_proj.bias": torch.randn(12 * 128),
        "talker.model.layers.0.self_attn.k_proj.bias": torch.randn(4 * 128),
        "talker.model.layers.0.self_attn.v_proj.bias": torch.randn(4 * 128),
        "talker.model.norm.weight": torch.randn(hidden_size),
    }

    # Create config
    neuron_config = TalkerNeuronConfig(
        tp_degree=4,
        batch_size=1,
        seq_len=512,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
        fused_qkv=True,
    )

    config = TalkerInferenceConfig(
        neuron_config=neuron_config,
        load_config=_make_load_config(
            hidden_size=896,
            num_attention_heads=12,
            num_hidden_layers=1,  # Just 1 layer for testing
            num_key_value_heads=4,
            vocab_size=8448,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            intermediate_size=18944,
            pad_token_id=0,
            max_position_embeddings=32768,
        ),
    )

    # Convert
    converted = NeuronQwen25OmniTalkerForCausalLM.convert_hf_to_neuron_state_dict(
        state_dict, config
    )

    # Check fused embedding
    assert "embed_tokens.weight" in converted
    fused_embed = converted["embed_tokens.weight"]
    assert fused_embed.shape == (vocab_size, hidden_size), \
        f"Expected ({vocab_size}, {hidden_size}), got {fused_embed.shape}"

    # Verify fused embedding is correct: embed @ proj.T + bias
    expected = embed_weight.float() @ proj_weight.float().T + proj_bias.float().unsqueeze(0)
    expected = expected.to(torch.bfloat16)
    assert torch.allclose(fused_embed, expected, atol=1e-2), \
        "Fused embedding values don't match expected computation"

    # Check codec_head → lm_head mapping
    assert "lm_head.weight" in converted
    assert "codec_head.weight" not in converted

    # Check prefix stripping
    assert "layers.0.self_attn.q_proj.weight" not in converted  # Should be fused
    assert "layers.0.self_attn.Wqkv.weight" in converted  # Fused QKV

    # Check fused QKV shape: q(1536) + k(512) + v(512) = 2560
    qkv = converted["layers.0.self_attn.Wqkv.weight"]
    assert qkv.shape[0] == 12 * 128 + 4 * 128 + 4 * 128, \
        f"Fused QKV wrong shape: {qkv.shape}"

    # Check projection weights saved for CPU context encoding
    assert "_thinker_proj_weight" in converted
    assert "_thinker_proj_bias" in converted
    assert converted["_thinker_proj_weight"].shape == (hidden_size, embed_dim)

    # Check rank utilities
    assert "rank_util.rank" in converted
    assert "layers.0.self_attn.rank_util.rank" in converted

    print("PASS: Fused embedding conversion")


# ============================================================================
# Test 3: ThinkerToTalkerProjection
# ============================================================================

def test_thinker_to_talker_projection():
    """Test CPU-side thinker state projection."""
    from modeling_qwen25_omni_talker import (
        ThinkerToTalkerProjection,
    )

    thinker_dim, talker_dim = 3584, 896

    # Create from state dict
    proj_weight = torch.randn(talker_dim, thinker_dim)
    proj_bias = torch.randn(talker_dim)
    state_dict = {
        "_thinker_proj_weight": proj_weight,
        "_thinker_proj_bias": proj_bias,
    }

    proj = ThinkerToTalkerProjection.from_state_dict(state_dict, dtype=torch.float32)

    # Test forward
    batch, seq = 2, 10
    thinker_states = torch.randn(batch, seq, thinker_dim)
    output = proj(thinker_states)

    assert output.shape == (batch, seq, talker_dim), \
        f"Expected ({batch}, {seq}, {talker_dim}), got {output.shape}"

    # Verify output matches manual computation (relax atol for large dim=3584)
    expected = thinker_states @ proj_weight.T + proj_bias
    assert torch.allclose(output, expected, atol=1e-3), \
        "Projection output doesn't match expected"

    print("PASS: ThinkerToTalkerProjection")


# ============================================================================
# Test 4: Talker RoPE
# ============================================================================

def test_talker_rotary_embedding():
    """Test TalkerRotaryEmbedding with both 1D and 3D positions."""
    from modeling_qwen25_omni_talker import (
        TalkerRotaryEmbedding,
    )

    class MockConfig:
        head_dim = 128
        rope_theta = 1000000.0

    emb = TalkerRotaryEmbedding(MockConfig())

    # Test with 2D position_ids (standard RoPE)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, 128)
    pos_2d = torch.arange(seq).unsqueeze(0).expand(batch, -1)  # (batch, seq)
    cos, sin = emb(x, pos_2d)
    assert cos.shape == (batch, seq, 128), f"2D RoPE cos shape: {cos.shape}"
    assert sin.shape == (batch, seq, 128), f"2D RoPE sin shape: {sin.shape}"

    # Test with 3D position_ids (mRoPE)
    pos_3d = torch.arange(seq).unsqueeze(0).unsqueeze(0).expand(3, batch, -1)  # (3, batch, seq)
    cos, sin = emb(x, pos_3d)
    assert cos.shape == (3, batch, seq, 128), f"3D mRoPE cos shape: {cos.shape}"
    assert sin.shape == (3, batch, seq, 128), f"3D mRoPE sin shape: {sin.shape}"

    print("PASS: TalkerRotaryEmbedding")


# ============================================================================
# Test 5: mRoPE application
# ============================================================================

def test_apply_multimodal_rotary_pos_emb():
    """Test mRoPE application function."""
    from modeling_qwen25_omni_talker import (
        _apply_multimodal_rotary_pos_emb,
    )

    batch, n_heads, seq, head_dim = 2, 12, 16, 128
    q = torch.randn(batch, n_heads, seq, head_dim)
    k = torch.randn(batch, n_heads, seq, head_dim)

    # cos/sin from mRoPE: (3, batch, seq, head_dim)
    cos = torch.randn(3, batch, seq, head_dim)
    sin = torch.randn(3, batch, seq, head_dim)
    mrope_section = [16, 24, 24]  # sum=64, *2=128=head_dim

    q_out, k_out = _apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

    assert q_out.shape == q.shape, f"Q shape mismatch: {q_out.shape} vs {q.shape}"
    assert k_out.shape == k.shape, f"K shape mismatch: {k_out.shape} vs {k.shape}"

    # Verify it's not identity (something changed)
    assert not torch.allclose(q_out, q), "Q unchanged after RoPE"
    assert not torch.allclose(k_out, k), "K unchanged after RoPE"

    print("PASS: apply_multimodal_rotary_pos_emb")


# ============================================================================
# Test 6: Token2Wav Neuron DiT class
# ============================================================================

def test_token2wav_neuron_dit_class():
    """Test NeuronQwen25OmniToken2WavWithNeuronDiT class structure."""
    from modeling_qwen25_omni_token2wav import (
        NeuronQwen25OmniToken2Wav,
        NeuronQwen25OmniToken2WavWithNeuronDiT,
    )

    # Check inheritance
    assert issubclass(NeuronQwen25OmniToken2WavWithNeuronDiT, NeuronQwen25OmniToken2Wav)

    # Check that Neuron DiT class has the expected methods
    assert hasattr(NeuronQwen25OmniToken2WavWithNeuronDiT, "compile_dit")
    assert hasattr(NeuronQwen25OmniToken2WavWithNeuronDiT, "load_dit")
    assert hasattr(NeuronQwen25OmniToken2WavWithNeuronDiT, "_get_dit_module")

    print("PASS: Token2Wav Neuron DiT class structure")


# ============================================================================
# Test 7: Orchestration methods
# ============================================================================

def test_orchestration_methods():
    """Test that orchestration class has the new methods."""
    import ast

    orchestration_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "modeling_qwen25_omni.py"
    )
    with open(orchestration_path) as f:
        tree = ast.parse(f.read())

    # Find NeuronQwen25OmniMultimodalForCausalLM
    target_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NeuronQwen25OmniMultimodalForCausalLM":
            target_cls = node
            break

    assert target_cls is not None, "NeuronQwen25OmniMultimodalForCausalLM not found"

    methods = {
        n.name for n in target_cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    # Check new methods exist
    expected_methods = {
        "enable_talker",
        "compile_talker",
        "load_talker",
        "enable_token2wav",
        "compile_token2wav_dit",
        "load_token2wav_dit",
        "_get_talker_cls",
        "_get_neuron_talker_cls",
        "_get_talker_config_cls",
        "_get_thinker_projection_cls",
        "_get_token2wav_cls",
        "_get_neuron_token2wav_cls",
    }

    missing = expected_methods - methods
    assert not missing, f"Missing methods in orchestration: {missing}"

    print("PASS: Orchestration methods")


# ============================================================================
# Test 8: Encode vision to input (thinker state injection)
# ============================================================================

def test_encode_vision_to_input():
    """Test NeuronTalkerModel.encode_vision_to_input for thinker state injection."""
    from modeling_qwen25_omni_talker import (
        NeuronTalkerModel,
    )

    batch, seq, hidden = 2, 16, 896

    # Placeholder embeddings from embed_tokens
    inputs_embeds = torch.zeros(batch, seq, hidden)
    # Projected thinker states
    vision_embeddings = torch.randn(batch, seq, hidden)
    # Full mask (all positions are thinker states)
    vision_mask = torch.ones(batch, seq, 1, dtype=torch.int32)

    # Call static-like method (doesn't need model instance)
    result = NeuronTalkerModel.encode_vision_to_input(None, inputs_embeds, vision_embeddings, vision_mask)

    assert result.shape == (batch, seq, hidden)
    assert torch.allclose(result, vision_embeddings), \
        "Full mask should replace all positions with thinker states"

    # Test partial mask
    vision_mask_partial = torch.zeros(batch, seq, 1, dtype=torch.int32)
    vision_mask_partial[:, :8, :] = 1  # First 8 positions are thinker states

    result_partial = NeuronTalkerModel.encode_vision_to_input(
        None, inputs_embeds, vision_embeddings, vision_mask_partial
    )
    assert torch.allclose(result_partial[:, :8, :], vision_embeddings[:, :8, :]), \
        "Masked positions should have thinker states"
    assert torch.allclose(result_partial[:, 8:, :], inputs_embeds[:, 8:, :]), \
        "Unmasked positions should keep original embeddings"

    print("PASS: encode_vision_to_input")


# ============================================================================
# Test 9: Class imports resolve correctly
# ============================================================================

def test_imports():
    """Test that all new classes can be imported."""
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalker,
        TalkerNeuronConfig,
        TalkerInferenceConfig,
        TalkerRotaryEmbedding,
        NeuronTalkerAttention,
        NeuronTalkerDecoderLayer,
        NeuronTalkerModel,
        NeuronQwen25OmniTalkerForCausalLM,
        ThinkerToTalkerProjection,
    )

    from modeling_qwen25_omni_token2wav import (
        NeuronQwen25OmniToken2Wav,
        NeuronQwen25OmniToken2WavWithNeuronDiT,
    )

    print("PASS: All imports successful")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    tests = [
        test_imports,
        test_talker_inference_config,
        test_fused_embedding_conversion,
        test_thinker_to_talker_projection,
        test_talker_rotary_embedding,
        test_apply_multimodal_rotary_pos_emb,
        test_token2wav_neuron_dit_class,
        test_orchestration_methods,
        test_encode_vision_to_input,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
