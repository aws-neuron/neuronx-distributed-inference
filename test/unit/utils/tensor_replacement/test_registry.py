import os
import re
import tempfile
from typing import Dict, List, Tuple
from types import SimpleNamespace

import pytest
import torch

from neuronx_distributed_inference.utils.tensor_replacement.registry import (
    TensorReplacementRegister,
    _overlap_slices,
    _ensure_rank_and_pad_ref_to_target,
    _apply_ref_equiv,
    _FN,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_register():
    TensorReplacementRegister.clear()
    yield
    TensorReplacementRegister.clear()


@pytest.fixture
def fake_neuron_config():
    """
    Minimal object with attrs that the registry/_configure might touch.
    Expand if your code reads more fields.
    """
    cfg = SimpleNamespace()
    # common fields we’ve seen used in this code path
    cfg.batch_size = 1
    cfg.ctx_batch_size = 1
    cfg.tkg_batch_size = 16
    cfg.output_logits = False
    cfg.on_device_sampling_config = None

    # replacement/capture configs default to None (tests don’t rely on them)
    cfg.tensor_capture_config = None
    cfg.tensor_replacement_config = None

    # misc toggles often checked in MoE/SP paths (safe defaults)
    cfg.sequence_parallel_enabled = False
    cfg.strided_context_parallel_kernel_enabled = False

    return cfg


def _save_pt(dirpath: str, phase: str, step: int, module: str,
             tensor: torch.Tensor, suffix: str = "output"):
    """
    Write a tensor with the filename pattern the register expects.

    IMPORTANT: Your source regex requires a trailing .(output|outputs).pt (or _output/_outputs).
    We therefore default to suffix="output". Pass suffix="outputs" to create a colliding duplicate.
    """
    os.makedirs(dirpath, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    fn = f"captured_tensors_{phase}_step_{step}_module_{module}{suffix_part}.pt"
    torch.save(tensor, os.path.join(dirpath, fn))


def _mk_dirs():
    cpu_td = tempfile.TemporaryDirectory()
    neu_td = tempfile.TemporaryDirectory()
    return cpu_td, neu_td


@pytest.fixture(autouse=True)
def _reset_singleton_after_test():
    """
    Ensure each test starts with a fresh register.
    """
    yield
    TensorReplacementRegister.clear()


# ----------------------------------------------------------------------
# Unit tests: small utilities
# ----------------------------------------------------------------------
def test_exact_match_shapes():
    src = (4, 8)
    dst = (4, 8)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 4), slice(0, 8))


def test_src_larger_than_dst():
    src = (10, 20)
    dst = (3, 5)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 3), slice(0, 5))


def test_dst_larger_than_src():
    src = (3, 5)
    dst = (10, 20)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 3), slice(0, 5))


def test_1d_shapes():
    src = (7,)
    dst = (4,)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 4),)


def test_3d_shapes_mixed():
    src = (5, 10, 2)
    dst = (3, 20, 4)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 3), slice(0, 10), slice(0, 2))


def test_zero_dimension():
    src = (0, 5)
    dst = (4, 3)
    result = _overlap_slices(src, dst)
    assert result == (slice(0, 0), slice(0, 3))


def test_empty_shapes():
    src = ()
    dst = ()
    result = _overlap_slices(src, dst)
    assert result == ()

def test_tensor_slice_behavior():
    src = torch.randn(5, 10)
    dst = torch.zeros(3, 8)
    slices = _overlap_slices(src.shape, dst.shape)

    # apply slices
    dst[slices] = src[slices]

    assert torch.allclose(dst[:3, :8], src[:3, :8])


def test__ensure_rank_and_pad_ref_to_target_unsqueeze_then_pad():
    # CPU missing leading batch; target has batch=1
    cpu = torch.zeros(44, 128)         # (S, H)
    target = torch.ones([1, 512, 128]) * 5.0 # (B, T, H)
    out = _ensure_rank_and_pad_ref_to_target(cpu, target)
    assert out.shape == (1, 512, 128)
    # original content sits at [:, :44, :], rest padded with zeros
    assert torch.all(out[0, :44, :] == cpu)


def test__apply_ref_equiv_basic_and_suffix_preserved():
    equiv = {"mlp.gate": "mlp.router.linear_router"}
    assert _apply_ref_equiv("layers.0.mlp.gate", equiv) == "layers.0.mlp.router.linear_router"
    assert _apply_ref_equiv("layers.0.mlp.gate_output", equiv) == "layers.0.mlp.router.linear_router_output"
    # no change if key absent
    assert _apply_ref_equiv("layers.0.attn", equiv) == "layers.0.attn"


def test__FN_regex_examples():
    ok = [
        "captured_tensors_cte_step_1_module_layers.0.mlp.router.linear_router_output.pt",
        "captured_tensors_tkg_step_2_module_layers.1.mlp.router.linear_router_outputs.pt",
        "captured_tensors_ctx_step_3_module_layers.2.mlp.gate_output.pt",  # ctx allowed
        "captured_tensors_cte_step_4_module_layers.9.attn.qkv.outputs.pt",
    ]
    for fn in ok:
        assert _FN.match(fn), f"Should match: {fn}"

    bad = [
        "captured_tensors_cte_step_1_module_layers.0.mlp.router.linear_router.pt",  # missing output(s) suffix
        "captured_tensors_tkg_step_2_module_layers.1.mlp.router.linear_router_inputs.pt",  # 'inputs' not allowed
        "captured_tensors_cte_step_X_module_layers.0.mlp.router.linear_router_output.pt",  # step must be int
        "random_name.pt",
    ]
    for fn in bad:
        assert not _FN.match(fn), f"Should NOT match: {fn}"


# ----------------------------------------------------------------------
# Main register behavior
# ----------------------------------------------------------------------

def test_register_happy_path_builds_shapes_dtypes_packs_and_masks(fake_neuron_config):
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    # We want to test two modules across three steps (step1=ctx/cte-like, 2-3=tkg-like).
    # Neuron is canonical; CPU uses 'mlp.gate' and must be mapped to router.linear_router.
    modules_neu = ["layers.0.mlp.router.linear_router", "layers.1.mlp.router.linear_router"]
    modules_cpu = ["layers.0.mlp.gate",               "layers.1.mlp.gate"]

    tr_map = {1: modules_neu, 2: modules_neu, 3: modules_neu}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Neuron shapes
    shp_ctx = torch.Size([1, 512, 128])
    shp_tkg = torch.Size([1, 1, 128])

    # Create Neuron files for each module & step (note: suffix required by regex)
    for m in modules_neu:
        _save_pt(neu_dir, "cte", 1, m, torch.ones(shp_ctx))  # step 1
        _save_pt(neu_dir, "tkg", 2, m, torch.ones(shp_tkg))  # step 2
        _save_pt(neu_dir, "tkg", 3, m, torch.ones(shp_tkg))  # step 3

    # Create CPU files: step1 missing batch (S,H), steps 2-3 have (1,H)
    cpu_ctx = torch.zeros(44, 128)
    cpu_tkg = torch.zeros(1, 128)
    for m_cpu in modules_cpu:
        _save_pt(cpu_dir, "cte", 1, m_cpu, cpu_ctx.clone())
        _save_pt(cpu_dir, "tkg", 2, m_cpu, cpu_tkg.clone())
        _save_pt(cpu_dir, "tkg", 3, m_cpu, cpu_tkg.clone())

    reg = TensorReplacementRegister.get_instance(
        ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
    )

    # module_superset in insertion order (stable)
    assert reg.module_superset == modules_neu

    # step_shapes present and correct
    assert reg.step_shapes[1][modules_neu[0]] == shp_ctx
    assert reg.step_shapes[2][modules_neu[0]] == shp_tkg
    assert reg.step_shapes[3][modules_neu[1]] == shp_tkg

    # dtype inference from CPU captures
    for s in (1, 2, 3):
        for m in modules_neu:
            assert reg.ref_step_dtypes[s][m] == torch.float32

    # packed_step tensors: CPU aligned to Neuron shapes
    assert reg.packed_step[1][modules_neu[0]].shape == shp_ctx
    assert reg.packed_step[2][modules_neu[1]].shape == shp_tkg
    # check padding happened for ctx (44 -> 512)
    assert torch.all(reg.packed_step[1][modules_neu[0]][0, :44, :] == 0)

    # masks: ones for modules present in tr_map for that step
    for s in (1, 2, 3):
        for m in modules_neu:
            expected = 1 if (s in tr_map and m in tr_map[s]) else 0
            assert reg.masks_step[s][m].dtype == torch.bool
            assert reg.masks_step[s][m].numel() == 1
            assert int(reg.masks_step[s][m].item()) == expected

    # example_args returns zeros matching shapes + zero masks
    ex_tensors, ex_masks = reg.example_args(step=2)
    # expected example shape uses config batch, not captured tensor batch
    expected_ex_shape = torch.Size([fake_neuron_config.tkg_batch_size, 1, 128])
    assert len(ex_tensors) == len(modules_neu) == len(ex_masks)
    for t, m, mname in zip(ex_tensors, ex_masks, modules_neu):
        assert t.shape == expected_ex_shape
        assert torch.all(t == 0)
        assert m.dtype == torch.bool and m.numel() == 1 and not m.item()

    # step_args returns aligned CPU tensors and the masks we built
    st_tensors, st_masks = reg.step_args(step=1)
    assert st_tensors[0].shape == shp_ctx
    assert st_masks[0].dtype == torch.bool and st_masks[0].item()


def test_register_ignores_non_requested_modules_and_handles_empty_dirs(fake_neuron_config):
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    tr_map = {1: ["layers.0.mlp.router.linear_router"]}  # only one module requested
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Put extra files that aren't requested
    _save_pt(neu_dir, "cte", 1, "layers.999.other", torch.randn(1, 4, 8))
    _save_pt(cpu_dir, "cte", 1, "layers.999.mlp.gate", torch.randn(4, 8))

    # Also add the ones we actually want
    _save_pt(neu_dir, "cte", 1, "layers.0.mlp.router.linear_router", torch.randn(1, 4, 8))
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8))

    reg = TensorReplacementRegister.get_instance(
        ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
    )
    # Only the requested module should appear
    assert reg.module_superset == ["layers.0.mlp.router.linear_router"]
    assert set(reg.step_shapes[1].keys()) == {"layers.0.mlp.router.linear_router"}


def test_register_errors_when_neuron_shape_missing_for_requested_module():
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    tr_map = {1: ["layers.0.mlp.router.linear_router"]}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Provide CPU capture but no Neuron capture for requested module
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8))

    with pytest.raises(ValueError, match=r"Expected non-zero tkg_shapes"):
        _ = TensorReplacementRegister.get_instance(
            ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
        )


def test_register_errors_on_mixed_cpu_dtypes_across_steps():
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    tr_map = {1: ["layers.0.mlp.router.linear_router"],
              2: ["layers.0.mlp.router.linear_router"]}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Neuron shapes
    _save_pt(neu_dir, "cte", 1, "layers.0.mlp.router.linear_router", torch.randn(1, 4, 8))
    _save_pt(neu_dir, "tkg", 2, "layers.0.mlp.router.linear_router", torch.randn(1, 1, 8))

    # CPU: step1 float32, step2 float16 -> should error (mixed dtypes)
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8).to(torch.float32))
    _save_pt(cpu_dir, "tkg", 2, "layers.0.mlp.gate", torch.randn(1, 8).to(torch.float16))

    with pytest.raises(ValueError, match="Mixed dtypes across steps"):
        _ = TensorReplacementRegister.get_instance(
            ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
        )


def test_register_errors_on_rank_mismatch_across_steps():
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    tr_map = {1: ["layers.0.mlp.router.linear_router"],
              2: ["layers.0.mlp.router.linear_router"]}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Neuron shapes: step1 rank-3, step2 rank-2 (rank mismatch)
    _save_pt(neu_dir, "cte", 1, "layers.0.mlp.router.linear_router", torch.randn(1, 4, 8))
    _save_pt(neu_dir, "tkg", 2, "layers.0.mlp.router.linear_router", torch.randn(1, 8))  # rank 2

    # CPU to satisfy dtype inference
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8))
    _save_pt(cpu_dir, "tkg", 2, "layers.0.mlp.gate", torch.randn(1, 8))

    with pytest.raises(ValueError, match="Rank mismatch"):
        _ = TensorReplacementRegister.get_instance(
            ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
        )


def test_register_errors_on_shape_mismatch_across_tkg_steps():
    """
    The code checks that tkg steps (everything except possibly the first) have the same shape.
    We'll create step1(ctx-like), step2(tkg-like), step3(tkg-like but different shape) -> error.
    """
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    modules = ["layers.0.mlp.router.linear_router"]
    tr_map = {1: modules, 2: modules, 3: modules}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Neuron shapes: step2 and step3 differ in T
    _save_pt(neu_dir, "cte", 1, modules[0], torch.randn(1, 512, 128))
    _save_pt(neu_dir, "tkg", 2, modules[0], torch.randn(1, 1, 128))
    _save_pt(neu_dir, "tkg", 3, modules[0], torch.randn(1, 2, 128))  # mismatch

    # CPU for dtype inference
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(44, 128))
    _save_pt(cpu_dir, "tkg", 2, "layers.0.mlp.gate", torch.randn(1, 128))
    _save_pt(cpu_dir, "tkg", 3, "layers.0.mlp.gate", torch.randn(1, 128))

    with pytest.raises(ValueError, match="Shape mismatch across tkg steps"):
        _ = TensorReplacementRegister.get_instance(
            ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
        )


def test_register_duplicate_files_for_same_module_step_raises():
    """
    Two files that normalize to the same (module, step) key should raise.
    We use an '_output' vs '_outputs' suffix to collide (regex strips suffix).
    """
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    modules = ["layers.0.mlp.router.linear_router"]
    tr_map = {1: modules}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    # Neuron duplicate for same key:
    _save_pt(neu_dir, "cte", 1, modules[0], torch.randn(1, 4, 8), suffix="output")
    _save_pt(neu_dir, "cte", 1, modules[0], torch.randn(1, 4, 8), suffix="outputs")  # same key after regex

    # CPU just to avoid earlier failures
    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8))

    with pytest.raises(ValueError, match="already been recorded"):
        _ = TensorReplacementRegister.get_instance(
            ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
        )


def test_example_and_step_args_lengths_and_order(fake_neuron_config):
    """
    Ensure example_args/step_args follow module_superset order and lengths.
    """
    cpu_td, neu_td = _mk_dirs()
    cpu_dir, neu_dir = cpu_td.name, neu_td.name

    mods = ["layers.0.mlp.router.linear_router", "layers.1.mlp.router.linear_router"]
    tr_map = {1: mods, 2: mods}
    cpu_equiv = {"mlp.gate": "mlp.router.linear_router"}

    _save_pt(neu_dir, "cte", 1, mods[0], torch.randn(1, 4, 8))
    _save_pt(neu_dir, "cte", 1, mods[1], torch.randn(1, 4, 8))
    _save_pt(neu_dir, "tkg", 2, mods[0], torch.randn(1, 1, 8))
    _save_pt(neu_dir, "tkg", 2, mods[1], torch.randn(1, 1, 8))

    _save_pt(cpu_dir, "cte", 1, "layers.0.mlp.gate", torch.randn(4, 8))
    _save_pt(cpu_dir, "cte", 1, "layers.1.mlp.gate", torch.randn(4, 8))
    _save_pt(cpu_dir, "tkg", 2, "layers.0.mlp.gate", torch.randn(1, 8))
    _save_pt(cpu_dir, "tkg", 2, "layers.1.mlp.gate", torch.randn(1, 8))

    reg = TensorReplacementRegister.get_instance(
        ref_dir=cpu_dir, neuron_dir=neu_dir, tr_map=tr_map, config=fake_neuron_config, ref_equiv_map=cpu_equiv
    )

    ex_t, ex_m = reg.example_args(2)
    st_t, st_m = reg.step_args(2)

    assert len(ex_t) == len(ex_m) == len(st_t) == len(st_m) == 2
    # Order matches module_superset
    assert reg.module_superset == mods
    # Step args should match packed tensors for that step
    assert torch.equal(st_t[0], reg.packed_step[2][mods[0]])
    assert torch.equal(st_t[1], reg.packed_step[2][mods[1]])
