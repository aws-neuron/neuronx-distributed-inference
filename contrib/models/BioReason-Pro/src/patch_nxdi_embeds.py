#!/usr/bin/env python3
"""
Patch NxDI (neuronx-distributed-inference) 0.8.0 to support inputs_embeds passthrough.

V3b: Complete 8-patch set for multimodal embedding injection into NxDI-compiled models.
Enables HuggingFaceGenerationAdapter.generate(inputs_embeds=...) for models that
inject pre-computed embeddings into placeholder token positions before LLM generation.

Architecture: Uniform 19-arg signature for both CTE (context encoding) and TKG (token
generation) models. CTE uses torch.where for embedding injection. TKG passes empty(0)
which takes the plain embed_tokens path.

Patches (all 8):
  1. model_base.py: Skip set_none_if_empty for inputs_embeds
  2. model_base.py: torch.where for CTE embedding injection, embed_tokens for TKG
  3. model_base.py: 19-arg calls for both CTE and TKG in _get_model_outputs
  4. model_base.py: Thread inputs_embeds into llava_args in forward()
  5. model_wrapper.py: 19-arg tuples for CTE (3D zeros) and TKG (empty(0))
  6. model_wrapper.py: Pad inputs_embeds at pos 18 for CTE
  7. hf_adapter.py: Fix inputs_embeds drop on first generation step (DynamicCache)
  8. model_base.py: Create dummy input_ids when HF passes only inputs_embeds

Tested on: NxDI 0.8.0 (SDK 2.28), trn2.3xlarge, Qwen3-4B

Usage (as module):
    from patch_nxdi_embeds import apply_all_patches
    apply_all_patches()

Usage (CLI):
    python patch_nxdi_embeds.py           # Apply all 8 patches
    python patch_nxdi_embeds.py --restore  # Restore original files from backups
    python patch_nxdi_embeds.py --verify   # Verify patches without applying
"""

import os
import sys
import shutil
import logging

log = logging.getLogger(__name__)

# Auto-detect NxDI installation path
_NXDI_BASE = None


def _find_nxdi_base():
    """Find the NxDI installation directory."""
    global _NXDI_BASE
    if _NXDI_BASE is not None:
        return _NXDI_BASE

    # Try import-based detection first
    try:
        import neuronx_distributed_inference

        _NXDI_BASE = os.path.dirname(neuronx_distributed_inference.__file__)
        return _NXDI_BASE
    except ImportError:
        pass

    # Fallback: known venv path
    fallback = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/lib/python3.12/site-packages/neuronx_distributed_inference"
    if os.path.exists(fallback):
        _NXDI_BASE = fallback
        return _NXDI_BASE

    raise RuntimeError(
        "Cannot find neuronx-distributed-inference installation. "
        "Ensure it is installed or set the NXDI path manually."
    )


def _get_paths():
    base = _find_nxdi_base()
    return {
        "model_base": os.path.join(base, "models", "model_base.py"),
        "model_wrapper": os.path.join(base, "models", "model_wrapper.py"),
        "hf_adapter": os.path.join(base, "utils", "hf_adapter.py"),
    }


def _backup_file(filepath):
    backup = filepath + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(filepath, backup)
        log.info(f"  Backed up {filepath} -> {backup}")
    else:
        log.debug(f"  Backup already exists: {backup}")


def restore_from_backups():
    """Restore original files from backups."""
    paths = _get_paths()
    log.info("Restoring from backups...")
    for key, filepath in paths.items():
        bak = filepath + ".bak"
        if os.path.exists(bak):
            shutil.copy2(bak, filepath)
            log.info(f"  Restored {filepath}")
        else:
            log.info(f"  No backup found for {filepath}")


# ---------------------------------------------------------------------------
# Patches 1-2: model_base.py -- NeuronBaseModel (embedding handling)
# ---------------------------------------------------------------------------


def _patch_model_base(content):
    """Patch 1: Comment out set_none_if_empty for inputs_embeds.
    Patch 2: Replace embed_tokens block with torch.where for CTE/TKG.
    """
    # --- Patch 1 ---
    old_line = "        inputs_embeds = self.set_none_if_empty(inputs_embeds)"
    new_line = "        # PATCHED-V3b: Keep inputs_embeds as tensor; handled in get_model_output\n        # inputs_embeds = self.set_none_if_empty(inputs_embeds)"

    if old_line in content:
        content = content.replace(old_line, new_line)
        log.info("  Patch 1 applied: Commented out set_none_if_empty")
    elif "PATCHED" in content and "# inputs_embeds = self.set_none_if_empty" in content:
        log.info("  Patch 1 already applied")
    else:
        raise RuntimeError("Patch 1: Could not find set_none_if_empty target")

    # --- Patch 2 ---
    old_embed_block = """        if inputs_embeds is None:
            inputs_embeds = (
                self.embed_tokens(input_ids)
                if not is_lora_module(self.embed_tokens)
                else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
            )"""

    new_embed_block = """        # PATCHED-V3b: inputs_embeds injection via torch.where (CTE only)
        token_embeds = (
            self.embed_tokens(input_ids)
            if not is_lora_module(self.embed_tokens)
            else self.embed_tokens(input_ids, adapter_ids=adapter_ids)
        )
        if inputs_embeds is not None and inputs_embeds.dim() == 3:
            has_embeds = (inputs_embeds.sum() != 0).reshape(1, 1, 1)
            inputs_embeds = torch.where(has_embeds, inputs_embeds, token_embeds)
        else:
            inputs_embeds = token_embeds"""

    if old_embed_block in content:
        content = content.replace(old_embed_block, new_embed_block)
        log.info("  Patch 2 applied: torch.where for CTE, embed_tokens for TKG")
    elif "PATCHED-V3" in content:
        log.info("  Patch 2 already applied")
    else:
        raise RuntimeError("Patch 2: Could not find embed block")

    return content


# ---------------------------------------------------------------------------
# Patches 3a-3b: model_base.py -- _get_model_outputs (19-arg CTE/TKG calls)
# ---------------------------------------------------------------------------


def _patch_get_model_outputs(content):
    """Patch 3: Both CTE and TKG calls pass 19 args."""
    # --- CTE standard path ---
    old_prefill = """            else:
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *llava_args,
                )"""

    new_prefill = """            else:
                # PATCHED-V3b: CTE with 19 args (includes inputs_embeds at pos 18)
                _empties_cte = [torch.empty(0) for _ in range(11)]
                if len(llava_args) > 11 and llava_args[11].dim() == 3:
                    _ie_cte = llava_args[11]
                else:
                    _ie_cte = torch.zeros(
                        input_ids.shape[0], input_ids.shape[1],
                        self.config.hidden_size,
                        dtype=self.config.neuron_config.torch_dtype,
                    )
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *_empties_cte,
                    _ie_cte,
                )"""

    if old_prefill in content:
        content = content.replace(old_prefill, new_prefill, 1)
        log.info("  Patch 3a applied: CTE call with 19 args")
    elif "PATCHED-V3" in content and "_empties_cte" in content:
        log.info("  Patch 3a already applied")
    else:
        raise RuntimeError("Patch 3a: Could not find CTE prefill block")

    # --- TKG standard path ---
    old_tkg = """                outputs = self.token_generation_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *llava_args,
                )"""

    new_tkg = """                # PATCHED-V3b: TKG with 19 args (matches CTE for uniform signature)
                _empties_tkg = [torch.empty(0) for _ in range(11)]
                _ie_tkg = torch.empty(0)
                outputs = self.token_generation_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    prev_hidden,
                    adapter_ids,
                    *_empties_tkg,
                    _ie_tkg,
                )"""

    if old_tkg in content:
        content = content.replace(old_tkg, new_tkg, 1)
        log.info("  Patch 3b applied: TKG call with 19 args")
    elif "PATCHED-V3" in content and "_empties_tkg" in content:
        log.info("  Patch 3b already applied")
    else:
        raise RuntimeError("Patch 3b: Could not find TKG block")

    return content


# ---------------------------------------------------------------------------
# Patch 4: model_base.py -- forward() threading inputs_embeds into llava_args
# ---------------------------------------------------------------------------


def _patch_forward_threading(content):
    """Patch 4: Thread inputs_embeds into llava_args for CTE."""
    old_transition = """            computed_context_lens=computed_context_lens)

        if self.async_mode:"""

    new_transition = """            computed_context_lens=computed_context_lens)

        # PATCHED-V3b: Thread inputs_embeds into llava_args for CTE consumption.
        if inputs_embeds is not None and inputs_embeds.dim() == 3:
            _empties_fwd = [torch.empty(0) for _ in range(11)]
            llava_args = _empties_fwd + [inputs_embeds]

        if self.async_mode:"""

    if old_transition in content:
        content = content.replace(old_transition, new_transition, 1)
        log.info("  Patch 4 applied: forward() threads inputs_embeds into llava_args")
    elif "PATCHED-V3" in content and "Thread inputs_embeds" in content:
        log.info("  Patch 4 already applied")
    else:
        raise RuntimeError("Patch 4: Could not find transition block")

    return content


# ---------------------------------------------------------------------------
# Patch 8: model_base.py -- Create dummy input_ids when inputs_embeds provided
# ---------------------------------------------------------------------------


def _patch_dummy_input_ids(content):
    """Patch 8: Create dummy input_ids when HF passes only inputs_embeds.

    This handles two NxDI versions:
    - With *llava_args in the unpack (older NxDI)
    - Without *llava_args (NxDI 0.8.0+ / SDK 2.28)
    """
    if "PATCHED-V3b: Create dummy input_ids" in content:
        log.info("  Patch 8 already applied")
        return content

    dummy_block = """        # PATCHED-V3b: Create dummy input_ids when HF passes only inputs_embeds
        if input_ids is None and inputs_embeds is not None and inputs_embeds.dim() == 3:
            batch_size, seq_len, _ = inputs_embeds.shape
            pad_id = getattr(self.config, 'pad_token_id', 0) or 0
            input_ids = torch.full(
                (batch_size, seq_len), pad_id, dtype=torch.long
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_len), dtype=torch.long
                )
            if position_ids is None:
                position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

"""

    # Insert BEFORE the is_context_encoding line (which crashes if input_ids is None)
    target_ctx = "        is_context_encoding = input_ids.shape[-1] > 1"
    if target_ctx in content:
        content = content.replace(target_ctx, dummy_block + target_ctx, 1)
        log.info("  Patch 8 applied: dummy input_ids before is_context_encoding check")
        return content

    # Fallback: try inserting before preprocess_inputs (older layout)
    target1 = "        input_ids, attention_mask, position_ids, seq_ids, sampling_params, *llava_args = self.preprocess_inputs("
    if target1 in content:
        content = content.replace(target1, dummy_block + target1, 1)
        log.info("  Patch 8 applied (fallback variant 1: with *llava_args)")
        return content

    target2 = "        input_ids, attention_mask, position_ids, seq_ids, sampling_params = self.preprocess_inputs("
    if target2 in content:
        content = content.replace(target2, dummy_block + target2, 1)
        log.info("  Patch 8 applied (fallback variant 2: without *llava_args)")
        return content

    raise RuntimeError(
        "Patch 8: Could not find is_context_encoding or preprocess_inputs anchor"
    )

    return content


# ---------------------------------------------------------------------------
# Patches 5-6: model_wrapper.py
# ---------------------------------------------------------------------------


def _patch_model_wrapper_input_gen(content):
    """Patch 5: Both CTE and TKG get 19-arg tuples in input_generator."""
    old_block = """            else:
                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        hidden_states,
                        adapter_ids,
                    )
                )"""

    new_block = """            else:
                # PATCHED-V3b: Both CTE and TKG get 19 args for uniform jit.trace signature
                empties = [torch.empty(0) for _ in range(11)]
                if self.tag == CONTEXT_ENCODING_MODEL_TAG:
                    inputs_embeds_slot = torch.zeros(
                        (batch_size, n_active_tokens, self.config.hidden_size),
                        dtype=self.config.neuron_config.torch_dtype,
                    )
                else:
                    inputs_embeds_slot = torch.empty(0)
                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        sampling_params,
                        hidden_states,
                        adapter_ids,
                        *empties,
                        inputs_embeds_slot,
                    )
                )"""

    if old_block in content:
        content = content.replace(old_block, new_block)
        log.info("  Patch 5 applied: Both CTE and TKG get 19-arg tuples")
    elif "PATCHED-V3" in content:
        log.info("  Patch 5 already applied")
    else:
        raise RuntimeError("Patch 5: Could not find input_generator block")

    return content


def _patch_pad_inputs(content):
    """Patch 6: Pad inputs_embeds at position 18 for CTE."""
    old_reassemble = """            else:
                args = (*padded_args, *args[3:])"""

    new_reassemble = """            else:
                args = (*padded_args, *args[3:])
            # PATCHED-V3b: Pad inputs_embeds at position 18 for CTE
            if len(args) > 18 and args[18].dim() == 3:
                ie = args[18]
                ie_pad_len = pad_length - ie.shape[1]
                if ie_pad_len > 0:
                    ie = F.pad(ie, (0, 0, 0, ie_pad_len), "constant", 0.0)
                elif ie_pad_len < 0:
                    ie = ie[:, :pad_length, :]
                args = (*args[:18], ie, *args[19:])"""

    if old_reassemble in content:
        content = content.replace(old_reassemble, new_reassemble, 1)
        log.info("  Patch 6 applied: pad_inputs pads inputs_embeds for CTE")
    elif "PATCHED-V3" in content and "Pad inputs_embeds" in content:
        log.info("  Patch 6 already applied")
    else:
        raise RuntimeError("Patch 6: Could not find reassemble block")

    return content


# ---------------------------------------------------------------------------
# Patch 7: hf_adapter.py
# ---------------------------------------------------------------------------


def _patch_hf_adapter(content):
    """Patch 7: Fix inputs_embeds drop on first generation step.

    HF's _prepare_cache_for_generation() creates an empty DynamicCache even
    on the first step, so `past_key_values is None` is always False after
    the cache is set up. This causes inputs_embeds to be silently dropped.

    Replace `past_key_values is None` with `not self.prev_kv_cache_populated`
    which correctly identifies the first generation step.

    Handles both the compound check form:
        if inputs_embeds is not None and past_key_values is None:
    and the standalone form:
        if past_key_values is None:
    """
    if "PATCHED-V3b" in content and "prev_kv_cache_populated" in content:
        log.info("  Patch 7 already applied")
        return content

    # Try compound form first (NxDI 0.8.0+)
    old_compound = "if inputs_embeds is not None and past_key_values is None:"
    new_compound = "if inputs_embeds is not None and not self.prev_kv_cache_populated:  # PATCHED-V3b"
    count = content.count(old_compound)
    if count > 0:
        content = content.replace(old_compound, new_compound)
        log.info(f"  Patch 7 applied: Replaced {count} compound checks")
        return content

    # Try standalone form (older NxDI)
    old_standalone = "if past_key_values is None:"
    new_standalone = "if not self.prev_kv_cache_populated:  # PATCHED-V3b"
    count = content.count(old_standalone)
    if count > 0:
        content = content.replace(old_standalone, new_standalone)
        log.info(f"  Patch 7 applied: Replaced {count} standalone checks")
        return content

    raise RuntimeError("Patch 7: Could not find past_key_values is None check")

    return content


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_all_patches():
    """Apply all 8 V3b patches to enable inputs_embeds passthrough.

    This function is idempotent -- it restores from backups before applying
    patches, so it can be called multiple times safely.
    """
    paths = _get_paths()

    log.info("Applying NxDI V3b patches (8 patches for inputs_embeds passthrough)...")

    # Restore from backups for clean slate
    restore_from_backups()

    # --- model_base.py (Patches 1, 2, 3, 4, 8) ---
    _backup_file(paths["model_base"])
    with open(paths["model_base"], "r") as f:
        content = f.read()

    content = _patch_model_base(content)  # Patches 1-2
    content = _patch_get_model_outputs(content)  # Patches 3a-3b
    content = _patch_forward_threading(content)  # Patch 4
    content = _patch_dummy_input_ids(content)  # Patch 8

    with open(paths["model_base"], "w") as f:
        f.write(content)

    # --- model_wrapper.py (Patches 5-6) ---
    _backup_file(paths["model_wrapper"])
    with open(paths["model_wrapper"], "r") as f:
        content = f.read()

    content = _patch_model_wrapper_input_gen(content)  # Patch 5
    content = _patch_pad_inputs(content)  # Patch 6

    with open(paths["model_wrapper"], "w") as f:
        f.write(content)

    # --- hf_adapter.py (Patch 7) ---
    _backup_file(paths["hf_adapter"])
    with open(paths["hf_adapter"], "r") as f:
        content = f.read()

    content = _patch_hf_adapter(content)  # Patch 7

    with open(paths["hf_adapter"], "w") as f:
        f.write(content)

    log.info("All 8 V3b patches applied successfully.")


def verify_patches() -> bool:
    """Verify all 8 patches are applied. Returns True if all OK."""
    paths = _get_paths()

    with open(paths["model_base"], "r") as f:
        base = f.read()
    with open(paths["model_wrapper"], "r") as f:
        wrapper = f.read()
    with open(paths["hf_adapter"], "r") as f:
        adapter = f.read()

    checks = [
        (
            "1. Skip set_none_if_empty",
            "# inputs_embeds = self.set_none_if_empty(inputs_embeds)" in base,
        ),
        ("2. torch.where pattern", "has_embeds = (inputs_embeds.sum() != 0)" in base),
        ("3a. CTE 19-arg", "_empties_cte" in base),
        ("3b. TKG 19-arg", "_empties_tkg" in base),
        ("4. forward() threading", "Thread inputs_embeds" in base),
        ("5. input_generator 19-arg", "inputs_embeds_slot" in wrapper),
        ("6. pad_inputs", "Pad inputs_embeds at position 18" in wrapper),
        (
            "7. hf_adapter fix",
            "not self.prev_kv_cache_populated" in adapter and "PATCHED-V3b" in adapter,
        ),
        ("8. dummy input_ids", "PATCHED-V3b: Create dummy input_ids" in base),
    ]

    all_ok = True
    for desc, result in checks:
        status = "OK" if result else "FAIL"
        log.info(f"  [{status}] {desc}")
        if not result:
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("NxDI Embedding Injection Patch V3b (8 patches, uniform 19-arg)")
    print("Target: neuronx-distributed-inference 0.8.0 (SDK 2.28)")
    print("=" * 70)

    if "--restore" in sys.argv:
        restore_from_backups()
        sys.exit(0)

    if "--verify" in sys.argv:
        ok = verify_patches()
        sys.exit(0 if ok else 1)

    apply_all_patches()
    ok = verify_patches()

    if ok:
        print("\nAll 8 V3b patches applied and verified!")
        print("\nIMPORTANT: Clear cache and recompile after patching:")
        print("  rm -rf /var/tmp/neuron-compile-cache/ /tmp/nxd_model/")
    else:
        print("\nWARNING: Some verification checks failed.")

    sys.exit(0 if ok else 1)
