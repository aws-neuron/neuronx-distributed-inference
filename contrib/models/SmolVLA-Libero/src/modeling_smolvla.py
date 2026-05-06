"""
SmolVLA top-level application class.

Orchestrates the three compiled subgraphs and the CPU-side Euler loop:

    images, lang_ids, state
        -> Vision NEFF (per camera) -> stack [B, 192, 960]
        -> Prefix NEFF              -> prefix_keys, prefix_values
        -> CPU loop over 10 Euler steps:
                Denoise NEFF(noisy_actions, t, K, V) -> v_t
                noisy_actions <- noisy_actions + dt * v_t
        -> action chunk [B, 50, 32]

The compiled subgraphs are compiled via NxDI's ModelBuilder, which initializes
parallel_state so ColumnParallelLinear / RowParallelLinear use the parallel
path. tp_degree=1 on this hardware (see config_constants.py — 15 attn heads
don't divide the 4 cores cleanly).

DEVIATIONS FROM "everything on Neuron":
  - The 10-step Euler loop runs on CPU. Static-shape compilation cannot host
    a Python `for step in range(N)` — the loop body is the compiled graph.
  - Image preprocessing (resize, normalize) and tokenization run on CPU.
    Dataloading, not model compute.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List

import torch
import torch.nn as nn

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
from safetensors.torch import load_file

import config_constants as C
from neuron_action_head_base import NeuronDenoisingConfig, COMPILED_MODEL_FILE_NAME
from modeling_smolvla_vision import SmolVLAVisionEncoder
from modeling_smolvla_text import SmolVLAPrefixModel, SmolVLADenoiseStep
from weight_mapping import load_hf_state_dict, split_hf_state_dict

logger = logging.getLogger("SmolVLA")

# ---------------------------------------------------------------------------
# NEFF directory layout
# ---------------------------------------------------------------------------

VISION_NEFF_SUBDIR  = "vision"
PREFIX_NEFF_SUBDIR  = "prefix"
DENOISE_NEFF_SUBDIR = "denoise"


def _make_config(tp_degree: int = C.DEFAULT_TP_DEGREE,
                 batch_size: int = C.BATCH_SIZE) -> NeuronDenoisingConfig:
    """Construct a config object compatible with NxDI's ModelWrapper."""
    return NeuronDenoisingConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        action_chunk_size=C.ACTION_CHUNK_SIZE,
        action_dim=C.MAX_ACTION_DIM,
        num_conditioning_tokens=C.PREFIX_LEN,
        conditioning_hidden_size=C.VLM_HIDDEN,
        timestep_embed_dim=C.TIMESTEP_EMBED_DIM,
        torch_dtype=torch.bfloat16,
    )


def _compiler_args() -> str:
    """Standard DiT-safe compiler args (see action_head_translation.md)."""
    return (
        "--auto-cast=none "
        "-O1 "
        "--tensorizer-options='"
        "--enable-ccop-compute-overlap "
        "--cc-pipeline-tiling-factor=1'"
    )


# ---------------------------------------------------------------------------
# SmolVLAPolicy — compile / load / generate
# ---------------------------------------------------------------------------

class SmolVLAPolicy(nn.Module):
    """
    End-to-end SmolVLA policy on Trainium.

    Lifecycle:
        compile(save_dir, hf_checkpoint) -> writes 3 NEFFs and 3 sharded weight dirs
        load(save_dir)                   -> loads 3 NEFFs to Neuron
        generate(images, lang_ids, state) -> [B, 50, 32] action chunk
    """

    def __init__(self,
                 hf_checkpoint_dir: str,
                 tp_degree: int = C.DEFAULT_TP_DEGREE,
                 batch_size: int = C.BATCH_SIZE):
        super().__init__()
        self.hf_checkpoint_dir = hf_checkpoint_dir
        self.config = _make_config(tp_degree=tp_degree, batch_size=batch_size)

        # ModelBuilder needs a callable that constructs the nn.Module after
        # parallel_state is initialized. Use BaseModelInstance with module_cls
        # set to a no-arg lambda.
        # Cache the HF state dict slices for use as ModelBuilder checkpoint loaders
        self._hf_sd = None
        self._vision_sd = None
        self._prefix_sd = None
        self._denoise_sd = None

        # Loaded NEFFs
        self._vision_traced = None
        self._prefix_traced = None
        self._denoise_traced = None

    # ----------------------------------------------------------------------
    # Checkpoint loaders for ModelBuilder
    # ----------------------------------------------------------------------

    def _ensure_hf_loaded(self):
        if self._hf_sd is None:
            self._hf_sd = load_hf_state_dict(self.hf_checkpoint_dir)
            self._vision_sd, self._prefix_sd, self._denoise_sd = split_hf_state_dict(self._hf_sd)
            # Cast everything to bf16 (cast_type='config' default)
            for sd in (self._vision_sd, self._prefix_sd, self._denoise_sd):
                for k, v in list(sd.items()):
                    if torch.is_floating_point(v) and v.dtype != torch.bfloat16:
                        sd[k] = v.to(torch.bfloat16)

    def _vision_loader(self, mmap=False):
        self._ensure_hf_loaded()
        return self._vision_sd

    def _prefix_loader(self, mmap=False):
        self._ensure_hf_loaded()
        return self._prefix_sd

    def _denoise_loader(self, mmap=False):
        self._ensure_hf_loaded()
        return self._denoise_sd

    # ----------------------------------------------------------------------
    # Compile
    # ----------------------------------------------------------------------

    def _build_one(self, tag: str, module_cls, example_inputs, save_dir: str, ckpt_loader):
        """Compile a single subgraph via ModelBuilder and shard its weights.

        Args:
            tag: Subgraph name (used as the key in builder.add).
            module_cls: No-arg callable returning an nn.Module. Called by
                BaseModelInstance.load_module() AFTER parallel_state is up.
            example_inputs: List[Tuple[Tensor, ...]] — one tuple per bucket.
            save_dir: Output dir for NEFF + sharded weights.
            ckpt_loader: Callable returning the state_dict for this subgraph.
        """
        os.makedirs(save_dir, exist_ok=True)
        builder = ModelBuilder(
            router=None,
            tp_degree=self.config.neuron_config.tp_degree,
            pp_degree=1,
            ep_degree=1,
            world_size=self.config.neuron_config.tp_degree,
            start_rank_id=0,
            local_ranks_size=self.config.neuron_config.tp_degree,
            checkpoint_loader=ckpt_loader,
            compiler_workdir=os.path.join(save_dir, "compiler_workdir"),
        )
        instance = BaseModelInstance(module_cls=module_cls, input_output_aliases={})
        builder.add(
            key=tag,
            model_instance=instance,
            example_inputs=example_inputs,
            compiler_args=_compiler_args(),
        )
        traced = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced, os.path.join(save_dir, COMPILED_MODEL_FILE_NAME))
        sharded_dir = os.path.join(save_dir, "weights")
        os.makedirs(sharded_dir, exist_ok=True)
        builder.shard_checkpoint(serialize_path=sharded_dir + "/")
        del traced
        logger.info(f"Compiled {tag} -> {save_dir}")

    def compile(self, save_root: str):
        """Compile all three NEFFs to save_root/{vision,prefix,denoise}/."""
        self._ensure_hf_loaded()

        B = self.config.neuron_config.batch_size

        # 1. Vision
        vision_inputs = [(
            torch.zeros(B, 3, C.VISION_IMAGE_SIZE, C.VISION_IMAGE_SIZE, dtype=torch.bfloat16),
        )]
        vdir = os.path.join(save_root, VISION_NEFF_SUBDIR)
        logger.info("=== Compiling vision encoder ===")
        t0 = time.monotonic()
        self._build_one("vision_encoder", SmolVLAVisionEncoder, vision_inputs, vdir, self._vision_loader)
        logger.info(f"Vision compile time: {time.monotonic()-t0:.1f}s")

        # 2. Prefix
        prefix_inputs = [(
            torch.zeros(B, C.NUM_VISION_TOKENS_TOTAL, C.VLM_HIDDEN, dtype=torch.bfloat16),
            torch.zeros(B, C.NUM_TEXT_TOKENS, dtype=torch.int32),
            torch.ones(B, C.NUM_TEXT_TOKENS, dtype=torch.bool),
            torch.zeros(B, C.MAX_STATE_DIM, dtype=torch.float32),
        )]
        pdir = os.path.join(save_root, PREFIX_NEFF_SUBDIR)
        logger.info("=== Compiling VLM prefix ===")
        t0 = time.monotonic()
        self._build_one("prefix", SmolVLAPrefixModel, prefix_inputs, pdir, self._prefix_loader)
        logger.info(f"Prefix compile time: {time.monotonic()-t0:.1f}s")

        # 3. Denoise step
        denoise_inputs = [(
            torch.zeros(B, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM, dtype=torch.float32),
            torch.zeros(B, dtype=torch.float32),
            torch.zeros(C.VLM_NUM_LAYERS, B, C.PREFIX_LEN, C.VLM_NUM_KV_HEADS, C.VLM_HEAD_DIM, dtype=torch.bfloat16),
            torch.zeros(C.VLM_NUM_LAYERS, B, C.PREFIX_LEN, C.VLM_NUM_KV_HEADS, C.VLM_HEAD_DIM, dtype=torch.bfloat16),
            torch.ones(B, C.PREFIX_LEN, dtype=torch.bool),
        )]
        ddir = os.path.join(save_root, DENOISE_NEFF_SUBDIR)
        logger.info("=== Compiling denoise step ===")
        t0 = time.monotonic()
        self._build_one("denoise_step", SmolVLADenoiseStep, denoise_inputs, ddir, self._denoise_loader)
        logger.info(f"Denoise compile time: {time.monotonic()-t0:.1f}s")

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------

    def _load_one(self, save_dir: str):
        traced = torch.jit.load(os.path.join(save_dir, COMPILED_MODEL_FILE_NAME))
        weights = []
        local_ranks = self.config.neuron_config.tp_degree
        for rank in range(local_ranks):
            ckpt = load_file(os.path.join(save_dir, "weights", f"tp{rank}_sharded_checkpoint.safetensors"))
            weights.append(ckpt)
        start_rank = torch.tensor([0], dtype=torch.int32)
        traced.nxd_model.initialize(weights, start_rank)
        return traced

    def load(self, save_root: str):
        """Load three NEFFs and their pre-sharded weights to Neuron device."""
        logger.info("Loading vision NEFF...")
        self._vision_traced  = self._load_one(os.path.join(save_root, VISION_NEFF_SUBDIR))
        logger.info("Loading prefix NEFF...")
        self._prefix_traced  = self._load_one(os.path.join(save_root, PREFIX_NEFF_SUBDIR))
        logger.info("Loading denoise NEFF...")
        self._denoise_traced = self._load_one(os.path.join(save_root, DENOISE_NEFF_SUBDIR))

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    def _embed_cameras(self, images: List[torch.Tensor]) -> torch.Tensor:
        """Run vision NEFF once per camera and concat -> [B, 192, 960]."""
        outs = []
        for img in images:
            assert img.shape == (self.config.neuron_config.batch_size, 3,
                                 C.VISION_IMAGE_SIZE, C.VISION_IMAGE_SIZE), (
                f"Each camera image must be [B, 3, {C.VISION_IMAGE_SIZE}, {C.VISION_IMAGE_SIZE}]"
            )
            out = self._vision_traced.nxd_model.forward([img.to(torch.bfloat16)])
            # ModelBuilder traced forward returns a list/tuple — unwrap
            out = out[0] if isinstance(out, (list, tuple)) else out
            outs.append(out)
        return torch.cat(outs, dim=1)   # [B, 192, 960]

    @torch.no_grad()
    def generate(
        self,
        images: List[torch.Tensor],     # length-NUM_CAMERAS list, each [B, 3, 512, 512]
        lang_token_ids: torch.Tensor,   # [B, NUM_TEXT_TOKENS] INT32
        state: torch.Tensor,            # [B, 32] FP32
        lang_mask: torch.Tensor = None, # [B, NUM_TEXT_TOKENS] BOOL  (defaults to all-True)
        num_steps: int = C.NUM_DENOISE_STEPS,
        noise: torch.Tensor = None,     # [B, ACTION_CHUNK_SIZE, MAX_ACTION_DIM] FP32, optional
    ) -> torch.Tensor:
        """Run the full pipeline: vision -> prefix -> N denoise steps -> action chunk.

        ``noise`` controls the Euler-loop initial state. Pass an explicit tensor
        (e.g. for parity testing against a CPU reference) to bypass the
        ``torch.randn`` default.
        """
        assert len(images) == C.NUM_CAMERAS, f"Expected {C.NUM_CAMERAS} camera tensors"
        assert isinstance(num_steps, int)

        B = state.shape[0]
        if lang_mask is None:
            lang_mask = torch.ones(B, C.NUM_TEXT_TOKENS, dtype=torch.bool)
        lang_mask = lang_mask.to(torch.bool)

        # 1. Vision (returns fp32 from compiled graph; cast to bf16 for prefix)
        vision_features = self._embed_cameras(images).to(torch.bfloat16)   # [B, NV, 960]

        # 2. Prefix -> KV cache (with attention_mask)
        prefix_out = self._prefix_traced.nxd_model.forward([
            vision_features,
            lang_token_ids.to(torch.int32),
            lang_mask,
            state.to(torch.float32),
        ])
        pk, pv = prefix_out
        pk = pk.to(torch.bfloat16)
        pv = pv.to(torch.bfloat16)

        # Build the prefix-wide pad mask: vision (all valid) + lang_mask + state (valid)
        prefix_pad = torch.cat([
            torch.ones(B, C.NUM_VISION_TOKENS_TOTAL, dtype=torch.bool),
            lang_mask,
            torch.ones(B, C.NUM_STATE_TOKENS, dtype=torch.bool),
        ], dim=1)                                                               # [B, PREFIX_LEN]

        # 3. Euler loop on CPU
        if noise is not None:
            assert tuple(noise.shape) == (B, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM), (
                f"noise must have shape [B={B}, {C.ACTION_CHUNK_SIZE}, {C.MAX_ACTION_DIM}], "
                f"got {tuple(noise.shape)}"
            )
            x_t = noise.to(torch.float32)
        else:
            x_t = torch.randn(B, C.ACTION_CHUNK_SIZE, C.MAX_ACTION_DIM, dtype=torch.float32)
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor([t] * B, dtype=torch.float32)
            v_t = self._denoise_traced.nxd_model.forward([x_t, t_tensor, pk, pv, prefix_pad])
            v_t = v_t[0] if isinstance(v_t, (list, tuple)) else v_t
            x_t = x_t + dt * v_t.to(torch.float32)

        return x_t   # [B, 50, 32]
