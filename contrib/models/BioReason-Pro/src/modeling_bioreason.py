#!/usr/bin/env python3
"""
BioReason-Pro: Multimodal protein function prediction on AWS Neuron.

This module provides the `BioReasonPipeline` class that wraps a Qwen3-4B backbone
(compiled via NxDI) with ESM3 protein encoding and embedding injection for
multimodal protein function prediction.

Architecture:
  - ESM3-small (~1.4B) encodes protein sequence on CPU -> per-residue embeddings
  - Protein projection layer maps ESM3 dim (1536) -> Qwen3 hidden dim (2560)
  - Pre-computed GO graph embedding (200 x 2560) is projected into Qwen3 space
  - Placeholder tokens (<|protein_pad|>, <|go_graph_pad|>) in the prompt are
    replaced with projected embeddings before Qwen3-4B generation
  - Qwen3-4B runs on a single NeuronCore via NxDI with 8 runtime patches
    that enable `inputs_embeds` passthrough through the compiled model

Requires:
  - neuronx-distributed-inference >= 0.8.0 (SDK 2.28)
  - esm (EvolutionaryScale ESM3, gated model, requires HF auth)
  - wanglab/bioreason-pro-rl checkpoint (includes projection weights + GO embedding)
  - trn2.3xlarge (or any Neuron instance with >= 1 NeuronCore)

Usage:
    from modeling_bioreason import BioReasonPipeline

    pipeline = BioReasonPipeline(
        model_path="/mnt/models/bioreason-pro-rl",
        esm3_model="esm3_sm_open_v1",
    )
    result = pipeline.predict(
        sequence="MSSQQYQ...",
        organism="Mus musculus (Mouse)",
        interpro="- IPR000762: Midkine...",
        gogpt="GO:0005576 (extracellular region)...",
    )
    print(result)  # Generated protein function analysis
"""

import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Valid amino acids for sequence cleaning
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Prompt templates (from bioreason2.dataset.prompts.cafa5)
SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a scientific assistant specialized in protein function prediction. "
    "Given a protein sequence, organism information, and additional context "
    "(InterPro domain annotations and/or initial GO term speculations), "
    "step-by-step reason about the InterPro terms, Gene Ontology (GO) terms "
    "regarding molecular function, biological process, and cellular component, "
    "protein-protein interactions (PPI), and overall function. Use the provided "
    "information as a starting point and improve upon it with deeper analysis. "
    "Provide a summary of your findings in your final answer."
)

SYSTEM_PROMPT_NO_CONTEXT = (
    "You are a scientific assistant specialized in protein function prediction. "
    "Given a protein sequence and organism information, step-by-step reason about "
    "the InterPro terms, Gene Ontology (GO) terms regarding molecular function, "
    "biological process, and cellular component, protein-protein interactions (PPI), "
    "and overall function. Provide a summary of your findings in your final answer."
)

GO_ASPECTS_SUFFIX = (
    " and focus more on its Molecular Function, Biological Process, Cellular Component."
)
UNIPROT_SUFFIX = " Summarize in UniProt format."


# ---------------------------------------------------------------------------
# ESM3 Encoder (CPU)
# ---------------------------------------------------------------------------


class ESM3Encoder:
    """ESM3-small encoder on CPU for protein sequence embedding."""

    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        device: str = "cpu",
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, SamplingConfig

        log.info(f"Loading ESM3 '{model_name}' on {device}...")
        t0 = time.time()
        self.model = ESM3.from_pretrained(model_name, device=torch.device(device))
        self.model.eval()
        self.device = device
        self.output_dtype = output_dtype
        self.ESMProtein = ESMProtein
        self.SamplingConfig = SamplingConfig
        self.embedding_dim = self.model.encoder.sequence_embed.embedding_dim
        log.info(f"ESM3 loaded in {time.time() - t0:.1f}s (dim={self.embedding_dim})")

    @torch.no_grad()
    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a protein sequence to per-residue embeddings.

        Args:
            sequence: Amino acid sequence string (e.g., "MSSQQYQ...")

        Returns:
            Tensor of shape (seq_len+2, embedding_dim) in output_dtype.
            The +2 accounts for BOS/EOS tokens added by ESM3.
        """
        protein = self.ESMProtein(sequence=sequence)
        protein_tensor = self.model.encode(protein)
        output = self.model.forward_and_sample(
            protein_tensor,
            self.SamplingConfig(return_per_residue_embeddings=True),
        )
        return output.per_residue_embedding.to(self.output_dtype)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def clean_sequence(seq: str) -> str:
    """Clean and validate a protein sequence."""
    seq = seq.strip()
    seq = re.sub(r"[\s\n\t\r]+", "", seq)
    seq = seq.upper()
    return "".join(c for c in seq if c in VALID_AA)


def load_projection(
    path: str,
    in_dim: int,
    out_dim: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Sequential:
    """Load a 2-layer projection MLP (Linear -> GELU -> Linear)."""
    proj = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim),
    ).to(device=device, dtype=dtype)
    state = torch.load(path, map_location=device)
    proj.load_state_dict(state, strict=True)
    proj.eval()
    return proj


def load_go_embeddings(
    model_path: str,
    text_hidden_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Optional[torch.Tensor], Optional[nn.Sequential]]:
    """Load pre-computed GO graph embedding and projection layer."""
    go_embedding_path = os.path.join(model_path, "go_embedding.pt")
    go_proj_path = os.path.join(model_path, "go_projection.pt")
    if not os.path.exists(go_embedding_path):
        return None, None
    go_embedding = torch.load(go_embedding_path, map_location=device).to(dtype=dtype)
    go_projection = None
    if os.path.exists(go_proj_path):
        go_embedding_dim = go_embedding.shape[1]
        go_projection = load_projection(
            go_proj_path, go_embedding_dim, text_hidden_size, device, dtype
        )
    return go_embedding, go_projection


def load_token_embedding_layer(
    model_path: str,
    vocab_size: int,
    hidden_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Embedding:
    """Load the token embedding layer from model checkpoint."""
    from safetensors import safe_open
    import glob as glob_mod

    embed_layer = nn.Embedding(vocab_size, hidden_size).to(device=device, dtype=dtype)
    st_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st_path):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            embed_layer.weight.data = f.get_tensor("model.embed_tokens.weight").to(
                dtype=dtype
            )
        return embed_layer
    shards = sorted(glob_mod.glob(os.path.join(model_path, "model-*.safetensors")))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            if "model.embed_tokens.weight" in f.keys():
                embed_layer.weight.data = f.get_tensor("model.embed_tokens.weight").to(
                    dtype=dtype
                )
                return embed_layer
    raise RuntimeError("Could not find model.embed_tokens.weight in checkpoint")


# ---------------------------------------------------------------------------
# NxDI Model Loading
# ---------------------------------------------------------------------------


def load_nxdi_model(
    model_path: str,
    max_context_length: int = 1024,
    max_new_tokens: int = 2048,
    batch_size: int = 1,
    tp_degree: int = 1,
    compiled_model_path: str = None,
):
    """Load BioReason-Pro's Qwen3-4B backbone via NxDI.

    This function:
    1. Applies the V3b patches to enable inputs_embeds passthrough
    2. Configures NxDI for the Qwen3-4B architecture
    3. Compiles or loads a pre-compiled model
    4. Returns an HF-compatible generation adapter

    Args:
        model_path: Path to the wanglab/bioreason-pro-rl checkpoint
        max_context_length: Maximum input context length (default: 1024)
        max_new_tokens: Maximum generation length (default: 2048)
        batch_size: Batch size for compiled model (default: 1)
        tp_degree: Tensor parallelism degree (default: 1)

    Returns:
        Tuple of (hf_adapter, neuron_model)
    """
    # Apply V3b patches before importing NxDI model classes
    try:
        from .patch_nxdi_embeds import apply_all_patches
    except ImportError:
        from patch_nxdi_embeds import apply_all_patches

    apply_all_patches()

    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import (
        NeuronQwen3ForCausalLM,
        Qwen3InferenceConfig,
    )

    with open(os.path.join(model_path, "config.json")) as f:
        hf_config_dict = json.load(f)
    if "pad_token_id" not in hf_config_dict:
        hf_config_dict["pad_token_id"] = hf_config_dict.get("eos_token_id", 151645)

    max_length = max_context_length + max_new_tokens
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        tp_degree=tp_degree,
        max_context_length=max_context_length,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        on_device_sampling_config=OnDeviceSamplingConfig(do_sample=False, top_k=1),
        torch_dtype=torch.bfloat16,
    )

    config = Qwen3InferenceConfig(neuron_config=neuron_config, **hf_config_dict)
    config.output_attentions = False
    config.output_hidden_states = False
    config.use_cache = True
    config.is_encoder_decoder = False

    # Use separate compiled_model_path if provided (allows per-batch-size compilation)
    out_path = compiled_model_path or model_path
    if compiled_model_path:
        os.makedirs(compiled_model_path, exist_ok=True)

    log.info(f"Loading NxDI Qwen3-4B from {model_path} (compiled to {out_path})...")
    t0 = time.time()
    neuron_model = NeuronQwen3ForCausalLM(model_path, config=config)

    compiled_path = os.path.join(out_path, "model.pt")
    if os.path.exists(compiled_path):
        neuron_model.load(out_path)
    else:
        log.info("Compiling (15-20 min for first run)...")
        neuron_model.compile(out_path)
        neuron_model.load(out_path)

    log.info(f"NxDI model ready in {time.time() - t0:.1f}s")
    hf_adapter = HuggingFaceGenerationAdapter(neuron_model)
    return hf_adapter, neuron_model


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------


def build_prompt(organism: str, interpro: str = "", gogpt: str = "") -> str:
    """Build the text portion of the BioReason-Pro prompt.

    Args:
        organism: Organism name (e.g., "Mus musculus (Mouse)")
        interpro: InterPro domain annotations (may be empty)
        gogpt: GO-GPT GO term predictions (may be empty)

    Returns:
        Formatted prompt string (without chat template wrapper).
    """
    if interpro or gogpt:
        system = SYSTEM_PROMPT_WITH_CONTEXT
        user = (
            f"Given the protein above from organism {organism} with the following "
            f"InterPro annotations:\n{interpro if interpro else 'None'}\n\n"
            f"And the following initial GO term speculations:\n"
            f"{gogpt if gogpt else 'None'}\n\n"
            f"Reason about the function of the protein."
        )
    else:
        system = SYSTEM_PROMPT_NO_CONTEXT
        user = (
            f"Given the protein above from organism {organism}, "
            f"reason about the function of the protein."
        )

    user = user.rstrip(".") + GO_ASPECTS_SUFFIX + UNIPROT_SUFFIX
    return f"{system.strip()}\n\n{user.strip()}"


# ---------------------------------------------------------------------------
# BioReason Pipeline
# ---------------------------------------------------------------------------


class BioReasonPipeline:
    """End-to-end BioReason-Pro inference pipeline on Neuron.

    Handles ESM3 protein encoding, embedding injection into Qwen3-4B
    placeholder tokens, and autoregressive generation via NxDI.

    Example:
        pipeline = BioReasonPipeline("/mnt/models/bioreason-pro-rl")
        result = pipeline.predict(
            sequence="MSSQQYQ...",
            organism="Mus musculus (Mouse)",
        )
    """

    def __init__(
        self,
        model_path: str,
        esm3_model: str = "esm3_sm_open_v1",
        max_context_length: int = 1024,
        max_new_tokens: int = 2048,
        batch_size: int = 1,
        tp_degree: int = 1,
        compiled_model_path: str = None,
    ):
        """Initialize the BioReason-Pro pipeline.

        Args:
            model_path: Path to wanglab/bioreason-pro-rl checkpoint
            esm3_model: ESM3 model name (default: esm3_sm_open_v1)
            max_context_length: Max input context for NxDI (default: 1024)
            max_new_tokens: Max generation tokens (default: 2048)
            batch_size: NxDI batch size (default: 1)
            tp_degree: NxDI tensor parallelism (default: 1)
            compiled_model_path: Separate path for compiled model artifacts
                (default: None, uses model_path)
        """
        from transformers import AutoTokenizer, AutoConfig

        self.model_path = model_path
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = hf_config.hidden_size
        self.vocab_size = hf_config.vocab_size

        # Add special tokens for embedding injection
        PROTEIN_PAD_TOKEN = "<|protein_pad|>"
        GO_GRAPH_PAD_TOKEN = "<|go_graph_pad|>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [PROTEIN_PAD_TOKEN, GO_GRAPH_PAD_TOKEN]}
        )
        self.protein_token_id = self.tokenizer.convert_tokens_to_ids(PROTEIN_PAD_TOKEN)
        self.go_token_id = self.tokenizer.convert_tokens_to_ids(GO_GRAPH_PAD_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load PLProcessor for chat template formatting
        # bioreason2 package lives at ~/bioreason2; need parent dir on path
        sys.path.insert(0, os.path.expanduser("~"))
        from bioreason2.models.pl.processing_pl import PLProcessor
        from bioreason2.models.pl.chat_template_pl import get_chat_template

        self.processor = PLProcessor(tokenizer=self.tokenizer)
        self.tokenizer.chat_template = get_chat_template("Qwen/Qwen3-4B-Thinking-2507")

        # Load ESM3 encoder (CPU)
        self.esm3 = ESM3Encoder(esm3_model, device="cpu", output_dtype=torch.bfloat16)

        # Load projection layers
        self.protein_proj = load_projection(
            os.path.join(model_path, "protein_projection.pt"),
            self.esm3.embedding_dim,
            self.hidden_size,
            "cpu",
            torch.bfloat16,
        )
        go_raw, go_proj = load_go_embeddings(
            model_path, self.hidden_size, "cpu", torch.bfloat16
        )

        # Pre-compute GO projections (same for all proteins)
        if go_raw is not None and go_proj is not None:
            with torch.no_grad():
                self.go_projected = go_proj(go_raw)
        else:
            self.go_projected = torch.zeros(0, self.hidden_size, dtype=torch.bfloat16)

        # Load token embedding layer (for building inputs_embeds)
        self.embed_layer = load_token_embedding_layer(
            model_path, self.vocab_size, self.hidden_size, "cpu", torch.bfloat16
        )

        # Load NxDI model (applies V3b patches)
        self.hf_adapter, self.neuron_model = load_nxdi_model(
            model_path,
            max_context_length=max_context_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            tp_degree=tp_degree,
            compiled_model_path=compiled_model_path,
        )

    def _build_inputs_embeds(
        self,
        sequence: str,
        organism: str,
        interpro: str = "",
        gogpt: str = "",
    ) -> torch.Tensor:
        """Build inputs_embeds with protein and GO embeddings injected.

        Args:
            sequence: Cleaned protein sequence
            organism: Organism name
            interpro: InterPro annotations
            gogpt: GO-GPT predictions

        Returns:
            inputs_embeds tensor of shape (1, seq_len, hidden_size) in BF16
        """
        # ESM3 encode + project
        protein_embeds = self.esm3.encode(sequence)
        with torch.no_grad():
            protein_projected = self.protein_proj(protein_embeds)

        # Build prompt and tokenize
        prompt_text = build_prompt(organism, interpro, gogpt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "protein", "text": None},
                    {"type": "go_graph", "text": None},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        prompt_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        processed = self.processor(
            text=[prompt_string],
            batch_protein_sequences=[[sequence]],
            batch_go_aspects=["all"],
            max_length_text=512,
            max_length_protein=2048,
            return_tensors="pt",
        )
        input_ids = processed["input_ids"]
        if input_ids.shape[1] > self.max_context_length:
            input_ids = input_ids[:, : self.max_context_length]

        # Build inputs_embeds from token embeddings
        with torch.no_grad():
            inputs_embeds = self.embed_layer(input_ids).to(torch.bfloat16)

        # Replace protein placeholder tokens with projected embeddings
        protein_mask = input_ids[0] == self.protein_token_id
        n_protein = protein_mask.sum().item()
        if n_protein > 0 and protein_projected.shape[0] > 0:
            n_fill = min(n_protein, protein_projected.shape[0])
            if n_protein <= protein_projected.shape[0]:
                inputs_embeds[0, protein_mask] = protein_projected[:n_protein]
            else:
                protein_indices = protein_mask.nonzero().squeeze(-1)
                inputs_embeds[0, protein_indices[:n_fill]] = protein_projected[:n_fill]

        # Replace GO placeholder tokens with projected embeddings
        go_mask = input_ids[0] == self.go_token_id
        n_go = go_mask.sum().item()
        if n_go > 0 and self.go_projected.shape[0] > 0:
            n_fill = min(n_go, self.go_projected.shape[0])
            if n_go <= self.go_projected.shape[0]:
                inputs_embeds[0, go_mask] = self.go_projected[:n_go]
            else:
                go_indices = go_mask.nonzero().squeeze(-1)
                inputs_embeds[0, go_indices[:n_fill]] = self.go_projected[:n_fill]

        return inputs_embeds

    def predict(
        self,
        sequence: str,
        organism: str,
        interpro: str = "",
        gogpt: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run BioReason-Pro inference for a single protein.

        Args:
            sequence: Amino acid sequence
            organism: Organism name (e.g., "Mus musculus (Mouse)")
            interpro: InterPro annotations (optional)
            gogpt: GO-GPT GO term predictions (optional)
            max_new_tokens: Override max generation length (optional)

        Returns:
            Dict with keys: 'text', 'num_tokens', 'gen_time_s', 'total_time_s'
        """
        t_start = time.time()
        sequence = clean_sequence(sequence)
        max_tokens = max_new_tokens or self.max_new_tokens

        # Build inputs_embeds with protein + GO embeddings injected
        inputs_embeds = self._build_inputs_embeds(sequence, organism, interpro, gogpt)

        # Generate
        t_gen = time.time()
        with torch.no_grad():
            output_ids = self.hf_adapter.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        gen_time = time.time() - t_gen

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_tokens = len(output_ids[0])
        total_time = time.time() - t_start

        return {
            "text": text,
            "num_tokens": num_tokens,
            "gen_time_s": gen_time,
            "total_time_s": total_time,
            "tok_per_s": num_tokens / gen_time if gen_time > 0 else 0,
        }

    def generate_with_scores(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ):
        """Generate with output scores for logit validation.

        Args:
            inputs_embeds: Pre-built inputs_embeds (1, seq_len, hidden_size)
            max_new_tokens: Override max generation length

        Returns:
            HF GenerateOutput with scores attribute
        """
        max_tokens = max_new_tokens or self.max_new_tokens
        with torch.no_grad():
            output = self.hf_adapter.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        return output
