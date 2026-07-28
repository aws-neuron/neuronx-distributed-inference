#!/usr/bin/env python3
"""YuE Stage 1 Worker -- runs S1 (7B) on Neuron in isolated process.

Called by yue_e2e_neuron.py orchestrator. Args passed via YUE_STAGE_ARGS env var.

CFG (Classifier-Free Guidance) implementation using a CUSTOM generation loop:
  - Model compiled with batch_size=2, padding_side=right
  - Row 0 = conditional input (full lyrics/genre prompt)
  - Row 1 = unconditional input (last token only, right-padded)
  - Custom loop calls neuron_model.forward() directly
  - After each forward pass, CFG blends logits and samples ONE token
  - The SAME sampled token is fed to BOTH rows, keeping KV caches synchronized
  - guidance_scale=1.5 (first segment), 1.2 (subsequent) -- matches original YuE
  - NxDI requires padding_side=right for batch_size > 1 (assertion in attention_base.py)

Environment variables:
    MODEL_DIR: Root directory for models and compiled artifacts (default: /mnt/models)
    YUE_STAGE_ARGS: JSON-encoded arguments from orchestrator
"""

import os
import sys
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, LogitsProcessor, LogitsProcessorList

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/models")

sys.path.insert(0, os.path.join(MODEL_DIR, "YuE/inference"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer/descriptaudiocodec"))

from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
from mmtokenizer import _MMSentencePieceTokenizer
from codecmanipulator import CodecManipulator

# Paths (derived from MODEL_DIR)
S1_MODEL_PATH = os.path.join(MODEL_DIR, "YuE-s1-7B-anneal-en-cot")
TOKENIZER_PATH = os.path.join(
    MODEL_DIR, "YuE/inference/mm_tokenizer_v0.2_hf/tokenizer.model"
)

S1_SEQ_LEN = 4096
S1_MAX_CONTEXT = 2048
S1_MAX_NEW_TOKENS = S1_SEQ_LEN - S1_MAX_CONTEXT


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def cfg_generate(
    neuron_model,
    cond_ids,
    uncond_ids,
    attention_mask,
    guidance_scale,
    max_new_tokens,
    min_new_tokens,
    eos_token_id,
    pad_token_id,
    logits_processor,
    top_p=0.93,
    temperature=1.0,
    repetition_penalty=1.1,
):
    """Custom generation loop with CFG for NxDI.

    Both rows (cond=row0, uncond=row1) must already be right-padded to the same
    length. attention_mask distinguishes real tokens from padding.

    NxDI right-padding convention for token generation:
      - Prepend 1 to attention_mask each step (mask grows, no trimming)
      - position_ids = amax(cumsum - 1) + 1

    Returns: tensor of shape (1, cond_len + num_generated) containing the
    conditional input followed by the generated tokens.
    """
    batch_size = 2
    input_len = cond_ids.shape[1]

    # -- STEP 1: Reset KV cache --
    neuron_model.reset()

    # -- STEP 2: Context encoding --
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    from neuronx_distributed_inference.utils.hf_adapter import prepare_sampling_params

    sampling_params = prepare_sampling_params(batch_size=batch_size)

    input_ids = torch.cat([cond_ids, uncond_ids], dim=0)

    print(
        f"  CFG context encoding: input_ids={input_ids.shape}, "
        f"cond real tokens={attention_mask[0].sum().item()}, "
        f"uncond real tokens={attention_mask[1].sum().item()}",
        flush=True,
    )

    with torch.no_grad():
        outputs = neuron_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sampling_params=sampling_params,
        )

    cond_logits = outputs.logits[0, -1, :]
    uncond_logits = outputs.logits[1, -1, :]

    blended = _cfg_blend(cond_logits, uncond_logits, guidance_scale)

    generated_ids = cond_ids[0].tolist()

    blended = blended.unsqueeze(0)
    gen_tensor = torch.tensor([generated_ids], dtype=torch.long)
    blended = logits_processor(gen_tensor, blended)

    if repetition_penalty != 1.0:
        blended = _apply_rep_pen(blended, gen_tensor, repetition_penalty)

    if min_new_tokens > 0:
        blended[0, eos_token_id] = -float("inf")

    next_token = _top_p_sample(blended[0], top_p, temperature)
    generated_ids.append(next_token)
    all_new_tokens = [next_token]

    next_tokens = torch.tensor([[next_token], [next_token]], dtype=torch.long)

    # -- STEP 3: Update attention mask (right-pad: prepend 1, no trim) --
    attention_mask = torch.cat(
        [attention_mask.new_ones((batch_size, 1)), attention_mask], dim=-1
    )

    # -- STEP 4: Auto-regressive token generation --
    for step in range(1, max_new_tokens):
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 1)
        position_ids = torch.amax(pos, dim=1, keepdim=True) + 1

        with torch.no_grad():
            outputs = neuron_model.forward(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                sampling_params=sampling_params,
            )

        cond_logits = outputs.logits[0, -1, :]
        uncond_logits = outputs.logits[1, -1, :]

        blended = _cfg_blend(cond_logits, uncond_logits, guidance_scale)

        blended = blended.unsqueeze(0)
        gen_tensor = torch.tensor([generated_ids], dtype=torch.long)
        blended = logits_processor(gen_tensor, blended)

        if repetition_penalty != 1.0:
            blended = _apply_rep_pen(blended, gen_tensor, repetition_penalty)

        if step < min_new_tokens:
            blended[0, eos_token_id] = -float("inf")

        next_token = _top_p_sample(blended[0], top_p, temperature)
        generated_ids.append(next_token)
        all_new_tokens.append(next_token)

        if next_token == eos_token_id:
            break

        next_tokens = torch.tensor([[next_token], [next_token]], dtype=torch.long)

        attention_mask = torch.cat(
            [attention_mask.new_ones((batch_size, 1)), attention_mask],
            dim=-1,
        )

    print(f"  CFG generated {len(all_new_tokens)} tokens", flush=True)

    result = torch.tensor([generated_ids], dtype=torch.long)
    return result


def _cfg_blend(cond_logits, uncond_logits, scale):
    """Apply CFG formula in log-probability space."""
    cond_lp = F.log_softmax(cond_logits, dim=-1)
    uncond_lp = F.log_softmax(uncond_logits, dim=-1)
    blended = uncond_lp + scale * (cond_lp - uncond_lp)
    blended = torch.nan_to_num(blended, nan=-1e9, posinf=-1e9, neginf=-1e9)
    return blended


def _apply_rep_pen(logits, input_ids, penalty):
    """Apply repetition penalty to logits based on tokens in input_ids."""
    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


def _top_p_sample(logits, top_p, temperature):
    """Top-p (nucleus) sampling from logits vector."""
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_indices_to_remove] = 0.0

    sorted_probs = sorted_probs / sorted_probs.sum()

    idx = torch.multinomial(sorted_probs, 1)
    token = sorted_indices[idx].item()
    return token


def main():
    args = json.loads(os.environ["YUE_STAGE_ARGS"])

    # Seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    # CFG parameters
    use_cfg = args.get("use_cfg", True)
    guidance_scale_first = args.get("guidance_scale_first", 1.5)
    guidance_scale_rest = args.get("guidance_scale_rest", 1.2)
    batch_size = 2 if use_cfg else 1

    # TP degree (configurable for LNC=1 testing)
    s1_tp_degree = args.get("s1_tp_degree", 2)
    print(f"S1 TP degree: {s1_tp_degree}", flush=True)

    # Create neuron config
    hf_config = AutoConfig.from_pretrained(S1_MODEL_PATH)

    # NKI TKG kernel flags (fuse operations to reduce HBM round-trips)
    use_nki_kernels = args.get("use_nki_kernels", False)

    neuron_cfg_dict = {
        "tp_degree": s1_tp_degree,
        "batch_size": batch_size,
        "seq_len": S1_SEQ_LEN,
        "max_context_length": S1_MAX_CONTEXT,
        "max_new_tokens": S1_MAX_NEW_TOKENS,
        "torch_dtype": "bfloat16",
        "start_rank_id": 0,
    }
    if use_cfg:
        neuron_cfg_dict["padding_side"] = "right"

    if use_nki_kernels:
        # Enable TKG NKI MLP kernel only. The CTE MLP kernel (mlp_kernel_enabled)
        # has a hard limit of intermediate_size <= 4096 in the compiler backend
        # (walrus/inline_bir_kernel/src/kernels_impl/mlp.cpp:196). S1 has
        # intermediate_size=11008 so CTE compilation fails.
        #
        # Strategy: enable both flags in config, then monkeypatch NeuronLlamaMLP.forward
        # to skip the CTE kernel path (fall back to native) while keeping the TKG
        # NKI kernel for small B*S (token generation).
        neuron_cfg_dict.update(
            {
                "mlp_kernel_enabled": True,
                "mlp_tkg_nki_kernel_enabled": True,
            }
        )
        print("NKI TKG kernels ENABLED (MLP TKG only, CTE native fallback)", flush=True)

    neuron_cfg = {
        "neuron_config": neuron_cfg_dict,
        "architectures": hf_config.architectures,
        "model_type": hf_config.model_type,
        "hidden_size": hf_config.hidden_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "intermediate_size": hf_config.intermediate_size,
        "vocab_size": hf_config.vocab_size,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "rms_norm_eps": hf_config.rms_norm_eps,
        "hidden_act": hf_config.hidden_act,
        "rope_theta": hf_config.rope_theta,
        "tie_word_embeddings": hf_config.tie_word_embeddings,
        "attention_bias": hf_config.attention_bias,
        "mlp_bias": hf_config.mlp_bias,
        "torch_dtype": "bfloat16",
        "bos_token_id": hf_config.bos_token_id,
        "eos_token_id": hf_config.eos_token_id,
        "pad_token_id": hf_config.eos_token_id,
        "output_attentions": False,
        "output_hidden_states": False,
    }
    with open(os.path.join(S1_MODEL_PATH, "neuron_config.json"), "w") as f:
        json.dump(neuron_cfg, f, indent=2)

    # Build compiled path dynamically
    nki_suffix = "_nki" if use_nki_kernels else ""
    compiled_path = os.path.join(
        MODEL_DIR,
        f"compiled/s1_tp{s1_tp_degree}_bs{batch_size}_ctx{S1_MAX_CONTEXT}{nki_suffix}",
    )

    # Load model
    model = NeuronLlamaForCausalLM(S1_MODEL_PATH)

    # Apply MLP monkeypatch: TKG NKI kernel for token gen, manual matmul for CTE.
    if use_nki_kernels:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nki_mlp_patch  # noqa: F401 -- auto-applies on import

        print(
            "Applied nki_mlp_patch (TKG kernel for TKG, manual matmul for CTE)",
            flush=True,
        )

    skip_compile = args["skip_compile"]
    compiled_exists = os.path.exists(os.path.join(compiled_path, "model.pt"))

    if not skip_compile or not compiled_exists:
        os.makedirs(compiled_path, exist_ok=True)
        print(f"Compiling S1 (batch_size={batch_size})...", flush=True)
        t0 = time.time()
        model.compile(compiled_model_path=compiled_path)
        print(f"S1 compiled in {time.time() - t0:.1f}s", flush=True)

    print("Loading S1 to Neuron...", flush=True)
    t0 = time.time()
    model.load(compiled_path)
    print(f"S1 loaded in {time.time() - t0:.1f}s", flush=True)

    if use_cfg:
        print("CFG mode: using custom generation loop", flush=True)
    else:
        hf_model = HuggingFaceGenerationAdapter(model)
        hf_model.generation_config.do_sample = True
        hf_model.generation_config.top_p = 0.93
        hf_model.generation_config.temperature = 1.0
        hf_model.generation_config.repetition_penalty = 1.5

    # Tokenizer and tools
    mmtokenizer = _MMSentencePieceTokenizer(TOKENIZER_PATH)
    codectool = CodecManipulator("xcodec", 0, 1)

    # Prepare prompts
    genres = args["genres"]
    lyrics = args["lyrics"]
    max_new_tokens = args["max_new_tokens"]
    run_n_segments = args["run_n_segments"]

    full_lyrics = "\n".join(lyrics)
    prompt_texts = [
        f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"
    ]
    prompt_texts += lyrics

    top_p = 0.93
    temperature = 1.0

    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")

    pad_token_id = mmtokenizer.eoa

    raw_output = None
    run_n = min(run_n_segments + 1, len(lyrics))

    print(
        f"CFG: {use_cfg}, guidance_scale: first={guidance_scale_first}, rest={guidance_scale_rest}",
        flush=True,
    )

    gen_start = time.time()
    for i, p in enumerate(tqdm(prompt_texts[:run_n], desc="Stage 1")):
        section_text = p.replace("[start_of_segment]", "").replace(
            "[end_of_segment]", ""
        )

        if i == 0:
            continue

        if i == 1:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = (
                head_id
                + start_of_segment
                + mmtokenizer.tokenize(section_text)
                + [mmtokenizer.soa]
                + codectool.sep_ids
            )
            guidance_scale = guidance_scale_first
        else:
            prompt_ids = (
                end_of_segment
                + start_of_segment
                + mmtokenizer.tokenize(section_text)
                + [mmtokenizer.soa]
                + codectool.sep_ids
            )
            guidance_scale = guidance_scale_rest

        prompt_ids_t = torch.as_tensor(prompt_ids).unsqueeze(0)
        cond_input = (
            torch.cat([raw_output, prompt_ids_t], dim=1) if i > 1 else prompt_ids_t
        )

        max_ctx = S1_SEQ_LEN - max_new_tokens - 1
        if cond_input.shape[-1] > max_ctx:
            print(
                f"Section {i}: truncating {cond_input.shape[-1]} -> {max_ctx}",
                flush=True,
            )
            cond_input = cond_input[:, -max_ctx:]

        cond_len = cond_input.shape[1]

        print(
            f"Section {i}: input_len={cond_len}, max_new_tokens={max_new_tokens}",
            flush=True,
        )

        if use_cfg:
            # -- CFG path: custom generation loop --
            uncond_input = torch.full(
                (1, cond_len), pad_token_id, dtype=cond_input.dtype
            )
            uncond_input[0, 0] = cond_input[0, -1]

            cond_mask = torch.ones(1, cond_len, dtype=torch.long)
            uncond_mask = torch.zeros(1, cond_len, dtype=torch.long)
            uncond_mask[0, 0] = 1
            attention_mask = torch.cat([cond_mask, uncond_mask], dim=0)

            block_processors = LogitsProcessorList(
                [
                    BlockTokenRangeProcessor(0, 32002),
                    BlockTokenRangeProcessor(32016, 32017),
                ]
            )

            t0 = time.time()
            output_seq = cfg_generate(
                neuron_model=model,
                cond_ids=cond_input,
                uncond_ids=uncond_input,
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=pad_token_id,
                logits_processor=block_processors,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.1,
            )
        else:
            # -- Non-CFG path: standard HF generate --
            attention_mask = torch.ones_like(cond_input)
            t0 = time.time()
            with torch.no_grad():
                output_seq = hf_model.generate(
                    input_ids=cond_input,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.5,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=LogitsProcessorList(
                        [
                            BlockTokenRangeProcessor(0, 32002),
                            BlockTokenRangeProcessor(32016, 32017),
                        ]
                    ),
                )

        new_tok = output_seq.shape[1] - cond_len
        elapsed = time.time() - t0
        print(
            f"Section {i}: {new_tok} tokens in {elapsed:.1f}s "
            f"({new_tok / max(elapsed, 0.001):.1f} tok/s)",
            flush=True,
        )

        if output_seq[0][-1].item() != mmtokenizer.eoa:
            output_seq = torch.cat(
                (output_seq, torch.as_tensor([[mmtokenizer.eoa]])), dim=1
            )

        if i > 1:
            raw_output = torch.cat(
                [raw_output, prompt_ids_t, output_seq[:, cond_len:]], dim=1
            )
        else:
            raw_output = output_seq

    gen_time = time.time() - gen_start

    # Extract vocal and instrumental tracks
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

    if len(soa_idx) != len(eoa_idx):
        raise ValueError(f"soa/eoa mismatch: {len(soa_idx)} vs {len(eoa_idx)}")

    vocals_list, inst_list = [], []
    for idx in range(len(soa_idx)):
        codec_ids = ids[soa_idx[idx] + 1 : eoa_idx[idx]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_list.append(
            codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        )
        inst_list.append(
            codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        )

    vocals = np.concatenate(vocals_list, axis=1)
    instrumentals = np.concatenate(inst_list, axis=1)

    # Save
    rid = args["random_id"]
    sdir = args["stage1_dir"]
    np.save(os.path.join(sdir, f"vocals_{rid}.npy"), vocals)
    np.save(os.path.join(sdir, f"instrumentals_{rid}.npy"), instrumentals)

    print(
        f"Stage 1 DONE: {gen_time:.1f}s, vocals={vocals.shape}, instrumentals={instrumentals.shape}",
        flush=True,
    )
    print(f"Audio duration: {vocals.shape[1] / 50:.1f}s", flush=True)


if __name__ == "__main__":
    main()
