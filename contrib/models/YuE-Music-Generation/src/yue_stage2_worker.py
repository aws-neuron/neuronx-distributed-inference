#!/usr/bin/env python3
"""YuE Stage 2 Worker -- runs S2 (1B) teacher-forcing on Neuron in isolated process.

Called by yue_e2e_neuron.py orchestrator. Args passed via YUE_STAGE_ARGS env var.

Environment variables:
    MODEL_DIR: Root directory for models and compiled artifacts (default: /mnt/models)
    YUE_STAGE_ARGS: JSON-encoded arguments from orchestrator
"""

import os
import sys
import json
import time
import copy
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, LogitsProcessor, LogitsProcessorList

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/models")

sys.path.insert(0, os.path.join(MODEL_DIR, "YuE/inference"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer"))
sys.path.insert(0, os.path.join(MODEL_DIR, "xcodec_mini_infer/descriptaudiocodec"))

from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    prepare_sampling_params,
)
from mmtokenizer import _MMSentencePieceTokenizer
from codecmanipulator import CodecManipulator

# Paths (derived from MODEL_DIR)
S2_MODEL_PATH = os.path.join(MODEL_DIR, "YuE-s2-1B-general")
S2_COMPILED_PATH = os.path.join(MODEL_DIR, "compiled/s2_tp1")
S2_COMPILED_PATH_NKI = os.path.join(MODEL_DIR, "compiled/s2_tp1_nki")
TOKENIZER_PATH = os.path.join(
    MODEL_DIR, "YuE/inference/mm_tokenizer_v0.2_hf/tokenizer.model"
)

S2_TP_DEGREE = 1
S2_SEQ_LEN = 4096
S2_MAX_CONTEXT = 4096 - 8
S2_MAX_NEW_TOKENS = 8


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def stage2_generate_chunk_kvcache(neuron_model, mmtokenizer, codectool, prompt_npy):
    """KV-cache-aware teacher-forcing on a single 6s chunk (300 tokens).
    Kept as fallback for batch_size=1 compilation."""
    return stage2_generate_batched_kvcache(
        neuron_model, mmtokenizer, codectool, [prompt_npy]
    )[0]


def stage2_generate_batched_kvcache(neuron_model, mmtokenizer, codectool, chunk_list):
    """KV-cache-aware teacher-forcing on multiple chunks simultaneously.

    All chunks are batched into a single forward pass. Each chunk has the same
    prompt structure (soa + stage_1 + 300_codec_ids + stage_2) so they naturally
    align in sequence length.

    Args:
        chunk_list: list of np arrays, each shape (1, n_frames) with codec IDs

    Returns:
        list of np arrays, one per chunk, containing generated token IDs
    """
    batch_size = len(chunk_list)

    # Prepare all chunks
    all_codec_ids = []
    all_prompt_ids = []
    for chunk_npy in chunk_list:
        codec_ids = codectool.unflatten(chunk_npy, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
            codec_ids,
            global_offset=codectool.global_offset,
            codebook_size=codectool.codebook_size,
            num_codebooks=codectool.num_codebooks,
        ).astype(np.int32)
        all_codec_ids.append(codec_ids)

        prompt_ids = np.concatenate(
            [
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),
                np.array([mmtokenizer.stage_2]),
            ]
        ).astype(np.int32)
        all_prompt_ids.append(prompt_ids)

    # Stack into batch tensors
    prompt_ids_batch = torch.tensor(
        np.stack(all_prompt_ids, axis=0), dtype=torch.long
    )  # (B, prompt_len)
    codec_ids_batch = torch.tensor(
        np.concatenate(all_codec_ids, axis=0), dtype=torch.long
    )  # (B, n_frames)

    n_frames = codec_ids_batch.shape[1]
    sampling_params = prepare_sampling_params(batch_size=batch_size)

    # Token range for blocking invalid tokens (S2 codec range)
    block_range_low = 46358
    block_range_high = 53526

    # Step 1: Reset and context-encode all prompts simultaneously
    neuron_model.reset()

    prompt_len = prompt_ids_batch.shape[1]
    attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long)
    position_ids = (
        torch.arange(prompt_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )

    with torch.no_grad():
        outputs = neuron_model.forward(
            input_ids=prompt_ids_batch,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sampling_params=sampling_params,
        )

    cur_pos = prompt_len
    all_generated = [[] for _ in range(batch_size)]

    # Step 2: For each frame, force-feed codebook-0 tokens + generate 7 tokens
    for f_idx in range(n_frames):
        forced_ids = codec_ids_batch[:, f_idx].unsqueeze(1)  # (B, 1)

        attention_mask = torch.ones(batch_size, cur_pos + 1, dtype=torch.long)
        pos_ids = torch.full((batch_size, 1), cur_pos, dtype=torch.long)

        with torch.no_grad():
            outputs = neuron_model.forward(
                input_ids=forced_ids,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                sampling_params=sampling_params,
            )
        cur_pos += 1
        for b in range(batch_size):
            all_generated[b].append(codec_ids_batch[b, f_idx].item())

        # Generate 7 tokens autoregressively
        for t in range(7):
            logits = outputs.logits[:, -1, :]  # (B, vocab_size)

            logits[:, :block_range_low] = -float("inf")
            logits[:, block_range_high:] = -float("inf")

            next_tokens = logits.argmax(dim=-1)  # (B,)

            next_ids = next_tokens.unsqueeze(1)  # (B, 1)
            attention_mask = torch.ones(batch_size, cur_pos + 1, dtype=torch.long)
            pos_ids = torch.full((batch_size, 1), cur_pos, dtype=torch.long)

            with torch.no_grad():
                outputs = neuron_model.forward(
                    input_ids=next_ids,
                    attention_mask=attention_mask,
                    position_ids=pos_ids,
                    sampling_params=sampling_params,
                )
            cur_pos += 1
            for b in range(batch_size):
                all_generated[b].append(next_tokens[b].item())

    return [np.array(gen) for gen in all_generated]


def stage2_generate_chunk(hf_model, mmtokenizer, codectool, prompt_npy):
    """Teacher-forcing on a single 6s chunk (300 tokens). Returns refined token IDs.
    Legacy path: calls generate() per frame (resets KV cache each time)."""
    codec_ids = codectool.unflatten(prompt_npy, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)

    prompt_ids = np.concatenate(
        [
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),
            np.array([mmtokenizer.stage_2]),
        ]
    ).astype(np.int32)
    prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids_t = torch.as_tensor(codec_ids)
    prompt_ids_t = torch.as_tensor(prompt_ids)
    len_prompt = prompt_ids_t.shape[-1]

    block_list = LogitsProcessorList(
        [
            BlockTokenRangeProcessor(0, 46358),
            BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size),
        ]
    )

    for f_idx in range(codec_ids_t.shape[1]):
        cb0 = codec_ids_t[:, f_idx : f_idx + 1]
        prompt_ids_t = torch.cat([prompt_ids_t, cb0], dim=1)
        attn = torch.ones_like(prompt_ids_t)

        with torch.no_grad():
            out = hf_model.generate(
                input_ids=prompt_ids_t,
                attention_mask=attn,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )
        assert out.shape[1] - prompt_ids_t.shape[1] == 7, (
            f"Expected 7 new tokens, got {out.shape[1] - prompt_ids_t.shape[1]}"
        )
        prompt_ids_t = out

    return prompt_ids_t[0].cpu().numpy()[len_prompt:]


def process_tracks_batched(
    neuron_model,
    mmtokenizer,
    codectool,
    codectool_s2,
    vocals_npy,
    inst_npy,
    batch_size,
):
    """Process vocal and instrumental tracks with batched S2 teacher-forcing.

    Collects all chunks from both tracks, batches them together (up to batch_size
    chunks per forward pass), and processes them simultaneously.

    Returns: (vocals_refined, instrumentals_refined) as numpy arrays
    """

    # Split both tracks into 6s chunks
    def get_chunks(track_npy):
        prompt = track_npy.astype(np.int32)
        dur = prompt.shape[-1] // 50 // 6 * 6
        n_chunks = dur // 6
        chunks = [prompt[:, ci * 300 : (ci + 1) * 300] for ci in range(n_chunks)]
        remainder = None
        if dur * 50 != prompt.shape[-1]:
            remainder = prompt[:, dur * 50 :]
        return chunks, remainder

    vocal_chunks, vocal_remainder = get_chunks(vocals_npy)
    inst_chunks, inst_remainder = get_chunks(inst_npy)

    # Combine all chunks: vocals first, then instrumentals
    all_chunks = vocal_chunks + inst_chunks
    n_vocal = len(vocal_chunks)
    n_inst = len(inst_chunks)
    total_chunks = len(all_chunks)

    print(
        f"S2 batched: {n_vocal} vocal + {n_inst} inst = {total_chunks} chunks, "
        f"batch_size={batch_size}",
        flush=True,
    )

    # Process in batches of batch_size
    all_results = []
    for batch_start in tqdm(range(0, total_chunks, batch_size), desc="S2 batched"):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_chunks = all_chunks[batch_start:batch_end]

        # Pad last batch to match compiled batch_size
        actual_count = len(batch_chunks)
        if actual_count < batch_size:
            batch_chunks = batch_chunks + [batch_chunks[-1]] * (
                batch_size - actual_count
            )

        results = stage2_generate_batched_kvcache(
            neuron_model, mmtokenizer, codectool, batch_chunks
        )
        all_results.extend(results[:actual_count])

    # Split back into vocal and instrumental
    vocal_segments = all_results[:n_vocal]
    inst_segments = all_results[n_vocal : n_vocal + n_inst]

    # Concatenate segments
    vocal_output = np.concatenate(vocal_segments, axis=0)
    inst_output = np.concatenate(inst_segments, axis=0)

    # Handle remainders (process individually with batch padding)
    if vocal_remainder is not None:
        batch_chunks = [vocal_remainder]
        if batch_size > 1:
            batch_chunks = batch_chunks + [vocal_remainder] * (batch_size - 1)
        rem_results = stage2_generate_batched_kvcache(
            neuron_model, mmtokenizer, codectool, batch_chunks
        )
        vocal_output = np.concatenate([vocal_output, rem_results[0]], axis=0)

    if inst_remainder is not None:
        batch_chunks = [inst_remainder]
        if batch_size > 1:
            batch_chunks = batch_chunks + [inst_remainder] * (batch_size - 1)
        rem_results = stage2_generate_batched_kvcache(
            neuron_model, mmtokenizer, codectool, batch_chunks
        )
        inst_output = np.concatenate([inst_output, rem_results[0]], axis=0)

    # Convert to multi-codebook format and fix invalid codes
    def finalize(output):
        output = codectool_s2.ids2npy(output)
        fixed = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, el in enumerate(line):
                if el < 0 or el > 1023:
                    ctr = Counter(line)
                    fixed[i, j] = sorted(ctr.items(), key=lambda x: x[1], reverse=True)[
                        0
                    ][0]
        return fixed

    return finalize(vocal_output), finalize(inst_output)


def process_track(
    hf_model,
    neuron_model,
    mmtokenizer,
    codectool,
    codectool_s2,
    track_npy,
    label,
    use_kv_cache=True,
):
    """Run Stage 2 on a single track. Legacy path for non-batched processing."""
    prompt = track_npy.astype(np.int32)
    dur = prompt.shape[-1] // 50 // 6 * 6  # 6s chunks only
    n_chunks = dur // 6

    mode = "KV-cache" if use_kv_cache else "legacy"
    print(
        f"S2 {label}: {n_chunks} chunks ({prompt.shape[-1] / 50:.1f}s) [{mode}]",
        flush=True,
    )

    segments = []
    for ci in tqdm(range(n_chunks), desc=f"S2 {label}"):
        chunk = prompt[:, ci * 300 : (ci + 1) * 300]
        if use_kv_cache:
            seg = stage2_generate_chunk_kvcache(
                neuron_model, mmtokenizer, codectool, chunk
            )
        else:
            seg = stage2_generate_chunk(hf_model, mmtokenizer, codectool, chunk)
        segments.append(seg)

    output = np.concatenate(segments, axis=0)

    # Process remainder if not 6s-aligned
    if dur * 50 != prompt.shape[-1]:
        remainder = prompt[:, dur * 50 :]
        if use_kv_cache:
            ending = stage2_generate_chunk_kvcache(
                neuron_model, mmtokenizer, codectool, remainder
            )
        else:
            ending = stage2_generate_chunk(hf_model, mmtokenizer, codectool, remainder)
        output = np.concatenate([output, ending], axis=0)

    # Convert to multi-codebook format
    output = codectool_s2.ids2npy(output)

    # Fix invalid codes
    fixed = copy.deepcopy(output)
    for i, line in enumerate(output):
        for j, el in enumerate(line):
            if el < 0 or el > 1023:
                ctr = Counter(line)
                fixed[i, j] = sorted(ctr.items(), key=lambda x: x[1], reverse=True)[0][
                    0
                ]

    return fixed


def main():
    args = json.loads(os.environ["YUE_STAGE_ARGS"])
    use_nki = args.get("nki_mlp", False)
    s2_batch_size = args.get("s2_batch_size", 1)

    # Create neuron config
    hf_config = AutoConfig.from_pretrained(S2_MODEL_PATH)
    neuron_cfg_inner = {
        "tp_degree": S2_TP_DEGREE,
        "batch_size": s2_batch_size,
        "seq_len": S2_SEQ_LEN,
        "max_context_length": S2_MAX_CONTEXT,
        "max_new_tokens": S2_MAX_NEW_TOKENS,
        "torch_dtype": "bfloat16",
        "start_rank_id": 0,
    }
    if s2_batch_size > 1:
        neuron_cfg_inner["padding_side"] = "right"
    if use_nki:
        neuron_cfg_inner["mlp_kernel_enabled"] = True
        neuron_cfg_inner["mlp_tkg_nki_kernel_enabled"] = True
        print(
            "NKI TKG kernels ENABLED for S2 (MLP only)",
            flush=True,
        )

    neuron_cfg = {
        "neuron_config": neuron_cfg_inner,
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
    with open(os.path.join(S2_MODEL_PATH, "neuron_config.json"), "w") as f:
        json.dump(neuron_cfg, f, indent=2)

    # Choose compiled path
    base_path = S2_COMPILED_PATH_NKI if use_nki else S2_COMPILED_PATH
    if s2_batch_size > 1:
        compiled_path = f"{base_path}_bs{s2_batch_size}"
    else:
        compiled_path = base_path

    # Load model
    model = NeuronLlamaForCausalLM(S2_MODEL_PATH)

    # Apply MLP monkeypatch: TKG NKI kernel for token gen, manual matmul for CTE.
    if use_nki:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nki_mlp_patch  # noqa: F401 -- auto-applies on import

        print("Applied nki_mlp_patch for S2", flush=True)

    skip_compile = args["skip_compile"]
    compiled_exists = os.path.exists(os.path.join(compiled_path, "model.pt"))

    if not skip_compile or not compiled_exists:
        os.makedirs(compiled_path, exist_ok=True)
        print(
            f"Compiling S2 (nki={use_nki}, batch_size={s2_batch_size})...", flush=True
        )
        t0 = time.time()
        model.compile(compiled_model_path=compiled_path)
        print(f"S2 compiled in {time.time() - t0:.1f}s", flush=True)

    print("Loading S2 to Neuron...", flush=True)
    t0 = time.time()
    model.load(compiled_path)
    print(f"S2 loaded in {time.time() - t0:.1f}s", flush=True)

    hf_model = HuggingFaceGenerationAdapter(model)

    # Tokenizer and tools
    mmtokenizer = _MMSentencePieceTokenizer(TOKENIZER_PATH)
    codectool = CodecManipulator("xcodec", 0, 1)
    codectool_s2 = CodecManipulator("xcodec", 0, 8)

    rid = args["random_id"]
    sdir1 = args["stage1_dir"]
    sdir2 = args["stage2_dir"]
    use_kv_cache = args.get("use_kv_cache", True)

    t_start = time.time()

    vocals = np.load(os.path.join(sdir1, f"vocals_{rid}.npy"))
    inst = np.load(os.path.join(sdir1, f"instrumentals_{rid}.npy"))

    if use_kv_cache and s2_batch_size > 1:
        # Batched path: process all chunks from both tracks simultaneously
        print(f"Using batched KV-cache path (batch_size={s2_batch_size})", flush=True)
        v_refined, i_refined = process_tracks_batched(
            model,
            mmtokenizer,
            codectool,
            codectool_s2,
            vocals,
            inst,
            s2_batch_size,
        )
    else:
        # Sequential path: process tracks one at a time
        print("Processing vocals...", flush=True)
        v_refined = process_track(
            hf_model,
            model,
            mmtokenizer,
            codectool,
            codectool_s2,
            vocals,
            "vocals",
            use_kv_cache=use_kv_cache,
        )
        print("Processing instrumentals...", flush=True)
        i_refined = process_track(
            hf_model,
            model,
            mmtokenizer,
            codectool,
            codectool_s2,
            inst,
            "instrumentals",
            use_kv_cache=use_kv_cache,
        )

    np.save(os.path.join(sdir2, f"vocals_{rid}.npy"), v_refined)
    np.save(os.path.join(sdir2, f"instrumentals_{rid}.npy"), i_refined)

    s2_time = time.time() - t_start
    print(f"Stage 2 DONE: {s2_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
