#!/usr/bin/env python3
"""Minimal generate smoke test for MiMo-V2.5-Pro FP8 on Trn2.

Assumes the compiled NEFF already exists at MIMO_V25_PRO_COMPILED_PATH
(from smoke_compile_mimo_v2.py). Rebuilds the same MoENeuronConfig /
Flash wrapper, loads with skip_warmup=False, and generates 20 tokens for a
single prompt via HuggingFaceGenerationAdapter. Purpose: sanity-check that
the FP8 MoE + preprocessed scales actually produce coherent tokens.

Run under /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16.
"""

import os
import sys
import time
import traceback

# AWS Llama-3.1-405B FP8 tutorial env vars — these are RUNTIME flags that
# affect XLA's fp8 special-scalar handling. Safe to set at generate time;
# do NOT set at compile time (it changes HLO and busts the neuronx-cc cache).
os.environ.setdefault("XLA_HANDLE_SPECIAL_SCALAR", "1")
os.environ.setdefault("UNSAFE_FP8FNCAST", "1")

MODEL_PATH = os.environ.get(
    "MIMO_V25_PRO_MODEL_PATH",
    "/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8",
)
COMPILED_PATH = os.environ.get(
    "MIMO_V25_PRO_COMPILED_PATH",
    "/opt/dlami/nvme/compiled/mimo_v2_5_pro_bs48_moetp1_ep64_fp8moe_bf16attn_seq512/",
)

# Must match smoke_compile_mimo_v2.py exactly, else load() sees a
# mismatched NEFF.
TP_DEGREE = int(os.environ.get("TP_DEGREE", "64"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))  # must match smoke_compile
CTX_BATCH_SIZE = int(os.environ.get("CTX_BATCH_SIZE", "1"))
MOE_TP = int(os.environ.get("MOE_TP", "1"))
MOE_EP = int(os.environ.get("MOE_EP", "64"))

PROMPT = os.environ.get(
    "MIMO_V25_PRO_PROMPT",
    "Hello! Please introduce yourself in one sentence.",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "20"))

# Keep the per-compile BASE_COMPILE_WORK_DIR in sync with
# smoke_compile_mimo_v2.py so load() under the same COMPILED_PATH
# doesn't collide with a concurrent compile or reuse a stale workdir.
os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))),
)


def main():
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
        load_pretrained_config,
    )

    contrib_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "src",
    )
    sys.path.insert(0, os.path.abspath(contrib_src))

    from modeling_mimo_v2 import (
        MiMoV2InferenceConfig,
        NeuronMiMoV2ForCausalLM,
    )

    print(f"[gen] MODEL_PATH={MODEL_PATH}")
    print(f"[gen] COMPILED_PATH={COMPILED_PATH}")
    print(f"[gen] TP={TP_DEGREE}, SEQ={SEQ_LEN}, BS={BATCH_SIZE}")

    # Outer ep_degree must match the compile-time value (kept at 1 so
    # world_size = tp_degree; see smoke_compile_mimo_v2.py comment).
    neuron_config = MoENeuronConfig(
        tp_degree=TP_DEGREE,
        ep_degree=1,
        logical_nc_config=2,
        batch_size=BATCH_SIZE,
        max_batch_size=BATCH_SIZE,
        ctx_batch_size=CTX_BATCH_SIZE,
        tkg_batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        n_active_tokens=128,
        torch_dtype="bfloat16",
        capacity_factor=1.0,
        glu_mlp=True,
        moe_ep_degree=MOE_EP,
        moe_tp_degree=MOE_TP,
        context_encoding_buckets=[SEQ_LEN],
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config={
            "use_shard_on_block_dynamic_while": True,
            "block_sharding_strategy": "PING_PONG",
        },
        save_sharded_checkpoint=True,
        quantized=True,
        quantized_checkpoints_path=MODEL_PATH,
        quantization_dtype="f8e4m3",
        quantization_type="blockwise_symmetric",
        quantization_block_axis=[1, 2],
        quantization_block_size=[128, 128],
        modules_to_not_convert=[
            "embed_tokens",
            "lm_head",
            "norm",
            "router",
            "o_proj",
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    )

    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_config)
    )

    print("[gen] Instantiating model...")
    t0 = time.time()
    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    print(f"[gen] Instantiated in {time.time() - t0:.1f}s")

    # skip_warmup=False so generate() hits a primed graph (the warmup forward
    # allocates the shared scratchpad the generation path needs).
    print(f"[gen] Loading from {COMPILED_PATH} (skip_warmup=False)")
    t0 = time.time()
    model.load(COMPILED_PATH, skip_warmup=False)
    print(f"[gen] Loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Decoder-only LM requires left-padding so the last token of each batch
    # slot is the real prompt ending, not a pad token. Default HF tokenizer
    # padding_side is 'right' which silently corrupts batched prefill.
    tokenizer.padding_side = "left"
    adapter = HuggingFaceGenerationAdapter(model)

    # When CHAT_TEMPLATE=1, wrap the raw prompt in the checkpoint's chat
    # template (system + user turns with <|im_start|>/<|im_end|> markers and
    # trailing assistant cue). Matches how vllm /v1/chat/completions prepares
    # inputs. Without this, the model free-continues the prompt as raw text
    # instead of answering it.
    use_chat_template = os.environ.get("CHAT_TEMPLATE", "0") == "1"
    minimal_chat = os.environ.get("MINIMAL_CHAT", "0") == "1"
    if minimal_chat:
        # Skip the Pro default system prompt entirely; wrap prompt in bare
        # <|im_start|>user ... <|im_end|><|im_start|>assistant\n framing.
        templated = (
            f"<|im_start|>user\n{PROMPT}<|im_end|>"
            f"<|im_start|>assistant\n"
        )
        print(f"[gen] minimal-chat prompt ({len(templated)} chars, no system)")
        inputs = tokenizer(
            [templated] * BATCH_SIZE,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    elif use_chat_template:
        system = os.environ.get("CHAT_SYSTEM", "")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": PROMPT})
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        print(f"[gen] chat-templated prompt ({len(templated)} chars)")
        inputs = tokenizer([templated] * BATCH_SIZE, return_tensors="pt", padding=True)
    else:
        inputs = tokenizer([PROMPT] * BATCH_SIZE, return_tensors="pt", padding=True)
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
    )

    # DUMP_LOGITS=1 -> request scores so we can see top-k per step.
    dump_logits = os.environ.get("DUMP_LOGITS", "0") == "1"
    if dump_logits:
        gen_config.output_scores = True
        gen_config.return_dict_in_generate = True

    print(f"[gen] prompt: {PROMPT!r}")
    print(f"[gen] input_ids.shape={tuple(inputs['input_ids'].shape)}")
    t0 = time.time()
    output = adapter.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=gen_config,
    )
    dt = time.time() - t0

    if dump_logits and hasattr(output, "sequences"):
        output_ids = output.sequences
        scores = output.scores  # tuple of [bs, vocab] per step
    else:
        output_ids = output
        scores = None

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"[gen] generated {new_tokens.numel()} tokens in {dt:.2f}s "
          f"({new_tokens.numel() / dt:.2f} tok/s)")
    print(f"[gen] new token ids: {new_tokens.tolist()}")
    print(f"[gen] new text     : {decoded!r}")
    print(f"[gen] full text    : {full!r}")

    if scores is not None:
        import torch as _t
        print("[gen] === top-5 per decode step (batch slot 0) ===")
        for step, step_logits in enumerate(scores):
            lp = _t.log_softmax(step_logits[0].float(), dim=-1)
            top_lp, top_id = _t.topk(lp, 5)
            parts = []
            for l, i in zip(top_lp.tolist(), top_id.tolist()):
                tok = tokenizer.decode([i]).replace("\n", "\\n")
                parts.append(f"({tok!r}:{i}:{l:.2f})")
            chosen = new_tokens[step].item()
            print(f"  step {step:3d} chose id={chosen}  top5={' '.join(parts)}")

    print("[gen] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
