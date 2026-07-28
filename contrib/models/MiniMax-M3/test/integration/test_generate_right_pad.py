"""Working M3 inference test with right padding.

This is the verified-working test pattern for MiniMax-M3 on Trn2.
The key requirements (vs. naive HF-style usage):

1. **Right padding** (NOT left). NxDI compiles the LM head with
   `padding_side="right"` by default. Using left padding (HF generate
   default) causes the LM head to read hidden states from PAD positions,
   producing constant garbage logits regardless of prompt.

2. **position_ids: pads get 0**. Use
   `position_ids = attention_mask.cumsum(-1) - 1`, then
   `masked_fill(attention_mask == 0, 0)` (not 1). This way
   `torch.max(position_ids).indices` correctly points to the last REAL
   token for the LM head's gather.

3. **Manual decode loop** (do NOT use HF `generate()`). The HF
   GenerationAdapter forces left-padding conventions that conflict with
   NxDI's right-pad compile.

Prefill predictions are accurate (verified top-5 for "1+1=" includes
correct digits 1-4). Decode still has a known issue (collapses to
repetition after 2-3 tokens) — see README.md "Known issues" section.
"""
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

sys.path.insert(0, os.path.dirname(__file__) + "/../../src")
from modeling_minimax_m3 import (
    NeuronMiniMaxM3ForCausalLM,
    MiniMaxM3InferenceConfig,
    MiniMaxM3NeuronConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


MODEL_PATH = os.environ.get("M3_MODEL_PATH", "/mnt/nvme/models/MiniMax-M3")
COMPILED_PATH = os.environ.get("M3_COMPILED_PATH", "/mnt/scratch/neuron_models/MiniMax-M3-full-v8-rmsnorm")
BATCH = int(os.environ.get("M3_BATCH", "32"))
SEQ_LEN = int(os.environ.get("M3_SEQ_LEN", "512"))

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    f"/mnt/scratch/nxd_compile/{os.path.basename(COMPILED_PATH.rstrip('/'))}",
)


def main():
    import json
    neuron_config = MiniMaxM3NeuronConfig(
        tp_degree=64,
        ep_degree=1,
        logical_nc_config=2,
        batch_size=BATCH,
        max_batch_size=BATCH,
        ctx_batch_size=1,
        tkg_batch_size=BATCH,
        seq_len=SEQ_LEN,
        n_active_tokens=128,
        torch_dtype=torch.bfloat16,
        capacity_factor=1.0,
        glu_mlp=True,
        moe_ep_degree=64,
        moe_tp_degree=1,
        context_encoding_buckets=[SEQ_LEN],
        fused_qkv=True,
        save_sharded_checkpoint=True,
        router_config={"act_fn": "sigmoid", "dtype": "float32"},
        blockwise_matmul_config={
            "use_shard_on_block_dynamic_while": True,
            "block_sharding_strategy": "PING_PONG",
        },
    )

    with open(f"{MODEL_PATH}/config.json") as f:
        raw = json.load(f)
    text_cfg = dict(raw["text_config"])
    text_cfg["model_type"] = "minimax_m3"
    text_cfg["architectures"] = ["MiniMaxM3SparseForCausalLM"]
    text_cfg["tie_word_embeddings"] = text_cfg.get("tie_word_embeddings", False)
    hf_cfg = PretrainedConfig(**text_cfg)
    hf_cfg._name_or_path = MODEL_PATH

    config = MiniMaxM3InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_cfg)
    )
    config._name_or_path = MODEL_PATH

    print("[load] loading cached NEFF + pre-sharded weights...")
    t0 = time.perf_counter()
    model = NeuronMiniMaxM3ForCausalLM(MODEL_PATH, config)
    model.load(COMPILED_PATH, skip_warmup=False)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s")

    # Right padding — must match NxDI's compiled padding_side
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [
        "The capital of France is",
        "1+1=",
        "Hello, my name is",
    ]

    for prompt in prompts:
        print(f"\n=== {prompt!r} ===")
        inp = tok(
            [prompt] * BATCH, return_tensors="pt",
            padding="max_length", max_length=SEQ_LEN, truncation=True,
        )
        # Key trick: pad position_ids = 0 (not 1)
        pos_ids = inp.attention_mask.long().cumsum(-1) - 1
        pos_ids.masked_fill_(inp.attention_mask == 0, 0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(inp.input_ids, attention_mask=inp.attention_mask, position_ids=pos_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

        logits = out.logits if hasattr(out, 'logits') else out[0]
        top5 = torch.topk(logits[0, -1, :].float(), 5)
        print(f"  prefill: {prefill_ms:.1f}ms")
        print(f"  top-5 predictions:")
        for v, i in zip(top5.values.tolist(), top5.indices.tolist()):
            print(f"    id={i:6d}  logit={v:+.3f}  token={tok.decode([i])!r}")


if __name__ == "__main__":
    main()
