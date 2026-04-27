#!/usr/bin/env python3
"""Minimal compile+load smoke test for MiMo-V2.5-Pro FP8 on Trn2.

Bypasses vLLM entirely so we can iterate on the preprocessed Neuron-FP8
checkpoint without paying vllm-neuron's startup cost. Compiles the model
(TP=64, configurable MoE TP/EP, blockwise FP8 for routed experts) to a
temp dir, then loads. For `moe_ep_degree > 1` the TKG path raises
`NotImplementedError: Selective Loading with Expert parallelism` unless
`batch_size * top_k / num_experts >= 1.0` → `batch_size >= 384 / 8 = 48`.

STAGE controls how far we go:
  instantiate | compile | load | all   (default: all)

DRY_RUN=1 does HLO-only compile (no torch.jit.save + shard). Fastest sanity
check for the preprocessed checkpoint. SKIP_WARMUP=1 on load() skips the
forward pass that allocates the shared scratchpad — useful when HBM is
tight.

Run under /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 (same venv used
by the bench script).
"""

import os
import sys
import time
import traceback

MODEL_PATH = os.environ.get(
    "MIMO_V25_PRO_MODEL_PATH",
    "/opt/dlami/nvme/models/MiMo-V2.5-Pro-Neuron-FP8",
)
COMPILED_PATH = os.environ.get(
    "MIMO_V25_PRO_COMPILED_PATH",
    "/opt/dlami/nvme/compiled/mimo_v25_pro_moetp1_ep64_bs48/",
)

TP_DEGREE = int(os.environ.get("TP_DEGREE", "64"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
# BS=48 is the minimum that avoids forward_selective_loading on decode:
# `BS * top_k / num_experts >= 1.0` → BS >= 384/8 = 48. At BS=1 the TKG
# path raises `NotImplementedError: Selective Loading with Expert parallelism`.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))
CTX_BATCH_SIZE = int(os.environ.get("CTX_BATCH_SIZE", "1"))
# moe_tp=1 / moe_ep=64: first recipe to try on V2.5-Pro. Lowest compile time
# (no intra-expert TP split) and output quality should be comparable to
# Flash, which uses the same recipe. On V2-Pro this produced garbage
# prefill ("0.0.0.0:8080"), but we're re-testing on V2.5-Pro because the
# V2-Pro root cause ended up being FP8 expert-MLP precision loss, which
# V2.5 may or may not inherit.
MOE_TP = int(os.environ.get("MOE_TP", "1"))
MOE_EP = int(os.environ.get("MOE_EP", "64"))

STAGE = os.environ.get("STAGE", "all").lower()

os.makedirs(COMPILED_PATH, exist_ok=True)

# NxDI's model builder uses a per-process temp workdir for HLO/NEFF staging
# (BASE_COMPILE_WORK_DIR, default "/tmp/nxd_model/"). If two compiles run in
# parallel with the same default, they silently overwrite each other's
# .hlo_module.pb files and one or both compilations crash with
# "neuronx-cc returned non-zero exit status 70". Pin the workdir to a
# unique per-COMPILED_PATH subdir to stay safe under any parallel invocation.
os.environ.setdefault(
    "BASE_COMPILE_WORK_DIR",
    os.path.join("/tmp/nxd_model", os.path.basename(COMPILED_PATH.rstrip("/"))),
)


def main():
    from neuronx_distributed_inference.models.config import MoENeuronConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    # Import the contrib wrapper (sibling src dir).
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

    print(f"[smoke] MODEL_PATH={MODEL_PATH}")
    print(f"[smoke] COMPILED_PATH={COMPILED_PATH}")
    print(f"[smoke] TP_DEGREE={TP_DEGREE}, SEQ_LEN={SEQ_LEN}, BS={BATCH_SIZE}")
    print(f"[smoke] MOE_TP={MOE_TP}, MOE_EP={MOE_EP}")
    print(f"[smoke] STAGE={STAGE}")

    print("[smoke] Building MoENeuronConfig (quantized FP8 MoE, blockwise_symmetric)...")
    # NOTE: ep_degree at the top level controls the OUTER (full model)
    # expert-parallel factor, which multiplies world_size to
    # tp_degree * ep_degree and duplicates non-MoE weights per replica.
    # At world_size > 64 on a 64-NC Trn2, sharded weights grow accordingly
    # (e.g. tp=64 + ep=4 -> 256 ranks -> 4x the sharded checkpoint size,
    # and at runtime the model doesn't fit on the device). For MoE-only
    # EP we want ep_degree=1 at the outer level and the per-MoE split
    # controlled solely by moe_ep_degree. Keep ep_degree=1 unconditionally.
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
        # SDK 2.29 ships only bwmm_shard_on_block / bwmm_shard_on_intermediate;
        # default routes to _call_shard_hidden_kernel which is missing, so we
        # take the shard-on-block path via this flag. Matches Flash + Kimi.
        blockwise_matmul_config={
            "use_shard_on_block_dynamic_while": True,
            "block_sharding_strategy": "PING_PONG",
        },
        # Persist sharded FP8 weights to disk so subsequent load()s skip the
        # ~10-minute shard_checkpoint step (writes weights/tp{0..63}_*.safetensors
        # on NVMe; NxDI load() reads these directly when present).
        save_sharded_checkpoint=True,
        # FP8 blockwise for routed experts (Kimi-K2 recipe).
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
        ],
    )

    print("[smoke] Building MiMoV2InferenceConfig...")
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = MiMoV2InferenceConfig(
        neuron_config, load_config=load_pretrained_config(hf_config=hf_config)
    )
    print(f"[smoke] config.hidden_size={config.hidden_size}")
    print(f"[smoke] config.num_hidden_layers={config.num_hidden_layers}")
    print(f"[smoke] config.n_routed_experts={config.n_routed_experts}")
    print(f"[smoke] config.num_experts_per_tok={config.num_experts_per_tok}")
    print(f"[smoke] config.layer_uses_moe[:5]={config.layer_uses_moe[:5]}")
    print(f"[smoke] config.layer_attention_types[:5]={config.layer_attention_types[:5]}")

    print("[smoke] Instantiating NeuronMiMoV2ForCausalLM (build model-on-cpu)...")
    t0 = time.time()
    model = NeuronMiMoV2ForCausalLM(MODEL_PATH, config)
    print(f"[smoke] Instantiated in {time.time() - t0:.1f}s")

    if STAGE == "instantiate":
        print("[smoke] STAGE=instantiate only, skipping compile/load.")
        return

    DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
    if STAGE in ("compile", "all"):
        label = "Dry-run compile (HLO only)" if DRY_RUN else "Full compile"
        print(f"[smoke] {label} -> {COMPILED_PATH}")
        t0 = time.time()
        try:
            model.compile(COMPILED_PATH, dry_run=DRY_RUN)
            print(f"[smoke] {label} OK in {time.time() - t0:.1f}s")
        except Exception:
            print(f"[smoke] {label} FAILED:")
            traceback.print_exc()
            raise

    if STAGE in ("load", "all") and not DRY_RUN:
        SKIP_WARMUP = os.environ.get("SKIP_WARMUP", "1") == "1"
        print(f"[smoke] Loading compiled model from {COMPILED_PATH} (skip_warmup={SKIP_WARMUP})")
        t0 = time.time()
        model.load(COMPILED_PATH, skip_warmup=SKIP_WARMUP)
        print(f"[smoke] Loaded in {time.time() - t0:.1f}s")

    print("[smoke] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
