#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
COMPILED_ARTIFACTS=""
MAX_MODEL_LEN="512"
SEQ_LEN="512"
CTE_BUCKET="512"
CTE_BUCKETS=""
CTE_BUCKET_PROFILE="single"
CONTEXT_ENCODING_BUCKET_PAIRS=""
TP_DEGREE="4"
LNC="2"
MAX_NUM_SEQS="1"
CTX_BATCH_SIZE="1"
TOKEN_GENERATION_BUCKETS=""
TOKEN_GENERATION_BATCHES=""
ASYNC_MODE="0"
PORT="8000"
HOST="0.0.0.0"
ENABLE_CHUNKED_PREFILL="0"
ENABLE_PREFIX_CACHING="0"
ENABLE_HYBRID_APC="0"
MAMBA_CACHE_MODE=""
MAMBA_CACHE_DTYPE=""
MAMBA_SSM_CACHE_DTYPE=""
BLOCK_SIZE=""
GDN_CHECKPOINT_INTERVAL="256"
MAX_GDN_CHECKPOINT_SLOTS="8"
GDN_RECURRENT_CACHE_DTYPE="float32"
GDN_CONV_CACHE_DTYPE="bfloat16"
HYBRID_GDN_RECURRENT_CACHE_DTYPE=""
HYBRID_GDN_CONV_CACHE_DTYPE=""
HYBRID_CACHE_MODE="all"
HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"
HYBRID_CACHE_VALIDATE_EXACT="0"
HYBRID_APC_REQUIRE_VLLM_METADATA="1"
HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="0"
HYBRID_APC_ENABLE_BACKED_PREFIX_READS="0"
HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE="0"
HYBRID_APC_PREFILL_CHUNK_TOKENS="0"
NUM_GPU_BLOCKS_OVERRIDE=""
GPU_MEMORY_UTILIZATION=""
KV_CACHE_DTYPE=""
KV_CACHE_MEMORY_BYTES=""
KERNEL_Q_TILE_SIZE="128"
KERNEL_KV_TILE_SIZE="1024"
TEXT_ONLY_CTE="1"
COMPACT_CTE_ATTENTION_MASK="1"
COLD_ZERO_CONV_FAST_PATH="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --compiled-artifacts) COMPILED_ARTIFACTS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --cte-bucket) CTE_BUCKET="$2"; shift 2 ;;
    --cte-buckets) CTE_BUCKETS="$2"; shift 2 ;;
    --cte-bucket-profile) CTE_BUCKET_PROFILE="$2"; shift 2 ;;
    --context-encoding-bucket-pairs) CONTEXT_ENCODING_BUCKET_PAIRS="$2"; shift 2 ;;
    --tensor-parallel-size) TP_DEGREE="$2"; shift 2 ;;
    --logical-nc-config) LNC="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --ctx-batch-size) CTX_BATCH_SIZE="$2"; shift 2 ;;
    --token-generation-buckets) TOKEN_GENERATION_BUCKETS="$2"; shift 2 ;;
    --token-generation-batches) TOKEN_GENERATION_BATCHES="$2"; shift 2 ;;
    --async-mode) ASYNC_MODE="1"; shift ;;
    --no-async-mode) ASYNC_MODE="0"; shift ;;
    --enable-vllm-chunked-prefill) ENABLE_CHUNKED_PREFILL="1"; shift ;;
    --enable-prefix-caching) ENABLE_PREFIX_CACHING="1"; shift ;;
    --disable-prefix-caching|--no-enable-prefix-caching) ENABLE_PREFIX_CACHING="0"; shift ;;
    --enable-hybrid-apc) ENABLE_HYBRID_APC="1"; shift ;;
    --mamba-cache-mode) MAMBA_CACHE_MODE="$2"; shift 2 ;;
    --mamba-cache-dtype) MAMBA_CACHE_DTYPE="$2"; shift 2 ;;
    --mamba-ssm-cache-dtype) MAMBA_SSM_CACHE_DTYPE="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --gdn-checkpoint-interval) GDN_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
    --max-gdn-checkpoint-slots) MAX_GDN_CHECKPOINT_SLOTS="$2"; shift 2 ;;
    --gdn-recurrent-cache-dtype) GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --gdn-conv-cache-dtype) GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-recurrent-cache-dtype) HYBRID_GDN_RECURRENT_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-gdn-conv-cache-dtype) HYBRID_GDN_CONV_CACHE_DTYPE="$2"; shift 2 ;;
    --hybrid-cache-mode) HYBRID_CACHE_MODE="$2"; shift 2 ;;
    --hybrid-cache-prefix-boundary-only|--hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="1"; shift ;;
    --no-hybrid-cache-prefix-boundary-only|--no-hybrid-cache-block-boundary-only) HYBRID_CACHE_PREFIX_BOUNDARY_ONLY="0"; shift ;;
    --hybrid-cache-validate-exact) HYBRID_CACHE_VALIDATE_EXACT="1"; shift ;;
    --hybrid-apc-require-vllm-metadata) HYBRID_APC_REQUIRE_VLLM_METADATA="1"; shift ;;
    --no-hybrid-apc-require-vllm-metadata|--allow-hybrid-apc-local-hash-fallback) HYBRID_APC_REQUIRE_VLLM_METADATA="0"; shift ;;
    --hybrid-apc-disable-unbacked-prefix-reads) HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="1"; shift ;;
    --no-hybrid-apc-disable-unbacked-prefix-reads) HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="0"; shift ;;
    --hybrid-apc-enable-backed-prefix-reads) HYBRID_APC_ENABLE_BACKED_PREFIX_READS="1"; shift ;;
    --no-hybrid-apc-enable-backed-prefix-reads) HYBRID_APC_ENABLE_BACKED_PREFIX_READS="0"; shift ;;
    --hybrid-apc-allow-mixed-prefill-decode) HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE="1"; shift ;;
    --no-hybrid-apc-allow-mixed-prefill-decode) HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE="0"; shift ;;
    --hybrid-apc-prefill-chunk-tokens) HYBRID_APC_PREFILL_CHUNK_TOKENS="$2"; shift 2 ;;
    --num-gpu-blocks-override) NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
    --kv-cache-memory-bytes) KV_CACHE_MEMORY_BYTES="$2"; shift 2 ;;
    --kernel-q-tile-size) KERNEL_Q_TILE_SIZE="$2"; shift 2 ;;
    --kernel-kv-tile-size) KERNEL_KV_TILE_SIZE="$2"; shift 2 ;;
    --text-only-cte) TEXT_ONLY_CTE="1"; shift ;;
    --no-text-only-cte|--multimodal-cte) TEXT_ONLY_CTE="0"; shift ;;
    --compact-cte-attention-mask) COMPACT_CTE_ATTENTION_MASK="1"; shift ;;
    --no-compact-cte-attention-mask) COMPACT_CTE_ATTENTION_MASK="0"; shift ;;
    --cold-zero-conv-fast-path) COLD_ZERO_CONV_FAST_PATH="1"; shift ;;
    --no-cold-zero-conv-fast-path) COLD_ZERO_CONV_FAST_PATH="0"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: --model-path is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTRIB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONTRIB_ROOT}/../../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${CONTRIB_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export VLLM_PLUGINS="${VLLM_PLUGINS:-neuron}"

if [[ -n "${COMPILED_ARTIFACTS}" ]]; then
  export NEURON_COMPILED_ARTIFACTS="${COMPILED_ARTIFACTS}"
  export XLA_HANDLE_SPECIAL_SCALAR="${XLA_HANDLE_SPECIAL_SCALAR:-1}"
  export UNSAFE_FP8FNCAST="${UNSAFE_FP8FNCAST:-1}"
fi
if [[ -z "${BLOCK_SIZE}" ]]; then
  BLOCK_SIZE="128"
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export DISABLE_NEURON_CUSTOM_SCHEDULER="1"
fi
if [[ "${ENABLE_HYBRID_APC}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  export QWEN36_HYBRID_APC_INSTALL_PATCH="1"
fi
if [[ "${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}" == "1" ]]; then
  export QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS="1"
fi
if [[ "${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}" == "1" ]]; then
  export QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS="1"
fi
if [[ -z "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}" ]]; then
  HYBRID_GDN_RECURRENT_CACHE_DTYPE="${GDN_RECURRENT_CACHE_DTYPE}"
fi
if [[ -z "${HYBRID_GDN_CONV_CACHE_DTYPE}" ]]; then
  HYBRID_GDN_CONV_CACHE_DTYPE="${GDN_CONV_CACHE_DTYPE}"
fi
case "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}" in
  fp32|float32|torch.float32)
    HYBRID_GDN_RECURRENT_CACHE_DTYPE="float32"
    ;;
  bf16|bfloat16|torch.bfloat16)
    if [[ "${ENABLE_HYBRID_APC}" == "1" && "${HYBRID_CACHE_MODE}" == "all" ]]; then
      echo "ERROR: Hybrid APC all-mode requires float32 recurrent GDN checkpoint cache state; use --gdn-recurrent-cache-dtype float32." >&2
      exit 2
    fi
    HYBRID_GDN_RECURRENT_CACHE_DTYPE="bfloat16"
    ;;
  *)
    echo "ERROR: unsupported --hybrid-gdn-recurrent-cache-dtype ${HYBRID_GDN_RECURRENT_CACHE_DTYPE}; expected float32 or bfloat16" >&2
    exit 2
    ;;
esac
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_HYBRID_APC}" == "1" ]]; then
  ENABLE_PREFIX_CACHING="1"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  if [[ -z "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    NUM_GPU_BLOCKS_OVERRIDE=$(( ((SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE) * MAX_NUM_SEQS ))
  fi
fi
if [[ "${ENABLE_HYBRID_APC}" == "1" ]]; then
  if [[ "${HYBRID_CACHE_MODE}" != "all" ]]; then
    echo "ERROR: --enable-hybrid-apc requires --hybrid-cache-mode all" >&2
    exit 2
  fi
  if [[ "${GDN_CHECKPOINT_INTERVAL}" != "${BLOCK_SIZE}" ]]; then
    echo "ERROR: --enable-hybrid-apc v0 requires --gdn-checkpoint-interval to equal --block-size" >&2
    exit 2
  fi
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_CACHE_MODE}" ]]; then
  MAMBA_CACHE_MODE="all"
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" && -z "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
  case "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}" in
    auto|float16|float32)
      MAMBA_SSM_CACHE_DTYPE="${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
      ;;
    *)
      MAMBA_SSM_CACHE_DTYPE="auto"
      ;;
  esac
fi
case "${MAMBA_SSM_CACHE_DTYPE}" in
  ""|auto|float16|float32)
    ;;
  bfloat16|bf16)
    echo "WARNING: vLLM --mamba-ssm-cache-dtype does not accept ${MAMBA_SSM_CACHE_DTYPE}; using auto while preserving hybrid GDN cache dtype in Neuron config." >&2
    MAMBA_SSM_CACHE_DTYPE="auto"
    ;;
  *)
    echo "ERROR: unsupported --mamba-ssm-cache-dtype ${MAMBA_SSM_CACHE_DTYPE}; expected auto, float16, or float32" >&2
    exit 2
    ;;
esac

CTE_BUCKETS_JSON="$(
  python3 - <<PY
import json

profiles = {
    "short": [128, 256, 512, 1024],
    "general": [256, 512, 1024, 2048],
    "long": [4096, 8192, 16384, 32768],
    "262k": [256],
}
profile = "${CTE_BUCKET_PROFILE}"
if profile != "single":
    if profile not in profiles:
        raise SystemExit(f"unknown --cte-bucket-profile: {profile}")
    buckets = profiles[profile]
else:
    raw = "${CTE_BUCKETS}".replace(",", " ").split()
    buckets = [int(x) for x in raw] if raw else [int("${CTE_BUCKET}")]
buckets = sorted(set(buckets))
if not buckets:
    raise SystemExit("at least one CTE bucket is required")
for bucket in buckets:
    if bucket <= 0:
        raise SystemExit(f"CTE buckets must be positive, got {bucket}")
    if bucket % 128 != 0:
        raise SystemExit(
            f"CTE bucket {bucket} is not 128-aligned; DeltaNet CTE uses 128-token chunks"
        )
if buckets[-1] > int("${SEQ_LEN}"):
    raise SystemExit(
        f"largest CTE bucket {buckets[-1]} exceeds --seq-len ${SEQ_LEN}"
    )
print(json.dumps(buckets))
PY
)"
MAX_CTE_BUCKET="$(
  python3 - <<PY
import json
print(json.loads('${CTE_BUCKETS_JSON}')[-1])
PY
)"
MAX_BATCHED_TOKENS="$(
  python3 - <<PY
import json

buckets = json.loads('${CTE_BUCKETS_JSON}')
max_bucket = buckets[-1]
checkpoint_interval = int("${GDN_CHECKPOINT_INTERVAL}")
if "${ENABLE_CHUNKED_PREFILL}" == "1" and "${ENABLE_HYBRID_APC}" == "1":
    requested_chunk = int("${HYBRID_APC_PREFILL_CHUNK_TOKENS}" or "0")
    if requested_chunk <= 0:
        candidates = [bucket for bucket in buckets if bucket % checkpoint_interval == 0]
        if not candidates:
            raise SystemExit(
                "--enable-hybrid-apc with chunked prefill requires at least one "
                "compiled CTE bucket that is a multiple of --gdn-checkpoint-interval "
                f"({checkpoint_interval}); got {buckets}"
            )
        print(candidates[-1])
    else:
        if requested_chunk % checkpoint_interval != 0:
            raise SystemExit(
                "--hybrid-apc-prefill-chunk-tokens must be a multiple of "
                f"--gdn-checkpoint-interval ({checkpoint_interval}), got {requested_chunk}"
            )
        if requested_chunk not in buckets:
            raise SystemExit(
                "--hybrid-apc-prefill-chunk-tokens must match a compiled CTE bucket, "
                f"got {requested_chunk} with buckets {buckets}"
            )
        print(min(max_bucket, requested_chunk))
else:
    print(max_bucket)
PY
)"

ADDITIONAL_CONFIG="$(
  python3 - <<PY
import json
import os
from pathlib import Path


def parse_int_list(name, raw):
    raw = raw.replace(",", " ").split()
    if not raw:
        return None
    values = sorted(set(int(item) for item in raw))
    for value in values:
        if value <= 0:
            raise SystemExit(f"{name} values must be positive, got {value}")
    return values


def parse_bucket_pairs(raw):
    tokens = raw.replace(",", " ").split()
    if not tokens:
        return None
    pairs = set()
    for token in tokens:
        if ":" in token:
            active, prefix = token.split(":", 1)
        elif "x" in token:
            active, prefix = token.split("x", 1)
        else:
            raise SystemExit(
                "CONTEXT_ENCODING_BUCKET_PAIRS entries must use "
                f"ACTIVE:PREFIX syntax, got {token!r}"
            )
        active_tokens = int(active)
        prefix_tokens = int(prefix)
        if active_tokens <= 0 or prefix_tokens < 0:
            raise SystemExit(
                "CONTEXT_ENCODING_BUCKET_PAIRS must use positive active "
                f"tokens and non-negative prefix tokens, got {token!r}"
            )
        pairs.add((active_tokens, prefix_tokens))
    return [[active, prefix] for active, prefix in sorted(pairs)]

enable_chunked = "${ENABLE_CHUNKED_PREFILL}" == "1"
enable_prefix_caching = "${ENABLE_PREFIX_CACHING}" == "1"
enable_hybrid_apc = "${ENABLE_HYBRID_APC}" == "1"
async_mode = "${ASYNC_MODE}" == "1"
cte_buckets = json.loads('${CTE_BUCKETS_JSON}')
context_encoding_bucket_pairs = parse_bucket_pairs("${CONTEXT_ENCODING_BUCKET_PAIRS}")
max_cte_bucket = cte_buckets[-1]
seq_len = int("${SEQ_LEN}")
max_num_seqs = int("${MAX_NUM_SEQS}")
token_generation_buckets = parse_int_list(
    "TOKEN_GENERATION_BUCKETS",
    "${TOKEN_GENERATION_BUCKETS}",
)
token_generation_batches = parse_int_list(
    "TOKEN_GENERATION_BATCHES",
    "${TOKEN_GENERATION_BATCHES}",
)
if token_generation_batches is not None and token_generation_batches[-1] > max_num_seqs:
    raise SystemExit(
        "TOKEN_GENERATION_BATCHES cannot contain values greater than "
        f"MAX_NUM_SEQS ({max_num_seqs})"
    )
compiled_artifacts = "${COMPILED_ARTIFACTS}"
compiled_max_prompt = 0
compiled_uses_prefix_caching = False
compiled_prefix_buckets = None
compiled_prefix_cte_attention_backend = None
compiled_prefix_cte_attention_segment_size = None
compiled_ctx_batch_size = 0
compiled_tkg_batch_size = 0
compiled_token_generation_buckets = None
compiled_token_generation_batches = None
compiled_kernel_flags = {}
compiled_decode_memory_flags = {}
compiled_weights_to_skip_layout_optimization = None
compiled_disable_token_generation_wlo = (
    os.environ.get("QWEN36_DISABLE_TOKEN_GENERATION_WLO") == "1"
)
if compiled_artifacts:
    config_path = Path(compiled_artifacts).expanduser() / "neuron_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            compiled_config = json.load(handle)
        compiled_disable_token_generation_wlo = (
            compiled_disable_token_generation_wlo
            or bool(compiled_config.get("disable_token_generation_wlo"))
        )
        nested_config = compiled_config.get("neuron_config")
        if isinstance(nested_config, dict):
            compiled_config = nested_config
            compiled_disable_token_generation_wlo = (
                compiled_disable_token_generation_wlo
                or bool(compiled_config.get("disable_token_generation_wlo"))
            )
        compiled_max_prompt = int(
            compiled_config.get("max_context_length")
            or compiled_config.get("max_length")
            or compiled_config.get("seq_len")
            or 0
        )
        if context_encoding_bucket_pairs is None:
            context_encoding_bucket_pairs = compiled_config.get(
                "context_encoding_bucket_pairs"
            )
        compiled_uses_prefix_caching = bool(
            compiled_config.get("is_prefix_caching")
        )
        compiled_prefix_buckets = compiled_config.get("prefix_buckets")
        compiled_prefix_cte_attention_backend = compiled_config.get(
            "prefix_cte_attention_backend"
        )
        compiled_prefix_cte_attention_segment_size = compiled_config.get(
            "prefix_cte_attention_segment_size"
        )
        compiled_ctx_batch_size = int(
            compiled_config.get("ctx_batch_size")
            or compiled_config.get("batch_size")
            or compiled_config.get("max_batch_size")
            or 0
        )
        compiled_tkg_batch_size = int(
            compiled_config.get("tkg_batch_size")
            or compiled_config.get("batch_size")
            or compiled_config.get("max_batch_size")
            or 0
        )
        compiled_token_generation_batches = compiled_config.get(
            "token_generation_batches"
        )
        compiled_token_generation_buckets = compiled_config.get(
            "token_generation_buckets"
        )
        compiled_weights_to_skip_layout_optimization = compiled_config.get(
            "weights_to_skip_layout_optimization"
        )
        for flag_name in (
            "fused_qkv",
            "qkv_kernel_enabled",
            "qkv_nki_kernel_enabled",
            "qkv_tkg_nki_kernel_enabled",
            "attn_block_tkg_nki_kernel_enabled",
            "attn_block_tkg_nki_kernel_cascaded_attention",
            "attn_block_tkg_nki_kernel_cache_update",
            "attn_block_tkg_nki_kernel_use_online_softmax",
            "attn_block_tkg_nki_kernel_disable_gpsimd_sb2sb",
            "out_proj_kernel_enabled",
            "mlp_kernel_enabled",
            "mlp_tkg_nki_kernel_enabled",
            "quantized_mlp_kernel_enabled",
            "rmsnorm_quantize_kernel_enabled",
            "quantize_clamp_bound",
        ):
            if flag_name in compiled_config:
                compiled_kernel_flags[flag_name] = compiled_config[flag_name]
        for flag_name in (
            "k_cache_transposed",
            "kv_cache_quant",
            "kv_quant_config",
            "quantized",
            "quantization_dtype",
            "quantization_type",
            "quantization_block_size",
            "quantization_block_axis",
            "quantization_scale_dtype",
            "quantized_checkpoints_path",
            "modules_to_not_convert",
            "draft_model_modules_to_not_convert",
            "activation_quantization_type",
        ):
            if flag_name in compiled_config:
                compiled_decode_memory_flags[flag_name] = compiled_config[flag_name]
runtime_max_prompt = compiled_max_prompt or max_cte_bucket
if compiled_artifacts and max_num_seqs > 1:
    if compiled_tkg_batch_size and max_num_seqs > compiled_tkg_batch_size:
        raise SystemExit(
            "compiled artifact cannot serve requested continuous batching: "
            f"MAX_NUM_SEQS={max_num_seqs} but compiled tkg_batch_size="
            f"{compiled_tkg_batch_size}"
        )
    if compiled_ctx_batch_size and int("${CTX_BATCH_SIZE}") > compiled_ctx_batch_size:
        raise SystemExit(
            "compiled artifact cannot serve requested CTE batch: "
            f"CTX_BATCH_SIZE=${CTX_BATCH_SIZE} but compiled ctx_batch_size="
            f"{compiled_ctx_batch_size}"
        )

def normalize_int_list(values):
    if values is None:
        return None
    if isinstance(values, str):
        return parse_int_list("compiled int list", values)
    normalized = sorted(set(int(value) for value in values))
    return normalized or None

if token_generation_batches is None:
    token_generation_batches = normalize_int_list(compiled_token_generation_batches)
if token_generation_buckets is None:
    token_generation_buckets = (
        normalize_int_list(compiled_token_generation_buckets) or [seq_len]
    )
if token_generation_buckets[-1] > seq_len:
    raise SystemExit(
        f"TOKEN_GENERATION_BUCKETS cannot contain values greater than SEQ_LEN ({seq_len})"
    )
if token_generation_batches is not None:
    token_generation_batches = [
        batch for batch in token_generation_batches if batch <= max_num_seqs
    ]
    if not token_generation_batches:
        token_generation_batches = None
num_gpu_blocks_override = "${NUM_GPU_BLOCKS_OVERRIDE}"
pa_num_blocks = (
    int(num_gpu_blocks_override)
    if num_gpu_blocks_override
    else max(
        1,
        ((int("${SEQ_LEN}") + int("${BLOCK_SIZE}") - 1) // int("${BLOCK_SIZE}"))
        * int("${MAX_NUM_SEQS}"),
    )
)
neuron_config = {
    "tp_degree": int("${TP_DEGREE}"),
    "batch_size": max_num_seqs,
    "ctx_batch_size": int("${CTX_BATCH_SIZE}"),
    "tkg_batch_size": max_num_seqs,
    "seq_len": seq_len,
    "max_length": seq_len,
    "max_context_length": runtime_max_prompt,
    "context_encoding_buckets": cte_buckets,
    "token_generation_buckets": token_generation_buckets,
    "enable_bucketing": len(cte_buckets) > 1 or len(token_generation_buckets) > 1,
    "logical_nc_config": int("${LNC}"),
    "torch_dtype": "bfloat16",
    "save_sharded_checkpoint": True,
    "pa_block_size": int("${BLOCK_SIZE}"),
    "pa_num_blocks": pa_num_blocks,
    "gdn_checkpoint_interval": int("${GDN_CHECKPOINT_INTERVAL}"),
    "max_gdn_checkpoint_slots": int("${MAX_GDN_CHECKPOINT_SLOTS}"),
    "gdn_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "gdn_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "hybrid_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_cache_mode": "${HYBRID_CACHE_MODE}",
}
if async_mode:
    neuron_config["async_mode"] = True
if token_generation_batches is not None:
    neuron_config["token_generation_batches"] = token_generation_batches
if compiled_weights_to_skip_layout_optimization is not None:
    neuron_config["weights_to_skip_layout_optimization"] = (
        compiled_weights_to_skip_layout_optimization
    )
neuron_config.update(compiled_kernel_flags)
neuron_config.update(compiled_decode_memory_flags)
if enable_prefix_caching or enable_hybrid_apc or enable_chunked:
    neuron_config["is_block_kv_layout"] = True
uses_prefix_cte_contract = (
    context_encoding_bucket_pairs is not None or compiled_uses_prefix_caching
)
if enable_prefix_caching or enable_hybrid_apc or uses_prefix_cte_contract:
    neuron_config["is_prefix_caching"] = True
    if context_encoding_bucket_pairs is not None:
        neuron_config["context_encoding_bucket_pairs"] = context_encoding_bucket_pairs
    if compiled_prefix_buckets is not None:
        neuron_config["prefix_buckets"] = compiled_prefix_buckets
    if compiled_prefix_cte_attention_backend is not None:
        neuron_config["prefix_cte_attention_backend"] = (
            compiled_prefix_cte_attention_backend
        )
    if compiled_prefix_cte_attention_segment_size is not None:
        neuron_config["prefix_cte_attention_segment_size"] = (
            compiled_prefix_cte_attention_segment_size
        )
# NeuronConfig.chunked_prefill_config trips the built-in block TKG attention
# kernel validation. Qwen Hybrid APC chunking still uses the top-level
# use_qwen_hybrid_chunked_prefill flags below.
if enable_chunked and not compiled_kernel_flags.get(
    "attn_block_tkg_nki_kernel_enabled",
    False,
):
    neuron_config.update({
        "chunked_prefill_config": {
            "max_num_seqs": int("${MAX_NUM_SEQS}"),
            "tkg_model_enabled": True,
            "kernel_q_tile_size": int("${KERNEL_Q_TILE_SIZE}"),
            "kernel_kv_tile_size": int("${KERNEL_KV_TILE_SIZE}"),
        },
    })
print(json.dumps({
    "max_prompt_length": runtime_max_prompt,
    "use_hybrid_apc_manager": enable_hybrid_apc,
    "use_text_only_cte_inputs": "${TEXT_ONLY_CTE}" == "1",
    "use_compact_cte_attention_mask": "${COMPACT_CTE_ATTENTION_MASK}" == "1",
    "use_cold_zero_conv_fast_path": "${COLD_ZERO_CONV_FAST_PATH}" == "1",
    "gdn_checkpoint_interval": int("${GDN_CHECKPOINT_INTERVAL}"),
    "max_gdn_checkpoint_slots": int("${MAX_GDN_CHECKPOINT_SLOTS}"),
    "gdn_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "gdn_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_recurrent_cache_dtype": "${HYBRID_GDN_RECURRENT_CACHE_DTYPE}",
    "hybrid_conv_cache_dtype": "${HYBRID_GDN_CONV_CACHE_DTYPE}",
    "hybrid_cache_mode": "${HYBRID_CACHE_MODE}",
    "hybrid_cache_prefix_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_block_boundary_only": "${HYBRID_CACHE_PREFIX_BOUNDARY_ONLY}" == "1",
    "hybrid_cache_validate_exact": "${HYBRID_CACHE_VALIDATE_EXACT}" == "1",
    "hybrid_apc_require_vllm_metadata": enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1",
    "hybrid_apc_allow_local_hash_fallback": not (enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1"),
    "hybrid_apc_require_attention_block_refs": enable_hybrid_apc and "${HYBRID_APC_REQUIRE_VLLM_METADATA}" == "1",
    "hybrid_apc_disable_unbacked_prefix_reads": enable_hybrid_apc and "${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}" == "1",
    "hybrid_apc_enable_backed_prefix_reads": enable_hybrid_apc and "${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}" == "1",
    "hybrid_apc_allow_mixed_prefill_decode": enable_hybrid_apc and "${HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE}" == "1",
    "hybrid_apc_prefill_chunk_tokens": int("${MAX_BATCHED_TOKENS}") if enable_hybrid_apc and enable_chunked else 0,
    "qwen_prefill_group_size": int("${MAX_BATCHED_TOKENS}") if enable_chunked else max_cte_bucket,
    "use_qwen_hybrid_chunked_prefill": enable_chunked,
    "use_qwen_hybrid_chunked_prefill_nki": enable_chunked,
    "disable_token_generation_wlo": compiled_disable_token_generation_wlo,
    "override_neuron_config": neuron_config,
}))
PY
)"

echo "Starting vLLM for Qwen3.6-27B"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NEURON_COMPILED_ARTIFACTS=${NEURON_COMPILED_ARTIFACTS:-}"
echo "XLA_HANDLE_SPECIAL_SCALAR=${XLA_HANDLE_SPECIAL_SCALAR:-}"
echo "UNSAFE_FP8FNCAST=${UNSAFE_FP8FNCAST:-}"
echo "QWEN36_DISABLE_TOKEN_GENERATION_WLO=${QWEN36_DISABLE_TOKEN_GENERATION_WLO:-}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
echo "ENABLE_HYBRID_APC=${ENABLE_HYBRID_APC}"
echo "MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}"
echo "MAMBA_CACHE_DTYPE=${MAMBA_CACHE_DTYPE:-}"
echo "MAMBA_SSM_CACHE_DTYPE=${MAMBA_SSM_CACHE_DTYPE:-}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "CTE_BUCKETS=${CTE_BUCKETS_JSON}"
echo "CONTEXT_ENCODING_BUCKET_PAIRS=${CONTEXT_ENCODING_BUCKET_PAIRS}"
echo "CTX_BATCH_SIZE=${CTX_BATCH_SIZE}"
echo "KERNEL_Q_TILE_SIZE=${KERNEL_Q_TILE_SIZE}"
echo "KERNEL_KV_TILE_SIZE=${KERNEL_KV_TILE_SIZE}"
echo "TEXT_ONLY_CTE=${TEXT_ONLY_CTE}"
echo "COMPACT_CTE_ATTENTION_MASK=${COMPACT_CTE_ATTENTION_MASK}"
echo "COLD_ZERO_CONV_FAST_PATH=${COLD_ZERO_CONV_FAST_PATH}"
echo "GDN_CHECKPOINT_INTERVAL=${GDN_CHECKPOINT_INTERVAL}"
echo "MAX_GDN_CHECKPOINT_SLOTS=${MAX_GDN_CHECKPOINT_SLOTS}"
echo "HYBRID_GDN_RECURRENT_CACHE_DTYPE=${HYBRID_GDN_RECURRENT_CACHE_DTYPE}"
echo "HYBRID_GDN_CONV_CACHE_DTYPE=${HYBRID_GDN_CONV_CACHE_DTYPE}"
echo "HYBRID_APC_REQUIRE_VLLM_METADATA=${HYBRID_APC_REQUIRE_VLLM_METADATA}"
echo "HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS=${HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS}"
echo "HYBRID_APC_ENABLE_BACKED_PREFIX_READS=${HYBRID_APC_ENABLE_BACKED_PREFIX_READS}"
echo "HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE=${HYBRID_APC_ALLOW_MIXED_PREFILL_DECODE}"
echo "HYBRID_APC_PREFILL_CHUNK_TOKENS=${HYBRID_APC_PREFILL_CHUNK_TOKENS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES}"
echo "ADDITIONAL_CONFIG=${ADDITIONAL_CONFIG}"

VLLM_ARGS=(
  "${MODEL_PATH}"
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "${TP_DEGREE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --generation-config vllm \
  --additional-config "${ADDITIONAL_CONFIG}"
)
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
else
  VLLM_ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "${MAMBA_CACHE_MODE}" ]]; then
  VLLM_ARGS+=(--mamba-cache-mode "${MAMBA_CACHE_MODE}")
fi
if [[ -n "${MAMBA_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--mamba-cache-dtype "${MAMBA_CACHE_DTYPE}")
fi
if [[ -n "${MAMBA_SSM_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE}")
fi
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  VLLM_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  VLLM_ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  VLLM_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  VLLM_ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi
if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_HYBRID_APC}" == "1" || "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(--block-size "${BLOCK_SIZE}")
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  VLLM_ARGS+=(
    --enable-chunked-prefill
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  )
else
  VLLM_ARGS+=(--no-enable-chunked-prefill)
fi

exec python "${SCRIPT_DIR}/serve_qwen36.py" "${VLLM_ARGS[@]}"
