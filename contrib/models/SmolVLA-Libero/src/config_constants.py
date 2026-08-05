"""
SmolVLA architecture constants.

All numbers extracted directly from the HuggingFaceVLA/smolvla_libero
checkpoint and config.json. Every other file in this port imports from here
— no hardcoded shapes anywhere else.

Source:
    HF model id : HuggingFaceVLA/smolvla_libero
    Backbone    : HuggingFaceTB/SmolVLM2-500M-Instruct (full 32-layer text model)
    expert_width_multiplier : 0.5  (expert hidden = 480)
    chunk size  : 50  (action prediction horizon)
    num steps   : 10  (Euler denoising steps, on CPU)
"""

# ---------------------------------------------------------------------------
# Vision encoder (SigLIP)
# ---------------------------------------------------------------------------

VISION_NUM_LAYERS         = 12
VISION_HIDDEN             = 768
VISION_INTERMEDIATE       = 3072
VISION_NUM_HEADS          = 12          # 768 / 64
VISION_HEAD_DIM           = 64
VISION_PATCH_SIZE         = 16
VISION_IMAGE_SIZE         = 512
VISION_NUM_PATCHES        = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_LAYER_NORM_EPS     = 1e-6

# Connector pixel-shuffle: scale_factor=4 → 4x4 spatial merge → 16x token reduction
PIXEL_SHUFFLE_SCALE       = 4
VISION_TOKENS_PER_IMAGE   = VISION_NUM_PATCHES // (PIXEL_SHUFFLE_SCALE ** 2)  # 64
CONNECTOR_INPUT_DIM       = VISION_HIDDEN * (PIXEL_SHUFFLE_SCALE ** 2)        # 12288

# ---------------------------------------------------------------------------
# VLM text backbone (SmolLM2-style — LlamaDecoderLayer with GQA)
# ---------------------------------------------------------------------------

VLM_NUM_LAYERS            = 32
VLM_HIDDEN                = 960
VLM_INTERMEDIATE          = 2560
VLM_NUM_HEADS             = 15
VLM_NUM_KV_HEADS          = 5
VLM_HEAD_DIM              = 64
VLM_KV_DIM                = VLM_NUM_KV_HEADS * VLM_HEAD_DIM   # 320
VLM_RMS_NORM_EPS          = 1e-5
VLM_ROPE_THETA            = 10000.0     # lerobot.smolvlm_with_expert.apply_rope hardcodes max_wavelength=10000
VLM_VOCAB_SIZE            = 49280

# ---------------------------------------------------------------------------
# Action expert (Llama-style with cross-attn override every other layer)
# ---------------------------------------------------------------------------

EXPERT_NUM_LAYERS         = 32
EXPERT_HIDDEN             = 480         # round(960 * 0.5)
EXPERT_INTERMEDIATE       = 1280        # confirmed from checkpoint
EXPERT_NUM_HEADS          = 15          # same as VLM
EXPERT_NUM_KV_HEADS       = 5           # same as VLM
EXPERT_HEAD_DIM           = 64
EXPERT_KV_DIM             = EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM   # 320
EXPERT_Q_DIM              = EXPERT_NUM_HEADS * EXPERT_HEAD_DIM      # 960
SELF_ATTN_EVERY_N_LAYERS  = 2
# Even-indexed expert layers (0,2,...,14) are "self-attn" — concat past VLM K/V
# with new expert K/V along seq dim. k/v_proj input dim = expert hidden (720).
# Odd-indexed expert layers (1,3,...,15) are "cross-attn" — Q from expert,
# K/V from past VLM K/V projected through k/v_proj. k/v_proj input dim = 320.

# ---------------------------------------------------------------------------
# Action / state projections (flow-matching action head)
# ---------------------------------------------------------------------------

MAX_STATE_DIM             = 32
MAX_ACTION_DIM            = 32
ACTION_CHUNK_SIZE         = 50          # config.chunk_size / n_action_steps
NUM_DENOISE_STEPS         = 10

# Sinusoidal timestep embedding parameters
TIMESTEP_EMBED_DIM        = EXPERT_HIDDEN     # 720 — output dim of sin/cos block
TIMESTEP_MIN_PERIOD       = 0.004
TIMESTEP_MAX_PERIOD       = 4.0

# action_time_mlp_in: Linear(1440, 720)  — (action_emb 720 ⊕ time_emb 720) → 720
# action_time_mlp_out: Linear(720, 720)
ACTION_TIME_MLP_IN_DIM    = EXPERT_HIDDEN * 2  # 1440

# ---------------------------------------------------------------------------
# Sequence layout (static at compile time)
# ---------------------------------------------------------------------------

NUM_CAMERAS               = 2                   # image (agentview), image2 (wrist)
NUM_TEXT_TOKENS           = 48                  # tokenizer_max_length
NUM_VISION_TOKENS_TOTAL   = NUM_CAMERAS * VISION_TOKENS_PER_IMAGE  # 192
NUM_STATE_TOKENS          = 1
PREFIX_LEN                = NUM_VISION_TOKENS_TOTAL + NUM_TEXT_TOKENS + NUM_STATE_TOKENS  # 241
SUFFIX_LEN                = ACTION_CHUNK_SIZE                                              # 50
FULL_LEN                  = PREFIX_LEN + SUFFIX_LEN                                        # 291

# ---------------------------------------------------------------------------
# Neuron runtime
# ---------------------------------------------------------------------------

# DEVIATION FLAG: tp_degree=1 because num_attention_heads=15 and
# num_kv_heads=5 — neither divides cleanly into the 4 cores available on
# trn3pd98.3xlarge. Production NxDI parallel primitives are still used so the
# code stays portable to instances where head counts allow real TP, but on
# this hardware sharding effectively no-ops.
DEFAULT_TP_DEGREE         = 1
BATCH_SIZE                = 1
TORCH_DTYPE_STR           = "bfloat16"
