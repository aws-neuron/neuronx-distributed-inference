from neuronx_distributed_inference.models.dbrx.modeling_dbrx import NeuronDbrxForCausalLM
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
from neuronx_distributed_inference.models.llama4.modeling_llama4 import NeuronLlama4ForCausalLM
from neuronx_distributed_inference.models.mixtral.modeling_mixtral import NeuronMixtralForCausalLM
from neuronx_distributed_inference.models.mllama.modeling_mllama import NeuronMllamaForCausalLM
from neuronx_distributed_inference.models.mllama.utils import add_instruct
from neuronx_distributed_inference.models.qwen2.modeling_qwen2 import NeuronQwen2ForCausalLM
from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import NeuronQwen3ForCausalLM

END_TO_END_MODEL = "e2e_model"
CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"
SPECULATION_MODEL = "speculation_model"
FUSED_SPECULATION_MODEL = "fused_speculation_model"
MEDUSA_MODEL = "medusa_speculation_model"
LAYOUT_OPT = "layout_opt"
LM_HEAD_NAME = "lm_head.pt"
FUSED_SPECULATION_MODEL = "fused_speculation_model"
VISION_ENCODER_MODEL = "vision_encoder_model"

BENCHMARK_REPORT_PATH = "./benchmark_report.json"

BASE_COMPILER_WORK_DIR = "/tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"
LAYOUT_OPT_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + LAYOUT_OPT + "/"
SPEC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + SPECULATION_MODEL + "/"

SUBMODEL_TO_COMPILER_WORK_DIR = {
    CONTEXT_ENCODING_MODEL: CTX_ENC_MODEL_COMPILER_WORK_DIR,
    TOKEN_GENERATION_MODEL: TKN_GEN_MODEL_COMPILER_WORK_DIR,
    LAYOUT_OPT: LAYOUT_OPT_COMPILER_WORK_DIR,
}

TEST_PROMPT = "Hello, I am a language model, and I am here to help,"
MM_TEST_PROMPT = add_instruct("What is in this image? Tell me a story", has_image=[1])

MODEL_TYPES = {
    "llama": {"causal-lm": NeuronLlamaForCausalLM},
    "llama4": {"causal-lm": NeuronLlama4ForCausalLM},
    "mllama": {"causal-lm": NeuronMllamaForCausalLM},
    "mixtral": {"causal-lm": NeuronMixtralForCausalLM},
    "dbrx": {"causal-lm": NeuronDbrxForCausalLM},
    "qwen2": {"causal-lm": NeuronQwen2ForCausalLM},
    "qwen3": {"causal-lm": NeuronQwen3ForCausalLM},
}
