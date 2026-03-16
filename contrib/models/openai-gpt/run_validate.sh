#!/bin/bash
#SBATCH --job-name=oaigpt-contrib
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/openai-gpt/validate_%j.out
#SBATCH --error=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/openai-gpt/validate_%j.err
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --exclusive

set -euo pipefail

# Environment setup
source /shared/dhwanw/venv2/.venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt

# Paths
MODEL_PATH="/shared/dhwanw2/models/openai-gpt"
MODELING_FILE="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/openai-gpt/src/modeling_openai_gpt.py"
COMPILED_PATH="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/openai-gpt/compiled_model"
RESULT_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/openai-gpt"

# Configuration
TP_DEGREE=1
BATCH_SIZE=1
SEQ_LEN=128
DTYPE="bfloat16"

echo "=========================================="
echo "OpenAI GPT-1 Contrib Validation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP_DEGREE}, BS: ${BATCH_SIZE}, SeqLen: ${SEQ_LEN}"
echo "Dtype: ${DTYPE}"
echo ""

# Step 1: Compile
echo "Step 1: Compiling..."
python3 -c "
import sys
sys.path.insert(0, '$(dirname ${MODELING_FILE})')
import torch
from modeling_openai_gpt import NeuronOpenAIGPTForCausalLM, OpenAIGPTInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig

neuron_config = NeuronConfig(
    tp_degree=${TP_DEGREE},
    batch_size=${BATCH_SIZE},
    seq_len=${SEQ_LEN},
    max_context_length=${SEQ_LEN},
    torch_dtype=torch.${DTYPE},
)

config = OpenAIGPTInferenceConfig.from_pretrained(
    '${MODEL_PATH}', neuron_config=neuron_config,
)

model = NeuronOpenAIGPTForCausalLM('${MODEL_PATH}', config)
model.compile('${COMPILED_PATH}')
print('Compilation complete')
"

echo ""
echo "Step 2: Token Match Validation..."
python3 -c "
import sys
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src')
from amzn.neuron.neuroboros.model_validation.validator import test_accuracy

config = {
    'model_name': 'openai-gpt',
    'model_path': '${MODEL_PATH}',
    'compiled_model_path': '${COMPILED_PATH}',
    'model_class': '${MODELING_FILE}:NeuronOpenAIGPTForCausalLM',
    'config_class': '${MODELING_FILE}:OpenAIGPTInferenceConfig',
    'num_tokens_to_check': 20,
}
test_params = {
    'batch_size': ${BATCH_SIZE},
    'seq_len': ${SEQ_LEN},
}

passed, details = test_accuracy(config, test_params)
print(f'Validation passed: {passed}')
print(f'Details: {details}')
"

echo ""
echo "Step 3: Profiling..."
python3 -c "
import sys
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src')
from amzn.neuron.neuroboros.utils.run_profiling import run_profiling, ProfilingConfig

config = ProfilingConfig(
    model_class='${MODELING_FILE}:NeuronOpenAIGPTForCausalLM',
    config_class='${MODELING_FILE}:OpenAIGPTInferenceConfig',
    model_path='${MODEL_PATH}',
    compiled_model_path='${COMPILED_PATH}',
)
result = run_profiling(config)
if result.success:
    print(f'Tokens/sec: {result.tokens_per_second:.1f}')
    print(f'TTFT (ms): {result.ttft_ms:.2f}')
    print(f'Latency p50 (ms): {result.latency_p50_ms:.2f}')
    print(f'Latency p99 (ms): {result.latency_p99_ms:.2f}')
else:
    print(f'Profiling failed: {result.error}')
"

echo ""
echo "=========================================="
echo "Validation complete"
echo "=========================================="
