#!/bin/bash
#SBATCH --job-name=gptneo-contrib
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/gpt-neo-1.3B/validate_%j.out
#SBATCH --error=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/gpt-neo-1.3B/validate_%j.err
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --exclusive

set -euo pipefail

echo "========================================="
echo "GPT-Neo-1.3B Contrib Validation"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Environment setup
source /shared/dhwanw/venv2/.venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt

# Paths
MODEL_PATH="/shared/dhwanw2/models/gpt-neo-1.3B"
CONTRIB_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/gpt-neo-1.3B"
COMPILED_MODEL_PATH="${CONTRIB_DIR}/compiled_model"
MODELING_FILE="${CONTRIB_DIR}/src/modeling_gpt_neo.py"

# Add contrib src to PYTHONPATH (NXDI is already in the venv)
export PYTHONPATH="${CONTRIB_DIR}/src:${PYTHONPATH:-}"

# Clean compile cache
rm -rf /tmp/neuron-compile-cache

echo ""
echo "========================================="
echo "Phase 1: Compile (TP=1, seq_len=128, batch=1, bf16)"
echo "========================================="

python3 -c "
import sys
sys.path.insert(0, '${CONTRIB_DIR}/src')
import torch
from modeling_gpt_neo import NeuronGPTNeoForCausalLM, GPTNeoInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = '${MODEL_PATH}'
compiled_path = '${COMPILED_MODEL_PATH}'

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
)

config = GPTNeoInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

model = NeuronGPTNeoForCausalLM(model_path, config)
print('Compiling...')
model.compile(compiled_path)
print('Compilation complete!')
"

echo ""
echo "========================================="
echo "Phase 2: Token-Level Accuracy Validation"
echo "========================================="

python3 -c "
import sys
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src')
from amzn.neuron.neuroboros.model_validation.validator import test_accuracy

config = {
    'model_name': 'gpt-neo-1.3B',
    'model_path': '${MODEL_PATH}',
    'compiled_model_path': '${COMPILED_MODEL_PATH}',
    'model_class': '${MODELING_FILE}:NeuronGPTNeoForCausalLM',
    'config_class': '${MODELING_FILE}:GPTNeoInferenceConfig',
    'num_tokens_to_check': 64,
}
test_params = {
    'batch_size': 1,
    'seq_len': 128,
}
passed, details = test_accuracy(config, test_params)
print(f'Accuracy test passed: {passed}')
print(f'Details: {details}')
"

echo ""
echo "========================================="
echo "Phase 3: Profiling"
echo "========================================="

python3 -c "
import sys
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src')
from amzn.neuron.neuroboros.utils.run_profiling import run_profiling, ProfilingConfig

config = ProfilingConfig(
    model_class='${MODELING_FILE}:NeuronGPTNeoForCausalLM',
    config_class='${MODELING_FILE}:GPTNeoInferenceConfig',
    model_path='${MODEL_PATH}',
    compiled_model_path='${COMPILED_MODEL_PATH}',
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
echo "========================================="
echo "All phases complete!"
echo "========================================="
echo "Date: $(date)"
