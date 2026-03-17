#!/bin/bash
#SBATCH --job-name=minicpm-moe-8x2b-validate
#SBATCH --partition=compute1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/minicpm-moe-8x2b/slurm-%j.out

set -euo pipefail

# Environment setup
source /shared/dhwanw/venv2/.venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt
export PYTHONPATH="/shared/dhwanw/agents/experiments/workdir/5459_FlexOlmo-7x7B-1T/workdir/NeuronxDistributedInference/src/:${PYTHONPATH}"

MODEL_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/minicpm-moe-8x2b"
HF_MODEL_PATH="/shared/dhwanw2/models/MiniCPM-MoE-8x2b"
COMPILED_MODEL_PATH="${MODEL_DIR}/compiled_model"
MODELING_FILE="${MODEL_DIR}/src/modeling_minicpm_moe_neuronx.py"

echo "=== MiniCPM-MoE-8x2b: Compile + Validate + Profile ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"
neuron-ls

# Clean compile cache
rm -rf /tmp/neuron-compile-cache

# Step 1: Compile model
echo ""
echo "=== Step 1: Compile ==="
python3 -c "
import sys
sys.path.insert(0, '${MODEL_DIR}/src')
from modeling_minicpm_moe_neuronx import NeuronMiniCPMMoEForCausalLM, MiniCPMMoEInferenceConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from amzn.neuron.neuroboros.utils.model_compiler import DirectModelCompiler, CompilationConfig

config = CompilationConfig(
    model_class=NeuronMiniCPMMoEForCausalLM,
    config_class=MiniCPMMoEInferenceConfig,
    neuron_config_class=MoENeuronConfig,
    model_path='${HF_MODEL_PATH}',
    output_path='${COMPILED_MODEL_PATH}',
    batch_size=1,
    seq_len=2048,
    tp_degree=2,
    use_fp16=True,
)
compiler = DirectModelCompiler(config)
success = compiler.compile()
assert success, 'Compilation failed!'
print('Compilation successful!')
"

# Step 2: Validate (token match accuracy)
echo ""
echo "=== Step 2: Validate ==="
python3 -c "
import sys
sys.path.insert(0, '${MODEL_DIR}/src')
from amzn.neuron.neuroboros.model_validation.validator import test_accuracy

config = {
    'model_name': 'minicpm-moe-8x2b',
    'model_path': '${HF_MODEL_PATH}',
    'compiled_model_path': '${COMPILED_MODEL_PATH}',
    'model_class': '${MODELING_FILE}:NeuronMiniCPMMoEForCausalLM',
    'config_class': '${MODELING_FILE}:MiniCPMMoEInferenceConfig',
    'num_tokens_to_check': 256,
}

test_params = {
    'batch_size': 1,
    'seq_len': 2048,
}
passed, details = test_accuracy(config, test_params)
print(f'Validation passed: {passed}')
print(f'Details: {details}')
"

# Step 3: Profile throughput
echo ""
echo "=== Step 3: Profile ==="
python3 -c "
import sys
sys.path.insert(0, '${MODEL_DIR}/src')
from amzn.neuron.neuroboros.utils.run_profiling import ProfilingConfig, run_profiling

config = ProfilingConfig(
    model_class='${MODELING_FILE}:NeuronMiniCPMMoEForCausalLM',
    config_class='${MODELING_FILE}:MiniCPMMoEInferenceConfig',
    model_path='${HF_MODEL_PATH}',
    compiled_model_path='${COMPILED_MODEL_PATH}',
)
result = run_profiling(config)
print(f'Profiling result: {result}')
"

echo ""
echo "=== Done ==="
echo "End time: $(date)"
