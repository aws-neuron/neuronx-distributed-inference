#!/bin/bash
#SBATCH --job-name=ctrl-contrib
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/ctrl/validate_%j.out
#SBATCH --error=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/ctrl/validate_%j.err
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=compute1-st-kaena-training-1-4

set -euo pipefail

echo "=========================================="
echo "CTRL Contrib: Compile + Validate + Profile"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Environment setup
source /shared/dhwanw/venv2/.venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt

# Paths
MODEL_PATH="/shared/dhwanw2/models/ctrl"
CONTRIB_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/ctrl"
COMPILED_MODEL_PATH="${CONTRIB_DIR}/compiled_model"
MODELING_FILE="${CONTRIB_DIR}/src/modeling_ctrl.py"
PROFILING_DIR="${CONTRIB_DIR}/neff_profiles"
RESULTS_FILE="${CONTRIB_DIR}/validation_results.json"

# NXDI source - use compatible version from workdir (master branch requires newer NxD)
NXDI_SRC="/shared/dhwanw/agents/experiments/workdir/5459_FlexOlmo-7x7B-1T/workdir/NeuronxDistributedInference/src"
export PYTHONPATH="${NXDI_SRC}:${CONTRIB_DIR}/src:${PYTHONPATH:-}"

echo "Model path: ${MODEL_PATH}"
echo "Compiled path: ${COMPILED_MODEL_PATH}"
echo "Modeling file: ${MODELING_FILE}"
echo ""

# ============================================
# Step 1: Compile
# ============================================
echo "=========================================="
echo "Step 1: Compiling CTRL model"
echo "=========================================="

python3 -c "
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_ctrl import NeuronCTRLForCausalLM, CTRLInferenceConfig

model_path = '${MODEL_PATH}'
compiled_path = '${COMPILED_MODEL_PATH}'

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
    save_sharded_checkpoint=True,
)

config = CTRLInferenceConfig.from_pretrained(
    model_path, neuron_config=neuron_config,
)

model = NeuronCTRLForCausalLM(model_path, config)
print('Compiling...')
model.compile(compiled_path)
print('Compilation complete!')
"

echo ""
echo "Compilation finished. Checking artifacts..."
ls -la "${COMPILED_MODEL_PATH}/"
echo ""

# ============================================
# Step 2: Validate (token match via test_accuracy)
# ============================================
echo "=========================================="
echo "Step 2: Running accuracy validation"
echo "=========================================="

python3 -c "
import sys
import json
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src/')

from amzn.neuron.neuroboros.model_validation.validator import test_accuracy

config = {
    'model_name': 'ctrl',
    'model_path': '${MODEL_PATH}',
    'compiled_model_path': '${COMPILED_MODEL_PATH}',
    'model_class': '${MODELING_FILE}:NeuronCTRLForCausalLM',
    'config_class': '${MODELING_FILE}:CTRLInferenceConfig',
    'num_tokens_to_check': 64,
}

test_params = {
    'batch_size': 1,
    'seq_len': 128,
}

passed, details = test_accuracy(config, test_params, hf_dtype_override='float32')

# Save results
results = {
    'model': 'ctrl',
    'passed': passed,
    'match_rate': details.get('match_rate', 0),
    'avg_teacher_forced_match_rate': details.get('avg_teacher_forced_match_rate', 0),
    'avg_best_match_rate': details.get('avg_best_match_rate', 0),
    'total_tokens_compared': details.get('total_tokens_compared', 0),
    'matching_tokens': details.get('matching_tokens', 0),
}

with open('${RESULTS_FILE}', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to ${RESULTS_FILE}')
print(f'Passed: {passed}')
print(f'Match rate: {details.get(\"match_rate\", 0)*100:.2f}%')
print(f'Teacher-forced rate: {details.get(\"avg_teacher_forced_match_rate\", 0)*100:.2f}%')
"

echo ""

# ============================================
# Step 3: Profile
# ============================================
echo "=========================================="
echo "Step 3: Running profiling"
echo "=========================================="

python3 -c "
import sys
import json
sys.path.insert(0, '/home/dhwanw/workplace/NeuroborosFoundations/src/')

from amzn.neuron.neuroboros.utils.run_profiling import run_profiling, ProfilingConfig

config = ProfilingConfig(
    model_class='${MODELING_FILE}:NeuronCTRLForCausalLM',
    config_class='${MODELING_FILE}:CTRLInferenceConfig',
    model_path='${MODEL_PATH}',
    compiled_model_path='${COMPILED_MODEL_PATH}',
    num_tokens=32,
    warmup_tokens=5,
    model_name='ctrl',
)

result = run_profiling(config)

print(f'Profiling success: {result.success}')
if result.success:
    print(f'Tokens/sec: {result.tokens_per_second:.1f}')
    print(f'Elapsed: {result.elapsed_seconds:.3f}s')
    print(f'Output preview: {result.output_preview[:200]}')
else:
    print(f'Error: {result.error}')
    print(f'Stage: {result.error_stage}')

# Save profiling results
profiling_file = '${CONTRIB_DIR}/profiling_results.json'
with open(profiling_file, 'w') as f:
    json.dump(result.to_dict(), f, indent=2)
print(f'Profiling results saved to {profiling_file}')
"

echo ""
echo "=========================================="
echo "All steps complete!"
echo "=========================================="
echo "Date: $(date)"
