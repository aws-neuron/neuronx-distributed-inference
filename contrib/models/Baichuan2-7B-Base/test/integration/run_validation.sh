#!/bin/bash
#SBATCH --job-name=baichuan2-contrib-validate
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/Baichuan2-7B-Base/test/integration/validation_%j.out
#SBATCH --error=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/Baichuan2-7B-Base/test/integration/validation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=01:00:00

echo "=== Baichuan2-7B-Base Contrib Model - Token Match Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"

source /shared/dhwanw/venv2/.venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt

echo ""
echo "Python: $(which python)"
echo "neuron-ls:"
neuron-ls
echo ""

SCRIPT_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/Baichuan2-7B-Base/test/integration"
python "$SCRIPT_DIR/test_token_match.py"
EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
