#!/bin/bash
#SBATCH --job-name=git-contrib
#SBATCH --output=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/git-base/validate_%j.out
#SBATCH --error=/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/git-base/validate_%j.err
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --exclusive

set -euo pipefail

echo "=== Git-base Contrib Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Environment setup
source /shared/ouroboros/venv/bin/activate
bash /home/dhwanw/workplace/OuroborosCodeGeneration/scripts/basic_experiment/retry_deps.sh /shared/dhwanw/deps.sh /shared/dhwanw/apt

# Paths
MODEL_PATH="/shared/dhwanw2/models/git-base"
COMPILED_PATH="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/git-base/compiled_model"
CONTRIB_DIR="/home/dhwanw/workplace/neuronx-distributed-inference/contrib/models/git-base"

# Use the NXDI already installed in the ouroboros venv (matched with its NxD version)
# Only add the contrib src dir to path for the modeling file

echo ""
echo "=== Step 1: Compile ==="
python3 -c "
import torch
import sys
sys.path.insert(0, '${CONTRIB_DIR}/src')
from modeling_git import NeuronGitForCausalLM, GitInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig

neuron_config = NeuronConfig(
    tp_degree=1,
    batch_size=1,
    seq_len=128,
    max_context_length=128,
    torch_dtype=torch.bfloat16,
)

config = GitInferenceConfig.from_pretrained(
    '${MODEL_PATH}', neuron_config=neuron_config,
)

model = NeuronGitForCausalLM('${MODEL_PATH}', config)
model.compile('${COMPILED_PATH}')
print('Compilation complete!')
"

echo ""
echo "=== Step 2: Validate (Token Matching) ==="
python3 -c "
import torch, sys, json
sys.path.insert(0, '${CONTRIB_DIR}/src')
from modeling_git import NeuronGitForCausalLM, GitInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = '${MODEL_PATH}'
COMPILED_PATH = '${COMPILED_PATH}'

with open(f'{COMPILED_PATH}/neuron_config.json') as f:
    nc_data = json.load(f)
nc_dict = nc_data.get('neuron_config', nc_data)

neuron_config = NeuronConfig(
    tp_degree=nc_dict.get('tp_degree', 1),
    batch_size=nc_dict.get('batch_size', 1),
    seq_len=nc_dict.get('seq_len', 128),
    max_context_length=nc_dict.get('seq_len', 128),
    torch_dtype=torch.bfloat16,
)

config = GitInferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
neuron_model = NeuronGitForCausalLM(MODEL_PATH, config)
neuron_model.load(COMPILED_PATH)

hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
hf_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompts = ['a photo of a cat sitting on', 'the weather today is', 'in the year 2025']
NUM_TOKENS = 64
total_match = 0
total_tokens = 0

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    with torch.no_grad():
        hf_out = hf_model.generate(input_ids, max_new_tokens=NUM_TOKENS, do_sample=False)
    hf_tokens = hf_out[0, input_ids.shape[1]:]

    generated = input_ids.clone()
    for _ in range(NUM_TOKENS):
        pos = torch.arange(generated.shape[1]).unsqueeze(0)
        with torch.no_grad():
            outputs = neuron_model(generated, position_ids=pos)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
    neuron_tokens = generated[0, input_ids.shape[1]:]

    min_len = min(len(hf_tokens), len(neuron_tokens))
    matches = (hf_tokens[:min_len] == neuron_tokens[:min_len]).sum().item()
    total_match += matches
    total_tokens += min_len
    pct = 100.0 * matches / min_len if min_len > 0 else 0
    print(f'Prompt: \"{prompt}\" -> {matches}/{min_len} ({pct:.1f}%)')

overall = 100.0 * total_match / total_tokens if total_tokens > 0 else 0
print(f'Overall: {total_match}/{total_tokens} ({overall:.1f}%)')
print('PASS' if overall >= 90 else 'WARN')
"

echo ""
echo "=== Step 3: Profile ==="
python3 -c "
import torch, sys, time, json
sys.path.insert(0, '${CONTRIB_DIR}/src')
from modeling_git import NeuronGitForCausalLM, GitInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoTokenizer

MODEL_PATH = '${MODEL_PATH}'
COMPILED_PATH = '${COMPILED_PATH}'

with open(f'{COMPILED_PATH}/neuron_config.json') as f:
    nc_data = json.load(f)
nc_dict = nc_data.get('neuron_config', nc_data)

neuron_config = NeuronConfig(
    tp_degree=nc_dict.get('tp_degree', 1), batch_size=nc_dict.get('batch_size', 1),
    seq_len=nc_dict.get('seq_len', 128), max_context_length=nc_dict.get('seq_len', 128),
    torch_dtype=torch.bfloat16,
)
config = GitInferenceConfig.from_pretrained(MODEL_PATH, neuron_config=neuron_config)
model = NeuronGitForCausalLM(MODEL_PATH, config)
model.load(COMPILED_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
inputs = tokenizer('a photo of', return_tensors='pt')
input_ids = inputs.input_ids

for _ in range(5):
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.no_grad(): _ = model(input_ids, position_ids=pos)

ttft_times = []
for _ in range(20):
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
    start = time.perf_counter()
    with torch.no_grad(): _ = model(input_ids, position_ids=pos)
    ttft_times.append((time.perf_counter() - start) * 1000)

generated = input_ids.clone()
start = time.perf_counter()
for _ in range(100):
    pos = torch.arange(generated.shape[1]).unsqueeze(0)
    with torch.no_grad():
        outputs = model(generated, position_ids=pos)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    generated = torch.cat([generated, torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)], dim=-1)
throughput = 100 / (time.perf_counter() - start)

print(f'TTFT: avg={sum(ttft_times)/len(ttft_times):.2f}ms, min={min(ttft_times):.2f}ms')
print(f'Throughput: {throughput:.2f} tokens/sec')
"

echo ""
echo "=== All steps complete ==="
echo "Date: $(date)"
