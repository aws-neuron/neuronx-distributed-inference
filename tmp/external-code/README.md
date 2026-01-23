Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

# Daanggn Neuron Inference Migration

This project demonstrates migrating inference workloads to AWS Neuron for the Gemma-3-27B model using NeuronX Distributed Inference (NxDI).

## Prerequisites

- AWS EC2 instance with Neuron support (inf2/trn1 instance types)
- HuggingFace account and token
- Optional: IAM instance profile for AWS service access

## Quick Start

### 1. Launch Neuron DLAMI Instance

Launch an EC2 instance using:
- **AMI**: `Deep Learning AMI Neuron (Ubuntu 22.04) 20250919`
- **Storage**: 500GiB gp3 root volume
- **Instance Type**: inf2 or trn1 family

> **Note**: Neuron DLAMIs come with the Neuron SDK pre-installed. See [NxDI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html#option-1-launch-an-instance-using-a-neuron-dlami) for details.

### 2. Environment Setup

Activate the pre-installed virtual environment:
```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
```

Install NxDI as editable package:
```bash
git clone https://github.com/aws-neuron/neuronx-distributed-inference.git
cd neuronx-distributed-inference
git checkout e07f0567ad8b77969b0f6eec650234ecb7359419
pip install -e .
cd ..
```

### 3. Download Model

Authenticate and download the Gemma-3-27B model:
```bash
huggingface-cli login --token <YOUR_HF_TOKEN>
huggingface-cli download google/gemma-3-27b-it --local-dir /home/ubuntu/model_hf/gemma-3-27b-it
```

### 4. Run Inference

The script automatically handles model compilation and inference:
```bash
export PYTHONPATH="/home/ubuntu/daanggn-neuron-inference-migration:$PYTHONPATH"
cd daanggn-neuron-inference-migration
python scripts/generation_gemma3.py
```

> **Info**: The script checks for compiled artifacts in `TRACED_MODEL_PATH`. If not found, it compiles the model first, then runs inference.

## Alternative: vLLM Inference

For vLLM-based inference, see the detailed guide: [scripts/README.md](scripts/README.md)

## Troubleshooting

- Ensure your instance type supports Neuron (inf2/trn1)
- Verify sufficient disk space for model compilation
- Check HuggingFace token permissions for model access