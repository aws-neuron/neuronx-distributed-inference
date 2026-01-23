Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.

# Daanggn Neuron Inference Migration - E2E Pipeline

This folder contains the end-to-end pipeline for compiling the Gemma 3 27B model on AWS Neuron and running performance benchmarks with vLLM Bench.

## Prerequisites

- AWS EC2 instance with Neuron support (inf2/trn1 instance types)
- HuggingFace account and token
- Optional: IAM instance profile for AWS service access

## Quick Start

For users who want to get started quickly:

1. Launch an inf2/trn1 instance with Neuron DLAMI
2. Run the setup script (see detailed setup below)
3. Prepare your model configurations in `configs/`
4. Run `./run_multiple_tracing.sh` to compile models
5. Run `./run_multiple_benchmark.sh` to benchmark
6. Visualize results with `vis.ipynb`

## Detailed Setup

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

Install NxDI as an editable package:
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

### 4. Install vLLM with Neuron Support

```bash
git clone -b 2.26.1 https://github.com/aws-neuron/upstreaming-to-vllm.git
cd upstreaming-to-vllm
# Skip if using Neuron DLAMI: pip install -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

### 5. Configure Gemma3 Support in vLLM

Copy the modified vLLM files to add Gemma3 support:

```bash
# Set paths for convenience
SOURCE_DIR="/home/ubuntu/daanggn-neuron-inference-migration/vllm_modified"
TARGET_DIR="/home/ubuntu/upstreaming-to-vllm/vllm"

# Copy modified files
cp "$SOURCE_DIR/model_executor/model_loader/neuronx_distributed.py" \
   "$TARGET_DIR/model_executor/model_loader/"

cp "$SOURCE_DIR/worker/neuronx_distributed_model_runner.py" \
   "$TARGET_DIR/worker/"
```

### 6. Create vLLM Bench Environment for Benchmarking
```bash
deactivate
python3 -m venv vllm_orig_venv
source vllm_orig_venv/bin/activate
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
pip install --upgrade pip
pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation
```

### 7. Download Benchmarking Datasets

```bash
mkdir -p ~/datasets/coco
cd ~/datasets/coco/
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip 
rm -rf train2017.zip
```

## Usage

### 1. Model Tracing and Compilation

Prepare configurations for each model in `configs/`. Sample configurations are provided in the `configs/` folder. Make sure to move or delete them if they are not your target configurations for compilation.

Run compilation for each configuration sequentially:
```bash
deactivate
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
export PYTHONPATH="/home/ubuntu/daanggn-neuron-inference-migration:$PYTHONPATH"
cd /home/ubuntu/daanggn-neuron-inference-migration/e2e_pipeline
./run_multiple_tracing.sh
```

Tracing logs will be saved to `tracing_logs/`. Check the logs to verify compilation success. Successful compilations will print sample outputs at the end. 


### 2. Performance Benchmarking

Run benchmarks across all compiled models. Compiled models are read from the `/home/ubuntu/traced_model/` directory.
```bash
cd /home/ubuntu/daanggn-neuron-inference-migration/e2e_pipeline
./run_multiple_benchmark.sh
```

Benchmarking results will be saved in the `results/` folder with metrics including:
- Latency (P50, P95, P99)
- Throughput (tokens/second)


### 3. Results Visualization

Use the provided Jupyter notebook `vis.ipynb` to analyze and visualize results.

The notebook provides:
- Latency/throughput across different concurrency levels
- Cost per 1K tokens across different concurrency levels