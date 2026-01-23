#!/bin/bash
# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.


cd /home/ubuntu/daanggn-neuron-inference-migration/e2e_pipeline

for config in configs/v18*.py; do
    config_name=$(basename "$config" .py)
    echo "Running config: $config_name"
    CONFIG_MODULE=e2e_pipeline.configs.$config_name python generation_gemma3.py 2>&1 | tee tracing_logs/${config_name}_log.txt
    echo "Completed: $config_name"
done
