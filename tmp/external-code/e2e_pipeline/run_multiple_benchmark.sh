#!/bin/bash
# Copyright 2025 © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms.


# Activate virtual environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# ================= CONFIGURATION =================
# List of models to benchmark from traced_model directory
TRACED_MODEL_DIR="/home/ubuntu/traced_model"
MODELS=($(ls -d "$TRACED_MODEL_DIR"/*/))

# vLLM Sever settings
PORT=8080
HOST="http://localhost:$PORT"
# =================================================

# Function to wait for Neuron cores to be available
wait_for_neuron_cores() {
    echo "Checking Neuron core availability..."
    local retries=0
    local max_retries=100
    local wait_seconds=2
    
    while [ $retries -lt $max_retries ]; do
        # Check if there are any processes using Neuron cores
        if ! pgrep -f "neuron" > /dev/null 2>&1; then
            echo "Neuron cores are available."
            return 0
        fi
        echo "Neuron cores still in use. Waiting... ($retries/$max_retries)"
        sleep $wait_seconds
        ((retries++))
    done
    
    echo "Warning: Timeout waiting for Neuron cores to be released."
    return 1
}

# Function to check if the server is ready
wait_for_server() {
    echo "Waiting for vLLM server to start at $HOST..."
    local retries=0
    local max_retries=200  # Wait up to 60 * 5 = 300 seconds
    local wait_seconds=5

    while true; do
        # Check health endpoint (suppress output, check for 200 OK)
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/health")
        
        if [ "$HTTP_STATUS" -eq 200 ]; then
            echo "Server is up and running!"
            return 0
        fi

        if [ "$retries" -ge "$max_retries" ]; then
            echo "Timeout waiting for server to start."
            return 1
        fi

        echo "Server not ready yet (Status: $HTTP_STATUS). Retrying in $wait_seconds seconds..."
        sleep "$wait_seconds"
        ((retries++))
    done
}

# Trap Ctrl+C (SIGINT) to ensure we kill the background server if the script is stopped
cleanup() {
    echo "Cleaning up processes..."
    # Kill all child processes
    pkill -P $$ 2>/dev/null
    # Kill all vLLM processes
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    # Kill vLLM v1 EngineCore processes
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    # Kill multiprocessing processes
    pkill -9 -f "multiprocessing" 2>/dev/null
    # Kill any remaining Python processes from the venv
    pkill -9 -f "aws_neuronx_venv_pytorch_2_9_nxd_inference" 2>/dev/null
    # Kill processes on port 8080
    lsof -t -i :8080 | xargs kill -9 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# ================= MAIN LOOP =================
for MODEL_PATH in "${MODELS[@]}"; do
    # Extract MAX_CONCURRENCY from folder name (e.g., bsx where x is the number)
    MODEL_NAME=$(basename "$MODEL_PATH")
    if [[ $MODEL_NAME =~ bs([0-9]+)$ ]]; then
        MAX_CONCURRENCY=${BASH_REMATCH[1]}
    else
        MAX_CONCURRENCY=1  # Default if no bsx pattern found
    fi
    
    echo "------------------------------------------------------------------"
    echo "Model: $MODEL_PATH | Concurrency: $MAX_CONCURRENCY"
    echo "------------------------------------------------------------------"

    # 1) Check if port is free and kill existing processes
    if lsof -i :8080 > /dev/null 2>&1; then
        echo "Port 8080 is in use. Killing existing processes..."
        lsof -t -i :8080 | xargs kill -9 2>/dev/null
        sleep 5
        # Verify port is now free
        if lsof -i :8080 > /dev/null 2>&1; then
            echo "Error: Failed to free port 8080. Exiting."
            exit 1
        fi
        echo "Port 8080 is now free."
    fi
    
    # 2) Read config from neuron_config.json
    CONFIG_FILE="${MODEL_PATH}neuron_config.json"
    MAX_MODEL_LEN=$(jq -r '.text_config.neuron_config.context_encoding_buckets[0]' "$CONFIG_FILE")
    TP_SIZE=$(jq -r '.text_config.neuron_config.tp_degree' "$CONFIG_FILE")
    ON_DEVICE_SAMPLING_CONFIG=$(jq -r '.text_config.neuron_config.on_device_sampling_config' "$CONFIG_FILE")
    
    # Set ON_DEVICE_SAMPLING: 1 if null, 0 otherwise
    if [[ "$ON_DEVICE_SAMPLING_CONFIG" == "null" ]]; then
        ON_DEVICE_SAMPLING="1"
    else
        ON_DEVICE_SAMPLING="0"
    fi
    
    echo "Config: MAX_MODEL_LEN=$MAX_MODEL_LEN, TP_SIZE=$TP_SIZE, ON_DEVICE_SAMPLING=$ON_DEVICE_SAMPLING"
    
    # 3) Launch vLLM server
    echo "Launching vLLM server..."
    bash ./start_vllm_server.sh "$MODEL_PATH" "$MAX_CONCURRENCY" "$MAX_MODEL_LEN" "$TP_SIZE" "$ON_DEVICE_SAMPLING" &
    
    SERVER_PID=$!
    echo "vLLM Server PID: $SERVER_PID"

    # 4) Check if server is running properly
    if wait_for_server; then
        
        # 5) Run benchmark
        echo "Running benchmark..."
        MODEL_NAME=$(basename "$MODEL_PATH")
        RESULT_FILENAME="${MODEL_NAME}_len${MAX_MODEL_LEN}_tp${TP_SIZE}_concurrency${MAX_CONCURRENCY}.json"
        bash ./run_mm_benchmark.sh "$MAX_CONCURRENCY" "$RESULT_FILENAME" &
        BENCHMARK_PID=$!
        wait $BENCHMARK_PID
        echo "Benchmark complete."

    else
        echo "Skipping benchmark due to server failure."
        kill "$SERVER_PID" 2>/dev/null
        break
    fi

    # 6) Kill the server and all related processes
    echo "Killing vLLM server (PID: $SERVER_PID)..."
    kill -9 "$SERVER_PID" 2>/dev/null
    # Kill all vLLM processes to ensure Neuron cores are released
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    # Kill vLLM v1 EngineCore processes
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    pkill -9 -f "multiprocessing" 2>/dev/null
    # Kill any remaining Python processes from the venv
    pkill -9 -f "aws_neuronx_venv_pytorch_2_9_nxd_inference" 2>/dev/null
    
    wait "$SERVER_PID" 2>/dev/null
    # Wait for Neuron cores to be released
    wait_for_neuron_cores
    
    echo ""
done

echo "All benchmarks completed."
