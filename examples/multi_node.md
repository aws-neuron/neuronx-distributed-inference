## Using NxDI distributed launcher

If your environment has ``MPI`` installed, you can make use of distributed laucher utility provided in NxDI library.

Use ``--`` to separate distributed launcher arguments and program command. Specify hostnames/IP addresses through ``--hosts`` argument.

```bash
nxdi_distributed_launcher \
  --hosts ip-172-31-35-58 ip-172-31-34-25 \
  --nproc_per_node 1 \
  -- \
  inference_demo \
  --model-type llama \
  --task-type causal-lm \
  run \
    --model-path TinyLLama-v0 \
    --compiled-model-path traced_models/TinyLLama-v0-multi-node-0/ \
    --enable-torch-dist  \
    --local_ranks_size 32 \
    --tp-degree 64 \
    --batch-size 2 \
    --max-context-length 32 \
    --seq-len 64 \
    --on-device-sampling \
    --enable-bucketing \
    --top-k 1 \
    --do-sample \
    --pad-token-id 2 \
    --prompt "I believe the meaning of life is" \
    --prompt "The color of the sky is" 2>&1 |& tee log
```

To run ``torchrun`` command with launcher, pass ``--torchrun`` argument to the launcher. E.g.

```bash
nxdi_distributed_launcher \
  --hosts ip-172-31-35-58 ip-172-31-34-25 \
  --nproc_per_node 1 \
  --torchrun \
  -- \
  example.py
```


## Manually running processes on each node

User can run one process on each node to execute NxDI model on multiple nodes. In order to distinguish ranks,
please provide ``--start_rank_id <int>`` argument. Please ensure ``--start_rank_id <int>``
is multiple of ``--local_ranks_size <int>`` argument.


### Example to run two process with tp=16 on single Trn1 node.

Process 1:
```bash
MASTER_PORT=65111  NEURON_RT_VISIBLE_CORES=0-15  NEURON_CPP_LOG_LEVEL=1 NEURON_RT_ROOT_COMM_ID=10.1.201.64:63423 inference_demo \
       --model-type llama \
       --task-type causal-lm \
        run \
         --model-path TinyLLama-v0 \
         --compiled-model-path traced_models/TinyLLama-v0-multi-node-0/ \
         --torch-dtype bfloat16 \
         --start_rank_id 0 \
         --local_ranks_size 16 \
         --tp-degree 32 \
         --batch-size 2 \
         --max-context-length 32 \
         --seq-len 64 \
         --on-device-sampling \
         --enable-bucketing \
         --top-k 1 \
         --do-sample \
         --pad-token-id 2 \
         --prompt "I believe the meaning of life is" \
         --prompt "The color of the sky is" 2>&1 | tee log
```


Process 2:
```bash
NEURON_RT_VISIBLE_CORES=16-31 NEURON_CPP_LOG_LEVEL=1 NEURON_RT_ROOT_COMM_ID=10.1.201.64:63423 inference_demo \
       --model-type llama \
       --task-type causal-lm \
        run \
         --model-path TinyLLama-v0 \
         --compiled-model-path traced_models/TinyLLama-v0-multi-node-1/ \
         --torch-dtype bfloat16 \
         --start_rank_id 16 \
         --local_ranks_size 16 \
         --tp-degree 32 \
         --batch-size 2 \
         --max-context-length 32 \
         --seq-len 64 \
         --on-device-sampling \
         --enable-bucketing \
         --top-k 1 \
         --do-sample \
         --pad-token-id 2 \
         --prompt "I believe the meaning of life is" \
         --prompt "The color of the sky is" 2>&1 | tee log


```


### Example to run two process with tp=64 on two Trn1 nodes.

```bash
NEURON_CPP_LOG_LEVEL=1 NEURON_RT_ROOT_COMM_ID=10.1.201.64:63423 inference_demo \
              --model-type llama \
              --task-type causal-lm \
              run \
                --model-path TinyLLama-v0 \
                --compiled-model-path  traced_models/TinyLLama-v0-multi-node_0/ \
                --torch-dtype bfloat16 \
                --start_rank_id 0 \
                --local_ranks 32 \
                --tp-degree 64 \
                --batch-size 2 \
                --max-context-length 32 \
                --seq-len 64 \
                --on-device-sampling \
                --enable-bucketing \
                --top-k 1 \
                --do-sample \
                --pad-token-id 2 \
                --prompt "I believe the meaning of life is" \
                --prompt "The color of the sky is" 2>&1 | tee rank_0.log
```

```bash
NEURON_CPP_LOG_LEVEL=1 NEURON_RT_ROOT_COMM_ID=10.1.201.64:63423 inference_demo \
              --model-type llama \
              --task-type causal-lm \
              run \
                --model-path TinyLLama-v0 \
                --compiled-model-path  traced_models/TinyLLama-v0-multi-node_1/ \
                --torch-dtype bfloat16 \
                --start_rank_id 32 \
                --local_ranks 32 \
                --tp-degree 64 \
                --batch-size 2 \
                --max-context-length 32 \
                --seq-len 64 \
                --on-device-sampling \
                --enable-bucketing \
                --top-k 1 \
                --do-sample \
                --pad-token-id 2 \
                --prompt "I believe the meaning of life is" \
                --prompt "The color of the sky is" 2>&1 | tee rank_1.log
```
