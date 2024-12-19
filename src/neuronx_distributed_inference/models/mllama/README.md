# Llama3.2 Multimodal

This is the Neuronx Distributed (NxD) Inference implementation of Llama3.2 Multimodal models. We currently support 11B and 90B instruct models using PyTorch checkpoints, and one image per prompt.

## Example to use Meta's PyTorch checkpoint
You can reference `examples/generation_mllama.py` to compile and run inference of Llama3.2 Multimodal model on Neuron. Please follow these steps:
- Download the PyTorch checkpoint from [Meta's official website](https://www.llama.com/llama-downloads/).
- `cd examples`
- Install the additional required packages `pip install -r requirements.txt`.
- Convert the PyTorch checkpoint to Neuron checkpoint: `python checkpoint_conversion_utils/convert_mllama_weights_to_neuron.py  --input-dir <path_to_meta_pytorch_checkpoint> --output-dir <path_to_neuron_checkpoint> --instruct`.
- Run compilation and inference example on neuron: `python generation_mllama.py |&tee out.txt`.

## Example to use HuggingFace checkpoint
Upcoming.