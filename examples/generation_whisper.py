import os
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.whisper.modeling_whisper import (
    WhisperInferenceConfig,
    NeuronApplicationWhisper,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


DTYPE = torch.float16
BATCH_SIZE = 1
TP_DEGREE = 8
MODEL_PATH = "/home/ubuntu/models/whisper-large-v3-turbo/"
TRACED_MODEL_PATH = "/home/ubuntu/traced_models/whisper-large-v3-turbo/"


def main():
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Define configs
    neuron_config = NeuronConfig(
        batch_size=BATCH_SIZE,
        torch_dtype=DTYPE,
        tp_degree=TP_DEGREE,
    )
    inference_config = WhisperInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    # Compile model
    if not os.path.exists(TRACED_MODEL_PATH):
        print("\nCompiling and saving model...")
        neuron_model = NeuronApplicationWhisper(MODEL_PATH, config=inference_config)
        neuron_model.compile(TRACED_MODEL_PATH)

    # Load from compiled checkpoint
    print("\nLoading model from compiled checkpoint...")
    neuron_model = NeuronApplicationWhisper(TRACED_MODEL_PATH, config=inference_config)
    neuron_model.load(TRACED_MODEL_PATH)

    # Transcribe an audio file
    print("\nTranscribing audio file...")
    result = neuron_model.transcribe(
        "audio-sample.mp3",
        # language="en",  # Uncomment to specify language, otherwise auto-detect
        verbose=True,
    )
    print(result["text"])


if __name__ == "__main__":
    main()
