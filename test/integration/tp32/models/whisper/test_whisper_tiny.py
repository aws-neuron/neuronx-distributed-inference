import os
import tempfile
import pytest
import torch
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.whisper.modeling_whisper import (
    WhisperInferenceConfig,
    NeuronApplicationWhisper,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from neuronx_distributed_inference.utils.accuracy import check_accuracy_embeddings
from transformers import WhisperForConditionalGeneration, WhisperConfig


def save_checkpoint(config_path):
    hf_config = WhisperConfig.from_pretrained(config_path)
    hf_model = WhisperForConditionalGeneration._from_config(hf_config)

    model_tempdir = tempfile.TemporaryDirectory()
    model_path = model_tempdir.name
    print(f"Saving model with random weights to {model_path}")
    hf_model.save_pretrained(model_path)
    return model_tempdir


@pytest.mark.parametrize(
    "tp_degree, batch_size, dtype",
    [
        # TODO: add bs > 1 tests when supported
        pytest.param(8, 1, torch.float16),
        pytest.param(8, 1, torch.float32),
    ],
)
def test_whisper_tiny(tp_degree, batch_size, dtype):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    model_tempdir = save_checkpoint(config_path)
    model_path = model_tempdir.name
    traced_model_path = os.path.join(model_path, "traced_model")

    os.makedirs(model_path, exist_ok=True)

    # transformers model
    cpu_model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)

    # Define configs
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        torch_dtype=dtype,
        tp_degree=tp_degree,
    )
    inference_config = WhisperInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Compile model
    print("\nCompiling and saving model...")
    neuron_model = NeuronApplicationWhisper(model_path, config=inference_config)
    neuron_model.compile(traced_model_path)

    # Load from compiled checkpoint
    print("\nLoading model from compiled checkpoint...")
    neuron_model.load(traced_model_path)

    # random input
    audio_features = torch.randn(
        batch_size, cpu_model.config.num_mel_bins, 2 * cpu_model.config.max_source_positions, dtype=dtype
    )
    decoder_input_ids = torch.randint(0, cpu_model.config.vocab_size, (batch_size, 10), dtype=torch.int32)
    padded_tokens, last_pos, pad_mask = neuron_model._prepare_decoder_inputs(decoder_input_ids)

    # CPU inference
    print("\nRunning CPU inference...")
    cpu_output = cpu_model(input_features=audio_features, decoder_input_ids=decoder_input_ids).logits
    print(f"CPU Output shape: {cpu_output.shape}")
    print(f"CPU Output: {cpu_output}")

    # Neuron inference
    print("\nRunning Neuron inference...")
    audio_embeddings = neuron_model.encoder(audio_features)
    neuron_output = neuron_model.decoder(padded_tokens, audio_embeddings, last_pos, pad_mask)[:, : last_pos + 1].to(
        dtype
    )
    print(f"NxD Output shape: {neuron_output.shape}")
    print(f"NxD Output: {neuron_output}")

    # Compare outputs
    passed, max_err = check_accuracy_embeddings(
        neuron_output,
        cpu_output,
    )
    print(f"Passed: {passed}, Max Error: {max_err}")
    assert passed, f"Output validation failed with max error: {max_err}"


if __name__ == "__main__":
    test_whisper_tiny(8, 1, torch.float16)
    test_whisper_tiny(8, 1, torch.float32)
    print("All tests passed.")
