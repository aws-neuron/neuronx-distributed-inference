"""LaughterSegmentation Neuron model utilities.

Provides loading, compilation, and inference for omine-me/LaughterSegmentation
(Wav2Vec2ForAudioFrameClassification) on AWS Neuron using torch_neuronx.trace().

The model classifies 7-second audio windows (16 kHz, 112,000 samples) into
349 per-frame binary laughter/non-laughter predictions.

IMPORTANT: Wav2Vec2 uses weight_norm parametrizations on pos_conv_embed.conv.
These must be removed before tracing or torch_neuronx.trace() will crash with
a PjRt buffer null error. Use ``remove_parametrizations()`` or the
``LaughterNeuronPipeline`` (which handles this automatically).
"""

import os
import time
from typing import Optional

import torch
import torch.nn.utils.parametrize as parametrize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
HF_MODEL_ID = "omine-me/LaughterSegmentation"
INPUT_SEC = 7
SAMPLE_RATE = 16000
INPUT_SAMPLES = INPUT_SEC * SAMPLE_RATE  # 112,000
FRAMES_PER_WINDOW = 349  # Wav2Vec2 output frames for 7-sec input

DEFAULT_COMPILER_ARGS = [
    "--model-type",
    "transformer",
    "--optlevel",
    "2",
    "--auto-cast",
    "matmult",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def remove_parametrizations(model: torch.nn.Module) -> None:
    """Remove weight_norm and other parametrizations for XLA compatibility.

    Wav2Vec2 uses weight_norm on ``pos_conv_embed.conv``, which creates
    XLA-incompatible hooks that cause PjRt buffer null crashes during trace.
    This function strips all parametrizations in-place.
    """
    for name, module in model.named_modules():
        if hasattr(module, "parametrizations"):
            for param_name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, param_name)


def load_cpu_model(model_path: str) -> torch.nn.Module:
    """Load LaughterSegmentation model on CPU with parametrizations removed.

    Args:
        model_path: Path to either:
            - A directory containing model.safetensors (HuggingFace layout)
            - A direct path to model.safetensors

    Returns:
        Wav2Vec2ForAudioFrameClassification model ready for tracing.
    """
    import safetensors.torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForAudioFrameClassification

    # Initialize model from base config
    config = Wav2Vec2Config.from_pretrained(AUDIO_MODEL_NAME)
    config.num_labels = 1
    config.problem_type = "single_label_classification"
    model = Wav2Vec2ForAudioFrameClassification(config)

    # Resolve weights path
    if os.path.isdir(model_path):
        weights_path = os.path.join(model_path, "model.safetensors")
    else:
        weights_path = model_path

    # Load fine-tuned weights
    state_dict = safetensors.torch.load_file(weights_path, device="cpu")

    # Strip "audio_model." prefix if present (from the original Model wrapper)
    prefix = "audio_model."
    stripped = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(stripped)
    model.eval()

    # Remove weight_norm parametrizations
    remove_parametrizations(model)

    return model


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_neuron_model(
    model: torch.nn.Module,
    batch_size: int = 1,
    compiler_args: Optional[list] = None,
    lnc: Optional[int] = None,
    save_path: Optional[str] = None,
) -> torch.jit.ScriptModule:
    """Compile model with torch_neuronx.trace().

    Args:
        model: CPU model (must have parametrizations already removed).
        batch_size: Batch size for the compiled model.
        compiler_args: Compiler arguments. Defaults to DEFAULT_COMPILER_ARGS.
        lnc: Logical NeuronCore count (trn2 only). None = use default.
        save_path: If provided, save the compiled model to this path.

    Returns:
        Compiled Neuron model (torch.jit.ScriptModule).
    """
    import torch_neuronx

    if compiler_args is None:
        compiler_args = list(DEFAULT_COMPILER_ARGS)
    else:
        compiler_args = list(compiler_args)

    if lnc is not None:
        compiler_args.extend(["--lnc", str(lnc)])

    example_input = torch.randn(batch_size, INPUT_SAMPLES)

    neuron_model = torch_neuronx.trace(
        model,
        example_input,
        compiler_args=compiler_args,
        inline_weights_to_neff=True,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.jit.save(neuron_model, save_path)

    return neuron_model


def extract_logits(output) -> torch.Tensor:
    """Extract logits tensor from model output (handles dict/tuple/tensor)."""
    if isinstance(output, dict):
        return output["logits"]
    elif isinstance(output, (tuple, list)):
        return output[0]
    return output


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class LaughterNeuronPipeline:
    """High-level pipeline for laughter detection on Neuron.

    Handles model loading, compilation, and inference.

    Usage::

        pipeline = LaughterNeuronPipeline(
            model_path="/home/ubuntu/models/LaughterSegmentation",
        )
        pipeline.compile(batch_size=2)

        # Or load pre-compiled:
        # pipeline.load("/path/to/compiled.pt")

        # Run inference
        import torch
        audio = torch.randn(2, 112000)  # 2 x 7-second windows at 16 kHz
        logits = pipeline.predict(audio)  # (2, 349, 1)
    """

    def __init__(self, model_path: str):
        """Initialize pipeline.

        Args:
            model_path: Path to model weights (directory or .safetensors file).
        """
        self.model_path = model_path
        self.cpu_model = None
        self.neuron_model = None

    def compile(
        self,
        batch_size: int = 1,
        compiler_args: Optional[list] = None,
        lnc: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Load weights and compile for Neuron.

        Args:
            batch_size: Batch size for the compiled model.
            compiler_args: Compiler arguments (defaults to recommended set).
            lnc: Logical NeuronCore count (trn2 only).
            save_path: If provided, save compiled model to this path.
        """
        if self.cpu_model is None:
            self.cpu_model = load_cpu_model(self.model_path)

        self.neuron_model = compile_neuron_model(
            self.cpu_model,
            batch_size=batch_size,
            compiler_args=compiler_args,
            lnc=lnc,
            save_path=save_path,
        )

    def load(self, compiled_path: str) -> None:
        """Load a pre-compiled Neuron model.

        Args:
            compiled_path: Path to the saved .pt file.
        """
        self.neuron_model = torch.jit.load(compiled_path)

    def predict(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Run inference and return logits.

        Args:
            audio_input: Tensor of shape (batch_size, 112000) -- 7-sec windows
                at 16 kHz sample rate.

        Returns:
            Logits tensor of shape (batch_size, 349, 1).
        """
        if self.neuron_model is None:
            raise RuntimeError(
                "Model not compiled or loaded. Call compile() or load() first."
            )

        with torch.no_grad():
            output = self.neuron_model(audio_input)

        return extract_logits(output)
