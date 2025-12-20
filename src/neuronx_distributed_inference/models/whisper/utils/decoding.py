# Modified from https://github.com/openai/whisper/blob/main/whisper/decoding.py

from dataclasses import replace
from typing import TYPE_CHECKING, List, Union

import torch
from torch import Tensor

from whisper.decoding import DecodingOptions, DecodingResult, DecodingTask, Inference

if TYPE_CHECKING:
    from neuronx_distributed_inference.models.whisper.modeling_whisper import NeuronApplicationWhisper as Whisper


class NeuronInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        tokens = tokens.to(torch.int32)
        padded_tokens, last_pos, pad_mask = self.model._prepare_decoder_inputs(tokens)

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]
            return self.model.decoder(tokens, audio_features, last_pos, pad_mask)
        else:
            tokens = padded_tokens
        return self.model.decoder(tokens, audio_features, last_pos, pad_mask)[:, : last_pos + 1]


class NeuronDecodingTask(DecodingTask):
    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        self.inference = NeuronInference(model, len(self.initial_tokens))


@torch.no_grad()
def decode(
    model: "Whisper", mel: Tensor, options: DecodingOptions = DecodingOptions(), **kwargs
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    dtype = model.config.neuron_config.torch_dtype
    assert dtype in [torch.float16, torch.float32], f"Unsupported dtype: {dtype}"
    options = replace(options, fp16=(dtype == torch.float16))

    result = NeuronDecodingTask(model, options).run(mel)

    return result[0] if single else result
