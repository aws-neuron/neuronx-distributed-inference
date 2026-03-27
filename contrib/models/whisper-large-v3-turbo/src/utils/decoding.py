# Modified from https://github.com/openai/whisper/blob/main/whisper/decoding.py

from dataclasses import replace
from typing import TYPE_CHECKING, List, Union

import torch
from torch import Tensor

from whisper.decoding import DecodingOptions, DecodingResult, DecodingTask, Inference

if TYPE_CHECKING:
    from modeling_whisper import NeuronApplicationWhisper as Whisper


class NeuronInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        tokens = tokens.to(torch.int32)
        model_dtype = self.model.config.neuron_config.torch_dtype
        padded_tokens, last_pos, pad_mask = self.model._prepare_decoder_inputs(tokens)

        if tokens.shape[-1] > self.initial_token_length:
            # Decode: only need the last token, pass dummy xa since
            # cross-attention K/V caches were populated during prefill
            tokens = tokens[:, -1:]
            dummy_audio = torch.zeros(
                audio_features.shape[0],
                1,
                audio_features.shape[2],
                dtype=model_dtype,
            )
            return self.model.decoder(tokens, dummy_audio, last_pos, pad_mask)
        else:
            # Prefill: return logits for all real (non-padded) token positions.
            # The upstream _main_loop indexes logits[:, sot_index] and logits[:, -1],
            # so we must return the full unpadded sequence (not just last_pos).
            xa = audio_features.to(model_dtype)
            tokens = padded_tokens
            logits = self.model.decoder(tokens, xa, last_pos, pad_mask)
            # Slice to real token length (last_pos is 0-indexed, so +1 for length).
            # For batched decoding, all samples share the same initial_token_length,
            # so last_pos is uniform — use the first sample's value.
            seq_len = last_pos[0].item() + 1
            return logits[:, :seq_len, :]


class NeuronDecodingTask(DecodingTask):
    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        self.inference = NeuronInference(model, len(self.initial_tokens))


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
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
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"Unsupported dtype: {dtype}"
    )
    # For fp16, set fp16=True so upstream whisper casts audio_features to float16.
    # For bfloat16, set fp16=False — we cast to bfloat16 ourselves in NeuronInference.logits().
    # (Upstream whisper only knows float16/float32; setting fp16=True with bfloat16
    #  would cast to float16, causing a dtype mismatch with the traced model.)
    options = replace(options, fp16=(dtype == torch.float16))

    result = NeuronDecodingTask(model, options).run(mel)

    return result[0] if single else result
