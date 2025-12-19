import copy
from typing import List

import torch
import torch.nn.functional as F


def generate(
    model: torch.nn.Module,
    max_len: int,
    prompt_tokens: List[List[int]],
    stop_tokens: List[int],
    pad_token: int,
):
    """
    A simple greedy generation implementation.

    For now it only supports running on device with AoT, and greedy sampling.

    TODO: Generalize it to support more use cases.
    """
    prompt_tokens = copy.deepcopy(prompt_tokens)

    # Track max pos per batch
    last_pos = torch.tensor([len(prompt) - 1 for prompt in prompt_tokens], dtype=torch.int32)

    # Pad all batch lines to the same sequence length, to put them in a tensor
    max_prompt_len = max([len(prompt) for prompt in prompt_tokens])
    padded_tokens = [prompt + [pad_token] * (max_prompt_len - len(prompt)) for prompt in prompt_tokens]
    tokens = torch.tensor(padded_tokens, dtype=torch.int32)

    input_tokens = tokens
    input_bs, _ = input_tokens.shape

    attention_mask = torch.where(tokens != pad_token, 1, 0).to(torch.int32)

    # A tensor to keep track of generation completion per batch line
    is_gen_complete = torch.full((input_bs, 1), False)

    while True:
        kwargs = {
            "input_tokens": input_tokens,
            "last_pos": last_pos,
            "attention_mask": attention_mask,
        }
        logits = model.forward(**kwargs)

        last_pos = last_pos + 1

        # assuming we are doing greedy sampling
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        input_tokens = next_token.to(torch.int32)

        # Add the new token to prompt
        for idx, prompt in enumerate(prompt_tokens):
            if not is_gen_complete[idx][0].item():
                prompt.append(next_token[idx].item())

        for stop_token in stop_tokens:
            is_gen_complete = is_gen_complete.logical_or(next_token == stop_token)

        # Stop generation when all batch lines are complete
        if is_gen_complete.all():
            break

        if torch.max(last_pos).item() >= max_len:
            break

        # Update mask
        attention_mask = F.pad(attention_mask, (0, 1), "constant", 0)
        attention_mask[torch.arange(last_pos.shape[0]), last_pos] = 1

    return prompt_tokens
