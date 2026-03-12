import os
import collections
import sys
from enum import Enum
from typing import Union, Tuple, List, Optional, Callable, Any

import torch
import numpy as np
from torch_neuronx.testing.validation import neuron_allclose


# Tolerances tend to be tighter at smaller top_k values because the accuracy of
# more likely tokens is more important than less likely tokens.
DEFAULT_TOLERANCE_MAP = {
    "5": (1e-5, 0.01),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001

# Summary formatting constants
_SUMMARY_WIDTH = 80
_ERROR_PRECISION = 4


class _Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class _ValidationStatus(Enum):
    """Enum representing different validation status classifications for logit comparison."""

    MATCHED = "matched"
    TOPK_ERRORS = "topk_errors"
    ACCEPTABLE_DIVERGENCE = "acceptable_divergence"
    ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS = "acceptable_divergence_with_topk_errors"
    DIVERGED = "diverged"
    DIVERGED_WITH_TOPK_ERRORS = "diverged_with_topk_errors"

    def get_display_text(self) -> str:
        """Returns the display text for this validation status."""
        display_map = {
            _ValidationStatus.MATCHED: "✓ Matched",
            _ValidationStatus.TOPK_ERRORS: "✗ TopK errors",
            _ValidationStatus.ACCEPTABLE_DIVERGENCE: "✓ Acceptable divergence",
            _ValidationStatus.ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS: "✗ Acceptable divergence + TopK errors",
            _ValidationStatus.DIVERGED: "✗ Diverged",
            _ValidationStatus.DIVERGED_WITH_TOPK_ERRORS: "✗ Diverged + TopK errors",
        }
        return display_map[self]

    def get_color(self) -> str:
        """Returns the appropriate color for this validation status."""
        if self == _ValidationStatus.MATCHED:
            return "GREEN"
        elif self == _ValidationStatus.ACCEPTABLE_DIVERGENCE:
            return "YELLOW"
        else:  # All error states
            return "RED"

    def is_passing(self) -> bool:
        """Returns True if this status represents a passing validation."""
        return self in (_ValidationStatus.MATCHED, _ValidationStatus.ACCEPTABLE_DIVERGENCE)


def logit_validation(
    input_ids: List[List[int]],
    generate_fn: Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
    expected_logits: torch.Tensor,
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    suppress_passing: bool = True,
    colorize: bool = True,
) -> bool:
    """
    Validates model accuracy by comparing raw prediction scores (logits) against reference values.

    This function performs comprehensive logit-based validation designed specifically for
    hardware-specific model accuracy testing. Unlike token-based validation that only compares
    final predictions, this approach compares the raw prediction scores at each position,
    providing deeper insights into numerical precision differences and model behavior across
    different hardware platforms (CPU, GPU, Neuron).

    What is Logit Matching?
    -----------------------
    Logit matching compares the raw prediction scores (logits) produced by language
    models before any sampling or argmax operations are applied. These logits represent the
    model's confidence in each possible next token as unnormalized log probabilities.

    **Logit Matching vs Token Matching:**

    - **Token Matching**: Only compares the final sampled tokens after argmax operation
      - Example: Comparing "Paris" vs "London" as final predictions
      - Limited insight: Only shows if the top choice differs
      - Misses subtle differences: Can't detect when model is "almost wrong"

    - **Logit Matching**: Compares the full probability distribution before sampling
      - Example: Comparing [Paris: 8.2, London: 8.1, Berlin: 7.9] vs [Paris: 8.15, London: 8.12, Berlin: 7.88]
      - Rich insight: Shows confidence levels and ranking of all possibilities
      - Detects drift: Can identify when models are converging toward different answers

    **Why Logit Matching is Superior:**

    1. **Captures Full Distribution**: Reveals the model's uncertainty and confidence levels
    2. **Detects Subtle Drift**: Identifies numerical differences before they affect final predictions
    3. **Hardware Validation**: Essential for validating model behavior across different hardware
    4. **Debugging Power**: Helps isolate where and why models diverge
    5. **Precision Analysis**: Quantifies the impact of different floating-point formats

    Hardware Precision Challenges:
    -----------------------------
    Different hardware platforms compute floating-point math slightly differently due to:

    - **Precision Formats**: CPU (FP32), GPU (FP16/BF16), Neuron (FP16/BF16)
    - **Operation Ordering**: Different platforms may reorder operations for optimization
    - **Parallel Reduction**: Various strategies for summing/reducing across parallel units
    - **Accumulation Effects**: In models with billions of operations, tiny differences
      accumulate through layers, causing numerical drift in final logits

    Teacher Forcing for Fair Comparison:
    -----------------------------------
    To isolate hardware-specific differences at each position, this validation uses
    teacher forcing: even when the model under test would sample a different token
    than the reference, we force it to use the reference token as input for the next position.

    **Without Teacher Forcing:**
    ```
    Reference: "The capital of France is Paris"
    Neuron:    "The capital of France is London" (due to drift at position 4)
    → Sequences diverge completely, remaining logits incomparable
    → Can't determine if position 5+ errors are due to position 4 or accumulated drift
    ```

    **With Teacher Forcing:**
    ```
    Position 4: Neuron wants "London", but we force it to use "Paris"
    → Both models continue with identical context: "The capital of France is Paris"
    → Can compare logits at positions 5, 6, 7... fairly
    → Isolates WHERE numerical drift occurs without cascading effects
    ```

    **Benefits of Teacher Forcing:**

    1. **Isolation of Errors**: Determines exactly which positions have numerical drift
    2. **Fair Comparison**: Ensures both models process identical context at each step
    3. **Cascading Prevention**: Stops early errors from contaminating later positions
    4. **Debugging Precision**: Identifies if errors accumulate or occur at specific layers
    5. **Comprehensive Coverage**: Validates the entire sequence length, not just until first divergence

    Tolerance Strategy:
    ------------------
    The validation uses a multi-tiered tolerance approach because accuracy requirements
    vary by token importance:

    - **Top-1 tokens**: Strictest tolerance (most important for generation quality)
    - **Top-5 tokens**: Moderate tolerance (affects sampling diversity)
    - **Top-50 tokens**: Relaxed tolerance (less critical for most applications)
    - **All tokens**: Most relaxed tolerance (tail of distribution less important)

    Args:
        input_ids: List of input token sequences for each batch.
            Each inner list represents a sequence of token IDs that serve as the
            initial context for generation.

        generate_fn: Function that takes input_ids as a tensor and returns either:
            - torch.Tensor: Logits tensor of shape (seq_len, batch_size, vocab_size)
            - Tuple[torch.Tensor, torch.Tensor]: (logits, sequences) where sequences
              are the sampled token IDs of shape (batch_size, seq_len)
            This function represents the model under test (e.g., Neuron-compiled model).

        expected_logits: Reference logits tensor of shape
            (seq_len, batch_size, vocab_size) to validate against. These typically
            come from a reference implementation (e.g., CPU/GPU model).

        tol_map: Dictionary mapping top-k values to (atol, rtol) tolerance
            tuples. Keys can be strings representing top-k values or "all" (all tokens).
            Defaults to DEFAULT_TOLERANCE_MAP if None.
            Example: {"all": (1e-5, 0.05), "50": (1e-5, 0.02), "5": (1e-5, 0.01)}

        divergence_difference_tol: Tolerance for divergence difference
            when sequences would naturally diverge. This measures how much the logit
            for the expected token differs from the maximum logit in the actual output.
            Defaults to DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE (0.001) if None.

        suppress_passing: If True, suppresses display of tokens with matched status
            in the validation results, showing only tokens that require attention
            (errors, divergences, etc.).

        colorize: If True, applies ANSI color codes to the validation output for
            enhanced readability in terminals that support color. If False, outputs
            plain text without color formatting. Defaults to True.

    Returns:
        True if validation passes for all tokens and batches within specified
        tolerances, False if any position fails validation or divergence
        tolerance is exceeded.

    Raises:
        AssertionError: If generate_fn returns logits and sequences with mismatched shapes.
        KeyError: If expected_logits tensor has incompatible dimensions.

    Examples:
        ```python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 1. Load your model
        model_name = 'openlm-research/open_llama_3b'
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # 2. Prepare your input
        prompt = 'I am a fun tutorial.'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # 3. Retrieve your goldens (in a real example, you wouldn't use exactly the
        #    same model in steps 3 and 4)
        generation_result = model.generate(
            input_ids,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        expected_logits = torch.stack(generation_result['scores'])

        # 4. Build your generate function
        def generate_fn(input_ids):
            input_ids = torch.tensor(input_ids)
            generation_result = model.generate(
                input_ids,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            return torch.stack(generation_result['scores'])

        # 5. Validate
        passed = logit_validation(input_ids.tolist(), generate_fn, expected_logits)
        ```

        Custom Tolerance Configuration:
        ```python
        ...
        # Strict validation for critical applications
        strict_tolerances = {
            "all": (1e-6, 0.01),    # Very strict for all tokens
            "50": (1e-6, 0.005),     # Extremely strict for top-50
            "5": (1e-7, 0.001),      # Ultra strict for top-5
            "1": (1e-8, 0.0005),     # Maximum precision for top-1
        }

        passed = logit_validation(
            input_ids=input_ids,
            generate_fn=neuron_generate_fn,
            expected_logits=expected_logits,
            tol_map=strict_tolerances,
            divergence_difference_tol=0.0001  # Very strict divergence tolerance
        )
        ```

        Batch Processing:
        ```python
        ...
        # Multiple sequences in batch
        batch_input_ids = [
            [1, 2, 3, 4],      # Sequence 1
            [5, 6],            # Sequence 2
        ]

        # Expected logits shape: (seq_len=10, batch_size=2, vocab_size=50257)
        expected_logits = torch.randn(10, 2, 50257)

        passed = logit_validation(batch_input_ids, generate_fn, expected_logits)
        ```

    Note:
        The `generate_fn` should handle teacher forcing internally or return logits
        that correspond to the expected sequence length. The validation will use
        teacher forcing by extending input_ids with reference tokens as needed.

        For optimal performance with large vocabularies, consider using appropriate
        top-k values in tol_map to focus validation on the most relevant tokens.
    """
    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    batch_size = len(input_ids)
    current_output_start_idx = 0
    expected_sequences = expected_logits.argmax(dim=2).T
    expected_sequence_length = expected_sequences.shape[1]

    passed = True
    results = [[] for _ in range(batch_size)]

    while current_output_start_idx < expected_sequence_length:
        generate_result = generate_fn(input_ids)

        client_sampling = False
        if isinstance(generate_result, tuple):
            actual_logits, actual_sequences = generate_result
            assert actual_logits.shape[:-1] == actual_sequences.T.shape, (
                f"Shape mismatch between logits and sequences returned by generate_fn: "
                f"actual_logits.shape={actual_logits.shape}, actual_sequences.shape={actual_sequences.shape}"
            )
            client_sampling = True
        else:
            actual_logits = generate_result
            actual_sequences = actual_logits.argmax(dim=2).T
        divergence_idx = _get_divergence_idx(expected_sequences[:, current_output_start_idx:], actual_sequences)
        divergence_idx += current_output_start_idx

        for batch_idx in range(batch_size):
            for token_idx in range(divergence_idx - current_output_start_idx):
                actual_token_id = None
                if client_sampling:
                    actual_token_id = actual_sequences[batch_idx, token_idx].item()
                single_token_passed, single_token_results = _validate_single_token_logits(
                    expected_logits=expected_logits[token_idx + current_output_start_idx, batch_idx, :],
                    actual_logits=actual_logits[token_idx, batch_idx, :],
                    tol_map=tol_map,
                    divergence_difference_tol=divergence_difference_tol,
                    remove_shift=True,
                    actual_token_id=actual_token_id,
                )

                results[batch_idx].append(single_token_results)
                passed &= single_token_passed

        for i in range(len(input_ids)):
            input_ids[i].extend(expected_sequences[i, current_output_start_idx:divergence_idx].tolist())

        current_output_start_idx = divergence_idx

    _print_logit_validation_results(results, tol_map, divergence_difference_tol, suppress_passing, colorize)

    return passed


def _get_divergence_idx(expected_sequences: torch.Tensor, actual_sequences: torch.Tensor) -> int:
    """Get the index of the first divergent token across all batches."""
    min_seq_len = min(actual_sequences.shape[1], expected_sequences.shape[1])
    diff = torch.ne(actual_sequences[:, :min_seq_len], expected_sequences[:, :min_seq_len])

    if torch.sum(diff) == 0:
        return min_seq_len
    else:
        return torch.min(torch.nonzero(diff), 0).values[1].item() + 1


def _preprocess_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    remove_shift: bool,
    return_removed_indices: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
    """
    This function handles two preprocessing tasks: omitting -inf values and
    removing the shift. -inf values sometimes appear in logits when a token is
    prohibited, and they cause issues in some downstream tasks. This function
    identifies indices at which either the actual or expected logits are -inf
    and omits them. To return the indices of the removed values, enable
    return_removed_indices.

    For instance, the input tensors:
        actual = [1, -inf, 3, 4], expected = [5, 6, 7, -inf]
    would be output as:
        actual = [1, 3], expected = [5, 7]

    This function also optionally finds and removes a constant shift in the
    logits by finding the least-squares approximation for p in the following
    system of linear equations:
        actual_logits = A @ p
    where
        A = [expected_logits | 1]
        p = [slope, shift].T
    In other words, it does a linear regression. Then, it subtracts the shift
    from actual_logits.

    For instance, the input tensors:
        actual = [1, 2, 3, 4], expected = [5, 6, 7, 8]
    would be output as:
        actual = [5, 6, 7, 8], expected = [5, 6, 7, 8], shift = -4
    """
    # Omit indices at which logits are -inf
    vocab_size = len(expected_logits)
    assert vocab_size == len(actual_logits)
    ninf_idxs = torch.nonzero(torch.logical_or(actual_logits == float('-inf'), expected_logits == float('-inf')))
    expected_logits = expected_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    actual_logits = actual_logits[~torch.isin(torch.arange(vocab_size), ninf_idxs)]
    shift = 0
    if remove_shift:  # Calculate and remove shift
        A = np.vstack([expected_logits.float(), np.ones(len(expected_logits))]).T
        _, shift = np.linalg.lstsq(A, actual_logits.float(), rcond=None)[0]
        actual_logits -= shift
    if return_removed_indices:
        return expected_logits, actual_logits, shift, ninf_idxs.reshape(-1)
    else:
        return expected_logits, actual_logits, shift


def _validate_top_k_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    top_k: Union[str, int],
    atol: float,
    rtol: float
) -> Tuple[bool, float]:
    """Validates logits for top-k tokens using specified tolerances."""
    if top_k != "all":  # filter only the top k most likely tokens
        if isinstance(top_k, str):
            top_k = int(top_k)
        top_k_result = torch.topk(expected_logits, top_k)
        expected_logits = top_k_result.values
        actual_logits = torch.index_select(actual_logits, 0, top_k_result.indices)

    result = neuron_allclose(actual_logits, expected_logits, rtol=rtol, atol=atol)

    return result.allclose, result.max_rel_error


def _validate_single_token_logits(
    expected_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    tol_map: dict,
    divergence_difference_tol: float,
    remove_shift: bool,
    actual_token_id: Optional[int] = None,
) -> Tuple[bool, dict]:
    """Validates logits for a single token position across all tolerance levels."""
    divergence_difference = 0
    error_map = {k: 0 for k in tol_map.keys()}
    divergence = False
    passed = torch.tensor(True)

    expected_logits, actual_logits, shift, removed_indices = _preprocess_logits(
        expected_logits, actual_logits, remove_shift, return_removed_indices=True,
    )
    if actual_token_id is not None:
        actual_token_id = _calculate_new_index(actual_token_id, removed_indices)
        assert actual_token_id is not None, "Provided actual token ID has -inf score"

    def get_top2_values_indices_diff(logits):
        top2_values, top2_indices = torch.topk(logits, 2)
        top1_top2_diff = top2_values[0] - top2_values[1]
        # Calculate relative difference between top1 and top2 logits (signed)
        top1_top2_relative_diff = (
            (top1_top2_diff / torch.abs(top2_values[0]))
            if torch.abs(top2_values[0]) > torch.tensor(1e-8)
            else torch.tensor(0.0)
        )
        # TODO: Shift indices to original indices based on removed_indices
        return top2_values, top2_indices, top1_top2_diff, top1_top2_relative_diff

    # expected
    (
        expected_top2_values,
        expected_top2_indices,
        expected_top1_top2_diff,
        expected_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(expected_logits)
    # actual
    (
        actual_top2_values,
        actual_top2_indices,
        actual_top1_top2_diff,
        actual_top1_top2_relative_diff,
    ) = get_top2_values_indices_diff(actual_logits)

    # Relative error = (actual - expected) / |expected| for each top2 index (signed)
    def get_relative_error(actual_val, expected_val):
        return (
            (
                (actual_val - expected_val) / torch.abs(expected_val)
            )
            if torch.abs(expected_val) > 1e-8
            else torch.abs(actual_val - expected_val)
        )

    top1_relative_error = get_relative_error(
        actual_top2_values[0], expected_top2_values[0]
    )
    top2_relative_error = get_relative_error(
        actual_top2_values[1], expected_top2_values[1]
    )
    # Calculate actual relative difference using expected top2 indices: (actual[top1] - actual[top2]) / |actual[top1]|
    actual_values_with_expected_top1_top2_indices_relative_diff = (
        get_relative_error(
            actual_logits[expected_top2_indices[0]],
            actual_logits[expected_top2_indices[1]],
        )
    )

    # check if the logits are in bounds for each value of k
    total_errors = collections.defaultdict(dict)
    max_abs_expected = torch.max(torch.abs(expected_logits)).item()
    total_errors["all"]["mean_abs_error"] = torch.nn.functional.l1_loss(
        actual_logits, expected_logits, reduction="mean"
    ).item() / max_abs_expected
    total_errors["all"]["mean_squared_error"] = torch.nn.functional.mse_loss(
        actual_logits, expected_logits, reduction="mean"
    ).item() / (max_abs_expected ** 2)
    for top_k, tols in tol_map.items():
        atol, rtol = tols
        in_bounds, error = _validate_top_k_logits(
            expected_logits, actual_logits, top_k, atol, rtol
        )

        total_errors[top_k]["max_abs_error"] = abs(error)
        total_errors[top_k]["max_squared_error"] = error**2

        passed &= in_bounds
        error_map[top_k] = error

    # determine if the sequences diverge and evaluate the divergence difference if they do
    greedy_next_token_id = expected_logits.argmax().item()
    if actual_token_id is None:
        actual_token_id = actual_logits.argmax().item()
    if greedy_next_token_id != actual_token_id:
        divergence = True
        divergence_difference = (
            torch.max(actual_logits) - actual_logits[greedy_next_token_id]
        )
        in_bounds = divergence_difference < divergence_difference_tol
        passed &= in_bounds
    passed = passed.item()
    # report the results and return
    results = {
        "passed": passed,
        "divergence": divergence,
        "divergence_difference": divergence_difference,
        "total_errors": total_errors,
        "error_map": error_map,
        "shift": shift,
        "expected_logits": expected_logits,
        "actual_logits": actual_logits,
        "expected_top2_values": expected_top2_values.tolist(),
        "actual_top2_values": actual_top2_values.tolist(),
        "expected_top1_top2_diff": expected_top1_top2_diff.item(),
        "actual_top1_top2_diff": actual_top1_top2_diff.item(),
        "expected_top1_top2_relative_diff": expected_top1_top2_relative_diff.item(),
        "actual_top1_top2_relative_diff": actual_top1_top2_relative_diff.item(),
        "actual_with_expected_top1_top2_relative_diff":
            actual_values_with_expected_top1_top2_indices_relative_diff.item(),
        "expected_top2_indices": expected_top2_indices.tolist(),
        "actual_top2_indices": actual_top2_indices.tolist(),
        "top1_relative_errors": top1_relative_error,
        "top2_relative_errors": top2_relative_error,
    }
    return passed, results


def _calculate_new_index(original_index: int, removed_indices: torch.Tensor) -> Optional[int]:
    """Calculate new index after removing elements from a tensor."""
    if original_index in removed_indices:
        return None

    # Count how many elements were removed before this index
    removed_before = 0
    for removed_idx in removed_indices:
        if removed_idx < original_index:
            removed_before += 1

    new_index = original_index - removed_before
    return new_index


def _get_logit_validation_max_topk_error_results_summary(results: List[List[dict]]) -> dict:
    """Extracts maximum errors across all tokens and batches for each top-k value."""
    summary = {
        "max_divergence": {"error": -1},
        "max_top_k_errors": {},
    }
    for batch_index in range(len(results)):
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            if token_results["divergence_difference"] > summary["max_divergence"]["error"]:
                summary["max_divergence"]["error"] = token_results["divergence_difference"]
                summary["max_divergence"]["batch_index"] = batch_index
                summary["max_divergence"]["token_index"] = token_index
            for top_k, error in token_results["error_map"].items():
                if top_k not in summary["max_top_k_errors"]:
                    summary["max_top_k_errors"][top_k] = {"error": -1}
                if error > summary["max_top_k_errors"][top_k]["error"]:
                    summary["max_top_k_errors"][top_k]["error"] = error
                    summary["max_top_k_errors"][top_k]["batch_index"] = batch_index
                    summary["max_top_k_errors"][top_k]["token_index"] = token_index
    return summary


def _get_logit_validation_average_over_tokens_results_summary(results: List[List[dict]]) -> dict:
    """Calculates average errors across all tokens for each error metric."""
    summary = {
        "average_over_tokens": collections.defaultdict(dict),
    }

    count = 0
    total_error_dict = collections.defaultdict(dict)
    for batch_index in range(len(results)):
        for token_index in range(len(results[batch_index])):
            token_results = results[batch_index][token_index]
            count += 1
            for top_k, top_k_errors in token_results["total_errors"].items():
                for error_key, error in top_k_errors.items():
                    total_error_dict[top_k][error_key] = total_error_dict.get(top_k, {}).get(error_key, 0) + error

    if count != 0:
        for top_k, error_dict in total_error_dict.items():
            for error_key, error in error_dict.items():
                summary["average_over_tokens"][top_k][error_key] = error / count

    return summary


def _get_logit_validation_results_summary(results: List[List[dict]]) -> dict:
    """Combines maximum and average error summaries into a single results dictionary."""
    max_error_summary = _get_logit_validation_max_topk_error_results_summary(results)
    average_over_tokens_summary = _get_logit_validation_average_over_tokens_results_summary(results)

    summary = max_error_summary | average_over_tokens_summary

    return summary


def _classify_validation_result(token_results: dict, divergence_difference_tol: float, tol_map: dict) -> _ValidationStatus:
    """
    Classifies validation result status for a single token validation result.

    Args:
        token_results: Dictionary containing validation results for a single token
        divergence_difference_tol: Threshold for acceptable divergence difference
        tol_map: Dictionary mapping top-k values to (atol, rtol) tolerance tuples

    Returns:
        _ValidationStatus enum representing the classification status
    """
    # Check for top-k errors by comparing error_map against tolerances
    has_topk_errors = False
    for top_k, error in token_results["error_map"].items():
        if top_k in tol_map:
            _, rtol = tol_map[top_k]  # Use rtol as the threshold
            if error > rtol:
                has_topk_errors = True
                break

    # Check if there's any divergence
    if not token_results["divergence"]:
        if has_topk_errors:
            return _ValidationStatus.TOPK_ERRORS
        else:
            return _ValidationStatus.MATCHED

    # There is divergence, check if it's within threshold
    divergence_within_threshold = token_results["divergence_difference"] <= divergence_difference_tol

    if divergence_within_threshold:
        if has_topk_errors:
            return _ValidationStatus.ACCEPTABLE_DIVERGENCE_WITH_TOPK_ERRORS
        else:
            return _ValidationStatus.ACCEPTABLE_DIVERGENCE
    else:
        if has_topk_errors:
            return _ValidationStatus.DIVERGED_WITH_TOPK_ERRORS
        else:
            return _ValidationStatus.DIVERGED


def _format_error_with_threshold(error: float, threshold: float, colorize: bool = True) -> str:
    """Format error value with color based on threshold compliance."""
    error_exceeds_threshold = error > threshold
    colored_error = _colorize_text(f"{error:.{_ERROR_PRECISION}f}", "RED" if error_exceeds_threshold else "GREEN", colorize)
    colored_threshold = f"(threshold: {threshold})"
    colored_status_icon = _colorize_text("✗", "RED", colorize) if error_exceeds_threshold else _colorize_text("✓", "GREEN", colorize)
    return f"{colored_error} {colored_threshold} {colored_status_icon}"


def _calculate_summary_statistics(results: List[List[dict]]) -> dict:
    """Calculate summary statistics from validation results."""
    if not results or not results[0]:
        return {
            "batch_size": 0,
            "total_tokens": 0,
            "passed_tokens": 0,
            "failed_tokens": 0,
            "tokens_per_batch": 0
        }

    batch_size = len(results)
    total_tokens = sum(len(batch_results) for batch_results in results)
    passed_tokens = sum(1 for batch_results in results for token_result in batch_results if token_result["passed"])
    failed_tokens = total_tokens - passed_tokens
    tokens_per_batch = len(results[0]) if results and results[0] else 0

    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "passed_tokens": passed_tokens,
        "failed_tokens": failed_tokens,
        "tokens_per_batch": tokens_per_batch
    }


def _format_summary_header(stats: dict, colorize: bool = True) -> List[str]:
    """Format the summary header section."""
    lines = [_colorize_text("=== VALIDATION SUMMARY ===", "BOLD", colorize)]

    # Show token count and batch configuration
    if stats["tokens_per_batch"] > 0:
        lines.append(f"Total Tokens: {stats['total_tokens']} ({stats['tokens_per_batch']} per batch, {stats['batch_size']} batches)")
    else:
        lines.append(f"Total Tokens: {stats['total_tokens']}")

    # Overall status
    overall_status = _colorize_text('PASSED', 'GREEN', colorize) if stats['failed_tokens'] == 0 else _colorize_text('FAILED', 'RED', colorize)
    lines.extend([
        f"Overall Status: {overall_status}",
        "",
        _colorize_text("Max Errors:", "BOLD", colorize)
    ])

    return lines


def _format_error_metrics_section(summary: dict, tol_map: dict, divergence_difference_tol: float, colorize: bool = True) -> List[str]:
    """Format the error metrics section with thresholds and coloring."""
    lines = []

    # Add divergence error with threshold
    max_div = summary["max_divergence"]
    if max_div["error"] >= 0:
        formatted_error = _format_error_with_threshold(max_div["error"], divergence_difference_tol, colorize)
        lines.append(f"- Divergence: {formatted_error} at Batch {max_div['batch_index']} Token {max_div['token_index']}")

    # Add top-k errors with thresholds - use tol_map keys to determine order
    for top_k in tol_map.keys():
        if top_k in summary["max_top_k_errors"]:
            max_error = summary["max_top_k_errors"][top_k]
            threshold = tol_map[top_k][1]  # Use rtol as threshold
            formatted_error = _format_error_with_threshold(max_error["error"], threshold, colorize)

            k_label = f"K{top_k}" if top_k != "all" else "All"
            lines.append(f"- {k_label}: {formatted_error} at Batch {max_error['batch_index']} Token {max_error['token_index']}")

    return lines


def _format_token_results_table(results: List[List[dict]], divergence_difference_tol: float, tol_map: dict, suppress_passing: bool, colorize: bool = True) -> str:
    """Formats per-token validation results in a clean table format with automatic color coding."""
    if not results or not results[0]:
        return "No token results to display."

    batch_size = len(results)
    max_tokens = max(len(batch_results) for batch_results in results)

    lines = []
    suppressed_count = 0

    for token_idx in range(max_tokens):
        for batch_idx in range(batch_size):
            if token_idx < len(results[batch_idx]):
                token_results = results[batch_idx][token_idx]

                # Determine status with enhanced classification
                status_enum = _classify_validation_result(token_results, divergence_difference_tol, tol_map)

                # Check if we should suppress this result (only suppress MATCHED status)
                if suppress_passing and status_enum == _ValidationStatus.MATCHED:
                    suppressed_count += 1
                    continue

                # Get display text with color
                status = _colorize_text(status_enum.get_display_text(), status_enum.get_color(), colorize)

                # Format divergence info
                divergence_info = ""
                if token_results["divergence"]:
                    divergence_info = f" Δ = {token_results['divergence_difference']:.4f}"

                # Extract and color-code error values based on thresholds
                def format_error_value(top_k: str, error_value: float) -> str:
                    """Format error value with color based on threshold compliance."""
                    if top_k in tol_map:
                        _, rtol = tol_map[top_k]  # Use rtol as threshold
                        if error_value <= rtol:
                            return _colorize_text(f"{error_value:.4f}", "GREEN", colorize)
                        else:
                            return _colorize_text(f"{error_value:.4f}", "RED", colorize)
                    else:
                        return f"{error_value:.4f}"

                k5_error = token_results["error_map"].get("5", 0.0)
                k50_error = token_results["error_map"].get("50", 0.0)
                k1000_error = token_results["error_map"].get("1000", 0.0)
                all_error = token_results["error_map"].get("all", 0.0)

                k5_formatted = format_error_value("5", k5_error)
                k50_formatted = format_error_value("50", k50_error)
                k1000_formatted = format_error_value("1000", k1000_error)
                all_formatted = format_error_value("all", all_error)

                status_and_divergence = f"{status}{divergence_info}"
                padding_needed = max(0, 50 - len(status_and_divergence.replace('\033[92m', '').replace('\033[91m', '').replace('\033[93m', '').replace('\033[0m', '')))
                padding = " " * padding_needed

                line = (f"Batch {batch_idx} Token {token_idx:2d}: {status_and_divergence}{padding} | "
                        f"K5: {k5_formatted}  K50: {k50_formatted}  K1000: {k1000_formatted}  All: {all_formatted}")
                lines.append(line)

                if token_results["divergence"]:
                    lines.append("")
                    lines.append("⟲ Teacher Forcing Applied: Models diverged but continuing validation with expected tokens")
                    lines.append("")

    if suppress_passing and suppressed_count > 0:
        if not lines:  # All results were suppressed
            lines.append(f"All {suppressed_count} tokens passed validation")

    return "\n".join(lines)


def _format_validation_summary(results: List[List[dict]], tol_map: dict, divergence_difference_tol: float, colorize: bool = True) -> str:
    """Formats a comprehensive validation summary with threshold information and color coding."""
    if not results:
        return "No results to summarize."

    # Calculate summary statistics using helper function
    stats = _calculate_summary_statistics(results)
    if stats["total_tokens"] == 0:
        return "No results to summarize."

    # Get max errors
    summary = _get_logit_validation_results_summary(results)

    lines = _format_summary_header(stats, colorize)
    lines.extend(_format_error_metrics_section(summary, tol_map, divergence_difference_tol, colorize))
    lines.append("=" * _SUMMARY_WIDTH)

    return "\n".join(lines)


def _format_thresholds_legend(tol_map: dict, divergence_difference_tol: float, colorize: bool = True) -> str:
    """Format thresholds legend showing divergence and TopK tolerances."""
    lines = [
        _colorize_text("VALIDATION THRESHOLDS", "BOLD", colorize),
        f"Divergence Difference: {divergence_difference_tol}",
    ]

    topk_tolerances = []
    for top_k, (_, rtol) in tol_map.items():
        if top_k == "all":
            topk_tolerances.append(f"All: {rtol}")
        else:
            topk_tolerances.append(f"K{top_k}: {rtol}")

    lines.append(f"TopK Error Tolerances (rtol): {', '.join(topk_tolerances)}")

    return "\n".join(lines)


def _print_logit_validation_results(
    results: List[List[Any]],
    tol_map: dict = None,
    divergence_difference_tol: float = None,
    suppress_passing: bool = True,
    colorize: bool = True,
) -> None:
    """Print logit validation results"""
    # Use defaults if not provided
    if tol_map is None:
        tol_map = DEFAULT_TOLERANCE_MAP
    if divergence_difference_tol is None:
        divergence_difference_tol = DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE

    print("\n" + "=" * 80)
    print("LOGIT VALIDATION RESULTS")
    print("=" * 80)

    print(_format_thresholds_legend(tol_map, divergence_difference_tol, colorize))
    print()
    print(_format_token_results_table(results, divergence_difference_tol, tol_map, suppress_passing, colorize))
    print()
    print(_format_validation_summary(results, tol_map, divergence_difference_tol, colorize))


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check if output is being redirected
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False

    # Check for common terminals that support color
    term = os.environ.get('TERM', '').lower()
    if 'color' in term or term in ['xterm', 'xterm-256color', 'screen', 'linux']:
        return True

    return False


def _colorize_text(text: str, color: str, colorize: bool = True) -> str:
    """Apply color to text if colors are supported by the terminal and colorize is True."""
    if not colorize or not _supports_color():
        return text

    color_code = getattr(_Colors, color.upper(), '')
    if color_code:
        return f"{color_code}{text}{_Colors.RESET}"
    return text
