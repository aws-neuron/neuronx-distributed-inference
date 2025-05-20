from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper


class AsyncTensorWrapper:
    """
    Wrapper class for tensors from models executed with async runtime.

    Attributes:

    1. `async_result`: A 2d list of tensors representing a collection of outputs from all tp ranks.
        **Exception:** If on_cpu=True, ranked_tensor is a 2d list of tensors containing  to be concatenated together
    2. `batch_padded`: A boolean indicating if the result has been padded along batch dimension.
    3. `on_cpu`: A boolean indicating if the ranked tensors have been synced to CPU.

    All 4 possiblilities are to be handled:
    1. `(batch_padded=False, on_cpu=False)` which implies request_batch_size = compiled_batch_size
    2. `(batch_padded=True, on_cpu=False)` which implies request_batch_size < compiled_batch_size
    3. `(batch_padded=True, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size != 0
    4. `(batch_padded=False, on_cpu=True)` which implies request_batch_size > compiled_batch_size and request_batch_size % compiled_batch_size == 0
    """

    def __init__(self, async_result: List[List[torch.Tensor]], batch_padded: bool, on_cpu: bool):
        self.async_result = async_result
        self.batch_padded = batch_padded
        self.on_cpu = on_cpu

        if self.on_cpu:
            assert not is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on Neuron."
        else:
            assert is_ranked_io(
                self.async_result
            ), f"Initialized with {on_cpu=} but found that async_result is still on CPU."

    def get_ranked_tensor(self):
        assert not self.on_cpu, "Can't get ranked tensor if async_result is already on CPU."
        return self.async_result

    def sync_async_result_to_cpu(
        self, seq_ids: torch.Tensor, is_fused_speculation: bool = False, early_exit: bool = False
    ):
        if not self.on_cpu:  # cases 1 and 2
            synced_result = get_async_output(self.async_result)
        else:  # cases 3 and 4
            if is_fused_speculation:
                synced_result = [torch.cat(x, dim=0) for x in zip(*self.async_result)]
            else:
                synced_result = torch.cat([x[0] for x in self.async_result], dim=0)
        if early_exit:  # used for discarding results
            return
        # handle unpadding based on supplied seq_ids
        batch_size = seq_ids.shape[0]
        seq_ids = seq_ids.reshape(batch_size)  # make sure it's 1d tensor for index_select
        if isinstance(synced_result, torch.Tensor):
            return torch.index_select(synced_result, 0, seq_ids)

        index_select = lambda x: torch.index_select(x, 0, seq_ids)  # noqa: E731
        return list(map(index_select, synced_result))


def execute_model(
    neuron_base_instance: "NeuronBaseForCausalLM",
    model_to_execute: "ModelWrapper",
    input_dict: Dict[str, Any],
    hits_bucket_boundary: bool = False,
) -> Tuple[AsyncTensorWrapper, bool]:
    ordered_tuple_inputs = neuron_base_instance._convert_input_dict_to_ordered_tuple(input_dict)

    pad_type = "first_fit" if not hits_bucket_boundary else "second_fit"
    return model_to_execute(*ordered_tuple_inputs, pad_type=pad_type), model_to_execute.is_neuron()


def get_async_output(ranked_async_tensor: Any, clone: bool = False):
    if not is_ranked_io(ranked_async_tensor):
        return ranked_async_tensor

    maybe_clone = lambda x: x.clone().detach() if clone else x  # noqa: E731
    return [maybe_clone(async_tensor.cpu()) for async_tensor in ranked_async_tensor[0]]


def is_ranked_io(input_ids: Any):
    # make sure the contents are List[List[torch.Tensor]]
    # and that tensor is a privateuseone device tensor (neuron)
    return (
        isinstance(input_ids, list)
        and isinstance(input_ids[0], list)
        and isinstance(input_ids[0][0], torch.Tensor)
        and input_ids[0][0].device.type == "privateuseone"
    )


def within_bounds(inputs: Dict[str, Any], max_length: int, generation_length: int):
    return (max_length - inputs["position_ids"].max().item()) > generation_length * 2


def will_hit_bucket_boundary(known_length: int, buckets: List[int], max_num_tokens_generated=1):
    hits_bucket_boundary = False
    for bucket in buckets:
        # we do max_num_tokens_generated * 2 because we speculatively execute one step
        # ahead, before knowing the results of the current neff execution. In the worst case
        # both neffs will have full matching tokens, which is problematic near the bucket boundary
        # therefore, we will execute at a higher bucket in such cases.
        if known_length < bucket and (known_length + max_num_tokens_generated * 2) >= bucket:
            hits_bucket_boundary = True
            return hits_bucket_boundary

    return hits_bucket_boundary


def causal_lm_async_execution(
    neuron_base_instance: "NeuronBaseForCausalLM",
    inputs: Dict[str, Any],
    is_fused_speculation: bool = False,
):
    # PREFILL STAGE:
    is_prefill = neuron_base_instance._is_prefill(inputs["position_ids"])
    neuron_base_instance.async_should_stop = False
    prefill_outputs = None
    is_run_on_neuron = None
    if is_prefill:
        prefill_outputs, is_run_on_neuron = execute_model(
            neuron_base_instance, neuron_base_instance.context_encoding_model, inputs
        )

        # Sequence IDs from vLLM will be in sorted order, but the maximum range of sequence IDs is
        # not [0, num_requested_prefills] but [0, max_num_seqs]. To prevent out-of-bound accesses,
        # we convert the sequence IDs to their argsorted values.
        _seq_ids = torch.argsort(inputs["seq_ids"])
        outputs = prefill_outputs.sync_async_result_to_cpu(
            _seq_ids, is_fused_speculation=is_fused_speculation
        )

        # clean up async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None

        return outputs, is_run_on_neuron

    # GENERATION STAGE:
    generation_model = (
        neuron_base_instance.token_generation_model
        if not is_fused_speculation
        else neuron_base_instance.fused_spec_model
    )
    generation_length = (
        1 if not is_fused_speculation else neuron_base_instance.neuron_config.speculation_length
    )
    known_seqlen = inputs["attention_mask"].shape[1]
    hits_bucket_boundary = will_hit_bucket_boundary(
        known_seqlen,
        buckets=generation_model.neuron_config.buckets,
        max_num_tokens_generated=generation_length,
    )

    stay_in_sync_mode = (
        not torch.equal(neuron_base_instance.prior_seq_ids, inputs["seq_ids"])
        or hits_bucket_boundary
    )
    start_async = not stay_in_sync_mode and neuron_base_instance.prior_outputs is None
    continue_async = not stay_in_sync_mode and not start_async

    if stay_in_sync_mode:
        # reset async state
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None

    if stay_in_sync_mode or start_async:
        next_outputs, is_run_on_neuron = execute_model(
            neuron_base_instance,
            generation_model,
            inputs,
            hits_bucket_boundary=hits_bucket_boundary,
        )
        if start_async:
            neuron_base_instance.prior_outputs = next_outputs
            neuron_base_instance.prior_seq_ids = inputs["seq_ids"]

    if start_async or continue_async:
        if within_bounds(inputs, neuron_base_instance.neuron_config.seq_len, generation_length):
            next_outputs = neuron_base_instance.prior_outputs
            inputs["input_ids"] = next_outputs.get_ranked_tensor()
            if neuron_base_instance.next_cpu_inputs is not None:
                inputs["attention_mask"] = neuron_base_instance.next_cpu_inputs["attention_mask"]
                inputs["position_ids"] = neuron_base_instance.next_cpu_inputs["position_ids"]
            elif not is_fused_speculation:
                raise RuntimeError(
                    "Expected next_cpu_inputs to be generated for a non fused_spec model."
                )

            next_outputs, is_run_on_neuron = execute_model(
                neuron_base_instance, generation_model, inputs
            )
        else:
            if neuron_base_instance.prior_outputs is not None:
                outputs = neuron_base_instance.prior_outputs.sync_async_result_to_cpu(
                    inputs["seq_ids"], is_fused_speculation=is_fused_speculation
                )
                neuron_base_instance.prior_outputs = None
                neuron_base_instance.prior_seq_ids = None
                neuron_base_instance.async_should_stop = True
            else:
                raise RuntimeError(
                    "The stopping criteria for fused async should have been triggered, but it wasn't."
                )

            return outputs, True  # assume async mode only runs on neuron

    # output to be returned
    outputs: AsyncTensorWrapper = neuron_base_instance.prior_outputs if not stay_in_sync_mode else next_outputs
    outputs = outputs.sync_async_result_to_cpu(
        inputs["seq_ids"], is_fused_speculation=is_fused_speculation
    )

    if stay_in_sync_mode:
        # make sure prior outputs is not set
        neuron_base_instance.prior_outputs = None
        neuron_base_instance.prior_seq_ids = None
        return outputs, is_run_on_neuron

    # next step
    neuron_base_instance.prior_outputs = next_outputs
    neuron_base_instance.prior_seq_ids = inputs["seq_ids"]

    return outputs, is_run_on_neuron
