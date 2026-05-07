#!/usr/bin/env python3
"""
Full 48-layer Gemma3 text encoder: compile, load, forward test.
"""

import gc
import os
import sys
import time

import torch

os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TP_DEGREE = 4
SEQ_LEN = 1024
BATCH = 1
NUM_LAYERS = 48
COMPILE_DIR = "/home/ubuntu/gemma3_encoder_compiled_1024"


def get_model_fn(tp_degree=TP_DEGREE):
    from modeling_gemma3_encoder import Gemma3TextEncoderModel

    model = Gemma3TextEncoderModel(
        vocab_size=262208,
        hidden_size=3840,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        intermediate_size=15360,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=131072,
        query_pre_attn_scalar=256,
        pad_token_id=0,
        dtype=torch.bfloat16,
    )
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, None


def main():
    print("=" * 60)
    print(
        "Compiling full 48-layer Gemma3 encoder (TP=%d, seq=%d)" % (TP_DEGREE, SEQ_LEN)
    )
    print("=" * 60)

    import torch_neuronx
    from neuronx_distributed.trace import parallel_model_trace, parallel_model_save

    os.makedirs(COMPILE_DIR, exist_ok=True)

    input_ids = torch.zeros(BATCH, SEQ_LEN, dtype=torch.int64)
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.int64)

    compiler_args = "--model-type=transformer -O1 --auto-cast=none --enable-saturate-infinity --enable-mixed-precision-accumulation --lnc=2"
    os.environ["NEURON_CC_FLAGS"] = compiler_args
    print("  Compiler flags: %s" % compiler_args)

    t0 = time.time()
    traced = parallel_model_trace(
        get_model_fn,
        (input_ids, attention_mask),
        tp_degree=TP_DEGREE,
        compiler_workdir=os.path.join(COMPILE_DIR, "compiler_workdir"),
        compiler_args=compiler_args,
        inline_weights_to_neff=False,
    )
    elapsed = time.time() - t0
    print("  Compile: %.1fs (%.1f min)" % (elapsed, elapsed / 60))

    parallel_model_save(traced, COMPILE_DIR)
    tp0_size = os.path.getsize(os.path.join(COMPILE_DIR, "tp_0.pt")) / 1e9
    print("  Saved tp_0.pt: %.2f GB" % tp0_size)

    # Quick forward test with random weights
    print("\nLoading for forward test...")
    from neuronx_distributed.trace.trace import (
        _mock_parallel_state,
        init_on_device,
        get_sharded_checkpoint,
        replace_weights,
        TensorParallelNeuronModel,
    )

    _mock_parallel_state(1, 0)
    with init_on_device(torch.device("cpu")):
        ref_model, _ = get_model_fn()
    checkpoint = ref_model.state_dict()
    total_params = sum(v.numel() for v in checkpoint.values())
    print(
        "  Checkpoint: %d keys, %.2f B params" % (len(checkpoint), total_params / 1e9)
    )
    del ref_model
    gc.collect()

    models = []
    for rank in range(TP_DEGREE):
        t0r = time.time()
        ckpt = {k: v.clone() for k, v in checkpoint.items()}
        _mock_parallel_state(TP_DEGREE, rank)
        with init_on_device(torch.device("meta")):
            model, _ = get_model_fn()
        get_sharded_checkpoint(ckpt, model, rank, TP_DEGREE)
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(os.path.join(COMPILE_DIR, "tp_0.pt"))
        replace_weights(traced_model, ckpt)
        models.append(traced_model)
        print("  [rank %d] %.1fs" % (rank, time.time() - t0r))
        gc.collect()
    del checkpoint
    gc.collect()

    compiled = TensorParallelNeuronModel(models)
    print("  All %d ranks loaded" % TP_DEGREE)

    print("\nForward pass...")
    _ = compiled(input_ids, attention_mask)  # warmup
    t0 = time.time()
    output = compiled(input_ids, attention_mask)
    elapsed = time.time() - t0

    expected = (BATCH, SEQ_LEN, 3840, NUM_LAYERS + 1)
    print("  Time: %.3fs" % elapsed)
    print("  Output shape: %s (expected %s)" % (tuple(output.shape), expected))
    print("  Output dtype: %s" % output.dtype)
    print("  NaN: %s" % ("FAIL" if torch.isnan(output).any() else "PASS"))
    print("  Inf: %s" % ("FAIL" if torch.isinf(output).any() else "PASS"))
    if tuple(output.shape) == expected:
        print("\n  *** 48-LAYER COMPILE + LOAD + FORWARD: PASSED ***")
    else:
        print("\n  *** SHAPE MISMATCH -- FAILED ***")


if __name__ == "__main__":
    main()
