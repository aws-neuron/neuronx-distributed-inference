#!/usr/bin/env python3
"""Compile code2wav (vocoder) on Neuron.

code2wav is the ConvNeXt + upsample + BigVGAN stack that maps 16-channel codec
tokens to 24 kHz audio. It ran on CPU in the streaming bench, spending ~390 ms
on the first chunk and blocking TTFB.

The model is a fixed-size graph given a fixed input length T (in codec tokens).
We trace one NEFF per bucket and dispatch at runtime by rounding T up to the
next bucket. Compile via ``torch_neuronx.trace`` (not the SPMD ModelBuilder) —
single-core, fp32 weights, no tensor parallelism.

Output: /tmp/qwen3_omni_compiled/code2wav_buckets/model_T{T}.pt for each T.

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  NEURON_RT_VISIBLE_CORES=0-7 python compile_code2wav.py
"""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0-7")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import time
from pathlib import Path

import torch
import torch_neuronx
from transformers import Qwen3OmniMoeForConditionalGeneration

MODEL_PATH = "/home/ubuntu/models/Qwen3-Omni-30B-A3B-Instruct"
# Bucket set tuned for the streaming bench:
#   * streaming chunk: CHUNK_SIZE + LEFT_CTX = 25 + 5 = 30
#   * finalize chunk (tail): LEFT_CTX + 0..CHUNK_SIZE-1 ≤ 30
#   * non-streaming `chunked_decode` default: chunk_size=300 + left_context=25
# We cover the streaming sizes and a large bucket for safety.
DEFAULT_BUCKETS = [30, 50, 128, 300, 512]


class Code2WavWrapper(torch.nn.Module):
    """Wraps ``Qwen3OmniMoeCode2Wav.forward`` so it is trace-friendly.

    The original forward does a shape check that raises a Python error if
    codes.shape[1] != num_quantizers. We keep that check out of the trace
    (it's a static invariant) and only expose the compute.
    """

    def __init__(self, c2w):
        super().__init__()
        self.c2w = c2w

    def forward(self, codes):
        # codes: [1, num_quantizers=16, T], long
        c2w = self.c2w
        hidden = c2w.code_embedding(codes + c2w.code_offset).mean(1)
        hidden = c2w.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in c2w.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in c2w.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def compile_one(c2w_wrapper, T, out_path):
    example = torch.randint(0, 2048, (1, 16, T), dtype=torch.long)
    print(f"  tracing T={T} ...")
    t0 = time.time()
    traced = torch_neuronx.trace(
        c2w_wrapper,
        example,
        compiler_workdir=f"/tmp/c2w_workdir_T{T}",
        # fp32 for correctness; c2w is fairly small so cost is modest.
        compiler_args="--auto-cast=none",
    )
    traced.save(str(out_path))
    # Quick sanity: run once
    out = traced(example)
    print(f"    done in {time.time()-t0:.0f}s, out shape={tuple(out.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/qwen3_omni_compiled/code2wav_buckets")
    parser.add_argument("--buckets", nargs="*", type=int, default=DEFAULT_BUCKETS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF model (we only need .code2wav) ...")
    t0 = time.time()
    hf_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu",
    )
    hf_model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    wrapper = Code2WavWrapper(hf_model.code2wav).eval()

    for T in args.buckets:
        out_path = out_dir / f"model_T{T}.pt"
        if out_path.exists():
            print(f"T={T}: already compiled at {out_path}, skipping")
            continue
        print(f"T={T}: compiling to {out_path}")
        compile_one(wrapper, T, out_path)


if __name__ == "__main__":
    main()
