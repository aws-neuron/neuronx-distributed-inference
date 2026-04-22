#!/usr/bin/env python3
"""
End-to-end speech synthesis for Qwen2.5-Omni-7B on NeuronX (TP=4).

Full pipeline: Thinker (text) -> Talker (codec tokens) -> Token2Wav (audio).

All three Neuron-compiled components (Thinker, Talker, Token2Wav DiT) are
loaded *once* into the same Python process on the same NeuronCores (TP=4,
core 0-3) and reused across runs. The first inference still pays the
full model-load cost, but subsequent runs are pure inference.

Two-step workflow:
  Step 1: Compile all Neuron components (one-time, ~30 min)
  Step 2: Run inference

Prerequisites:
  - Trn2 instance (trn2.48xlarge or trn2.xlarge, 4+ NeuronCores)
  - Neuron SDK 2.23+ with PyTorch 2.9
  - Model weights downloaded from Qwen/Qwen2.5-Omni-7B (auto-fetched on first run)
  - pip install soundfile

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd neuronx-distributed-inference

  # Step 1: Compile (one-time)
  python examples/generate_qwen25_omni_speech.py --compile

  # Step 2: Run inference
  python examples/generate_qwen25_omni_speech.py
  python examples/generate_qwen25_omni_speech.py --prompt "Tell me about the weather"
  python examples/generate_qwen25_omni_speech.py --speaker Chelsie --output hello.wav

  # Benchmark: load each model once, run N inferences, report avg latency
  python examples/generate_qwen25_omni_speech.py --num-runs 5
"""

# --- Qwen2.5-Omni contrib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
import _upstream_compat  # noqa: F401  (applies hf_adapter shim)
# --- end bootstrap ---

import argparse
import gc
import os
import sys
import time

import torch

try:
    import soundfile as sf
except ImportError:
    sys.exit(
        "soundfile is required for WAV output. Install with: pip install soundfile"
    )

from _model_path import resolve_model_path

MODEL_PATH = resolve_model_path()
COMPILED_PATH = os.environ.get(
    "QWEN25_OMNI_COMPILED_PATH", "/tmp/qwen25_omni_compiled"
)
TP_DEGREE = int(os.environ.get("QWEN25_OMNI_TP_DEGREE", "4"))

DEFAULT_PROMPT = "Say hello and briefly introduce yourself in two sentences."
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
DEFAULT_SPEAKER = "Ethan"

_ORIG_EMBEDDING_FORWARD = torch.nn.Embedding.forward


def _restore_embedding():
    """Restore original Embedding.forward if Neuron loading changed it."""
    if torch.nn.Embedding.forward is not _ORIG_EMBEDDING_FORWARD:
        torch.nn.Embedding.forward = _ORIG_EMBEDDING_FORWARD


class Timer:
    def __init__(self, label):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"  [{self.label}] {self.elapsed:.2f}s")


# ==========================================================================
# Compilation (--compile)
# ==========================================================================

def _compile_thinker(model_path, out_path):
    from neuronx_distributed_inference.models.config import (
        NeuronConfig, OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM, Qwen25OmniInferenceConfig,
    )

    nc = NeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, seq_len=2048, max_context_length=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.8,
        ),
    )
    cfg = Qwen25OmniInferenceConfig(nc, load_config=load_pretrained_config(model_path))
    model = NeuronQwen25OmniForCausalLM(model_path, cfg)
    model.compile(out_path)


def _compile_talker(model_path, out_path):
    from transformers import AutoConfig
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalkerForCausalLM, TalkerInferenceConfig, TalkerNeuronConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tc = hf.talker_config

    tnc = TalkerNeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, seq_len=2048, max_context_length=2048,
        torch_dtype=torch.bfloat16,
    )
    tic = TalkerInferenceConfig(
        neuron_config=tnc, load_config=load_pretrained_config(hf_config=tc),
    )
    talker = NeuronQwen25OmniTalkerForCausalLM(model_path, config=tic)
    talker.compile(out_path)


def _compile_dit(model_path, out_path):
    from transformers import AutoConfig
    from safetensors.torch import load_file
    from modeling_qwen25_omni_token2wav import (
        NeuronQwen25OmniToken2WavWithNeuronDiT,
    )

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    t2w = NeuronQwen25OmniToken2WavWithNeuronDiT(hf_config.token2wav_config)

    state_dict = {}
    for fn in sorted(os.listdir(model_path)):
        if fn.endswith(".safetensors"):
            sd = load_file(os.path.join(model_path, fn))
            for k, v in sd.items():
                if k.startswith("token2wav."):
                    state_dict[k[len("token2wav."):]] = v
    t2w.load_state_dict(state_dict, strict=False)
    t2w.compile_dit(out_path, max_mel_len=2048, batch_size=2)


def compile_all(model_path, compiled_path):
    """Compile all three Neuron components: Thinker, Talker, DiT.

    Each component is compiled sequentially in the current process. Compilation
    holds the Neuron compiler (not the runtime) so there's no core-conflict
    issue even when all three share TP=4 / core 0-3.
    """
    print("=" * 60)
    print("Compiling Qwen2.5-Omni Speech Components")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Output:   {compiled_path}")
    print(f"  TP:       {TP_DEGREE}")
    t_total = time.time()

    stages = [
        ("Thinker",  "thinker_tp4", ["neuron_config.json"],                  _compile_thinker),
        ("Talker",   "talker_tp4",  ["neuron_config.json"],                  _compile_talker),
        # DiT has two possible artifacts: TP-replicated directory (current)
        # or the legacy single-file .pt (pre-TP rewrite).
        ("DiT",      "dit_core",    ["dit_core_parallel", "dit_core_neuron.pt"], _compile_dit),
    ]
    for idx, (label, subdir, markers, fn) in enumerate(stages, 1):
        print(f"\n--- [{idx}/{len(stages)}] Compiling {label} ---")
        out_path = os.path.join(compiled_path, subdir)
        if any(os.path.exists(os.path.join(out_path, m)) for m in markers):
            print("  Already compiled, skipping.")
            continue
        t0 = time.time()
        fn(model_path, out_path)
        print(f"  {label} compiled in {time.time() - t0:.1f}s")

    print(f"\nAll components compiled in {time.time() - t_total:.0f}s")
    print(f"Artifacts saved to: {compiled_path}/")
    return True


# ==========================================================================
# Inference: model loading (once per process)
# ==========================================================================

def _check_compiled(compiled_path):
    """Confirm that each component has been compiled.

    Accepts both the new TP-replicated DiT artifact (``dit_core_parallel/``)
    and the legacy single-device one (``dit_core_neuron.pt``).
    """
    checks = [
        ([os.path.join(compiled_path, "thinker_tp4", "neuron_config.json")], "Thinker"),
        ([os.path.join(compiled_path, "talker_tp4",  "neuron_config.json")], "Talker"),
        (
            [
                os.path.join(compiled_path, "dit_core", "dit_core_parallel"),
                os.path.join(compiled_path, "dit_core", "dit_core_neuron.pt"),
            ],
            "DiT",
        ),
    ]
    missing = [name for paths, name in checks if not any(os.path.exists(p) for p in paths)]
    if missing:
        print(f"ERROR: Missing compiled artifacts for: {', '.join(missing)}")
        print(f"Run with --compile first:")
        print(f"  python {sys.argv[0]} --compile")
        return False
    return True


def load_thinker(model_path, compiled_path):
    """Load the Thinker (Qwen2.5-Omni text model) onto Neuron, return (adapter, tokenizer)."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig, OnDeviceSamplingConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config, HuggingFaceGenerationAdapter,
    )
    from modeling_qwen25_omni import (
        NeuronQwen25OmniForCausalLM, Qwen25OmniInferenceConfig,
    )
    from transformers import AutoTokenizer

    nc = NeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, seq_len=2048, max_context_length=2048,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True, temperature=0.7, top_k=20, top_p=0.8,
        ),
    )
    cfg = Qwen25OmniInferenceConfig(nc, load_config=load_pretrained_config(model_path))
    model = NeuronQwen25OmniForCausalLM(model_path, cfg)

    t0 = time.time()
    model.load(os.path.join(compiled_path, "thinker_tp4"))
    load_time = time.time() - t0
    print(f"  [Thinker] loaded in {load_time:.1f}s")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    adapter = HuggingFaceGenerationAdapter(model)

    # Warmup the NEFF so the first real inference isn't artificially slow.
    enc = tok(
        tok.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False, add_generation_prompt=True,
        ),
        return_tensors="pt",
    )
    _ = adapter.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=5, eos_token_id=[tok.eos_token_id, 151645],
    )
    print("  [Thinker] warmup done")
    return adapter, tok, load_time


def load_talker(model_path, compiled_path):
    """Load the Talker model onto Neuron and return (talker, adapter, talker_config)."""
    from transformers import AutoConfig
    from modeling_qwen25_omni_talker import (
        NeuronQwen25OmniTalkerForCausalLM, TalkerInferenceConfig, TalkerNeuronConfig,
    )
    from neuronx_distributed_inference.utils.hf_adapter import (
        load_pretrained_config, HuggingFaceGenerationAdapter,
    )

    hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tc = hf.talker_config

    tnc = TalkerNeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, seq_len=2048, max_context_length=2048,
        torch_dtype=torch.bfloat16,
    )
    tic = TalkerInferenceConfig(
        neuron_config=tnc, load_config=load_pretrained_config(hf_config=tc),
    )
    talker = NeuronQwen25OmniTalkerForCausalLM(model_path, config=tic)

    t0 = time.time()
    talker.load(os.path.join(compiled_path, "talker_tp4"))
    load_time = time.time() - t0
    print(f"  [Talker]  loaded in {load_time:.1f}s")

    adapter = HuggingFaceGenerationAdapter(talker)
    return talker, adapter, tc, load_time


def load_token2wav(model_path, compiled_path):
    """Load the Token2Wav model (DiT on Neuron + BigVGAN on CPU)."""
    from transformers import AutoConfig
    from safetensors.torch import load_file
    from modeling_qwen25_omni_token2wav import (
        NeuronQwen25OmniToken2WavWithNeuronDiT,
    )

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    t2w_cfg = hf_config.token2wav_config

    t2w = NeuronQwen25OmniToken2WavWithNeuronDiT(t2w_cfg)

    state_dict = {}
    for fn in sorted(os.listdir(model_path)):
        if fn.endswith(".safetensors"):
            sd = load_file(os.path.join(model_path, fn))
            for k, v in sd.items():
                if k.startswith("token2wav."):
                    state_dict[k[len("token2wav."):]] = v
    t2w.load_state_dict(state_dict, strict=False)

    t0 = time.time()
    t2w.load_dit(os.path.join(compiled_path, "dit_core"))
    load_time = time.time() - t0
    _restore_embedding()
    print(f"  [DiT]     loaded in {load_time:.1f}s")
    return t2w, t2w_cfg, load_time


def load_hf_cpu(model_path):
    """Load the HF Qwen2.5-Omni model on CPU in bfloat16 (for hidden-state extraction)."""
    from transformers import Qwen2_5OmniForConditionalGeneration

    t0 = time.time()
    hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    hf_model.eval()
    _restore_embedding()
    load_time = time.time() - t0
    print(f"  [HF CPU]  loaded in {load_time:.1f}s")
    return hf_model, load_time


# ==========================================================================
# Inference: per-run phases
# ==========================================================================

def run_thinker(thinker_adapter, tokenizer, prompt, system_prompt):
    """Phase 1: Thinker generates text."""
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    enc = tokenizer(
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
        return_tensors="pt",
    )

    t0 = time.time()
    out = thinker_adapter.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=200, eos_token_id=[tokenizer.eos_token_id, 151645],
    )
    elapsed = time.time() - t0

    prompt_len = enc["input_ids"].shape[1]
    all_ids = out[0].tolist()
    gen_ids = all_ids[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "all_ids": all_ids,
        "prompt_len": prompt_len,
        "gen_text": text,
        "n_tokens": len(gen_ids),
        "gen_time": elapsed,
    }


def extract_hidden_states(hf_model, thinker_result):
    """Phase 2: Run HF Thinker on CPU to capture hidden states.

    The compiled Neuron Thinker uses on-device sampling and only emits tokens,
    not hidden states. The Talker needs the per-token last-layer hidden states
    to condition on, so we re-run the prompt+reply through the HF CPU model in
    bfloat16 — all downstream consumers already round back to bf16 so float32
    here would be pure overhead.
    """
    full_ids = torch.tensor([thinker_result["all_ids"]], dtype=torch.long)
    prompt_len = thinker_result["prompt_len"]

    t0 = time.time()
    with torch.no_grad():
        outputs = hf_model.thinker(
            input_ids=full_ids, output_hidden_states=True, return_dict=True,
        )
    elapsed = time.time() - t0
    return outputs, full_ids, prompt_len, elapsed


def prepare_talker_input(model_path, hf_model, outputs, full_ids, prompt_len, speaker):
    """Phase 3: Build projected thinker states for the Talker."""
    from transformers import AutoConfig
    from safetensors.torch import load_file
    from modeling_qwen25_omni_talker import ThinkerToTalkerProjection

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    talker_cfg = hf_config.talker_config

    spk_dict = torch.load(os.path.join(model_path, "spk_dict.pt"), weights_only=True)
    sp = spk_dict[speaker]
    conditioning = sp["cond"].unsqueeze(0).float() if sp["cond"].dim() == 1 else sp["cond"].float()
    if conditioning.dim() == 1:
        conditioning = conditioning.unsqueeze(0)
    reference_mel = sp["ref_mel"].unsqueeze(0).float() if sp["ref_mel"].dim() == 2 else sp["ref_mel"].float()
    if reference_mel.dim() == 2:
        reference_mel = reference_mel.unsqueeze(0)
    bos_token = sp["bos_token"]
    if isinstance(bos_token, torch.Tensor):
        bos_token = bos_token.item()

    embedding_output = outputs.hidden_states[0]
    last_hidden = outputs.hidden_states[-1]
    total_len = full_ids.shape[1]

    context_embed = embedding_output[:, :prompt_len, :]
    context_hidden = last_hidden[:, :prompt_len, :]
    reply_embeds = [embedding_output[:, i:i+1, :] for i in range(prompt_len, total_len)]
    reply_hiddens = [last_hidden[:, i:i+1, :] for i in range(prompt_len, total_len)]

    thinker_token_embeds = [context_embed] + reply_embeds
    thinker_hidden_states_list = [context_hidden] + reply_hiddens

    thinker_reply_part = (
        torch.cat(thinker_hidden_states_list[1:], dim=1)
        + torch.cat(thinker_token_embeds[1:], dim=1)
    )
    talker_inputs_embeds = thinker_hidden_states_list[0] + thinker_token_embeds[0]

    thinker_embed_tokens = hf_model.thinker.get_input_embeddings()
    bos_embed = thinker_embed_tokens(torch.tensor([[bos_token]], dtype=torch.long))
    talker_inputs_embeds = torch.cat([
        talker_inputs_embeds, bos_embed, thinker_reply_part[:, :1, :],
    ], dim=1)

    talker_embed_weight = None
    for fn in sorted(os.listdir(model_path)):
        if fn.endswith(".safetensors"):
            sd = load_file(os.path.join(model_path, fn))
            if "talker.model.embed_tokens.weight" in sd:
                talker_embed_weight = sd["talker.model.embed_tokens.weight"]
                break
    if talker_embed_weight is not None:
        talker_embed_layer = torch.nn.Embedding(
            talker_embed_weight.shape[0], talker_embed_weight.shape[1],
        )
        talker_embed_layer.weight.data = talker_embed_weight.float()
        codec_bos_embed = talker_embed_layer(
            torch.tensor([talker_cfg.tts_codec_start_token_id]),
        )
        codec_pad_embed = talker_embed_layer(
            torch.tensor([talker_cfg.tts_codec_pad_token_id]),
        )
        talker_inputs_embeds[:, -1, :] += codec_bos_embed
        talker_inputs_embeds[:, -2, :] += codec_pad_embed

    eos_embed = thinker_embed_tokens(
        torch.tensor([[talker_cfg.tts_text_end_token_id]], dtype=torch.long),
    )
    pad_embed = thinker_embed_tokens(
        torch.tensor([[talker_cfg.tts_text_pad_token_id]], dtype=torch.long),
    )
    thinker_reply_part = torch.cat([thinker_reply_part[:, 1:, :], eos_embed, pad_embed], dim=1)

    context_len = talker_inputs_embeds.shape[1]
    n_reply = thinker_reply_part.shape[1]

    proj_weight = proj_bias = None
    for k, v in hf_model.state_dict().items():
        if "thinker_to_talker_proj.weight" in k:
            proj_weight = v
        if "thinker_to_talker_proj.bias" in k:
            proj_bias = v

    proj = ThinkerToTalkerProjection(proj_weight.shape[1], proj_weight.shape[0])
    proj.proj.weight.data = proj_weight
    if proj_bias is not None:
        proj.proj.bias.data = proj_bias
    proj.to(proj_weight.dtype)

    projected_context = proj(talker_inputs_embeds)
    projected_reply = proj(thinker_reply_part)

    return {
        "projected_context": projected_context,
        "projected_reply": projected_reply,
        "context_len": context_len,
        "n_reply": n_reply,
        "conditioning": conditioning,
        "reference_mel": reference_mel,
    }


def run_talker(talker_model, talker_adapter, talker_cfg, talker_input):
    """Phase 4: Talker generates codec tokens."""
    projected_context = talker_input["projected_context"]
    projected_reply = talker_input["projected_reply"]
    context_len = talker_input["context_len"]

    codec_bos = talker_cfg.tts_codec_start_token_id
    codec_eos = talker_cfg.tts_codec_end_token_id
    codec_pad = talker_cfg.tts_codec_pad_token_id
    codec_mask = talker_cfg.tts_codec_mask_token_id

    talker_input_ids = torch.cat([
        torch.full((1, context_len - 2), codec_mask, dtype=torch.long),
        torch.tensor([[codec_pad]], dtype=torch.long),
        torch.tensor([[codec_bos]], dtype=torch.long),
    ], dim=1)
    talker_attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long)

    max_gen = min(600, 2048 - context_len - 10)

    # Re-set vision embeddings before each run (context encoding consumes them).
    ve = projected_context.to(torch.bfloat16)
    vm = torch.ones(1, context_len, 1, dtype=torch.int32)
    reply = projected_reply.to(torch.bfloat16)
    talker_model.set_vision_embeddings(ve, vm, thinker_reply_embeds=reply)

    t0 = time.time()
    out = talker_adapter.generate(
        input_ids=talker_input_ids,
        attention_mask=talker_attention_mask,
        max_new_tokens=max_gen,
        eos_token_id=[codec_eos, codec_pad],
        suppress_tokens=[codec_bos],
        do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
        repetition_penalty=1.05,
    )
    elapsed = time.time() - t0

    gen_tokens = out[0, context_len:].tolist()
    while gen_tokens and gen_tokens[-1] == codec_eos:
        gen_tokens.pop()
    return gen_tokens, elapsed


def run_token2wav(t2w, t2w_cfg, codec_codes, conditioning, reference_mel):
    """Phase 5: Token2Wav DiT + BigVGAN synthesize a waveform."""
    code_tensor = torch.tensor([codec_codes], dtype=torch.long)
    num_embeds = getattr(t2w_cfg.dit_config, "num_embeds", 8193)
    if code_tensor.max() >= num_embeds:
        code_tensor = code_tensor.clamp(0, num_embeds)

    t0 = time.time()
    wav = t2w(
        code=code_tensor,
        conditioning=conditioning,
        reference_mel=reference_mel,
        num_steps=10,
        guidance_scale=0.5,
    )
    elapsed = time.time() - t0
    return wav, elapsed


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni-7B speech synthesis on Neuron"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile all Neuron components (one-time, ~30 min)",
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of inference runs per component for benchmarking (default: 1)",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Text prompt for speech generation",
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM,
        help="System prompt",
    )
    parser.add_argument(
        "--speaker", default=DEFAULT_SPEAKER, choices=["Ethan", "Chelsie"],
        help="Speaker voice (default: Ethan)",
    )
    parser.add_argument(
        "--model-path", default=MODEL_PATH,
        help=f"Model path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--compiled-path", default=COMPILED_PATH,
        help=f"Compiled artifacts path (default: {COMPILED_PATH})",
    )
    parser.add_argument(
        "--output", default="speech_output.wav",
        help="Output WAV file path (default: speech_output.wav)",
    )
    args = parser.parse_args()

    model_path = args.model_path
    compiled_path = args.compiled_path
    num_runs = args.num_runs

    if args.compile:
        ok = compile_all(model_path, compiled_path)
        sys.exit(0 if ok else 1)

    if not _check_compiled(compiled_path):
        sys.exit(1)

    print("=" * 60)
    print("Qwen2.5-Omni Speech Pipeline (Neuron, TP=4, single process)")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Compiled: {compiled_path}")
    print(f"  Speaker:  {args.speaker}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Output:   {args.output}")
    print(f"  Runs:     {num_runs}")
    t_total = time.time()

    # ----- Load everything once -----
    print("\n--- Loading models (one-time cost) ---")
    t_load_total = time.time()
    thinker_adapter, tokenizer, thinker_load = load_thinker(model_path, compiled_path)
    hf_model, hf_load = load_hf_cpu(model_path)
    talker_model, talker_adapter, talker_cfg, talker_load = load_talker(model_path, compiled_path)
    t2w, t2w_cfg, dit_load = load_token2wav(model_path, compiled_path)
    total_load = time.time() - t_load_total
    print(f"  Total model load time: {total_load:.1f}s")

    # ----- Run the pipeline num_runs times -----
    thinker_times, talker_times, t2w_times = [], [], []
    hidden_times, prep_times = [], []
    first_text = first_codes = first_wav = None
    first_audio_duration = 0.0

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")

        thinker_result = run_thinker(
            thinker_adapter, tokenizer, args.prompt, args.system_prompt,
        )
        thinker_times.append(thinker_result["gen_time"])
        print(
            f"  [Thinker]   {thinker_result['n_tokens']} tokens in "
            f"{thinker_result['gen_time']:.3f}s - {thinker_result['gen_text'][:80]}"
        )

        outputs, full_ids, prompt_len, hidden_time = extract_hidden_states(
            hf_model, thinker_result,
        )
        hidden_times.append(hidden_time)
        print(f"  [Hidden]    forward pass in {hidden_time:.2f}s")

        t0 = time.time()
        talker_input = prepare_talker_input(
            model_path, hf_model, outputs, full_ids, prompt_len, args.speaker,
        )
        prep_time = time.time() - t0
        prep_times.append(prep_time)
        print(
            f"  [Prep]      context={talker_input['context_len']} tokens, "
            f"reply={talker_input['n_reply']} tokens ({prep_time:.2f}s)"
        )

        codec_codes, talker_time = run_talker(
            talker_model, talker_adapter, talker_cfg, talker_input,
        )
        talker_times.append(talker_time)
        print(f"  [Talker]    {len(codec_codes)} codec tokens in {talker_time:.3f}s")
        if not codec_codes:
            print("  Talker produced no tokens, aborting run.")
            continue

        wav, t2w_time = run_token2wav(
            t2w, t2w_cfg, codec_codes,
            talker_input["conditioning"], talker_input["reference_mel"],
        )
        t2w_times.append(t2w_time)
        print(f"  [Token2Wav] synthesized in {t2w_time:.2f}s")

        if first_text is None:
            first_text = thinker_result["gen_text"]
            first_codes = codec_codes
            first_wav = wav

        # Free the per-run temporaries so the heap doesn't grow across runs.
        del outputs, full_ids, talker_input
        gc.collect()

    # ----- Write first run's audio -----
    if first_wav is not None and isinstance(first_wav, torch.Tensor) and first_wav.numel() > 0:
        wav_np = first_wav.detach().cpu().float().numpy().flatten()
        sf.write(args.output, wav_np, 24000)
        first_audio_duration = len(wav_np) / 24000
        print(f"\n  Audio: {first_audio_duration:.1f}s saved to {args.output}")

    total_time = time.time() - t_total

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if first_text:
        print(f"  Text:      {first_text[:200]}")
    print("\n  Model load time (one-time cost, excluded from pipeline avg):")
    print(f"    Thinker:   {thinker_load:.1f}s")
    print(f"    HF CPU:    {hf_load:.1f}s")
    print(f"    Talker:    {talker_load:.1f}s")
    print(f"    DiT:       {dit_load:.1f}s")
    print(f"    Total:     {total_load:.1f}s")
    print(f"\n  Per-run latency (avg of {num_runs} runs):")
    print(f"    Thinker:     {_avg(thinker_times):.3f}s")
    print(f"    Hidden:      {_avg(hidden_times):.3f}s (HF CPU forward)")
    print(f"    Prep:        {_avg(prep_times):.3f}s")
    print(f"    Talker:      {_avg(talker_times):.3f}s")
    print(f"    Token2Wav:   {_avg(t2w_times):.2f}s")
    pipeline_avg = (
        _avg(thinker_times) + _avg(hidden_times) + _avg(prep_times)
        + _avg(talker_times) + _avg(t2w_times)
    )
    print(f"    Pipeline:    {pipeline_avg:.2f}s total")
    if first_audio_duration > 0:
        print(f"\n  Audio:     {first_audio_duration:.1f}s")
        print(f"  RTF:       {pipeline_avg/first_audio_duration:.2f}x")
    if num_runs > 1:
        print(f"\n  Per-run breakdown ({num_runs} runs):")
        print(f"    Thinker:     {['%.3f' % t for t in thinker_times]}")
        print(f"    Hidden:      {['%.3f' % t for t in hidden_times]}")
        print(f"    Talker:      {['%.3f' % t for t in talker_times]}")
        print(f"    Token2Wav:   {['%.2f' % t for t in t2w_times]}")
    print(f"\n  Wall time:   {total_time:.1f}s (load + {num_runs} run(s))")
    print(f"  Output:      {args.output}")


if __name__ == "__main__":
    main()
