#!/usr/bin/env python3
"""
End-to-end speech synthesis for Qwen2.5-Omni-7B on NeuronX (TP=4).

Full pipeline: Thinker (text) -> Talker (codec tokens) -> Token2Wav (audio)

Two-step workflow:
  Step 1: Compile all Neuron components (one-time, ~30 min)
  Step 2: Run inference (loads compiled artifacts, ~15s per utterance)

Architecture note:
  Thinker (TP=4) and Talker (TP=4) each require exclusive Neuron access,
  so they run in separate subprocesses. Within each subprocess, the model
  is loaded ONCE and reused for all --num-runs iterations.
  Token2Wav DiT runs in the main process.

Prerequisites:
  - Trn2 instance (trn2.48xlarge or trn2.xlarge, 4+ NeuronCores)
  - Neuron SDK 2.23+ with PyTorch 2.9
  - Model weights: huggingface-cli download Qwen/Qwen2.5-Omni-7B

Usage:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
  cd neuronx-distributed-inference

  # Step 1: Compile (one-time, ~30 min)
  python examples/generate_qwen25_omni_speech.py --compile

  # Step 2: Run inference
  python examples/generate_qwen25_omni_speech.py
  python examples/generate_qwen25_omni_speech.py --prompt "Tell me about the weather"
  python examples/generate_qwen25_omni_speech.py --speaker Chelsie --output hello.wav

  # Benchmark: load each model once, run N inferences, report avg latency
  python examples/generate_qwen25_omni_speech.py --num-runs 5

Pipeline timing (trn2.48xlarge, TP=4, model already loaded):
  Thinker:  ~0.3s  (text generation)
  Talker:   ~2-3s  (codec token generation)
  Token2Wav: ~10s  (mel spectrogram + vocoder)
  Total:    ~15s   for ~10s of audio (RTF ~1.5x)
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
import json
import os
import subprocess
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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


_SUBPROCESS_BOOTSTRAP = (
    "import sys, os\n"
    f"sys.path.insert(0, {str(_SRC)!r})\n"
    "import _upstream_compat  # noqa: F401\n"
)


def _run_subprocess(script_code, label, temp_dir):
    """Run Python code as a subprocess (required for Neuron process isolation)."""
    script_path = os.path.join(temp_dir, f"{label}.py")
    with open(script_path, "w") as f:
        f.write(_SUBPROCESS_BOOTSTRAP + script_code)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC}{os.pathsep}{existing}" if existing else str(_SRC)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=600, env=env,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [{label}] FAILED ({elapsed:.1f}s)")
        for line in result.stderr.strip().split("\n")[-15:]:
            if line.strip() and not any(
                x in line for x in ["WARN", "TDRV", "NMGR", "NRT", "nccl", "blockwise"]
            ):
                print(f"    {line}")
        for line in result.stdout.strip().split("\n")[-5:]:
            if line.strip():
                print(f"    {line}")
        return False
    print(f"  [{label}] subprocess finished ({elapsed:.1f}s)")
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"    {line}")
    return True


# ==========================================================================
# Compilation (--compile)
# ==========================================================================

def compile_all(model_path, compiled_path):
    """Compile all three Neuron components: Thinker, Talker, DiT."""
    print("=" * 60)
    print("Compiling Qwen2.5-Omni Speech Components")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Output:   {compiled_path}")
    print(f"  TP:       {TP_DEGREE}")
    t_total = time.time()

    temp_dir = os.path.join(compiled_path, "_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # --- 1. Compile Thinker ---
    print("\n--- [1/3] Compiling Thinker ---")
    thinker_compiled = os.path.join(compiled_path, "thinker_tp4")
    if os.path.exists(os.path.join(thinker_compiled, "neuron_config.json")):
        print("  Already compiled, skipping.")
    else:
        script = f'''
import torch, os
MODEL_PATH = "{model_path}"
COMPILED = "{thinker_compiled}"

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM, Qwen25OmniInferenceConfig,
)

nc = NeuronConfig(
    tp_degree={TP_DEGREE}, batch_size=1, seq_len=2048, max_context_length=2048,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.7, top_k=20, top_p=0.8
    ),
)
cfg = Qwen25OmniInferenceConfig(nc, load_config=load_pretrained_config(MODEL_PATH))
model = NeuronQwen25OmniForCausalLM(MODEL_PATH, cfg)
model.compile(COMPILED)
print("Thinker compiled successfully")
'''
        ok = _run_subprocess(script, "compile_thinker", temp_dir)
        if not ok:
            print("FATAL: Thinker compilation failed.")
            return False

    # --- 2. Compile Talker ---
    print("\n--- [2/3] Compiling Talker ---")
    talker_compiled = os.path.join(compiled_path, "talker_tp4")
    if os.path.exists(os.path.join(talker_compiled, "neuron_config.json")):
        print("  Already compiled, skipping.")
    else:
        script = f'''
import torch, os
MODEL_PATH = "{model_path}"
COMPILED = "{talker_compiled}"

from transformers import AutoConfig
from modeling_qwen25_omni_talker import (
    NeuronQwen25OmniTalkerForCausalLM, TalkerInferenceConfig, TalkerNeuronConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

hf = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tc = hf.talker_config

tnc = TalkerNeuronConfig(
    tp_degree={TP_DEGREE}, batch_size=1, seq_len=2048, max_context_length=2048,
    torch_dtype=torch.bfloat16,
)
tic = TalkerInferenceConfig(neuron_config=tnc, load_config=load_pretrained_config(hf_config=tc))
talker = NeuronQwen25OmniTalkerForCausalLM(MODEL_PATH, config=tic)
talker.compile(COMPILED)
print("Talker compiled successfully")
'''
        ok = _run_subprocess(script, "compile_talker", temp_dir)
        if not ok:
            print("FATAL: Talker compilation failed.")
            return False

    # --- 3. Compile DiT ---
    print("\n--- [3/3] Compiling Token2Wav DiT ---")
    dit_compiled = os.path.join(compiled_path, "dit_core")
    if os.path.exists(os.path.join(dit_compiled, "dit_core_neuron.pt")):
        print("  Already compiled, skipping.")
    else:
        script = f'''
import torch, os
MODEL_PATH = "{model_path}"
COMPILED = "{dit_compiled}"

from transformers import AutoConfig
from safetensors.torch import load_file
from modeling_qwen25_omni_token2wav import (
    NeuronQwen25OmniToken2WavWithNeuronDiT,
)

hf_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
t2w = NeuronQwen25OmniToken2WavWithNeuronDiT(hf_config.token2wav_config)

state_dict = {{}}
for fn in sorted(os.listdir(MODEL_PATH)):
    if fn.endswith(".safetensors"):
        sd = load_file(os.path.join(MODEL_PATH, fn))
        for k, v in sd.items():
            if k.startswith("token2wav."):
                state_dict[k[len("token2wav."):]] = v
t2w.load_state_dict(state_dict, strict=False)

t2w.compile_dit(COMPILED, max_mel_len=2048, batch_size=2)
print("DiT compiled successfully")
'''
        ok = _run_subprocess(script, "compile_dit", temp_dir)
        if not ok:
            print("FATAL: DiT compilation failed.")
            return False

    total = time.time() - t_total
    print(f"\nAll components compiled in {total:.0f}s")
    print(f"Artifacts saved to: {compiled_path}/")
    print(f"  thinker_tp4/    - Thinker (7B text model)")
    print(f"  talker_tp4/     - Talker (690M codec model)")
    print(f"  dit_core/       - Token2Wav DiT (85M transformer)")
    return True


# ==========================================================================
# Inference
# ==========================================================================

def _check_compiled(compiled_path):
    """Verify all compiled artifacts exist."""
    checks = [
        (os.path.join(compiled_path, "thinker_tp4", "neuron_config.json"), "Thinker"),
        (os.path.join(compiled_path, "talker_tp4", "neuron_config.json"), "Talker"),
        (os.path.join(compiled_path, "dit_core", "dit_core_neuron.pt"), "DiT"),
    ]
    missing = [name for path, name in checks if not os.path.exists(path)]
    if missing:
        print(f"ERROR: Missing compiled artifacts for: {', '.join(missing)}")
        print(f"Run with --compile first:")
        print(f"  python {sys.argv[0]} --compile")
        return False
    return True


def run_thinker(model_path, compiled_path, prompt, system_prompt, num_runs, temp_dir):
    """Phase 1: Load Thinker once, generate num_runs times, return first result + avg time."""
    print(f"\n--- Phase 1: Thinker (text generation, {num_runs} runs) ---")

    thinker_compiled = os.path.join(compiled_path, "thinker_tp4")
    output_file = os.path.join(temp_dir, "thinker_output.json")

    script = f'''
import torch, os, json, time
MODEL_PATH = "{model_path}"
COMPILED = "{thinker_compiled}"
OUTPUT = "{output_file}"
NUM_RUNS = {num_runs}

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from modeling_qwen25_omni import (
    NeuronQwen25OmniForCausalLM, Qwen25OmniInferenceConfig,
)
from transformers import AutoTokenizer

nc = NeuronConfig(
    tp_degree={TP_DEGREE}, batch_size=1, seq_len=2048, max_context_length=2048,
    torch_dtype=torch.bfloat16,
    on_device_sampling_config=OnDeviceSamplingConfig(
        do_sample=True, temperature=0.7, top_k=20, top_p=0.8
    ),
)
cfg = Qwen25OmniInferenceConfig(nc, load_config=load_pretrained_config(MODEL_PATH))
model = NeuronQwen25OmniForCausalLM(MODEL_PATH, cfg)

t_load = time.time()
model.load(COMPILED)
t_load = time.time() - t_load
print(f"Model loaded in {{t_load:.1f}}s")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
adp = HuggingFaceGenerationAdapter(model)

# Warmup (first inference is slower due to Neuron warm-up)
enc = tok(tok.apply_chat_template(
    [{{"role":"user","content":"Hi"}}], tokenize=False, add_generation_prompt=True
), return_tensors="pt")
_ = adp.generate(
    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
    max_new_tokens=5, eos_token_id=[tok.eos_token_id, 151645],
)
print("Warmup done")

chat = [
    {{"role":"system","content":"{system_prompt}"}},
    {{"role":"user","content":"{prompt}"}},
]
enc = tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt")

print(f"Running {{NUM_RUNS}} inferences (model stays loaded)...")
first_result = None
times = []
for i in range(NUM_RUNS):
    t0 = time.time()
    out = adp.generate(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=200, eos_token_id=[tok.eos_token_id, 151645],
    )
    elapsed = time.time() - t0
    times.append(elapsed)

    prompt_len = enc["input_ids"].shape[1]
    all_ids = out[0].tolist()
    gen_ids = all_ids[prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    n_tokens = len(gen_ids)
    print(f"  Run {{i+1}}/{{NUM_RUNS}}: {{n_tokens}} tokens in {{elapsed:.3f}}s - {{text[:80]}}")

    if first_result is None:
        first_result = {{"all_ids": all_ids, "prompt_len": prompt_len,
                        "gen_text": text, "n_tokens": n_tokens}}

avg_time = sum(times) / len(times)
first_result["gen_time"] = avg_time
first_result["all_times"] = times
first_result["load_time"] = t_load
print(f"Avg inference: {{avg_time:.3f}}s (load: {{t_load:.1f}}s, not included in avg)")

with open(OUTPUT, "w") as f:
    json.dump(first_result, f)
'''
    ok = _run_subprocess(script, "thinker", temp_dir)
    if not ok:
        return None

    with open(output_file) as f:
        return json.load(f)


def extract_hidden_states(model_path, thinker_result):
    """Phase 2: Extract thinker hidden states via CPU forward pass."""
    print("\n--- Phase 2: Hidden state extraction (CPU) ---")
    from transformers import Qwen2_5OmniForConditionalGeneration

    with Timer("Load HF model"):
        hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True,
        )
        hf_model.eval()
    _restore_embedding()

    full_ids = torch.tensor([thinker_result["all_ids"]], dtype=torch.long)
    prompt_len = thinker_result["prompt_len"]

    with Timer("Forward pass"):
        with torch.no_grad():
            outputs = hf_model.thinker(
                input_ids=full_ids, output_hidden_states=True, return_dict=True,
            )

    return hf_model, outputs, full_ids, prompt_len


def prepare_talker_input(model_path, hf_model, outputs, full_ids, prompt_len, speaker, temp_dir):
    """Phase 3: Build projected thinker states for the Talker."""
    print("\n--- Phase 3: Talker input preparation ---")
    from transformers import AutoConfig
    from safetensors.torch import load_file
    from modeling_qwen25_omni_talker import (
        ThinkerToTalkerProjection,
    )

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
            talker_embed_weight.shape[0], talker_embed_weight.shape[1]
        )
        talker_embed_layer.weight.data = talker_embed_weight.float()
        codec_bos_embed = talker_embed_layer(
            torch.tensor([talker_cfg.tts_codec_start_token_id])
        )
        codec_pad_embed = talker_embed_layer(
            torch.tensor([talker_cfg.tts_codec_pad_token_id])
        )
        talker_inputs_embeds[:, -1, :] += codec_bos_embed
        talker_inputs_embeds[:, -2, :] += codec_pad_embed

    eos_embed = thinker_embed_tokens(
        torch.tensor([[talker_cfg.tts_text_end_token_id]], dtype=torch.long)
    )
    pad_embed = thinker_embed_tokens(
        torch.tensor([[talker_cfg.tts_text_pad_token_id]], dtype=torch.long)
    )
    thinker_reply_part = torch.cat([thinker_reply_part[:, 1:, :], eos_embed, pad_embed], dim=1)

    context_len = talker_inputs_embeds.shape[1]
    n_reply = thinker_reply_part.shape[1]
    print(f"  Context: {context_len} tokens, Reply: {n_reply} tokens")

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

    projected_context = proj(talker_inputs_embeds)
    projected_reply = proj(thinker_reply_part)

    talker_input = {
        "projected_context": projected_context,
        "projected_reply": projected_reply,
        "context_len": context_len,
        "n_reply": n_reply,
        "prompt_len": prompt_len,
        "conditioning": conditioning,
        "reference_mel": reference_mel,
    }
    torch.save(talker_input, os.path.join(temp_dir, "talker_input.pt"))

    del hf_model, outputs
    gc.collect()
    return context_len


def run_talker(model_path, compiled_path, context_len, num_runs, temp_dir):
    """Phase 4: Load Talker once, generate num_runs times, return first result + avg time."""
    print(f"\n--- Phase 4: Talker (codec token generation, {num_runs} runs) ---")

    talker_compiled = os.path.join(compiled_path, "talker_tp4")
    output_file = os.path.join(temp_dir, "talker_output.json")

    script = f'''
import torch, os, json, time
MODEL_PATH = "{model_path}"
COMPILED = "{talker_compiled}"
TEMP_DIR = "{temp_dir}"
NUM_RUNS = {num_runs}

from transformers import AutoConfig
from modeling_qwen25_omni_talker import (
    NeuronQwen25OmniTalkerForCausalLM, TalkerInferenceConfig, TalkerNeuronConfig,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

hf = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
tc = hf.talker_config

tnc = TalkerNeuronConfig(
    tp_degree={TP_DEGREE}, batch_size=1, seq_len=2048, max_context_length=2048,
    torch_dtype=torch.bfloat16,
)
tic = TalkerInferenceConfig(neuron_config=tnc, load_config=load_pretrained_config(hf_config=tc))
talker = NeuronQwen25OmniTalkerForCausalLM(MODEL_PATH, config=tic)

t_load = time.time()
talker.load(COMPILED)
t_load = time.time() - t_load
print(f"Model loaded in {{t_load:.1f}}s")

adp = HuggingFaceGenerationAdapter(talker)

inp = torch.load(os.path.join(TEMP_DIR, "talker_input.pt"), weights_only=False)
projected_context = inp["projected_context"]
projected_reply = inp["projected_reply"]
context_len = inp["context_len"]

codec_bos = tc.tts_codec_start_token_id
codec_eos = tc.tts_codec_end_token_id
codec_pad = tc.tts_codec_pad_token_id
codec_mask = tc.tts_codec_mask_token_id

talker_input_ids = torch.cat([
    torch.full((1, context_len - 2), codec_mask, dtype=torch.long),
    torch.tensor([[codec_pad]], dtype=torch.long),
    torch.tensor([[codec_bos]], dtype=torch.long),
], dim=1)
talker_attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long)

max_gen = min(600, 2048 - context_len - 10)

print(f"Running {{NUM_RUNS}} inferences (model stays loaded)...")
first_codes = None
times = []
for i in range(NUM_RUNS):
    # Re-set vision embeddings before each run (context encoding consumes them)
    ve = projected_context.to(torch.bfloat16)
    vm = torch.ones(1, context_len, 1, dtype=torch.int32)
    reply = projected_reply.to(torch.bfloat16)
    talker.set_vision_embeddings(ve, vm, thinker_reply_embeds=reply)

    t0 = time.time()
    out = adp.generate(
        input_ids=talker_input_ids,
        attention_mask=talker_attention_mask,
        max_new_tokens=max_gen,
        eos_token_id=[codec_eos, codec_pad],
        suppress_tokens=[codec_bos],
        do_sample=True, temperature=0.9, top_k=40, top_p=0.8,
        repetition_penalty=1.05,
    )
    elapsed = time.time() - t0
    times.append(elapsed)

    gen_tokens = out[0, context_len:].tolist()
    while gen_tokens and gen_tokens[-1] == codec_eos:
        gen_tokens.pop()
    print(f"  Run {{i+1}}/{{NUM_RUNS}}: {{len(gen_tokens)}} codec tokens in {{elapsed:.3f}}s")

    if first_codes is None:
        first_codes = gen_tokens

avg_time = sum(times) / len(times)
print(f"Avg inference: {{avg_time:.3f}}s (load: {{t_load:.1f}}s, not included in avg)")

result = {{"codes": first_codes, "gen_time": avg_time, "all_times": times, "load_time": t_load}}
with open(os.path.join(TEMP_DIR, "talker_output.json"), "w") as f:
    json.dump(result, f)
'''
    ok = _run_subprocess(script, "talker", temp_dir)
    if not ok:
        return None

    with open(output_file) as f:
        return json.load(f)


def run_token2wav(model_path, compiled_path, codec_codes, num_runs, temp_dir, output_wav):
    """Phase 5: Load DiT once, run num_runs times, save first result, report avg time."""
    print(f"\n--- Phase 5: Token2Wav (waveform synthesis, {num_runs} runs) ---")

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

    dit_compiled = os.path.join(compiled_path, "dit_core")
    with Timer("Load compiled DiT"):
        t2w.load_dit(dit_compiled)
    _restore_embedding()

    code_tensor = torch.tensor([codec_codes], dtype=torch.long)
    num_embeds = getattr(t2w_cfg.dit_config, "num_embeds", 8193)
    if code_tensor.max() >= num_embeds:
        code_tensor = code_tensor.clamp(0, num_embeds)

    inp = torch.load(os.path.join(temp_dir, "talker_input.pt"), weights_only=False)
    conditioning = inp["conditioning"]
    reference_mel = inp["reference_mel"]

    print(f"  Running {num_runs} inferences (DiT stays loaded)...")
    first_wav = None
    times = []
    for i in range(num_runs):
        t0 = time.time()
        wav = t2w(
            code=code_tensor,
            conditioning=conditioning,
            reference_mel=reference_mel,
            num_steps=10,
            guidance_scale=0.5,
        )
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.2f}s")

        if first_wav is None:
            first_wav = wav

    avg_time = sum(times) / len(times)
    print(f"  Avg inference: {avg_time:.2f}s")

    audio_duration = 0
    if first_wav is not None and isinstance(first_wav, torch.Tensor) and first_wav.numel() > 0:
        import soundfile as sf
        wav_np = first_wav.detach().cpu().float().numpy().flatten()
        sf.write(output_wav, wav_np, 24000)
        audio_duration = len(wav_np) / 24000
        print(f"  Audio: {audio_duration:.1f}s saved to {output_wav}")

    return audio_duration, avg_time, times


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

    # --- Compile mode ---
    if args.compile:
        ok = compile_all(model_path, compiled_path)
        sys.exit(0 if ok else 1)

    # --- Inference mode ---
    if not _check_compiled(compiled_path):
        sys.exit(1)

    temp_dir = os.path.join(compiled_path, "_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print("Qwen2.5-Omni Speech Pipeline (Neuron, TP=4)")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Compiled: {compiled_path}")
    print(f"  Speaker:  {args.speaker}")
    print(f"  Prompt:   {args.prompt}")
    print(f"  Output:   {args.output}")
    print(f"  Runs:     {num_runs}")
    t_total = time.time()

    # Phase 1: Thinker (load once, run N times in subprocess)
    thinker_result = run_thinker(
        model_path, compiled_path, args.prompt, args.system_prompt,
        num_runs, temp_dir,
    )
    if not thinker_result:
        print("Thinker failed, aborting.")
        return
    print(f"  Text: {thinker_result['gen_text'][:200]}")

    # Phase 2: Hidden states (CPU, run once)
    hf_model, outputs, full_ids, prompt_len = extract_hidden_states(
        model_path, thinker_result
    )

    # Phase 3: Talker input prep (CPU, run once)
    context_len = prepare_talker_input(
        model_path, hf_model, outputs, full_ids, prompt_len,
        args.speaker, temp_dir,
    )

    # Phase 4: Talker (load once, run N times in subprocess)
    talker_result = run_talker(
        model_path, compiled_path, context_len, num_runs, temp_dir,
    )
    if not talker_result or not talker_result["codes"]:
        print("Talker failed or produced no tokens, aborting.")
        return
    print(f"  {len(talker_result['codes'])} codec tokens (avg {talker_result['gen_time']:.3f}s)")

    # Phase 5: Token2Wav (load DiT once, run N times in main process)
    audio_duration, t2w_avg, t2w_times = run_token2wav(
        model_path, compiled_path, talker_result["codes"],
        num_runs, temp_dir, args.output,
    )

    # Summary
    total_time = time.time() - t_total
    thinker_avg = thinker_result["gen_time"]
    talker_avg = talker_result["gen_time"]
    thinker_load = thinker_result.get("load_time", 0)
    talker_load = talker_result.get("load_time", 0)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Text:      {thinker_result['gen_text'][:200]}")
    print(f"\n  Model load time (one-time cost, excluded from pipeline avg):")
    print(f"    Thinker:   {thinker_load:.1f}s")
    print(f"    Talker:    {talker_load:.1f}s")
    print(f"\n  Inference latency (avg of {num_runs} runs, model already loaded):")
    print(f"    Thinker:   {thinker_avg:.3f}s ({thinker_result['n_tokens']} tokens)")
    print(f"    Talker:    {talker_avg:.3f}s ({len(talker_result['codes'])} codec tokens)")
    print(f"    Token2Wav: {t2w_avg:.2f}s")
    pipeline_avg = thinker_avg + talker_avg + t2w_avg
    print(f"    Pipeline:  {pipeline_avg:.2f}s total")
    if audio_duration > 0:
        print(f"\n  Audio:     {audio_duration:.1f}s")
        print(f"  RTF:       {pipeline_avg/audio_duration:.2f}x (pipeline_avg / audio_duration)")
    if num_runs > 1:
        thinker_times = thinker_result.get("all_times", [])
        talker_times = talker_result.get("all_times", [])
        print(f"\n  Per-run breakdown ({num_runs} runs):")
        print(f"    Thinker:   {['%.3f' % t for t in thinker_times]}")
        print(f"    Talker:    {['%.3f' % t for t in talker_times]}")
        print(f"    Token2Wav: {['%.2f' % t for t in t2w_times]}")
    print(f"\n  Wall time:  {total_time:.1f}s (includes model loading + CPU phases)")
    print(f"  Output:     {args.output}")


if __name__ == "__main__":
    main()
