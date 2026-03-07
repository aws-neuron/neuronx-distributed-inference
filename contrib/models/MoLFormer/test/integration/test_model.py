"""
Integration tests for MoLFormer on Neuron (inf2).

Tests accuracy (cosine similarity), compilation, inference, DataParallel,
and performance for ibm/MoLFormer-XL-both-10pct compiled with torch_neuronx.trace().

Run:
    python -m pytest contrib/models/MoLFormer/test/integration/test_model.py -v
    python contrib/models/MoLFormer/test/integration/test_model.py  # standalone
"""

import json
import os
import shutil
import subprocess
import time

import numpy as np
import pytest
import torch
import torch_neuronx
from transformers import AutoModel, AutoTokenizer

# --- Constants ---

MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
MAX_LENGTH = 202

# Resolve saved_models: use contrib root if in proper directory structure,
# otherwise fall back to ./saved_models in the current working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTRIB_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
if os.path.isdir(os.path.join(_CONTRIB_ROOT, "test")):
    SAVED_DIR = os.path.join(_CONTRIB_ROOT, "saved_models")
else:
    SAVED_DIR = os.path.join(os.getcwd(), "saved_models")
COMPILER_ARGS = ["--auto-cast", "matmult", "--auto-cast-type", "bf16"]

# Accuracy thresholds
COSINE_SIM_THRESHOLD = 0.9999
MAX_DIFF_THRESHOLD = 0.02

# Performance thresholds (conservative, for inf2.xlarge)
MIN_THROUGHPUT_BS1 = 400  # inf/s, single core, matmult
MIN_THROUGHPUT_DP = 800  # inf/s, DataParallel 2 cores
MAX_P50_LATENCY_MS = 5.0  # ms, single core BS=1

# Test SMILES molecules
SMILES_EXAMPLES = [
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
    "C1CCCCC1",  # cyclohexane
    "c1ccccc1",  # benzene
]


# --- Helpers ---


def encode_smiles(tokenizer, smiles_list, max_length=MAX_LENGTH, batch_size=1):
    """Encode SMILES strings for MoLFormer input."""
    tokens = tokenizer(
        smiles_list,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return (
        torch.repeat_interleave(tokens["input_ids"], batch_size, 0),
        torch.repeat_interleave(tokens["attention_mask"], batch_size, 0),
    )


def get_pooler_output(outputs):
    """Extract pooler_output from Neuron model outputs.

    Neuron traced models return a dict (default) or tuple (torchscript=True).
    """
    if isinstance(outputs, dict):
        return outputs["pooler_output"]
    elif isinstance(outputs, tuple):
        return outputs[1]
    else:
        return outputs.pooler_output


def get_last_hidden_state(outputs):
    """Extract last_hidden_state from Neuron model outputs."""
    if isinstance(outputs, dict):
        return outputs["last_hidden_state"]
    elif isinstance(outputs, tuple):
        return outputs[0]
    else:
        return outputs.last_hidden_state


def get_neuron_core_count():
    """Detect number of NeuronCores available."""
    try:
        result = subprocess.run(
            ["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10
        )
        info = json.loads(result.stdout)
        return sum(d["nc_count"] for d in info)
    except Exception:
        return 0


def load_cpu_model():
    """Load MoLFormer CPU reference model."""
    model = AutoModel.from_pretrained(
        MODEL_ID, deterministic_eval=True, trust_remote_code=True
    )
    model.eval()
    return model


def compile_neuron_model(cpu_model, batch_size=1):
    """Compile MoLFormer for Neuron, with caching."""
    os.makedirs(SAVED_DIR, exist_ok=True)
    model_file = os.path.join(SAVED_DIR, f"molformer_bs{batch_size}.pt")

    if os.path.isfile(model_file):
        return torch.jit.load(model_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    trace_inputs = encode_smiles(
        tokenizer, [SMILES_EXAMPLES[0]], max_length=MAX_LENGTH, batch_size=batch_size
    )

    workdir = os.path.join(SAVED_DIR, f"workdir_bs{batch_size}")
    shutil.rmtree(workdir, ignore_errors=True)

    neuron_model = torch_neuronx.trace(
        cpu_model,
        trace_inputs,
        compiler_args=COMPILER_ARGS,
        compiler_workdir=workdir,
    )
    neuron_model.save(model_file)
    return torch.jit.load(model_file)


# --- Fixtures ---


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def cpu_model():
    return load_cpu_model()


@pytest.fixture(scope="module")
def neuron_model(cpu_model):
    return compile_neuron_model(cpu_model, batch_size=1)


@pytest.fixture(scope="module")
def n_cores():
    return get_neuron_core_count()


# --- Test Classes ---


class TestModelLoads:
    """Smoke tests: model loads and produces correct output shape."""

    def test_neuron_model_loads(self, neuron_model, tokenizer):
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        outputs = neuron_model(*inputs)
        assert isinstance(outputs, (tuple, dict))

    def test_output_shape(self, neuron_model, tokenizer):
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        outputs = neuron_model(*inputs)
        # last_hidden_state: [batch, seq_len, hidden_dim]
        assert get_last_hidden_state(outputs).shape == (1, MAX_LENGTH, 768)
        # pooler_output: [batch, hidden_dim]
        assert get_pooler_output(outputs).shape == (1, 768)


class TestAccuracy:
    """Accuracy tests: Neuron vs CPU cosine similarity for various SMILES."""

    @pytest.mark.parametrize("smiles", SMILES_EXAMPLES)
    def test_cosine_similarity(self, neuron_model, cpu_model, tokenizer, smiles):
        inputs = encode_smiles(tokenizer, [smiles])

        with torch.no_grad():
            cpu_out = cpu_model(*inputs)
            neuron_out = neuron_model(*inputs)

        cpu_pooler = cpu_out.pooler_output
        neuron_pooler = get_pooler_output(neuron_out)

        cosine_sim = torch.nn.functional.cosine_similarity(
            cpu_pooler.flatten().unsqueeze(0),
            neuron_pooler.flatten().unsqueeze(0),
        ).item()

        assert cosine_sim >= COSINE_SIM_THRESHOLD, (
            f"Cosine similarity {cosine_sim:.6f} < {COSINE_SIM_THRESHOLD} for {smiles}"
        )

    @pytest.mark.parametrize("smiles", SMILES_EXAMPLES)
    def test_max_diff(self, neuron_model, cpu_model, tokenizer, smiles):
        inputs = encode_smiles(tokenizer, [smiles])

        with torch.no_grad():
            cpu_out = cpu_model(*inputs)
            neuron_out = neuron_model(*inputs)

        max_diff = torch.max(
            torch.abs(cpu_out.pooler_output - get_pooler_output(neuron_out))
        ).item()

        assert max_diff <= MAX_DIFF_THRESHOLD, (
            f"Max diff {max_diff:.6f} > {MAX_DIFF_THRESHOLD} for {smiles}"
        )


class TestDataParallel:
    """DataParallel tests: verify multi-core scaling."""

    def test_dp_runs(self, cpu_model, tokenizer, n_cores):
        if n_cores < 2:
            pytest.skip("Need >= 2 NeuronCores for DataParallel test")

        model_file = os.path.join(SAVED_DIR, "molformer_bs1.pt")
        models = [torch.jit.load(model_file) for _ in range(2)]
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])

        # Run both models
        for m in models:
            out = m(*inputs)
            assert get_pooler_output(out).shape == (1, 768)

    def test_dp_speedup(self, cpu_model, tokenizer, n_cores):
        if n_cores < 2:
            pytest.skip("Need >= 2 NeuronCores for DataParallel test")

        model_file = os.path.join(SAVED_DIR, "molformer_bs1.pt")
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        n_iters = 50

        # Single model throughput
        model_single = torch.jit.load(model_file)
        for _ in range(10):
            model_single(*inputs)
        start = time.time()
        for _ in range(n_iters):
            model_single(*inputs)
        single_duration = time.time() - start
        single_throughput = n_iters / single_duration

        # DP=2 throughput
        import concurrent.futures

        models = [torch.jit.load(model_file) for _ in range(2)]
        for m in models:
            for _ in range(10):
                m(*inputs)

        latencies = []

        def worker(m):
            for _ in range(n_iters):
                s = time.time()
                m(*inputs)
                latencies.append(time.time() - s)

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, models[i]) for i in range(2)]
            for f in futures:
                f.result()
        dp_duration = time.time() - start
        dp_throughput = len(latencies) / dp_duration

        speedup = dp_throughput / single_throughput
        assert speedup > 1.3, f"DP speedup {speedup:.2f}x < 1.3x"


class TestPerformance:
    """Performance tests: throughput and latency thresholds."""

    def test_single_core_throughput(self, neuron_model, tokenizer):
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        n_iters = 100

        # Warmup
        for _ in range(10):
            neuron_model(*inputs)

        start = time.time()
        for _ in range(n_iters):
            neuron_model(*inputs)
        duration = time.time() - start
        throughput = n_iters / duration

        assert throughput >= MIN_THROUGHPUT_BS1, (
            f"Throughput {throughput:.0f} inf/s < {MIN_THROUGHPUT_BS1}"
        )

    def test_single_core_latency(self, neuron_model, tokenizer):
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        n_iters = 100

        # Warmup
        for _ in range(10):
            neuron_model(*inputs)

        latencies = []
        for _ in range(n_iters):
            start = time.time()
            neuron_model(*inputs)
            latencies.append((time.time() - start) * 1000.0)

        p50 = np.percentile(latencies, 50)
        assert p50 <= MAX_P50_LATENCY_MS, (
            f"P50 latency {p50:.2f}ms > {MAX_P50_LATENCY_MS}ms"
        )

    def test_dp_throughput(self, cpu_model, tokenizer, n_cores):
        if n_cores < 2:
            pytest.skip("Need >= 2 NeuronCores for DP throughput test")

        import concurrent.futures

        model_file = os.path.join(SAVED_DIR, "molformer_bs1.pt")
        models = [torch.jit.load(model_file) for _ in range(2)]
        inputs = encode_smiles(tokenizer, [SMILES_EXAMPLES[0]])
        n_iters = 50

        for m in models:
            for _ in range(10):
                m(*inputs)

        latencies = []

        def worker(m):
            for _ in range(n_iters):
                s = time.time()
                m(*inputs)
                latencies.append(time.time() - s)

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, models[i]) for i in range(2)]
            for f in futures:
                f.result()
        duration = time.time() - start
        throughput = len(latencies) / duration

        assert throughput >= MIN_THROUGHPUT_DP, (
            f"DP throughput {throughput:.0f} inf/s < {MIN_THROUGHPUT_DP}"
        )


# --- Standalone Runner ---

if __name__ == "__main__":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    print("=" * 60)
    print("MoLFormer Neuron Integration Tests (standalone)")
    print("=" * 60)

    n_cores = get_neuron_core_count()
    print(f"\nNeuronCores detected: {n_cores}")

    print("\n--- Step 1: Load CPU model ---")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cpu_mod = load_cpu_model()
    print(f"CPU model loaded: {sum(p.numel() for p in cpu_mod.parameters()):,} params")

    print("\n--- Step 2: Compile for Neuron ---")
    neuron_mod = compile_neuron_model(cpu_mod, batch_size=1)
    print("Compilation complete")

    print("\n--- Step 3: Smoke test ---")
    inputs = encode_smiles(tok, [SMILES_EXAMPLES[0]])
    out = neuron_mod(*inputs)
    print(
        f"Output shapes: last_hidden={get_last_hidden_state(out).shape}, "
        f"pooler={get_pooler_output(out).shape}"
    )

    print("\n--- Step 4: Accuracy (5 SMILES molecules) ---")
    all_pass = True
    for smiles in SMILES_EXAMPLES:
        inp = encode_smiles(tok, [smiles])
        with torch.no_grad():
            cpu_out = cpu_mod(*inp)
            neuron_out = neuron_mod(*inp)

        cosine_sim = torch.nn.functional.cosine_similarity(
            cpu_out.pooler_output.flatten().unsqueeze(0),
            get_pooler_output(neuron_out).flatten().unsqueeze(0),
        ).item()
        max_diff = torch.max(
            torch.abs(cpu_out.pooler_output - get_pooler_output(neuron_out))
        ).item()

        status = "PASS" if cosine_sim >= COSINE_SIM_THRESHOLD else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  [{status}] {smiles}: cosine={cosine_sim:.6f}, max_diff={max_diff:.6f}"
        )
    print(f"Accuracy: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    print("\n--- Step 5: Performance (BS=1, single core) ---")
    for _ in range(10):
        neuron_mod(*inputs)
    latencies = []
    for _ in range(100):
        s = time.time()
        neuron_mod(*inputs)
        latencies.append((time.time() - s) * 1000.0)
    throughput = 100 / (sum(latencies) / 1000.0)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    print(f"  Throughput: {throughput:.0f} inf/s")
    print(f"  P50: {p50:.2f}ms, P99: {p99:.2f}ms")
    print(
        f"  [{'PASS' if throughput >= MIN_THROUGHPUT_BS1 else 'FAIL'}] >= {MIN_THROUGHPUT_BS1} inf/s"
    )

    if n_cores >= 2:
        print("\n--- Step 6: DataParallel (DP=2) ---")
        import concurrent.futures

        # Release single-core model to free NeuronCore
        del neuron_mod

        model_file = os.path.join(SAVED_DIR, "molformer_bs1.pt")
        models = [torch.jit.load(model_file) for _ in range(2)]
        for m in models:
            for _ in range(10):
                m(*inputs)

        dp_lats = []

        def dp_worker(m):
            for _ in range(50):
                s = time.time()
                m(*inputs)
                dp_lats.append(time.time() - s)

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(dp_worker, models[i]) for i in range(2)]
            for f in futs:
                f.result()
        dp_duration = time.time() - start
        dp_throughput = len(dp_lats) / dp_duration
        print(f"  DP throughput: {dp_throughput:.0f} inf/s")
        print(
            f"  [{'PASS' if dp_throughput >= MIN_THROUGHPUT_DP else 'FAIL'}] >= {MIN_THROUGHPUT_DP} inf/s"
        )
    else:
        print("\n--- Step 6: DataParallel SKIPPED (need >= 2 cores) ---")

    print("\n--- Step 7: Summary ---")
    print(f"NeuronCores: {n_cores}")
    print(f"Single-core: {throughput:.0f} inf/s, P50={p50:.2f}ms")
    if n_cores >= 2:
        print(f"DP=2: {dp_throughput:.0f} inf/s")
    print("ALL TESTS COMPLETE")
