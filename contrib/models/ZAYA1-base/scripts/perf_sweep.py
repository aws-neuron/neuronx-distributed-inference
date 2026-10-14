#!/usr/bin/env python3
"""
ZAYA Performance Optimization Sweep via vLLM.

Launches vLLM servers with different configs, benchmarks each one,
then produces a comparison summary.

Configs tested:
  A) Baseline:          max_model_len=256, no MLP ISA
  B) Multi-bucket:      max_model_len=1024, no MLP ISA
  C) MLP ISA:           max_model_len=256, MLP ISA kernel
  D) Multi-bucket+ISA:  max_model_len=1024, MLP ISA kernel

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python3 -u ~/zaya/perf_sweep.py 2>&1 | tee ~/zaya/perf_sweep.log
"""

import json
import os
import signal
import subprocess
import sys
import time

import requests

MODEL = os.path.expanduser("~/models/ZAYA1-base")
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
RESULTS_DIR = os.path.expanduser("~/zaya/perf_sweep_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ensure Neuron target is set
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")


CONFIGS = [
    {
        "name": "A_baseline_256",
        "max_model_len": 256,
        "max_num_seqs": 4,
        "env": {},
        "description": "Baseline: max_model_len=256",
    },
    {
        "name": "B_multibucket_1024",
        "max_model_len": 1024,
        "max_num_seqs": 4,
        "env": {},
        "description": "Multi-bucket: max_model_len=1024",
    },
    {
        "name": "C_mlpisa_256",
        "max_model_len": 256,
        "max_num_seqs": 4,
        "env": {"ZAYA_MLP_ISA_KERNEL": "1"},
        "description": "MLP ISA kernel: max_model_len=256",
    },
    {
        "name": "D_multibucket_mlpisa_1024",
        "max_model_len": 1024,
        "max_num_seqs": 4,
        "env": {"ZAYA_MLP_ISA_KERNEL": "1"},
        "description": "Multi-bucket + MLP ISA: max_model_len=1024",
    },
]


def kill_server():
    """Kill any running vLLM server."""
    subprocess.run(["pkill", "-f", f"vllm.*{MODEL}"], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", f"vllm.*{MODEL}"], capture_output=True)
    time.sleep(2)


def start_server(cfg):
    """Start vLLM server with the given config. Returns (process, startup_time)."""
    env = os.environ.copy()
    env.update(cfg["env"])

    # Remove stale env vars from previous configs
    for key in ["ZAYA_MLP_ISA_KERNEL", "ZAYA_DISABLE_NKI"]:
        if key not in cfg["env"]:
            env.pop(key, None)

    override = json.dumps({"override_neuron_config": {"is_continuous_batching": True}})

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        str(cfg["max_model_len"]),
        "--max-num-seqs",
        str(cfg["max_num_seqs"]),
        "--port",
        str(PORT),
        "--host",
        "0.0.0.0",
        "--trust-remote-code",
        "--block-size",
        "128",
        "--no-enable-prefix-caching",
        "--additional-config",
        override,
    ]

    log_path = os.path.join(RESULTS_DIR, f"{cfg['name']}_server.log")
    log_file = open(log_path, "w")

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file, t0


def wait_for_server(timeout=2400):
    """Wait for server to become healthy. Returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(10)
        elapsed = int(time.time() - t0)
        if elapsed % 60 == 0:
            print(f"    Still waiting... ({elapsed}s)")
    raise TimeoutError(f"Server not ready after {timeout}s")


def benchmark_request(prompt, max_tokens, n=1):
    """Send a single completion request and return (completion_tokens, latency_ms, ttft_ms)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "n": n,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/v1/completions", json=payload, timeout=120)
    latency = (time.time() - t0) * 1000
    data = r.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    return completion_tokens, latency, prompt_tokens


def run_benchmark(name, prompt_len, max_tokens, num_requests, warmup=3):
    """Run a benchmark: warmup + measured requests. Returns dict of results."""
    # Build prompt (~prompt_len tokens)
    prompt = "hello " * prompt_len

    # Warmup
    for _ in range(warmup):
        try:
            benchmark_request(prompt, max_tokens)
        except Exception:
            pass

    # Measured requests (sequential for precise latency)
    latencies = []
    total_gen_tokens = 0
    for _ in range(num_requests):
        try:
            gen_tokens, latency_ms, _ = benchmark_request(prompt, max_tokens)
            latencies.append(latency_ms)
            total_gen_tokens += gen_tokens
        except Exception as e:
            print(f"      Request failed: {e}")

    if not latencies:
        return {"error": "All requests failed"}

    import numpy as np

    latencies = np.array(latencies)
    total_time_s = latencies.sum() / 1000

    result = {
        "config": name,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "num_requests": num_requests,
        "success": len(latencies),
        "total_gen_tokens": int(total_gen_tokens),
        "avg_tokens_per_req": float(total_gen_tokens / len(latencies)),
        "total_time_s": float(total_time_s),
        "throughput_tok_s": float(total_gen_tokens / total_time_s)
        if total_time_s > 0
        else 0,
        "avg_latency_ms": float(latencies.mean()),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(latencies.min()),
        "max_latency_ms": float(latencies.max()),
    }
    return result


def run_config(cfg):
    """Run full benchmark suite for a single config."""
    name = cfg["name"]
    print(f"\n{'=' * 70}")
    print(f"Config: {name}")
    print(f"  {cfg['description']}")
    print(f"  max_model_len={cfg['max_model_len']}, env={cfg['env']}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    kill_server()
    # Clear compile cache
    subprocess.run(["rm", "-rf", "/var/tmp/neuron-compile-cache/"], capture_output=True)

    print("  Starting vLLM server (compilation may take 15-30 min)...")
    sys.stdout.flush()
    proc, log_file, start_time = start_server(cfg)

    try:
        startup_s = wait_for_server()
        print(f"  Server ready in {startup_s:.0f}s ({startup_s / 60:.1f} min)")

        results = {
            "config": cfg,
            "startup_time_s": startup_s,
            "benchmarks": {},
        }

        # Benchmark matrix
        tests = [
            # (prompt_len, max_tokens, num_requests, label)
            (4, 30, 10, "short_prompt"),
            (32, 30, 10, "med_prompt_32"),
            (64, 30, 10, "med_prompt_64"),
            (128, 30, 10, "long_prompt_128"),
        ]

        # Add longer prompts if max_model_len supports them
        if cfg["max_model_len"] >= 512:
            tests.append((256, 30, 10, "long_prompt_256"))
        if cfg["max_model_len"] >= 768:
            tests.append((512, 30, 10, "long_prompt_512"))

        for prompt_len, max_tokens, num_req, label in tests:
            print(
                f"\n  Benchmark: {label} (prompt={prompt_len}, gen={max_tokens}, n={num_req})"
            )
            sys.stdout.flush()
            bm = run_benchmark(name, prompt_len, max_tokens, num_req)
            results["benchmarks"][label] = bm
            if "error" not in bm:
                print(
                    f"    {bm['throughput_tok_s']:.1f} tok/s, "
                    f"avg={bm['avg_latency_ms']:.0f}ms, "
                    f"p99={bm['p99_latency_ms']:.0f}ms"
                )
            else:
                print(f"    ERROR: {bm['error']}")
            sys.stdout.flush()

        # Save results
        results_path = os.path.join(RESULTS_DIR, f"{name}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {"config": cfg, "error": str(e)}

    finally:
        # Kill server
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        time.sleep(3)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        log_file.close()


def main():
    print("=" * 70)
    print("ZAYA Performance Optimization Sweep")
    print("=" * 70)
    print(f"Instance: trn2.3xlarge, TP=2, LNC=2")
    print(f"Model: {MODEL}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    all_results = {}
    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i + 1}/{len(CONFIGS)}]", end="")
        result = run_config(cfg)
        all_results[cfg["name"]] = result

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "sweep_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':<30} {'Startup':>8} {'Short4':>8} {'Med64':>8} {'Long128':>8}")
    print(f"{'':30} {'(min)':>8} {'tok/s':>8} {'tok/s':>8} {'tok/s':>8}")
    print("-" * 70)

    for name, r in all_results.items():
        if "error" in r and "benchmarks" not in r:
            print(f"{name:<30} {'ERROR':>8}")
            continue
        startup = r.get("startup_time_s", 0) / 60
        bms = r.get("benchmarks", {})
        short_tps = bms.get("short_prompt", {}).get("throughput_tok_s", 0)
        med_tps = bms.get("med_prompt_64", {}).get("throughput_tok_s", 0)
        long_tps = bms.get("long_prompt_128", {}).get("throughput_tok_s", 0)
        print(
            f"{name:<30} {startup:>7.1f} {short_tps:>7.1f} {med_tps:>7.1f} {long_tps:>7.1f}"
        )

    print(f"\nResults: {combined_path}")
    print(f"Finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")


if __name__ == "__main__":
    main()
