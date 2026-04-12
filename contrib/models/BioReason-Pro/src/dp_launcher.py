#!/usr/bin/env python3
"""
Data-parallel launcher for BioReason-Pro on multi-core Neuron instances.

Runs N independent BioReasonPipeline workers, each pinned to a separate
NeuronCore via NEURON_RT_VISIBLE_CORES. Proteins are distributed round-robin
across workers and processed in parallel.

Usage:
    from src.dp_launcher import DataParallelRunner

    runner = DataParallelRunner(
        model_path="/mnt/models/bioreason-pro-rl",
        num_workers=4,          # Number of NeuronCores to use
        batch_size=1,           # Per-worker batch size
        compiled_model_path="/mnt/compiled/bs1",
    )

    proteins = [
        {"sequence": "MSSQQYQ...", "organism": "Mouse", "interpro": "...", "gogpt": "..."},
        {"sequence": "MKFLILF...", "organism": "Human", "interpro": "...", "gogpt": "..."},
        ...
    ]
    results = runner.run(proteins)  # Returns list of result dicts

On trn2.3xlarge:
    - LNC=2 (default): 4 cores -> num_workers=4
    - LNC=1: 8 cores -> num_workers=8

Requires:
    - NEURON_RT_VISIBLE_CORES support (Neuron SDK 2.28+)
    - Pre-compiled model at compiled_model_path (compile once, share across workers)
    - Sufficient CPU memory for N copies of ESM3 (~1.4B each, ~3GB FP32)
"""

import logging
import os
import time
from multiprocessing import Process, Queue
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _worker_fn(
    worker_id: int,
    core_id: int,
    model_path: str,
    esm3_model: str,
    max_context_length: int,
    max_new_tokens: int,
    batch_size: int,
    tp_degree: int,
    compiled_model_path: Optional[str],
    task_queue: "Queue",
    result_queue: "Queue",
):
    """Worker process: loads pipeline on assigned core and processes proteins.

    Each worker:
    1. Sets NEURON_RT_VISIBLE_CORES to pin to a single NeuronCore
    2. Loads a BioReasonPipeline (ESM3 on CPU, Qwen3-4B on Neuron)
    3. Pulls protein tasks from the shared queue
    4. Pushes results back to the result queue
    """
    # Pin to specific NeuronCore BEFORE importing anything Neuron-related
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_RT_LOG_LEVEL"] = os.environ.get("NEURON_RT_LOG_LEVEL", "WARNING")

    import torch  # noqa: delayed import after env setup

    # Import pipeline (triggers NxDI import and patch application)
    try:
        from modeling_bioreason import BioReasonPipeline
    except ImportError:
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from modeling_bioreason import BioReasonPipeline

    log.info(f"Worker {worker_id}: loading pipeline on core {core_id}...")
    t0 = time.time()
    pipeline = BioReasonPipeline(
        model_path=model_path,
        esm3_model=esm3_model,
        max_context_length=max_context_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        tp_degree=tp_degree,
        compiled_model_path=compiled_model_path,
    )
    load_time = time.time() - t0
    log.info(f"Worker {worker_id}: pipeline ready in {load_time:.1f}s")

    # Process proteins from the task queue until sentinel (None)
    while True:
        item = task_queue.get()
        if item is None:
            break

        idx, protein = item
        try:
            result = pipeline.predict(
                sequence=protein["sequence"],
                organism=protein["organism"],
                interpro=protein.get("interpro", ""),
                gogpt=protein.get("gogpt", ""),
                max_new_tokens=protein.get("max_new_tokens"),
            )
            result["worker_id"] = worker_id
            result["core_id"] = core_id
            result["protein_idx"] = idx
            result_queue.put((idx, result))
        except Exception as e:
            result_queue.put(
                (
                    idx,
                    {
                        "error": str(e),
                        "worker_id": worker_id,
                        "core_id": core_id,
                        "protein_idx": idx,
                    },
                )
            )

    log.info(f"Worker {worker_id}: shutting down")


class DataParallelRunner:
    """Distributes protein inference across multiple NeuronCores.

    Each core runs an independent BioReasonPipeline in a separate process.
    Proteins are distributed round-robin across workers.
    """

    def __init__(
        self,
        model_path: str,
        num_workers: int = 4,
        esm3_model: str = "esm3_sm_open_v1",
        max_context_length: int = 1024,
        max_new_tokens: int = 2048,
        batch_size: int = 1,
        tp_degree: int = 1,
        compiled_model_path: Optional[str] = None,
        start_core: int = 0,
    ):
        """Initialize the data-parallel runner.

        Args:
            model_path: Path to wanglab/bioreason-pro-rl checkpoint
            num_workers: Number of NeuronCores to use (default: 4 for LNC=2)
            esm3_model: ESM3 model name (default: esm3_sm_open_v1)
            max_context_length: Max input context per worker (default: 1024)
            max_new_tokens: Max generation tokens per worker (default: 2048)
            batch_size: Per-worker batch size (default: 1)
            tp_degree: TP degree per worker (default: 1)
            compiled_model_path: Path for compiled model artifacts
                (shared across all workers)
            start_core: First NeuronCore ID to use (default: 0)
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.esm3_model = esm3_model
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.compiled_model_path = compiled_model_path
        self.start_core = start_core

    def run(self, proteins: List[Dict]) -> List[Dict]:
        """Run inference on a list of proteins using all workers.

        Args:
            proteins: List of dicts with keys: sequence, organism,
                interpro (optional), gogpt (optional)

        Returns:
            List of result dicts, ordered by input index.
            Each result has: text, num_tokens, gen_time_s, total_time_s,
            tok_per_s, worker_id, core_id, protein_idx
        """
        if not proteins:
            return []

        task_queue = Queue()
        result_queue = Queue()

        # Enqueue all proteins
        for idx, protein in enumerate(proteins):
            task_queue.put((idx, protein))

        # Add sentinel values (one per worker) to signal shutdown
        for _ in range(self.num_workers):
            task_queue.put(None)

        # Start worker processes
        workers = []
        for i in range(self.num_workers):
            core_id = self.start_core + i
            p = Process(
                target=_worker_fn,
                args=(
                    i,
                    core_id,
                    self.model_path,
                    self.esm3_model,
                    self.max_context_length,
                    self.max_new_tokens,
                    self.batch_size,
                    self.tp_degree,
                    self.compiled_model_path,
                    task_queue,
                    result_queue,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        # Collect results
        results = [None] * len(proteins)
        for _ in range(len(proteins)):
            idx, result = result_queue.get()
            results[idx] = result

        # Wait for workers to finish
        for p in workers:
            p.join(timeout=30)

        return results

    def benchmark(
        self,
        proteins: List[Dict],
        warmup_count: int = 0,
    ) -> Dict:
        """Run inference with timing and aggregate statistics.

        Args:
            proteins: List of protein dicts
            warmup_count: Number of initial proteins to discard from timing
                (useful if first compilation happens during the run)

        Returns:
            Dict with aggregate benchmark statistics
        """
        t_start = time.time()
        results = self.run(proteins)
        wall_time = time.time() - t_start

        # Separate warmup from measured results
        measured = results[warmup_count:]
        errors = [r for r in measured if "error" in r]
        successes = [r for r in measured if "error" not in r]

        total_tokens = sum(r["num_tokens"] for r in successes)
        total_gen_time = sum(r["gen_time_s"] for r in successes)

        # Per-worker stats
        worker_tokens = {}
        worker_gen_time = {}
        for r in successes:
            wid = r["worker_id"]
            worker_tokens[wid] = worker_tokens.get(wid, 0) + r["num_tokens"]
            worker_gen_time[wid] = worker_gen_time.get(wid, 0) + r["gen_time_s"]

        per_worker_tok_s = {
            wid: worker_tokens[wid] / worker_gen_time[wid]
            for wid in worker_tokens
            if worker_gen_time[wid] > 0
        }

        return {
            "num_workers": self.num_workers,
            "num_proteins": len(proteins),
            "num_measured": len(measured),
            "num_errors": len(errors),
            "wall_time_s": wall_time,
            "total_tokens": total_tokens,
            "aggregate_tok_s": total_tokens / wall_time if wall_time > 0 else 0,
            "per_worker_tok_s": per_worker_tok_s,
            "mean_worker_tok_s": (
                sum(per_worker_tok_s.values()) / len(per_worker_tok_s)
                if per_worker_tok_s
                else 0
            ),
            "results": results,
        }
