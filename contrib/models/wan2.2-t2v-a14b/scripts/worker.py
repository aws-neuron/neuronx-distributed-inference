"""
Persistent Neuron model worker.

Loads a single NEFF and processes forward-pass requests via filesystem IPC.
Runs as a single Python process — NxDI handles TP internally (no torchrun needed).

Usage:
  WORKER_TYPE=tp4 NEURON_RT_VISIBLE_CORES=0-3 python worker.py
  WORKER_TYPE=cp_first NEURON_RT_VISIBLE_CORES=4-7 python worker.py
  WORKER_TYPE=cp_second NEURON_RT_VISIBLE_CORES=8-11 python worker.py

IPC protocol (all files in WORK_DIR/{worker_type}/):
  _worker_ready     — created after NEFF is loaded
  _request.pt       — request tensor dict, written by orchestrator
  _request_ready    — signal file, created by orchestrator AFTER _request.pt is fully written
  _response.pt      — response tensor dict, written by worker
  _response_ready   — signal file, created by worker AFTER _response.pt is fully written
  _shutdown         — signal file, created by orchestrator to stop the worker
"""
import os, sys, time, torch
sys.path.insert(0, os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-lint-fix"))

WORKER_TYPE = os.environ["WORKER_TYPE"]
WORK_DIR = os.environ.get("WORK_DIR", "/dev/shm/wan_workers")
MODEL_PATH = os.environ.get("WAN_MODEL_PATH", "/mnt/work/.cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7")
SUBFOLDER = os.environ.get("SUBFOLDER", "transformer")
WORKER_NAME = os.environ.get("WORKER_NAME", WORKER_TYPE)

worker_dir = os.path.join(WORK_DIR, WORKER_NAME)
os.makedirs(worker_dir, exist_ok=True)
for f in ["_request_ready", "_response_ready", "_shutdown", "_worker_ready"]:
    p = os.path.join(worker_dir, f)
    if os.path.exists(p):
        os.remove(p)

# ---- Load model (NxDI initializes TP internally, no torchrun needed) ----
from neuronx_distributed_inference.models.config import NeuronConfig

if WORKER_TYPE == "tp4":
    from nxdi_wan.application import NeuronWanBackboneApplication, WanBackboneInferenceConfig
    nc = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16, batch_size=1)
    config = WanBackboneInferenceConfig.from_pretrained(
        model_path=MODEL_PATH, neuron_config=nc, num_frames=13, height=480, width=832)
    compiled_path = os.environ.get("COMPILED_PATH",
        f"{os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-lint-fix")}/compiled_tp4_bf16_{SUBFOLDER}")
    print(f"[{WORKER_TYPE}] Loading {compiled_path}...", flush=True)
    app = NeuronWanBackboneApplication(
        model_path=os.path.join(MODEL_PATH, SUBFOLDER), config=config)
    app.load(compiled_path)

elif WORKER_TYPE == "cp_first":
    from nxdi_wan.application_cp import NeuronWanCPApplication, WanCPInferenceConfig
    from nxdi_wan.modeling_wan_cp import CPWanFirstHalf
    nc = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16, batch_size=1)
    config = WanCPInferenceConfig.from_pretrained(
        model_path=MODEL_PATH, neuron_config=nc, num_frames=13, height=480, width=832,
        subfolder=SUBFOLDER)
    compiled_path = os.environ.get("COMPILED_PATH",
        f"{os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-lint-fix")}/compiled_cp_{SUBFOLDER}_first")
    print(f"[{WORKER_TYPE}] Loading {compiled_path}...", flush=True)
    app = NeuronWanCPApplication(
        model_path=os.path.join(MODEL_PATH, SUBFOLDER),
        config=config, model_cls=CPWanFirstHalf)
    app.load(compiled_path)

elif WORKER_TYPE == "cp_second":
    from nxdi_wan.application_cp import NeuronWanCPSecondHalfApplication, WanCPInferenceConfig
    from nxdi_wan.modeling_wan_cp import CPWanSecondHalf
    nc = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16, batch_size=1)
    config = WanCPInferenceConfig.from_pretrained(
        model_path=MODEL_PATH, neuron_config=nc, num_frames=13, height=480, width=832,
        subfolder=SUBFOLDER)
    compiled_path = os.environ.get("COMPILED_PATH",
        f"{os.environ.get("WAN_PORT_DIR", "/mnt/work/wan2.2-lint-fix")}/compiled_cp_{SUBFOLDER}_second")
    print(f"[{WORKER_TYPE}] Loading {compiled_path}...", flush=True)
    app = NeuronWanCPSecondHalfApplication(
        model_path=os.path.join(MODEL_PATH, SUBFOLDER),
        config=config, model_cls=CPWanSecondHalf)
    app.load(compiled_path)

else:
    raise ValueError(f"Unknown WORKER_TYPE: {WORKER_TYPE}")

print(f"[{WORKER_TYPE}] Ready.", flush=True)
open(os.path.join(worker_dir, "_worker_ready"), "w").close()

# ---- Request loop ----
request_count = 0

while True:
    # Poll for request or shutdown
    while True:
        if os.path.exists(os.path.join(worker_dir, "_shutdown")):
            print(f"[{WORKER_TYPE}] Shutting down after {request_count} requests.", flush=True)
            sys.exit(0)
        if os.path.exists(os.path.join(worker_dir, "_request_ready")):
            break
        time.sleep(0.002)

    data = torch.load(os.path.join(worker_dir, "_request.pt"), weights_only=False)

    with torch.no_grad():
        if WORKER_TYPE == "tp4":
            output = app(data["hs"], data["ts"], data["enc"], data["rc"], data["rs"])
        elif WORKER_TYPE == "cp_first":
            output = app(data["hs"], data["ts"], data["enc"], data["rc"], data["rs"])
        elif WORKER_TYPE == "cp_second":
            output = app(data["hs"], data["temb"], data["ts_proj"],
                         data["enc_proj"], data["rc"], data["rs"])

    if isinstance(output, (tuple, list)):
        save_data = {f"out_{i}": t.cpu() for i, t in enumerate(output)}
    else:
        save_data = {"output": output.cpu()}
    torch.save(save_data, os.path.join(worker_dir, "_response.pt"))
    # Signal AFTER file is fully written
    os.remove(os.path.join(worker_dir, "_request_ready"))
    open(os.path.join(worker_dir, "_response_ready"), "w").close()

    request_count += 1

if __name__ == "__main__":
    pass  # Worker is launched as a subprocess; main logic runs at module level by design

