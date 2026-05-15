"""Compile first half and second half CP models."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuronx_distributed_inference.models.config import NeuronConfig
from nxdi_wan.application_cp import NeuronWanCPApplication, NeuronWanCPSecondHalfApplication, WanCPInferenceConfig

MODEL_PATH = os.environ.get("WAN_MODEL_PATH", "/mnt/work/.cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7")
SUBFOLDER = os.environ.get("TRANSFORMER_SUBFOLDER", "transformer")
HALF = os.environ.get("HALF", "first")  # "first" or "second"

if HALF == "first":
    COMPILED_PATH = os.environ.get("COMPILED_PATH", f"/mnt/work/wan2.2-port/compiled_cp_{SUBFOLDER}_first")
    from nxdi_wan.modeling_wan_cp import CPWanFirstHalf as model_cls
else:
    COMPILED_PATH = os.environ.get("COMPILED_PATH", f"/mnt/work/wan2.2-port/compiled_cp_{SUBFOLDER}_second")
    from nxdi_wan.modeling_wan_cp import CPWanSecondHalf as model_cls

os.makedirs(COMPILED_PATH, exist_ok=True)

nc = NeuronConfig(tp_degree=4, world_size=4, torch_dtype=torch.bfloat16, batch_size=1)
config = WanCPInferenceConfig.from_pretrained(
    model_path=MODEL_PATH, neuron_config=nc, num_frames=13, height=480, width=832)

print(f"Compiling {SUBFOLDER} {HALF} half, CP=4...")
sys.stdout.flush()

app_cls = NeuronWanCPSecondHalfApplication if HALF == "second" else NeuronWanCPApplication
app = app_cls(
    model_path=os.path.join(MODEL_PATH, SUBFOLDER),
    config=config,
    model_cls=model_cls,
)
t0 = time.time()
app.compile(COMPILED_PATH)
print(f"Compiled in {time.time()-t0:.0f}s")
