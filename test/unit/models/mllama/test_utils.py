import logging
import os
import time

import torch
import torch_xla
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

CHECKPOINT_DIR = os.path.join("/tmp", "llama3_mm_unit_test")
if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")


logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)


class TestModelBuilderInstance(BaseModelInstance):
    def __init__(self, module_cls, input_output_aliases, **module_init_kwargs):
        self.module_init_kwargs = module_init_kwargs
        super().__init__(module_cls, input_output_aliases)

    def load_module(self):
        self.module = self.module_cls(**self.module_init_kwargs)
        self.module.eval()


def setup_debug_env():
    os.environ["XLA_FALLBACK_CPU"] = "0"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    torch_xla._XLAC._set_ir_debug(True)
    torch.manual_seed(0)


def trace_nxd_model(model_class, example_inputs, tp_degree=1, **model_init_kwargs):
    model_builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=load_checkpoint,
        compiler_workdir=f"/tmp/compiler_workdir_tp{tp_degree}/",
    )
    logger.info("Initialized model builder")

    model_builder.add(
        key="test_nxd_model",
        model_instance=TestModelBuilderInstance(model_class, {}, **model_init_kwargs),
        example_inputs=[example_inputs],
        compiler_args="--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type=transformer -O1 --internal-hlo2tensorizer-options='--verify-hlo=true'",
    )
    logger.info("Added models. Starting trace.")

    start_time = time.time()
    traced_model = model_builder.trace()
    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor)
    elapsed_time = time.time() - start_time
    logger.info(f"Done with trace in {elapsed_time}s!")
    return traced_model


def load_checkpoint():
    return torch.load(CHECKPOINT_PATH, map_location="cpu")


def save_checkpoint(model):
    torch.save(model.state_dict(), CHECKPOINT_PATH)
