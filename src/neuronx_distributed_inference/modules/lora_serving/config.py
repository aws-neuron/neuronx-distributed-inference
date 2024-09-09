from typing import List


class LoraServingConfig:
    def __init__(
        self,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        max_loras_on_cpu: int = 2,
        target_modules: List[str] = None,
        lora_bias: str = "none",
        lora_ckpt_paths: List[str] = None,
    ):
        # The maximum number of concurrent LoRA adapters in device memory
        self.max_loras = max_loras
        # The highest LoRA rank that needs to be supported
        self.max_lora_rank = max_lora_rank
        # The maximum number of LoRA adapters stored in CPU memory
        self.max_loras_on_cpu = max_loras_on_cpu
        # List of module names or regex expression of the module names to replace with LoRA.
        self.target_modules = target_modules
        # Bias type for LoRA. Can be 'none', 'all'
        self.lora_bias = lora_bias
        # List of checkpoint paths for LoRA adapters
        self.lora_ckpt_paths = lora_ckpt_paths
