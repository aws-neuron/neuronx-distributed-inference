from dataclasses import dataclass


@dataclass
class BuildConfig:
    batch_size: int
    sequence_length: int
    world_size: int
