from dataclasses import dataclass


@dataclass
class baseConfig:
    d_model: int = 384
    heads: int = 8
    layers: int = 10

    context_length: int = 512
    dropout: float = 0.0

    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    hold_steps: int = 20_000

    max_steps: int = 10_0000
    batch_size: int = 32