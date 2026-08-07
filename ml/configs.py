from dataclasses import dataclass, field


@dataclass
class baseConfig:
    d_model: int = 384
    heads: int = 8
    layers: int = 10

    context_length: int = 768
    dropout: float = 0.0

    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    hold_steps: int = 20_000

    vocab_size: int = 16384

    max_steps: int = 60_000
    batch_size: int = 32
    datasets: list[tuple[str, str, float]] = field(
        default_factory=lambda: [
            ("HuggingFaceFW/fineweb", "sample-10BT", 0.8),
            ("Salesforce/wikitext", "wikitext-103-raw-v1", 0.2),
        ]
    )