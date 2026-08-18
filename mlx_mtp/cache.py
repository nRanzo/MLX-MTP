"""Cache ownership and snapshots used by speculative decoding."""
from dataclasses import dataclass
from typing import Any

@dataclass
class MTPCache:
    """Separate cache collection: exactly one entry per MTP attention layer."""
    layers: list[Any]

    @classmethod
    def create(cls, factory, num_layers: int) -> "MTPCache":
        return cls([factory() for _ in range(num_layers)])

    def reset(self) -> None:
        self.layers.clear()
