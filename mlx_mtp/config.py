"""Explicit, model-independent MTP contracts."""
from dataclasses import dataclass
from typing import Literal

HiddenVariant = Literal["fc", "pre_norm", "post_norm", "embedding", "prev"]
ConcatOrder = Literal["embedding_hidden", "hidden_embedding"]

@dataclass(frozen=True)
class MTPConfig:
    hidden_size: int
    vocab_size: int
    num_layers: int
    hidden_variant: HiddenVariant = "pre_norm"
    concat_order: ConcatOrder = "embedding_hidden"
    rms_norm_eps: float = 1e-6
