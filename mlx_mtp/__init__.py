"""MLX-native multi-token prediction and speculative decoding."""

from .config import MTPConfig
from .sampling import SamplerConfig, sample_from_logits
from .speculative import SpeculativeDecoder
from .loader import load_adapter

__all__ = ["MTPConfig", "SamplerConfig", "SpeculativeDecoder", "load_adapter", "sample_from_logits"]
