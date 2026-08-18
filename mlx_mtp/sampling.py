"""Sampling and exact speculative-rejection primitives."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

def filtered_probabilities(logits: np.ndarray, config: SamplerConfig) -> np.ndarray:
    """Return the distribution after the same truncation used by draft and target."""
    x = np.asarray(logits, dtype=np.float64)
    if config.temperature == 0:
        out = np.zeros_like(x)
        out[np.argmax(x)] = 1.0
        return out
    if config.temperature < 0 or not 0 < config.top_p <= 1 or config.top_k < 0:
        raise ValueError("invalid sampler configuration")
    x = x / config.temperature
    if config.top_k:
        keep = np.argpartition(x, -min(config.top_k, x.size))[-min(config.top_k, x.size):]
        mask = np.full(x.size, False); mask[keep] = True
        x = np.where(mask, x, -np.inf)
    x = x - np.max(x)
    p = np.exp(x); p /= p.sum()
    if config.top_p < 1:
        order = np.argsort(-p); cumulative = np.cumsum(p[order])
        mask = np.zeros(x.size, dtype=bool); mask[order[cumulative <= config.top_p]] = True
        mask[order[0]] = True
        p = np.where(mask, p, 0); p /= p.sum()
    return p

def sample_from_logits(logits, config: SamplerConfig, rng: np.random.Generator):
    p = filtered_probabilities(logits, config)
    return int(rng.choice(len(p), p=p)), p

def acceptance_probability(target_p: float, draft_q: float) -> float:
    return min(1.0, float(np.float32(target_p)) / max(float(np.float32(draft_q)), np.finfo(np.float32).tiny))

def residual_distribution(target_p: np.ndarray, draft_q: np.ndarray) -> np.ndarray:
    residual = np.maximum(np.asarray(target_p) - np.asarray(draft_q), 0.0)
    total = residual.sum()
    return residual / total if total > 0 else np.asarray(target_p) / np.asarray(target_p).sum()
