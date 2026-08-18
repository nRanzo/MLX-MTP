"""Exact rejection-sampling speculative decoder.

Cache mutation is transactional: verification runs against a cloned target
cache, and a rejected suffix is replayed from the last committed cache.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .sampling import (
    SamplerConfig,
    acceptance_probability,
    filtered_probabilities,
    residual_distribution,
    sample_from_logits,
)

@dataclass
class GenerationStats:
    proposed: int = 0
    accepted: int = 0
    target_forwards: int = 0
    mtp_forwards: int = 0

class SpeculativeDecoder:
    def __init__(self, adapter, sampler: SamplerConfig = SamplerConfig(), seed: int | None = None):
        self.adapter, self.sampler, self.rng = adapter, sampler, np.random.default_rng(seed)

    @staticmethod
    def _last(x):
        return x[0, -1]

    @staticmethod
    def _numpy(x):
        """MLX BF16 has no direct NumPy buffer representation."""
        import mlx.core as mx
        return np.array(x.astype(mx.float32))

    def generate(self, input_ids, max_tokens: int, depth: int = 1):
        """Generate with a batched `[anchor, MTP proposal]` target verification.

        The anchor is sampled from the target's current distribution and is
        therefore already an exact target token.  Qwythos' one MTP layer then
        proposes its successor.  Only that successor enters the rejection
        test.  A model that exports more physical MTP layers can extend the
        same transaction to more than one proposal; Qwythos exports one.
        """
        if depth < 1 or depth > self.adapter.config.num_layers:
            raise ValueError(f"checkpoint supports depth 1..{self.adapter.config.num_layers}, got {depth}")
        import mlx.core as mx
        cache = self.adapter.make_target_cache(); mtp_cache = self.adapter.make_mtp_cache()
        logits, hidden = self.adapter.forward(input_ids, cache=cache, return_hidden=True)
        stats, output = GenerationStats(target_forwards=1), []
        while len(output) < max_tokens:
            anchor, _ = sample_from_logits(self._numpy(self._last(logits)), self.sampler, self.rng)
            draft_logits, _ = self.adapter.mtp_forward(
                hidden[:, -1:, :], mx.array([[anchor]]), cache=mtp_cache, return_hidden=True)
            proposal, q = sample_from_logits(self._numpy(self._last(draft_logits)), self.sampler, self.rng)
            stats.mtp_forwards += 1; stats.proposed += 1
            # One multi-token call incorporates the anchor and verifies draft.
            trial_cache = self.adapter.clone_cache(cache)
            verified_logits, verified_hidden = self.adapter.forward(
                mx.array([[anchor, proposal]]), cache=trial_cache, return_hidden=True
            )
            stats.target_forwards += 1
            p = filtered_probabilities(self._numpy(verified_logits[0, 0]), self.sampler)
            if self.rng.random() <= acceptance_probability(p[proposal], q[proposal]):
                # Commit candidate cache; the final verification logit supplies
                # the required bonus target token.
                cache, logits, hidden = trial_cache, verified_logits, verified_hidden
                bonus, _ = sample_from_logits(self._numpy(self._last(logits)), self.sampler, self.rng)
                logits, hidden = self.adapter.forward(mx.array([[bonus]]), cache=cache, return_hidden=True)
                stats.target_forwards += 1
                output.extend([anchor, proposal, bonus]); stats.accepted += 1
            else:
                replacement = int(self.rng.choice(len(p), p=residual_distribution(p, q)))
                # The trial cache includes rejected proposal. Rebuild it from
                # the last committed state with anchor and residual only.
                cache = self.adapter.clone_cache(cache)
                logits, hidden = self.adapter.forward(
                    mx.array([[anchor, replacement]]), cache=cache, return_hidden=True
                )
                stats.target_forwards += 1
                output.extend([anchor, replacement])
            mtp_cache = self.adapter.make_mtp_cache()
            if len(output) >= max_tokens:
                output = output[:max_tokens]
        return output, stats
