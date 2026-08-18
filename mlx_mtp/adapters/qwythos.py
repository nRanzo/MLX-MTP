"""Qwythos 9B OptiQ adapter.

The MTP overlay is one Qwen3.5 full-attention decoder layer and is loaded
separately from ``optiq/mtp.safetensors``.  The target contract is the
pre-final-norm residual stream and [embedding, hidden] concatenation.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any

from ..config import MTPConfig
from ..model_adapter import MTPModelAdapter

def inspect_qwythos_config(model_path: str | Path) -> MTPConfig:
    path = Path(model_path)
    with (path / "config.json").open() as f:
        raw = json.load(f)
    text = raw["text_config"]
    if raw.get("model_type") != "qwen3_5" or not (path / raw.get("mtp_file", "optiq/mtp.safetensors")).is_file():
        raise ValueError("not a Qwythos Qwen3.5 checkpoint with an MTP sidecar")
    return MTPConfig(hidden_size=text["hidden_size"], vocab_size=text["vocab_size"],
                     num_layers=text["mtp_num_hidden_layers"], hidden_variant="pre_norm",
                     concat_order="embedding_hidden", rms_norm_eps=text["rms_norm_eps"])

class QwythosAdapter(MTPModelAdapter):
    def __init__(self, model: Any, config: MTPConfig):
        self.model, self.config = model, config
        self.text = getattr(model, "language_model", model)
        if not hasattr(self.text, "mtp"):
            raise ValueError("MTP overlay is not attached; use QwythosAdapter.load")

    @classmethod
    def load(cls, model_path: str | Path):
        """Load the quantized trunk, then attach only the separate MTP overlay."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.cache import KVCache
        from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
        path = Path(model_path); config = inspect_qwythos_config(path)
        model, _tokenizer = load(str(path))
        text = getattr(model, "language_model", model)
        raw = json.loads((path / "config.json").read_text()); args = TextModelArgs.from_dict(raw["text_config"])
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
                self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
                self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
                self.layers = [DecoderLayer(args=args, layer_idx=args.full_attention_interval - 1)]
                self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        head = Head()
        q = raw["mtplx_mtp_quantization"]
        # The sidecar's FC is BF16; only attention/MLP linears have the
        # packed weight + scales + biases representation.
        nn.quantize(
            head, group_size=q["group_size"], bits=q["bits"], mode=q["mode"],
            class_predicate=lambda name, module: name.startswith("layers.") and isinstance(module, nn.Linear),
        )
        # The sidecar is namespaced as ``mtp.*``; the overlay itself is the
        # MTP module, so strict MLX loading needs local (prefix-free) names.
        weights = mx.load(str(path / raw["mtp_file"]))
        local = {name.removeprefix("mtp."): value for name, value in weights.items()}
        # Some Qwythos sidecars store all RMSNorm scales as raw deltas. The
        # trunk loader restores this Qwen convention, but a separate MTP file
        # bypasses that sanitize step. Detect the characteristic low q/k and
        # residual norm means before restoring every one-dimensional scale.
        qk = [v for k, v in local.items() if k.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight"))]
        low = [v for k, v in local.items() if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight"))]
        if qk and low and max(float(v.mean().item()) for v in qk) < 1.25 and min(float(v.mean().item()) for v in low) < .5:
            local = {k: (v + 1.0 if v.ndim == 1 else v) for k, v in local.items()}
        head.load_weights(list(local.items()), strict=True)
        mx.eval(head.parameters())
        text.mtp = head
        return cls(model, config)

    def _lm_logits(self, hidden):
        return self.text.lm_head(hidden) if getattr(self.text, "lm_head", None) is not None else self.text.model.embed_tokens.as_linear(hidden)

    def forward(self, input_ids, cache=None, return_hidden=False):
        if not return_hidden:
            return self.text(input_ids, cache=cache)
        # Qwen3.5 normally returns post-norm. Swap only for this call to expose
        # the residual stream required by this checkpoint's MTP overlay.
        inner = self.text.model; norm = inner.norm
        try:
            inner.norm = lambda x: x
            pre_norm = inner(input_ids, cache=cache)
        finally:
            inner.norm = norm
        return self._lm_logits(norm(pre_norm)), pre_norm

    def mtp_forward(self, hidden_states, next_token_ids, cache=None, position_offset=None, return_hidden=False):
        from mlx_lm.models.base import create_attention_mask
        import mlx.core as mx
        layer_cache = cache.layers[0] if hasattr(cache, "layers") else (cache[0] if isinstance(cache, list) else cache)
        e = self.text.mtp.pre_fc_norm_embedding(self.text.model.embed_tokens(next_token_ids))
        h = self.text.mtp.pre_fc_norm_hidden(hidden_states)
        x = self.text.mtp.fc(mx.concatenate([e, h], axis=-1))
        hidden = self.text.mtp.layers[0](x, mask=create_attention_mask(x, layer_cache), cache=layer_cache)
        logits = self._lm_logits(self.text.mtp.norm(hidden))
        return (logits, hidden) if return_hidden else logits

    def make_target_cache(self):
        return self.text.make_cache()
    def make_mtp_cache(self):
        from mlx_lm.models.cache import KVCache
        from ..cache import MTPCache
        return MTPCache.create(KVCache, self.config.num_layers)
    def clone_cache(self, cache):
        return copy.deepcopy(cache)
