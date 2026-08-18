# Qwythos weight mapping

Inspected checkpoint: `mlx-community/Qwythos-9B-v2-OptiQ-4bit`.

- Target: Qwen3.5 hybrid 32-layer text trunk, width 4096, vocab 248320.
- Target attention is hybrid; full attention occurs every fourth layer.
- Sidecar: `optiq/mtp.safetensors`, 29 tensors, one physical MTP layer.
- MTP block: full attention, Q=8192 (32 heads), K/V=1024 (4 heads), head dim 256,
  MLP 4096 → 12288 → 4096.
- `fc.weight` is BF16 `(4096, 8192)`; attention and MLP projections are stored
  affine group-size-64 INT4 as `.weight` U32 plus BF16 `.scales`/`.biases`.
- No dedicated MTP embedding or MTP LM head: use target embeddings and its untied
  `lm_head`.

The adapter therefore quantizes the fresh MTP module before strict weight load.
It never dequantizes the full target model.

## Hidden-state and normalization contract

The target value passed into MTP is the pre-final-RMSNorm residual stream
(`pre_norm`), not the normal output of the text model. The MTP FC receives
`[normalized embedding, normalized hidden]` (`embedding_hidden`). These are
checkpoint properties and are declared by `QwythosAdapter`, rather than being
general defaults for future model families.

This sidecar stores one-dimensional RMSNorm weights in Qwen's raw delta form.
The loader detects that format from its low norm means and restores `+1.0`
before strict loading. This is required for numerical parity; it applies only
to the loaded MTP sidecar, not to the target trunk.
