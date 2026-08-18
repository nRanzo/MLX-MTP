# MLX-MTP

Minimal native-MTP speculative decoding for Apple Silicon.

MLX-MTP runs a model's internal multi-token-prediction (MTP) head with MLX:
there is no external draft model. It uses probability-ratio rejection sampling
and residual correction, so stochastic decoding preserves the target
distribution when draft and target use the same sampler.

**Current scope:** Qwythos 9B OptiQ only, with its one physical MTP layer.
The package is structured for additional adapters, but no other checkpoint is
supported yet.

MLX-MTP is an independent implementation inspired by and numerically validated
against MTPLX. It neither imports MTPLX nor requires it at runtime.

## Requirements

- Apple Silicon Mac running macOS (MLX/Metal)
- Python 3.10 or newer
- An MLX-compatible local Qwythos checkpoint containing `config.json` and
  `optiq/mtp.safetensors`

The initial checkpoint is `mlx-community/Qwythos-9B-v2-OptiQ-4bit`. This
project intentionally does not download, modify, redistribute, or include it.
The loader currently accepts a **local checkpoint directory**, not a Hugging
Face model ID.

## Install

```sh
git clone https://github.com/nranzo/mlx-mtp.git
cd mlx-mtp
python -m pip install -e '.[dev]'
```

MLX and mlx-lm evolve quickly. The project declares minimum versions in
`pyproject.toml`; its validated development environment uses the MLX stack
bundled with the local MTPLX runtime. Please report the output of
`python -m pip show mlx mlx-lm` with compatibility issues.

## Quick start

Inspect the checkpoint first. This confirms the Qwythos-specific sidecar and
prints its 29 tensor names, shapes, and storage dtypes.

```sh
python scripts/inspect_model.py --model /path/to/Qwythos-9B-v2-OptiQ-4bit
python scripts/inspect_mtp_weights.py --model /path/to/Qwythos-9B-v2-OptiQ-4bit
```

Expected contract summary:

```text
MTPConfig(hidden_size=4096, vocab_size=248320, num_layers=1,
          hidden_variant='pre_norm', concat_order='embedding_hidden', ...)
```

Run standard autoregressive generation:

```sh
python examples/qwythos_ar.py \
  --model /path/to/Qwythos-9B-v2-OptiQ-4bit \
  --prompt 'Explain speculative decoding.'
```

Run native MTP speculative decoding:

```sh
python examples/qwythos_mtp.py \
  --model /path/to/Qwythos-9B-v2-OptiQ-4bit \
  --prompt 'Explain speculative decoding.' \
  --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 128
```

## Python API

```python
import mlx.core as mx
from mlx_mtp import SamplerConfig, SpeculativeDecoder, load_adapter

adapter = load_adapter("/path/to/Qwythos-9B-v2-OptiQ-4bit")
input_ids = mx.array([[...token_ids...]])
decoder = SpeculativeDecoder(
    adapter,
    sampler=SamplerConfig(temperature=0.6, top_p=0.95, top_k=20),
    seed=0,
)
token_ids, stats = decoder.generate(input_ids, max_tokens=128)
```

## Correctness and validation

Qwythos has one MTP layer. At each step MLX-MTP samples target anchor `N+1`,
uses MTP to propose `N+2`, and verifies `[N+1, N+2]` in one target forward.
Only the MTP proposal is accepted with `min(1, p/q)` in FP32. On rejection,
the replacement is sampled from normalized `max(p-q, 0)`—not from `p` and not
by argmax matching. A full acceptance receives a bonus target token.

Target and MTP caches are distinct. Verification uses a cloned target cache;
a rejected draft is replayed from the committed state, so it never leaks into
the target cache.

On the Qwythos checkpoint, against the installed MTPLX oracle, the current
implementation has been checked with one prompt for:

| Comparison | Maximum absolute difference |
| --- | ---: |
| Target logits | 0.0 |
| Target pre-norm hidden states | 0.0 |
| MTP logits | 0.0 |

Greedy (`temperature=0`) AR and speculative generation also produced the same
six-token continuation in that integration check. This is a validation result,
not a throughput claim or a substitute for broader stochastic testing.

Run the CPU suite:

```sh
pytest
```

Run development-only parity with MTPLX (optional; MTPLX is never installed by
this project). Set the interpreter path explicitly:

```sh
MTPLX_PYTHON='/Users/nicola/Library/Application Support/MTPLX/runtime-venv/bin/python'
PYTHONPATH=. "$MTPLX_PYTHON" scripts/parity_mtplx.py --model /path/to/model
```

## Benchmarking

Benchmarks must be paired AR/MTP runs on the same machine and model settings.
They write JSON containing platform, tokens/sec, and speedup:

```sh
python benchmarks/compare.py \
  --model /path/to/Qwythos-9B-v2-OptiQ-4bit \
  --prompt 'Explain speculative decoding.' \
  --tokens 256
```

No benchmark result is claimed or published by this repository yet.

## Limitations

- Qwythos text generation only; vision input is unsupported.
- One physical MTP layer; `depth > 1` is rejected rather than approximated.
- Local model directories only; remote IDs, server/API, graph compilation, and
  custom Metal kernels are intentionally out of scope.
- The current cache-correction path prioritizes correctness; profile it before
  claiming speedups.
- MTPLX parity is an optional local-development check, not a CI requirement.

See [Qwythos weight mapping](docs/qwythos.md),
[architecture/reference map](docs/REFERENCE.md), and
[benchmarking notes](docs/benchmarking.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*.
