# MLX-MTP

Minimal native-MTP speculative decoding for Apple Silicon.

MLX-MTP runs a model's internal multi-token-prediction (MTP) head with MLX,
without an external draft model.

**Current scope:** Qwythos 9B OptiQ only, with its one physical MTP layer.
The package is structured for additional adapters, but no other checkpoint is
supported yet.

MLX-MTP is an independent implementation inspired by and numerically validated
against the MTPLX oracle. It neither imports MTPLX nor requires it at runtime.

## Results

MTPLX Forge generated an MTP-enabled MLX variant of the Qwythos 9B OptiQ
checkpoint and verified the following result. This is a Forge verification
result, not a throughput benchmark independently reproduced by this
repository's benchmark harness.

| Configuration | Throughput |
| --- | ---: |
| Autoregressive baseline | 36.3 tok/s |
| MTP D2 | 67.1 tok/s |
| Speedup | **1.85x** |

The mean acceptance rate was 93% at D2, measured on an Apple M4 Pro running
macOS 26.6.2 (arm64), with temperature 0.6, top_p 0.95, and top_k 20.

## Forged model

The public [MTP-enabled Qwythos model on Hugging Face](https://huggingface.co/nRanzo/mlx-community-Qwythos-9B-v2-OptiQ-4bit-MTPLX)
was generated with MTPLX Forge from
`mlx-community/Qwythos-9B-v2-OptiQ-4bit`. It provides the distributable
MTP-enabled artifact corresponding to the Forge verification described above.

MLX-MTP provides the independent native MLX implementation, technical details,
and tests in this repository. MTPLX is the runtime/oracle used for optional
numerical validation, while MTPLX Forge produced and benchmarked the public
MTP-enabled model.

## What MLX-MTP implements

MLX-MTP runs the model's internal MTP head with MLX, without an external draft
model. It uses probability-ratio rejection sampling and residual correction so
stochastic decoding preserves the target distribution when draft and target
use the same sampler, and targets Apple Silicon.

## Requirements

- Apple Silicon Mac running macOS (MLX/Metal)
- Python 3.10 or newer
- An MLX-compatible local Qwythos checkpoint containing `config.json` and
  `optiq/mtp.safetensors`

The initial checkpoint is `mlx-community/Qwythos-9B-v2-OptiQ-4bit`. This
project intentionally does not download, modify, redistribute, or include it.
The loader currently accepts a **local checkpoint directory**, not a Hugging
Face model ID. This repository does not contain model weights; the Forge model
is published separately on [Hugging Face](https://huggingface.co/nRanzo/mlx-community-Qwythos-9B-v2-OptiQ-4bit-MTPLX).

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

The repository provides this harness for paired local AR/MTP runs. The
published 1.85x result above comes from the MTPLX Forge verification run; this
README does not claim that `benchmarks/compare.py` independently produced it.
When reproducing performance, use the same machine, model, prompt, and sampling
settings for both runs.

### Published Forge benchmark

The Forge-verified result used the MTP-enabled Qwythos 9B v2 OptiQ 4-bit model
at D2 on an Apple M4 Pro running macOS 26.6.2 (arm64), with temperature 0.6,
top_p 0.95, and top_k 20. It measured 67.1 tok/s versus 36.3 tok/s for the
autoregressive baseline (1.85x), with 93% mean acceptance. This result is
configuration-specific and should not be generalized to other models, prompts,
samplers, or Apple Silicon devices.

## Limitations

- Qwythos text generation only; vision input is unsupported.
- One physical MTP layer; `depth > 1` is rejected rather than approximated.
- Local model directories only; remote IDs, server/API, graph compilation, and
  custom Metal kernels are intentionally out of scope.
- The cache-correction path prioritizes correctness. The published 1.85x
  result is a Forge-verified result on the specified hardware and configuration
  and should not be generalized to all models, prompts, samplers, or Apple
  Silicon devices.
- MTPLX parity is an optional local-development check, not a CI requirement.

See [Qwythos weight mapping](docs/qwythos.md),
[architecture/reference map](docs/REFERENCE.md), and
[benchmarking notes](docs/benchmarking.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*.
