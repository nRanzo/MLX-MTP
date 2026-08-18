#!/usr/bin/env python3
import argparse, time
import mlx.core as mx
from mlx_lm import load
from mlx_mtp.adapters import QwythosAdapter
from mlx_mtp.sampling import SamplerConfig
from mlx_mtp.speculative import SpeculativeDecoder

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--prompt", required=True); p.add_argument("--max-tokens", type=int, default=128); p.add_argument("--temperature", type=float, default=.6); p.add_argument("--top-p", type=float, default=.95); p.add_argument("--top-k", type=int, default=20); p.add_argument("--seed", type=int, default=0); args = p.parse_args()
# Load tokenizer independently; adapter loads the trunk plus MTP overlay.
_model, tokenizer = load(args.model)
adapter = QwythosAdapter.load(args.model)
ids = mx.array([tokenizer.encode(args.prompt)])
start = time.perf_counter()
tokens, stats = SpeculativeDecoder(adapter, SamplerConfig(args.temperature, args.top_p, args.top_k), args.seed).generate(ids, args.max_tokens)
mx.eval(mx.array(tokens)); elapsed = time.perf_counter() - start
print(tokenizer.decode(tokens))
print({**stats.__dict__, "seconds": elapsed, "tokens_per_second": len(tokens) / elapsed})
