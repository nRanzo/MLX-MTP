#!/usr/bin/env python3
import argparse, time
import mlx.core as mx
from mlx_lm import load, generate

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--prompt", required=True); p.add_argument("--max-tokens", type=int, default=128); args = p.parse_args()
model, tokenizer = load(args.model)
start = time.perf_counter()
text = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens, verbose=False)
mx.eval(model.parameters())
elapsed = time.perf_counter() - start
print(text)
print(f"generation_time_s={elapsed:.3f}")
