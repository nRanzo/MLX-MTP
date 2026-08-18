#!/usr/bin/env python3
"""Paired AR/MTP benchmark. Run only on the same Apple Silicon machine."""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--prompt", required=True); p.add_argument("--tokens", type=int, default=256); p.add_argument("--output", default="benchmark-results/result.json"); args = p.parse_args()
root = Path(__file__).parents[1]
def run(script):
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(root / script), "--model", args.model, "--prompt", args.prompt, "--max-tokens", str(args.tokens)], text=True, capture_output=True)
    if result.returncode: raise RuntimeError(result.stderr)
    return time.perf_counter() - start, result.stdout
ar_s, ar = run("examples/qwythos_ar.py")
mtp_s, mtp = run("examples/qwythos_mtp.py")
data = {"model": Path(args.model).name, "hardware": platform.platform(), "generated_tokens": args.tokens, "temperature": .6, "top_p": .95, "top_k": 20, "ar_tokens_per_second": args.tokens/ar_s, "mtp_d1_tokens_per_second": args.tokens/mtp_s, "speedup_vs_ar": ar_s/mtp_s, "ar_output": ar, "mtp_output": mtp}
Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(data, indent=2)); print(json.dumps(data, indent=2))
