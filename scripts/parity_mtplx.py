#!/usr/bin/env python3
"""Numerical oracle comparison (development-only; requires MLX + MTPLX)."""
import argparse
import json
import sys

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompt", default="Explain speculative decoding.")
p.add_argument("--atol", type=float, default=1e-4)
args = p.parse_args()

try:
    import mlx.core as mx
    import mtplx.runtime as mtplx_runtime
except ImportError as exc:
    print(f"SKIP: optional parity dependency unavailable: {exc}", file=sys.stderr)
    raise SystemExit(0)

from mlx_mtp.adapters.qwythos import QwythosAdapter
from mlx_mtp.verification import assert_close

reference = mtplx_runtime.load(args.model, mtp=True)
ours = QwythosAdapter.load(args.model)
ids = mx.array([reference.tokenizer.encode(args.prompt)])

reference_logits, reference_hidden = reference.forward_ar(ids, return_hidden=True, hidden_variant="pre_norm")
ours_logits, ours_hidden = ours.forward(ids, return_hidden=True)
anchor = mx.argmax(reference_logits[:, -1, :], axis=-1).reshape(1, 1)
reference_draft = reference.draft_mtp(reference_hidden[:, -1:, :], anchor, mtp_cache=reference.make_mtp_cache())
ours_draft = ours.mtp_forward(ours_hidden[:, -1:, :], anchor, cache=ours.make_mtp_cache())
mx.eval(reference_logits, reference_hidden, reference_draft, ours_logits, ours_hidden, ours_draft)

def fp32(x):
    return mx.array(x).astype(mx.float32)

report = {
    "target_logits_max_abs_diff": assert_close(fp32(ours_logits), fp32(reference_logits), args.atol, "target logits"),
    "target_hidden_max_abs_diff": assert_close(fp32(ours_hidden), fp32(reference_hidden), args.atol, "target hidden"),
    "mtp_logits_max_abs_diff": assert_close(fp32(ours_draft), fp32(reference_draft), args.atol, "MTP logits"),
    "anchor": int(anchor.item()),
    "draft_top1": int(mx.argmax(ours_draft[:, -1, :], axis=-1).item()),
}
print(json.dumps(report, indent=2))
