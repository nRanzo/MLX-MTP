#!/usr/bin/env python3
"""Print MTP SafeTensors metadata without loading a model into MLX."""
import argparse
from pathlib import Path
from safetensors import safe_open

p = argparse.ArgumentParser(); p.add_argument("--model", required=True); args = p.parse_args()
path = Path(args.model) / "optiq" / "mtp.safetensors"
with safe_open(str(path), framework="np") as f:
    keys = list(f.keys()); print(f"{path}: {len(keys)} tensors")
    for key in keys:
        s = f.get_slice(key); shape = tuple(s.get_shape()); print(f"{key:58} {str(shape):20} {s.get_dtype()}")
