#!/usr/bin/env python3
import argparse
from mlx_mtp.adapters.qwythos import inspect_qwythos_config
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); args = p.parse_args()
print(inspect_qwythos_config(args.model))
