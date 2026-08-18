#!/usr/bin/env python3
"""Launch an optional MTPLX oracle process; never imports MTPLX in mlx-mtp."""
import argparse, shutil, subprocess, sys
p = argparse.ArgumentParser(); p.add_argument("--mtplx-python", required=True); p.add_argument("--model", required=True); args = p.parse_args()
if not shutil.which(args.mtplx_python) and not __import__("pathlib").Path(args.mtplx_python).is_file():
    print("SKIP: MTPLX Python is unavailable", file=sys.stderr); raise SystemExit(0)
code = "import mtplx.runtime; print('MTPLX oracle import succeeded')"
result = subprocess.run([args.mtplx_python, "-c", code], text=True, capture_output=True)
if result.returncode: print("SKIP: MTPLX unavailable or Metal inaccessible:\n" + result.stderr); raise SystemExit(0)
print(result.stdout, end="")
