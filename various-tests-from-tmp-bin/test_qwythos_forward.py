import sys
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

import mtplx.runtime as runtime

model_path = Path(sys.argv[1]).resolve()

print("Loading...")
r = runtime.load(model_path, mtp=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

prompt = "Explain why the sky appears blue in one short paragraph."
input_ids = tokenizer.encode(prompt, return_tensors="np")

# Convert to MLX
input_ids = mx.array(input_ids)

print("\n=== INPUT ===")
print("shape:", input_ids.shape)
print("dtype:", input_ids.dtype)

# ------------------------------------------------------------
# AR PREFILL
# ------------------------------------------------------------

cache = r.make_cache()

print("\n=== FORWARD AR ===")

out = r.forward_ar(
    input_ids,
    cache=cache,
    return_hidden=True,
    emit_logits=True,
)

print("type:", type(out))

if isinstance(out, tuple):
    print("tuple length:", len(out))

    for i, x in enumerate(out):
        print(f"\n[{i}]")
        print("  type:", type(x))
        print("  shape:", getattr(x, "shape", None))
        print("  dtype:", getattr(x, "dtype", None))
else:
    print("shape:", getattr(out, "shape", None))
    print("dtype:", getattr(out, "dtype", None))

print("\n=== DONE ===")
