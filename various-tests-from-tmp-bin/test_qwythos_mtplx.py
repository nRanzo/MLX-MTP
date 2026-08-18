import json
import sys
from pathlib import Path

import mtplx.qwen3_5_mtp_patch as qwen_mtp
import mtplx.runtime as runtime

p = qwen_mtp

model_path = Path(sys.argv[1]).resolve()
config = json.loads((model_path / "config.json").read_text())

# ------------------------------------------------------------
# Temporary compatibility patch:
# Qwythos/OptiQ declares MTP as:
#   model_type = qwen3_5
#   mtp_num_hidden_layers = 1
#
# MTPLX currently expects:
#   model_type = qwen3_5_mtp
#   num_nextn_predict_layers = 1
# ------------------------------------------------------------

qwen_mtp.QWEN3_5_MTP_MODEL_TYPES = {
    "qwen3_5",
    "qwen3_5_mtp",
}

def patched_num_mtp_layers(config):
    tcfg = qwen_mtp.text_config(config)
    return int(
        config.get("num_nextn_predict_layers")
        or config.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or 0
    )

qwen_mtp._num_mtp_layers = patched_num_mtp_layers

print("=== CONFIG ===")
print("model_type:", config.get("model_type"))
print("mtp_num_hidden_layers:", config.get("mtp_num_hidden_layers"))
print("is_qwen3_5_mtp_config:",
      qwen_mtp.is_qwen3_5_mtp_config(config))

print("\n=== MTP FILE ===")
mtp_file = qwen_mtp.expected_mtp_file(model_path, config)
print(mtp_file)
print("exists:", mtp_file.exists())

print("\n=== STARTING MTPLX LOAD ===")


def patched_quantize_like_trunk(mtp, config, contract):
    q = config.get("quantization") or p.text_config(config).get("quantization")
    if not q:
        return

    import mlx.nn as nn

    nn.quantize(
        mtp.layers[0],
        group_size=int(q.get("group_size", 64)),
        bits=int(q.get("bits", 4)),
        mode=str(q.get("mode", "affine")),
    )

    print("AFTER QUANTIZATION:")
    print("  fc.weight:", mtp.fc.weight.shape, mtp.fc.weight.dtype)
    print(
        "  q_proj.weight:",
        mtp.layers[0].self_attn.q_proj.weight.shape,
        mtp.layers[0].self_attn.q_proj.weight.dtype,
    )


p._quantize_like_trunk = patched_quantize_like_trunk


# IMPORTANT:
# Call MTPLX runtime.load(), NOT mlx_lm.load().
runtime_obj = runtime.load(
    model_path,
    mtp=True,
)

print("\n=== MTPLX LOAD COMPLETE ===")
print("runtime:", type(runtime_obj))

model = getattr(runtime_obj, "model", None)

print("model:", type(model) if model is not None else None)
print("has mtp:", getattr(model, "mtp", None) is not None)

if model is not None:
    mtp = getattr(model, "mtp", None)
    print(
        "mtp layers:",
        len(getattr(mtp, "layers", []))
        if mtp is not None else 0
    )

print("\n=== VALIDATION ===")

if model is not None:
    print(
        "validate:",
        qwen_mtp.validate_qwen3_5_mtp_support(model)
    )

print("\n=== DONE ===")

print("validate:", qwen_mtp.validate_qwen3_5_mtp_support(model))
print("\n=== GENERATION TEST ===")

tokenizer = runtime_obj.tokenizer

from mlx_lm import generate

response = generate(
    model,
    tokenizer,
    prompt="Explain why the sky appears blue in one short paragraph.",
    max_tokens=100,
)

print(response)