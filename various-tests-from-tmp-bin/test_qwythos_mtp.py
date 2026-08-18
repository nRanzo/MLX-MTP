import sys
import json
from pathlib import Path

import mtplx.qwen3_5_mtp_patch as p

# ---- Patch detector ----
p.QWEN3_5_MTP_MODEL_TYPES = {"qwen3_5", "qwen3_5_mtp"}

def patched_num_mtp_layers(config):
    tcfg = p.text_config(config)
    return int(
        config.get("num_nextn_predict_layers")
        or config.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or 0
    )

p._num_mtp_layers = patched_num_mtp_layers

# ---- Verify ----
model_path = Path(sys.argv[1]).resolve()
config = json.loads((model_path / "config.json").read_text())

print("=== QWYTHOS MTP TEST ===")
print("model:", model_path)
print("model_type:", config.get("model_type"))
print("mtp_num_hidden_layers:", config.get("mtp_num_hidden_layers"))
print("is_mtp:", p.is_qwen3_5_mtp_config(config))

mtp_file = p.expected_mtp_file(model_path, config)
print("mtp_file:", mtp_file)
print("mtp_exists:", mtp_file.exists())

# ---- Install Qwen3.5 MTP trunk shim ----
p.install_qwen3_5_mtp_trunk_shim()

print("trunk shim: installed")

# ---- Now load the actual model ----
from mlx_lm import load

print("\nLoading model...")
model, tokenizer = load(str(model_path))

print("\n=== LOAD COMPLETE ===")
print("model class:", type(model))
print("has mtp:", getattr(model, "mtp", None) is not None)
print("mtp layers:", len(getattr(getattr(model, "mtp", None), "layers", [])))

print("\n=== VALIDATION ===")
print(
    "MTP supported:",
    p.validate_qwen3_5_mtp_support(model)
)
