"""Model-family dispatch; no checkpoint paths are hard-coded."""
from pathlib import Path
import json
from .adapters.qwythos import QwythosAdapter

def load_adapter(model: str):
    path = Path(model)
    if not path.is_dir():
        raise ValueError("MTP sidecars require a local MLX checkpoint directory")
    with (path / "config.json").open() as f:
        config = json.load(f)
    if config.get("model_type") == "qwen3_5" and config.get("mtp_file"):
        return QwythosAdapter.load(path)
    raise ValueError(f"no native MTP adapter for model_type={config.get('model_type')!r}")
