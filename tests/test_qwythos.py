import json
from pathlib import Path
import pytest
from mlx_mtp.adapters.qwythos import inspect_qwythos_config

MODEL = Path("/Users/nicola/.mtplx/models/mlx-community--Qwythos-9B-v2-OptiQ-4bit")

@pytest.mark.skipif(not MODEL.is_dir(), reason="local Qwythos fixture is unavailable")
def test_qwythos_contract_matches_checkpoint():
    c = inspect_qwythos_config(MODEL)
    assert (c.hidden_size, c.vocab_size, c.num_layers) == (4096, 248320, 1)
    assert c.hidden_variant == "pre_norm"
    assert c.concat_order == "embedding_hidden"

@pytest.mark.skipif(not MODEL.is_dir(), reason="local Qwythos fixture is unavailable")
def test_qwythos_mtp_sidecar_is_declared_and_present():
    raw = json.loads((MODEL / "config.json").read_text())
    assert raw["mtp_file"] == "optiq/mtp.safetensors"
    assert (MODEL / raw["mtp_file"]).is_file()
