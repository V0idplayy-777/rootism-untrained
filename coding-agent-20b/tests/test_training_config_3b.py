import json
from pathlib import Path


def test_3b_pretraining_token_budget():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "training_config_3b.json"
    config = json.loads(config_path.read_text())

    assert config["model_size"] == "3B"
    assert config["training"]["total_tokens"] == 18_000_000_000
    assert 0 < config["training"]["warmup_tokens"] < config["training"]["total_tokens"]

    # A full-global-batch schedule rounded down still falls in the requested range.
    tokens_per_step = config["batching"]["global_batch_size"] * config["batching"]["sequence_length"]
    full_batch_tokens = (config["training"]["total_tokens"] // tokens_per_step) * tokens_per_step
    assert 15_000_000_000 <= full_batch_tokens <= 20_000_000_000
