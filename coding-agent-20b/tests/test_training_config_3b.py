import json
from pathlib import Path


def test_3b_pretraining_token_budget():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "training_config_3b.json"
    config = json.loads(config_path.read_text())

    assert config["model_size"] == "3B"
    assert config["training"]["total_tokens"] == 18_000_000_000
    assert 0 < config["training"]["warmup_tokens"] < config["training"]["total_tokens"]

    tokens_per_step = config["batching"]["global_batch_size"] * config["batching"]["sequence_length"]
    full_batch_tokens = (config["training"]["total_tokens"] // tokens_per_step) * tokens_per_step
    assert 15_000_000_000 <= full_batch_tokens <= 20_000_000_000


def test_data_block_records_the_synthetic_token_cap():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("configs/training_config_3b.json", "configs/training_config.json"):
        data = json.loads((root / name).read_text())["data"]
        budget = data["synthetic_budget"]
        assert budget["synthetic_token_cap"] == 0.05
        assert budget["synthetic_token_plan_share"] <= budget["synthetic_token_cap"]
        assert budget["synthetic_document_share"] == 0.21
        from dataset.mixture import SYNTHETIC_TOKEN_SLICES, synthetic_share

        assert set(budget["synthetic_token_slices"]) == set(SYNTHETIC_TOKEN_SLICES)
        assert budget["synthetic_document_share"] == synthetic_share()
