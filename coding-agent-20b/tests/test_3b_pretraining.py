import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from dataset.processor import MemoryMappedDataset  # noqa: E402
from model.config import CodingAgentConfig  # noqa: E402
from training.train_3b import (  # noqa: E402
    Pretraining3BModel,
    PretrainState,
    TokenBatchStream,
    build_3b_model_config,
    build_deepspeed_config,
    build_training_plan,
    checkpoint_metadata,
    resolve_deepspeed_path,
    restore_3b_checkpoint,
    run_3b_training,
)


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def recipe():
    return json.loads((ROOT / "configs" / "training_config_3b.json").read_text())


def _small_recipe(recipe):
    config = copy.deepcopy(recipe)
    config["training"].update(total_tokens=63, warmup_tokens=16, precision="fp32")
    config["batching"].update(global_batch_size=4, micro_batch_size_per_gpu=2, sequence_length=4)
    config["deepspeed"].update(offload_optimizer=False, offload_param=False)
    config["checkpointing"].update(save_every_steps=1, save_every_tokens=0, keep_last_n=2)
    config["logging"].update(log_every_steps=0, eval_every_steps=1)
    config["evaluation"].update(eval_batch_size=2, eval_tokens=16)
    return config


def _write_dataset(directory, split="train", length=16, shard_sizes=(1, 2, 5, 8)):
    directory.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(length, dtype=np.uint32)
    assert sum(shard_sizes) == length
    shards = []
    offset = 0
    for index, size in enumerate(shard_sizes):
        name = f"{split}_{index:05d}.bin"
        tokens[offset : offset + size].tofile(directory / name)
        shards.append({"path": name, "tokens": size})
        offset += size
    manifest = {"total_tokens": length, "num_shards": len(shards), "shards": shards}
    (directory / f"{split}_manifest.json").write_text(json.dumps(manifest))
    return MemoryMappedDataset(str(directory), split=split, seq_length=4)


def _tiny_model():
    config = CodingAgentConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        max_position_embeddings=32,
        sliding_window=8,
        use_gradient_checkpointing=True,
    )
    return Pretraining3BModel(config)


class LocalAccumulatingEngine:
    """CPU test double for DeepSpeed's loss scaling, updates and checkpoints."""

    def __init__(self, model, accumulation):
        self.module = model
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
        self.accumulation = accumulation
        self.micro_steps = 0
        self.global_steps = 0
        self.training_inputs = []
        self.saved = []

    def __call__(self, *, input_ids, labels):
        assert labels is input_ids
        if self.module.training:
            self.training_inputs.append(input_ids.detach().clone())
        return self.module(input_ids=input_ids, labels=labels)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def backward(self, loss):
        (loss / self.accumulation).backward()

    def step(self):
        self.micro_steps += 1
        if self.micro_steps % self.accumulation == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_steps += 1

    def save_checkpoint(self, directory, tag, client_state):
        target = Path(directory) / tag
        target.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "micro_steps": self.micro_steps,
                "global_steps": self.global_steps,
                "client_state": client_state,
            },
            target / "state.pt",
        )
        (Path(directory) / "latest").write_text(tag)
        self.saved.append(tag)
        return True

    def load_checkpoint(self, directory):
        root = Path(directory)
        checkpoint = root / (root / "latest").read_text() / "state.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.micro_steps = state["micro_steps"]
        self.global_steps = state["global_steps"]
        return str(checkpoint), state["client_state"]


def test_3b_budget_and_accumulation_use_all_ranks(recipe):
    for world_size, accumulation in [(1, 256), (2, 128), (8, 32), (16, 16), (256, 1)]:
        plan = build_training_plan(recipe, world_size)
        assert plan.gradient_accumulation_steps == accumulation
        assert plan.tokens_per_step == 8_388_608
        assert plan.total_steps == 2_145
        assert plan.warmup_steps == 119
        assert plan.planned_tokens == 17_993_564_160
        assert plan.planned_tokens <= recipe["training"]["total_tokens"]
    assert build_training_plan(recipe, 8, max_steps=3).run_steps == 3
    assert build_training_plan(recipe, 8, max_steps=100_000).run_steps == 2_145


def test_3b_plan_rejects_bad_topology_and_budget(recipe):
    with pytest.raises(ValueError, match="divisible"):
        build_training_plan(recipe, 3)
    bad = copy.deepcopy(recipe)
    bad["batching"]["gradient_accumulation_steps"] = 1
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        build_training_plan(bad, 8)
    bad = copy.deepcopy(recipe)
    bad["training"]["total_tokens"] = 100
    bad["training"]["warmup_tokens"] = 0
    with pytest.raises(ValueError, match="one complete global batch"):
        build_training_plan(bad, 8)
    with pytest.raises(ValueError, match="max_steps"):
        build_training_plan(recipe, 8, max_steps=0)
    bad = copy.deepcopy(recipe)
    bad["model_size"] = "20B"
    with pytest.raises(ValueError, match="3B"):
        build_training_plan(bad, 8)


def test_deepspeed_config_resolves_3b_without_changing_template(recipe, tmp_path, monkeypatch):
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    original = copy.deepcopy(template)
    plan = build_training_plan(recipe, 8)
    model_config = build_3b_model_config(recipe)
    assert model_config.num_hidden_layers == 23
    assert model_config.hidden_size == 3072
    config = build_deepspeed_config(template, recipe, plan, model_config.hidden_size)
    assert template == original
    assert config["train_batch_size"] == 1024
    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 32
    assert config["bf16"]["enabled"] and not config["fp16"]["enabled"]
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert "offload_param" not in config["zero_optimization"]
    assert "auto" not in json.dumps(config)
    assert "scheduler" not in config
    assert config["optimizer"]["params"]["betas"] == [0.9, 0.95]
    monkeypatch.chdir(tmp_path)
    assert (
        resolve_deepspeed_path(str(ROOT / "configs" / "training_config_3b.json"), "configs/ds_config_zero3.json")
        == ROOT / "configs" / "ds_config_zero3.json"
    )


def test_3b_entrypoints_dispatch_without_initializing_other_models(recipe, monkeypatch):
    import training.train as standard
    import training.train_3b as implementation
    import training.train_production as production

    calls = []
    sentinel = object()
    monkeypatch.setattr(implementation, "train_3b", lambda args, config: calls.append((args, config)) or sentinel)
    config_path = str(ROOT / "configs" / "training_config_3b.json")
    args = SimpleNamespace(config=config_path, model_size=None)
    assert standard.train(args) is sentinel
    assert calls[-1][1]["training"]["total_tokens"] == 18_000_000_000
    assert standard.train(SimpleNamespace(config=config_path, model_size="3B")) is sentinel
    assert (
        production.train(SimpleNamespace(config=config_path, data_dir="data/processed", output_dir="checkpoints-3b"))
        is sentinel
    )
    with pytest.raises(ValueError, match="must be used together"):
        standard.train(SimpleNamespace(config=config_path, model_size="20B"))
    with pytest.raises(ValueError, match="must be used together"):
        standard.train(SimpleNamespace(config=str(ROOT / "configs" / "training_config.json"), model_size="3B"))
    with pytest.raises(ValueError, match="does not support --use_wandb"):
        production.train(SimpleNamespace(config=config_path, use_wandb=True))
    assert len(calls) == 3


def test_token_stream_shards_data_across_ranks_and_resumes(tmp_path):
    dataset = _write_dataset(tmp_path)
    rank0 = TokenBatchStream(dataset, batch_size=1, rank=0, world_size=2, seed=7)
    rank1 = TokenBatchStream(dataset, batch_size=1, rank=1, world_size=2, seed=7)
    first0, first1 = next(rank0), next(rank1)
    assert not torch.equal(first0, first1)
    assert (first0[0, 1:] - first0[0, :-1] == 1).all()
    assert (first1[0, 1:] - first1[0, :-1] == 1).all()
    saved_epoch, saved_batch = rank0.epoch, rank0.batch_in_epoch
    expected = next(rank0)
    resumed = TokenBatchStream(dataset, 1, 0, 2, 7, epoch=saved_epoch, batch_in_epoch=saved_batch)
    assert torch.equal(next(resumed), expected)
    saved_epoch, saved_batch = rank0.epoch, rank0.batch_in_epoch
    expected = next(rank0)
    resumed = TokenBatchStream(dataset, 1, 0, 2, 7, epoch=saved_epoch, batch_in_epoch=saved_batch)
    assert torch.equal(next(resumed), expected)


def test_3b_training_stops_at_budget_and_resume_matches_uninterrupted(tmp_path, recipe):
    torch.set_num_threads(1)
    config = _small_recipe(recipe)
    train_data = _write_dataset(tmp_path / "data")
    eval_data = _write_dataset(tmp_path / "data", "val")
    plan = build_training_plan(config, 1, max_steps=999)
    assert plan.total_steps == 3 and plan.planned_tokens == 48

    def new_engine(seed):
        torch.manual_seed(seed)
        return LocalAccumulatingEngine(_tiny_model(), plan.gradient_accumulation_steps)

    def new_stream():
        return TokenBatchStream(train_data, 2, 0, 1, config["training"]["seed"])

    stream = new_stream()
    metadata = checkpoint_metadata(plan, stream, tmp_path / "data" / "train_manifest.json")
    baseline = new_engine(10)
    initial = [parameter.detach().clone() for parameter in baseline.module.parameters()]
    unrelated = tmp_path / "full" / "step_00000000"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("unrelated")
    full = run_3b_training(
        baseline, baseline.optimizer, stream, config, plan, tmp_path / "full", metadata, 0, eval_dataset=eval_data
    )
    assert full == PretrainState(step=3, tokens_processed=48, epoch=2, batch_in_epoch=2)
    assert baseline.global_steps == 3 and baseline.micro_steps == 6
    assert any(not torch.equal(before, after) for before, after in zip(initial, baseline.module.parameters()))
    assert (unrelated / "keep.txt").read_text() == "unrelated"
    assert sorted(p.name for p in (tmp_path / "full").glob("step_*") if (p / ".3b-pretrain").is_file()) == [
        "step_00000002",
        "step_00000003",
    ]

    partial = new_engine(10)
    partial_plan = build_training_plan(config, 1, max_steps=1)
    partial_state = run_3b_training(
        partial,
        partial.optimizer,
        new_stream(),
        config,
        partial_plan,
        tmp_path / "resume",
        metadata,
        0,
        eval_dataset=eval_data,
    )
    assert partial_state.tokens_processed == 16 and partial.global_steps == 1
    resumed = new_engine(999)
    state = restore_3b_checkpoint(resumed, tmp_path / "resume", plan, metadata)
    final = run_3b_training(
        resumed,
        resumed.optimizer,
        new_stream(),
        config,
        plan,
        tmp_path / "resume",
        metadata,
        0,
        state=state,
        eval_dataset=eval_data,
    )
    assert final == full
    assert resumed.global_steps == 3 and resumed.micro_steps == 6
    for expected, actual in zip(baseline.module.parameters(), resumed.module.parameters()):
        torch.testing.assert_close(expected, actual)
    assert all(torch.equal(x, y) for x, y in zip(baseline.training_inputs[2:], resumed.training_inputs))
    assert (tmp_path / "resume" / "latest").read_text() == "step_00000003"
    with pytest.raises(ValueError, match="does not match"):
        restore_3b_checkpoint(resumed, tmp_path / "resume", plan, {**metadata, "world_size": 8})


def test_token_based_checkpoint_interval(tmp_path, recipe):
    config = _small_recipe(recipe)
    config["checkpointing"].update(save_every_steps=0, save_every_tokens=20, keep_last_n=5)
    config["logging"]["eval_every_steps"] = 0
    dataset = _write_dataset(tmp_path / "data")
    plan = build_training_plan(config, 1)
    stream = TokenBatchStream(dataset, 2, 0, 1, config["training"]["seed"])
    metadata = checkpoint_metadata(plan, stream, tmp_path / "data" / "train_manifest.json")
    engine = LocalAccumulatingEngine(_tiny_model(), plan.gradient_accumulation_steps)
    run_3b_training(engine, engine.optimizer, stream, config, plan, tmp_path / "ckpt", metadata, 0)
    assert engine.saved == ["step_00000002", "step_00000003"]


def test_3b_driver_uses_recipe_and_resumes(tmp_path, recipe, monkeypatch):
    import contextlib
    import sys
    import training.train_3b as trainer

    config = _small_recipe(recipe)
    config["logging"]["eval_every_steps"] = 0
    config["deepspeed"]["config_file"] = str(ROOT / "configs" / "ds_config_zero3.json")
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)
    output_dir = tmp_path / "output"
    recipe_path = tmp_path / "3b.json"
    recipe_path.write_text(json.dumps(config))
    initialized = []

    class FakeDeepSpeed:
        class zero:
            @staticmethod
            def Init(config_dict_or_path):
                initialized.append(config_dict_or_path)
                return contextlib.nullcontext()

        @staticmethod
        def init_distributed():
            pass

        @staticmethod
        def initialize(*, model, model_parameters, config, dist_init_required):
            assert isinstance(model, Pretraining3BModel)
            assert config["train_batch_size"] == 4
            assert config["gradient_accumulation_steps"] == 2
            assert "auto" not in json.dumps(config)
            engine = LocalAccumulatingEngine(model, config["gradient_accumulation_steps"])
            return engine, engine.optimizer, None, None

    tiny_cfg = CodingAgentConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        max_position_embeddings=32,
        sliding_window=8,
        use_gradient_checkpointing=True,
    )
    monkeypatch.setitem(sys.modules, "deepspeed", FakeDeepSpeed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(trainer, "build_3b_model_config", lambda cfg: tiny_cfg)
    args = SimpleNamespace(
        config=str(recipe_path), data_dir=str(data_dir), output_dir=str(output_dir), max_steps=1, resume=False
    )
    first = trainer.train_3b(args, config)
    assert first.step == 1 and first.tokens_processed == 16
    with pytest.raises(FileExistsError, match="--resume"):
        trainer.train_3b(args, config)
    args.resume = True
    args.max_steps = 999
    final = trainer.train_3b(args, config)
    assert final.step == 3 and final.tokens_processed == 48
    assert len(initialized) == 2
