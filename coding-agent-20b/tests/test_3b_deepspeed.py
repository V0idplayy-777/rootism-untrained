"""Optional CPU integration test of the actual ZeRO-3 training engine."""

import copy
import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from dataset.processor import MemoryMappedDataset  # noqa: E402
from model.config import CodingAgentConfig  # noqa: E402
from training.train_3b import (  # noqa: E402
    Pretraining3BModel,
    TokenBatchStream,
    build_deepspeed_config,
    build_training_plan,
    checkpoint_metadata,
    restore_3b_checkpoint,
    run_3b_training,
)


ROOT = Path(__file__).resolve().parent.parent


def test_real_zero3_accumulation_checkpoint_and_resume(tmp_path, monkeypatch):
    deepspeed = pytest.importorskip("deepspeed")
    if torch.cuda.is_available():
        pytest.skip("This smoke test is specific to DeepSpeed's CPU backend")
    import torch.distributed as dist
    from deepspeed.accelerator import get_accelerator

    if dist.is_initialized():
        pytest.skip("Test requires its own single-rank process group")
    torch.set_num_threads(1)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    # Avoid compiling optional shared-memory ops or requiring numactl in CI.
    monkeypatch.setattr(importlib.import_module("deepspeed.comm.torch"), "build_shm_op", lambda: None)
    monkeypatch.setattr(get_accelerator(), "device_count", lambda: 1)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
        recipe["training"].update(total_tokens=47, warmup_tokens=16, precision="bf16")
        recipe["batching"].update(global_batch_size=4, micro_batch_size_per_gpu=2, sequence_length=4)
        recipe["deepspeed"].update(offload_optimizer=False, offload_param=False)
        recipe["checkpointing"].update(save_every_steps=1, save_every_tokens=0, keep_last_n=2)
        recipe["logging"].update(log_every_steps=0, eval_every_steps=0)
        plan = build_training_plan(recipe, 1)
        template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
        ds_config = build_deepspeed_config(template, recipe, plan, hidden_size=16)
        ds_config["zero_optimization"]["overlap_comm"] = False
        ds_config.pop("optimizer")  # A native CPU optimizer avoids a CUDA/C++ build in this test.

        data = tmp_path / "data"
        data.mkdir()
        np.arange(16, dtype=np.uint32).tofile(data / "train_00000.bin")
        (data / "train_manifest.json").write_text(
            json.dumps(
                {
                    "total_tokens": 16,
                    "num_shards": 1,
                    "shards": [{"path": "train_00000.bin", "tokens": 16}],
                }
            )
        )
        dataset = MemoryMappedDataset(str(data), split="train", seq_length=4)

        def new_stream():
            return TokenBatchStream(dataset, 2, 0, 1, recipe["training"]["seed"])

        def new_engine():
            cfg = CodingAgentConfig(
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
            with deepspeed.zero.Init(config_dict_or_path=copy.deepcopy(ds_config)):
                model = Pretraining3BModel(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=copy.deepcopy(ds_config),
                dist_init_required=False,
            )
            return engine, optimizer

        stream = new_stream()
        metadata = checkpoint_metadata(plan, stream, data / "train_manifest.json")
        engine, optimizer = new_engine()
        first = run_3b_training(
            engine,
            optimizer,
            stream,
            recipe,
            build_training_plan(recipe, 1, max_steps=1),
            tmp_path / "checkpoints",
            metadata,
            0,
        )
        assert first.step == engine.global_steps == 1
        assert first.tokens_processed == 16
        assert (tmp_path / "checkpoints" / "latest").read_text() == "step_00000001"

        # ZeRO-3 requires a fresh engine when loading its own checkpoint.
        resumed_engine, resumed_optimizer = new_engine()
        restored = restore_3b_checkpoint(resumed_engine, tmp_path / "checkpoints", plan, metadata)
        final = run_3b_training(
            resumed_engine,
            resumed_optimizer,
            new_stream(),
            recipe,
            plan,
            tmp_path / "checkpoints",
            metadata,
            0,
            state=restored,
        )
        assert final.step == resumed_engine.global_steps == 2
        assert final.tokens_processed == 32 <= recipe["training"]["total_tokens"]
        assert (tmp_path / "checkpoints" / "latest").read_text() == "step_00000002"

        # Consolidation must retain the base model's keys for inference.
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        from model.model import CodingAgentModel

        weights = get_fp32_state_dict_from_zero_checkpoint(str(tmp_path / "checkpoints"))
        plain = CodingAgentModel(resumed_engine.module.config)
        plain.load_state_dict(weights, strict=True)
        plain.eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        assert torch.isfinite(plain(input_ids=ids, labels=ids).loss)
    finally:
        dist.destroy_process_group()


def _run_two_rank_worker(directory: Path):
    """Invoked under torchrun so each rank gets an isolated process."""
    import torch.distributed as dist
    import deepspeed
    from deepspeed.accelerator import get_accelerator

    importlib.import_module("deepspeed.comm.torch").build_shm_op = lambda: None
    get_accelerator().device_count = lambda: 2
    dist.init_process_group("gloo")
    deepspeed.init_distributed(dist_backend="gloo", dist_init_required=False)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
        recipe["training"].update(total_tokens=47, warmup_tokens=16, precision="bf16")
        recipe["batching"].update(global_batch_size=4, micro_batch_size_per_gpu=1, sequence_length=4)
        recipe["deepspeed"].update(offload_optimizer=False, offload_param=False)
        recipe["checkpointing"].update(save_every_steps=1, save_every_tokens=0, keep_last_n=2)
        recipe["logging"].update(log_every_steps=0, eval_every_steps=1)
        recipe["evaluation"].update(eval_tokens=16, eval_batch_size=1)
        plan = build_training_plan(recipe, world)
        template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
        ds_config = build_deepspeed_config(template, recipe, plan, hidden_size=16)
        ds_config["zero_optimization"]["overlap_comm"] = False
        ds_config.pop("optimizer")
        dataset = MemoryMappedDataset(str(directory / "data"), split="train", seq_length=4)
        eval_dataset = MemoryMappedDataset(str(directory / "data"), split="val", seq_length=4)

        def stream():
            return TokenBatchStream(dataset, 1, rank, world, recipe["training"]["seed"])

        def engine_and_optimizer():
            cfg = CodingAgentConfig(
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
            with deepspeed.zero.Init(config_dict_or_path=copy.deepcopy(ds_config)):
                model = Pretraining3BModel(cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config=copy.deepcopy(ds_config),
                dist_init_required=False,
            )
            return engine, optimizer

        sample = next(stream())
        gathered = [torch.empty_like(sample) for _ in range(world)]
        dist.all_gather(gathered, sample)
        assert not torch.equal(gathered[0], gathered[1])
        data_stream = stream()
        metadata = checkpoint_metadata(plan, data_stream, directory / "data" / "train_manifest.json")
        engine, optimizer = engine_and_optimizer()
        first = run_3b_training(
            engine,
            optimizer,
            data_stream,
            recipe,
            build_training_plan(recipe, world, max_steps=1),
            directory / "checkpoints",
            metadata,
            rank,
            eval_dataset=eval_dataset,
        )
        assert first.step == engine.global_steps == 1 and first.tokens_processed == 16
        resumed_engine, resumed_optimizer = engine_and_optimizer()
        restored = restore_3b_checkpoint(resumed_engine, directory / "checkpoints", plan, metadata)
        final = run_3b_training(
            resumed_engine,
            resumed_optimizer,
            stream(),
            recipe,
            plan,
            directory / "checkpoints",
            metadata,
            rank,
            state=restored,
            eval_dataset=eval_dataset,
        )
        assert final.step == resumed_engine.global_steps == 2 and final.tokens_processed == 32
        print(f"rank {rank} completed with {final.tokens_processed} global tokens", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_zero3_budget_and_resume(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("deepspeed")
    if torch.cuda.is_available():
        pytest.skip("This smoke test is specific to DeepSpeed's CPU backend")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "val"):
        np.arange(16, dtype=np.uint32).tofile(data / f"{split}_00000.bin")
        (data / f"{split}_manifest.json").write_text(
            json.dumps(
                {
                    "total_tokens": 16,
                    "num_shards": 1,
                    "shards": [{"path": f"{split}_00000.bin", "tokens": 16}],
                }
            )
        )
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", __file__, str(tmp_path)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-8000:]
    assert "rank 0 completed with 32 global tokens" in result.stdout
    assert "rank 1 completed with 32 global tokens" in result.stdout
    assert (tmp_path / "checkpoints" / "latest").read_text() == "step_00000002"


if __name__ == "__main__":
    import sys

    _run_two_rank_worker(Path(sys.argv[1]))
