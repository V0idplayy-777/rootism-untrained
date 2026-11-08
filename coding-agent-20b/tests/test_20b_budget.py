"""Budget, curriculum and optimization regression checks for the 20B recipe."""
import json
from pathlib import Path

import numpy as np
import pytest

from dataset.mixture import DEFAULT_MIXTURE, real_backbone_weight
from dataset.processor import MemoryMappedDataset
from training.schedules import curriculum_steps, default_pipeline, seq_length_at

ROOT = Path(__file__).resolve().parents[1]


def test_recipe_stage_budget_and_curriculum():
    config = json.loads((ROOT / 'configs/training_config.json').read_text())
    pipeline = json.loads((ROOT / 'configs/pipeline.json').read_text())['stages']
    budget = config['training']['total_tokens']
    batch = config['batching']['global_batch_size']
    curriculum = config['batching']['sequence_curriculum']
    assert 60e9 <= budget <= 80e9
    assert curriculum == pipeline[0]['data']['seq_curriculum']
    assert [x[1] for x in curriculum] == [2048, 4096, 8192]
    assert [seq_length_at(x, curriculum) for x in (0, 644, 1610)] == [2048, 4096, 8192]
    assert [s.total_steps for s in default_pipeline()] == [s['total_steps'] for s in pipeline]
    for stage in pipeline:
        schedule = stage['data']['seq_curriculum']
        tokens = sum(batch * seq_length_at(i, schedule) for i in range(stage['total_steps']))
        assert tokens == stage['total_tokens']
    assert sum(s['total_tokens'] for s in pipeline) == budget
    steps = curriculum_steps(budget, batch, curriculum)
    used = sum(batch * seq_length_at(i, curriculum) for i in range(steps))
    assert used <= budget < used + batch * seq_length_at(steps, curriculum)
    assert curriculum_steps(5, 1, [[0, 2], [2, 4]]) == 2
    with pytest.raises(ValueError):
        curriculum_steps(100, 1, [[1, 2]])


def test_high_value_mix_and_fast_paths():
    cfg = json.loads((ROOT / 'configs/training_config.json').read_text())
    weights = {s.name: s.weight for s in DEFAULT_MIXTURE}
    assert cfg['data']['mixture'] == weights
    assert weights['the-stack-code'] == .40
    assert real_backbone_weight() >= .70
    for name, value in {'security-sft': .03, 'repair-trace-sft': .04,
                        'app-sft': .05, 'app-repair-sft': .03,
                        'app-preference-dpo': .02, 'recovery-sft': .02}.items():
        assert weights[name] == value
    for flag in ('fast_kernels', 'share_rope_tables', 'fused_linear_cross_entropy',
                 'fuse_mlp_down_projection', 'sync_free_moe', 'auto_offload'):
        assert cfg['optimizations'][flag] is True
    assert cfg['deepspeed']['stage'] == 3
    assert cfg['deepspeed']['offload_policy'] == 'auto'


def test_flat_shards_can_be_rewindowed_without_short_final_sample(tmp_path):
    np.arange(33, dtype=np.uint32).tofile(tmp_path / 'a.bin')
    (tmp_path / 'train_manifest.json').write_text(json.dumps({
        'total_tokens': 33, 'seq_length': 8, 'shards': [{'path': 'a.bin'}]}))
    for length in (2, 4, 8):
        ds = MemoryMappedDataset(str(tmp_path), seq_length=length)
        assert all(len(ds[i]) == length + 1 for i in range(len(ds)))
        assert len(ds) == 32 // length
