import pytest
from conftest import requires_torch

torch = pytest.importorskip("torch")


@requires_torch
def test_forward_and_loss(tiny_model):

    ids = torch.randint(0, 320, (2, 16))

    out = tiny_model(input_ids=ids, labels=ids.clone())

    assert out.logits.shape == (2, 16, 320)

    assert out.loss is not None and torch.isfinite(out.loss)


@requires_torch
def test_generate_sampling_controls(tiny_model):

    from model.generation import GenerationConfig

    ids = torch.randint(0, 320, (1, 8))

    out = tiny_model.generate(
        ids.clone(),
        generation_config=GenerationConfig(
            max_new_tokens=5, temperature=0.8, top_p=0.9, top_k=10, min_p=0.05, seed=123
        ),
    )

    assert out.shape[1] == 13

    out2 = tiny_model.generate(ids.clone(), max_new_tokens=4, do_sample=False)

    assert out2.shape[1] == 12


@requires_torch
def test_kv_cache_trim_and_yarn_config():

    from model.config import get_20b_32k_config
    from model.model import PagedKVCache

    cfg = get_20b_32k_config()

    assert cfg.max_position_embeddings == 32768 and cfg.rope_scaling_type == "yarn"

    cache = [(torch.zeros(1, 2, 100, 8), torch.zeros(1, 2, 100, 8))]

    trimmed = PagedKVCache(max_tokens=32, sink_tokens=4).trim(cache)

    assert trimmed[0][0].shape[2] == 32


@requires_torch
def test_samplers_keep_valid_distribution():

    from model.generation import (
        GenerationConfig,
        filter_min_p,
        filter_top_p,
        sample_next,
    )

    logits = torch.randn(320)

    assert torch.isfinite(filter_top_p(logits.clone(), 0.9)).any()

    assert torch.isfinite(filter_min_p(logits.clone(), 0.1)).any()

    tok = sample_next(logits, [1, 2], GenerationConfig(do_sample=True, seed=0))

    assert 0 <= tok < 320
