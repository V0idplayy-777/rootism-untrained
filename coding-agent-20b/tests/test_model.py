import pytest
from conftest import requires_torch

torch = pytest.importorskip("torch")


@requires_torch
def test_moe_top_k_routing_shapes_and_backprop(tiny_model):

    from model.model import MoEExperts

    moe = MoEExperts(tiny_model.config, num_experts=3, top_k=2)
    x = torch.randn(2, 4, moe.hidden_size, requires_grad=True)

    output = moe(x)

    assert len(moe.experts) == 3
    assert output.shape == (2, 4, moe.intermediate_size)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@requires_torch
def test_mlp_projects_moe_output_to_hidden_size(tiny_model):

    mlp = tiny_model.layers[0].mlp
    x = torch.randn(2, 4, mlp.hidden_size, requires_grad=True)

    output = mlp(x)

    assert output.shape == (2, 4, mlp.hidden_size)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


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
def test_dense_configs_match_advertised_parameter_counts():
    """The named 3B and 20B configs must land on their reported sizes.

    Regression guard: MoE used to be hardcoded on in SwiGLUMLP, which
    silently doubled per-layer params (3B -> 4.31B, 20B -> 49.14B).
    """
    from model.config import get_20b_config
    from model.config_3b import get_3b_config, get_3b_26l_config

    cfg3 = get_3b_config()
    assert abs(cfg3.total_parameters() / 1e9 - 2.55) < 0.01, (
        f"3B primary drifted: {cfg3.total_parameters()/1e9:.3f}B"
    )

    cfg3d = get_3b_26l_config()
    assert 3.0 < cfg3d.total_parameters() / 1e9 < 3.15, (
        f"3B deep drifted: {cfg3d.total_parameters()/1e9:.3f}B"
    )

    cfg20 = get_20b_config()
    assert abs(cfg20.total_parameters() / 1e9 - 24.98) < 0.05, (
        f"20B drifted: {cfg20.total_parameters()/1e9:.3f}B"
    )

    assert cfg3.use_moe is False
    assert cfg20.use_moe is False


@requires_torch
def test_total_parameters_matches_pytorch_count(tiny_model):
    """config.total_parameters() must equal sum(p.numel())."""
    cfg = tiny_model.config
    actual = sum(p.numel() for p in tiny_model.parameters())
    assert cfg.total_parameters() == actual, (
        f"estimator {cfg.total_parameters():,} != pytorch {actual:,}"
    )


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
