"""Regression tests for the quality-neutral training optimisations.

These are deliberately strict: any change that reduces precision, drops
tokens/parameters, changes the architecture or alters the learning problem
must fail here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model.config import CodingAgentConfig, get_20b_config  # noqa: E402
from model.config_3b import get_3b_config  # noqa: E402
from model.fast_ops import clear_rope_tables, fused_linear_cross_entropy, rope_tables  # noqa: E402
from model.model import CodingAgentModel, MoEExperts, SwiGLUMLP  # noqa: E402
from training.optimizations import (  # noqa: E402
    apply_model_optimizations,
    model_parameter_count,
    resolve_offload,
    tune_deepspeed_config,
)
from training.train import resolve_batch_settings  # noqa: E402
from training.train_3b import build_3b_model_config, build_deepspeed_config, build_training_plan  # noqa: E402


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "_quality_gate", ROOT / "scripts" / "verify_no_quality_loss.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny(**flags):
    cfg = CodingAgentConfig(
        vocab_size=71,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        max_position_embeddings=64,
        sliding_window=32,
        use_gradient_checkpointing=False,
    )
    for key, value in flags.items():
        setattr(cfg, key, value)
    return cfg


def test_quality_gate_passes():
    gate = _load_gate()
    ok, checks = gate.run_all(quick=True)
    assert ok, "\n".join(checks)
    assert any("bitwise" in line for line in checks)


def test_learning_problem_is_unchanged():
    gate = _load_gate()
    assert gate.recipe_invariants() == []


def test_architecture_and_precision_not_reduced():
    for cfg in (get_3b_config(), get_20b_config()):
        summary = apply_model_optimizations(cfg, {})
        assert summary["fused_linear_cross_entropy"] is True
        assert cfg.mixed_precision == "bf16"
    import ast

    tree = ast.parse((ROOT / "model" / "fast_ops.py").read_text())
    identifiers = [
        (node.attr if isinstance(node, ast.Attribute) else node.id)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    ]
    for banned in ("float8", "int8", "quantize", "allow_tf32", "float16", "half"):
        assert all(banned not in str(name) for name in identifiers), (
            f"precision-reducing construct {banned!r} found"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 3, 8, 64])
def test_fused_ce_matches_full_logits_cross_entropy(dtype, chunk):
    torch.manual_seed(0)
    vocab, seq, batch, hidden = 71, 11, 2, 32
    head = torch.nn.Linear(hidden, vocab, bias=False).to(dtype)
    hidden_states = torch.randn(batch, seq, hidden, dtype=dtype, requires_grad=True)
    ref_hidden = hidden_states.detach().clone().requires_grad_(True)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[0, :3] = -100

    full_logits = head(ref_hidden)[..., :-1, :].contiguous()
    reference = torch.nn.functional.cross_entropy(
        full_logits.view(-1, vocab), labels[..., 1:].contiguous().view(-1), ignore_index=-100
    )
    fused = fused_linear_cross_entropy(
        hidden_states, labels, head, chunk_tokens=chunk, ignore_index=-100
    )
    assert abs(reference.item() - fused.item()) <= 0.04
    reference.backward()
    fused.backward()
    rel = (
        (ref_hidden.grad.float() - hidden_states.grad.float()).pow(2).sum()
        / ref_hidden.grad.float().pow(2).sum()
    ).item() ** 0.5
    assert rel < 0.02
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(head.weight.grad).all()


def test_fused_ce_handles_all_ignored_labels_like_pytorch():
    torch.manual_seed(3)
    head = torch.nn.Linear(16, 23, bias=False)
    hidden = torch.randn(2, 5, 16)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    ref = torch.nn.functional.cross_entropy(
        head(hidden)[:, :-1, :].reshape(-1, 23), labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    fused = fused_linear_cross_entropy(hidden, labels, head, chunk_tokens=2)
    assert torch.isnan(ref) and torch.isnan(fused)


def test_shared_rope_tables_are_identical_to_the_original_and_survive_bfloat16():
    clear_rope_tables()
    cfg = _tiny(num_hidden_layers=3)
    model = CodingAgentModel(cfg)
    ropes = [layer.self_attn.rotary_emb for layer in model.layers]
    assert all(r.cos_cached is ropes[0].cos_cached for r in ropes)
    assert all(r.sin_cached is ropes[0].sin_cached for r in ropes)

    inv = ropes[0]._inv_freq_fp32
    t = torch.arange(16, dtype=torch.float32)
    emb = torch.cat((torch.outer(t, inv), torch.outer(t, inv)), dim=-1)
    assert torch.equal(ropes[0].cos_cached[:16], emb.cos())
    assert torch.equal(ropes[0].sin_cached[:16], emb.sin())

    model.bfloat16()
    assert ropes[0]._inv_freq_fp32.dtype is torch.float32
    assert torch.equal(ropes[0]._inv_freq_fp32, inv)
    q = torch.randn(2, cfg.num_attention_heads, 16, cfg.head_dim, dtype=torch.bfloat16)
    cos, sin = ropes[0](q)
    assert cos.dtype is torch.bfloat16
    assert torch.equal(cos[0, :, 0, :], emb.cos().bfloat16())


def test_fused_mlp_and_sync_free_moe_match_reference_outputs():
    torch.manual_seed(2)
    ref_cfg = _tiny(fast_kernels=False, use_moe=True, moe_num_experts=4, moe_top_k=2)
    reference = SwiGLUMLP(ref_cfg)
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    fast_cfg = _tiny(fast_kernels=True, use_moe=True, moe_num_experts=4, moe_top_k=2)
    fast = SwiGLUMLP(fast_cfg)
    fast.load_state_dict(state)
    x = torch.randn(2, 8, ref_cfg.hidden_size, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    a, b = reference(x_ref), fast(x)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    a.sum().backward()
    b.sum().backward()
    assert torch.allclose(x_ref.grad, x.grad, atol=1e-5, rtol=1e-5)


def test_long_context_beyond_native_max_position_matches_reference():
    """The 20B recipe trains at 32768 despite native max_position_embeddings=8192.

    Both implementations rebuild their RoPE tables on demand; the optimised
    one must produce identical values there, and skipping explicit
    position_ids during training must also be identical.
    """
    torch.manual_seed(9)
    kwargs = dict(max_position_embeddings=32, sliding_window=16,
                  fast_kernels=False, fused_linear_cross_entropy=False)
    reference = CodingAgentModel(_tiny(**kwargs))
    state = {k: v.clone() for k, v in reference.state_dict().items()}
    clear_rope_tables()
    fast_cfg = _tiny(fast_kernels=True, fused_linear_cross_entropy=False,
                     fuse_mlp_down_projection=False,
                     max_position_embeddings=32, sliding_window=16,
                     kv_cache_max_tokens=128)
    fast = CodingAgentModel(fast_cfg)
    fast.load_state_dict(state)

    seq = 96
    ids = torch.randint(0, 71, (2, seq))
    labels = ids.clone()
    pos = torch.arange(seq).unsqueeze(0).expand(2, -1).contiguous()

    a = reference(input_ids=ids, position_ids=pos, labels=labels)
    b = fast(input_ids=ids, position_ids=pos, labels=labels)
    assert torch.allclose(a.logits, b.logits, atol=1e-5, rtol=1e-5)
    assert abs(a.loss.item() - b.loss.item()) < 1e-5

    c = fast(input_ids=ids, labels=labels)
    assert torch.allclose(a.logits, c.logits, atol=1e-5, rtol=1e-5)
    assert abs(a.loss.item() - c.loss.item()) < 1e-5

    a.loss.backward()
    c.loss.backward()
    worst = 0.0
    for (_, pa), (_, pc) in zip(reference.named_parameters(), fast.named_parameters()):
        rel = ((pa.grad.float() - pc.grad.float()).pow(2).sum()
               / pa.grad.float().pow(2).sum().clamp_min(1e-30)).item() ** 0.5
        worst = max(worst, rel)
    assert worst < 1e-5


def test_fused_ce_only_hides_logits_on_the_grad_enabled_loss_path():
    cfg = _tiny(fast_kernels=True, fused_linear_cross_entropy=True, loss_chunk_tokens=4)
    model = CodingAgentModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = ids.clone()

    for training in (True, False):
        model.train(training)
        with torch.no_grad():
            assert model(input_ids=ids, labels=labels).logits.shape == (
                2, 8, cfg.vocab_size
            )

    assert model(input_ids=ids).logits.shape == (2, 8, cfg.vocab_size)

    for training in (True, False):
        model.train(training)
        out = model(input_ids=ids, labels=labels)
        assert out.logits is None
        assert torch.isfinite(out.loss)
    model.train(True)


def test_strict_mode_is_bitwise_identical_in_bfloat16():
    gate = _load_gate()
    result = gate.strict_parity(torch.bfloat16)
    assert result["loss_diff"] == 0.0
    assert result["grad_rel_l2"] == 0.0
    assert result["grad_tensors_bitwise"] == result["grad_tensors"]


def test_20b_auto_gradient_accumulation_resolves():
    recipe = json.loads((ROOT / "configs" / "training_config.json").read_text())
    micro, accumulation, global_batch = resolve_batch_settings(recipe, 8)
    assert (micro, accumulation, global_batch) == (2, 128, 2048)
    with pytest.raises(ValueError):
        bad = copy.deepcopy(recipe)
        bad["batching"]["global_batch_size"] = 2047
        resolve_batch_settings(bad, 8)


def test_deepspeed_tuning_removes_precision_and_overhead_footguns():
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    tuned = tune_deepspeed_config(template, {})
    assert "communication_data_type" not in tuned
    assert tuned["flops_profiler"]["enabled"] is False
    assert tuned["activation_checkpointing"]["cpu_checkpointing"] is False
    assert template["flops_profiler"]["enabled"] is False


def test_both_training_recipes_enable_fast_path_without_changing_hyperparameters():
    for name in ("training_config.json", "training_config_3b.json"):
        recipe = json.loads((ROOT / "configs" / name).read_text())
        assert recipe["optimizations"]["fast_kernels"] is True
        assert recipe["optimizations"]["fused_linear_cross_entropy"] is True
        assert recipe["training"]["precision"] == "bf16"
        assert recipe["training"]["beta2"] == 0.95
        assert recipe["training"]["weight_decay"] == 0.1
        assert recipe["training"]["seed"] == 42


def test_3b_model_config_gets_optimisations_without_architecture_changes():
    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    baseline = get_3b_config()
    cfg = build_3b_model_config(recipe)
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_kv_heads", "max_position_embeddings",
                "sliding_window", "rope_theta"):
        assert getattr(cfg, key) == getattr(baseline, key)
    assert cfg.fused_linear_cross_entropy is True
    assert cfg.fast_kernels is True
    assert cfg.deepspeed_stage == 3
    plan = build_training_plan(recipe, 8)
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    ds = build_deepspeed_config(template, recipe, plan, cfg.hidden_size)
    assert ds["gradient_accumulation_steps"] == plan.gradient_accumulation_steps
    assert "auto" not in json.dumps(ds)


def test_model_parameter_count_matches_instantiated_model():
    tiny = CodingAgentModel(_tiny())
    assert model_parameter_count(tiny.config) == sum(p.numel() for p in tiny.parameters())


def test_offload_policy_is_safe_and_deterministic():
    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    kwargs = dict(
        world_size=8, params=4_310_000_000, micro_batch=1, seq_len=4096,
        vocab_size=128256, layers=20, hidden_size=3072,
    )
    assert resolve_offload(recipe, free_vram_bytes=80 * 1024**3, **kwargs)[:2] == (False, False)
    assert resolve_offload(recipe, free_vram_bytes=2 * 1024**3, **kwargs)[:2] == (True, True)
    explicit = copy.deepcopy(recipe)
    explicit["deepspeed"]["offload_policy"] = "off"
    assert resolve_offload(explicit, free_vram_bytes=2 * 1024**3, **kwargs)[:2] == (False, False)
    explicit["deepspeed"]["offload_policy"] = "on"
    assert resolve_offload(explicit, free_vram_bytes=200 * 1024**3, **kwargs)[:2] == (True, True)


def test_a100_40gb_and_80gb_projections_are_consistent():
    """Both A100 cards have the same compute; only HBM/VRAM differ."""
    from benchmark_training import GPU_PROFILES, projection

    assert set(GPU_PROFILES) == {"a100-40gb", "a100-80gb", "a100-80gb-pcie"}
    assert all(p["bf16_flops_per_second"] == 3.12e14 for p in GPU_PROFILES.values())
    assert GPU_PROFILES["a100-40gb"]["hbm_bytes_per_second"] < GPU_PROFILES["a100-80gb"]["hbm_bytes_per_second"]
    assert GPU_PROFILES["a100-40gb"]["vram_bytes"] == 40 * 1024**3
    assert GPU_PROFILES["a100-80gb"]["vram_bytes"] == 80 * 1024**3

    for gpu_key in GPU_PROFILES:
        small = projection("3B", get_3b_config(), 1, 4096, 18_000_000_000,
                           512, gpu_key=gpu_key)
        large = projection("20B", get_20b_config(), 2, 32768, 300_000_000_000,
                           2048, gpu_key=gpu_key)
        assert small["fits_in_vram"] and not small["offload_needed"]
        assert small["full_run_after_hours_incl_offload"] < small["full_run_before_hours_incl_offload"]
        assert not large["baseline_fits_in_vram"]
        assert large["fits_in_vram"] is (gpu_key != "a100-40gb")
        assert large["full_run_hours_saved_incl_offload"] > 1000
        assert large["logits_after_bytes"] < large["logits_before_bytes"] * 0.01
    p40 = projection("3B", get_3b_config(), 1, 4096, 18_000_000_000, 512,
                     gpu_key="a100-40gb")
    p80 = projection("3B", get_3b_config(), 1, 4096, 18_000_000_000, 512,
                     gpu_key="a100-80gb")
    assert p40["full_run_hours_saved_incl_offload"] > p80["full_run_hours_saved_incl_offload"]


def test_activation_memory_saving_is_large():
    from model.fast_ops import logits_activation_bytes

    before = logits_activation_bytes(2, 32768, 128256, fused=False)
    after = logits_activation_bytes(2, 32768, 128256, fused=True)
    assert after < before * 0.01
    for cfg, micro, seq in ((get_3b_config(), 1, 4096), (get_20b_config(), 2, 32768)):
        b = logits_activation_bytes(micro, seq, cfg.vocab_size, fused=False)
        a = logits_activation_bytes(micro, seq, cfg.vocab_size, fused=True)
        assert a < b * 0.06
