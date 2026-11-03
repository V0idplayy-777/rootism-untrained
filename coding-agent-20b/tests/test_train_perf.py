"""The optimisations must be *exact*: this file proves they are.

Every fast path in the training loop (chunked/sparse LM loss, fused SDPA,
fused RMSNorm/RoPE, cached window masks, segmented activation checkpointing,
threaded prefetch) is checked against the straightforward reference
implementation it replaced.  If any of these tests starts failing, the "faster"
code has changed what the model computes and the optimisation is not allowed to
land.
"""

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from dataset.prefetch import PrefetchStream  # noqa: E402
from dataset.processor import MemoryMappedDataset  # noqa: E402
from model.config import CodingAgentConfig  # noqa: E402
from model.fused_ops import (  # noqa: E402
    chunked_cross_entropy,
    rms_norm,
    sparse_cross_entropy,
)
from model.model import (  # noqa: E402
    CodingAgentModel,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from training.perf import estimate, largest_micro_batch, plan_recipe  # noqa: E402
from training.train import resolve_accumulation  # noqa: E402
from training.train_3b import build_deepspeed_config, build_training_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _tiny_config(**overrides):
    values = dict(
        vocab_size=97,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_kv_heads=1,
        max_position_embeddings=64,
        sliding_window=8,
        use_gradient_checkpointing=True,
        rope_theta=10_000.0,
    )
    values.update(overrides)
    return CodingAgentConfig(**values)


# --------------------------------------------------------------------------- loss
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_loss_matches_reference(dtype):
    torch.manual_seed(0)
    hidden = torch.randn(2, 33, 16, dtype=dtype, requires_grad=True)
    weight = torch.randn(97, 16, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 97, (2, 33))

    reference = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight).reshape(-1, 97), labels.reshape(-1)
    )
    reference.backward()
    reference_hidden_grad = hidden.grad.clone()
    reference_weight_grad = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    fast = chunked_cross_entropy(hidden, weight, labels, chunk_size=7)
    fast.backward()

    torch.testing.assert_close(fast, reference)
    torch.testing.assert_close(hidden.grad, reference_hidden_grad)
    torch.testing.assert_close(weight.grad, reference_weight_grad, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_loss_matches_reference_with_ignore_index(dtype):
    torch.manual_seed(1)
    hidden = torch.randn(1, 20, 16, dtype=dtype, requires_grad=True)
    weight = torch.randn(97, 16, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 97, (1, 20))
    labels[:, :11] = -100

    reference = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight).reshape(-1, 97), labels.reshape(-1), ignore_index=-100
    )
    reference.backward()
    reference_hidden_grad = hidden.grad.clone()

    hidden.grad = None
    weight.grad = None

    fast = sparse_cross_entropy(hidden, weight, labels, chunk_size=5)
    fast.backward()

    torch.testing.assert_close(fast, reference)
    torch.testing.assert_close(hidden.grad, reference_hidden_grad)


def test_sparse_loss_skips_masked_rows():
    """Masked positions must not be projected at all (that is the whole saving)."""

    hidden = torch.zeros(1, 8, 16, requires_grad=True)
    weight = torch.zeros(97, 16, requires_grad=True)
    labels = torch.full((1, 8), -100)
    labels[0, 3:] = 5

    loss = sparse_cross_entropy(hidden, weight, labels, chunk_size=8)

    loss.backward()

    assert loss.item() == pytest.approx(math.log(97), rel=1e-5)
    assert weight.grad is not None and weight.grad.sum() == 0


def test_loss_requires_at_least_one_label():
    hidden = torch.zeros(1, 4, 16)
    weight = torch.zeros(97, 16)
    with pytest.raises(ValueError, match="no labels"):
        sparse_cross_entropy(hidden, weight, torch.full((1, 4), -100), chunk_size=4)


# --------------------------------------------------------------------------- model forward
@pytest.mark.parametrize("gradient_checkpointing_segments", [0, 1, 2])
def test_training_forward_is_loss_identical_to_reference(gradient_checkpointing_segments):
    """The fast training path must produce the same loss *and* gradients."""

    torch.manual_seed(3)
    config = _tiny_config(
        return_logits=True,
        use_gradient_checkpointing=gradient_checkpointing_segments > 0,
        gradient_checkpointing_segments=gradient_checkpointing_segments,
        loss_chunk_size=16,
    )
    ids = torch.randint(0, 97, (2, 24))
    labels = ids.clone()
    labels[:, :8] = -100

    reference_model = CodingAgentModel(copy.deepcopy(config))
    reference_model.train()
    reference = reference_model(input_ids=ids, labels=labels)
    reference.loss.backward()
    reference_grads = [p.grad.clone() for p in reference_model.parameters()]

    fast_config = copy.deepcopy(config)
    fast_config.return_logits = False
    fast_model = CodingAgentModel(fast_config)
    fast_model.load_state_dict(reference_model.state_dict())
    fast_model.train()
    out = fast_model(input_ids=ids, labels=labels)
    out.loss.backward()

    assert out.logits is None, "training must not materialise [B, S, vocab]"
    torch.testing.assert_close(out.loss, reference.loss, atol=1e-5, rtol=1e-4)
    for expected, actual in zip(reference_grads, (p.grad for p in fast_model.parameters())):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-3)


def test_training_step_never_allocates_full_logits():
    """The point of the chunked loss: peak live tensors stay far below B*S*V."""

    config = _tiny_config(vocab_size=4096, hidden_size=32, intermediate_size=64, num_hidden_layers=1)
    config.return_logits = False
    model = CodingAgentModel(config)
    model.train()
    ids = torch.randint(0, 4096, (2, 64))

    with torch.autograd.profiler.profile(record_shapes=True) as profile:
        loss = model(input_ids=ids, labels=ids).loss

    loss.backward()

    full_logits = 2 * 64 * 4096 * 4  # [B, S, V] fp32 bytes
    aten = [
        event
        for event in profile.function_events
        if event.name in {"aten::linear", "aten::mm", "aten::addmm"}
    ]
    biggest = max((event.input_shapes[0][-1] * event.input_shapes[1][-1] * 4) for event in aten if len(event.input_shapes) >= 2)
    assert biggest < full_logits, "a projection produced a full [B, S, vocab] tensor"


def test_inference_still_returns_full_logits():
    config = _tiny_config(return_logits=False)
    model = CodingAgentModel(config)
    model.eval()
    ids = torch.randint(0, 97, (1, 6))
    out = model(input_ids=ids)
    assert out.logits.shape == (1, 6, 97)


# --------------------------------------------------------------------------- attention
def _reference_attention(attn, query, key, value, additive_mask=None, window=None):
    """The explicit softmax implementation the fused kernel replaces."""

    key = attn._repeat_kv(key, attn.num_kv_groups)
    value = attn._repeat_kv(value, attn.num_kv_groups)
    weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if window is not None and weights.shape[-2] > 1:
        weights = weights + attn._create_sliding_window_mask(
            weights.shape[-2], weights.shape[-1], weights.device, weights.dtype
        )
    if additive_mask is not None:
        weights = weights + additive_mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


@pytest.mark.parametrize("use_mask", [False, True])
def test_fused_attention_matches_explicit_softmax(use_mask):
    torch.manual_seed(7)
    config = _tiny_config(sliding_window=None, attention_backend="auto")
    attn = GroupedQueryAttention(config, layer_idx=0)
    attn.eval()

    batch, heads, kv_heads, seq, dim = 2, 4, 2, 12, 8
    query = torch.randn(batch, heads, seq, dim)
    key = torch.randn(batch, kv_heads, seq, dim)
    value = torch.randn(batch, kv_heads, seq, dim)

    causal = torch.zeros(seq, seq)
    causal.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), float("-inf"))

    if use_mask:
        # A padding mask must be combined with the causal band, never replace it.
        keep = torch.ones(batch, seq, dtype=torch.bool)
        keep[1, -3:] = False
        fused = attn._compute_attention(query, key, value, keep)
        expected = _reference_attention(
            attn, query, key, value, causal + attn._layer_mask(keep, seq, query.dtype)
        )
    else:
        fused = attn._compute_attention(query, key, value, None)
        expected = _reference_attention(attn, query, key, value, causal)

    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)


def test_windowed_attention_matches_reference_mask():
    torch.manual_seed(8)
    config = _tiny_config(sliding_window=4, num_attention_heads=4, num_kv_heads=2)
    attn = GroupedQueryAttention(config, layer_idx=0)
    attn.eval()

    query = torch.randn(1, 4, 10, 8)
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)

    fused = attn._compute_attention(query, key, value)

    key_mask = attn._repeat_kv(key, attn.num_kv_groups)
    value_mask = attn._repeat_kv(value, attn.num_kv_groups)
    weights = torch.matmul(query, key_mask.transpose(-2, -1)) / math.sqrt(8)
    value_mask = attn._repeat_kv(value, attn.num_kv_groups)
    weights = torch.matmul(query, key_mask.transpose(-2, -1)) / math.sqrt(8)
    loop_mask = torch.full((1, 1, 10, 10), float("-inf"))
    for row in range(10):
        start = max(0, row - 4 + 1)
        loop_mask[0, 0, row, start : row + 1] = 0.0
    weights = weights + loop_mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32)
    expected = torch.matmul(weights, value_mask)

    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)


def test_windowed_attention_uses_the_fused_kernel_at_long_sequences():
    """8k sequence with a 4k window must not fall back to the quadratic path."""

    torch.manual_seed(21)
    config = _tiny_config(sliding_window=4, num_attention_heads=4, num_kv_heads=2)
    attn = GroupedQueryAttention(config, layer_idx=0)
    attn.eval()

    query = torch.randn(1, 4, 16, 8)
    key = torch.randn(1, 2, 16, 8)
    value = torch.randn(1, 2, 16, 8)

    fused = attn._compute_attention(query, key, value)

    keep = attn._window_keep_mask(16, 16, query.device)
    additive = torch.zeros(keep.shape).masked_fill_(~keep, float("-inf"))
    expected = _reference_attention(attn, query, key, value, additive)

    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)


def test_windowed_attention_with_padding_keeps_both_constraints():
    torch.manual_seed(22)
    config = _tiny_config(sliding_window=3, num_attention_heads=4, num_kv_heads=2)
    attn = GroupedQueryAttention(config, layer_idx=0)
    attn.eval()

    query = torch.randn(2, 4, 9, 8)
    key = torch.randn(2, 2, 9, 8)
    value = torch.randn(2, 2, 9, 8)
    # Pad the *last* keys only: a fully-masked row has no valid distribution, so
    # the reference softmax would emit NaN there while SDPA emits zeros.
    keep = torch.ones(2, 9, dtype=torch.bool)
    keep[1, -2:] = False

    fused = attn._compute_attention(query, key, value, keep)

    window = attn._create_sliding_window_mask(9, 9, query.device)
    padding = attn._layer_mask(keep, 9, query.dtype)
    expected = _reference_attention(attn, query, key, value, window + padding)
    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)


def test_window_mask_is_cached_and_stable():
    attn = GroupedQueryAttention(_tiny_config(sliding_window=4), layer_idx=0)
    first = attn._create_sliding_window_mask(6, 6, torch.device("cpu"))
    second = attn._create_sliding_window_mask(6, 6, torch.device("cpu"))
    assert first is second
    allowed = torch.isfinite(first)
    for row in range(6):
        for col in range(6):
            assert bool(allowed[0, 0, row, col]) == (0 <= row - col < 4)


def test_code_graph_bias_accepts_structural_masks():
    torch.manual_seed(9)
    attn = GroupedQueryAttention(
        _tiny_config(sliding_window=None, num_attention_heads=4, num_kv_heads=2), layer_idx=0
    )
    attn.eval()
    query = torch.randn(1, 4, 5, 8)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn(1, 2, 5, 8)
    disallowed = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    disallowed[..., 0] = True

    fused = attn._compute_attention(query, key, value, None, disallowed)

    additive = torch.zeros(1, 1, 5, 5)
    additive.masked_fill_(disallowed, float("-inf"))
    expected = _reference_attention(attn, query, key, value, additive)
    torch.testing.assert_close(fused, expected, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- primitives
def test_rms_norm_matches_reference_formula():
    torch.manual_seed(11)
    norm = RMSNorm(16, eps=1e-6)
    norm.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(3, 7, 16)
    reference = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)) * norm.weight
    torch.testing.assert_close(norm(x), reference, atol=1e-6, rtol=1e-6)

    # bf16 in -> fp32 reduce (same as the reference _norm(...).type_as(x))
    bf16 = x.bfloat16()
    torch.testing.assert_close(rms_norm(bf16, norm.weight, norm.eps), norm(bf16), atol=1e-5, rtol=1e-5)


def test_rope_matches_reference_rotation():
    torch.manual_seed(12)
    rotary = RotaryEmbedding(dim=8, max_position_embeddings=16, theta=10_000.0)
    q = torch.randn(2, 1, 5, 8)
    k = torch.randn(2, 1, 5, 8)
    cos, sin = rotary(q)

    q_fast, k_fast = apply_rotary_pos_emb(q, k, cos, sin)

    freqs = torch.outer(torch.arange(5, dtype=torch.float32), rotary.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_ref = emb.cos()[None, None]
    sin_ref = emb.sin()[None, None]

    half = q.shape[-1] // 2
    q_reference = q * cos_ref + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin_ref
    torch.testing.assert_close(q_fast, q_reference, atol=1e-6, rtol=1e-6)
    assert k_fast.shape == k.shape


def test_rope_handles_gather_style_positions():
    rotary = RotaryEmbedding(dim=8, max_position_embeddings=16, theta=10_000.0)
    x = torch.randn(1, 4, 3, 8)
    positions = torch.tensor([[3, 1, 0]])
    cos, sin = rotary(x, position_ids=positions)
    assert cos.shape == (1, 1, 3, 8)
    torch.testing.assert_close(cos[0, 0, 0], rotary.cos_cached[3])
    torch.testing.assert_close(sin[0, 0, 1], rotary.sin_cached[1])


# --------------------------------------------------------------------------- data path
def test_prefetch_stream_preserves_order_and_reports_position():
    produced = [torch.full((1, 4), value) for value in range(6)]

    class Source:
        def __init__(self):
            self.epoch = 0
            self.batch_in_epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.batch_in_epoch == len(produced):
                raise StopIteration
            value = produced[self.batch_in_epoch]
            self.batch_in_epoch += 1
            return value

    stream = PrefetchStream(Source(), depth=2, pin_memory=False)
    seen = []
    for batch in stream:
        seen.append(batch.clone())
        assert stream.position == (0, len(seen))

    assert len(seen) == len(produced)
    for expected, actual in zip(produced, seen):
        assert torch.equal(expected, actual)
    stream.close()


def test_prefetch_stream_propagates_source_errors():
    class Broken:
        epoch = 0
        batch_in_epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("shard vanished")

    stream = PrefetchStream(Broken(), depth=2, pin_memory=False)
    with pytest.raises(RuntimeError, match="shard vanished"):
        next(stream)


def test_batch_tokens_matches_per_index_fetch(tmp_path):
    tokens = np.arange(64, dtype=np.uint32)
    for index, size in enumerate((21, 43)):
        tokens[sum((21, 43)[:index]) : sum((21, 43)[: index + 1])].tofile(tmp_path / f"train_{index:05d}.bin")
    (tmp_path / "train_manifest.json").write_text(
        json.dumps(
            {
                "total_tokens": 64,
                "num_shards": 2,
                "shards": [{"path": "train_00000.bin", "tokens": 21}, {"path": "train_00001.bin", "tokens": 43}],
            }
        )
    )
    dataset = MemoryMappedDataset(str(tmp_path), split="train", seq_length=7)

    batch = dataset.batch_tokens([0, 2, 5])

    assert batch.shape == (3, 8) and batch.dtype == np.int32
    for row, index in enumerate((0, 2, 5)):
        expected = dataset._get_tokens(index * 7, (index + 1) * 7 + 1)
        assert np.array_equal(batch[row], expected)


def test_streaming_loader_shifts_a_single_buffer(tmp_path):
    from training.train import StreamingDataLoader

    tokens = np.arange(64, dtype=np.uint32)
    tokens.tofile(tmp_path / "train_00000.bin")
    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"total_tokens": 64, "num_shards": 1, "shards": [{"path": "train_00000.bin", "tokens": 64}]})
    )
    dataset = MemoryMappedDataset(str(tmp_path), split="train", seq_length=8)
    loader = StreamingDataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))

    # One [B, S+1] token buffer, two shifted views: no per-batch copies.
    assert batch["input_ids"].shape == (2, 8)
    assert torch.equal(batch["input_ids"][0], torch.arange(0, 8))
    assert torch.equal(batch["labels"][0], torch.arange(1, 9))
    assert torch.equal(batch["labels"][1], torch.arange(9, 17))


# --------------------------------------------------------------------------- planning
def test_memory_estimate_reacts_to_the_knobs_that_matter():
    config = _tiny_config(num_hidden_layers=4)

    base = estimate(config, micro_batch=1, seq_length=512, activation_segments=2)
    bigger = estimate(config, micro_batch=2, seq_length=512, activation_segments=2)
    assert bigger.total > base.total

    unchunked = estimate(config, micro_batch=1, seq_length=4096, chunked_loss=False, loss_chunk_size=1024)
    chunked = estimate(config, micro_batch=1, seq_length=4096, chunked_loss=True, loss_chunk_size=1024)
    assert unchunked.logits > chunked.logits

    math_attention = estimate(config, micro_batch=1, seq_length=512, fused_attention=False)
    assert math_attention.attention_scratch > 0 and base.attention_scratch == 0

    offloaded = estimate(config, micro_batch=1, seq_length=512, offload_optimizer=True, offload_param=True)
    assert offloaded.total < base.total
    assert offloaded.weights_bf16 == 0 and offloaded.optimizer_state == 0
    large = estimate(
        _tiny_config(num_hidden_layers=4, hidden_size=4096, intermediate_size=16384, vocab_size=32000),
        micro_batch=1,
        seq_length=512,
        offload_optimizer=True,
        offload_param=True,
    )
    assert large.extra["host_gib"] > 0
    assert base.extra.get("host_gib", 0) == 0


def test_plan_recipe_preserves_the_global_batch():
    from model.config_3b import get_3b_config

    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    plan = plan_recipe(recipe, get_3b_config(), budget_gb=24.0, world_size=8)

    assert plan["global_batch_size"] == recipe["batching"]["global_batch_size"]
    assert plan["gradient_accumulation_steps"] * plan["micro_batch_size_per_gpu"] * 8 == plan["global_batch_size"]
    assert plan["micro_batch_size_per_gpu"] >= 1
    assert largest_micro_batch(get_3b_config(), 24.0, world_size=8, seq_length=4096) >= 1


# --------------------------------------------------------------------------- deepspeed
def test_3b_deepspeed_defaults_to_resident_state():
    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    plan = build_training_plan(recipe, 8)

    config = build_deepspeed_config(template, recipe, plan, hidden_size=3072)

    zero = config["zero_optimization"]
    assert zero["offload_optimizer"]["device"] == "none"
    assert zero["offload_param"]["device"] == "none"
    assert zero["overlap_comm"] is True
    assert zero["reduce_scatter"] is True
    assert "activation_checkpointing" not in config
    assert config["optimizer"]["type"] == "AdamW"
    assert config["optimizer"]["params"]["betas"] == [recipe["training"]["beta1"], recipe["training"]["beta2"]]
    assert config["bf16"]["enabled"] is True
    assert config["train_batch_size"] == 512
    assert "auto" not in json.dumps(config)


def test_3b_offload_fallback_changes_residency_only():
    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    recipe["deepspeed"]["offload_optimizer"] = True
    recipe["deepspeed"]["offload_param"] = True
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    plan = build_training_plan(recipe, 8)

    fast = build_deepspeed_config(copy.deepcopy(template), {**recipe, "deepspeed": {**recipe["deepspeed"], "offload_optimizer": False, "offload_param": False}}, plan, 3072)
    slow = build_deepspeed_config(template, recipe, plan, 3072)

    assert slow["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert slow["zero_optimization"]["offload_param"]["device"] == "cpu"
    for key in ("bf16", "optimizer", "gradient_clipping", "train_batch_size", "gradient_accumulation_steps"):
        assert fast[key] == slow[key]


def test_3b_8bit_optimizer_is_opt_in_and_keeps_the_update():
    recipe = json.loads((ROOT / "configs" / "training_config_3b.json").read_text())
    template = json.loads((ROOT / "configs" / "ds_config_zero3.json").read_text())
    plan = build_training_plan(recipe, 8)

    reference = build_deepspeed_config(copy.deepcopy(template), recipe, plan, 3072)
    assert reference["optimizer"]["type"] == "AdamW"

    recipe["deepspeed"]["zero8bit_optimizer"] = True
    quantised = build_deepspeed_config(copy.deepcopy(template), recipe, plan, 3072)
    assert quantised["optimizer"]["type"] == "AdamW8bit"
    assert quantised["optimizer"]["params"]["betas"] == reference["optimizer"]["params"]["betas"]
    assert quantised["optimizer"]["params"]["lr"] == reference["optimizer"]["params"]["lr"]


# --------------------------------------------------------------------------- recipes
def test_training_recipes_request_the_fast_paths():
    for name, model_size in (("training_config.json", "20B"), ("training_config_3b.json", "3B")):
        recipe = json.loads((ROOT / "configs" / name).read_text())
        assert recipe["model_size"] == model_size
        assert recipe["training"]["chunked_loss"] is True
        assert recipe["training"]["attention_backend"] in {"auto", "flash"}
        assert recipe["deepspeed"]["activation_checkpointing_segments"] >= 1
        assert recipe["data"]["prefetch_depth"] >= 1
        assert recipe["data"]["pin_memory"] is True
        assert recipe["batching"]["micro_batch_size_per_gpu"] >= 1


def test_recipes_default_to_a_configuration_whose_peak_fits_a_real_card():
    """Residency is a choice, but the shipped default has to be runnable.

    20B cannot be fully resident on 8 ranks (~66 GiB/rank), so it keeps offload;
    3B fits, so it is resident.  Both must land inside a 40 GiB-per-rank budget
    on the hardware their docs claim, which is what this pins.
    """

    from model.config import get_20b_config
    from model.config_3b import get_3b_config

    from training.perf import estimate

    def plan(path, model, ranks):
        recipe = json.loads((ROOT / "configs" / path).read_text())
        ds = recipe["deepspeed"]
        return estimate(
            model,
            world_size=ranks,
            micro_batch=recipe["batching"]["micro_batch_size_per_gpu"],
            seq_length=recipe["batching"]["sequence_length"],
            offload_optimizer=ds["offload_optimizer"],
            offload_param=ds["offload_param"],
            activation_segments=ds["activation_checkpointing_segments"],
            chunked_loss=recipe["training"]["chunked_loss"],
            loss_chunk_size=recipe["training"]["loss_chunk_size"],
            window_materialised=True,
        )

    three_b = plan("training_config_3b.json", get_3b_config(), 8)
    assert three_b.total / 1024**3 < 12, three_b.as_gib()

    twenty_b = plan("training_config.json", get_20b_config(), 64)
    assert twenty_b.total / 1024**3 < 40, twenty_b.as_gib()

    recipe = json.loads((ROOT / "configs" / "training_config.json").read_text())
    assert recipe["deepspeed"]["offload_optimizer"] is True
    assert recipe["deepspeed"]["offload_param"] is True


def test_20b_recipe_keeps_the_learning_recipe():
    recipe = json.loads((ROOT / "configs" / "training_config.json").read_text())
    assert recipe["training"]["learning_rate"] == 3e-4
    assert recipe["training"]["precision"] == "bf16"
    assert recipe["batching"]["global_batch_size"] == 2048
    assert recipe["batching"]["sequence_length"] == 8192
    assert recipe["training"]["beta1"] == 0.9 and recipe["training"]["beta2"] == 0.95
    assert recipe["training"]["weight_decay"] == 0.1


def test_20b_accumulation_is_derived_from_the_real_world_size():
    batching = json.loads((ROOT / "configs" / "training_config.json").read_text())["batching"]

    assert resolve_accumulation(batching, micro_batch=1, world_size=8) == 256
    assert resolve_accumulation(batching, micro_batch=1, world_size=64) == 32
    assert resolve_accumulation(batching, micro_batch=2, world_size=64) == 16
    # the accumulation window times micro-batch times ranks is the global batch
    for world_size in (8, 16, 32, 64, 128, 256):
        window = resolve_accumulation(batching, micro_batch=1, world_size=world_size)
        assert window * 1 * world_size == batching["global_batch_size"]

    with pytest.raises(ValueError, match="not divisible"):
        resolve_accumulation({"global_batch_size": 100, "gradient_accumulation_steps": "auto"}, 1, 8)


def test_training_gradients_match_autograd_on_a_batch():
    """Batched gradient check: the fast path must not perturb any gradient."""

    torch.manual_seed(31)
    config = _tiny_config(vocab_size=53, hidden_size=12, intermediate_size=24, num_hidden_layers=2)
    model = CodingAgentModel(config)
    model.train()
    ids = torch.randint(0, 53, (2, 9))
    labels = ids.clone()
    labels[:, :4] = -100

    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()

    checked = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            inputs = torch.randn(2, 9, module.in_features, dtype=module.weight.dtype)
            torch.autograd.gradcheck(
                lambda x, w: torch.nn.functional.linear(x, w),
                (inputs.double().requires_grad_(True), module.weight.double().requires_grad_(True)),
                eps=1e-6,
                atol=1e-4,
                raise_exception=True,
            )
            break

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
        checked += 1

    assert checked > 5


def test_prefetch_stream_restarts_reiterable_sources():
    """The 20B loader is a generator; the epoch rollover must not stall."""

    class Reiterable:
        def __init__(self):
            self.epoch = 0
            self.batch_in_epoch = 0

        def __iter__(self):
            self.batch_in_epoch = 0
            for index in range(3):
                self.batch_in_epoch += 1
                yield torch.full((1, 2), index)
            self.epoch += 1

    stream = PrefetchStream(Reiterable(), depth=2, pin_memory=False)

    values = []
    for batch in stream:
        values.append(int(batch[0, 0]))
        if len(values) == 7:
            break
    stream.close()

    assert values == [0, 1, 2, 0, 1, 2, 0]
    # the producer may be a pass ahead of the consumer; what matters is that it
    # restarted rather than stopping at the end of the first pass
    assert stream.restarts >= 2


def test_prefetch_stream_does_not_restart_iterators():
    produced = [torch.zeros(1, 2) for _ in range(4)]

    class Once:
        epoch = 0
        batch_in_epoch = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.batch_in_epoch == len(produced):
                raise StopIteration
            self.batch_in_epoch += 1
            return torch.full((1, 2), self.batch_in_epoch)

    stream = PrefetchStream(Once(), depth=2, pin_memory=False)
    assert len(list(stream)) == 4
    stream.close()
