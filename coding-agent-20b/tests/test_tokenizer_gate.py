import json
import os

import pytest

from model.tokenizer import (
    SPECIAL_TOKENS,
    CodeTokenizer,
    EfficientBPETrainer,
    TokenizerValidator,
)


def _corpus():
    from evaluation.tokenizer_bench import SAMPLE_CORPUS

    docs = [d for v in SAMPLE_CORPUS.values() for d in v]

    return docs, docs * 30


def test_vocab_is_padded_to_exact_size_and_ids_are_contiguous():

    _, corpus = _corpus()

    tok = EfficientBPETrainer(vocab_size=1000, min_frequency=1).train_from_iterator(
        iter(corpus), verbose=False
    )

    assert len(tok) == 1000

    assert len(tok.id_to_token) == 1000

    assert sorted(tok.id_to_token) == list(range(1000))

    # Padding is reserved specials, not silently missing ids.
    reserved = [t for t in tok.token_to_id if t.startswith("<|reserved_")]

    assert reserved

    # Round-trip and byte fallback survive padding.
    text = "def f(x: int) -> int:\n\treturn x + 1  # ✓\n"

    ids = tok.encode(text, add_special_tokens=False)

    assert tok.decode(ids) == text

    assert SPECIAL_TOKENS["<unk>"] not in ids


def test_capability_special_tokens_exist_and_are_stable():

    for name in (
        "<|scratchpad|>",
        "<|plan|>",
        "<|repo_map|>",
        "<|retrieval|>",
        "<|diff|>",
        "<|tool_error|>",
        "<|clarify|>",
    ):

        assert name in SPECIAL_TOKENS

    # Existing ids must not shift (checkpoints depend on them).
    assert SPECIAL_TOKENS["<|tool_call|>"] == 48

    assert SPECIAL_TOKENS["<|tool_result|>"] == 49


def test_save_load_roundtrip_and_integrity(tmp_path):

    _, corpus = _corpus()

    tok = EfficientBPETrainer(vocab_size=700, min_frequency=1).train_from_iterator(
        iter(corpus), verbose=False
    )

    tok.save(str(tmp_path))

    loaded = CodeTokenizer.load(str(tmp_path))

    assert len(loaded) == len(tok)

    assert loaded.encode("def g(): pass", add_special_tokens=False) == tok.encode(
        "def g(): pass", add_special_tokens=False
    )

    data = json.loads((tmp_path / "tokenizer.json").read_text())

    data["vocab_size"] = 123

    (tmp_path / "tokenizer.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="vocab_size mismatch"):

        CodeTokenizer.load(str(tmp_path))


def test_validator_reports_fertility_identifiers_and_diff():

    _, corpus = _corpus()

    from evaluation.tokenizer_bench import SAMPLE_CORPUS

    docs = [d for v in SAMPLE_CORPUS.values() for d in v]

    tok = EfficientBPETrainer(vocab_size=2000, min_frequency=1).train_from_iterator(
        iter(corpus), verbose=False
    )

    report = TokenizerValidator.validate(tok, docs)

    for key in ("per_language", "identifiers", "whitespace", "diff", "vocab_size"):

        assert key in report

    assert report["roundtrip_ok"]

    assert report["whitespace"]["tabs_roundtrip"]

    assert report["whitespace"]["blank_line_roundtrip"]

    assert report["identifiers"]["whole_identifier_rate"] > 0.5

    assert report["diff"]["chars_per_token"] > 1.0

    # Report must survive JSON (it is written beside the vocab).
    json.dumps({k: v for k, v in report.items() if k != "per_sample"})


def test_benchmark_gate_rejects_untrained_and_accepts_trained():

    from evaluation.tokenizer_bench import TokenizerBenchmark

    bench = TokenizerBenchmark()

    untrained = bench.run(CodeTokenizer())

    assert not untrained["gate"]["passed"]

    with pytest.raises(AssertionError):

        TokenizerBenchmark.assert_competitive(untrained)

    _, corpus = _corpus()

    trained = EfficientBPETrainer(vocab_size=2000, min_frequency=1).train_from_iterator(
        iter(corpus), verbose=False
    )

    report = bench.run(trained)

    assert report["tokenizer"] == "candidate"

    assert report["gate"]["wins"] == report["gate"]["corpora"]

    TokenizerBenchmark.assert_competitive(report)


def test_trainer_script_streams_jsonl_shards_with_manifest(tmp_path, monkeypatch):

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_prod_tok", os.path.join(os.path.dirname(__file__), "..", "scripts", "train_production_tokenizer.py")
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    shard = tmp_path / "corpus.jsonl"

    lines = []

    for i in range(60):

        code = f"def handler_{i}(request: dict) -> dict:\n" + "\n".join(
            f"    value_{j} = request.get('key_{j}', {j})\n    if value_{j} is None:\n        raise ValueError('missing key_{j}')"
            for j in range(6)
        ) + "\n    return {'ok': True}\n"

        lines.append(json.dumps({"language": "python", "content": code}))

    shard.write_text("\n".join(lines))

    pairs = list(module.iter_corpus(str(tmp_path), per_language_cap=100))

    assert len(pairs) == 60

    # Duplicate content is dropped by the content-hash dedupe.
    shard.write_text("\n".join(lines + lines[:10]))

    pairs = list(module.iter_corpus(str(tmp_path), per_language_cap=100))

    assert len(pairs) == 60


def test_tokenizer_trainer_requires_real_documents(tmp_path):

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_prod_tok2", os.path.join(os.path.dirname(__file__), "..", "scripts", "train_production_tokenizer.py")
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    trainer = module.ProductionTokenizerTrainer(vocab_size=400)

    with pytest.raises(RuntimeError, match="no usable documents"):

        trainer.train(str(tmp_path), str(tmp_path / "tok"))

    src = tmp_path / "src"

    src.mkdir()

    for i in range(20):

        (src / f"m{i}.py").write_text(
            "\n".join(
                [
                    "import os",
                    "",
                    f"def compute_{i}(values: list[int]) -> int:",
                    '    """Sum values with a guard."""',
                    "    if not values:",
                    "        return 0",
                    "    total = 0",
                    "    for value in values:",
                    "        total += value",
                    "    return total",
                    "",
                ]
            )
        )

    tok = trainer.train(str(src), str(tmp_path / "tok"))

    assert len(tok) == 400

    manifest = json.loads((tmp_path / "tok" / "manifest.json").read_text())

    assert manifest["documents"] == 20

    assert manifest["languages"]["python"] == 20

    assert len(manifest["corpus_sha256"]) == 64
