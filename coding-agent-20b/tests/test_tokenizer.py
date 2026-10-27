from model.tokenizer import CodeTokenizer, EfficientBPETrainer, TokenizerValidator

SAMPLES = [
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n",
    "héllo wörld — unicode ✓ def f(): pass\n",
    "class Foo:\n    def bar(self, x=1):\n        return [i*i for i in range(x)]\n",
]


def test_roundtrip_byte_fallback(tokenizer):
    for s in SAMPLES:
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert tokenizer.decode(ids, skip_special_tokens=True) == s
        assert tokenizer.special_tokens["<unk>"] not in ids


def test_efficient_trainer_learns_merges_and_validates():
    corpus = SAMPLES * 40 + ["print('hello world')\n" * 5, "import os\nimport sys\n" * 5]
    tok = EfficientBPETrainer(vocab_size=600, min_frequency=2).train_from_iterator(iter(corpus), verbose=False)
    report = TokenizerValidator.validate(tok, SAMPLES)
    assert report["roundtrip_ok"] and report["unk_rate"] == 0.0
    assert len(tok.merges) > 10
    # Trained merges compress repeated code better than raw bytes.
    ids = tok.encode(SAMPLES[0], add_special_tokens=False)
    assert len(ids) < len(SAMPLES[0].encode("utf-8"))


def test_trained_vocab_still_byte_fallback():
    tok = EfficientBPETrainer(vocab_size=400, min_frequency=2).train(["def a():\n    pass\n" * 30], verbose=False)
    ids = tok.encode("Ωμέγα ✓ unseen text", add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=True) == "Ωμέγα ✓ unseen text"
