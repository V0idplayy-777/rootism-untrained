"""FastBPETrainer: Rust-engine training with exact CodeTokenizer parity."""

import pytest

from model.tokenizer import CodePreTokenizer, CodeTokenizer, SPECIAL_TOKENS

fast_bpe = pytest.importorskip("model.fast_bpe", reason="tokenizers not installed")


from model.fast_bpe import (  # noqa: E402
    FastBPETrainer,
    GPT2_BYTES_TO_UNICODE,
    REPO_BYTES_TO_CHARS,
    gpt2_token_to_repo,
    pre_tokenize_rust,
    verify_parity,
)

ADVERSARIAL = [
    "def f():\r\n    return 'a\\'b'  # comment\r\n",
    '"""docstring"""\nx = 0x1F + 1.5e-3\n',
    "--- a/f.py\n+++ b/f.py\n@@ -1,4 +1,6 @@\n-old\n+new\n",
    "const x = a?.b ?? c; let y = `t${x}`;\n",
    "a\u00a0b\u2028c\rd\n\n\n \t \n",
    "x = 10 // 3; y = 10 /* blk */ / 2\n",
    "λ = lambda Ω: Ω**2  # unicode\n",
    "package main\n\nimport (\n\t\"fmt\"\n)\n",
    "0b1010 + 0o777 + 0xDEAD\n",
    "f(a, b[1:2], {'k': v})\n",
]


def test_byte_alphabets_are_exact_bijections():
    assert len(GPT2_BYTES_TO_UNICODE) == 256
    assert len(set(GPT2_BYTES_TO_UNICODE.values())) == 256
    assert len(REPO_BYTES_TO_CHARS) == 256
    assert len(set(REPO_BYTES_TO_CHARS.values())) == 256
    for b in range(256):
        gpt2 = GPT2_BYTES_TO_UNICODE[b]
        repo = gpt2_token_to_repo(gpt2)
        assert REPO_BYTES_TO_CHARS[b] == repo


def test_pretokenizer_parity_on_adversarial_inputs():
    py = CodePreTokenizer()
    for sample in ADVERSARIAL:
        assert py.pre_tokenize(sample) == pre_tokenize_rust(sample), sample


def test_pretokenizer_parity_on_real_files():
    import glob
    from pathlib import Path

    root = Path(__file__).parent.parent
    py = CodePreTokenizer()
    files = sorted(glob.glob(str(root / "agent" / "*.py")))[:10]
    assert files
    for f in files:
        text = Path(f).read_text(encoding="utf-8")
        assert py.pre_tokenize(text) == pre_tokenize_rust(text), f


def test_train_produces_exact_vocab_and_roundtrips():
    corpus = [
        "def add(a, b):\n    return a + b\n",
        "def sub(a, b):\n    return a - b\n",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
        "const handler = (req, res) => res.json({ok: true});\n",
        "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,4 @@\n-a\n+b\n",
    ] * 30
    target = len(SPECIAL_TOKENS) + 256 + 150
    trainer = FastBPETrainer(vocab_size=target, min_frequency=2)
    tok = trainer.train(corpus, verbose=False)
    assert len(tok) == target
    assert len(tok.merges) == min(150, len(tok.merges))
    assert len(tok) == len(SPECIAL_TOKENS) + 256 + len(tok.merges) + (
        target - len(SPECIAL_TOKENS) - 256 - len(tok.merges)
    )
    for sample in ADVERSARIAL:
        ids = tok.encode(sample, add_special_tokens=False)
        assert tok.decode(ids) == sample
    report = verify_parity(trainer.rust_tokenizer, tok, corpus[:5] + ADVERSARIAL)
    assert report["parity"]


def test_train_pads_when_corpus_cannot_fill_vocab():
    corpus = ["def f():\n    return 1\n"]
    target = len(SPECIAL_TOKENS) + 256 + 500
    trainer = FastBPETrainer(vocab_size=target, min_frequency=2)
    tok = trainer.train(corpus, verbose=False)
    assert len(tok) == target
    tok2 = FastBPETrainer(vocab_size=target, min_frequency=2).train(corpus, verbose=False)
    assert tok.token_to_id == tok2.token_to_id


def test_save_and_load_preserves_behaviour(tmp_path):
    corpus = [
        "def add(a, b):\n    return a + b\n",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n",
    ] * 40
    target = len(SPECIAL_TOKENS) + 256 + 120
    tok = FastBPETrainer(vocab_size=target, min_frequency=2).train(corpus, verbose=False)
    tok.save(str(tmp_path))
    loaded = CodeTokenizer.load(str(tmp_path))
    sample = "class Point:\n    def __init__(self, x, y):\n        self.x = x\n"
    assert loaded.encode(sample, add_special_tokens=False) == tok.encode(
        sample, add_special_tokens=False
    )
    assert len(loaded) == target
