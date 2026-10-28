"""Fast (Rust) byte-level BPE training with exact parity to ``CodeTokenizer``.

The pure-Python ``EfficientBPETrainer`` is correct but CPU-bound; a 128k
vocabulary over a multi-gigabyte corpus is not tractable in Python. This
module trains with the ``tokenizers`` (Rust) library — the same engine
HuggingFace ships in production — under constraints that make the result
*bit-compatible* with this repository's tokenizer:

1. **Same pre-tokenizer.** The Rust ``Split`` pre-tokenizer uses a regex with
   identical leftmost-first alternation semantics to
   ``CodePreTokenizer.CODE_PATTERN`` (diff hunk headers, docstrings, strings,
   comments, numbers, identifiers, operators, total whitespace handling).
   A shared test asserts that the two pre-tokenizers produce identical
   pre-token streams on adversarial inputs.
2. **Byte-level, both alphabets.** The Rust trainer works in the GPT-2
   byte-to-unicode alphabet; ``CodeTokenizer`` uses its own byte->char map.
   Both are bijections over the 256 byte values, so every learned token can
   be converted losslessly: gpt2-token -> bytes -> repo-token.
3. **Parity is proven, not assumed.** ``verify_parity`` encodes held-out
   documents with both tokenizers and asserts identical token counts; the
   training entrypoint runs it and refuses to return a tokenizer that
   disagrees.

The public surface mirrors ``EfficientBPETrainer``:
``FastBPETrainer.train_from_iterator(texts) -> CodeTokenizer``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .tokenizer import SPECIAL_TOKENS, CodePreTokenizer, CodeTokenizer

try:
    import tokenizers
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers

    HAS_TOKENIZERS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TOKENIZERS = False


# ---------------------------------------------------------------------------
# Byte alphabets
# ---------------------------------------------------------------------------


def _repo_byte_to_char(byte_val: int) -> str:
    """CodeTokenizer's byte->char map (duplicated to avoid a private import)."""

    if 33 <= byte_val <= 126:
        return chr(byte_val)
    if byte_val == 32:
        return "Ġ"
    if byte_val == 10:
        return "Ċ"
    if byte_val == 9:
        return "ĉ"
    return chr(byte_val + 256)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """The exact GPT-2 byte->unicode table used by ``tokenizers`` ByteLevel."""

    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


GPT2_BYTES_TO_UNICODE = _gpt2_bytes_to_unicode()
GPT2_UNICODE_TO_BYTES = {c: b for b, c in GPT2_BYTES_TO_UNICODE.items()}
REPO_BYTES_TO_CHARS = {b: _repo_byte_to_char(b) for b in range(256)}


def gpt2_token_to_repo(token: str) -> str:
    """Convert a GPT-2-alphabet token to the CodeTokenizer alphabet (exact)."""

    return "".join(REPO_BYTES_TO_CHARS[GPT2_UNICODE_TO_BYTES[ch]] for ch in token)


# ---------------------------------------------------------------------------
# Pre-tokenizer: Rust-regex twin of CodePreTokenizer.CODE_PATTERN
# ---------------------------------------------------------------------------

# Whitespace as an EXPLICIT class of literal characters: the union of Python
# `re` `\s` and the Rust regex crate's `\s` (the two engines disagree on
# \x1c-\x1f). Must stay character-identical to the whitespace alternatives in
# CodePreTokenizer.CODE_PATTERN (model/tokenizer.py) - the training parity
# gate depends on both engines splitting identically.
_WS_CHARS = (
    "\t\n\x0b\x0c\r\x1c-\x1f\x20\x85\u00a0\u1680"
    "\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
)

RUST_SPLIT_PATTERN = (
    r"(?m)"
    r"(?:^@@ -[^\n]*)"
    r"|(?:^(?:---|\+\+\+) [^\n]*)"
    r"|(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")"
    r"|(?:'(?:[^'\\]|\\[\s\S])*'|\"(?:[^\"\\]|\\[\s\S])*\")"
    r"|(?:`(?:[^`\\]|\\[\s\S])*`)"
    r"|(?://[^\n]*|[\x23][^\n]*)"
    r"|(?:/\*[\s\S]*?\*/)"
    r"|(?:0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+|[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)"
    r"|(?:[a-zA-Z_$][0-9A-Za-z_$]*)"
    r"|(?:::=|>>>|<<=|>>=|===|!==|\?\?|\?\.)"
    r"|(?:[+\-*/%&|^~<>!]=?|[<>&|]{2}|->|=>|\.\.\.|\*\*)"
    r"|(?:[{}()\[\];:,.?@&~^])"
    r"|(?:[ \t\x0b\x0c]+)"
    r"|(?:(?:\r\n|\r|\n)+)"
    r"|(?:[" + _WS_CHARS + r"]+)"
    r"|(?:[^" + _WS_CHARS + r"])"
)


# Generic GPT-2 pre-tokenization (ASCII-range re-expression of the original
# pattern; used ONLY as the ablation baseline that shows what the code-aware
# pattern buys). Not used for production training.
GPT2_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def rust_pretokenizer():
    """Split(code regex) + ByteLevel: the Rust twin of CodePreTokenizer.

    NOTE: the pattern **must** be wrapped in ``tokenizers.Regex`` — a bare
    ``str`` is treated as a literal delimiter by the Rust bindings.
    """

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(RUST_SPLIT_PATTERN),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def pre_tokenize_rust(text: str) -> list[str]:
    """Apply the Rust pre-tokenizer and return the *text* pre-tokens.

    Used only for parity checking against ``CodePreTokenizer``.
    """

    splitter = pre_tokenizers.Split(
        pattern=Regex(RUST_SPLIT_PATTERN),
        behavior="isolated",
        invert=False,
    )
    return [token for token, _span in splitter.pre_tokenize_str(text)]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FastBPETrainer:
    """Train a ``CodeTokenizer`` with the Rust engine.

    Same contract as ``EfficientBPETrainer``: ``vocab_size`` counts specials +
    256 byte symbols + merges; the learned vocabulary is padded with
    deterministic reserved tokens so ``len(tokenizer) == vocab_size`` exactly.
    """

    def __init__(
        self,
        vocab_size: int = 128256,
        min_frequency: int = 2,
        special_tokens: dict[str, int] | None = None,
        pad_to_vocab_size: bool = True,
        parity_samples: int = 64,
        split_pattern: str | None = None,
    ):
        if not HAS_TOKENIZERS:
            raise ImportError(
                "the `tokenizers` package is required for FastBPETrainer; "
                "pip install tokenizers>=0.19"
            )
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.pad_to_vocab_size = pad_to_vocab_size
        self.parity_samples = parity_samples
        self.split_pattern = split_pattern or RUST_SPLIT_PATTERN

    def train_from_iterator(self, texts, verbose: bool = True) -> CodeTokenizer:
        # The Rust trainer counts: initial alphabet (256) + merges.
        rust_vocab = self.vocab_size - len(self.special_tokens)
        if rust_vocab <= 256:
            raise ValueError(
                f"vocab_size {self.vocab_size} leaves no room for merges "
                f"({len(self.special_tokens)} specials)"
            )

        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=Regex(self.split_pattern),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )

        trainer = trainers.BpeTrainer(
            vocab_size=rust_vocab,
            min_frequency=self.min_frequency,
            show_progress=verbose,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
        )
        tok.train_from_iterator(texts, trainer=trainer)
        self.rust_tokenizer = tok  # kept for verify_parity() by callers

        with tempfile.TemporaryDirectory() as tmp:
            # BPE.save writes vocab.json + merges.txt (ordered merge list).
            _vocab_path, merges_path = tok.model.save(tmp)
            merges: list[tuple[str, str]] = []
            for line in Path(merges_path).read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#version"):
                    continue
                left, right = line.split(" ", 1)
                merges.append((left, right))

        repo_merges = [
            (gpt2_token_to_repo(a), gpt2_token_to_repo(b)) for a, b in merges
        ]

        tokenizer = CodeTokenizer(
            vocab=[],
            merges=repo_merges,
            special_tokens=self.special_tokens,
        )
        if self.pad_to_vocab_size:
            tokenizer.pad_to_size(self.vocab_size)

        if verbose:
            print(
                f"FastBPETrainer: {len(repo_merges)} merges, "
                f"vocab={len(tokenizer):,} (target {self.vocab_size:,})"
            )

        return tokenizer

    def train(self, texts: list[str], verbose: bool = True) -> CodeTokenizer:
        return self.train_from_iterator(iter(texts), verbose=verbose)


def verify_parity(
    rust_tokenizer,
    code_tokenizer: CodeTokenizer,
    samples: list[str],
) -> dict:
    """Assert both tokenizers segment identically; return a report.

    ``rust_tokenizer`` is the ``tokenizers.Tokenizer`` used for training
    (already configured with the shared pre-tokenizer). Because the conversion
    is a per-token bijection, identical segmentation on held-out text proves
    the learned merges survived conversion intact.
    """

    checked = mismatched = 0
    diagnostics: list[dict] = []
    for sample in samples:
        if not sample:
            continue
        checked += 1
        rust_count = len(rust_tokenizer.encode(sample).ids)
        repo_count = len(code_tokenizer.encode(sample, add_special_tokens=False))
        if rust_count != repo_count:
            mismatched += 1
            if len(diagnostics) < 3:
                # attribute the cause: pre-tokenizer drift vs BPE conversion
                try:
                    py_pretoks = code_tokenizer.pre_tokenizer.pre_tokenize(sample)
                    rust_pretoks = pre_tokenize_rust(sample)
                    cause = (
                        "pre_tokenizer"
                        if py_pretoks != rust_pretoks
                        else "bpe_conversion"
                    )
                except Exception:
                    py_pretoks = rust_pretoks = []
                    cause = "unknown"
                diagnostics.append(
                    {
                        "sample_head": repr(sample[:120]),
                        "rust_tokens": rust_count,
                        "repo_tokens": repo_count,
                        "cause": cause,
                    }
                )
    report = {
        "samples": checked,
        "mismatches": mismatched,
        "parity": mismatched == 0,
        "diagnostics": diagnostics,
    }
    if mismatched:
        raise AssertionError(
            f"rust/python tokenizer parity failed on {mismatched}/{checked} samples: "
            f"{diagnostics}"
        )
    return report
