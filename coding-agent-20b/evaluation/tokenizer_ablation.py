"""Pre-tokenizer ablation: code-aware vs generic GPT-2 pre-tokenization.

Trains two equal-vocabulary BPE tokenizers on the same corpus slice - one
under the repository's code-aware pre-tokenizer, one under a generic GPT-2
pattern - and measures what code-awareness actually buys on held-out code:

- **fertility**: tokens per character (compression at equal vocab),
- **atomicity**: average tokens needed to re-encode a whole string literal,
  comment, or identifier extracted from held-out files (1.0 = the unit
  survives as one token).

The gate: the code-aware pre-tokenizer must win atomicity on strings and
comments (the units code models care about) at equal vocabulary, or there
would be no reason to prefer it over the generic pattern.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.fast_bpe import GPT2_SPLIT_PATTERN, FastBPETrainer
from scripts.train_production_tokenizer import iter_corpus

CORPUS = "/home/user/corpus/tokenizer_corpus"
HELDOUT = Path("/home/user/corpus/tokenizer_heldout")

VOCAB = 16384
TRAIN_CHARS = 3_000_000

STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')
COMMENT_RE = re.compile(r"//[^\n]*|#[^\n]*")
IDENT_RE = re.compile(r"\b[a-z_][a-z0-9_]{3,}\b")
NUMBER_RE = re.compile(r"\b0x[0-9a-fA-F]+\b|\b\d+\.\d+\b|\b\d{2,}\b")


def train_engine(pattern: str | None, docs: list[str]) -> FastBPETrainer:
    trainer = FastBPETrainer(
        vocab_size=VOCAB, min_frequency=2, split_pattern=pattern, pad_to_vocab_size=False
    )
    trainer.train(docs, verbose=False)
    return trainer


def tokens_per_unit(encoder, units: list[str]) -> float:
    if not units:
        return float("nan")
    total = sum(len(encoder(u)) for u in units)
    return total / len(units)


def fertility(encoder, docs: list[str]) -> float:
    chars = sum(len(d) for d in docs)
    toks = sum(len(encoder(d)) for d in docs)
    return toks / max(chars, 1)


def run(docs: list[str], heldout: list[str], quiet: bool = False) -> dict:
    code_trainer = train_engine(None, docs)
    gpt2_trainer = train_engine(GPT2_SPLIT_PATTERN, docs)

    code_enc = code_trainer.rust_tokenizer.encode
    gpt2_enc = gpt2_trainer.rust_tokenizer.encode

    strings, comments, idents, numbers = [], [], [], []
    for doc in heldout:
        strings.extend(STRING_RE.findall(doc)[:20])
        comments.extend(COMMENT_RE.findall(doc)[:20])
        idents.extend(IDENT_RE.findall(doc)[:40])
        numbers.extend(NUMBER_RE.findall(doc)[:20])
        if len(strings) > 4000:
            break

    report = {
        "vocab": VOCAB,
        "train_chars": sum(len(d) for d in docs),
        "heldout_docs": len(heldout),
        "units": {
            "strings": len(strings),
            "comments": len(comments),
            "identifiers": len(idents),
            "numbers": len(numbers),
        },
        "code_aware": {
            "fertility": fertility(code_enc, heldout),
            "tokens_per_string": tokens_per_unit(code_enc, strings),
            "tokens_per_comment": tokens_per_unit(code_enc, comments),
            "tokens_per_identifier": tokens_per_unit(code_enc, idents),
            "tokens_per_number": tokens_per_unit(code_enc, numbers),
        },
        "gpt2_generic": {
            "fertility": fertility(gpt2_enc, heldout),
            "tokens_per_string": tokens_per_unit(gpt2_enc, strings),
            "tokens_per_comment": tokens_per_unit(gpt2_enc, comments),
            "tokens_per_identifier": tokens_per_unit(gpt2_enc, idents),
            "tokens_per_number": tokens_per_unit(gpt2_enc, numbers),
        },
    }

    c, g = report["code_aware"], report["gpt2_generic"]
    FERTILITY_TOLERANCE = 1.25
    report["fertility_ratio"] = c["fertility"] / g["fertility"]
    report["gate"] = {
        "strings_win": c["tokens_per_string"] < g["tokens_per_string"],
        "comments_win": c["tokens_per_comment"] < g["tokens_per_comment"],
        "numbers_win": c["tokens_per_number"] <= g["tokens_per_number"],
        "identifiers_win": c["tokens_per_identifier"] <= g["tokens_per_identifier"],
        "fertility_within_bound": (
            c["fertility"] <= g["fertility"] * FERTILITY_TOLERANCE
        ),
        "fertility_tolerance": FERTILITY_TOLERANCE,
    }
    report["passed"] = (
        report["gate"]["strings_win"]
        and report["gate"]["comments_win"]
        and report["gate"]["numbers_win"]
        and report["gate"]["identifiers_win"]
        and report["gate"]["fertility_within_bound"]
    )

    if not quiet:
        print(f"pre-tokenizer ablation (vocab={VOCAB:,}, trained on {report['train_chars']:,} chars)")
        print(f"{'metric':<28}{'code-aware':>12}{'gpt2-generic':>14}")
        for key in ("fertility", "tokens_per_string", "tokens_per_comment",
                    "tokens_per_identifier", "tokens_per_number"):
            print(f"{key:<28}{c[key]:>12.3f}{g[key]:>14.3f}")
        print("gate:", report["gate"])
        print("PASSED" if report["passed"] else "FAILED")
    return report


def main():
    train_docs = []
    for _lang, doc in iter_corpus(CORPUS, max_chars=TRAIN_CHARS, per_language_cap=100_000):
        train_docs.append(doc)
    heldout = []
    for f in sorted(p for p in HELDOUT.rglob("*") if p.is_file()):
        try:
            doc = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        heldout.append(doc)
        if len(heldout) >= 400:
            break
    report = run(train_docs, heldout)
    out = Path(__file__).parent.parent / "tokenizer" / "pretok_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
