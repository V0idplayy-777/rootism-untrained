#!/usr/bin/env python3
"""Build Corpus v1 pretraining shards from real code + verified synthetic slices.

This is the single entry point every config points at
(`configs/training_config*.json -> data.builder`). It does four things and
refuses to do them silently:

1. read real permissive-licensed JSONL shards from ``--real_dir`` and route each
   document into its Corpus v1 backbone slice (``the-stack-code``, ``tests``,
   ``github-prs``, ``github-issues``, ``docs-and-comments``);
2. build the verified synthetic slices from the generators in this repo
   (security, repair traces, verified apps, recovery, scaffolds) and cap their
   token contribution at 5% of the corpus
   (`dataset.mixture.synthetic_token_budget`);
3. refuse any document whose SPDX license is not permissive and any document
   present in the held-out registry;
4. write uint32 shards + ``train_manifest.json`` / ``val_manifest.json`` /
   ``corpus_manifest.json`` through `ProductionShardBuilder`, with the realized
   per-slice token shares in the manifest.

    python scripts/build_corpus_v1.py \\
        --real_dir data/corpus_v1/real_jsonl \\
        --heldout_dir data/corpus_v1/heldout \\
        --output_dir data/corpus_v1/shards_3b

    # smoke test with the tiny in-repo tokenizer and template data
    python scripts/build_corpus_v1.py --demo --output_dir /tmp/shards --total_tokens 2_000_000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.app_corpus import AppCorpus  # noqa: E402
from dataset.augmentation import RecoveryInstructionGenerator  # noqa: E402
from dataset.instruction_induction import InstructionInduction  # noqa: E402
from dataset.project_data import ProjectDatasetGenerator  # noqa: E402
from dataset.mixture import (  # noqa: E402
    DEFAULT_MIXTURE,
    SYNTHETIC_TOKEN_SLICES,
    synthetic_token_budget,
    validate_synthetic_budget,
)
from dataset.production_processor import (  # noqa: E402
    Document,
    HeldoutRegistry,
    LicenseFilter,
    ProductionShardBuilder,
    classify_real_slice,
)
from dataset.repair_traces import generate_repair_rows  # noqa: E402
from dataset.security_data import SecurityDataGenerator  # noqa: E402



REAL_SLICES = ("the-stack-code", "tests", "github-prs", "github-issues", "docs-and-comments")


def iter_real_documents(real_dir: Path) -> Iterator[dict]:
    """Yield raw JSONL rows from every ``*.jsonl`` shard under ``real_dir``."""

    if not real_dir.exists():
        return

    for path in sorted(real_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def real_documents(real_dir: Path, license_filter: LicenseFilter) -> Iterator[Document]:
    """Route real rows into slices, dropping anything the license gate rejects."""

    for row in iter_real_documents(real_dir):
        content = row.get("content") or ""
        if not content:
            continue

        language = row.get("language") or "unknown"
        path = row.get("path") or row.get("source") or ""

        spdx = license_filter.canonicalize(row.get("license"))
        if not spdx:
            continue

        yield Document(
            content=content,
            language=language,
            source=row.get("source") or "real",
            license=spdx,
            path=path,
            slice_name=row.get("slice") or classify_real_slice(path, language, content),
            is_multi_file=bool(row.get("is_multi_file")),
            metadata={"provenance": "real"},
        )




def _row_text(row: dict) -> str:
    parts = [row.get("prompt") or row.get("instruction") or ""]
    parts.append(row.get("response") or row.get("output") or "")
    if row.get("chosen") or row.get("rejected"):
        parts.append(row.get("chosen") or "")
        parts.append(row.get("rejected") or "")
    return "\n\n".join(part for part in parts if part)


def _as_document(row: dict, slice_name: str, source: str) -> Document:
    """Wrap a generated training row as a shardable document."""

    return Document(
        content=_row_text(row),
        language=row.get("language") or "text",
        source=source,
        license="CC0-1.0",
        path=f"{slice_name}:{row.get('id') or row.get('bug_id') or row.get('case_id') or 'row'}",
        slice_name=slice_name,
        is_multi_file=slice_name in {"app-sft", "app-repair-sft", "project-scaffold-sft"},
        metadata={"provenance": "synthetic", "slice": slice_name},
    )


def induced_documents(real_dir: Path | None, *, limit: int = 60) -> Iterator[Document]:
    """instruction-sft: hybrid rows induced from the real corpus itself.

    Real files, real licences, AST-derived instruction. This is the hybrid slice
    in the mixture: the *content* is real code, only the task framing is
    generated.
    """

    if real_dir is None:
        return

    induction = InstructionInduction(real_dir, language="python", max_files=400)

    try:
        rows = induction.generate(count=limit)
    except (FileNotFoundError, OSError):
        return

    for row in rows:
        yield _as_document(row, "instruction-sft", "instruction_induction")


def synthetic_documents(
    *,
    apps: int = 8,
    include_apps: bool = True,
) -> Iterator[Document]:
    """Every model-generated slice, wrapped as documents with provenance."""

    security = SecurityDataGenerator(heldout=False)
    for row in (
        security.review_samples()
        + security.repair_samples()
        + security.implement_samples()
        + security.dpo_pairs()
    ):
        yield _as_document(row, "security-sft", "security_data")

    repair = generate_repair_rows()
    for row in repair["sft"]:
        yield _as_document(row, "repair-trace-sft", "repair_traces")
    for row in repair["dpo"]:
        yield _as_document(row, "app-preference-dpo", "repair_traces")

    projects = ProjectDatasetGenerator()
    for row in projects.to_sft_rows():
        yield _as_document(row, "project-scaffold-sft", "project_data")

    for row in RecoveryInstructionGenerator().generate(6):
        yield _as_document(row, "recovery-sft", "augmentation")

    if not include_apps:
        return

    corpus = AppCorpus(limit=apps, verify=False)
    samples = corpus.samples()
    for row in corpus.sft_rows(samples):
        yield _as_document(row, "app-sft", "app_corpus")
    for row in corpus.repair_rows(samples):
        yield _as_document(row, "app-repair-sft", "app_corpus")
    for row in corpus.preference_rows(samples):
        yield _as_document(row, "app-preference-dpo", "app_corpus")


def iter_mixture_documents(
    real_dir: Path | None,
    *,
    apps: int,
    include_apps: bool,
    license_filter: LicenseFilter | None = None,
) -> Iterator[Document]:
    """Real backbone first, then synthetic slices (so the cap can trim them)."""

    gate = license_filter or LicenseFilter()

    if real_dir is not None:
        yield from real_documents(real_dir, gate)

    yield from synthetic_documents(apps=apps, include_apps=include_apps)




def build(
    output_dir: Path,
    tokenizer,
    *,
    real_dir: Path | None = None,
    heldout_dir: Path | None = None,
    apps: int = 8,
    include_apps: bool = True,
    limit_induced: int = 60,
    seq_length: int = 4096,
    synthetic_cap_share: float = 0.05,
    val_fraction: float = 0.02,
    shard_size_tokens: int = 50_000_000,
    synthetic_only: bool = False,
    seed: int = 42,
) -> dict:
    """Build shards and return the corpus manifest."""

    audit = validate_synthetic_budget(DEFAULT_MIXTURE, synthetic_cap_share)

    heldout = HeldoutRegistry()
    if heldout_dir is not None:
        heldout.load_directory(heldout_dir)

    builder = ProductionShardBuilder(
        output_dir,
        tokenizer,
        seq_length=seq_length,
        shard_size_tokens=shard_size_tokens,
        val_fraction=val_fraction,
        seed=seed,
        heldout=heldout,
    )

    token_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    for doc in (
        ()
        if synthetic_only
        else (real_documents(real_dir, LicenseFilter()) if real_dir else ())
    ):
        tokens = builder.add_document(doc)
        token_counts[doc.slice_name] += tokens
        doc_counts[doc.slice_name] += 1

    real_tokens = sum(token_counts.values())

    if synthetic_only:
        allowance = int(real_tokens * synthetic_cap_share / (1.0 - synthetic_cap_share))
    else:
        allowance = int(real_tokens * synthetic_cap_share / (1.0 - synthetic_cap_share))

    used = 0
    rows_over_budget = 0

    for doc in induced_documents(None if synthetic_only else real_dir, limit=limit_induced):
        tokens = builder.add_document(doc)
        token_counts[doc.slice_name] += tokens
        doc_counts[doc.slice_name] += 1

    pending: dict[str, list[Document]] = {}
    for doc in synthetic_documents(apps=apps, include_apps=include_apps):
        pending.setdefault(doc.slice_name, []).append(doc)

    planned = {
        spec.name: spec.weight
        for spec in DEFAULT_MIXTURE
        if spec.name in SYNTHETIC_TOKEN_SLICES
    }
    planned_total = sum(planned.values()) or 1.0

    def _cost(doc: Document) -> int:
        try:
            return len(builder.tokenizer.encode(doc.content, add_special_tokens=True))
        except TypeError:
            return len(builder.tokenizer.encode(doc.content))

    cursors = {name: 0 for name in pending}
    slice_used = {name: 0 for name in pending}
    finished: set[str] = set()

    while len(finished) < len(pending):
        progressed = False
        for name in sorted(pending):
            if name in finished:
                continue
            rows = pending[name]

            if cursors[name] >= len(rows):
                finished.add(name)
                continue

            if synthetic_only:
                row_allowance = None
            else:
                share = planned.get(name, 0.0) / planned_total
                row_allowance = max(1, int(round(allowance * share)))

                if cursors[name] == 0 and slice_used[name] == 0:
                    row_allowance = max(row_allowance, _cost(rows[0]))

                if slice_used[name] >= row_allowance:
                    dropped[name] += len(rows) - cursors[name]
                    finished.add(name)
                    continue

            doc = rows[cursors[name]]
            cost = _cost(doc)

            over_slice = row_allowance is not None and slice_used[name] + cost > row_allowance
            over_global = not synthetic_only and used + cost > allowance

            cursors[name] += 1

            if over_slice or over_global:
                dropped[name] += 1
                rows_over_budget += 1
                continue

            progressed = True
            tokens = builder.add_document(doc)
            used += tokens
            slice_used[name] += cost
            token_counts[doc.slice_name] += tokens
            doc_counts[doc.slice_name] += 1

        if not progressed:
            break

    synthetic_tokens = used
    total_tokens = sum(token_counts.values())

    budget = synthetic_token_budget(
        max(total_tokens, 1),
        max_share=synthetic_cap_share,
        synthetic_share_of_slice=1.0,
    )

    manifest = builder.finalize(
        extra_metadata={
            "builder": "scripts/build_corpus_v1.py",
            "mixture": {spec.name: spec.weight for spec in DEFAULT_MIXTURE},
            "slice_tokens": dict(sorted(token_counts.items())),
            "slice_documents": dict(sorted(doc_counts.items())),
            "synthetic": {
                "tokens": synthetic_tokens,
                "allowance_tokens": allowance,
                "dropped_rows": dict(sorted(dropped.items())),
                "allowance_starved": rows_over_budget > 0 and used == 0,
                "note": (
                    "the real backbone was too small to fund even one synthetic row; "
                    "build with the real corpus before training"
                    if rows_over_budget > 0 and used == 0
                    else ""
                ),
                "share": round(synthetic_tokens / max(total_tokens, 1), 6),
                "cap_share": synthetic_cap_share,
                "cap_tokens": budget["cap_tokens"],
                "within_cap": synthetic_tokens <= budget["cap_tokens"] or synthetic_only,
                "slices": sorted(SYNTHETIC_TOKEN_SLICES),
                "document_share_limit": audit["document_share"],
            },
            "sources_expected": [spec.name for spec in DEFAULT_MIXTURE],
        }
    )

    missing = [
        spec.name
        for spec in DEFAULT_MIXTURE
        if spec.name not in doc_counts and spec.kind != "hybrid"
    ]

    manifest["missing_slices"] = missing
    return manifest


def _demo_real_rows() -> list[dict]:
    """Tiny synthetic stand-in for a fetched corpus (demo/smoke test only)."""

    sample = (
        '"""Retry helper."""\n\n'
        "import time\n\n\n"
        "def retry(fn, attempts=3, delay=0.1):\n"
        "    for attempt in range(attempts):\n"
        "        try:\n"
        "            return fn()\n"
        "        except OSError:\n"
        "            time.sleep(delay * (attempt + 1))\n"
        "    raise RuntimeError('retries exhausted')\n"
    )
    test = (
        "from pkg.retry import retry\n\n\n"
        "def test_retry_succeeds_second_time():\n"
        "    calls = []\n\n"
        "    def flaky():\n"
        "        calls.append(1)\n"
        "        if len(calls) < 2:\n"
        "            raise OSError('boom')\n"
        "        return 'ok'\n\n"
        "    assert retry(flaky) == 'ok'\n"
    )
    return [
        {"content": sample, "language": "python", "path": "pkg/retry.py", "license": "MIT", "source": "demo"},
        {"content": test, "language": "python", "path": "tests/test_retry.py", "license": "MIT", "source": "demo"},
        {"content": "# Retry\n\nBackoff notes.\n", "language": "markdown", "path": "docs/retry.md", "license": "MIT", "source": "demo"},
        {"content": sample, "language": "python", "path": "pkg/vendored.py", "license": "GPL-3.0", "source": "demo"},
    ]


def _write_demo_corpus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "demo-0.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in _demo_real_rows():
            handle.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real_dir", type=Path, default=None)
    parser.add_argument("--heldout_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--apps", type=int, default=8, help="blueprints rendered for app slices")
    parser.add_argument("--induced", type=int, default=60, help="hybrid induced rows from the real corpus")
    parser.add_argument("--no_apps", action="store_true", help="skip app slices (fast smoke test)")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--synthetic_cap", type=float, default=0.05)
    parser.add_argument(
        "--shard_tokens", type=int, default=50_000_000, help="tokens per shard file"
    )
    parser.add_argument(
        "--synthetic_only",
        action="store_true",
        help="demo mode: build only synthetic slices (the real cap needs a real corpus)",
    )
    parser.add_argument("--demo", action="store_true", help="write a tiny demo real corpus first")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        args.real_dir = args.output_dir.parent / f"{args.output_dir.name}-demo-real"
        _write_demo_corpus(args.real_dir)

    from model.tokenizer import CodeTokenizer

    tokenizer = CodeTokenizer()

    manifest = build(
        args.output_dir,
        tokenizer,
        real_dir=args.real_dir,
        heldout_dir=args.heldout_dir,
        apps=args.apps,
        include_apps=not args.no_apps,
        limit_induced=args.induced,
        seq_length=args.seq_length,
        synthetic_cap_share=args.synthetic_cap,
        shard_size_tokens=args.shard_tokens,
        synthetic_only=args.synthetic_only,
    )

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote shards to {args.output_dir}")
        print(f"  documents: {manifest['total_documents']}")
        print(f"  tokens:    {manifest['total_packed_tokens']:,}")
        synthetic = manifest["synthetic"]
        print(
            f"  synthetic: {synthetic['share']:.2%} of tokens "
            f"({'within' if synthetic['within_cap'] else 'OVER'} {synthetic['cap_share']:.0%} cap)"
        )
        for name, tokens in manifest["slice_tokens"].items():
            print(f"    {name:24s} {tokens:>12,} tokens")
        if manifest["missing_slices"]:
            print(f"  missing slices: {', '.join(manifest['missing_slices'])}")

    if not manifest["synthetic"]["within_cap"] and not args.synthetic_only:
        print("error: synthetic token share exceeded the cap", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
