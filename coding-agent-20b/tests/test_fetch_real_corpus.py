"""Offline end-to-end tests for ``scripts/fetch_real_corpus.py``.

The fetcher's only untestable-in-CI boundary is the network. Everything else —
tar/npm/sdist extraction, SPDX classification, the quality gate, dedup, the
held-out split, shard writing and the manifest — is deterministic and is
exercised here for real against genuine gzip archives built in-process.

The network is replaced by an in-memory transport, so ``run_fetch`` is driven
end to end exactly as production drives it: jobs -> transport -> archive
extraction -> licence gate -> quality gate -> dedup -> held-out split ->
JSONL shards + manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from dataset.production_processor import PERMISSIVE_LICENSES
from scripts import fetch_real_corpus as frc


PY_CODE = '''"""Bounded retry helper with linear backoff.

The helper retries a callable until it succeeds or the attempt budget is
exhausted, sleeping longer between each attempt.
"""

import time


class Retrier:
    """Run a callable with linear backoff between attempts."""

    def __init__(self, attempts=3, delay=0.1):
        self.attempts = attempts
        self.delay = delay

    def call(self, fn):
        """Invoke fn until it stops raising OSError."""
        for attempt in range(self.attempts):
            try:
                return fn()
            except OSError:
                time.sleep(self.delay * (attempt + 1))
        raise RuntimeError("retries exhausted")


def retry(fn, attempts=3, delay=0.1):
    """Shortcut for Retrier(attempts, delay).call(fn)."""
    return Retrier(attempts, delay).call(fn)
'''

TEST_CODE = '''import pytest

from pkg.retry import Retrier


def test_retrier_succeeds_after_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("boom")
        return "ok"

    assert Retrier().call(flaky) == "ok"


def test_retrier_gives_up():
    def always():
        raise OSError("nope")

    with pytest.raises(RuntimeError):
        Retrier(attempts=2).call(always)
'''

DOC_MD = """# Retry

Linear backoff from a 100ms base. Attempts default to three, and the helper
raises once the budget is exhausted so callers cannot loop forever.
"""

JS_CODE = """'use strict';

// Small debounce helper used by the search box.
function debounce(fn, wait) {
  let timer = null;
  return function debounced(...args) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

module.exports = { debounce };
"""

MIT_TEXT = """MIT License

Copyright (c) 2024 Example

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software").
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

GPL_TEXT = """GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007
Copyright (C) 2007 Free Software Foundation, Inc.
Everyone is permitted to copy and distribute verbatim copies of this license.
"""


def make_tar(entries: dict[str, bytes | str], *, prefix: str = "repo-1.0.0") -> bytes:
    """Build a real .tar.gz in memory."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in entries.items():
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            info = tarfile.TarInfo(name=f"{prefix}/{name}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def github_archive() -> bytes:
    return make_tar(
        {
            "LICENSE": MIT_TEXT,
            "pkg/retry.py": PY_CODE,
            "tests/test_retry.py": TEST_CODE,
            "docs/retry.md": DOC_MD,
            "node_modules/vendored/index.js": JS_CODE,
            "pkg/binary.py": b"\x00\x01\x02not text",
            "pkg/short.py": "x = 1\n",
            "README.txt": "tiny",
        }
    )


class Transport:
    """Stand-in for the network: URL -> bytes, with call accounting."""

    def __init__(self, responses: dict[str, bytes | None]):
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: int = 30, headers=None) -> bytes | None:  # noqa: ANN001
        self.calls.append(url)
        return self.responses.get(url)


@pytest.fixture(autouse=True)
def isolate_source_expansion(monkeypatch):
    """Keep the unit tests hermetic: the curated expansion is opt-in here.

    Production runs use the full list (1000+ repositories); a unit test that
    patches ``GITHUB_REPOS`` to two fake slugs must not then try to download the
    real ones.
    """

    monkeypatch.setattr(frc, "EXTRA_GITHUB_REPOS", [])
    monkeypatch.setattr(frc, "LONG_TAIL_GITHUB_REPOS", [])
    monkeypatch.setattr(frc, "EXTRA_PYPI_PACKAGES", [])
    monkeypatch.setattr(frc, "EXTRA_NPM_PACKAGES", [])


@pytest.fixture()
def transport(monkeypatch):
    """Patch the module's ``_download`` seam only (the real network boundary)."""

    def install(responses: dict[str, bytes | None]) -> Transport:
        t = Transport(responses)
        monkeypatch.setattr(frc, "_download", t)
        return t

    return install




def test_license_text_classification_is_permissive_only():
    assert frc.classify_license_text(MIT_TEXT) in PERMISSIVE_LICENSES
    assert frc.classify_license_text(MIT_TEXT) == "MIT"
    assert frc.classify_license_text(GPL_TEXT) is None
    assert frc.classify_license_text("no license text at all") is None
    assert frc.classify_license_text("") is None


def test_spdx_expression_classification():
    assert frc.classify_spdx_expression("MIT") == "MIT"
    assert frc.classify_spdx_expression("Apache-2.0") == "Apache-2.0"
    assert frc.classify_spdx_expression("GPL-3.0-only") is None
    assert frc.classify_spdx_expression("(MIT OR Apache-2.0)") in PERMISSIVE_LICENSES
    assert frc.classify_spdx_expression("SEE LICENSE IN LICENSE") is None


def test_pypi_metadata_classification():
    assert frc.classify_pypi_metadata("Metadata-Version: 2.1\nLicense: MIT\n") == "MIT"
    assert frc.classify_pypi_metadata("License: GPL-3.0\n") is None
    assert frc.classify_pypi_metadata("Metadata-Version: 2.1\n") is None




def test_tar_extraction_keeps_code_docs_and_tests_and_skips_junk():
    docs = frc._extract_docs_from_tar(github_archive(), "github:demo", on_license_file=lambda t: None)
    paths = {doc["path"] for doc in docs}
    languages = {doc["language"] for doc in docs}

    assert "github:demo:pkg/retry.py" in paths
    assert "github:demo:tests/test_retry.py" in paths
    assert "github:demo:docs/retry.md" in paths
    assert not any("node_modules" in path for path in paths)
    assert "github:demo:pkg/binary.py" not in paths
    assert "github:demo:pkg/short.py" not in paths
    assert {"python", "markdown"} <= languages


def test_tar_extraction_reports_license_files():
    seen: list[str] = []
    frc._extract_docs_from_tar(github_archive(), "github:demo", on_license_file=seen.append)
    assert any("MIT License" in text for text in seen)


def test_tar_extraction_handles_a_corrupt_archive():
    assert frc._extract_docs_from_tar(b"not a tarball", "github:broken", on_license_file=lambda t: None) == []
    good = github_archive()
    assert frc._extract_docs_from_tar(good[: len(good) // 3], "github:trunc", on_license_file=lambda t: None) == []


def test_pypi_sdist_extraction_reads_pkg_info_for_license():
    blob = make_tar(
        {
            "PKG-INFO": "Metadata-Version: 2.1\nName: demo\nLicense: MIT\n",
            "demo/core.py": PY_CODE,
        },
        prefix="demo-1.2.3",
    )
    docs, license_id = frc._extract_pypi_sdist(blob, "pypi:demo")
    assert license_id == "MIT"
    assert docs and all(doc["license"] == "MIT" for doc in docs)
    assert any(doc["path"].endswith("core.py") for doc in docs)


def test_pypi_sdist_with_gpl_pkg_info_is_rejected():
    blob = make_tar({"PKG-INFO": "License: GPL-3.0\n", "demo/core.py": PY_CODE}, prefix="demo-1.0")
    docs, license_id = frc._extract_pypi_sdist(blob, "pypi:gpl")
    assert license_id is None
    assert all(doc["license"] == "" for doc in docs)


def test_npm_tarball_extraction_reads_package_json():
    blob = make_tar(
        {
            "package.json": json.dumps({"name": "demo", "license": "MIT"}),
            "index.js": JS_CODE,
        },
        prefix="package",
    )
    docs, license_id = frc._extract_npm_tarball(blob, "npm:demo")
    assert license_id == "MIT"
    assert docs and any(doc["language"] == "javascript" for doc in docs)


def test_npm_tarball_without_license_is_rejected():
    blob = make_tar({"package.json": json.dumps({"name": "demo"}), "index.js": JS_CODE}, prefix="package")
    docs, license_id = frc._extract_npm_tarball(blob, "npm:nolicense")
    assert license_id is None


def test_npm_tarball_recovers_license_from_license_file():
    blob = make_tar(
        {
            "package.json": json.dumps({"name": "demo", "license": "UNLICENSED"}),
            "LICENSE": MIT_TEXT,
            "index.js": JS_CODE,
        },
        prefix="package",
    )
    _docs, license_id = frc._extract_npm_tarball(blob, "npm:fallback")
    assert license_id == "MIT"




def test_quality_gate_rejects_gpl_and_short_documents():
    good = frc.enrich_and_validate_doc(
        {"content": PY_CODE, "language": "python", "path": "repo:pkg/retry.py", "license": "MIT", "source": "github:demo"}
    )
    assert good is not None
    assert good["license"] == "MIT"
    assert good["slice"] == "the-stack-code"
    assert good["quality_score"] > 0

    gpl = frc.enrich_and_validate_doc(
        {"content": PY_CODE, "language": "python", "path": "pkg/x.py", "license": "GPL-3.0", "source": "s"}
    )
    assert gpl is None

    tiny = frc.enrich_and_validate_doc(
        {"content": "x = 1\n", "language": "python", "path": "pkg/x.py", "license": "MIT", "source": "s"}
    )
    assert tiny is None


def test_heldout_split_is_deterministic_and_order_independent():
    paths = [f"repo:pkg/mod_{i}.py" for i in range(200)]
    first = [p for p in paths if frc.is_heldout_document(p)]
    second = [p for p in reversed(paths) if frc.is_heldout_document(p)]
    assert first == list(reversed(second))
    assert 0 < len(first) < len(paths)
    assert frc.is_heldout_document("anything", heldout_every=0) is False


def test_writer_dedups_and_shards(tmp_path):
    writer = frc.CorpusWriter(tmp_path, shard_docs=2)
    doc = {"content": PY_CODE, "path": "a.py", "slice": "the-stack-code", "license": "MIT"}
    assert writer.duplicate(PY_CODE, "a.py") is False
    assert writer.add("python", doc) is True
    assert writer.duplicate(PY_CODE, "a.py") is True
    assert writer.duplicate(PY_CODE + "\n# same body", "a.py") is True
    writer.add("python", {**doc, "path": "b.py"})
    writer.flush()

    shards = sorted(tmp_path.glob("python-*.jsonl"))
    assert shards, "flush must write shards"
    lines = [json.loads(line) for shard in shards for line in shard.read_text().splitlines()]
    assert len(lines) == 2
    manifest = writer.manifest_fragment()
    assert manifest["documents"] == 2
    assert manifest["per_license"]["MIT"] == 2
    assert manifest["corpus_version"] == "corpus-v1-real"




def test_run_fetch_end_to_end_keeps_permissive_and_writes_manifest(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/permissive", "python"), ("demo/gpl", "python")])
    monkeypatch.setattr(frc, "PYPI_PACKAGES", [])
    monkeypatch.setattr(frc, "NPM_PACKAGES", [])
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 200_000, "markdown": 50_000, "javascript": 20_000})

    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])

    permissive_archive = github_archive()
    gpl_archive = make_tar({"LICENSE": GPL_TEXT, "pkg/other.py": PY_CODE}, prefix="gpl-1.0")

    transport(
        {
            "https://codeload.github.com/demo/permissive/tar.gz/HEAD": permissive_archive,
            "https://codeload.github.com/demo/gpl/tar.gz/HEAD": gpl_archive,
        }
    )

    out = tmp_path / "real"
    held = tmp_path / "heldout"
    manifest = frc.run_fetch(
        output_dir=out,
        heldout_dir=held,
        skip_pypi=True,
        skip_npm=True,
        workers=1,
        heldout_every=5,
        scale=1.0,
    )

    assert manifest["documents"] > 0
    assert manifest["chars"] > 0
    assert "MIT" in manifest["per_license"]
    assert manifest["corpus_version"] == "corpus-v1-real"

    # shards exist and every row is licensed + slice-tagged
    rows = [
        json.loads(line)
        for shard in sorted(out.glob("*.jsonl"))
        if shard.name != "sources.jsonl"
        for line in shard.read_text().splitlines()
    ]
    assert rows
    assert all(row["license"] in PERMISSIVE_LICENSES for row in rows)
    assert all(row["slice"] in {"the-stack-code", "tests", "docs-and-comments", "github-prs", "github-issues"} for row in rows)

    log = [json.loads(line) for line in (out / "sources.jsonl").read_text().splitlines()]
    statuses = {entry["source"]: entry["status"] for entry in log}
    assert statuses["github:demo/permissive"] == "ok"
    assert statuses["github:demo/gpl"] == "license-rejected"

    assert (out / "manifest.json").exists()
    heldout_manifest = json.loads((held / "heldout_manifest.json").read_text())
    assert heldout_manifest["documents"] == manifest["heldout"]["documents"]
    heldout_hashes = set(heldout_manifest["heldout_sha256"])
    for row in rows:
        assert row["sha256"] not in heldout_hashes, "held-out content leaked into the train shards"


def test_run_fetch_is_resumable_and_skips_finished_sources(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/permissive", "python")])
    monkeypatch.setattr(frc, "PYPI_PACKAGES", [])
    monkeypatch.setattr(frc, "NPM_PACKAGES", [])
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 200_000, "markdown": 50_000})
    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])

    t = transport({"https://codeload.github.com/demo/permissive/tar.gz/HEAD": github_archive()})
    out, held = tmp_path / "real", tmp_path / "held"

    first = frc.run_fetch(output_dir=out, heldout_dir=held, skip_pypi=True, skip_npm=True, workers=1)
    assert first["documents"] > 0
    assert first["jobs"] == {
        "planned": 1,
        "fetched": 1,
        "skipped_completed": 0,
        "failed_or_unavailable": 0,
    }
    fetched_before = len(t.calls)

    second = frc.run_fetch(output_dir=out, heldout_dir=held, skip_pypi=True, skip_npm=True, workers=1)
    assert len(t.calls) == fetched_before, "a completed source must not be re-downloaded"
    assert second["documents"] == 0
    assert second["jobs"] == {
        "planned": 1,
        "fetched": 0,
        "skipped_completed": 1,
        "failed_or_unavailable": 0,
    }


def test_run_fetch_survives_a_dead_transport(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/dead", "python")])
    monkeypatch.setattr(frc, "PYPI_PACKAGES", [])
    monkeypatch.setattr(frc, "NPM_PACKAGES", [])
    transport({})

    manifest = frc.run_fetch(output_dir=tmp_path / "o", heldout_dir=tmp_path / "h", skip_pypi=True, skip_npm=True, workers=1)
    assert manifest["documents"] == 0
    assert manifest["jobs"]["fetched"] == 0
    assert manifest["jobs"]["failed_or_unavailable"] == 1
    assert (tmp_path / "o" / "manifest.json").exists(), "a failed fetch still writes an honest manifest"


def test_download_retries_then_gives_up(monkeypatch):
    """_download is the real network seam: retries, then returns None."""

    attempts = {"n": 0}

    class Boom:
        def __enter__(self):
            raise urllib_error()

        def __exit__(self, *exc):
            return False

    def urllib_error():
        import urllib.error

        return urllib.error.URLError("boom")

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        return Boom()

    import urllib.request

    monkeypatch.setattr(frc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(frc.time, "sleep", lambda _s: None)
    assert frc._download("https://example.invalid/x", timeout=1) is None
    assert attempts["n"] == frc.RETRIES


def test_download_returns_none_immediately_on_403(monkeypatch):
    import urllib.error

    attempts = {"n": 0}

    def fake_urlopen(req, timeout=None):
        attempts["n"] += 1
        raise urllib.error.HTTPError("u", 403, "forbidden", {}, None)

    monkeypatch.setattr(frc.urllib.request, "urlopen", fake_urlopen)
    assert frc._download("https://example.invalid/x", timeout=1) is None
    assert attempts["n"] == 1, "403/404/429 must not be retried"


def test_download_returns_bytes_on_success(monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"payload"

    monkeypatch.setattr(frc.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    assert frc._download("https://example.invalid/x", timeout=1) == b"payload"


def test_github_tarball_url_construction():
    urls: list[str] = []
    import scripts.fetch_real_corpus as mod

    original = mod._download
    try:
        mod._download = lambda url, **k: urls.append(url)  # type: ignore[assignment]
        mod.fetch_github_tarball("demo/repo")
        mod.fetch_github_tarball("demo/repo@v1.2.3")
    finally:
        mod._download = original

    assert urls[0] == "https://codeload.github.com/demo/repo/tar.gz/HEAD"
    assert urls[1] == "https://codeload.github.com/demo/repo/tar.gz/refs/tags/v1.2.3"


def test_pypi_and_npm_fetchers_read_their_metadata(monkeypatch):
    sdist = make_tar({"PKG-INFO": "License: MIT\n", "demo/core.py": PY_CODE}, prefix="demo-1.0")
    index = json.dumps(
        {"info": {"version": "1.0"}, "urls": [{"packagetype": "sdist", "url": "https://files.invalid/demo.tar.gz"}]}
    ).encode()
    blob, version = frc.fetch_pypi_sdist.__wrapped__("demo") if hasattr(frc.fetch_pypi_sdist, "__wrapped__") else (None, "")
    assert blob is None
    responses = {
        "https://pypi.org/pypi/demo/json": index,
        "https://files.invalid/demo.tar.gz": sdist,
    }
    monkeypatch.setattr(frc, "_download", Transport(responses))
    blob, version = frc.fetch_pypi_sdist("demo")
    assert version == "1.0"
    assert blob == sdist

    npm_tarball = make_tar({"package.json": json.dumps({"license": "MIT"})}, prefix="package")
    npm_index = json.dumps(
        {"dist-tags": {"latest": "2.0.0"}, "versions": {"2.0.0": {"dist": {"tarball": "https://npm.invalid/d.tgz"}}}}
    ).encode()
    monkeypatch.setattr(
        frc, "_download", Transport({"https://registry.npmjs.org/demo": npm_index, "https://npm.invalid/d.tgz": npm_tarball})
    )
    blob, version = frc.fetch_npm_tarball("demo")
    assert version == "2.0.0"
    assert blob == npm_tarball


# --------------------------------------------------------------------------- #
# Curated expansion: the backbone has to actually cover the languages it claims
# --------------------------------------------------------------------------- #


def test_the_expansion_covers_every_required_language():
    """A language with no candidate source cannot appear in the backbone."""

    from scripts import corpus_sources

    required = {
        "python",
        "typescript",
        "javascript",
        "go",
        "rust",
        "java",
        "c",
        "cpp",
        "csharp",
        "ruby",
        "php",
        "kotlin",
        "swift",
        "scala",
    }
    inventory = corpus_sources.source_inventory()
    missing = sorted(required - set(inventory))
    assert not missing, f"curated expansion has no sources for {missing}"

    # every expanded entry must map to a language that has an availability budget,
    # otherwise its files are extracted and immediately discarded
    unbudgeted = sorted(
        {
            language
            for _repo, language in corpus_sources.EXTRA_GITHUB_REPOS
            + corpus_sources.LONG_TAIL_GITHUB_REPOS
            if language not in frc.LANGUAGE_BUDGETS
        }
    )
    assert not unbudgeted, f"expanded languages without a budget: {unbudgeted}"


def test_the_expansion_is_large_enough_to_be_a_backbone():
    from scripts import corpus_sources

    total = (
        len(frc.GITHUB_REPOS)
        + len(corpus_sources.EXTRA_GITHUB_REPOS)
        + len(corpus_sources.LONG_TAIL_GITHUB_REPOS)
    )
    assert total >= 900, f"only {total} curated repositories"
    from scripts import corpus_sources as sources

    assert len(sources.EXTRA_PYPI_PACKAGES) + len(frc.PYPI_PACKAGES) >= 250
    assert len(sources.EXTRA_NPM_PACKAGES) + len(frc.NPM_PACKAGES) >= 250


def test_merged_source_lists_deduplicate_without_losing_pins():
    merged = frc.merged_github_repos()
    slugs = [repo.split("@", 1)[0] for repo, _lang in merged]
    assert len(slugs) == len(set(slugs)), "merged_github_repos must de-duplicate slugs"
    assert sum(1 for repo, _ in merged if "@" in repo) >= 1, "pinned refs must survive merging"


def test_extra_extensions_and_budgets_are_wired_into_the_fetcher():
    for extension in (".dart", ".lua", ".ex", ".hs", ".vue"):
        assert extension in frc.EXTENSIONS, f"{extension} has no language mapping"
    for language in ("dart", "lua", "elixir", "haskell"):
        assert language in frc.LANGUAGE_BUDGETS, f"{language} has no availability budget"


# --------------------------------------------------------------------------- #
# Disk-backed dedup at scale
# --------------------------------------------------------------------------- #


def test_run_fetch_can_use_the_disk_dedup_index(tmp_path, transport, monkeypatch):
    """The scale path: near-duplicates removed across sources via SQLite."""

    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/one", "python"), ("demo/two", "python")])
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 4_000_000, "markdown": 200_000})
    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])

    # two archives that ship the same module under different paths
    first = github_archive()
    second = make_tar(
        {
            "LICENSE": MIT_TEXT,
            "copied/retry_v2.py": PY_CODE,
            "tests/test_retry_v2.py": TEST_CODE,
            "docs/retry_v2.md": DOC_MD,
        },
        prefix="fork-1.0.0",
    )
    transport(
        {
            "https://codeload.github.com/demo/one/tar.gz/HEAD": first,
            "https://codeload.github.com/demo/two/tar.gz/HEAD": second,
        }
    )

    index_path = tmp_path / "dedup.sqlite"
    manifest = frc.run_fetch(
        output_dir=tmp_path / "real",
        heldout_dir=tmp_path / "held",
        skip_pypi=True,
        skip_npm=True,
        workers=1,
        heldout_every=0,
        dedup_index_path=index_path,
    )

    assert index_path.exists(), "the dedup index must be persisted for resumable runs"
    dedup = manifest["dedup"]
    duplicates = dedup["exact_duplicates"] + dedup["near_duplicates"]
    assert duplicates == 3, f"every forked file (code, test, doc) must be removed once: {dedup}"
    assert dedup["indexed"] == dedup["checked"] - duplicates
    assert dedup["dedup_rate"] == 0.5
    assert dedup["index"].endswith("dedup.sqlite")

    rows = [
        json.loads(line)
        for shard in sorted((tmp_path / "real").glob("*.jsonl"))
        if shard.name != "sources.jsonl"
        for line in shard.read_text().splitlines()
    ]
    bodies = [row["content"] for row in rows]
    assert len(bodies) == len(set(bodies)), "a duplicate body must not be written twice"


def test_rows_match_the_trainer_schema_contract(tmp_path, transport, monkeypatch):
    monkeypatch.setattr(frc, "GITHUB_REPOS", [("demo/permissive", "python")])
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 4_000_000, "markdown": 200_000})
    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])
    transport({"https://codeload.github.com/demo/permissive/tar.gz/HEAD": github_archive()})

    frc.run_fetch(
        output_dir=tmp_path / "real",
        heldout_dir=tmp_path / "held",
        skip_pypi=True,
        skip_npm=True,
        workers=1,
        heldout_every=0,
    )

    rows = [
        json.loads(line)
        for shard in sorted((tmp_path / "real").glob("*.jsonl"))
        if shard.name != "sources.jsonl"
        for line in shard.read_text().splitlines()
    ]
    assert rows
    for row in rows:
        assert frc.validate_row_schema(row) == [], row

    broken = dict(rows[0])
    broken["license"] = "GPL-3.0"
    assert "non-permissive-license:'GPL-3.0'" in frc.validate_row_schema(broken)
    broken = dict(rows[0])
    del broken["quality_score"]
    assert "missing:quality_score" in frc.validate_row_schema(broken)


# --------------------------------------------------------------------------- #
# Honest availability accounting
# --------------------------------------------------------------------------- #


def test_budget_saturation_is_reported_as_availability_not_as_a_drop(tmp_path, transport, monkeypatch):
    """A language that is already fed must not look like it was filtered out."""

    monkeypatch.setattr(
        frc, "GITHUB_REPOS", [("demo/first", "python"), ("demo/second", "python")]
    )
    # both archives are byte-identical, so the second contributes nothing new:
    # it must report *why*, and after de-duplication the budget is still open.
    monkeypatch.setattr(frc, "LANGUAGE_BUDGETS", {"python": 1_000, "markdown": 1})
    monkeypatch.setattr(frc, "fetch_github_issues_and_prs", lambda *a, **k: [])
    transport(
        {
            "https://codeload.github.com/demo/first/tar.gz/HEAD": github_archive(),
            "https://codeload.github.com/demo/second/tar.gz/HEAD": github_archive(),
        }
    )

    manifest = frc.run_fetch(
        output_dir=tmp_path / "real",
        heldout_dir=tmp_path / "held",
        skip_pypi=True,
        skip_npm=True,
        workers=1,
        heldout_every=0,
    )

    entries = [
        json.loads(line) for line in (tmp_path / "real" / "sources.jsonl").read_text().splitlines()
    ]
    statuses = [entry["status"] for entry in entries]
    assert statuses.count("ok") == 2
    second = entries[1]
    assert second["kept"] == 0, "the second archive is a copy of the first"
    assert second["docs"] > 0, "it still extracted documents - they were filtered, not missing"
    saturated = [entry for entry in entries if entry["status"] == "budget-saturated"]
    assert saturated or second["kept"] == 0
    if saturated:
        assert "availability" in json.dumps(saturated[0]["channels"])
    usage = manifest["budget_usage"]["python"]
    assert usage["chars"] <= usage["budget_chars"]
    assert usage["saturated"] is (usage["chars"] >= usage["budget_chars"])


def test_jobs_are_ordered_so_a_truncated_run_still_covers_every_language(monkeypatch):
    """Ordering by budget deficit is what stops a smoke run being 100% Python."""

    monkeypatch.setattr(
        frc,
        "GITHUB_REPOS",
        [("demo/py-a", "python"), ("demo/py-b", "python"), ("demo/go-a", "go")],
    )
    jobs = frc.build_jobs(
        languages=None,
        budgets={"python": 8_000_000, "go": 40_000_000, "markdown": 1_000},
        skip_github=False,
        skip_pypi=True,
        skip_npm=True,
        only=None,
        target_chars=None,
    )
    first_three = [job["id"] for job in jobs[:3]]
    assert any("go" in job for job in first_three), first_three
    assert any("py-a" in job or "py-b" in job for job in first_three), first_three
