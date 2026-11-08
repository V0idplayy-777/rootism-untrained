#!/usr/bin/env python3
"""Audit the curated real-code source list before a multi-hour fetch runs.

A fetch job that 404s or is copyleft is not a crash — it is a wasted download
and a misleading line in ``sources.jsonl``. This script answers, *before* the
fetch, three questions the corpus plan depends on:

1. **Does every candidate exist and is it permissively licensed?** GitHub's own
   licence classification (``GET /repos/{owner}/{name}``) is advisory, but it
   turns a 300 MB guess into a decision. The fetch still re-classifies the
   archive's LICENSE text fail-closed; this only prunes the obvious misses.
2. **Is there enough material?** GitHub reports repository size in KB, which
   bounds the working-tree size a ``codeload`` tarball can produce. Summing it
   per language gives an honest availability estimate instead of a hand-wave.
3. **Is the list internally sane?** Duplicate slugs, unknown languages and
   extensions with no budget are cheap to catch here.

    python scripts/source_inventory.py                 # audit (uses network)
    python scripts/source_inventory.py --offline       # static checks only
    python scripts/source_inventory.py --json docs/corpus/repo_inventory.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import corpus_sources  # noqa: E402
from scripts import fetch_real_corpus as frc  # noqa: E402

logger = logging.getLogger("source-inventory")

PERMISSIVE = set(frc.PERMISSIVE_LICENSES)

TARBALL_FRACTION_OF_REPO_SIZE = 0.35
BYTES_PER_TOKEN = 3.6

REQUIRED_CODE_LANGUAGES: frozenset[str] = frozenset(
    {
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
        "dart",
        "lua",
        "elixir",
        "haskell",
        "vue",
    }
)


def merged_repos() -> list[tuple[str, str]]:
    """Core + expanded candidates, de-duplicated, order preserved."""

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for repo, language in (
        list(frc.GITHUB_REPOS)
        + corpus_sources.EXTRA_GITHUB_REPOS
        + corpus_sources.LONG_TAIL_GITHUB_REPOS
    ):
        slug = repo.split("@", 1)[0]
        if slug in seen:
            continue
        seen.add(slug)
        out.append((repo, language))
    return out


def merged_budgets() -> dict[str, int]:
    budgets = dict(frc.LANGUAGE_BUDGETS)
    for language, chars in corpus_sources.EXTRA_LANGUAGE_BUDGETS.items():
        budgets[language] = budgets.get(language, 0) + chars
    return budgets


def merged_extensions() -> dict[str, str]:
    extensions = dict(frc.EXTENSIONS)
    extensions.update(corpus_sources.EXTRA_EXTENSIONS)
    return extensions


def static_audit() -> dict[str, Any]:
    """Everything checkable without touching the network."""

    repos = merged_repos()
    extensions = merged_extensions()
    budgets = merged_budgets()

    raw = [
        r.split("@", 1)[0]
        for r, _ in list(frc.GITHUB_REPOS)
        + corpus_sources.EXTRA_GITHUB_REPOS
        + corpus_sources.LONG_TAIL_GITHUB_REPOS
    ]
    core = [r.split("@", 1)[0] for r, _ in frc.GITHUB_REPOS]
    expanded = [
        r.split("@", 1)[0]
        for r, _ in corpus_sources.EXTRA_GITHUB_REPOS + corpus_sources.LONG_TAIL_GITHUB_REPOS
    ]
    duplicates = sorted({slug for slug, count in Counter(expanded).items() if count > 1})
    overlap_with_core = sorted(set(core) & set(expanded))

    languages = {language for _repo, language in repos}
    unmapped_extensions = sorted(
        ext for ext, language in extensions.items() if language not in budgets
    )
    languages_without_extensions = sorted(
        language
        for language in languages
        if language not in set(extensions.values()) and language != "markdown"
    )

    return {
        "github_repos": len(repos),
        "core_repos": len(frc.GITHUB_REPOS),
        "expanded_repos": len(corpus_sources.EXTRA_GITHUB_REPOS),
        "long_tail_repos": len(corpus_sources.LONG_TAIL_GITHUB_REPOS),
        "pypi_extra": len(corpus_sources.EXTRA_PYPI_PACKAGES),
        "npm_extra": len(corpus_sources.EXTRA_NPM_PACKAGES),
        "pypi_packages": len(frc.PYPI_PACKAGES) + len(corpus_sources.EXTRA_PYPI_PACKAGES),
        "npm_packages": len(frc.NPM_PACKAGES) + len(corpus_sources.EXTRA_NPM_PACKAGES),
        "duplicate_slugs": duplicates,
        "overlap_with_core": overlap_with_core,
        "languages": sorted(languages),
        "extensions": len(extensions),
        "unmapped_extensions": unmapped_extensions,
        "languages_without_extensions": languages_without_extensions,
        "language_budgets_mb": {
            language: round(chars / 1e6, 1) for language, chars in sorted(budgets.items())
        },
        "candidates_by_language": dict(
            sorted(Counter(language for _repo, language in repos).items())
        ),
    }


def _api_get(url: str, token: str | None) -> dict[str, Any] | None:
    headers = {
        "User-Agent": frc.USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as error:
            if error.code in (404, 403, 451):
                return {"__error__": error.code}
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1.5 * (attempt + 1))
    return None


def probe_repo(repo: str, language: str, token: str | None) -> dict[str, Any]:
    slug = repo.split("@", 1)[0]
    payload = _api_get(f"https://api.github.com/repos/{slug}", token)
    record: dict[str, Any] = {
        "repo": repo,
        "slug": slug,
        "language": language,
        "pinned_ref": repo.split("@", 1)[1] if "@" in repo else "",
    }
    if payload is None:
        record.update(status="unreachable")
        return record
    if "__error__" in payload:
        record.update(status="missing" if payload["__error__"] == 404 else "blocked")
        return record

    license_info = payload.get("license") or {}
    spdx = str(license_info.get("spdx_id") or "")
    size_kb = int(payload.get("size") or 0)
    record.update(
        status="ok",
        spdx=spdx,
        permissive=spdx in PERMISSIVE,
        archived=bool(payload.get("archived")),
        size_kb=size_kb,
        estimated_bytes=int(size_kb * 1024 * TARBALL_FRACTION_OF_REPO_SIZE),
        stars=int(payload.get("stargazers_count") or 0),
        pushed_at=str(payload.get("pushed_at") or ""),
        default_branch=str(payload.get("default_branch") or ""),
        actual_language=str(payload.get("language") or ""),
    )
    return record


def network_audit(workers: int = 8, limit: int | None = None) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repos = merged_repos()
    if limit:
        repos = repos[:limit]

    records: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe_repo, repo, language, token) for repo, language in repos]
        for index, future in enumerate(cf.as_completed(futures), 1):
            records.append(future.result())
            if index % 100 == 0:
                logger.info("probed %d/%d repositories", index, len(repos))

    records.sort(key=lambda record: record["slug"])

    permissive = [r for r in records if r.get("permissive")]
    by_language_bytes: dict[str, int] = defaultdict(int)
    by_language_repos: Counter[str] = Counter()
    for record in permissive:
        by_language_bytes[record["language"]] += record.get("estimated_bytes", 0)
        by_language_repos[record["language"]] += 1

    total_bytes = sum(by_language_bytes.values())
    return {
        "probed": len(records),
        "permissive": len(permissive),
        "rejected_license": sorted(
            {r["slug"]: r.get("spdx") for r in records if r["status"] == "ok" and not r.get("permissive")}.items()
        ),
        "missing": sorted(r["slug"] for r in records if r["status"] == "missing"),
        "blocked_or_unreachable": sorted(
            r["slug"] for r in records if r["status"] in {"blocked", "unreachable"}
        ),
        "archived": sorted(r["slug"] for r in permissive if r.get("archived")),
        "estimated_source_mb": round(total_bytes / 1e6, 1),
        "estimated_tokens_billions": round(total_bytes / BYTES_PER_TOKEN / 1e9, 3),
        "per_language": {
            language: {
                "repositories": by_language_repos[language],
                "estimated_mb": round(by_language_bytes[language] / 1e6, 1),
                "budget_mb": round(merged_budgets().get(language, 0) / 1e6, 1),
            }
            for language in sorted(by_language_repos, key=lambda l: -by_language_bytes[l])
        },
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="static checks only")
    parser.add_argument("--json", type=Path, default=None, help="write the full inventory here")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    audit: dict[str, Any] = {"static": static_audit()}
    static = audit["static"]
    print(
        f"candidates: {static['github_repos']} repos "
        f"(core {static['core_repos']} + expanded {static['expanded_repos']}), "
        f"{static['pypi_packages']} PyPI, {static['npm_packages']} npm"
    )
    print(f"languages:  {len(static['languages'])} -> {', '.join(static['languages'])}")
    if static["overlap_with_core"]:
        print(
            f"overlap:    {len(static['overlap_with_core'])} slugs are in both the core and "
            f"expanded list (de-duplicated at fetch time)"
        )
    if static["duplicate_slugs"]:
        print(f"duplicates: {static['duplicate_slugs'][:10]}")
    if static["unmapped_extensions"]:
        print(f"extensions without a budget: {static['unmapped_extensions']}")

    problems: list[str] = []
    if static["unmapped_extensions"]:
        problems.append(f"{len(static['unmapped_extensions'])} extensions have no budget")
    if static["duplicate_slugs"]:
        problems.append("duplicate slugs inside the expanded list")

    if not args.offline:
        network = network_audit(workers=args.workers, limit=args.limit)
        audit["network"] = {k: v for k, v in network.items() if k != "records"}
        print(
            f"verified:   {network['permissive']}/{network['probed']} permissively licensed, "
            f"~{network['estimated_source_mb']:,.0f} MB source "
            f"(~{network['estimated_tokens_billions']:.2f}B tokens at "
            f"{BYTES_PER_TOKEN} bytes/token)"
        )
        thin = [
            f"{language} {payload['estimated_mb']:.0f}MB < budget {payload['budget_mb']:.0f}MB"
            for language, payload in network["per_language"].items()
            if language in REQUIRED_CODE_LANGUAGES
            and payload["budget_mb"]
            and payload["estimated_mb"] < payload["budget_mb"] * 0.25
        ]
        for language in sorted(REQUIRED_CODE_LANGUAGES - set(network["per_language"])):
            thin.append(f"{language} has no verified source")
        if thin:
            print("thin coverage: " + "; ".join(thin))
            problems.extend(thin)
        if network["missing"]:
            print(f"missing ({len(network['missing'])}): {network['missing'][:12]}")

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(audit | {"records": network["records"]}, indent=2) + "\n")
            print(f"wrote {args.json}")
    elif args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"wrote {args.json}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
