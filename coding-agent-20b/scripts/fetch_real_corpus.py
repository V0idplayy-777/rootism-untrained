"""Fetch a real, license-clean, multi-language code corpus for tokenizer training.

Why this exists: `sample_tokenizer_corpus.py` streams HuggingFace datasets,
which is the right shape for a GPU cluster but is not always available (air-
gapped CI, licence audits, bandwidth). Everything needed for a *real* corpus
run is reachable from three package/source hosts that every build machine
already talks to:

- GitHub  — codeload tarballs of a curated list of popular, permissively
            licensed repositories across 14 languages,
- PyPI    — sdists of popular packages (Python plus the C/Rust they vendor),
- npm     — package tarballs (JavaScript/TypeScript, tests included).

Guarantees (same contract as `sample_tokenizer_corpus.py`):

- **Licence gate, fail-closed.** Every source is classified from its actual
  licence text/metadata (MIT / Apache-2.0 / BSD-2 / BSD-3 / ISC / MPL-2.0 /
  Unlicense / CC0 / 0BSD / Zlib). Anything unclassifiable is skipped and
  logged. Each emitted document records its source licence; the manifest
  records every source decision.
- **Quality gate.** 10..5000 lines per file (the same gate the trainer
  applies), <=400 KB per file, no vendored/generated/minified/lockfiles,
  no VCS/build directories.
- **Balance.** Per-language character budgets with a per-source cap, so one
  monorepo cannot dominate a language.
- **Deterministic.** Sources are processed in manifest order, archive members
  in sorted order, content-hash dedup; reruns skip already-fetched sources.
- **Held-out split.** Every `--heldout_every`-th accepted document is written
  to a separate directory that never enters the training shards.

Output: JSONL shards (`{lang}-NNNNNN.jsonl`) with rows
``{"content", "language", "path", "license", "source"}`` plus `manifest.json`
— directly consumable by `train_production_tokenizer.py --data_dir`.

Usage:
    python scripts/fetch_real_corpus.py --output_dir data/tokenizer_corpus \
        --heldout_dir data/tokenizer_heldout
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

USER_AGENT = "coding-agent-20b-tokenizer-corpus/1.0 (+local build)"
TIMEOUT = 120
RETRIES = 3


EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "vendor", "vendors", "third_party", "thirdparty", "external", "deps",
    ".idea", "fixtures", "testdata", "test_data", "snapshots", "__snapshots__",
}

SKIP_NAME_PATTERNS = (
    re.compile(r"\.min\.(js|ts)$"),
    re.compile(r"\.d\.ts$"),
    re.compile(r"^(package-lock|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|"
               r"Cargo\.lock|composer\.lock|Gemfile\.lock|go\.sum)$"),
    re.compile(r"\.gen\.(ts|py|go|rs)$"),
    re.compile(r"^(.?generated[_-].*)\.(py|ts|go|rs)$"),
)

MIN_LINES = 10
MAX_LINES = 5000
MAX_FILE_BYTES = 400_000

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
    "Unlicense", "CC0-1.0", "0BSD", "Zlib",
}

LANGUAGE_BUDGETS = {
    "python": 40_000_000,
    "javascript": 20_000_000,
    "typescript": 20_000_000,
    "java": 14_000_000,
    "go": 14_000_000,
    "rust": 12_000_000,
    "c": 12_000_000,
    "cpp": 12_000_000,
    "ruby": 8_000_000,
    "php": 10_000_000,
    "csharp": 8_000_000,
    "kotlin": 5_000_000,
    "swift": 5_000_000,
    "sql": 4_000_000,
    "shell": 4_000_000,
    "yaml": 4_000_000,
    "toml": 1_500_000,
}

PER_SOURCE_FRACTION = 0.30

# the actual archive licence is classified from its LICENSE text and the

GITHUB_REPOS: list[tuple[str, str]] = [
    ("psf/requests", "python"), ("psf/black", "python"),
    ("encode/httpx", "python"), ("encode/starlette", "python"),
    ("tiangolo/fastapi", "python"), ("pallets/werkzeug", "python"),
    ("pallets/click", "python"), ("pallets/jinja", "python"),
    ("pallets/itsdangerous", "python"), ("sphinx-doc/sphinx", "python"),
    ("scrapy/scrapy", "python"), ("python-attrs/attrs", "python"),
    ("marshmallow-code/marshmallow", "python"), ("aio-libs/aiofiles", "python"),
    ("httpie/cli", "python"), ("jd/tenacity", "python"),
    ("pytest-dev/pytest", "python"), ("pydantic/pydantic", "python"),
    ("python-poetry/poetry", "python"), ("scanny/pathlib2", "python"),
    ("jquery/jquery", "javascript"), ("d3/d3", "javascript"),
    ("axios/axios", "javascript"), ("facebook/react", "javascript"),
    ("moment/moment", "javascript"), ("expressjs/express", "javascript"),
    ("lodash/lodash", "javascript"), ("mrdoob/three.js", "javascript"),
    ("chartjs/Chart.js", "javascript"), ("caolan/async", "javascript"),
    ("socketio/socket.io", "javascript"), ("jashkenas/underscore", "javascript"),
    ("ramda/ramda", "javascript"), ("ml5js/ml5-library", "javascript"),
    ("microsoft/TypeScript", "typescript"), ("rollup/rollup", "typescript"),
    ("evanw/esbuild", "typescript"), ("vitejs/vite", "typescript"),
    ("vercel/swr", "typescript"), ("TanStack/query", "typescript"),
    ("trpc/trpc", "typescript"), ("prisma/prisma", "typescript"),
    ("pmndrs/zustand", "typescript"), ("reduxjs/redux-toolkit", "typescript"),
    ("typeorm/typeorm", "typescript"), ("nestjs/nest", "typescript"),
    ("colinhacks/zod", "typescript"), ("Effect-TS/effect", "typescript"),
    ("google/guava", "java"), ("google/auto", "java"),
    ("google/gson", "java"), ("brettwooldridge/HikariCP", "java"),
    ("redis/jedis", "java"), ("mockito/mockito", "java"),
    ("apache/commons-lang", "java"), ("apache/commons-io", "java"),
    ("apache/commons-codec", "java"), ("square/okhttp", "java"),
    ("square/retrofit", "java"), ("spring-projects/spring-framework", "java"),
    ("redis/redis@7.4.1", "c"), ("libuv/libuv", "c"), ("lua/lua@v5.4.7", "c"),
    ("jqlang/jq", "c"), ("antirez/sds", "c"), ("memcached/memcached", "c"),
    ("redis/hiredis", "c"), ("nginx/nginx", "c"), ("tmux/tmux", "c"),

    ("nlohmann/json", "cpp"), ("fmtlib/fmt", "cpp"), ("google/re2", "cpp"),
    ("Tencent/rapidjson", "cpp"), ("gulrak/filesystem", "cpp"),
    ("protocolbuffers/protobuf", "cpp"), ("abseil/abseil-cpp", "cpp"),
    ("spf13/cobra", "go"), ("spf13/viper", "go"), ("gin-gonic/gin", "go"),
    ("gorilla/mux", "go"), ("go-kit/kit", "go"), ("golang/crypto", "go"),
    ("golang/net", "go"), ("BurntSushi/toml", "go"), ("golang/sync", "go"),
    ("grpc/grpc-go", "go"), ("Masterminds/sprig", "go"), ("golang/go", "go"),
    ("serde-rs/serde", "rust"), ("serde-rs/json", "rust"),
    ("tokio-rs/tokio", "rust"), ("BurntSushi/ripgrep", "rust"),
    ("clap-rs/clap", "rust"), ("rayon-rs/rayon", "rust"),
    ("seanmonstar/reqwest", "rust"), ("dtolnay/anyhow", "rust"),
    ("dtolnay/thiserror", "rust"), ("hyperium/hyper", "rust"),
    ("rust-lang/regex", "rust"), ("rust-random/rand", "rust"),
    ("rails/rails", "ruby"), ("sinatra/sinatra", "ruby"),
    ("jekyll/jekyll", "ruby"), ("puma/puma", "ruby"),
    ("rubygems/rubygems", "ruby"), ("lostisland/faraday", "ruby"),
    ("redis/redis-rb", "ruby"),
    ("laravel/framework", "php"), ("symfony/symfony", "php"),
    ("composer/composer", "php"), ("guzzle/guzzle", "php"),
    ("doctrine/orm", "php"), ("doctrine/inflector", "php"),
    ("JamesNK/Newtonsoft.Json", "csharp"), ("dotnet/efcore", "csharp"),
    ("dotnet/orleans", "csharp"), ("autofac/Autofac", "csharp"),
    ("serilog/serilog", "csharp"), ("dotnet/command-line-api", "csharp"),
    ("moq/moq", "csharp"),
    ("JetBrains/Exposed", "kotlin"), ("cashapp/sqldelight", "kotlin"),
    ("ktorio/ktor", "kotlin"), ("square/okio", "kotlin"),
    ("square/moshi", "kotlin"),
    ("apple/swift-log", "swift"), ("apple/swift-argument-parser", "swift"),
    ("apple/swift-nio", "swift"), ("apple/swift-collections", "swift"),
    ("Alamofire/Alamofire", "swift"), ("vapor/vapor", "swift"),
    ("pointfreeco/swift-snapshot-testing", "swift"),
    ("duckdb/duckdb", "sql"),
]

PYPI_PACKAGES = [
    "flask", "django", "jinja2", "markupsafe", "click", "itsdangerous",
    "werkzeug", "sqlalchemy", "alembic", "celery", "kombu", "requests",
    "urllib3", "certifi", "chardet", "idna", "rich", "typer", "pydantic",
    "attrs", "pytest", "pluggy", "iniconfig", "packaging", "setuptools",
    "wheel", "pip", "virtualenv", "tox", "coverage", "black", "mypy",
    "cryptography", "cffi", "pycparser", "pyyaml", "toml", "tomli", "tqdm",
    "colorama", "docutils", "pygments", "sphinx", "numpy", "scipy",
    "pillow", "greenlet", "aiohttp", "httpx", "h11", "h2", "ujson",
    "orjson", "msgpack", "pyzmq", "tornado", "twisted", "pyasn1", "bcrypt",
    "pynacl", "html5lib", "bleach", "markdown", "jmespath", "botocore",
    "boto3", "s3transfer", "pytz", "python-dateutil", "arrow", "pendulum",
    "croniter", "jsonschema", "faker", "factory-boy", "marshmallow",
    "webargs", "gunicorn", "uvicorn", "starlette", "anyio", "sniffio",
    "tenacity", "loguru", "prometheus-client", "psutil", "platformdirs",
    "filelock", "cachetools", "decorator", "wrapt", "six", "more-itertools",
    "toolz", "xxhash", "mmh3", "semver", "webob", "bottle", "falcon",
]

NPM_PACKAGES = [
    "express", "lodash", "axios", "chalk", "commander", "postcss", "webpack",
    "mocha", "date-fns", "three", "d3", "async", "bluebird", "debug",
    "minimist", "yargs", "uuid", "qs", "ws", "got", "glob", "micromatch",
    "rxjs", "typescript", "prettier", "eslint", "vue", "react", "redux",
    "svelte", "vite", "rollup", "esbuild", "zod", "acorn", "lodash-es",
    "node-fetch", "p-limit", "jest", "tar", "ini", "form-data",
    "follow-redirects", "mime-types", "handlebars", "underscore", "jquery",
    "bootstrap", "socket.io", "moment", "dayjs", "ramda", "immutable",
    "jest-mock", "espree",
]


_LICENSE_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("Apache-2.0", re.compile(r"apache license\b.{0,80}version 2\.0|"
                              r"www\.apache\.org/licenses/license-2\.0", re.I | re.S)),
    ("MPL-2.0", re.compile(r"mozilla public license.{0,60}version 2\.0", re.I | re.S)),
    ("MIT", re.compile(r"permission is hereby granted, free of charge", re.I)),
    ("BSD-3-Clause", re.compile(r"redistribution and use in source and binary forms.{0,400}"
                                r"neither the name", re.I | re.S)),
    ("BSD-2-Clause", re.compile(r"redistribution and use in source and binary forms", re.I | re.S)),
    ("ISC", re.compile(r"\bisc license\b|permission to use, copy, modify.{0,200}and/or distribute",
                       re.I | re.S)),
    ("Unlicense", re.compile(r"free and unencumbered software released into the public domain", re.I)),
    ("CC0-1.0", re.compile(r"creative commons zero|cc0 1\.0 universal", re.I)),
    ("0BSD", re.compile(r"bsd zero clause license", re.I)),
    ("Zlib", re.compile(r"zlib license|this software is provided .as-is.", re.I)),
]


def classify_license_text(text: str) -> str | None:
    """Best-effort SPDX id from licence text; None when unclassifiable."""
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text[:8000])
    for name, pattern in _LICENSE_SIGNATURES:
        if pattern.search(normalized):
            return name
    return None


_CANONICAL_LICENSES = {name.upper(): name for name in PERMISSIVE_LICENSES}


def classify_spdx_expression(expr: str) -> str | None:
    """Accept an SPDX expression if every branch is permissive (case-safe)."""
    if not expr:
        return None
    expr = expr.strip().strip("`\"'")
    if " OR " in expr.upper():
        for branch in re.split(r"\s+OR\s+", expr, flags=re.I):
            got = classify_spdx_expression(branch)
            if got:
                return got
        return None
    if " AND " in expr.upper() or " WITH " in expr.upper():
        return None
    token = expr.strip("()").strip()
    return _CANONICAL_LICENSES.get(token.upper())


def classify_pypi_metadata(pkg_info: str) -> str | None:
    classifiers = re.findall(r"Classifier:\s*License :: OSI Approved :: (.+)", pkg_info)
    name_map = {
        "MIT License": "MIT",
        "Apache Software License": "Apache-2.0",
        "BSD License": "BSD-3-Clause",
        "ISC License (ISCL)": "ISC",
        "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
        "The Unlicense (Unlicense)": "Unlicense",
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
    }
    for c in classifiers:
        if c.strip() in name_map:
            return name_map[c.strip()]
    m = re.search(r"^License(?:-Expression)?:\s*(.+)$", pkg_info, re.M | re.I)
    if m:
        return classify_spdx_expression(m.group(1))
    return None




def _download(url: str, timeout: int = TIMEOUT) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            logger.warning(f"download failed ({attempt + 1}/{RETRIES}): {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_github_tarball(repo: str) -> bytes | None:
    """Tarball from codeload: `owner/repo[@tag]` (no API call, no rate limit).

    Pin a tag when the default branch changed licence (e.g. Redis 8 is
    source-available; 7.4.x is BSD-3).
    """
    if "@" in repo:
        repo, ref = repo.split("@", 1)
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{ref}"
    else:
        url = f"https://codeload.github.com/{repo}/tar.gz/HEAD"
    return _download(url)


def fetch_pypi_sdist(name: str) -> tuple[bytes, str] | None:
    """Latest sdist for a package; returns (archive bytes, version)."""
    data = _download(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    if data is None:
        return None
    try:
        info = json.loads(data)
    except json.JSONDecodeError:
        return None
    version = info.get("info", {}).get("version", "")
    for url_info in info.get("urls", []):
        if url_info.get("packagetype") == "sdist":
            blob = _download(url_info["url"])
            if blob is not None:
                return blob, str(version)
    return None


def fetch_npm_tarball(name: str) -> tuple[bytes, str] | None:
    """Latest-version package tarball from the npm registry."""
    data = _download(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='/')}")
    if data is None:
        return None
    try:
        info = json.loads(data)
    except json.JSONDecodeError:
        return None
    version = info.get("dist-tags", {}).get("latest")
    if not version:
        return None
    tarball = (info.get("versions", {}).get(version, {}) or {}).get("dist", {}).get("tarball")
    if not tarball:
        return None
    blob = _download(tarball)
    if blob is None:
        return None
    return blob, str(version)




class CorpusWriter:
    """Streaming, deterministic JSONL shard writer per language."""

    def __init__(self, out_dir: Path, shard_docs: int = 200):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self.shard_index: Counter = Counter()
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.counts: Counter = Counter()
        self.chars: Counter = Counter()
        self.hasher = hashlib.sha256()
        self.seen_hashes: set[str] = set()

    def duplicate(self, content: str) -> bool:
        digest = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:32]
        if digest in self.seen_hashes:
            return True
        self.seen_hashes.add(digest)
        self.hasher.update(digest.encode())
        return False

    def add(self, language: str, doc: dict) -> bool:
        self.buffers[language].append(doc)
        self.counts[language] += 1
        self.chars[language] += len(doc["content"])
        if len(self.buffers[language]) >= self.shard_docs:
            self.flush(language)
        return True

    def flush(self, language: str | None = None) -> list[Path]:
        written = []
        for lang in ([language] if language else list(self.buffers)):
            docs = self.buffers.get(lang) or []
            if not docs:
                continue
            path = self.out_dir / f"{lang}-{self.shard_index[lang]:06d}.jsonl"
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            self.shard_index[lang] += 1
            self.buffers[lang] = []
            written.append(path)
        return written

    def manifest_fragment(self) -> dict:
        return {
            "documents": sum(self.counts.values()),
            "chars": sum(self.chars.values()),
            "per_language": {
                lang: {"documents": self.counts[lang], "chars": self.chars[lang]}
                for lang in sorted(self.counts)
            },
            "shards": sum(self.shard_index.values()),
            "corpus_sha256": self.hasher.hexdigest(),
        }


def _member_ok(name: str, size: int) -> bool:
    if size <= 0 or size > MAX_FILE_BYTES:
        return False
    parts = Path(name).parts
    if any(part in SKIP_DIRS for part in parts):
        return False
    base = Path(name).name
    if any(p.search(base) for p in SKIP_NAME_PATTERNS):
        return False
    return True


def _extract_docs_from_tar(
    blob: bytes,
    source_id: str,
    on_license_file,
) -> list[dict]:
    """Read accepted code files (sorted member order) from a tar.gz archive."""
    docs: list[dict] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted(
                (m for m in tar.getmembers() if m.isfile()),
                key=lambda m: m.name,
            )
            for member in members:
                name = member.name
                base = Path(name).name.lower()
                if base.startswith(("license", "copying", "copyright", "notice")):
                    if Path(base).suffix in {".txt", ".md", ".rst", ""}:
                        try:
                            text = tar.extractfile(member).read().decode("utf-8", "ignore")[:8000]
                            on_license_file(text)
                        except Exception:
                            pass
                    continue
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    raw = tar.extractfile(member).read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                if not (MIN_LINES <= lines <= MAX_LINES):
                    continue
                doc = {
                    "content": content,
                    "language": language,
                    "path": f"{source_id}:{name.split('/', 1)[-1]}",
                    "license": "",
                    "source": source_id,
                }
                docs.append(doc)
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning(f"bad archive {source_id}: {e}")
    return docs


def _extract_pypi_sdist(blob: bytes, source_id: str) -> tuple[list[dict], str | None]:
    """Docs + licence classification from an sdist (PKG-INFO + LICENSE)."""
    docs: list[dict] = []
    license_id: str | None = None
    pkg_info_text = ""
    license_text = ""

    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted(
                (m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name
            )
            for member in members:
                name = member.name
                base = Path(name).name
                if base == "PKG-INFO":
                    pkg_info_text = tar.extractfile(member).read().decode("utf-8", "ignore")[:12000]
                if base.startswith(("license", "copying")):
                    license_text = license_text or tar.extractfile(member).read().decode(
                        "utf-8", "ignore"
                    )[:8000]
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    raw = tar.extractfile(member).read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                if not (MIN_LINES <= lines <= MAX_LINES):
                    continue
                docs.append(
                    {
                        "content": content,
                        "language": language,
                        "path": f"{source_id}:{name.split('/', 1)[-1]}",
                        "license": "",
                        "source": source_id,
                    }
                )
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning(f"bad sdist archive {source_id}: {e}")
        return [], None

    license_id = classify_pypi_metadata(pkg_info_text) or classify_license_text(license_text)
    for doc in docs:
        doc["license"] = license_id or ""
    return docs, license_id


def _extract_npm_tarball(blob: bytes, source_id: str) -> tuple[list[dict], str | None]:
    """Docs + licence from an npm package tarball (package.json + LICENSE)."""
    docs: list[dict] = []
    license_id: str | None = None
    license_text = ""
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted(
                (m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name
            )
            for member in members:
                name = member.name
                base = Path(name).name
                if base == "package.json" and name.count("/") <= 1:
                    try:
                        meta = json.loads(tar.extractfile(member).read().decode("utf-8", "ignore"))
                        expr = meta.get("license") or ""
                        if isinstance(expr, dict):
                            expr = expr.get("type", "")
                        if isinstance(expr, list):
                            expr = " OR ".join(
                                e.get("type", str(e)) if isinstance(e, dict) else str(e)
                                for e in expr
                            )
                        license_id = classify_spdx_expression(str(expr))
                    except Exception:
                        pass
                if base.startswith(("license", "copying", "notice")):
                    license_text = license_text or tar.extractfile(member).read().decode(
                        "utf-8", "ignore"
                    )[:8000]
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    raw = tar.extractfile(member).read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                if not (MIN_LINES <= lines <= MAX_LINES):
                    continue
                docs.append(
                    {
                        "content": content,
                        "language": language,
                        "path": f"{source_id}:{name.split('/', 1)[-1]}",
                        "license": "",
                        "source": source_id,
                    }
                )
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning(f"bad npm archive {source_id}: {e}")
        return [], None

    license_id = license_id or classify_license_text(license_text)
    for doc in docs:
        doc["license"] = license_id or ""
    return docs, license_id


def _write_heldout(heldout_dir: Path, doc: dict) -> None:
    """Write one real file into the held-out tree (never used for training)."""
    lang = doc["language"]
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc["source"])[:60]
    directory = heldout_dir / lang / safe_source
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc["path"].split(":", 1)[-1])[-120:]
    suffix = Path(doc["path"]).suffix or ".txt"
    path = directory / f"{name}"
    if not path.suffix:
        path = path.with_suffix(suffix)
    try:
        path.write_text(doc["content"], encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a real tokenizer corpus")
    parser.add_argument("--output_dir", default="data/tokenizer_corpus")
    parser.add_argument("--heldout_dir", default="data/tokenizer_heldout")
    parser.add_argument("--source_log", default=None,
                        help="Where to record per-source decisions (default: <output_dir>/sources.jsonl)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for per-language char budgets")
    parser.add_argument("--languages", default=None,
                        help="Comma-separated allowlist of languages")
    parser.add_argument("--skip_github", action="store_true")
    parser.add_argument("--skip_pypi", action="store_true")
    parser.add_argument("--skip_npm", action="store_true")
    parser.add_argument("--only", default=None,
                        help="Only fetch sources whose id contains this substring")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--heldout_every", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    out_dir = Path(args.output_dir)
    heldout_dir = Path(args.heldout_dir)
    source_log = Path(args.source_log) if args.source_log else out_dir / "sources.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_log.parent.mkdir(parents=True, exist_ok=True)

    budgets = {
        lang: int(chars * args.scale) for lang, chars in LANGUAGE_BUDGETS.items()
    }
    if args.languages:
        allow = {s.strip() for s in args.languages.split(",") if s.strip()}
        budgets = {lang: n for lang, n in budgets.items() if lang in allow}

    done_sources: set[str] = set()
    if source_log.exists():
        with open(source_log) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        done_sources.add(row["source"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"resuming: {len(done_sources)} sources already fetched")

    writer = CorpusWriter(out_dir)
    totals: Counter = Counter()

    jobs: list[dict] = []
    if not args.skip_github:
        for repo, lang in GITHUB_REPOS:
            jobs.append({"kind": "github", "id": f"github:{repo}", "repo": repo})
    if not args.skip_pypi:
        for name in PYPI_PACKAGES:
            jobs.append({"kind": "pypi", "id": f"pypi:{name}", "name": name})
    if not args.skip_npm:
        for name in NPM_PACKAGES:
            jobs.append({"kind": "npm", "id": f"npm:{name}", "name": name})

    if args.only:
        jobs = [j for j in jobs if args.only in j["id"]]

    import random as _random

    _random.Random(1337).shuffle(jobs)

    cache_dir = out_dir / ".download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(job):
        if job["kind"] == "github":
            blob = fetch_github_tarball(job["repo"])
            version = ""
        elif job["kind"] == "pypi":
            got = fetch_pypi_sdist(job["name"])
            blob, version = (got[0], got[1]) if got else (None, "")
        else:
            got = fetch_npm_tarball(job["name"])
            blob, version = (got[0], got[1]) if got else (None, "")
        if blob is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", job["id"])
        cache_path = cache_dir / f"{job['index']:04d}-{safe}.bin"
        cache_path.write_bytes(blob)
        return {"path": cache_path, "version": version}

    for i, job in enumerate(jobs):
        job["index"] = i

    fetched: dict[str, dict] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in cf.as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"fetch error for {job['id']}: {e}")
                continue
            if result is not None:
                fetched[job["id"]] = result
            logger.info(f"fetched {job['id']} ({len(fetched)}/{len(jobs)})")

    with open(source_log, "a") as logf:

        def log_source(source_id: str, status: str, license_id, detail: dict):
            logf.write(
                json.dumps(
                    {
                        "source": source_id,
                        "status": status,
                        "license": license_id,
                        **detail,
                    }
                )
                + "\n"
            )
            logf.flush()

        for job in jobs:
            source_id = job["id"]
            if source_id in done_sources or source_id not in fetched:
                continue
            meta = fetched.pop(source_id)
            blob = meta["path"].read_bytes()
            meta["path"].unlink(missing_ok=True)
            version = meta["version"]

            if job["kind"] == "github":
                license_texts: list[str] = []
                docs = _extract_docs_from_tar(
                    blob, source_id, on_license_file=lambda t: license_texts.append(t)
                )
                license_id = classify_license_text("\n".join(license_texts))
                for doc in docs:
                    doc["license"] = license_id or ""
            elif job["kind"] == "pypi":
                docs, license_id = _extract_pypi_sdist(blob, source_id)
            else:
                docs, license_id = _extract_npm_tarball(blob, source_id)

            if license_id is None or license_id not in PERMISSIVE_LICENSES:
                log_source(source_id, "license-rejected", license_id,
                           {"bytes": len(blob), "docs": len(docs)})
                logger.info(f"skip {source_id}: licence {license_id!r} not permissive")
                continue

            archive_sha = hashlib.sha256(blob).hexdigest()[:16]

            source_chars: Counter = Counter()
            kept = 0
            per_source_cap = {
                lang: int(budgets.get(lang, 0) * PER_SOURCE_FRACTION)
                for lang in {d["language"] for d in docs}
            }
            for i, doc in enumerate(docs):
                lang = doc["language"]
                if budgets.get(lang, 0) <= 0:
                    continue
                if totals[lang] >= budgets[lang]:
                    continue
                if source_chars[lang] >= per_source_cap.get(lang, 0):
                    continue
                if writer.duplicate(doc["content"]):
                    continue
                if i % args.heldout_every == 0:
                    _write_heldout(heldout_dir, doc)
                    continue
                writer.add(lang, doc)
                totals[lang] += len(doc["content"])
                source_chars[lang] += len(doc["content"])
                kept += 1

            log_source(
                source_id,
                "ok",
                license_id,
                {
                    "bytes": len(blob),
                    "archive_sha256": archive_sha,
                    "version": version,
                    "docs": len(docs),
                    "kept": kept,
                    "chars": dict(source_chars),
                },
            )
            logger.info(
                f"kept {kept}/{len(docs)} docs from {source_id} "
                f"[{license_id}] chars={dict(source_chars)}"
            )

    writer.flush()
    manifest = writer.manifest_fragment()
    manifest["budgets"] = {k: v for k, v in budgets.items()}
    manifest["sources"] = len(done_sources) + len(
        [j for j in jobs if j["id"] not in done_sources]
    )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=== corpus complete ===")
    logger.info(f"documents: {manifest['documents']:,}")
    logger.info(f"chars:     {manifest['chars']:,}")
    for lang, row in manifest["per_language"].items():
        logger.info(f"  {lang:<12} {row['documents']:>6,} docs  {row['chars']:>12,} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
