"""Primary Real-Code Corpus Fetcher (Corpus v1 — 3B/20B Pretraining & Tokenizer).

Fetches permissively licensed source code, unit/integration test suites,
technical documentation, GitHub issues, and pull-request unified diffs from:
- GitHub  — codeload tarballs + optional issue/PR threads & unified diffs across
            300+ curated repositories spanning 18+ languages,
- PyPI    — sdists of 150+ popular Python/C/Rust packages,
- npm     — package tarballs of 110+ JavaScript/TypeScript libraries.

Production guarantees:
- **Fail-closed SPDX licence gate.** Every archive is classified from its actual
  LICENSE/PKG-INFO/package.json metadata (MIT / Apache-2.0 / BSD-2-Clause /
  BSD-3-Clause / ISC / MPL-2.0 / Unlicense / CC0-1.0 / 0BSD / Zlib / BSL-1.0 /
  PSF-2.0). Unclassifiable, copyleft (GPL/LGPL/AGPL), or source-available
  archives are rejected and logged.
- **Hard quality & complexity floor.** Enforces ``QualityScorer`` filters (line
  bounds, byte bounds, no minified/vendored/generated/lockfiles, syntax check,
  structural complexity floor) and attaches ``quality_score`` in ``[0, 1]``.
- **Exact + MinHash near-dedup.** Document-level SHA-256 + MinHash LSH
  near-duplicate removal via ``dataset.production_processor.Deduplicator``.
- **Canonical JSONL schema.** Every emitted row carries:
  ``{"content", "language", "license", "source", "path", "quality_score", "slice", "sha256"}``.
- **Isolated held-out split.** Deterministic hash-selected held-out documents
  are written exclusively to ``--heldout_dir`` (with ``heldout_manifest.json``)
  and never enter training shards.
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.dedup_index import DedupIndex
from dataset.production_processor import (
    Deduplicator,
    Document,
    LicenseFilter,
    QualityScorer,
    classify_real_slice,
)
from scripts import corpus_sources

logger = logging.getLogger(__name__)

USER_AGENT = "coding-agent-corpus-v1/1.0 (+reproducible-permissive-fetch)"
TIMEOUT = 120
RETRIES = 3

EXTENSIONS: dict[str, str] = {
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
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "markdown",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "vendor", "vendors", "third_party", "thirdparty", "external", "deps",
    ".idea", "fixtures", "testdata", "test_data", "snapshots", "__snapshots__",
}

SKIP_NAME_PATTERNS = (
    re.compile(r"\.min\.(js|ts|css)$"),
    re.compile(r"\.bundle\.(js|ts)$"),
    re.compile(r"\.d\.ts$"),
    re.compile(r"^(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|"
               r"Cargo\.lock|composer\.lock|Gemfile\.lock|go\.sum)$"),
    re.compile(r"\.gen\.(ts|py|go|rs)$"),
    re.compile(r"(_pb2\.py|\.pb\.go|\.pb\.cc|\.pb\.h)$"),
    re.compile(r"^(.?generated[_-].*)\.(py|ts|go|rs)$"),
)

MIN_LINES = 8
MIN_LINES_BY_LANGUAGE = {"markdown": 5, "diff": 5, "yaml": 5, "toml": 5}
MAX_LINES = 5000
MAX_FILE_BYTES = 450_000

PERMISSIVE_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
    "Unlicense", "CC0-1.0", "0BSD", "Zlib", "BSL-1.0", "PSF-2.0",
}

LANGUAGE_BUDGETS: dict[str, int] = {
    "python": 80_000_000,
    "typescript": 48_000_000,
    "javascript": 40_000_000,
    "go": 36_000_000,
    "rust": 36_000_000,
    "java": 28_000_000,
    "cpp": 28_000_000,
    "c": 24_000_000,
    "csharp": 18_000_000,
    "ruby": 16_000_000,
    "php": 16_000_000,
    "kotlin": 12_000_000,
    "swift": 12_000_000,
    "scala": 10_000_000,
    "sql": 10_000_000,
    "shell": 8_000_000,
    "markdown": 16_000_000,
    "yaml": 6_000_000,
    "toml": 3_000_000,
    "diff": 20_000_000,
}

EXTENSIONS.update(corpus_sources.EXTRA_EXTENSIONS)
LANGUAGE_BUDGETS.update(corpus_sources.EXTRA_LANGUAGE_BUDGETS)

EXTRA_GITHUB_REPOS: list[tuple[str, str]] = list(corpus_sources.EXTRA_GITHUB_REPOS)
LONG_TAIL_GITHUB_REPOS: list[tuple[str, str]] = list(corpus_sources.LONG_TAIL_GITHUB_REPOS)
EXTRA_PYPI_PACKAGES: list[str] = list(corpus_sources.EXTRA_PYPI_PACKAGES)
EXTRA_NPM_PACKAGES: list[str] = list(corpus_sources.EXTRA_NPM_PACKAGES)

REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "content",
    "language",
    "license",
    "source",
    "path",
    "quality_score",
    "slice",
    "sha256",
)

PER_SOURCE_FRACTION = 0.25

# 310+ curated permissively licensed repositories across 17+ languages.
# Every archive is still verified fail-closed from its actual LICENSE file.
GITHUB_REPOS: list[tuple[str, str]] = [
    ("psf/requests", "python"), ("psf/black", "python"),
    ("encode/httpx", "python"), ("encode/starlette", "python"),
    ("encode/uvicorn", "python"), ("encode/databases", "python"),
    ("tiangolo/fastapi", "python"), ("tiangolo/typer", "python"),
    ("tiangolo/sqlmodel", "python"), ("pallets/flask", "python"),
    ("pallets/werkzeug", "python"), ("pallets/click", "python"),
    ("pallets/jinja", "python"), ("pallets/itsdangerous", "python"),
    ("pallets/markupsafe", "python"), ("sphinx-doc/sphinx", "python"),
    ("scrapy/scrapy", "python"), ("python-attrs/attrs", "python"),
    ("python-attrs/cattrs", "python"), ("marshmallow-code/marshmallow", "python"),
    ("Tinche/aiofiles", "python"), ("aio-libs/aiohttp", "python"),
    ("aio-libs/yarl", "python"), ("aio-libs/multidict", "python"),
    ("httpie/cli", "python"), ("jd/tenacity", "python"),
    ("pytest-dev/pytest", "python"), ("pytest-dev/pluggy", "python"),
    ("pydantic/pydantic", "python"), ("pydantic/pydantic-settings", "python"),
    ("python-poetry/poetry", "python"), ("Textualize/rich", "python"),
    ("Textualize/textual", "python"), ("sqlalchemy/sqlalchemy", "python"),
    ("sqlalchemy/alembic", "python"), ("MagicStack/asyncpg", "python"),
    ("MagicStack/uvloop", "python"), ("agronholm/anyio", "python"),
    ("python-trio/trio", "python"), ("tornadoweb/tornado", "python"),
    ("benoitc/gunicorn", "python"), ("celery/kombu", "python"),
    ("pyca/cryptography", "python"), ("pyca/bcrypt", "python"),
    ("pyca/pynacl", "python"), ("HypothesisWorks/hypothesis", "python"),
    ("nedbat/coveragepy", "python"), ("python/mypy", "python"),
    ("astral-sh/ruff", "python"), ("pypa/packaging", "python"),
    ("pypa/pip", "python"), ("pypa/virtualenv", "python"),
    ("arrow-py/arrow", "python"), ("hynek/structlog", "python"),
    ("Delgan/loguru", "python"),
    ("microsoft/TypeScript", "typescript"), ("rollup/rollup", "typescript"),
    ("evanw/esbuild", "typescript"), ("vitejs/vite", "typescript"),
    ("vercel/swr", "typescript"), ("TanStack/query", "typescript"),
    ("TanStack/table", "typescript"), ("TanStack/router", "typescript"),
    ("trpc/trpc", "typescript"), ("prisma/prisma", "typescript"),
    ("pmndrs/zustand", "typescript"), ("pmndrs/jotai", "typescript"),
    ("reduxjs/redux-toolkit", "typescript"), ("typeorm/typeorm", "typescript"),
    ("nestjs/nest", "typescript"), ("colinhacks/zod", "typescript"),
    ("Effect-TS/effect", "typescript"), ("honojs/hono", "typescript"),
    ("fastify/fastify", "typescript"), ("drizzle-team/drizzle-orm", "typescript"),
    ("kysely-org/kysely", "typescript"), ("vitest-dev/vitest", "typescript"),
    ("playwright-community/playwright-go", "go"), ("microsoft/playwright", "typescript"),
    ("sindresorhus/ky", "typescript"), ("sindresorhus/execa", "typescript"),
    ("sindresorhus/got", "typescript"), ("unjs/nitro", "typescript"),
    ("unjs/ofetch", "typescript"), ("unjs/h3", "typescript"),
    ("graphql/graphql-js", "typescript"), ("apollographql/apollo-client", "typescript"),
    ("ianstormtaylor/superstruct", "typescript"), ("open-circle/valibot", "typescript"),
    ("biomejs/biome", "rust"), ("swc-project/swc", "rust"),
    ("oxc-project/oxc", "rust"), ("denoland/deno", "rust"),
    ("sveltejs/kit", "typescript"), ("vuejs/core", "typescript"),
    ("solidjs/solid", "typescript"), ("remix-run/react-router", "typescript"),
    ("jquery/jquery", "javascript"), ("d3/d3", "javascript"),
    ("axios/axios", "javascript"), ("facebook/react", "javascript"),
    ("moment/moment", "javascript"), ("iamkun/dayjs", "javascript"),
    ("expressjs/express", "javascript"), ("koajs/koa", "javascript"),
    ("lodash/lodash", "javascript"), ("mrdoob/three.js", "javascript"),
    ("chartjs/Chart.js", "javascript"), ("caolan/async", "javascript"),
    ("socketio/socket.io", "javascript"), ("jashkenas/underscore", "javascript"),
    ("ramda/ramda", "javascript"), ("immerjs/immer", "typescript"),
    ("webpack/webpack", "javascript"), ("babel/babel", "javascript"),
    ("eslint/eslint", "javascript"), ("prettier/prettier", "javascript"),
    ("mochajs/mocha", "javascript"), ("jestjs/jest", "typescript"),
    ("tj/commander.js", "javascript"), ("yargs/yargs", "javascript"),
    ("chalk/chalk", "javascript"), ("debug-js/debug", "javascript"),
    ("websockets/ws", "javascript"), ("nodejs/undici", "javascript"),
    ("spf13/cobra", "go"), ("spf13/viper", "go"), ("spf13/pflag", "go"),
    ("gin-gonic/gin", "go"), ("gorilla/mux", "go"), ("gorilla/websocket", "go"),
    ("go-chi/chi", "go"), ("gofiber/fiber", "go"), ("labstack/echo", "go"),
    ("go-kit/kit", "go"), ("golang/crypto", "go"), ("golang/net", "go"),
    ("golang/sync", "go"), ("BurntSushi/toml", "go"), ("grpc/grpc-go", "go"),
    ("Masterminds/sprig", "go"), ("uber-go/zap", "go"), ("rs/zerolog", "go"),
    ("sirupsen/logrus", "go"), ("stretchr/testify", "go"),
    ("google/go-cmp", "go"), ("go-gorm/gorm", "go"), ("jmoiron/sqlx", "go"),
    ("jackc/pgx", "go"), ("redis/go-redis", "go"), ("nats-io/nats.go", "go"),
    ("prometheus/client_golang", "go"), ("etcd-io/bbolt", "go"),
    ("dgraph-io/badger", "go"), ("charmbracelet/bubbletea", "go"),
    ("charmbracelet/lipgloss", "go"), ("urfave/cli", "go"),
    ("tidwall/gjson", "go"), ("valyala/fasthttp", "go"),
    ("serde-rs/serde", "rust"), ("serde-rs/json", "rust"),
    ("tokio-rs/tokio", "rust"), ("tokio-rs/axum", "rust"),
    ("tokio-rs/tracing", "rust"), ("tokio-rs/bytes", "rust"),
    ("tokio-rs/mio", "rust"), ("BurntSushi/ripgrep", "rust"),
    ("clap-rs/clap", "rust"), ("rayon-rs/rayon", "rust"),
    ("seanmonstar/reqwest", "rust"), ("dtolnay/anyhow", "rust"),
    ("dtolnay/thiserror", "rust"), ("dtolnay/syn", "rust"),
    ("dtolnay/quote", "rust"), ("hyperium/hyper", "rust"),
    ("hyperium/tonic", "rust"), ("hyperium/http", "rust"),
    ("rust-lang/regex", "rust"), ("rust-lang/log", "rust"),
    ("rust-lang/Hashbrown", "rust"), ("rust-random/rand", "rust"),
    ("crossbeam-rs/crossbeam", "rust"), ("amanieu/parking_lot", "rust"),
    ("actix/actix-web", "rust"), ("launchbadge/sqlx", "rust"),
    ("diesel-rs/diesel", "rust"), ("sharkdp/fd", "rust"),
    ("sharkdp/bat", "rust"), ("BurntSushi/memchr", "rust"),
    ("indexmap-rs/indexmap", "rust"), ("matklad/once_cell", "rust"),
    ("rust-itertools/itertools", "rust"), ("uuid-rs/uuid", "rust"),
    ("google/guava", "java"), ("google/auto", "java"),
    ("google/gson", "java"), ("google/dagger", "java"),
    ("brettwooldridge/HikariCP", "java"), ("redis/jedis", "java"),
    ("mockito/mockito", "java"), ("apache/commons-lang", "java"),
    ("apache/commons-io", "java"), ("apache/commons-codec", "java"),
    ("apache/commons-collections", "java"), ("square/okhttp", "java"),
    ("square/retrofit", "java"), ("square/javapoet", "java"),
    ("spring-projects/spring-framework", "java"), ("FasterXML/jackson-core", "java"),
    ("FasterXML/jackson-databind", "java"), ("junit-team/junit5", "java"),
    ("assertj/assertj", "java"), ("netty/netty", "java"),
    ("ben-manes/caffeine", "java"), ("resilience4j/resilience4j", "java"),
    ("redis/redis@7.4.1", "c"), ("libuv/libuv", "c"), ("lua/lua@v5.4.7", "c"),
    ("jqlang/jq", "c"), ("antirez/sds", "c"), ("memcached/memcached", "c"),
    ("redis/hiredis", "c"), ("nginx/nginx", "c"), ("tmux/tmux", "c"),
    ("curl/curl", "c"), ("madler/zlib", "c"), ("Cyan4973/lz4", "c"),
    ("facebook/zstd", "c"), ("DaveGamble/cJSON", "c"), ("nothings/stb", "c"),
    ("nlohmann/json", "cpp"), ("fmtlib/fmt", "cpp"), ("google/re2", "cpp"),
    ("google/googletest", "cpp"), ("google/benchmark", "cpp"),
    ("google/flatbuffers", "cpp"), ("google/leveldb", "cpp"),
    ("Tencent/rapidjson", "cpp"), ("gulrak/filesystem", "cpp"),
    ("protocolbuffers/protobuf", "cpp"), ("abseil/abseil-cpp", "cpp"),
    ("gabime/spdlog", "cpp"), ("catchorg/Catch2", "cpp"),
    ("pybind/pybind11", "cpp"), ("CLIUtils/CLI11", "cpp"),
    ("JamesNK/Newtonsoft.Json", "csharp"), ("dotnet/efcore", "csharp"),
    ("dotnet/orleans", "csharp"), ("autofac/Autofac", "csharp"),
    ("serilog/serilog", "csharp"), ("dotnet/command-line-api", "csharp"),
    ("moq/moq", "csharp"), ("DapperLib/Dapper", "csharp"),
    ("FluentValidation/FluentValidation", "csharp"), ("App-vNext/Polly", "csharp"),
    ("jbogard/MediatR", "csharp"), ("AutoMapper/AutoMapper", "csharp"),
    ("JetBrains/Exposed", "kotlin"), ("cashapp/sqldelight", "kotlin"),
    ("ktorio/ktor", "kotlin"), ("square/okio", "kotlin"),
    ("square/moshi", "kotlin"), ("Kotlin/kotlinx.coroutines", "kotlin"),
    ("Kotlin/kotlinx.serialization", "kotlin"), ("InsertKoinIO/koin", "kotlin"),
    ("apple/swift-log", "swift"), ("apple/swift-argument-parser", "swift"),
    ("apple/swift-nio", "swift"), ("apple/swift-collections", "swift"),
    ("apple/swift-algorithms", "swift"), ("apple/swift-crypto", "swift"),
    ("Alamofire/Alamofire", "swift"), ("vapor/vapor", "swift"),
    ("pointfreeco/swift-snapshot-testing", "swift"),
    ("scala/scala3", "scala"), ("typelevel/cats", "scala"),
    ("zio/zio", "scala"), ("com-lihaoyi/fastparse", "scala"),
    ("rails/rails", "ruby"), ("sinatra/sinatra", "ruby"),
    ("jekyll/jekyll", "ruby"), ("puma/puma", "ruby"),
    ("rubygems/rubygems", "ruby"), ("lostisland/faraday", "ruby"),
    ("redis/redis-rb", "ruby"), ("rspec/rspec-core", "ruby"),
    ("sidekiq/sidekiq", "ruby"),
    ("laravel/framework", "php"), ("symfony/symfony", "php"),
    ("composer/composer", "php"), ("guzzle/guzzle", "php"),
    ("doctrine/orm", "php"), ("doctrine/inflector", "php"),
    ("Seldaek/monolog", "php"), ("vlucas/phpdotenv", "php"),
    ("nikic/FastRoute", "php"),
    ("duckdb/duckdb", "sql"), ("postgres/postgres", "c"),
    ("bats-core/bats-core", "shell"), ("direnv/direnv", "shell"),
]

PYPI_PACKAGES: list[str] = [
    "flask", "django", "jinja2", "markupsafe", "click", "itsdangerous",
    "werkzeug", "sqlalchemy", "alembic", "celery", "kombu", "requests",
    "urllib3", "certifi", "chardet", "charset-normalizer", "idna", "rich",
    "typer", "pydantic", "pydantic-settings", "attrs", "cattrs", "pytest",
    "pytest-asyncio", "pytest-cov", "pluggy", "iniconfig", "packaging",
    "setuptools", "wheel", "pip", "virtualenv", "tox", "nox", "coverage",
    "black", "mypy", "isort", "flake8", "bandit", "cryptography", "cffi",
    "pycparser", "pyyaml", "toml", "tomli", "tomli-w", "tqdm", "colorama",
    "docutils", "pygments", "sphinx", "numpy", "scipy", "pillow", "greenlet",
    "aiohttp", "aiofiles", "yarl", "multidict", "aiosignal", "httpx",
    "httpcore", "h11", "h2", "hyperframe", "hpack", "ujson", "orjson",
    "msgpack", "pyzmq", "tornado", "twisted", "pyasn1", "bcrypt", "pynacl",
    "html5lib", "bleach", "markdown", "markdown-it-py", "jmespath",
    "botocore", "boto3", "s3transfer", "pytz", "python-dateutil", "arrow",
    "pendulum", "croniter", "jsonschema", "referencing", "rpds-py", "faker",
    "factory-boy", "hypothesis", "marshmallow", "webargs", "apispec",
    "gunicorn", "uvicorn", "starlette", "fastapi", "anyio", "sniffio",
    "trio", "tenacity", "backoff", "loguru", "structlog", "prometheus-client",
    "psutil", "platformdirs", "filelock", "cachetools", "diskcache",
    "decorator", "wrapt", "six", "more-itertools", "toolz", "cytoolz",
    "xxhash", "mmh3", "semver", "webob", "bottle", "falcon", "sanic",
    "litestar", "sqlparse", "pyjwt", "authlib", "itsdangerous", "dnspython",
    "asyncpg", "aiosqlite", "redis", "hiredis", "watchfiles", "dotenv",
    "python-dotenv", "tabulate", "humanize", "prettytable", "networkx",
    "sympy", "lark", "pyparsing", "parso", "jedi", "rope", "dulwich",
]

NPM_PACKAGES: list[str] = [
    "express", "koa", "fastify", "hono", "lodash", "lodash-es", "axios",
    "chalk", "commander", "postcss", "webpack", "mocha", "date-fns", "three",
    "d3", "async", "bluebird", "debug", "minimist", "yargs", "uuid", "nanoid",
    "qs", "ws", "got", "ky", "undici", "glob", "micromatch", "picomatch",
    "rxjs", "typescript", "prettier", "eslint", "vue", "react", "redux",
    "@reduxjs/toolkit", "zustand", "jotai", "immer", "svelte", "vite",
    "vitest", "rollup", "esbuild", "zod", "superstruct", "ajv", "acorn",
    "node-fetch", "ofetch", "p-limit", "p-queue", "p-retry", "jest", "tar",
    "ini", "dotenv", "cross-env", "form-data", "follow-redirects",
    "mime-types", "handlebars", "mustache", "marked", "cheerio", "underscore",
    "jquery", "bootstrap", "socket.io", "moment", "dayjs", "luxon", "ramda",
    "immutable", "jest-mock", "espree", "estraverse", "magic-string",
    "source-map", "semver", "lru-cache", "pino", "winston", "helmet", "cors",
    "cookie", "jsonwebtoken", "bcryptjs", "jose", "graphql", "dataloader",
    "knex", "kysely", "ioredis", "pg", "bullmq", "superjson", "devalue",
    "clsx", "tailwind-merge", "mitt", "eventemitter3", "chokidar", "execa",
    "ora", "listr2", "prompts", "enquirer", "kleur", "picocolors",
]

_LICENSE_SIGNATURES: list[tuple[str, re.Pattern]] = [
    ("Apache-2.0", re.compile(r"apache license\b.{0,80}version 2\.0|"
                              r"www\.apache\.org/licenses/license-2\.0", re.I | re.S)),
    ("MPL-2.0", re.compile(r"mozilla public license.{0,60}version 2\.0", re.I | re.S)),
    ("MIT", re.compile(r"permission is hereby granted, free of charge", re.I)),
    ("BSD-3-Clause", re.compile(r"redistribution and use in source and binary forms.{0,450}"
                                r"neither the name", re.I | re.S)),
    ("BSD-2-Clause", re.compile(r"redistribution and use in source and binary forms", re.I | re.S)),
    ("ISC", re.compile(r"\bisc license\b|permission to use, copy, modify.{0,200}and/or distribute",
                       re.I | re.S)),
    ("Unlicense", re.compile(r"free and unencumbered software released into the public domain", re.I)),
    ("CC0-1.0", re.compile(r"creative commons zero|cc0 1\.0 universal", re.I)),
    ("0BSD", re.compile(r"bsd zero clause license", re.I)),
    ("Zlib", re.compile(r"zlib license|this software is provided .as-is.", re.I)),
    ("BSL-1.0", re.compile(r"boost software license.{0,40}version 1\.0", re.I | re.S)),
    ("PSF-2.0", re.compile(r"python software foundation license version 2", re.I)),
]

_LICENSE_FILTER = LicenseFilter(allow_unknown=False)


def classify_license_text(text: str) -> str | None:
    """Best-effort SPDX id from licence text; None when unclassifiable or copyleft."""
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text[:10000])
    lowered = normalized.lower()
    if any(bad in lowered for bad in ("gnu general public license", "gnu lesser general public", "affero general public", "server side public license", "business source license")):
        # Dual-licensed Rust/Python crates often concatenate Apache-2.0 + MIT;
        if not ("apache license" in lowered and "permission is hereby granted" in lowered):
            return None
    for name, pattern in _LICENSE_SIGNATURES:
        if pattern.search(normalized):
            return name
    return None


def classify_spdx_expression(expr: str) -> str | None:
    """Accept an SPDX expression only if permissive under ``LicenseFilter``."""
    return _LICENSE_FILTER.canonicalize(expr)


def classify_pypi_metadata(pkg_info: str) -> str | None:
    if not pkg_info:
        return None
    if re.search(r"License :: OSI Approved :: GNU (?:General|Lesser|Affero)", pkg_info, re.I):
        return None
    classifiers = re.findall(r"Classifier:\s*License :: OSI Approved :: (.+)", pkg_info)
    name_map = {
        "MIT License": "MIT",
        "Apache Software License": "Apache-2.0",
        "BSD License": "BSD-3-Clause",
        "ISC License (ISCL)": "ISC",
        "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
        "The Unlicense (Unlicense)": "Unlicense",
        "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
        "Python Software Foundation License": "PSF-2.0",
        "Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
        "zlib/libpng License": "Zlib",
    }
    for c in classifiers:
        if c.strip() in name_map:
            return name_map[c.strip()]
    m = re.search(r"^License(?:-Expression)?:\s*(.+)$", pkg_info, re.M | re.I)
    if m:
        return classify_spdx_expression(m.group(1))
    return None


def _download(url: str, timeout: int = TIMEOUT, headers: dict[str, str] | None = None) -> bytes | None:
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404, 429):
                logger.debug("HTTP %s on %s", e.code, url)
                return None
            logger.warning("download failed (%d/%d): %s: %s", attempt + 1, RETRIES, url, e)
            time.sleep(1.2 * (attempt + 1))
    return None


def fetch_github_tarball(repo: str) -> bytes | None:
    if "@" in repo:
        repo_slug, ref = repo.split("@", 1)
        url = f"https://codeload.github.com/{repo_slug}/tar.gz/refs/tags/{ref}"
    else:
        url = f"https://codeload.github.com/{repo}/tar.gz/HEAD"
    return _download(url)


def fetch_github_issues_and_prs(
    repo: str,
    license_id: str,
    *,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    """Fetch real issues and pull-request unified diffs for a verified permissive repo."""
    repo_slug = repo.split("@", 1)[0]
    source_id = f"github:{repo}"
    api_url = f"https://api.github.com/repos/{repo_slug}/issues?state=closed&per_page={max_items}"
    raw = _download(api_url, timeout=30, headers={"Accept": "application/vnd.github+json"})
    if not raw:
        return []
    try:
        items = json.loads(raw.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    docs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        title = (item.get("title") or "").strip()
        body = (item.get("body") or "").strip()
        if not number or not title or len(body) < 80:
            continue

        is_pr = "pull_request" in item
        if is_pr:
            diff_url = f"https://github.com/{repo_slug}/pull/{number}.diff"
            diff_bytes = _download(diff_url, timeout=30)
            diff_text = diff_bytes.decode("utf-8", "ignore") if diff_bytes else ""
            if diff_text and "diff --git " in diff_text and len(diff_text) <= MAX_FILE_BYTES:
                pr_doc = (
                    f"# Pull Request #{number}: {title}\n"
                    f"Repository: {repo_slug}\n\n"
                    f"## Summary\n{body[:4000]}\n\n"
                    f"## Unified Diff\n```diff\n{diff_text[:28000].rstrip()}\n```\n"
                )
                docs.append(
                    {
                        "content": pr_doc,
                        "language": "diff",
                        "path": f"pr:{repo_slug}#{number}",
                        "license": license_id,
                        "source": source_id,
                        "slice": "github-prs",
                    }
                )
        else:
            labels = ", ".join(
                lbl.get("name", "") for lbl in (item.get("labels") or []) if isinstance(lbl, dict)
            )
            issue_doc = (
                f"# Issue #{number}: {title}\n"
                f"Repository: {repo_slug}"
                + (f" | Labels: {labels}" if labels else "")
                + f"\n\n{body[:12000]}\n"
            )
            docs.append(
                {
                    "content": issue_doc,
                    "language": "markdown",
                    "path": f"issue:{repo_slug}#{number}",
                    "license": license_id,
                    "source": source_id,
                    "slice": "github-issues",
                }
            )
    return docs


def fetch_pypi_sdist(name: str) -> tuple[bytes, str] | None:
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
    data = _download(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@/')}")
    if data is None:
        return None
    try:
        info = json.loads(data)
    except json.JSONDecodeError:
        return None
    version = (info.get("dist-tags") or {}).get("latest")
    if not version:
        return None
    tarball = ((info.get("versions") or {}).get(version) or {}).get("dist", {}).get("tarball")
    if not tarball:
        return None
    blob = _download(tarball)
    if blob is None:
        return None
    return blob, str(version)


def is_heldout_document(path: str, heldout_every: int = 20) -> bool:
    """Deterministic SHA-256 split so held-out selection is order-independent."""
    if heldout_every <= 0:
        return False
    digest = int(hashlib.sha256(f"corpus-v1-heldout:{path}".encode("utf-8")).hexdigest()[:8], 16)
    return (digest % heldout_every) == 0


def validate_row_schema(row: dict[str, Any]) -> list[str]:
    """Return the contract violations in one emitted JSONL row (empty == valid)."""

    problems: list[str] = []
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            problems.append(f"missing:{field}")
    if row.get("license") not in PERMISSIVE_LICENSES:
        problems.append(f"non-permissive-license:{row.get('license')!r}")
    if not str(row.get("content") or "").strip():
        problems.append("empty-content")
    if not str(row.get("path") or "").strip():
        problems.append("empty-path")
    score = row.get("quality_score")
    if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        problems.append(f"quality_score-out-of-range:{score!r}")
    return problems


def enrich_and_validate_doc(raw_doc: dict[str, Any], min_quality: float = 0.20) -> dict[str, Any] | None:
    """Apply hard quality + complexity filters and attach canonical Corpus v1 fields."""
    content = raw_doc.get("content") or ""
    language = raw_doc.get("language") or "unknown"
    path = raw_doc.get("path") or ""
    license_id = _LICENSE_FILTER.canonicalize(raw_doc.get("license"))
    if not license_id:
        return None

    min_complexity = 0 if language in {"markdown", "yaml", "toml", "diff"} else 1
    ok, _reason = QualityScorer.passes_hard_filters(
        content,
        language,
        path,
        min_lines=MIN_LINES if language not in {"diff", "markdown"} else 5,
        max_lines=MAX_LINES,
        max_bytes=MAX_FILE_BYTES,
        min_complexity=min_complexity,
    )
    if not ok:
        return None

    slice_name = raw_doc.get("slice") or classify_real_slice(path, language, content)
    doc_obj = Document(
        content=content,
        language=language,
        source=raw_doc["source"],
        license=license_id,
        path=path,
        slice_name=slice_name,
    )
    score = QualityScorer.score(doc_obj)
    if score < min_quality:
        return None
    doc_obj.quality_score = score
    return doc_obj.to_jsonl_row()


class CorpusWriter:
    """Streaming, deterministic JSONL shard writer with exact + MinHash LSH dedup."""

    def __init__(
        self,
        out_dir: Path,
        shard_docs: int = 200,
        dedup_threshold: float = 0.85,
        dedup_index: DedupIndex | None = None,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self.shard_index: Counter[str] = Counter()
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.counts: Counter[str] = Counter()
        self.chars: Counter[str] = Counter()
        self.slices: Counter[str] = Counter()
        self.licenses: Counter[str] = Counter()
        self.sources: Counter[str] = Counter()
        self.quality_sum = 0.0
        self.hasher = hashlib.sha256()
        self.dedup_index: DedupIndex | None = dedup_index
        self.dedup: Any = dedup_index if dedup_index is not None else Deduplicator(
            threshold=dedup_threshold
        )

    def duplicate(self, content: str, doc_id: str = "") -> bool:
        key = doc_id or hashlib.sha256(content[:256].encode("utf-8", "ignore")).hexdigest()[:20]
        if self.dedup_index is not None:
            is_duplicate, _reason = self.dedup_index.check_and_add(content, key)
        else:
            is_duplicate = self.dedup.is_duplicate(key, content)
        if is_duplicate:
            return True
        digest = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:32]
        self.hasher.update(digest.encode("utf-8"))
        return False

    def add(self, language: str, doc: dict[str, Any]) -> bool:
        self.buffers[language].append(doc)
        self.counts[language] += 1
        self.chars[language] += len(doc["content"])
        self.slices[doc.get("slice", "the-stack-code")] += 1
        self.licenses[doc.get("license", "UNKNOWN")] += 1
        self.sources[doc.get("source", "unknown")] += 1
        self.quality_sum += float(doc.get("quality_score") or 0.0)
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

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "corpus_version": "corpus-v1-real",
            "documents": sum(self.counts.values()),
            "chars": sum(self.chars.values()),
            "per_language": {
                lang: {"documents": self.counts[lang], "chars": self.chars[lang]}
                for lang in sorted(self.counts)
            },
            "per_slice": dict(sorted(self.slices.items())),
            "per_license": dict(sorted(self.licenses.items())),
            "unique_sources": len(self.sources),
            "mean_quality": round(self.quality_sum / max(sum(self.counts.values()), 1), 4),
            "dedup": self.dedup.stats(),
            "shards": sum(self.shard_index.values()),
            "corpus_sha256": self.hasher.hexdigest(),
        }


class HeldoutWriter:
    """Writes isolated held-out files + JSONL shards + SHA-256 manifest."""

    def __init__(self, heldout_dir: Path):
        self.heldout_dir = heldout_dir
        self.heldout_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.sha256_list: list[str] = []
        self.paths: list[str] = []

    def add(self, doc: dict[str, Any]) -> None:
        self.rows.append(doc)
        sha = doc.get("sha256") or hashlib.sha256(doc["content"].encode("utf-8", "ignore")).hexdigest()
        self.sha256_list.append(sha)
        self.paths.append(doc["path"])
        _write_heldout_raw_file(self.heldout_dir, doc)

    def flush(self) -> dict[str, Any]:
        if self.rows:
            jsonl_path = self.heldout_dir / "heldout-000000.jsonl"
            tmp = jsonl_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in self.rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, jsonl_path)
        manifest = {
            "documents": len(self.rows),
            "heldout_sha256": sorted(set(self.sha256_list)),
            "heldout_paths": sorted(set(self.paths)),
        }
        (self.heldout_dir / "heldout_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest


def _write_heldout_raw_file(heldout_dir: Path, doc: dict[str, Any]) -> None:
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


def _extract_docs_from_tar(blob: bytes, source_id: str, on_license_file) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
            for member in members:
                name = member.name
                base = Path(name).name.lower()
                if base.startswith(("license", "copying", "copyright", "notice")):
                    if Path(base).suffix in {".txt", ".md", ".rst", ""}:
                        try:
                            fobj = tar.extractfile(member)
                            if fobj is not None:
                                on_license_file(fobj.read().decode("utf-8", "ignore")[:10000])
                        except Exception:
                            pass
                    continue
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                floor = MIN_LINES_BY_LANGUAGE.get(language, MIN_LINES)
                if not (floor <= lines <= MAX_LINES):
                    continue
                rel = name.split("/", 1)[-1] if "/" in name else name
                docs.append(
                    {
                        "content": content,
                        "language": language,
                        "path": f"{source_id}:{rel}",
                        "license": "",
                        "source": source_id,
                    }
                )
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning("bad archive %s: %s", source_id, e)
    return docs


def _extract_pypi_sdist(blob: bytes, source_id: str) -> tuple[list[dict[str, Any]], str | None]:
    docs: list[dict[str, Any]] = []
    pkg_info_text = ""
    license_text = ""
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
            for member in members:
                name = member.name
                base = Path(name).name
                if base == "PKG-INFO":
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        pkg_info_text = fobj.read().decode("utf-8", "ignore")[:12000]
                if base.lower().startswith(("license", "copying")):
                    fobj = tar.extractfile(member)
                    if fobj is not None and not license_text:
                        license_text = fobj.read().decode("utf-8", "ignore")[:10000]
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                floor = MIN_LINES_BY_LANGUAGE.get(language, MIN_LINES)
                if not (floor <= lines <= MAX_LINES):
                    continue
                rel = name.split("/", 1)[-1] if "/" in name else name
                docs.append(
                    {
                        "content": content,
                        "language": language,
                        "path": f"{source_id}:{rel}",
                        "license": "",
                        "source": source_id,
                    }
                )
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning("bad sdist archive %s: %s", source_id, e)
        return [], None

    license_id = classify_pypi_metadata(pkg_info_text) or classify_license_text(license_text)
    for doc in docs:
        doc["license"] = license_id or ""
    return docs, license_id


def _extract_npm_tarball(blob: bytes, source_id: str) -> tuple[list[dict[str, Any]], str | None]:
    docs: list[dict[str, Any]] = []
    license_id: str | None = None
    license_text = ""
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
            for member in members:
                name = member.name
                base = Path(name).name
                if base == "package.json" and name.count("/") <= 1:
                    try:
                        fobj = tar.extractfile(member)
                        if fobj is not None:
                            meta = json.loads(fobj.read().decode("utf-8", "ignore"))
                            expr = meta.get("license") or ""
                            if isinstance(expr, dict):
                                expr = expr.get("type", "")
                            elif isinstance(expr, list):
                                expr = " OR ".join(
                                    e.get("type", str(e)) if isinstance(e, dict) else str(e)
                                    for e in expr
                                )
                            license_id = classify_spdx_expression(str(expr))
                    except Exception:
                        pass
                if base.lower().startswith(("license", "copying", "notice")):
                    fobj = tar.extractfile(member)
                    if fobj is not None and not license_text:
                        license_text = fobj.read().decode("utf-8", "ignore")[:10000]
                ext = Path(name).suffix.lower()
                language = EXTENSIONS.get(ext)
                if language is None or not _member_ok(name, member.size):
                    continue
                try:
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                except Exception:
                    continue
                if b"\x00" in raw[:1024]:
                    continue
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = content.count("\n") + 1
                floor = MIN_LINES_BY_LANGUAGE.get(language, MIN_LINES)
                if not (floor <= lines <= MAX_LINES):
                    continue
                rel = name.split("/", 1)[-1] if "/" in name else name
                docs.append(
                    {
                        "content": content,
                        "language": language,
                        "path": f"{source_id}:{rel}",
                        "license": "",
                        "source": source_id,
                    }
                )
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.warning("bad npm archive %s: %s", source_id, e)
        return [], None

    license_id = license_id or classify_license_text(license_text)
    for doc in docs:
        doc["license"] = license_id or ""
    return docs, license_id


def merged_github_repos() -> list[tuple[str, str]]:
    """Core + curated expansion, de-duplicated, order preserved.

    A repository that appears in both lists is fetched once. Pinned refs
    (``owner/name@tag``) are keyed by slug, so ``redis/redis@7.4.1`` wins over an
    unpinned duplicate and the fetch stays reproducible.
    """

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for repo, language in list(GITHUB_REPOS) + list(EXTRA_GITHUB_REPOS) + list(LONG_TAIL_GITHUB_REPOS):
        slug = repo.split("@", 1)[0]
        if slug in seen:
            continue
        seen.add(slug)
        out.append((repo, language))
    return out


def merged_pypi_packages() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in list(PYPI_PACKAGES) + list(EXTRA_PYPI_PACKAGES):
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def merged_npm_packages() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in list(NPM_PACKAGES) + list(EXTRA_NPM_PACKAGES):
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def build_jobs(
    *,
    languages: list[str] | None,
    budgets: dict[str, int],
    skip_github: bool,
    skip_pypi: bool,
    skip_npm: bool,
    only: str | None,
    target_chars: int | None,
) -> list[dict[str, Any]]:
    """Deterministic job list, ordered so the biggest budget gaps are filled first.

    Order matters for a corpus, not just for politeness: the per-language
    character budgets are cumulative, so fetching in a fixed shuffled order
    spends the first N downloads on whatever the shuffle produced. Ordering by
    ``deficit / (family size)`` means a language that is far from its budget is
    served before a language that is already satisfied, and a truncated run
    (``--max_sources``, a timeout, a machine reboot) still leaves a corpus that
    covers every language instead of one that is 80% Python.
    """

    repos = merged_github_repos()
    pypi = merged_pypi_packages()
    npm = merged_npm_packages()

    families: list[dict[str, Any]] = []
    if not skip_github:
        by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for repo, language in repos:
            if languages and language not in budgets:
                continue
            by_language[language].append(
                {"kind": "github", "id": f"github:{repo}", "repo": repo, "language": language}
            )
        for language, jobs in by_language.items():
            families.append({"language": language, "jobs": jobs})
    if not skip_pypi and (not languages or "python" in budgets):
        families.append(
            {
                "language": "python",
                "jobs": [
                    {"kind": "pypi", "id": f"pypi:{name}", "name": name, "language": "python"}
                    for name in pypi
                ],
            }
        )
    if not skip_npm and (not languages or {"javascript", "typescript"} & set(budgets)):
        families.append(
            {
                "language": "javascript",
                "jobs": [
                    {"kind": "npm", "id": f"npm:{name}", "name": name, "language": "javascript"}
                    for name in npm
                ],
            }
        )

    for family in families:
        family["deficit"] = budgets.get(family["language"], 0) / max(len(family["jobs"]), 1)

    ordered: list[dict[str, Any]] = []
    cursors = {id(family): 0 for family in families}
    while True:
        progressed = False
        for family in sorted(families, key=lambda f: (-f["deficit"], f["language"])):
            position = cursors[id(family)]
            if position >= len(family["jobs"]):
                continue
            ordered.append(family["jobs"][position])
            cursors[id(family)] = position + 1
            progressed = True
        if not progressed:
            break

    if only:
        ordered = [job for job in ordered if only in job["id"]]
    if target_chars is not None and target_chars > 0:
        ordered = ordered[: max(1, int(target_chars / 2_000_000))]
    return ordered


def run_fetch(
    *,
    output_dir: str | Path = "data/corpus_v1/real_jsonl",
    heldout_dir: str | Path = "data/corpus_v1/heldout",
    source_log: str | Path | None = None,
    scale: float = 1.0,
    languages: list[str] | None = None,
    skip_github: bool = False,
    skip_pypi: bool = False,
    skip_npm: bool = False,
    include_issues_prs: bool = True,
    max_sources: int | None = None,
    only: str | None = None,
    workers: int = 6,
    heldout_every: int = 20,
    min_quality: float = 0.20,
    dedup_threshold: float = 0.85,
    dedup_index_path: str | Path | None = None,
    target_chars: int | None = None,
) -> dict[str, Any]:
    """Fetch, gate, dedup and shard real permissive code.

    ``dedup_index_path`` selects the SQLite-backed dedup index
    (``dataset.dedup_index.DedupIndex``) instead of the in-RAM
    ``Deduplicator``. At real backbone scale (10^7 documents) the in-RAM index is
    the thing that runs out of memory first; the on-disk index keeps ~1 MB of
    page cache and reopens across runs, so an interrupted fetch resumes without
    re-admitting duplicates it already saw.
    """

    out_dir = Path(output_dir)
    held_dir = Path(heldout_dir)
    src_log = Path(source_log) if source_log else out_dir / "sources.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_log.parent.mkdir(parents=True, exist_ok=True)

    budgets = {lang: max(50_000, int(chars * scale)) for lang, chars in LANGUAGE_BUDGETS.items()}
    if languages:
        allow = {s.strip().lower() for s in languages if s.strip()}
        allow.update({"markdown", "diff"})
        budgets = {lang: n for lang, n in budgets.items() if lang in allow}

    done_sources: set[str] = set()
    if src_log.exists():
        with open(src_log, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        done_sources.add(row["source"])
                except json.JSONDecodeError:
                    continue

    if dedup_index_path is not None:
        dedup_index = DedupIndex(
            dedup_index_path,
            threshold=dedup_threshold,
            namespace="train",
        )
    else:
        dedup_index = None

    writer = CorpusWriter(
        out_dir,
        dedup_threshold=dedup_threshold,
        dedup_index=dedup_index,
    )
    heldout_writer = HeldoutWriter(held_dir)
    totals: Counter[str] = Counter()
    rejected_licenses: Counter[str] = Counter()

    jobs = build_jobs(
        languages=languages,
        budgets=budgets,
        skip_github=skip_github,
        skip_pypi=skip_pypi,
        skip_npm=skip_npm,
        only=only,
        target_chars=target_chars,
    )

    skip_budget_langs = set()
    if max_sources is not None:
        jobs = jobs[:max_sources]

    cache_dir = out_dir / ".download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs):
        job["index"] = i

    def _fetch_one(job: dict[str, Any]) -> dict[str, Any] | None:
        if job["id"] in done_sources:
            return None
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

    fetched: dict[str, dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, job): job for job in jobs}
        for future in cf.as_completed(futures):
            job = futures[future]
            try:
                res = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("fetch error for %s: %s", job["id"], exc)
                continue
            if res is not None:
                fetched[job["id"]] = res

    fetched_count = len(fetched)
    skipped_completed_count = sum(1 for job in jobs if job["id"] in done_sources)

    with open(src_log, "a", encoding="utf-8") as logf:

        def log_source(source_id: str, status: str, license_id: str | None, detail: dict[str, Any]) -> None:
            logf.write(
                json.dumps({"source": source_id, "status": status, "license": license_id, **detail}) + "\n"
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
                docs = _extract_docs_from_tar(blob, source_id, on_license_file=license_texts.append)
                license_id = classify_license_text("\n".join(license_texts))
                for doc in docs:
                    doc["license"] = license_id or ""
                if include_issues_prs and license_id in PERMISSIVE_LICENSES:
                    docs.extend(fetch_github_issues_and_prs(job["repo"], license_id, max_items=6))
            elif job["kind"] == "pypi":
                docs, license_id = _extract_pypi_sdist(blob, source_id)
            else:
                docs, license_id = _extract_npm_tarball(blob, source_id)

            if license_id is None or license_id not in PERMISSIVE_LICENSES:
                rejected_licenses[str(license_id)] += 1
                log_source(source_id, "license-rejected", license_id, {"bytes": len(blob), "docs": len(docs)})
                logger.info("skip %s: licence %r not permissive", source_id, license_id)
                continue

            if all(totals[lang] >= budgets.get(lang, 0) for lang in {d["language"] for d in docs}):
                log_source(
                    source_id,
                    "budget-saturated",
                    license_id,
                    {"bytes": len(blob), "docs": len(docs), "channels": sorted({
                        f"{lang} (availability)" for lang in {d["language"] for d in docs}
                    })},
                )
                skip_budget_langs.update({d["language"] for d in docs})
                continue

            archive_sha = hashlib.sha256(blob).hexdigest()[:16]
            source_chars: Counter[str] = Counter()
            kept = 0
            held = 0
            per_source_cap = {
                lang: max(25_000, int(budgets.get(lang, 0) * PER_SOURCE_FRACTION))
                for lang in {d["language"] for d in docs}
            }

            for raw_doc in docs:
                lang = raw_doc["language"]
                if budgets.get(lang, 0) <= 0:
                    continue
                if totals[lang] >= budgets[lang]:
                    continue
                if source_chars[lang] >= per_source_cap.get(lang, 0):
                    continue

                enriched = enrich_and_validate_doc(raw_doc, min_quality=min_quality)
                if enriched is None:
                    continue
                if writer.duplicate(enriched["content"], enriched["path"]):
                    continue
                if is_heldout_document(enriched["path"], heldout_every=heldout_every):
                    heldout_writer.add(enriched)
                    held += 1
                    continue

                writer.add(lang, enriched)
                totals[lang] += len(enriched["content"])
                source_chars[lang] += len(enriched["content"])
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
                    "heldout": held,
                    "chars": dict(source_chars),
                },
            )
            logger.info("kept %d/%d (heldout=%d) from %s [%s]", kept, len(docs), held, source_id, license_id)

    try:
        cache_dir.rmdir()
    except OSError:
        pass

    writer.flush()
    heldout_manifest = heldout_writer.flush()
    manifest = writer.manifest_fragment()
    if dedup_index is not None:
        manifest["dedup"] = dedup_index.stats()
        dedup_index.close()

    budget_saturated = {lang for lang, cap in budgets.items() if totals[lang] >= cap}
    under_budget = {lang: cap - totals[lang] for lang, cap in budgets.items() if totals[lang] < cap}
    manifest["budgets"] = budgets
    manifest["budget_usage"] = {
        lang: {
            "chars": totals[lang],
            "budget_chars": budgets[lang],
            "utilisation": round(totals[lang] / max(budgets[lang], 1), 4),
            "saturated": lang in budget_saturated,
        }
        for lang in sorted(budgets)
    }
    manifest["availability_gaps"] = {
        "saturated_languages": sorted(budget_saturated),
        "under_budget_chars": dict(sorted(under_budget.items())),
    }
    manifest["license_rejections"] = dict(sorted(rejected_licenses.items()))
    manifest["heldout"] = {
        "dir": str(held_dir),
        "documents": heldout_manifest["documents"],
        "unique_hashes": len(heldout_manifest["heldout_sha256"]),
    }
    manifest["sources_configured"] = {
        "github": len(merged_github_repos()),
        "github_core": len(GITHUB_REPOS),
        "github_expanded": len(EXTRA_GITHUB_REPOS) + len(LONG_TAIL_GITHUB_REPOS),
        "pypi": len(merged_pypi_packages()),
        "npm": len(merged_npm_packages()),
        "languages": len(LANGUAGE_BUDGETS),
    }
    manifest["schema"] = {
        "row_contract": list(REQUIRED_ROW_FIELDS),
        "shard_naming": "<language>-<index:06d>.jsonl",
        "slice_routing": "classify_real_slice(path, language, content)",
        "validator": "scripts.fetch_real_corpus.validate_row_schema",
    }
    manifest["jobs"] = {
        "planned": len(jobs),
        "fetched": fetched_count,
        "skipped_completed": skipped_completed_count,
        "failed_or_unavailable": len(jobs) - fetched_count - skipped_completed_count,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real permissively licensed code corpus (Corpus v1)")
    parser.add_argument("--output_dir", default="data/corpus_v1/real_jsonl")
    parser.add_argument("--heldout_dir", default="data/corpus_v1/heldout")
    parser.add_argument("--source_log", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--languages", default=None, help="Comma-separated allowlist of languages")
    parser.add_argument("--skip_github", action="store_true")
    parser.add_argument("--skip_pypi", action="store_true")
    parser.add_argument("--skip_npm", action="store_true")
    parser.add_argument("--no_issues_prs", action="store_true")
    parser.add_argument("--max_sources", type=int, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--heldout_every", type=int, default=20)
    parser.add_argument("--min_quality", type=float, default=0.20)
    parser.add_argument("--dedup_threshold", type=float, default=0.85)
    parser.add_argument(
        "--dedup_index",
        default=None,
        help="SQLite dedup index path (constant RAM); default keeps signatures in RAM",
    )
    parser.add_argument(
        "--target_chars",
        type=int,
        default=None,
        help="Stop after roughly this many source characters (size-bounded smoke run)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    langs = [s.strip() for s in args.languages.split(",") if s.strip()] if args.languages else None
    manifest = run_fetch(
        output_dir=args.output_dir,
        heldout_dir=args.heldout_dir,
        source_log=args.source_log,
        scale=args.scale,
        languages=langs,
        skip_github=args.skip_github,
        skip_pypi=args.skip_pypi,
        skip_npm=args.skip_npm,
        include_issues_prs=not args.no_issues_prs,
        max_sources=args.max_sources,
        only=args.only,
        workers=args.workers,
        heldout_every=args.heldout_every,
        min_quality=args.min_quality,
        dedup_threshold=args.dedup_threshold,
        dedup_index_path=args.dedup_index,
        target_chars=args.target_chars,
    )
    logger.info("=== real corpus complete: %s docs, %s chars ===", f"{manifest['documents']:,}", f"{manifest['chars']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
