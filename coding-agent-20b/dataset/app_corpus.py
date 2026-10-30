"""The app-generation corpus: request -> working product, and the near misses.

This is the data behind the product claim "describe an app, get a running,
tasteful, deployable full-stack app". One blueprint yields four kinds of row:

* ``app-sft`` - a natural request, the plan, every file, and the command that
  proves it runs. Honest about evidence: a row says ``verified`` only when this
  generator actually ran the tests in a temp directory.
* ``app-preference`` - the same request with the good answer and a plausible
  bad one (2018 UI, authenticated-but-not-authorised, relations without foreign
  keys), plus the measured reason it lost.
* ``app-repair`` - broken project + real failure output -> diagnosis -> minimal
  diff -> green. This is the trace the repair loop should imitate.
* ``app-taste`` - why the interface is shaped the way it is: tokens, states,
  responsive rules, microcopy. Text, not vibes.

Rows are deterministic and offline. Blueprints held out by
:func:`scaffold.blueprints.split` are excluded from generation, so the
evaluation suite measures generalisation rather than recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path

from quality.critic import CodeCritic
from quality.taste import taste_of
from scaffold.blueprint import render_blueprint
from scaffold.blueprint_variants import (
    apply_broken_owner_scope,
    apply_incomplete_entities,
    apply_legacy_ui,
)
from scaffold.blueprints import AppBlueprint, all_blueprints, split

RUN_COMMAND = "python -m pytest -q"

SYSTEM_PROMPT = (
    "You are a senior product engineer. You build complete, runnable full-stack "
    "applications from a plain-language request: typed schema, auth with real "
    "row-level isolation, a REST API, a responsive UI with deliberate design "
    "tokens, and tests that prove the core flows. You never claim success "
    "without running the verification command, and you never hide a failure."
)

VARIANT_LABELS = {
    "legacy-ui": "the layout is a fixed-width 2018 template with no states or dark mode",
    "broken-auth": "authentication checks the token but queries are not scoped to the viewer",
    "incomplete-entities": "relations are loose id columns with no foreign key or cascade",
}


@dataclass
class TestEvidence:
    """What a real test run said, captured verbatim for the row."""

    ok: bool = False
    command: str = RUN_COMMAND
    summary: str = ""
    failures: list[str] = field(default_factory=list)
    exit_code: int | None = None
    kind: str = "not-run"

    def to_dict(self) -> dict:
        """Serialise for metadata."""
        return {
            "kind": self.kind,
            "ok": self.ok,
            "command": self.command,
            "summary": self.summary,
            "failures": self.failures,
            "exit_code": self.exit_code,
        }

    def failure_signal(self, limit: int = 1600) -> str:
        """The part a model should actually read when repairing."""
        text = self.summary + "\n" + "\n".join(self.failures)
        return text.strip()[:limit]


def _write_tree(files: dict[str, str], root: Path) -> None:
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")



EXPECTED_FAILURES = {
    "legacy-ui": "taste review: no token layer, no dark scheme, no phone breakpoint, no empty state",
    "broken-auth": (
        "FAILED tests/test_api.py::test_isolation_between_users - "
        "AssertionError: the second user's row is visible to the first"
    ),
    "incomplete-entities": (
        "FAILED tests/test_schema.py::test_relations_foreign_key - "
        "AssertionError: PRAGMA foreign_key_list has no CASCADE entry"
    ),
}


def _expected_failure(variant: str) -> TestEvidence:
    """Static evidence for fast generation; ``--verify`` replaces it with a real run."""
    return TestEvidence(
        ok=False,
        summary="not executed during generation (use --verify for recorded output)",
        failures=[EXPECTED_FAILURES[variant]],
        kind="expected",
    )

def _run_tests(files: dict[str, str], timeout: int = 180) -> TestEvidence:
    """Run the generated project's own tests in a throwaway directory."""
    with tempfile.TemporaryDirectory(prefix="app-corpus-") as tmp:
        root = Path(tmp)
        _write_tree(files, root)
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return TestEvidence(kind="run-error", summary=str(exc))
        output = (run.stdout or "") + (run.stderr or "")
        failures = [
            line
            for line in output.splitlines()
            if line.startswith(("FAILED", "ERROR")) or "AssertionError" in line or "Error" in line
        ][:8]
        summary_lines = [line for line in output.splitlines() if line.strip()][-1:] or [""]
        return TestEvidence(
            ok=run.returncode == 0,
            summary=summary_lines[0][:500],
            failures=failures,
            exit_code=run.returncode,
            kind="pytest",
        )


def _priority(path: str) -> tuple[int, str]:
    if path.startswith("tests/") or "/test" in path:
        return 90, path
    order = (
        "README",
        "requirements",
        "schema.sql",
        "db.py",
        "schemas.py",
        "store.py",
        "api.py",
        "main.py",
        "index.html",
        "config.js",
        "app.js",
        "app.css",
    )
    for index, token in enumerate(order):
        if token in path:
            return index, path
    return len(order), path


def files_block(files: dict[str, str], max_files: int = 34, max_chars: int = 30000) -> str:
    """Render a file map as fenced blocks, reading the most important first."""
    parts: list[str] = []
    budget = max_chars
    for path in sorted(files, key=_priority)[:max_files]:
        content = files[path]
        if budget <= 0:
            break
        chunk = "### " + path + "\n```\n" + content[: max(200, budget)] + "\n```\n"
        budget -= len(chunk)
        parts.append(chunk)
    return "".join(parts)


def _base_prompt(blueprint: AppBlueprint) -> str:
    return (
        blueprint.prompt.strip().rstrip(".")
        + ".\n\n"
        + "Build it as a complete, runnable product: a typed schema, the API, a "
        + "responsive UI, tests, and the commands to run and verify it. Pick sensible "
        + "defaults; do not ask me follow-up questions."
    )


def _plan(blueprint: AppBlueprint, file_count: int) -> str:
    entities = ", ".join(entity.table for entity in blueprint.entities)
    steps = [
        "1. Model the domain: " + entities + " with typed fields, constraints and relations",
        "2. Persistence: schema + migrations, parameterised queries, viewer scoping",
    ]
    if blueprint.auth:
        steps.append("3. Auth: register/login, hashed passwords, token checks, row isolation")
    else:
        steps.append("3. Access: single-owner mode, explicit demo viewer, no fake login")
    steps.append("4. API: CRUD, search/pagination, stats, CSV, validation errors")
    steps.append("5. UI: design tokens, responsive shell, table/form, empty/loading/error states")
    steps.append("6. Tests: happy path, isolation, validation, relations, feature flow")
    steps.append("7. Deploy: Dockerfile, health check, one-command run (" + str(file_count) + " files)")
    return "\n".join(steps)


def _response(files: dict[str, str], plan: str, evidence: TestEvidence | None) -> str:
    parts = ["Plan:\n" + plan + "\n\nFiles:\n" + files_block(files)]
    if evidence is not None and evidence.kind == "pytest":
        verdict = "passing" if evidence.ok else "failing"
        parts.append(
            "\nVerification (actual run):\n```\n$ "
            + RUN_COMMAND
            + "\n"
            + evidence.summary
            + f"\n# {verdict}\n```\n"
        )
        if not evidence.ok:
            parts.append("Failures:\n```\n" + evidence.failure_signal() + "\n```\n")
    else:
        parts.append(
            "\nRun:\n```bash\npip install -r requirements.txt\n" + RUN_COMMAND + "\nuvicorn <module>.main:app\n```\n"
        )
    return "".join(parts)


def _diff(before: dict[str, str], after: dict[str, str], limit: int = 8000) -> str:
    chunks: list[str] = []
    for path in sorted(set(before) | set(after)):
        old = before.get(path, "").splitlines(keepends=True)
        new = after.get(path, "").splitlines(keepends=True)
        if old == new:
            continue
        chunks.extend(
            unified_diff(old, new, fromfile="a/" + path, tofile="b/" + path, n=2)
        )
    return "".join(chunks)[:limit]


@dataclass
class AppSample:
    """One rendered app plus the measurements taken on it."""

    blueprint: AppBlueprint
    files: dict[str, str]
    evidence: TestEvidence = field(default_factory=TestEvidence)
    critic_score: int = 100
    taste_score: int = 100
    taste_findings: list[str] = field(default_factory=list)

    @property
    def sample_id(self) -> str:
        digest = hashlib.sha256(self.blueprint.id.encode()).hexdigest()[:10]
        return "app-" + self.blueprint.id + "-" + digest

    @property
    def metadata(self) -> dict:
        """Provenance attached to every row built from this sample."""
        return {
            "source": "blueprint",
            "blueprint": self.blueprint.id,
            "title": self.blueprint.title,
            "tier": self.blueprint.tier,
            "feature": self.blueprint.feature,
            "auth": self.blueprint.auth,
            "entities": [entity.table for entity in self.blueprint.entities],
            "flows": self.blueprint.flows,
            "files": len(self.files),
            "critic_score": self.critic_score,
            "taste_score": self.taste_score,
            "verified": self.evidence.ok,
            "verification": self.evidence.to_dict(),
            "provenance": "rendered blueprint; tests executed locally" if self.evidence.kind == "pytest" else "rendered blueprint; tests not run",
        }


class AppCorpus:
    """Build SFT, preference, repair and taste rows from the blueprint catalogue."""

    def __init__(
        self,
        blueprints: list[AppBlueprint] | None = None,
        *,
        train_only: bool = True,
        verify: bool = False,
        limit: int | None = None,
    ):
        source = blueprints or (split("app")[0] if train_only else all_blueprints())
        self.blueprints = source[:limit] if limit else source
        self.verify = verify

    def sample(self, blueprint: AppBlueprint) -> AppSample:
        """Render one app and measure it with the same gates production uses."""
        project = render_blueprint(blueprint)
        sample = AppSample(blueprint=blueprint, files=project.files)
        if self.verify:
            sample.evidence = _run_tests(project.files)
        with tempfile.TemporaryDirectory(prefix="app-critic-") as tmp:
            root = Path(tmp)
            _write_tree(project.files, root)
            sample.critic_score = CodeCritic().review(root).score
            taste = taste_of(root)
            sample.taste_score = taste.score
            sample.taste_findings = [item.rule for item in taste.findings[:6]]
        return sample

    def samples(self) -> list[AppSample]:
        """Every selected app, rendered once."""
        return [self.sample(blueprint) for blueprint in self.blueprints]

    def sft_rows(self, samples: list[AppSample] | None = None) -> list[dict]:
        """Prompt -> complete project -> verification."""
        rows: list[dict] = []
        for sample in samples or self.samples():
            plan = _plan(sample.blueprint, len(sample.files))
            rows.append(
                {
                    "id": sample.sample_id + "-sft",
                    "system": SYSTEM_PROMPT,
                    "prompt": _base_prompt(sample.blueprint),
                    "response": _response(sample.files, plan, sample.evidence),
                    "metadata": {**sample.metadata, "kind": "app-sft", "split": "train"},
                }
            )
        return rows

    def _variant(self, sample: AppSample, variant: str) -> dict[str, str]:
        if variant == "legacy-ui":
            return apply_legacy_ui(sample.files, sample.blueprint.title)
        if variant == "broken-auth":
            return apply_broken_owner_scope(sample.files, sample.blueprint, auth=sample.blueprint.auth)
        return apply_incomplete_entities(sample.files, sample.blueprint)

    def preference_rows(self, samples: list[AppSample] | None = None) -> list[dict]:
        """Chosen = shipped default, rejected = one measured downgrade."""
        rows: list[dict] = []
        for sample in samples or self.samples():
            rows.append(self._preference(sample, "legacy-ui"))
            if sample.blueprint.auth:
                rows.append(self._preference(sample, "broken-auth"))
            if any(entity.relations for entity in sample.blueprint.entities):
                rows.append(self._preference(sample, "incomplete-entities"))
        return rows

    def _preference(self, sample: AppSample, variant: str) -> dict:
        rejected_files = self._variant(sample, variant)
        rejected_evidence = _run_tests(rejected_files) if self.verify else TestEvidence()
        good = _response(sample.files, _plan(sample.blueprint, len(sample.files)), sample.evidence)
        bad = _response(
            rejected_files,
            _plan(sample.blueprint, len(rejected_files)),
            rejected_evidence if rejected_evidence.kind == "pytest" else None,
        )
        return {
            "id": sample.sample_id + "-" + variant + "-dpo",
            "system": SYSTEM_PROMPT,
            "prompt": _base_prompt(sample.blueprint),
            "chosen": good,
            "rejected": bad,
            "metadata": {
                **sample.metadata,
                "kind": "app-preference",
                "variant": variant,
                "reason": VARIANT_LABELS[variant],
                "rejected_verification": rejected_evidence.to_dict(),
                "rejected_taste": taste_of_tree(rejected_files),
                "kind_of_failure": "visual" if variant == "legacy-ui" else "behaviour",
            },
        }

    def repair_rows(self, samples: list[AppSample] | None = None) -> list[dict]:
        """Broken app + real failure evidence -> diagnosis -> minimal diff."""
        rows: list[dict] = []
        for sample in samples or self.samples():
            variants = ["legacy-ui"]
            if sample.blueprint.auth:
                variants.append("broken-auth")
            if any(entity.relations for entity in sample.blueprint.entities):
                variants.append("incomplete-entities")
            for variant in variants:
                rows.append(self._repair(sample, variant))
        return rows

    def _repair(self, sample: AppSample, variant: str) -> dict:
        broken = self._variant(sample, variant)
        evidence = _run_tests(broken) if self.verify else _expected_failure(variant)
        taste = taste_of_tree(broken)
        if variant == "legacy-ui":
            finding = (
                "Taste review failed: "
                + ", ".join(item["rule"] for item in taste.get("findings", [])[:4])
                + f" (score {taste['score']}/100). "
                "Behavioural tests pass; the product surface does not meet the bar."
            )
        else:
            finding = evidence.failure_signal() or "the verification run failed"
        diff = _diff(broken, sample.files)
        changed = _changed_paths(broken, sample.files)
        excerpt = files_block({path: broken[path] for path in changed[:4]}, max_files=4, max_chars=9000)
        prompt = (
            _base_prompt(sample.blueprint)
            + "\n\nA previous attempt is close but does not ship. Evidence:\n```\n"
            + (finding or "no failure output was captured")
            + "\n```\n\nFiles in scope:\n"
            + excerpt
            + "\nDiagnose the cause, make the smallest fix, and state the command that proves it."
        )
        response = (
            "Diagnosis: " + _diagnosis(variant) + "\n\n"
            + "Patch:\n```diff\n" + diff + "```\n\n"
            + "Why this fixes it: " + _why(variant) + "\n\n"
            + "Verify: `" + RUN_COMMAND + "` must report all tests passing; then "
            + "`python -m app_suite`-style checks (health, taste, deploy plan) must also pass.\n"
        )
        return {
            "id": sample.sample_id + "-" + variant + "-repair",
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "response": response,
            "metadata": {
                **sample.metadata,
                "kind": "app-repair",
                "variant": variant,
                "changed_files": changed,
                "evidence": evidence.to_dict(),
                "failure": (finding or "")[:1200],
                "taste": taste,
            },
        }

    def taste_rows(self, samples: list[AppSample] | None = None) -> list[dict]:
        """The design-decision trace behind the default interface."""
        rows: list[dict] = []
        for sample in (samples or self.samples())[:8]:
            rows.append(self._taste_row(sample))
        return rows

    def _taste_row(self, sample: AppSample) -> dict:
        blueprint = sample.blueprint
        css = sample.files.get("web/app.css", "")
        config = sample.files.get("web/config.js", "")
        tokens = "\n".join(css.split(":root", 1)[-1].split("}", 1)[0].splitlines()[:26])
        choices = [
            "Palette: one accent (" + blueprint.accent + ") mixed into surfaces, with success/warn/danger reserved for state.",
            "Type: system-ui at a 14/16/20/28 scale; tabular numbers in tables so columns line up.",
            "Space: 4px base scale, page gutters that collapse below 720px; no fixed pixel container.",
            "Layout: sidebar -> top bar on phones; table rows become labelled cards instead of a horizontal scroll.",
            "States: skeleton rows while loading, a domain-specific empty sentence with the primary action, inline field errors, a polite toast.",
            "Access: lang, viewport, skip link, visible focus rings, labels bound with for=, aria-live on errors, reduced-motion honoured.",
            "Copy: the empty state names the object and the next action; buttons are verbs; errors say what to change.",
        ]
        response = (
            "Design decisions for " + blueprint.title + ":\n"
            + "\n".join("- " + item for item in choices)
            + "\n\nToken layer (web/app.css):\n```css\n:root"
            + tokens
            + "\n```\n\nEmpty-state copy (web/config.js):\n```js\n"
            + _excerpt(config, "empty", 700)
            + "\n```\n\nReject: fixed 960px containers, Helvetica defaults, tables that overflow on a "
            "phone, blank screens on first load, and success messages that hide failures.\n"
        )
        return {
            "id": sample.sample_id + "-taste",
            "system": SYSTEM_PROMPT,
            "prompt": (
                "You are shipping " + blueprint.title + " (" + blueprint.prompt + "). "
                "Explain the interface decisions that make it feel like a product rather than a "
                "template: hierarchy, palette, spacing, responsive behaviour, empty/loading/error "
                "states, accessibility and microcopy. Then show the tokens and one state."
            ),
            "response": response,
            "metadata": {
                **sample.metadata,
                "kind": "app-taste",
                "dimensions": taste_of_tree(sample.files).get("dimensions", {}),
                "taste_score": sample.taste_score,
            },
        }

    def stats(self) -> dict:
        """Counts for the mixture, docs and the manifest."""
        train, held_out = split("app")
        return {
            "blueprints": len(self.blueprints),
            "train_blueprints": len(train),
            "held_out_blueprints": len(held_out),
            "tiers": _count_by(self.blueprints, lambda item: item.tier),
            "features": _count_by(self.blueprints, lambda item: item.feature),
            "verify": self.verify,
            "sft_rows_estimate": len(self.blueprints),
            "preference_rows_estimate": sum(
                1 + int(item.auth) + int(any(entity.relations for entity in item.entities))
                for item in self.blueprints
            ),
            "repair_rows_estimate": sum(
                1 + int(item.auth) + int(any(entity.relations for entity in item.entities))
                for item in self.blueprints
            ),
            "taste_rows_estimate": min(8, len(self.blueprints)),
        }

    def write(self, output_dir: str | Path) -> dict:
        """Write JSONL files plus a manifest that records how they were made."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        samples = self.samples()
        payloads = {
            "app-sft.jsonl": self.sft_rows(samples),
            "app-preference.jsonl": self.preference_rows(samples),
            "app-repair.jsonl": self.repair_rows(samples),
            "app-taste.jsonl": self.taste_rows(samples),
        }
        manifest = {
            "kind": "app-corpus",
            "verify": self.verify,
            "stats": self.stats(),
            "files": {},
        }
        for name, rows in payloads.items():
            path = out / name
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            manifest["files"][name] = len(rows)
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def _excerpt(text: str, needle: str, limit: int = 700) -> str:
    index = text.lower().find(needle.lower())
    if index < 0:
        return text[:limit]
    start = max(0, index - 120)
    return text[start : start + limit]


def _diagnosis(variant: str) -> str:
    return {
        "legacy-ui": (
            "The behaviour is fine, but the interface was assembled from a fixed-width 2018 "
            "template: no token layer, no dark scheme, no phone breakpoint, no loading or empty "
            "state, and labels that are not associated with inputs."
        ),
        "broken-auth": (
            "`_scope` returns `1 = 1`, so every query returns every account's rows. The token is "
            "verified, which hides the bug until a second user exists: the isolation test fails "
            "with the second user's row in the first user's list."
        ),
        "incomplete-entities": (
            "The relation is just a text column: the schema has no foreign key, no cascade, and "
            "the store never checks the parent exists. One bogus id is accepted and deleting a "
            "parent leaves orphaned children."
        ),
    }[variant]


def _why(variant: str) -> str:
    return {
        "legacy-ui": (
            "It restores the token layer and the responsive shell, and adds the states a user "
            "hits before the happy path: loading, empty, validation and a toast."
        ),
        "broken-auth": (
            "Scoping the predicate to `user_id` (plus explicit shares where the product has "
            "them) makes the row count depend on the viewer, which the isolation test asserts."
        ),
        "incomplete-entities": (
            "The foreign key gives the database the invariant, cascade makes deletion total, and "
            "`_require_parent` turns an unknown id into a 404 instead of an orphan."
        ),
    }[variant]


def taste_of_tree(files: dict[str, str]) -> dict:
    """Taste report for an in-memory file map (no temp directory needed twice)."""
    with tempfile.TemporaryDirectory(prefix="app-taste-") as tmp:
        root = Path(tmp)
        _write_tree(files, root)
        return taste_of(root).to_dict()


def _count_by(items: list[AppBlueprint], key) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        value = str(key(item))
        counts[value] = counts.get(value, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    """Write the corpus to disk; ``--verify`` runs every project's tests."""
    parser = argparse.ArgumentParser(description="Generate the app-generation corpus")
    parser.add_argument("--out", default="data/app-corpus", help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="only the first N blueprints")
    parser.add_argument("--verify", action="store_true", help="run tests and record real evidence")
    parser.add_argument("--all-splits", action="store_true", help="include held-out blueprints")
    parser.add_argument("--stats", action="store_true", help="print stats and exit")
    args = parser.parse_args(argv)

    corpus = AppCorpus(
        train_only=not args.all_splits,
        verify=args.verify,
        limit=args.limit,
    )
    if args.stats:
        print(json.dumps(corpus.stats(), indent=2, sort_keys=True))
        return 0
    manifest = corpus.write(args.out)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
