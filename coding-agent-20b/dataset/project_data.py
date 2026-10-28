"""Multi-file "generate an entire small project, then prove it runs" rows.

Most instruction data is single-function. This generator produces whole-project
rows from the scaffold registry:

* ``prose request -> plan -> file tree with complete contents -> test command``,
  where the tests in the tree are the ones the starter actually ships and the
  command is the one that actually runs;
* **verification-aware** variants: a row generated from a project whose tests
  pass carries ``verified: true`` only when the caller supplies evidence from a
  real ``quality.verify`` run (never assumed);
* **structure-repair** rows built from ``quality.critic`` findings: the same
  project before/after the fix are a natural DPO pair ("ships with swallowed
  exceptions and no tests" is rejected, "fixed" is chosen).

Provenance and limits are recorded on every row: these are template-derived
seeds designed to teach *multi-file shape, wiring and test discipline*, not to
invent novel application logic. ``docs/DATASET.md`` says so explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from scaffold.registry import get_template, template_ids
from scaffold.spec import ProjectOptions, module_name, slugify, titleize

DEFAULT_REQUESTS = [
    ("fastapi-react", "build me a web app where users sign up and save tasks"),
    ("fastapi-react", "a small API with a dashboard for tracking expenses"),
    ("nextjs-fullstack", "a single Next.js app that stores notes and lists them"),
    ("flask-htmx", "an internal tool to track support tickets, minimal setup"),
    ("cli-tool", "a command line tool that counts lines and words in files"),
    ("discord-bot", "a discord bot that rolls dice and reports uptime"),
    ("static-site", "a one page landing site for my project with dark mode"),
]


@dataclass
class ProjectSample:
    template_id: str
    request: str
    options: ProjectOptions
    files: dict[str, str] = field(default_factory=dict)
    test_command: str = ""
    verified: bool = False
    evidence: dict = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        digest = hashlib.sha256((self.template_id + self.request).encode()).hexdigest()[:12]
        return f"{self.template_id}-{digest}"

    def render_tree(self, max_files: int = 24, max_chars: int = 6000) -> str:
        parts: list[str] = []
        budget = max_chars
        for path in sorted(self.files)[:max_files]:
            content = self.files[path]
            if budget <= 0:
                break
            chunk = f"### {path}\n```\n{content[: max(400, budget)]}\n```\n"
            budget -= len(chunk)
            parts.append(chunk)
        return "".join(parts)

    def to_sft_row(self) -> dict:
        tests = [path for path in sorted(self.files) if "test" in path.lower()]
        plan = "\n".join(
            [
                f"1. Scaffold {self.template_id} ({len(self.files)} files)",
                "2. Implement the requests in the tree below",
                f"3. Add/extend tests: {', '.join(tests[:4]) or 'create tests/'}",
                f"4. Verify: {self.test_command}",
                "5. Review for secrets, swallowed errors and missing tests before shipping",
            ]
        )
        return {
            "id": self.sample_id,
            "prompt": (
                f"Create a project: {self.request}.\n\n"
                "Return a complete, runnable multi-file project (not snippets). Include tests and "
                "the exact commands to run and verify it."
            ),
            "response": (
                f"Plan:\n{plan}\n\nFiles:\n{self.render_tree()}\n"
                f"Run:\n```bash\n{self.test_command}\n```\n"
            ),
            "metadata": {
                "source": "scaffold",
                "template": self.template_id,
                "files": len(self.files),
                "has_tests": bool(tests),
                "test_command": self.test_command,
                "verified": self.verified,
                "verification": self.evidence or {"kind": "not-run"},
                "provenance": "template-derived multi-file seed",
            },
        }

    def to_dpo_row(self, rejected_response: str, reason: str) -> dict:
        row = self.to_sft_row()
        return {
            "id": row["id"] + "-dpo",
            "prompt": row["prompt"],
            "chosen": row["response"],
            "rejected": rejected_response,
            "metadata": {**row["metadata"], "kind": "project-dpo", "rejected_reason": reason},
        }


class ProjectDatasetGenerator:
    """Build project samples from the scaffold registry (deterministic)."""

    def __init__(self, requests: list[tuple[str, str]] | None = None):
        self.requests = requests or list(DEFAULT_REQUESTS)

    def sample(
        self,
        template_id: str,
        request: str,
        *,
        name: str | None = None,
        use_auth: bool = False,
        database: str | None = None,
    ) -> ProjectSample:
        spec = get_template(template_id)
        project_name = slugify(name or template_id.replace("-", " "))
        options = ProjectOptions(
            name=project_name,
            display_name=titleize(name or template_id.replace("-", " ")),
            module=module_name(project_name),
            description=request[:200],
            port=spec.default_port,
            use_auth=bool(spec.supports_auth and use_auth),
            database=database or ("sqlite" if spec.supports_database else "none"),
        )
        files = spec.render(options)
        return ProjectSample(
            template_id=spec.id,
            request=request,
            options=options,
            files=files,
            test_command=spec.test_command,
        )

    def samples(self, include_auth: bool = True) -> list[ProjectSample]:
        out: list[ProjectSample] = []
        for template_id, request in self.requests:
            if template_id not in template_ids():
                continue
            out.append(self.sample(template_id, request))
            spec = get_template(template_id)
            if include_auth and spec.supports_auth:
                out.append(
                    self.sample(
                        template_id,
                        request + " (with user accounts and login)",
                        name=template_id + " auth",
                        use_auth=True,
                    )
                )
        return out

    def mark_verified(self, sample: ProjectSample, test_run: dict) -> ProjectSample:
        """Attach real evidence to a sample (only call this with an actual run)."""

        if not test_run or not test_run.get("ok"):
            return sample
        sample.verified = True
        sample.evidence = {
            "kind": "test-run",
            "command": test_run.get("command"),
            "passed": len(test_run.get("passed") or []),
            "duration": test_run.get("duration"),
        }
        return sample

    def to_sft_rows(self, samples: list[ProjectSample] | None = None) -> list[dict]:
        return [sample.to_sft_row() for sample in (samples or self.samples())]

    def structure_dpo_rows(self, sample: ProjectSample) -> list[dict]:
        """Contrast a complete project with common *structural* failure modes."""

        rows: list[dict] = []
        rows.append(
            sample.to_dpo_row(
                "Here is the core file (other files omitted for brevity):\n```python\n"
                + next(iter(sample.files.values()))[:1500]
                + "\n```",
                "single-file answer with the rest omitted: the project does not run",
            )
        )
        rows.append(
            sample.to_dpo_row(
                "Plan:\n1. Build it\n2. Ship it\n\nFiles:\n(main file only)\n",
                "no tests, no run command, no verification step",
            )
        )
        return rows

    def stats(self) -> dict:
        samples = self.samples()
        return {
            "samples": len(samples),
            "templates": sorted({sample.template_id for sample in samples}),
            "avg_files": round(sum(len(sample.files) for sample in samples) / max(len(samples), 1), 1),
            "sft_rows": len(self.to_sft_rows(samples)),
            "dpo_rows": sum(len(self.structure_dpo_rows(sample)) for sample in samples),
        }


def write_project_dataset(output_dir: str | Path, generator: ProjectDatasetGenerator | None = None) -> dict:
    """Write JSONL shards (sft.jsonl / dpo.jsonl) and return a manifest."""

    import json

    generator = generator or ProjectDatasetGenerator()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = generator.samples()
    sft_rows = generator.to_sft_rows(samples)
    dpo_rows = [row for sample in samples for row in generator.structure_dpo_rows(sample)]

    (output / "sft.jsonl").write_text("\n".join(json.dumps(row) for row in sft_rows) + "\n", encoding="utf-8")
    (output / "dpo.jsonl").write_text("\n".join(json.dumps(row) for row in dpo_rows) + "\n", encoding="utf-8")
    manifest = {
        "sft": len(sft_rows),
        "dpo": len(dpo_rows),
        "templates": sorted({sample.template_id for sample in samples}),
        "source": "scaffold templates (deterministic, no network)",
        "verified_samples": sum(1 for sample in samples if sample.verified),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
