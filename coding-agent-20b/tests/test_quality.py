"""Quality loop: static review, test execution/parsing, iteration, test writing.

These tests assert behaviour a user can feel:

* the critic finds the bugs that actually bite (bare except, hardcoded secret,
  no tests, mutable default, insecure file handling) and *says where*;
* a green project scores high and ``Review.ok`` is true;
* failures from pytest are parsed into a usable signal (not a stack-trace wall);
* the verification loop stops at green, stops when the fixer stops helping, and
  records a trace that can become training data;
* generated tests import the module from any cwd and actually pass.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from quality import CodeCritic, VerificationLoop, humanize, run_tests
from quality.testgen import generate_tests, write_generated_tests
from quality.verify import detect_test_command, parse_output, SAFE_COMMANDS


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def clean_project(root: Path) -> Path:
    write(
        root,
        "pkg/core.py",
        '''
        """Small, tidy module."""

        from __future__ import annotations


        def add(left: int, right: int) -> int:
            """Add two numbers."""

            return left + right


        def slugify(text: str) -> str:
            """Lower-case words joined by dashes."""

            return "-".join(text.lower().split())
        ''',
    )
    write(
        root,
        "tests/test_core.py",
        '''
        from pkg.core import add, slugify


        def test_add():
            assert add(2, 3) == 5


        def test_slugify():
            assert slugify("Hello There") == "hello-there"
        ''',
    )
    write(root, "pkg/__init__.py", "")
    write(root, ".gitignore", ".env\n__pycache__/\n")
    write(root, "README.md", "# demo\n\nRun `python -m pytest`.\n")
    return root


def test_clean_project_scores_well(tmp_path):
    review = CodeCritic().review(clean_project(tmp_path))
    assert review.score >= 80, review.render()
    assert review.ok
    assert review.critical == []


def test_critic_finds_bare_except_and_says_where(tmp_path):
    write(
        tmp_path,
        "pkg/risky.py",
        '''
        """Risky."""

        from __future__ import annotations


        def parse(text):
            try:
                return int(text)
            except:
                return 0
        ''',
    )
    review = CodeCritic().review(tmp_path, include_tests=True)
    hits = [f for f in review.findings if "except" in f.message.lower()]
    assert hits, review.render()
    assert hits[0].path == "pkg/risky.py"
    assert hits[0].line > 0
    assert hits[0].severity in {"critical", "high"}
    assert "risky.py" in review.fix_prompt()


def test_critic_finds_hardcoded_secret_and_committed_env(tmp_path):
    write(
        tmp_path,
        "app/settings.py",
        '''
        """Settings."""

        from __future__ import annotations

        API_KEY = "sk-live-abcdef1234567890abcdef1234567890"
        ''',
    )
    write(tmp_path, ".env", "SECRET=1\n")
    review = CodeCritic().review(tmp_path, include_tests=True)
    messages = " ".join(f.message.lower() for f in review.findings)
    assert "secret" in messages or "key" in messages
    assert any(".env" in f.message for f in review.findings)


def test_critic_finds_mutable_default_and_unsafe_constructs(tmp_path):
    write(
        tmp_path,
        "pkg/bad.py",
        '''
        """Bad patterns."""

        import subprocess


        def collect(items=[]):
            items.append(1)
            return items


        def run(cmd):
            return subprocess.run(cmd, shell=True)
        ''',
    )
    review = CodeCritic().review(tmp_path, include_tests=True)
    text = " ".join(f.message.lower() for f in review.findings)
    assert "mutable" in text or "default" in text
    assert "shell" in text


def test_project_rules_require_tests_and_gitignore(tmp_path):
    write(
        tmp_path,
        "main.py",
        '''
        """App."""

        from __future__ import annotations


        def main() -> int:
            """Entry point."""

            return 0
        ''',
    )
    review = CodeCritic().review(tmp_path)
    text = " ".join(f.message.lower() for f in review.findings)
    assert "test" in text
    assert "git-ignored" in text or "gitignore" in text


def test_review_serialises_for_ui(tmp_path):
    review = CodeCritic().review(clean_project(tmp_path))
    data = review.to_dict()
    assert {"score", "ok", "counts", "findings", "summary"} <= set(data)
    assert isinstance(data["findings"], list)


def test_detect_test_command_prefers_explicit_config(tmp_path):
    write(tmp_path, "pyproject.toml", "[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    command = detect_test_command(tmp_path)
    assert command and any("pytest" in part for part in command)


def test_detect_test_command_returns_none_without_tests(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    assert detect_test_command(tmp_path) is None


def test_run_tests_passes_and_reports_counts(tmp_path):
    clean_project(tmp_path)
    run = run_tests(tmp_path, [sys.executable, "-m", "pytest", "-q"])
    assert run.ok and run.exit_code == 0
    assert run.passed and any("2 tests" in item for item in run.passed)


def test_run_tests_parses_failures(tmp_path):
    clean_project(tmp_path)
    write(
        tmp_path,
        "tests/test_broken.py",
        '''
        from pkg.core import add


        def test_add_is_wrong():
            assert add(2, 2) == 5
        ''',
    )
    run = run_tests(tmp_path, [sys.executable, "-m", "pytest", "-q"])
    assert not run.ok
    assert run.failures
    failure = run.failures[0]
    assert "test_add_is_wrong" in failure["test"]
    signal = run.failure_signal()
    assert "test_add_is_wrong" in signal and "assert" in signal.lower()


def test_run_tests_refuses_arbitrary_commands(tmp_path):
    run = run_tests(tmp_path, ["rm", "-rf", "/"])
    assert not run.ok and run.exit_code == 126
    assert "refusing" in run.stderr
    assert "pytest" in SAFE_COMMANDS.pattern


def test_parse_output_handles_no_tests_exit_code(tmp_path):
    run = run_tests(tmp_path, [sys.executable, "-m", "pytest", "-q"], timeout=60)
    assert run.exit_code in {0, 5}
    assert isinstance(parse_output(run).failures, list)


def test_verification_loop_stops_when_green(tmp_path):
    clean_project(tmp_path)
    loop = VerificationLoop(tmp_path, command=[sys.executable, "-m", "pytest", "-q"])
    outcome = loop.run()
    assert outcome["ok"] and outcome["rounds"] == 0
    assert loop.to_trace()["green"] is True


def test_verification_loop_fixes_until_green(tmp_path):
    clean_project(tmp_path)
    (tmp_path / "pkg" / "core.py").write_text(
        textwrap.dedent(
            '''
            """Broken on purpose."""

            from __future__ import annotations


            def add(left: int, right: int) -> int:
                """Add two numbers."""

                return left - right


            def slugify(text: str) -> str:
                """Lower-case words joined by dashes."""

                return "-".join(text.lower().split())
            '''
        ),
        encoding="utf-8",
    )

    def fixer(prompt: str, run) -> bool:
        assert "test_add" in prompt
        source = (tmp_path / "pkg" / "core.py").read_text()
        (tmp_path / "pkg" / "core.py").write_text(source.replace("left - right", "left + right"))
        return True

    loop = VerificationLoop(tmp_path, command=[sys.executable, "-m", "pytest", "-q"], fix_fn=fixer)
    outcome = loop.run()
    assert outcome["ok"], outcome["run"].render()
    assert outcome["rounds"] == 1
    trace = loop.to_trace()
    assert trace["green"] is True
    assert trace["rounds"] == 1
    assert trace["attempts"][0]["changed"] is True
    assert "failure" in trace and trace["failure"]


def test_verification_loop_stops_when_fix_changes_nothing(tmp_path):
    clean_project(tmp_path)
    (tmp_path / "tests" / "test_fail.py").write_text(
        "def test_never():\n    assert False\n", encoding="utf-8"
    )
    loop = VerificationLoop(
        tmp_path,
        command=[sys.executable, "-m", "pytest", "-q"],
        fix_fn=lambda prompt, run: False,
        max_rounds=3,
    )
    outcome = loop.run()
    assert not outcome["ok"]
    assert outcome["rounds"] == 1
    assert loop.to_trace()["green"] is False


def test_generate_tests_are_valid_and_pass(tmp_path):
    clean_project(tmp_path)
    outcome = write_generated_tests(tmp_path)
    assert outcome["written"], outcome
    for rel in outcome["written"]:
        compile((tmp_path / rel).read_text(), rel, "exec")
    run = run_tests(tmp_path, [sys.executable, "-m", "pytest", "-q"])
    assert run.ok, run.render()


def test_generate_tests_for_fastapi_routes(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    write(
        tmp_path,
        "demo/app.py",
        '''
        """Tiny API."""

        from __future__ import annotations

        from fastapi import FastAPI

        app = FastAPI()


        @app.get("/ping")
        def ping() -> dict:
            """Ping."""

            return {"ok": True}
        ''',
    )
    text = generate_tests("demo/app.py", (tmp_path / "demo" / "app.py").read_text())
    assert "TestClient" in text and "/ping" in text
    compile(text, "generated.py", "exec")


def test_generate_tests_handles_empty_module():
    result = generate_tests("pkg/empty.py", "")
    assert result is None or "def test_" in result


def test_write_generated_tests_does_not_overwrite(tmp_path):
    clean_project(tmp_path)
    first = write_generated_tests(tmp_path)
    assert first["written"]
    target = tmp_path / first["written"][0]
    target.write_text("# mine\n", encoding="utf-8")
    again = write_generated_tests(tmp_path)
    assert again["written"] == []
    assert target.read_text() == "# mine\n"


def test_humanize_translates_common_failures():
    cases = {
        "ModuleNotFoundError: No module named 'fastapi'": "fastapi",
        "PermissionError: [Errno 13] Permission denied": "permission",
        "SyntaxError: invalid syntax": "syntax",
        "error: address already in use": "port",
    }
    for raw, needle in cases.items():
        error = humanize(raw)
        assert error.title and error.plain
        assert needle in (error.plain + " ".join(error.next_steps) + error.title).lower()


def test_humanize_falls_back_and_strips_ansi():
    error = humanize("\x1b[31msomething odd happened\x1b[0m")
    assert "\x1b" not in error.raw
    assert error.next_steps
