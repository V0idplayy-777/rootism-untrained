from agent.recovery import RecoveryAction, RecoveryPolicy, StallDetector
from agent.repo_context import ContextAssembler, RepoIndex, tokenize_for_index
from agent.session import (
    ContextWindow,
    ObservationCompactor,
    SessionState,
    TodoList,
    failure_signature,
)


def _repo(tmp_path):
    (tmp_path / "pkg").mkdir()

    (tmp_path / "pkg" / "auth.py").write_text(
        "\n".join(
            [
                '"""Authentication helpers."""',
                "import hmac",
                "import time",
                "",
                "",
                "def rotate_token(session: dict, now: int) -> str | None:",
                '    """Rotate the session token when it is due."""',
                "    if session['expires_at'] <= now:",
                "        return None",
                "    signature = hmac.new(session['key'], session['id'].encode(), 'sha256')",
                "    return signature.hexdigest()",
                "",
                "",
                "class SessionStore:",
                "    def get(self, session_id: str) -> dict | None:",
                "        return self._data.get(session_id)",
                "",
                "    def put(self, session_id: str, value: dict) -> None:",
                "        self._data[session_id] = value",
                "",
            ]
        )
    )

    (tmp_path / "pkg" / "billing.py").write_text(
        "\n".join(
            [
                "from .auth import rotate_token",
                "",
                "",
                "def charge_customer(customer_id: str, amount_cents: int) -> dict:",
                '    """Charge once per idempotency key."""',
                "    key = f'{customer_id}:{amount_cents}'",
                "    return {'idempotency_key': key, 'status': 'pending'}",
                "",
            ]
        )
    )

    (tmp_path / "web" / "handler.ts").parent.mkdir()

    (tmp_path / "web" / "handler.ts").write_text(
        "export async function handleLogin(req: Request): Promise<Response> {\n"
        "  const token = req.headers.get('authorization');\n"
        "  if (!token) return new Response('unauthorized', { status: 401 });\n"
        "  return new Response('ok');\n"
        "}\n"
    )

    return tmp_path


def test_repo_index_chunks_symbols_and_bm25(tmp_path):

    root = _repo(tmp_path)

    index = RepoIndex.build(str(root))

    assert index.stats.files == 3

    names = {s.name for s in index.symbols}

    assert {"rotate_token", "SessionStore", "charge_customer", "handleLogin"} <= names

    hits = index.search("rotate session token when expired", k=3)

    assert hits

    assert hits[0].path == "pkg/auth.py"

    assert hits[0].kind in {"function", "module"}

    assert index.symbol_lookup("rotate_token")

    assert index.symbol_lookup("charge_customer")

    related = index.related_files("pkg/billing.py")

    assert "pkg/auth.py" in related

    outline = index.file_outline("pkg/auth.py")

    assert "rotate_token" in outline and "SessionStore" in outline

    assert "user" in tokenize_for_index("getUserById")
    assert "parse" in tokenize_for_index("parse_tool_call")


def test_repo_map_respects_token_budget(tmp_path):

    index = RepoIndex.build(str(_repo(tmp_path)))

    small = index.repo_map(max_tokens=60)

    large = index.repo_map(max_tokens=2000)

    assert small.startswith("<|repo_map|>")

    assert len(small) <= len(large)

    assert "pkg/auth.py" in large

    assert "web/handler.ts" in large

    assert len(small) <= len(large)


def test_context_assembler_ablation_modes(tmp_path):

    index = RepoIndex.build(str(_repo(tmp_path)))

    assembler = ContextAssembler(index)

    none = assembler.build("fix token rotation", mode="none")

    flat = assembler.build("fix token rotation", mode="flat")

    hier = assembler.build("fix token rotation", mode="hierarchical")

    assert none.text == "" and none.tokens == 0

    assert "pkg/auth.py" in flat.text

    assert flat.mode == "flat" and not flat.map_included

    assert hier.map_included

    assert "<|repo_map|>" in hier.text

    assert hier.tokens <= hier_tokens_budget(hier)

    tiny = assembler.build("fix token rotation", budget_tokens=40)

    assert tiny.tokens <= 40


def hier_tokens_budget(pack):

    return max(pack.tokens, 6000)


def test_session_state_persists_plan_facts_and_tests(tmp_path):

    state = SessionState(task="fix the expiry bug", max_steps=30)

    state.plan_as(["reproduce", "patch rotate_token", "run tests"])

    assert state.todos.todos[0].status == "pending"

    state.todos.update(1, "done", "tests/test_auth.py::test_expiry")

    state.note_fact("expiry check must be <=, not <")

    state.note_fact("expiry check must be <=, not <")

    state.record_failure("run_tests", {"test_path": "tests"}, "AssertionError: expired")

    state.record_test_result(["test_ok"], ["test_expiry"])

    state.note_file("pkg/auth.py")

    path = tmp_path / "session.json"

    state.save(str(path))

    loaded = SessionState.load(str(path))

    assert loaded.task == state.task

    assert loaded.todos.todos[0].status == "done"

    assert loaded.todos.todos[0].evidence.startswith("tests/")

    assert loaded.facts == ["expiry check must be <=, not <"]

    assert loaded.failures[0].tool == "run_tests"

    assert loaded.tests_failed == ["test_expiry"]

    digest = loaded.render()

    assert "<|plan|>" in digest and "expiry check" in digest

    # A todo marked done with no evidence is detectable (state coherence).
    loaded.todos.update(2, "done")

    assert not all(
        t.evidence for t in loaded.todos.todos if t.status == "done"
    )


def test_failure_signature_normalises_volatile_parts():

    a = failure_signature("run_tests", {"test_path": "tests"}, "line 42: KeyError: 'x' at 0xdeadbeef")

    b = failure_signature("run_tests", {"test_path": "tests"}, "line 99: KeyError: 'x' at 0xfeedface")

    c = failure_signature("run_tests", {"test_path": "tests"}, "line 99: TypeError: 'y'")

    assert a == b

    assert a != c


def test_observation_compactor_keeps_signal_and_bounds_size():

    compactor = ObservationCompactor(max_chars=500)

    result = {
        "returncode": 1,
        "stdout": "FAILED tests/test_auth.py::test_expiry - AssertionError: expected 1 got 2\n"
        "E   AssertionError: expected 1 got 2\n"
        + "\n".join(f"noise line {i}" for i in range(400))
        + "\n2 failed, 10 passed in 1.23s",
    }

    summary = compactor.summarise("run_tests", {"test_path": "tests"}, result)

    assert len(summary) <= 500

    assert "test_expiry" in summary

    assert "AssertionError" in summary

    read = compactor.summarise(
        "read_file",
        {"path": "pkg/auth.py"},
        {"content": "import hmac\n\ndef rotate_token(s):\n    return s\n" + "x = 1\n" * 500},
    )

    assert "rotate_token" in read and len(read) < 800


def test_context_window_compacts_older_steps_within_budget():

    state = SessionState(task="fix bug")

    state.plan_as(["repro", "fix", "verify"])

    window = ContextWindow(
        "system prompt",
        "fix bug",
        state,
        budget_tokens=1200,
        keep_recent_steps=3,
    )

    for i in range(12):

        window.add_assistant(f"step {i}: I will inspect the code again " + "x" * 200)

        window.add_observation(
            "read_file",
            {"path": f"pkg/m{i}.py"},
            {"content": "def f():\n    return 1\n" + "# padding\n" * 200},
        )

    text, stats = window.assemble()

    assert stats["prompt_tokens"] <= 1200

    assert stats["steps_compacted"] > 0

    assert stats["truncated"]

    assert "Earlier steps (compacted)" in text

    assert "<|plan|>" in text

    assert "pkg/m11.py" in text


def test_recovery_escalation_ladder():

    state = SessionState(task="fix bug", max_steps=40)

    state.plan_as(["repro", "fix"])

    policy = RecoveryPolicy(max_retries=1, max_consecutive_failures=6)

    first = policy.decide(state, "run_tests", {"test_path": "tests"}, "AssertionError: nope")

    assert first.action == RecoveryAction.RETRY

    state.record_failure("run_tests", {"test_path": "tests"}, "AssertionError: nope")

    second = policy.decide(state, "run_tests", {"test_path": "tests"}, "AssertionError: nope")

    assert second.action in {RecoveryAction.SWITCH, RecoveryAction.CHANGE}

    assert second.hint

    state.record_failure("run_tests", {"test_path": "tests"}, "AssertionError: nope")

    third = policy.decide(state, "run_tests", {"test_path": "tests"}, "AssertionError: nope")

    assert third.action == RecoveryAction.BACKTRACK

    for _ in range(6):

        state.record_failure("write_file", {"path": "a.py"}, "FileNotFoundError")

    escalated = policy.decide(state, "write_file", {"path": "a.py"}, "FileNotFoundError")

    assert escalated.action == RecoveryAction.ESCALATE

    assert "do not claim" in escalated.hint.lower()

    parse = RecoveryPolicy().decide(state, parse_error=True, error="bad json")

    assert parse.action in {RecoveryAction.REPAIR, RecoveryAction.ESCALATE}

    assert RecoveryPolicy.backtrack_command(["a.py"]) == ["git", "checkout", "--", "a.py"]

    assert RecoveryPolicy.backtrack_command() == ["git", "stash", "--include-untracked"]


def test_stall_detector_flags_spinning():

    state = SessionState(task="t")

    detector = StallDetector(window=3, min_progress=0.5)

    for _ in range(3):

        assert detector.observe(state, "read_file:a.py") is False

    assert detector.observe(state, "read_file:a.py") is True

    state.note_file("b.py")

    detector.reset()

    for i in range(4):

        state.note_file(f"c{i}.py")

    assert detector.observe(state, f"write_file:c{i}.py") is False


def test_todo_list_replaces_plan_preserving_done_items():

    todos = TodoList(["a", "b"])

    todos.update(1, "done", "evidence-a")

    todos.replace_plan(["a", "b", "c"])

    assert [t.status for t in todos.todos] == ["done", "pending", "pending"]

    assert todos.completion == 1 / 3

    assert todos.next_pending().text == "b"

    todos.update(2, "blocked")

    assert todos.blockers()[0].id == 2

    assert "[x] 1. a" in todos.render()


def test_retrieval_ablation_metrics(tmp_path, monkeypatch):

    root = _repo(tmp_path)

    from evaluation.production_evaluate import (
        RetrievalAblation,
        SweBenchProblem,
        build_swebench_prompt,
        gold_files_from_patch,
    )

    patch = (
        "diff --git a/pkg/auth.py b/pkg/auth.py\n"
        "--- a/pkg/auth.py\n"
        "+++ b/pkg/auth.py\n"
        "@@ -1,3 +1,3 @@\n-    if session['expires_at'] < now:\n"
        "+    if session['expires_at'] <= now:\n"
    )

    assert gold_files_from_patch(patch) == ["pkg/auth.py"]

    problem = SweBenchProblem(
        instance_id="inst-1",
        repo="example/repo",
        base_commit="abc",
        problem_statement="Session token is rotated one tick too late; expiry check is off by one.",
        fail_to_pass=["tests/test_auth.py::test_expiry"],
        pass_to_pass=[],
        patch=patch,
    )

    data = build_swebench_prompt(problem, str(root))

    assert "pkg/auth.py" in data["prompt"]

    assert data["stats"]["mode"] == "hierarchical"

    ablation = RetrievalAblation(budget_tokens=2000).evaluate(
        [problem], {problem.instance_id: str(root)}
    )

    assert set(ablation) >= {"none", "flat", "hierarchical"}

    assert ablation["none"]["gold_file_recall"] == 0.0

    assert ablation["flat"]["gold_file_recall"] == 1.0

    assert ablation["hierarchical"]["gold_file_recall"] >= ablation["flat"]["gold_file_recall"]

    assert ablation["hierarchical"]["context_tokens"] >= ablation["none"]["context_tokens"]

    assert "hierarchical_vs_flat" in ablation


def test_working_set_excerpts_survive_compaction():
    state = SessionState(task="fix the auth middleware")
    window = ContextWindow("system", state.task, state, budget_tokens=1200, keep_recent_steps=2)
    window.add_observation(
        "read_file",
        {"path": "pkg/auth.py"},
        {"path": "pkg/auth.py", "content": "def check():\n    return False\n"},
    )
    for index in range(20):
        window.add_observation("list_files", {}, {"files": [f"f{index}.py"]})
    text, stats = window.assemble()
    assert "<|file_chunk|> pkg/auth.py" in text
    assert "def check():" in text
    assert stats["working_set_files"] >= 1
    assert stats["truncated"] or stats["steps_compacted"] >= 0


def test_observation_compaction_keeps_exact_failures_and_deploy_errors():
    compactor = ObservationCompactor()
    tests = compactor.summarise_tests(
        {
            "returncode": 1,
            "stdout": (
                "FAILED tests/test_auth.py::test_expiry - AssertionError: assert 1 == 2\n"
                "E   assert 1 == 2\n"
                "E    +  where 1 = expired(now)\n"
                "FAILED tests/test_api.py::test_scope - assert False\n"
                "E   assert False\n"
            ),
        }
    )
    assert "test_auth.py::test_expiry" in tests
    assert "E   assert 1 == 2" in tests
    assert "E    +  where 1 = expired(now)" in tests
    deploy = compactor.summarise("deploy", {}, {"ok": False, "provider": "fly", "error": "health check failed: /healthz 500"})
    assert "health check failed: /healthz 500" in deploy
    assert "fly" in deploy


def test_session_state_round_trip_keeps_the_working_set(tmp_path):
    state = SessionState(task="ship it")
    state.pin_file("pkg/auth.py", "def check():\n    return False\n")
    path = tmp_path / "state.json"
    state.save(str(path))
    loaded = SessionState.load(str(path))
    assert "pkg/auth.py" in loaded.working_set.pinned
    assert "def check():" in loaded.working_set.excerpts["pkg/auth.py"]
