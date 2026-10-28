"""Long-horizon machinery: plan DAGs, milestones, assumptions, memory, scratch."""

import json

import pytest

from agent.memory import MemoryStore
from agent.planning import (
    AssumptionLedger,
    MilestoneCheckpointer,
    PlanGraph,
    _slug,
)
from agent.scratch import ScratchWorkspace




def test_plan_graph_parses_explicit_and_implicit_dependencies():
    graph = PlanGraph.parse(
        [
            "write the parser",
            "[after 1] add tests for the parser",
            "write the CLI",
            "[after 1,2] wire CLI to parser and test end to end",
        ]
    )
    assert [s.id for s in graph.steps] == [1, 2, 3, 4]
    assert graph.steps[0].depends_on == []
    assert graph.steps[1].depends_on == [1]
    assert graph.steps[2].depends_on == [2]
    assert graph.steps[3].depends_on == [1, 2]
    assert graph.steps[3].description == "wire CLI to parser and test end to end"


def test_plan_graph_ready_respects_dependencies():
    graph = PlanGraph.parse(
        ["a", "[after 1] b", "[after 2] c", "[after] independent d"]
    )
    ready = {s.id for s in graph.ready()}
    assert ready == {1, 4}
    graph.mark(1, "done", evidence="test_a passes")
    ready = {s.id for s in graph.ready()}
    assert ready == {2, 4}
    graph.mark(2, "done", "test_b passes")
    ready = {s.id for s in graph.ready()}
    assert ready == {3, 4}


def test_plan_graph_downstream_and_local_replan():
    graph = PlanGraph.parse(
        ["a", "b", "[after 1,2] c", "[after 3] d", "e"]
    )
    assert graph.downstream(1) == {2, 3, 4, 5}
    assert graph.downstream(4) == {5}
    graph2 = PlanGraph.parse(
        [
            {"description": "a", "depends_on": []},
            {"description": "b", "depends_on": []},
            {"description": "c", "depends_on": [1, 2]},
            {"description": "d", "depends_on": [3]},
        ]
    )
    assert graph2.downstream(1) == {3, 4}
    assert graph2.downstream(4) == set()
    graph.mark(1, "done", "x")
    graph.mark(2, "done", "y")
    graph.mark(3, "done", "z")
    graph.mark(4, "done", "w")
    assert graph.completion == 0.8
    affected = graph.reset_subtree(3)
    assert affected == {3, 4, 5}
    assert graph.steps[2].status == "pending"
    assert graph.steps[3].status == "pending"
    assert graph.steps[4].status == "pending"
    assert graph.steps[0].status == "done"
    assert graph.steps[1].status == "done"


def test_plan_graph_serialisation_roundtrip():
    graph = PlanGraph.parse(["a", "[after 1] b"])
    graph.mark(1, "done", "evidence here")
    data = json.loads(json.dumps(graph.to_dict()))
    restored = PlanGraph.from_dict(data)
    assert restored.steps[0].status == "done"
    assert restored.steps[0].evidence == "evidence here"
    assert restored.steps[1].depends_on == [1]




def test_milestone_snapshot_and_rollback_scope(tmp_path):
    work = tmp_path / "repo"
    (work / "src").mkdir(parents=True)
    (work / "src" / "a.py").write_text("x = 1\n")
    (work / "src" / "b.py").write_text("y = 2\n")

    cp = MilestoneCheckpointer(str(work))
    cp.checkpoint("parser works", step=3, files_touched=["src/a.py"], tests_passed=["test_a"])
    assert len(cp.milestones) == 1

    files, milestone = cp.rollback_files(["src/a.py", "src/b.py"])
    assert files == ["src/b.py"]
    assert milestone.label == "parser works"

    cp.checkpoint("second", step=5, files_touched=["src/a.py", "src/b.py"], tests_passed=[])
    restored = cp.restore_snapshot(cp.milestones[0])
    assert "src/a.py" in restored


def test_milestone_serialisation(tmp_path):
    cp = MilestoneCheckpointer(str(tmp_path))
    cp.checkpoint("label one", step=1, files_touched=["a.py"], tests_passed=["t1"])
    data = json.loads(json.dumps(cp.to_dict()))
    restored = MilestoneCheckpointer.from_dict(data, str(tmp_path))
    assert restored.last_verified().label == "label one"




def test_assumption_ledger_lifecycle_and_render():
    ledger = AssumptionLedger()
    a1 = ledger.add("the API returns JSON", step=2)
    a2 = ledger.add("tests run with pytest", step=3)
    assert ledger.add("the API returns JSON") is a1
    ledger.record(a1.id, "confirmed", "read the openapi spec")
    assert ledger.unverified() == [a2]
    ledger.record(a2.id, "refuted", "repo uses unittest")
    text = ledger.render()
    assert "Refuted assumptions" in text
    assert "replan" in text or "wrong" in text
    with pytest.raises(ValueError):
        ledger.record(a2.id, "invalid-status")




def test_memory_store_remember_recall_render(tmp_path):
    mem = MemoryStore(tmp_path / "memory.json")
    mem.remember("test command", "pytest -q tests/unit", "command")
    mem.remember("lesson: import error", "resolved by installing requests first", "lesson")
    mem.remember("bad key", "x" * 10, "fact")
    hits = mem.recall("which test command should I run")
    assert hits and hits[0].key == "test command"
    text = mem.render()
    assert "test command" in text and "hints with history" in text
    mem.save()
    mem2 = MemoryStore(tmp_path / "memory.json")
    assert mem2.get("test command").value == "pytest -q tests/unit"
    assert mem2.get("test command").uses == 1


def test_memory_store_corrupt_file_never_bricks(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json")
    mem = MemoryStore(path)
    assert len(mem) == 0


def test_memory_store_learn_from_session(tmp_path):
    from agent.session import SessionState, FailureRecord

    state = SessionState(task="fix bug")
    state.note_fact("the suite is pytest -q tests/")
    state.failures.append(
        FailureRecord(step=2, tool="run_tests", signature="run_tests|err", error="boom", recovery="installed deps")
    )
    state.tests_passed.append("tests/test_a.py::test_one")
    mem = MemoryStore(tmp_path / "memory.json")
    stored = mem.learn_from_session(state)
    assert stored >= 3
    assert mem.get("fact:the-suite-is-pytest-q-tests").value == "the suite is pytest -q tests/"
    assert "lesson:" in mem.get("lesson:run_tests|err").key
    assert mem.learn_from_session(state) == 0


def test_memory_eviction_is_bounded(tmp_path):
    mem = MemoryStore(tmp_path / "memory.json")
    for i in range(250):
        mem.remember(f"key-{i}", f"value {i}")
    from agent.memory import MAX_MEMORIES
    assert len(mem) <= MAX_MEMORIES




def test_scratch_write_read_index_and_search(tmp_path):
    scratch = ScratchWorkspace(tmp_path / "scratch")
    receipt = scratch.write("algorithm trace", "hypothesis: the bug is an off-by-one in merge()")
    assert receipt["entries"] == 1
    scratch.write("algorithm trace", "refuted: merge is fine, the input is already sorted")
    assert len(scratch) == 1
    body = scratch.read("algorithm trace")
    assert "hypothesis" in body and "refuted" in body
    index = scratch.render_index()
    assert "algorithm-trace" in index and "2 entries" in index
    hits = scratch.search("where did the merge bug go")
    assert hits and hits[0]["topic"] == "algorithm-trace"
    assert scratch.list_topics()[0]["entries"] == 2


def test_scratch_replace_and_prune(tmp_path):
    scratch = ScratchWorkspace(tmp_path / "scratch")
    scratch.write("notes", "first version", mode="append")
    scratch.write("notes", "corrected version", mode="replace")
    body = scratch.read("notes")
    assert "corrected version" in body
    assert "first version" not in body
    from agent.scratch import MAX_TOPIC_CHARS
    big = "x" * (MAX_TOPIC_CHARS + 10)
    scratch.write("big", big, mode="append")
    assert len(scratch.read("big", max_chars=MAX_TOPIC_CHARS * 2)) <= MAX_TOPIC_CHARS + 40


def test_scratch_survives_restart(tmp_path):
    scratch = ScratchWorkspace(tmp_path / "scratch")
    scratch.write("findings", "auth middleware sets the wrong principal")
    again = ScratchWorkspace(tmp_path / "scratch")
    assert "auth middleware" in again.read("findings")
    assert "findings" in again.render_index()


def test_slug_is_filesystem_safe():
    assert _slug("My Notes: ../../etc/passwd?") == "my-notes-etc-passwd"
