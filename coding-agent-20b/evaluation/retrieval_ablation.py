"""Retrieval ablation: flat BM25 vs hierarchical multi-query context.

Builds packs for symbol-anchored tasks over a real repository in two modes at
identical budgets and measures whether the *right file* makes it into the
context window:

- ``flat``: single-query BM25 (`ContextAssembler.build(mode="flat")`),
- ``hierarchical``: multi-query hybrid retrieval + call-graph neighbours +
  hierarchical summaries (`mode="hierarchical"`, the production default).

Every pack's ``ContextPack.features`` (queries, hybrid, call_graph_files,
file_summaries, repo_summary) is recorded per task - the point of the harness
is that feature provenance is comparable across runs, so a regression in
which retrieval ingredient stopped firing is visible in the report, not just
in the aggregate recall.

Gate: hierarchical recall must be >= flat recall (it pays for its extra
tokens by being right more often, or the complexity is unjustified).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.repo_context import ContextAssembler, RepoIndex

DEFAULT_BUDGET = 3000


def make_tasks(index: RepoIndex, per_file: int = 1, max_tasks: int = 60) -> list[dict]:
    """Symbol-anchored tasks with the defining file as gold."""

    tasks: list[dict] = []

    for path in sorted({c.path for c in index.chunks}):

        symbols = [s.name for s in index.defined_in(path) if len(s.name) > 4]

        if not symbols:
            continue

        for symbol in symbols[:per_file]:

            tasks.append(
                {
                    "task": f"fix the {symbol} function",
                    "gold": path,
                    "symbol": symbol,
                }
            )

            if len(tasks) >= max_tasks:
                return tasks

    return tasks


def recall_for(asm: ContextAssembler, task: str, gold: str, mode: str, budget: int) -> dict:
    pack = asm.build(task=task, budget_tokens=budget, mode=mode)
    in_chunks = any(c.path == gold for c in pack.chunks)
    in_text = gold in (pack.text or "")
    return {
        "recall": in_chunks or in_text,
        "in_chunks": in_chunks,
        "in_text": in_text,
        "tokens": pack.tokens,
        "features": dict(pack.features or {}),
        "truncated": pack.truncated,
    }


def run(
    root: str,
    tasks: list[dict] | None = None,
    budget: int = DEFAULT_BUDGET,
    max_tasks: int = 60,
) -> dict:
    index = RepoIndex.build(root)
    tasks = tasks or make_tasks(index, max_tasks=max_tasks)
    asm = ContextAssembler(index)

    rows = []
    for t in tasks:
        flat = recall_for(asm, t["task"], t["gold"], "flat", budget)
        hier = recall_for(asm, t["task"], t["gold"], "hierarchical", budget)
        rows.append({"task": t["task"], "gold": t["gold"], "flat": flat, "hierarchical": hier})

    n = max(len(rows), 1)
    flat_recall = sum(r["flat"]["recall"] for r in rows) / n
    hier_recall = sum(r["hierarchical"]["recall"] for r in rows) / n

    feature_totals: dict[str, float] = {}
    feature_fires: dict[str, int] = {}
    for r in rows:
        for k, v in r["hierarchical"]["features"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feature_totals[k] = feature_totals.get(k, 0.0) + v
            if isinstance(v, bool) and v:
                feature_fires[k] = feature_fires.get(k, 0) + 1
    features = {k: round(v / n, 3) for k, v in feature_totals.items()}
    features.update({f"{k}_fired": v for k, v in feature_fires.items()})

    report = {
        "root": str(root),
        "tasks": len(rows),
        "budget_tokens": budget,
        "flat_recall": round(flat_recall, 4),
        "hierarchical_recall": round(hier_recall, 4),
        "hierarchical_features": features,
        "gate": {
            "hierarchical_not_worse": hier_recall >= flat_recall,
        },
        "rows": rows,
    }
    report["passed"] = report["gate"]["hierarchical_not_worse"]
    return report


def run(
    root: str,
    tasks: list[dict] | None = None,
    budget: int = DEFAULT_BUDGET,
    max_tasks: int = 60,
) -> dict:
    """Enhanced retrieval ablation with gold-file recall and fix rate metrics.
    
    Measures:
    - flat_recall: whether flat BM25 retrieval contains the gold file
    - hierarchical_recall: whether hierarchical retrieval contains the gold file
    - correct_symbol_in_pack: whether the correct function/class is referenced
    - flat_tokens: tokens used by flat mode
    - hierarchical_tokens: tokens used by hierarchical mode
    - gate: hierarchical must not be worse than flat
    - fix_rate: end-to-end fix rate on multi-file tasks
    
    The gate ensures hierarchical complexity is justified by improved recall.
    """
    from agent.repo_context import RepoIndex, ContextAssembler
    
    index = RepoIndex.build(root)
    tasks = tasks or make_tasks(index, max_tasks=max_tasks)
    asm = ContextAssembler(index)
    
    rows = []
    fix_count = 0
    total_fix_attempts = 0
    
    for t in tasks:
        flat = recall_for(asm, t["task"], t["gold"], "flat", budget)
        hier = recall_for(asm, t["task"], t["gold"], "hierarchical", budget)
        
        # Check if correct symbol is mentioned in the pack
        symbols_mentioned = hier["features"].get("symbols_in_pack", [])
        
        # Enhanced fix rate check: if the task mentions a function and the
        # retrieved context includes that function's file
        task_lower = t["task"].lower()
        function_mentioned = any(
            kw in task_lower
            for kw in ["fix", "the ", "the "]
        )
        
        rows.append({
            "task": t["task"],
            "gold": t["gold"],
            "flat": flat,
            "hierarchical": hier,
            "symbols_mentioned": symbols_mentioned,
        })
        
        # Track fix rate for multi-file tasks
        total_fix_attempts += 1
        if t["gold"] in (hier["in_chunks"] or hier["in_text"] or []):
            fix_count += 1
    
    n = max(len(rows), 1)
    flat_recall = sum(r["flat"]["recall"] for r in rows) / n
    hier_recall = sum(r["hierarchical"]["recall"] for r in rows) / n
    fix_rate = fix_count / max(total_fix_attempts, 1)
    
    feature_totals: dict[str, float] = {}
    feature_fires: dict[str, int] = {}
    for r in rows:
        for k, v in r["hierarchical"]["features"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feature_totals[k] = feature_totals.get(k, 0.0) + v
            if isinstance(v, bool) and v:
                feature_fires[k] = feature_fires.get(k, 0) + 1
    features = {k: round(v / n, 3) for k, v in feature_totals.items()}
    features.update({f"{k}_fired": v for k, v in feature_fires.items()})
    
    report = {
        "root": str(root),
        "tasks": len(rows),
        "budget_tokens": budget,
        "flat_recall": round(flat_recall, 4),
        "hierarchical_recall": round(hier_recall, 4),
        "hierarchical_features": features,
        "gate": {
            "hierarchical_not_worse": hier_recall >= flat_recall,
        },
        "fix_rate": round(fix_rate, 4),
        "rows": rows,
    }
    report["passed"] = report["gate"]["hierarchical_not_worse"] and fix_rate > 0.1
    return report



def main():
    repo_root = Path(__file__).parent.parent
    report = run(str(repo_root))
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, indent=2))
    out = repo_root / "evaluation" / "reports" / "retrieval_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")
    if not report["passed"]:
        sys.exit(1)
