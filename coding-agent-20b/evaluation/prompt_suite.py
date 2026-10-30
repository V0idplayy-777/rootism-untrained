"""Held-out, out-of-catalogue prompt suite for the blueprint resolver.

The app suite in :mod:`evaluation.app_suite` uses the same prompts the
blueprints were authored from, so it cannot tell whether the *router* works on
wording it has never seen. This suite is deliberately disjoint: 30 paraphrases
(including sibling-app ambiguity) and 10 prompts that no shipped blueprint
should claim, where the honest answer is to fall back to a starter or refuse.

Run it directly or from tests::

    python -m evaluation.prompt_suite
    python -m evaluation.prompt_suite --json evaluation/reports/prompt_suite.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from scaffold.blueprints import match_blueprint_scored

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PromptCase:
    """One prompt plus the set of answers a good resolver may give."""

    prompt: str
    expected: tuple[str, ...] = ()
    kind: str = "catalogue"  # catalogue | out-of-catalogue
    note: str = ""

    @property
    def is_out_of_catalogue(self) -> bool:
        return self.kind == "out-of-catalogue"


CATALOGUE_CASES: tuple[PromptCase, ...] = (
    PromptCase("a kanban board for our sprint work", ("task-board", "sprint-board")),
    PromptCase("track job applications through each interview stage", ("job-applications", "hiring-pipeline")),
    PromptCase("manage rental properties and repair requests", ("property-maintenance",)),
    PromptCase("a reading log for the books I finish", ("reading-list",)),
    PromptCase("a meal planner with recipes and shopping lists", ("family-groceries", "recipe-box")),
    PromptCase("keep a list of my plants and when to water them", ("plant-care",)),
    PromptCase("a habit tracker with streaks", ("habit-tracker", "habit-dashboard")),
    PromptCase("a budget tracker by category", ("budget-categories",)),
    PromptCase("a support ticket queue for a small team", ("support-queues", "support-board")),
    PromptCase("a patient appointment system for a clinic", ("clinic-slots",)),
    PromptCase("a warehouse stock count tool", ("warehouse-stock",)),
    PromptCase("a workout log", ("workout-log", "gym-tracker")),
    PromptCase("a travel itinerary builder", ("trip-itinerary",)),
    PromptCase("a task list shared with my partner", ("shared-tasks",)),
    PromptCase("a class roster for a teacher", ("class-roster",)),
    PromptCase("a freelance invoice tracker", ("client-invoices",)),
    PromptCase("a restaurant menu manager", ("restaurant-menu",)),
    PromptCase("a farm crop rotation planner", ("farm-fields",)),
    PromptCase("a bug tracker for QA", ("bug-triage",)),
    PromptCase("a wedding guest list", ("event-guests",)),
    PromptCase("a vehicle maintenance log", ("vehicle-service",)),
    PromptCase("a course lesson planner", ("course-lessons",)),
    PromptCase("a book club reading schedule", ("book-club",)),
    PromptCase("a volunteer shift scheduler", ("volunteer-roster",)),
    PromptCase("an inventory for a small shop", ("inventory-simple",)),
    PromptCase("a simple CRM for freelancers", ("shared-clients", "sales-pipeline")),
    PromptCase("book a meeting room for our office", ("meeting-rooms",)),
    PromptCase("salon appointments for a hair studio", ("salon-appointments",)),
    PromptCase("a grocery shopping list for the family", ("family-groceries",)),
    PromptCase("desk booking for a coworking space", ("desk-bookings",)),
)

OUT_OF_CATALOGUE_CASES: tuple[PromptCase, ...] = (
    PromptCase("a distributed GPU scheduler with Kubernetes operators", kind="out-of-catalogue"),
    PromptCase("a compiler backend for WebAssembly", kind="out-of-catalogue"),
    PromptCase("a real-time collaborative video editor", kind="out-of-catalogue"),
    PromptCase("a blockchain explorer", kind="out-of-catalogue"),
    PromptCase("a machine learning experiment tracker", kind="out-of-catalogue"),
    PromptCase("an operating system kernel module", kind="out-of-catalogue"),
    PromptCase("a ray tracer in Rust", kind="out-of-catalogue"),
    PromptCase("a database engine with B-tree indexes", kind="out-of-catalogue"),
    PromptCase("a stock trading bot with live market data", kind="out-of-catalogue"),
    PromptCase("a game engine editor", kind="out-of-catalogue"),
)

TUNING_CASES: tuple[PromptCase, ...] = CATALOGUE_CASES + OUT_OF_CATALOGUE_CASES

# Measured once, after the matcher was frozen. These prompts are intentionally
# kept out of every tuning loop: report the number, do not "fix" the cases.
# Catalogue accepted 15/23 (65%), out-of-catalogue refused 11/11 (100%).
HOLDOUT_CASES: tuple[PromptCase, ...] = (
    PromptCase("a shared whiteboard of sticky notes for the team", ("shared-notes", "notes-app")),
    PromptCase("a list of suppliers and their phone numbers", ("contact-book",)),
    PromptCase("keep a tally of my daily coffee spending", ("expense-log", "budget-categories")),
    PromptCase("a seating plan for the wedding", ("event-guests",)),
    PromptCase("an editorial plan for next month's newsletters", ("content-calendar",)),
    PromptCase("follow orders from checkout to doorstep", ("order-fulfilment",)),
    PromptCase("sign out cameras and laptops to staff", ("equipment-rentals",)),
    PromptCase("who is on shift at the cafe this week", ("team-roster", "volunteer-roster", "desk-bookings")),
    PromptCase("a company directory with job titles", ("org-structure", "team-roster")),
    PromptCase("monthly objectives for each squad", ("shared-goals",)),
    PromptCase("a stocktake of the stockroom", ("inventory-simple", "warehouse-stock")),
    PromptCase("a list of all the plants in the office and their watering days", ("plant-care",)),
    PromptCase("a log of gym classes I attend", ("gym-tracker", "workout-log")),
    PromptCase("a list of dishes we serve and what's in them", ("restaurant-menu", "recipe-ingredients")),
    PromptCase("a list of the flats we manage and their tenants", ("property-maintenance",)),
    PromptCase("a register of the children in my class", ("class-roster",)),
    PromptCase("track the books I lend to friends", ("reading-list",)),
    PromptCase("a queue of customer emails for the support team", ("support-queues", "support-board", "shared-inbox")),
    PromptCase("track candidate applications for each role", ("hiring-pipeline", "job-applications")),
    PromptCase("a weekly rota for the volunteers", ("volunteer-roster",)),
    PromptCase("a project timeline with milestones", ("project-milestones",)),
    PromptCase("a kanban for our design team", ("task-board", "sprint-board")),
    PromptCase("a list of invoices waiting to be paid", ("client-invoices",)),
    PromptCase("a booking sheet for the tennis courts", kind="out-of-catalogue"),
    PromptCase("a spell checker for old English", kind="out-of-catalogue"),
    PromptCase("a PDF renderer in C", kind="out-of-catalogue"),
    PromptCase("a packet sniffer for wifi", kind="out-of-catalogue"),
    PromptCase("a recommender system for movies at scale", kind="out-of-catalogue"),
    PromptCase("a programming language interpreter", kind="out-of-catalogue"),
    PromptCase("a blockchain smart contract auditor", kind="out-of-catalogue"),
    PromptCase("a real-time audio mixer", kind="out-of-catalogue"),
    PromptCase("a container runtime", kind="out-of-catalogue"),
    PromptCase("a satellite orbit simulator", kind="out-of-catalogue"),
    PromptCase("a TCP congestion control simulator", kind="out-of-catalogue"),
)

SPLITS: dict[str, tuple[PromptCase, ...]] = {
    "tuning": TUNING_CASES,
    "holdout": HOLDOUT_CASES,
    "all": TUNING_CASES + HOLDOUT_CASES,
}

CASES: tuple[PromptCase, ...] = TUNING_CASES


@dataclass
class PromptResult:
    """Outcome for one case."""

    prompt: str
    kind: str
    expected: tuple[str, ...]
    matched: str | None
    score: int
    exact: bool
    accepted: bool
    refused: bool
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "kind": self.kind,
            "expected": list(self.expected),
            "matched": self.matched,
            "score": self.score,
            "exact": self.exact,
            "accepted": self.accepted,
            "refused": self.refused,
            "evidence": self.evidence,
        }


def resolve(prompt: str, minimum: int = 3):
    """Return the matcher decision exposed to the app builder."""

    return match_blueprint_scored(prompt, minimum=minimum)


def evaluate_prompts(cases: Iterable[PromptCase] = CASES, minimum: int = 3, split: str = "tuning") -> dict:
    """Score routing accuracy, exact accuracy and refusal honesty."""

    results: list[PromptResult] = []
    for case in cases:
        match = resolve(case.prompt, minimum=minimum)
        matched = match.blueprint.id if match else None
        if case.is_out_of_catalogue:
            accepted = matched is None
            exact = matched is None
        else:
            accepted = matched in case.expected
            exact = bool(case.expected) and matched == case.expected[0]
        results.append(
            PromptResult(
                prompt=case.prompt,
                kind=case.kind,
                expected=case.expected,
                matched=matched,
                score=match.score if match else 0,
                exact=exact,
                accepted=accepted,
                refused=matched is None,
                evidence=list(match.evidence) if match else [],
            )
        )

    catalogue = [r for r in results if r.kind == "catalogue"]
    outside = [r for r in results if r.kind == "out-of-catalogue"]
    accepted = sum(1 for r in catalogue if r.accepted)
    exact = sum(1 for r in catalogue if r.exact)
    refusals = sum(1 for r in outside if r.refused)
    false_claims = len(outside) - refusals
    total = len(results)
    passed = sum(1 for r in results if r.accepted)
    return {
        "version": 1,
        "split": split,
        "cases": [r.to_dict() for r in results],
        "metrics": {
            "total": total,
            "catalogue": len(catalogue),
            "out_of_catalogue": len(outside),
            "catalogue_accepted": accepted,
            "catalogue_accuracy": round(accepted / max(1, len(catalogue)), 4),
            "catalogue_exact": exact,
            "catalogue_exact_accuracy": round(exact / max(1, len(catalogue)), 4),
            "out_refused": refusals,
            "out_refusal_rate": round(refusals / max(1, len(outside)), 4),
            "false_claims": false_claims,
            "overall_passed": passed,
            "overall_accuracy": round(passed / max(1, total), 4),
        },
        "failures": [r.to_dict() for r in results if not r.accepted],
    }


def guards(report: dict) -> list[str]:
    """Return human-readable guard failures (empty means the suite is healthy)."""

    metrics = report["metrics"]
    problems: list[str] = []
    if report.get("split") == "holdout":
        # The holdout is a measurement, not a tuning target: these are the
        # recorded floors from the frozen matcher, not aspirations.
        catalogue_floor, exact_floor = 0.60, 0.50
    else:
        catalogue_floor, exact_floor = 0.85, 0.60
    if metrics["catalogue_accuracy"] < catalogue_floor:
        problems.append(
            f"catalogue accuracy {metrics['catalogue_accuracy']:.0%} is below {catalogue_floor:.0%}"
        )
    if metrics["out_refusal_rate"] < 0.90:
        problems.append(
            f"out-of-catalogue refusal {metrics['out_refusal_rate']:.0%} is below 90%"
        )
    if metrics["catalogue_exact_accuracy"] < exact_floor:
        problems.append(
            f"exact accuracy {metrics['catalogue_exact_accuracy']:.0%} is below {exact_floor:.0%}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out prompt routing suite")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--split", choices=sorted(SPLITS), default="tuning", help="tuning, holdout, or all")
    parser.add_argument("--minimum", type=int, default=3, help="matcher minimum score")
    parser.add_argument("--guards", action="store_true", help="exit 1 when guards fail")
    parser.add_argument("--failures", action="store_true", help="print every failing case")
    args = parser.parse_args(argv)

    report = evaluate_prompts(SPLITS[args.split], minimum=args.minimum, split=args.split)
    metrics = report["metrics"]
    print(f"split:        {report['split']}")
    print(f"prompts:      {metrics['total']} ({metrics['catalogue']} catalogue, {metrics['out_of_catalogue']} out)")
    print(f"catalogue:    {metrics['catalogue_accepted']}/{metrics['catalogue']} accepted ({metrics['catalogue_accuracy']:.0%}), exact {metrics['catalogue_exact']}/{metrics['catalogue']} ({metrics['catalogue_exact_accuracy']:.0%})")
    print(f"out-of-cat:   {metrics['out_refused']}/{metrics['out_of_catalogue']} refused ({metrics['out_refusal_rate']:.0%}), false claims {metrics['false_claims']}")
    print(f"overall:      {metrics['overall_passed']}/{metrics['total']} ({metrics['overall_accuracy']:.0%})")
    if args.failures:
        for failure in report["failures"]:
            print(f"  FAIL [{failure['kind']}] {failure['prompt']!r} -> {failure['matched']} (expected {failure['expected']})")
    problems = guards(report)
    if problems:
        print("\nGuards failed:")
        for problem in problems:
            print("  -", problem)
    elif args.guards:
        print("\nAll prompt-routing guards pass.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.json}")
    return 1 if (args.guards and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
