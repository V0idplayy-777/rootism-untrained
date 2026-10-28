"""Long-form synthesis and explicit scratch space.

Two related weaknesses at 8k (or even 32k):

1. **Long outputs.** A 600-line module cannot be produced in one 1k-token
   generation, and a model that tries loses the plan halfway through. The fix
   is structural: plan -> skeleton -> per-unit fill, where each fill sees the
   plan and *signatures* of neighbours, not their bodies. Token use stays flat
   as the artifact grows.
2. **Deep reasoning.** Intermediate steps need somewhere to live. Rather than
   hoping the model does arithmetic silently, allocate an explicit scratch
   budget (`<|scratchpad|>`) sized from the context window, and verify
   candidates by execution where a checker exists.

Both are implemented generator-agnostically: pass any
``generate_fn(prompt: str, max_new_tokens: int) -> str`` (the served model, a
local model wrapper, or a test double). Nothing here imports torch.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

GenerateFn = Callable[[str, int], str]


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass
class ReasoningBudget:
    """Split the window between plan, scratch and output.

    Defaults are deliberately conservative for 8k: a large verbatim transcript
    plus a long scratchpad plus a long answer does not fit, and silently
    truncating is worse than planning for it.
    """

    context_window: int = 8192
    prompt_tokens: int = 4096
    max_new_tokens: int = 2048
    plan_fraction: float = 0.15
    scratch_fraction: float = 0.35
    answer_fraction: float = 0.50

    def __post_init__(self) -> None:

        total = self.plan_fraction + self.scratch_fraction + self.answer_fraction

        assert total > 0, "budget fractions must be positive"

        self.plan_fraction /= total

        self.scratch_fraction /= total

        self.answer_fraction /= total

    @property
    def available(self) -> int:
        """Tokens left for new content after the prompt."""

        return max(self.context_window - self.prompt_tokens, 0)

    def allocate(self) -> dict[str, int]:
        """Concrete budgets; `answer` is at least 256 tokens when possible."""

        available = min(self.max_new_tokens, self.available) or self.max_new_tokens

        plan = int(available * self.plan_fraction)

        scratch = int(available * self.scratch_fraction)

        answer = available - plan - scratch

        if answer < 256 and available >= 768:

            answer = 256

            scratch = max(128, available - plan - answer)

        return {
            "plan": max(plan, 64),
            "scratch": max(scratch, 64),
            "answer": max(answer, 128),
            "available": available,
        }

    def describe(self) -> str:
        alloc = self.allocate()

        return (
            f"window={self.context_window}, prompt={self.prompt_tokens}, "
            f"plan={alloc['plan']}, scratch={alloc['scratch']}, "
            f"answer={alloc['answer']} tokens"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def python_syntax_errors(code: str) -> list[str]:
    """Compile-time problems, phrased so they can be fed back to a model."""

    errors: list[str] = []

    try:

        ast.parse(code)

    except SyntaxError as e:

        errors.append(f"{type(e).__name__}: {e.msg} (line {e.lineno})")

        return errors

    try:

        compile(code, "<generated>", "exec")

    except SyntaxError as e:  # pragma: no cover - ast.parse catches these first

        errors.append(f"{type(e).__name__}: {e.msg} (line {e.lineno})")

    return errors


def missing_definitions(code: str) -> list[str]:
    """Top-level `from x import y`/attribute use that the module never defines.

    A cheap undefined-name heuristic: it only reports *called* names that are
    neither defined locally, imported, nor builtins. It is intentionally
    conservative (no false confidence in either direction).
    """

    try:

        tree = ast.parse(code)

    except SyntaxError:

        return []

    defined = set(__builtins__) if isinstance(__builtins__, dict) else set(dir(__builtins__))

    defined |= {
        "print",
        "len",
        "range",
        "int",
        "float",
        "str",
        "dict",
        "list",
        "set",
        "tuple",
        "bool",
        "open",
        "super",
        "object",
        "type",
        "isinstance",
        "ValueError",
        "TypeError",
        "KeyError",
        "Exception",
    }

    defined |= {"self", "cls"}

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            defined.add(node.name)

        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):

            defined.add(node.id)

        elif isinstance(node, ast.arg):

            defined.add(node.arg)

        elif isinstance(node, (ast.Import, ast.ImportFrom)):

            for alias in node.names:

                defined.add(alias.asname or alias.name.split(".")[0])

        elif isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(target, ast.Name):

                    defined.add(target.id)

    called = Counter()

    for node in ast.walk(tree):

        if isinstance(node, ast.Call):

            fn = node.func

            if isinstance(fn, ast.Name):

                called[fn.id] += 1

            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):

                called[fn.value.id] += 1

    return sorted(name for name in called if name not in defined)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class UnitPlan:
    name: str
    purpose: str
    signature: str = ""


@dataclass
class SynthesisPlan:
    instruction: str
    units: list[UnitPlan] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def size(self) -> int:

        return len(self.units)

    def render(self) -> str:

        lines = ["Plan:"] + [f"{i + 1}. {u.name}: {u.purpose}" for i, u in enumerate(self.units)]

        if self.invariants:

            lines.append("Invariants:")

            lines.extend(f"- {inv}" for inv in self.invariants)

        return "\n".join(lines)


_NAME = re.compile(
    r"^\s*(?:\d+[.)]|\-|\*)\s*`?([A-Za-z_][A-Za-z0-9_]*)`?"
    r"\s*(?:\(([^)]*)\))?\s*`?\s*[:\-—]\s*(.+)$"
)
_LOOSE = re.compile(r"^\s*(?:\d+[.)]|\-|\*)\s+(.+)$")


def parse_plan(text: str, fallback_instruction: str = "") -> SynthesisPlan:
    """Parse a numbered plan into units; tolerant of prose around it."""

    units: list[UnitPlan] = []

    invariants: list[str] = []

    in_invariants = False

    for line in (text or "").splitlines():

        stripped = line.strip()

        if not stripped:

            continue

        if re.match(r"(?i)^(invariants?|constraints?|must hold)\b", stripped):

            in_invariants = True

            continue

        if in_invariants and stripped.startswith(("-", "*")):

            invariants.append(stripped.lstrip("-* ").strip())

            continue

        match = _NAME.match(stripped)

        if match:

            name, args, purpose = match.group(1), match.group(2), match.group(3)

            units.append(
                UnitPlan(
                    name=name,
                    purpose=purpose.strip(),
                    signature=f"{name}({args})" if args else name,
                )
            )

            in_invariants = False

            continue

        loose = _LOOSE.match(stripped)

        if loose and not in_invariants:

            purpose = loose.group(1)

            name_match = re.search(r"[a-z_][a-z0-9_]{2,}", purpose)

            units.append(
                UnitPlan(
                    name=name_match.group(0) if name_match else f"unit_{len(units) + 1}",
                    purpose=purpose,
                )
            )

    if not units:

        units = [UnitPlan(name="solve", purpose=fallback_instruction or "implement the task")]

    return SynthesisPlan(
        instruction=fallback_instruction,
        units=units[:12],
        invariants=invariants[:6],
        raw=text or "",
    )


# ---------------------------------------------------------------------------
# Long-form synthesis
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    code: str
    plan: SynthesisPlan
    units: dict[str, str] = field(default_factory=dict)
    repairs: int = 0
    errors: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:

        return not self.errors

    def to_dict(self) -> dict:

        return {
            "ok": self.ok,
            "units": self.plan.size,
            "repairs": self.repairs,
            "errors": self.errors,
            "trace": self.trace,
            "chars": len(self.code),
        }


class LongFormSynthesizer:
    """Plan -> skeleton -> per-unit fill -> validate/repair.

    `max_unit_tokens` bounds each fill call, so total output length is
    independent of any single generation length. `neighbour_context` controls
    how much already-written code each fill sees (signatures only by default).
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        budget: ReasoningBudget | None = None,
        max_units: int = 12,
        max_repairs: int = 2,
        validator: Callable[[str], list[str]] | None = None,
        neighbour_context: str = "signatures",
    ):

        self.generate_fn = generate_fn

        self.budget = budget or ReasoningBudget()

        self.max_units = max_units

        self.max_repairs = max_repairs

        self.validator = validator or python_syntax_errors

        self.neighbour_context = neighbour_context

    # -- prompts -----------------------------------------------------------
    def _plan_prompt(self, instruction: str, constraints: str) -> str:

        alloc = self.budget.allocate()

        return (
            "<|plan|>\n"
            f"Task: {instruction}\n"
            f"Constraints: {constraints or 'standard library only, typed, testable'}\n"
            "Write a decomposition plan only — no code.\n"
            "Format: one numbered line per unit, `name(args): one-sentence purpose`.\n"
            "Then a line `Invariants:` followed by '- ...' bullets.\n"
            f"Keep it under {alloc['plan']} tokens. Use at most {self.max_units} units.\n"
            "<|endoftext|>"
        )

    def _skeleton_prompt(self, instruction: str, plan: SynthesisPlan, constraints: str) -> str:
        return (
            "<|plan|>\n"
            f"{plan.render()}\n<|endoftext|>\n"
            f"Implement the module interface for: {instruction}\n"
            f"Constraints: {constraints or 'standard library only'}\n"
            "Output ONLY Python: imports, module docstring, dataclasses/TypedDicts, "
            "and def/class stubs with docstrings and `...` bodies. No implementations.\n"
            "<|endoftext|>"
        )

    def _unit_prompt(
        self, instruction: str, plan: SynthesisPlan, name: str, neighbours: str
    ) -> str:
        unit = next((u for u in plan.units if u.name == name), None)

        alloc = self.budget.allocate()

        return (
            "<|plan|>\n"
            f"{plan.render()}\n<|endoftext|>\n"
            f"<|file_chunk|> existing interface\n{neighbours}\n<|endoftext|>\n"
            "Write ONLY the body of `"
            f"{name}` (no signature, no markdown), consistent with the plan and the "
            f"interface above.\nInvariants: {'; '.join(plan.invariants) or 'none stated'}\n"
            f"Unit purpose: {unit.purpose if unit else instruction}\n"
            f"Stay under {alloc['scratch']} tokens.\n"
            "<|endoftext|>"
        )

    def _repair_prompt(self, code: str, errors: list[str]) -> str:
        return (
            "<|tool_error|>\n"
            "The generated module does not compile:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n<|endoftext|>\n<|file_chunk|>\n"
            + code
            + "\n<|endoftext|>\nReturn the complete corrected Python file only.\n"
        )

    # -- steps -------------------------------------------------------------
    @staticmethod
    def _strip_fences(text: str) -> str:

        text = text or ""

        fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)

        return fence.group(1).strip("\n") if fence else text.strip("\n")

    def plan(self, instruction: str, constraints: str = "") -> SynthesisPlan:

        raw = self.generate_fn(self._plan_prompt(instruction, constraints), self.budget.allocate()["plan"])

        return parse_plan(raw, instruction)

    def _skeleton(self, instruction: str, plan: SynthesisPlan, constraints: str) -> str:

        code = self._strip_fences(
            self.generate_fn(
                self._skeleton_prompt(instruction, plan, constraints),
                self.budget.allocate()["answer"],
            )
        )

        return code

    def _fill_units(self, instruction: str, plan: SynthesisPlan, skeleton: str) -> dict[str, str]:
        """Fill each stubbed unit; returns the complete module text per unit."""

        filled: dict[str, str] = {}

        signatures = "\n".join(
            line
            for line in skeleton.splitlines()
            if re.match(r"^\s*(?:def|class|async def)\s", line)
        )

        for unit in plan.units:

            prompt = self._unit_prompt(instruction, plan, unit.name, signatures)

            body = self._strip_fences(
                self.generate_fn(prompt, self.budget.allocate()["scratch"])
            )

            filled[unit.name] = body

        return filled

    def _apply_bodies(self, skeleton: str, bodies: dict[str, str]) -> str:
        """Replace `...`/`pass` stub bodies with generated bodies (AST-guided)."""

        try:

            tree = ast.parse(skeleton)

        except SyntaxError:

            return skeleton

        lines = skeleton.splitlines()

        replacements: list[tuple[int, int, list[str]]] = []

        for node in tree.body:

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

                continue

            body = bodies.get(node.name)

            if not body:

                continue

            children = getattr(node, "body", [])

            if not children:

                continue

            first, last = children[0].lineno, getattr(children[-1], "end_lineno", children[-1].lineno)

            # Only replace stub bodies (single `...`/pass/docstring+pass).
            stub = all(
                isinstance(stmt, ast.Expr)
                and (
                    isinstance(stmt.value, ast.Constant)
                    and (stmt.value.value is Ellipsis or isinstance(stmt.value.value, str))
                )
                or isinstance(stmt, ast.Pass)
                for stmt in children
            )

            if not stub:

                continue

            indent = " " * (len(lines[first - 1]) - len(lines[first - 1].lstrip()))

            dedented = textwrap.dedent(body.replace("\t", "    ")).splitlines()

            new_body = [indent + line if line.strip() else "" for line in dedented]

            if not new_body:

                new_body = [indent + "pass"]

            replacements.append((first, last, new_body))

        for first, last, new_body in sorted(replacements, reverse=True):

            lines[first - 1 : last] = new_body

        return "\n".join(lines) + "\n"

    def synthesise(self, instruction: str, constraints: str = "") -> SynthesisResult:
        """Full pipeline with bounded repair."""

        trace: list[dict] = []

        plan = self.plan(instruction, constraints)

        trace.append({"step": "plan", "units": plan.size, "raw_chars": len(plan.raw)})

        skeleton = self._skeleton(instruction, plan, constraints)

        trace.append({"step": "skeleton", "chars": len(skeleton), "valid": not self.validator(skeleton)})

        bodies = self._fill_units(instruction, plan, skeleton)

        code = self._apply_bodies(skeleton, bodies)

        trace.append({"step": "fill", "units": len(bodies), "chars": len(code)})

        errors = self.validator(code)

        repairs = 0

        while errors and repairs < self.max_repairs:

            repairs += 1

            code = self._strip_fences(
                self.generate_fn(self._repair_prompt(code, errors), self.budget.allocate()["answer"])
            )

            errors = self.validator(code)

            trace.append({"step": "repair", "attempt": repairs, "errors": errors})

        # Undefined names are reported but not fatal (heuristic).
        undefined = missing_definitions(code)

        if undefined and not errors:

            trace.append({"step": "lint", "possibly_undefined": undefined[:8]})

        return SynthesisResult(
            code=code,
            plan=plan,
            units=bodies,
            repairs=repairs,
            errors=errors,
            trace=trace,
        )

    # -- deep reasoning ----------------------------------------------------
    def reason(
        self,
        question: str,
        checker: Callable[[str], bool] | None = None,
        samples: int = 3,
    ) -> dict:
        """Scratchpad reasoning with optional execution-checked self-consistency.

        Each sample writes `<|scratchpad|>` notes then `ANSWER:`. Candidates are
        ranked by `checker` when supplied, otherwise by majority answer, which
        is the best available signal without an oracle.
        """

        alloc = self.budget.allocate()

        candidates: list[dict] = []

        for _ in range(max(1, samples)):

            raw = self.generate_fn(
                "<|scratchpad|>\n"
                f"Problem: {question}\n"
                "Think step by step inside the scratchpad. State assumptions, "
                "check units/edge cases, then output a line `ANSWER: <final>`.\n"
                f"Keep scratch under {alloc['scratch']} tokens.\n",
                alloc["scratch"] + alloc["answer"],
            )

            answer_match = re.search(r"(?im)^\s*ANSWER:\s*(.+)$", raw)

            answer = answer_match.group(1).strip() if answer_match else raw.strip().splitlines()[-1][:200]

            candidates.append({"raw": raw, "answer": answer})

        verified = []

        if checker is not None:

            for cand in candidates:

                try:

                    cand["verified"] = bool(checker(cand["raw"]))

                except Exception:

                    cand["verified"] = False

                verified.append(cand)

        winner = None

        verified_candidates = [c for c in verified if c.get("verified")]

        if verified_candidates:

            winner = verified_candidates[0]

            strategy = "execution_check"

        else:

            counts = Counter(c["answer"] for c in candidates)

            top, _ = counts.most_common(1)[0]

            winner = next(c for c in candidates if c["answer"] == top)

            strategy = "majority"

        return {
            "answer": winner["answer"],
            "strategy": strategy,
            "samples": candidates,
            "agreement": (
                max(Counter(c["answer"] for c in candidates).values()) / len(candidates)
            ),
        }


def split_long_function(code: str, max_lines: int = 60) -> list[str]:
    """Propose split points for an over-long generated function.

    Returns human-readable chunks (signature + body segments) so the caller can
    ask the model to extract helpers rather than truncate.
    """

    try:

        tree = ast.parse(code)

    except SyntaxError:

        return [code]

    out: list[str] = []

    lines = code.splitlines()

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            span = getattr(node, "end_lineno", node.lineno) - node.lineno + 1

            if span > max_lines:

                segment = lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)]

                out.append(f"# {node.name}: {span} lines (>{max_lines})")

                for start in range(0, len(segment), max_lines):

                    out.append("\n".join(segment[start : start + max_lines]))

    return out or [code]
