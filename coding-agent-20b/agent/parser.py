"""Robust tool-call parsing: native tokens, fenced JSON, bare JSON, ReAct.

Pipeline for every candidate payload:
  extract -> json.loads -> deterministic repair (trailing commas, brace
  balancing) -> JSON-Schema validation -> typed coercion/defaults.
`parse_with_repair` returns (call, error); on error the agent feeds `error`
back to the model (repair loop) instead of silently dropping the call.
No constrained decoding is possible without a serving engine, so the
contract is: generation-time stop tokens + this validator + bounded retries.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:

    import jsonschema

    HAS_JSONSCHEMA = True


except ImportError:

    HAS_JSONSCHEMA = False


_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


_NATIVE = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*(?:<eos>|<\|tool_result\|>|$)", re.DOTALL
)


_REACT_ACTION = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(\{.*)", re.DOTALL)


def _balance_braces(text: str) -> str:
    """Close unclosed {/[ while respecting string literals (best-effort repair)."""

    stack, in_str, esc = [], False, False

    for ch in text:

        if in_str:

            if esc:

                esc = False

            elif ch == "\\":

                esc = True

            elif ch == '"':

                in_str = False

            continue

        if ch == '"':

            in_str = True

        elif ch in "{[":

            stack.append("}" if ch == "{" else "]")

        elif ch in "}]" and stack and stack[-1] == ch:

            stack.pop()

    if in_str:

        text += '"'

    return text + "".join(reversed(stack))


def repair_json(raw: str) -> str:

    s = raw.strip()

    s = re.sub(r",\s*([}\]])", r"\1", s)

    s = s.replace("'", '"') if s.count("'") > s.count('"') and '"' not in s else s

    return _balance_braces(s)


def _loads_maybe_repair(raw: str) -> tuple[Any | None, str | None]:

    try:

        return json.loads(raw), None

    except json.JSONDecodeError:

        try:

            return json.loads(repair_json(raw)), None

        except json.JSONDecodeError as e2:

            return None, f"JSON parse failed: {e2}"


def validate_against_schema(
    call: dict, schemas: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """Validate + normalise {tool, arguments}; returns (call, error)."""

    if not isinstance(call, dict) or "tool" not in call:

        return None, "tool call must be an object with a 'tool' field"

    tool = call["tool"]

    if tool not in schemas:

        return None, f"unknown tool {tool!r}; valid tools: {sorted(schemas)}"

    args = call.get("arguments", {})

    if not isinstance(args, dict):

        return None, "'arguments' must be an object"

    schema = schemas[tool].get("parameters", {})

    if HAS_JSONSCHEMA:

        try:

            jsonschema.validate(args, schema)

        except jsonschema.ValidationError as e:

            return (
                None,
                f"schema validation failed for {tool}: {e.message} (at {list(e.absolute_path)})",
            )

    else:

        err = _validate_fallback(args, schema)

        if err:

            return None, f"schema validation failed for {tool}: {err}"

    props = schema.get("properties", {})

    for key, spec in props.items():

        if key not in args and isinstance(spec, dict) and "default" in spec:

            args[key] = spec["default"]

    return {"tool": tool, "arguments": args}, None


def _validate_fallback(args: dict, schema: dict) -> str | None:

    for req in schema.get("required", []):

        if req not in args:

            return f"missing required property {req!r}"

    types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, spec in schema.get("properties", {}).items():

        if key in args and isinstance(spec, dict) and spec.get("type") in types:

            expected = types[spec["type"]]

            if spec["type"] == "integer" and isinstance(args[key], bool):

                return f"property {key!r} must be integer"

            if not isinstance(args[key], expected):

                return f"property {key!r} must be {spec['type']}"

    return None


def extract_candidates(text: str):

    for pat in (_NATIVE, _FENCED):

        m = pat.search(text)

        if m:

            yield m.group(1)

    m = re.search(r"(\{\s*\"tool\"\s*:.*)", text, re.DOTALL)

    if m:

        yield m.group(1)

    m = _REACT_ACTION.search(text)

    if m:

        yield json.dumps({"tool": m.group(1), "arguments": {}})

        m2 = re.search(r"(\{.*)", m.group(2), re.DOTALL)

        if m2:

            obj, _ = _loads_maybe_repair(m2.group(1))

            if isinstance(obj, dict):

                yield json.dumps({"tool": m.group(1), "arguments": obj})


def parse_tool_call(
    text: str, schemas: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """Return (validated_call, None) or (None, actionable_error_for_repair_loop)."""

    last_err = 'no tool call found; emit <|tool_call|> with a JSON object {"tool": ..., "arguments": {...}}'

    for raw in extract_candidates(text):

        obj, err = _loads_maybe_repair(raw)

        if err:

            last_err = err

            continue

        call, verr = validate_against_schema(obj, schemas)

        if call is not None:

            return call, None

        last_err = verr

    return None, last_err


def repair_prompt(task_error: str, bad_output: str) -> str:

    return (
        f"Your previous tool call was rejected: {task_error}\n"
        f"Previous output (truncated): {bad_output[:800]}\n"
        "Reply with ONLY a corrected call in this exact form:\n"
        '<|tool_call|>\n{"tool": "<tool_name>", "arguments": {...}}\n'
    )
