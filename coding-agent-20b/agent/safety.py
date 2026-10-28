"""Agent safety: secret redaction, dangerous-command blocking, output sanitising."""

from __future__ import annotations

import re

_SECRET_PATTERNS = [
    (
        re.compile(
            r"(?i)(api[_-]?key|secret|token|password|passwd|aws_secret_access_key)\s*[:=]\s*\S+"
        ),
        r"\1=<redacted>",
    ),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<redacted-github-token>"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "<redacted-api-key>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<redacted-aws-key>"),
    (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "-----BEGIN REDACTED KEY-----",
    ),
]


_DANGEROUS = [
    re.compile(r"rm\s+(-[a-z]*f[a-z]*\s+)?/(?:\s|$)"),
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}"),
    re.compile(r"(curl|wget)\s+[^\n|]*\|\s*(sudo\s+)?(sh|bash)"),
    re.compile(r"chmod\s+777\s+/"),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+.*of=/dev/"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"shutdown|reboot|init\s+0"),
]


def redact_secrets(text: str) -> str:

    out = str(text)

    for pat, repl in _SECRET_PATTERNS:

        out = pat.sub(repl, out)

    return out


def check_code_safety(code: str) -> tuple[bool, str]:
    """Return (allowed, reason). Blocks only unambiguous host-destruction patterns;
    the sandbox remains the real boundary — this is defence in depth."""

    for pat in _DANGEROUS:

        if pat.search(code):

            return False, f"Blocked dangerous pattern: {pat.pattern[:60]}"

    return True, "ok"


def sanitize_observation(text: str, max_chars: int = 8000) -> str:
    """Redact secrets and bound size before an observation re-enters the prompt."""

    cleaned = redact_secrets(text)

    cleaned = re.compile(r"\x1b\[[0-9;]*m").sub("", cleaned)

    if len(cleaned) > max_chars:

        cleaned = (
            cleaned[:max_chars] + f"\n...[truncated {len(cleaned) - max_chars} chars]"
        )

    return cleaned
