"""Turn raw tool/traceback output into something a non-engineer can act on.

Every rule is ``(pattern, title, plain explanation, next steps)``. ``humanize``
picks the most specific matching rule, strips ANSI/noise, keeps a short excerpt
of the raw text for power users, and never claims to know more than the text
supports: when nothing matches it says "unrecognised failure" and shows the last
line, instead of inventing a diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

RULES: list[tuple[str, str, str, list[str]]] = [
    (
        r"ModuleNotFoundError: No module named '?([^'\s]+)'?|ImportError: No module named '?([^'\s]+)'?|No module named '?([^'\s]+)'?",
        "A Python package is missing",
        "The code imports a library that is not installed in this environment.",
        [
            "Add it to requirements.txt / pyproject.toml, then reinstall",
            "If it is only used in tests, put it in the dev requirements",
            "Re-run the command after installing",
        ],
    ),
    (
        r"is not empty; pass --force|already contains files|directory not empty",
        "That folder already has files in it",
        "The agent will not write a starter over an existing project.",
        [
            "Scaffold into a new (or empty) folder",
            "Or pass --force to add the starter alongside the existing files (nothing is overwritten)",
        ],
    ),
    (
        r"EADDRINUSE|address already in use|port is already allocated",
        "That port is already taken",
        "Another process (often a previous run of this same app) is using the port.",
        [
            "Stop the old process, or let the deploy engine pick a free port",
            "Local check: `lsof -i :<port>` (macOS/Linux) or `netstat -ano | findstr :<port>`",
        ],
    ),
    (
        r"docker: command not found|Cannot connect to the Docker daemon|docker daemon is not running|permission denied while trying to connect to the Docker daemon",
        "Docker is not available",
        "The command needs a working Docker installation and a running daemon.",
        [
            "Install Docker Desktop or the docker engine",
            "Start the daemon: `sudo systemctl start docker`",
            "On Linux, add your user to the docker group (then log out and back in)",
            "No Docker? `agent deploy --provider vercel` (static sites) or `--provider github-pages`",
        ],
    ),
    (
        r"EACCES: permission denied|PermissionError: \[Errno 13\]|Permission denied \(publickey\)",
        "Permission denied",
        "Something tried to write outside the place it is allowed to, or key-based auth was rejected.",
        [
            "Check file ownership: the agent only writes inside the project folder",
            "For git pushes, confirm your SSH key is loaded (`ssh -T git@github.com`)",
            "For containers, avoid writing to system paths - use the workspace",
        ],
    ),
    (
        r"npm ERR! code ENOENT|ENOENT: no such file or directory, open '.*package.json'",
        "npm could not find the project files",
        "The command ran in a folder without a package.json (or node_modules is missing).",
        [
            "Run from the project root (where package.json lives)",
            "Install dependencies: `npm install` (or `pnpm install`)",
        ],
    ),
    (
        r"npm ERR! code ERESOLVE|ERESOLVE unable to resolve dependency tree",
        "Two npm packages want incompatible versions",
        "The dependency tree has a conflict, usually a peer-dependency mismatch.",
        [
            "Prefer fixing the versions over forcing them",
            "`npm install --legacy-peer-deps` is a temporary workaround, not a fix",
            "Check which package pulled in the conflicting version: `npm ls <pkg>`",
        ],
    ),
    (
        r"SyntaxError: (?:invalid syntax|unexpected)|IndentationError",
        "The code has a syntax error",
        "Python could not parse a file, so it never ran.",
        [
            "The message points at the file and line",
            "Look for a missing colon/bracket on the line above the reported one",
            "Run `python -m py_compile <file>` to check one file quickly",
        ],
    ),
    (
        r"AssertionError|assert .* ==|FAILED tests?/.*::|(\d+) failed",
        "A test failed",
        "The code ran, but behaved differently from what the test expects.",
        [
            "Read the first failing assertion - it shows expected vs actual",
            "Fix the code (or the test, if the test is wrong) and re-run",
            "`agent verify` repeats the loop until green",
        ],
    ),
    (
        r"Connection refused|ECONNREFUSED|Failed to establish a new connection",
        "Could not connect to a service",
        "Nothing is listening at that address, or it is still starting.",
        [
            "Start the backend/database first",
            "Check the host and port in the connection string",
            "In Docker, use the service name (e.g. `db`) instead of `localhost`",
        ],
    ),
    (
        r"Temporary failure in name resolution|Name or service not known|getaddrinfo failed|ENOTFOUND",
        "A hostname could not be resolved",
        "DNS lookup failed, usually a typo in the URL/host or no network access.",
        ["Check the spelling of the host", "Verify the container has network access"],
    ),
    (
        r"SSL: CERTIFICATE_VERIFY_FAILED|certificate verify failed|self signed certificate",
        "TLS certificate check failed",
        "The certificate is not trusted by this machine (expired, self-signed, or a proxy).",
        [
            "Do not disable verification - fix the certificate or trust the proxy's CA",
            "For local development, use http:// locally rather than weakening TLS",
        ],
    ),
    (
        r"fatal: Authentication failed|could not read Username|remote: Permission to .* denied",
        "Git authentication failed",
        "The push was rejected: no credentials, or no permission for that repository.",
        [
            "`gh auth login` (then `gh auth setup-git`) or add an SSH key",
            "Confirm the remote: `git remote -v`",
        ],
    ),
    (
        r"No space left on device|ENOSPC",
        "The disk is full",
        "A write failed because there is no free space left.",
        ["Free space (old images: `docker system prune`)", "Re-run the command"],
    ),
    (
        r"Killed|Out of memory|OOMKilled|MemoryError",
        "Ran out of memory",
        "The process was killed for using more memory than allowed.",
        [
            "Raise the container memory limit, or process data in smaller batches",
            "Check for an accidental infinite loop collecting results",
        ],
    ),
    (
        r"failed to solve|failed to compute cache key|dockerfile parse error|unknown instruction",
        "Docker could not build the image",
        "The Dockerfile referenced a file or instruction that does not exist.",
        [
            "The build log names the failing step and the missing path",
            "Check `.dockerignore` is not excluding a needed file",
            "`agent deploy --dry-run` shows the plan without building",
        ],
    ),
    (
        r"aes|InvalidToken|SECRET_KEY|Could not deserialize key|badly formed hexadecimal UUID",
        "A secret is missing or malformed",
        "The app needs a key/token and got an empty or invalid value.",
        ["Set the variable in .env or your host's secret store", "Generate one: `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`"],
    ),
    (
        r"not healthy after|never became healthy|health check failed",
        "The app started but never reported healthy",
        "The container is running, yet the health URL did not return a success status.",
        [
            "Check the app listens on 0.0.0.0 and the expected port",
            "Read the container logs: `docker logs <name>`",
            "Point the health check at a real route (default `/`)",
        ],
    ),
    (
        r"could not determine|nothing safe to start|project type could not be determined|cannot generate a Dockerfile|no start command",
        "The project type was not recognised",
        "The agent will not guess how to start an unknown project.",
        [
            "Add `[deploy] start_cmd = \"...\"` to agentdeploy.toml",
            "Or add a Dockerfile - it takes precedence",
        ],
    ),
]

COMPILED = [(re.compile(pattern, re.IGNORECASE), title, plain, steps) for pattern, title, plain, steps in RULES]


@dataclass
class HumanError:
    title: str
    plain: str
    next_steps: list[str] = field(default_factory=list)
    raw: str = ""
    matched: bool = True

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "plain": self.plain,
            "next_steps": list(self.next_steps),
            "raw": self.raw[-1200:],
            "matched": self.matched,
        }

    def render(self) -> str:
        lines = [self.title, f"  {self.plain}"]
        for step in self.next_steps:
            lines.append(f"  -> {step}")
        return "\n".join(lines)


def clean(text: str, max_chars: int = 6000) -> str:
    text = ANSI.sub("", text or "")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[-max_chars:]


def humanize(error: str | None, max_chars: int = 6000) -> HumanError:
    """Map raw failure text to an explanation + concrete next steps."""

    raw = clean(error or "", max_chars)
    for pattern, title, plain, steps in COMPILED:
        match = pattern.search(raw)
        if match:
            captured = [group for group in match.groups() if group]
            if captured:
                plain = f"{plain} Missing: {captured[0]}."
            return HumanError(title=title, plain=plain, next_steps=list(steps), raw=raw, matched=True)
    last_line = raw.splitlines()[-1][:200] if raw.splitlines() else "no output"
    return HumanError(
        title="Unrecognised failure",
        plain=f"The command failed and the agent does not have a specific explanation. Last line: {last_line}",
        next_steps=[
            "Read the full output above (power mode shows all of it)",
            "Run `agent doctor` to check the local toolchain",
        ],
        raw=raw,
        matched=False,
    )


def humanize_result(result: dict) -> HumanError:
    """Humanise a tool/test/deploy result dictionary."""

    parts = [
        str(result.get("error") or ""),
        str(result.get("stderr") or ""),
        str(result.get("stdout") or ""),
        str(result.get("output") or ""),
    ]
    return humanize("\n".join(part for part in parts if part))
