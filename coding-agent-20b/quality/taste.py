"""Taste gate: the difference between "it renders" and "a person would use it".

The code critic answers *is this safe and maintainable*. This module answers
*does the product look and feel intentional*: one accent, a spacing and type
scale, dark mode, a layout that survives a phone, empty/loading/error states,
labels and focus rings, sentences instead of "Lorem ipsum", and none of the
2018-template smells (fixed 960px container, Helvetica stack, a wall of
`!important`).

It is deliberately heuristic and offline: no screenshots, no vision model, no
network. Every finding names the file and the fix, so the same output can be
shown to a user, fed back into a repair loop, or used as the rejected side of a
preference pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DIMENSION_WEIGHTS = {
    "design_system": 0.22,
    "responsiveness": 0.20,
    "states": 0.18,
    "accessibility": 0.16,
    "typography": 0.10,
    "copy": 0.09,
    "anti_patterns": 0.05,
}

LEGACY_SMELLS = (
    ("fixed-width-container", r"(?<![-\w])width:\s*9[0-9]{2}px"),
    ("helvetica-stack", r"font-family:\s*[\"']?Helvetica Neue"),
    ("font-size-13", r"font-size:\s*13px"),
    ("bootstrap-cdn", r"(bootstrap(\.min)?\.css|jquery)"),
)


@dataclass(frozen=True)
class TasteFinding:
    """One concrete reason a UI reads as untended."""

    rule: str
    dimension: str
    severity: str
    message: str
    fix: str
    path: str = "web/app.css"

    def to_dict(self) -> dict:
        """Serialise for reports and dataset metadata."""
        return {
            "rule": self.rule,
            "dimension": self.dimension,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
            "path": self.path,
        }


@dataclass
class TasteReport:
    """Score plus the evidence behind it."""

    score: int = 100
    dimensions: dict[str, int] = field(default_factory=dict)
    findings: list[TasteFinding] = field(default_factory=list)
    files_reviewed: int = 0
    threshold: int = 70

    @property
    def ok(self) -> bool:
        """True when the UI clears the bar and has no critical finding."""
        blocking = any(item.severity == "critical" for item in self.findings)
        return self.score >= self.threshold and not blocking

    def count(self, severity: str) -> int:
        """Number of findings at one severity (used by reports)."""
        return sum(1 for item in self.findings if item.severity == severity)

    def summary(self) -> str:
        """One-line verdict for logs and CLI output."""
        state = "pass" if self.ok else "fail"
        worst = ", ".join(sorted({item.dimension for item in self.findings})) or "none"
        return f"taste {state}: {self.score}/100 (weakest: {worst})"

    def to_dict(self) -> dict:
        """Serialise for JSON reports."""
        return {
            "score": self.score,
            "ok": self.ok,
            "threshold": self.threshold,
            "dimensions": dict(self.dimensions),
            "findings": [item.to_dict() for item in self.findings],
            "files_reviewed": self.files_reviewed,
        }


def _read(root: Path, name: str) -> str:
    path = root / name
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def _design_findings(css: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    tokens = _count(r"--[a-z0-9-]+\s*:", css)
    if tokens < 12:
        findings.append(
            TasteFinding(
                "few-design-tokens",
                "design_system",
                "high",
                f"only {tokens} CSS custom properties: colours and spacing are hard-coded",
                "define colour, spacing, radius and type tokens once, then compose with var()",
            )
        )
    if not re.search(r":root\s*\{", css):
        findings.append(
            TasteFinding(
                "no-token-root",
                "design_system",
                "medium",
                "no :root token block",
                "put the palette and scale in :root so the theme is editable in one place",
            )
        )
    if "prefers-color-scheme" not in css:
        findings.append(
            TasteFinding(
                "no-dark-mode",
                "design_system",
                "high",
                "no dark colour scheme",
                "add a prefers-color-scheme block that swaps surface, text and border tokens",
            )
        )
    if not re.search(r"--(radius|radius-sm|radius-lg)\b", css):
        findings.append(
            TasteFinding(
                "no-radius-scale",
                "design_system",
                "low",
                "no radius tokens: corners will drift between components",
                "add --radius-sm/md/lg and use them on cards, inputs and buttons",
            )
        )
    return findings


def _responsiveness_findings(css: str, html: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    if _count(r"@media\s*\(", css) < 3:
        findings.append(
            TasteFinding(
                "thin-responsive-rules",
                "responsiveness",
                "high",
                "fewer than three breakpoints",
                "add breakpoints for narrow phones, the table-to-card switch and wide screens",
            )
        )
    if not re.search(r"@media\s*\(max-width:\s*(6[0-9]{2}|7[0-9]{2})px\)", css):
        findings.append(
            TasteFinding(
                "no-mobile-breakpoint",
                "responsiveness",
                "high",
                "no phone-sized breakpoint around 600-760px",
                "stack the sidebar, wrap the top bar and turn table rows into cards below 720px",
            )
        )
    if not re.search(r'<meta\s+name="viewport"', html):
        findings.append(
            TasteFinding(
                "no-viewport",
                "responsiveness",
                "critical",
                "missing viewport meta tag: phones render the desktop layout zoomed out",
                'add <meta name="viewport" content="width=device-width, initial-scale=1" />',
                "web/index.html",
            )
        )
    return findings


def _state_findings(css: str, js: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    for rule, label in (
        ("empty-state", "empty states"),
        ("skeleton", "loading skeletons"),
        ("error", "inline error copy"),
        ("toast", "a non-blocking success/error toast"),
    ):
        if rule not in css and rule not in js:
            findings.append(
                TasteFinding(
                    "missing-" + rule,
                    "states",
                    "high" if rule in {"empty-state", "skeleton"} else "medium",
                    f"no {label}",
                    f"give the {label.split()[0]} path a designed state instead of a blank screen",
                )
            )
    return findings


def _accessibility_findings(css: str, html: str, js: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    if not re.search(r"prefers-reduced-motion", css):
        findings.append(
            TasteFinding(
                "no-reduced-motion",
                "accessibility",
                "low",
                "animations ignore prefers-reduced-motion",
                "disable transitions and skeleton shimmer when the user asks for less motion",
            )
        )
    if not re.search(r"focus-visible", css):
        findings.append(
            TasteFinding(
                "no-focus-ring",
                "accessibility",
                "high",
                "no :focus-visible style: keyboard users cannot see where they are",
                "add a 2px accent outline with an offset on every interactive element",
            )
        )
    if 'lang="' not in html:
        findings.append(
            TasteFinding(
                "no-lang",
                "accessibility",
                "medium",
                "the document does not declare a language",
                'add lang="en" to <html> so screen readers use the right voice',
                "web/index.html",
            )
        )
    if "aria-live" not in html + js:
        findings.append(
            TasteFinding(
                "no-live-region",
                "accessibility",
                "medium",
                "errors and toasts are not announced",
                'mark the toast and form error containers with aria-live="polite"',
            )
        )
    if not re.search(r'<label[^>]*for="', html) and not re.search(r'el\("label",\s*\{\s*for:', js):
        findings.append(
            TasteFinding(
                "labels-not-bound",
                "accessibility",
                "medium",
                "form labels are not associated with their inputs",
                "give every input an id and a matching label for attribute",
                "web/index.html",
            )
        )
    return findings


def _typography_findings(css: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    if "system-ui" not in css and "Inter" not in css and "ui-sans-serif" not in css:
        findings.append(
            TasteFinding(
                "default-font-stack",
                "typography",
                "medium",
                "no deliberate font stack",
                "use system-ui with a readable fallback chain rather than a dated Helvetica stack",
            )
        )
    if not re.search(r"font-size:\s*clamp\(", css) and not re.search(r"--(text|font)-[a-z0-9]+", css):
        findings.append(
            TasteFinding(
                "no-type-scale",
                "typography",
                "medium",
                "headings and body text do not share a scale",
                "add --text-sm/md/lg/xl tokens (or clamp()) and use them consistently",
            )
        )
    return findings


def _copy_findings(html: str, js: str, config: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    if not re.search(r"empty[^\n]{0,200}", config + js, flags=re.I):
        findings.append(
            TasteFinding(
                "generic-empty-copy",
                "copy",
                "medium",
                "empty states have no product-specific sentence",
                "say what the screen is for and what the first action creates",
            )
        )
    if re.search(r"lorem ipsum|item \d|placeholder text", html + js, flags=re.I):
        findings.append(
            TasteFinding(
                "placeholder-copy",
                "copy",
                "high",
                "placeholder copy leaked into the UI",
                "replace it with a concrete noun from the product's domain",
            )
        )
    if not re.search(r"[A-Z][^.!?]{20,}[.!?]", config + js):
        findings.append(
            TasteFinding(
                "no-guidance-sentences",
                "copy",
                "low",
                "no full-sentence guidance in the interface",
                "explain the next step in one sentence under the title or above the form",
            )
        )
    return findings


def _anti_pattern_findings(css: str) -> list[TasteFinding]:
    findings: list[TasteFinding] = []
    if _count(r"!important", css) > 6:
        findings.append(
            TasteFinding(
                "important-wall",
                "anti_patterns",
                "medium",
                "more than three !important declarations",
                "raise specificity with component classes instead of forcing overrides",
            )
        )
    for rule, pattern in LEGACY_SMELLS:
        if re.search(pattern, css, flags=re.I):
            findings.append(
                TasteFinding(
                    rule,
                    "anti_patterns",
                    "critical" if rule in {"fixed-width-container", "bootstrap-cdn"} else "medium",
                    f"legacy template smell: {rule.replace('-', ' ')}",
                    "use a fluid max-width shell and the token layer instead of a fixed wireframe",
                )
            )
    return findings


def _dimension_findings(css: str, html: str, js: str, config: str) -> list[TasteFinding]:
    return (
        _design_findings(css)
        + _responsiveness_findings(css, html)
        + _state_findings(css, js)
        + _accessibility_findings(css, html, js)
        + _typography_findings(css)
        + _copy_findings(html, js, config)
        + _anti_pattern_findings(css)
    )


def _score_dimensions(findings: list[TasteFinding]) -> dict[str, int]:
    scores = {name: 100 for name in DIMENSION_WEIGHTS}
    penalty = {"critical": 55, "high": 28, "medium": 14, "low": 6}
    for item in findings:
        scores[item.dimension] = max(0, scores[item.dimension] - penalty[item.severity])
    return scores


class TasteCritic:
    """Score a rendered project's product surface from its own assets."""

    def __init__(self, threshold: int = 70):
        self.threshold = threshold

    def review(self, root: str | Path) -> TasteReport:
        """Review ``web/`` (or the repo root) and return a scored report."""
        base = Path(root)
        web = base / "web"
        if not web.is_dir():
            web = base
        files = {
            "css": _read(web, "app.css"),
            "html": _read(web, "index.html"),
            "js": _read(web, "app.js"),
            "config": _read(web, "config.js"),
        }
        reviewed = sum(1 for text in files.values() if text)
        if not files["css"] and not files["html"]:
            return TasteReport(
                score=0,
                dimensions={name: 0 for name in DIMENSION_WEIGHTS},
                findings=[
                    TasteFinding(
                        "no-ui",
                        "design_system",
                        "critical",
                        "the project ships no web assets to review",
                        "generate a UI (index.html + app.css + app.js) before claiming it is an app",
                        "(project)",
                    )
                ],
                files_reviewed=0,
                threshold=self.threshold,
            )
        findings = _dimension_findings(files["css"], files["html"], files["js"], files["config"])
        dimensions = _score_dimensions(findings)
        total = round(sum(dimensions[name] * weight for name, weight in DIMENSION_WEIGHTS.items()))
        return TasteReport(
            score=max(0, min(100, total)),
            dimensions=dimensions,
            findings=findings,
            files_reviewed=reviewed,
            threshold=self.threshold,
        )


def taste_of(root: str | Path, threshold: int = 70) -> TasteReport:
    """Convenience wrapper used by the corpus, the suite and the repair loop."""
    return TasteCritic(threshold=threshold).review(root)
