"""Quality loop: tests first, verify until green, then review before shipping.

* :mod:`quality.critic` - deterministic static review with actionable findings.
* :mod:`quality.verify` - run tests, parse failures, iterate to green.
* :mod:`quality.testgen` - generate starter tests for untested modules.
* :mod:`quality.errors` - plain-language explanations of raw failures.

Nothing here needs a model: the loop works with any ``generate_fn`` seam or none
at all (review/verify are useful on their own).
"""

from __future__ import annotations

from .critic import CodeCritic, CriticLoop, Finding, Review
from .errors import HumanError, humanize, humanize_result
from .testgen import generate_tests, write_generated_tests
from .verify import TestRun, VerificationLoop, detect_test_command, run_tests

__all__ = [
    "CodeCritic",
    "CriticLoop",
    "Finding",
    "HumanError",
    "Review",
    "TestRun",
    "VerificationLoop",
    "detect_test_command",
    "generate_tests",
    "humanize",
    "humanize_result",
    "run_tests",
    "write_generated_tests",
]
