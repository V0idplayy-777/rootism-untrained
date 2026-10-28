"""Post-deploy health verification.

A deploy is not "done" because an API returned 201. ``wait_for_http`` polls the
public URL until it answers with an acceptable status (or the budget expires)
and reports exactly what happened, so the CLI/UI never claim a live URL that is
actually a 502.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .http import DeployHTTPError, HttpClient


@dataclass
class HealthResult:
    ok: bool
    url: str
    status: int | None = None
    attempts: int = 0
    elapsed: float = 0.0
    detail: str = ""
    body_excerpt: str = ""
    history: list[int | None] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "url": self.url,
            "status": self.status,
            "attempts": self.attempts,
            "elapsed": round(self.elapsed, 2),
            "detail": self.detail,
            "body_excerpt": self.body_excerpt,
            "history": list(self.history),
        }


def check_http(
    url: str,
    client: HttpClient | None = None,
    timeout: float = 10.0,
    expect: tuple[int, ...] = (200, 201, 204, 301, 302, 304),
) -> HealthResult:
    """One request against ``url``; never raises."""

    client = client or HttpClient()
    started = time.monotonic()
    try:
        response = client.get(url, timeout=timeout)
    except DeployHTTPError as exc:
        return HealthResult(
            ok=False,
            url=url,
            elapsed=time.monotonic() - started,
            attempts=1,
            detail=str(exc),
            history=[None],
        )
    except Exception as exc:  # noqa: BLE001 - health checks must not explode
        return HealthResult(
            ok=False,
            url=url,
            elapsed=time.monotonic() - started,
            attempts=1,
            detail=f"{type(exc).__name__}: {exc}",
            history=[None],
        )

    ok = response.status in expect
    body = (response.body or "")[:280].replace("\n", " ")
    detail = "ok" if ok else f"unexpected status {response.status}"
    if not ok and body:
        detail += f": {body[:120]}"
    return HealthResult(
        ok=ok,
        url=url,
        status=response.status,
        attempts=1,
        elapsed=time.monotonic() - started,
        detail=detail,
        body_excerpt=body,
        history=[response.status],
    )


def wait_for_http(
    url: str,
    client: HttpClient | None = None,
    *,
    timeout: float = 90.0,
    interval: float = 3.0,
    expect: tuple[int, ...] = (200, 201, 204, 301, 302, 304),
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    emit: Callable[[str], None] | None = None,
) -> HealthResult:
    """Poll until healthy, or return the last (failed) measurement."""

    client = client or HttpClient()
    started = clock()
    history: list[int | None] = []
    attempts = 0
    last = HealthResult(ok=False, url=url, detail="no attempts made")

    while True:
        attempts += 1
        last = check_http(url, client=client, expect=expect)
        history.append(last.status)
        last.attempts = attempts
        last.history = list(history)
        last.elapsed = clock() - started
        if last.ok:
            last.detail = f"healthy after {attempts} attempt(s)"
            return last
        if clock() - started >= timeout:
            last.ok = False
            last.detail = (
                f"not healthy after {attempts} attempt(s) in {timeout:.0f}s: {last.detail}"
            )
            return last
        if emit is not None:
            emit(f"health: {last.detail} (attempt {attempts}, status {last.status})")
        sleep(interval)
