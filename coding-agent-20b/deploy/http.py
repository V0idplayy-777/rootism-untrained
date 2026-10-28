"""Small HTTP transport used by providers and health checks.

* ``HttpClient`` is the real transport (``requests`` when installed, otherwise
  ``urllib``), with timeouts on every call and secrets redacted from errors.
* ``FakeHttpClient`` records requests and returns scripted responses, so the
  provider tests never touch the network.
* No provider talks to the network directly; passing a different client is the
  only seam needed to test a full deploy end to end.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import redact

try:  # pragma: no cover - exercised implicitly when requests is installed
    import requests

    HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    HAS_REQUESTS = False


@dataclass
class HttpResponse:
    status: int
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body or "{}")
        except json.JSONDecodeError:
            return {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class HttpClient:
    """Thin transport with a uniform interface and no hidden retries."""

    def __init__(self, timeout: float = 30.0, secrets: dict[str, str] | None = None):
        self.timeout = timeout
        self.secrets = secrets or {}

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        timeout = self.timeout if timeout is None else timeout
        headers = dict(headers or {})
        body = data
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        if HAS_REQUESTS:
            try:
                response = requests.request(
                    method, url, data=body, headers=headers, timeout=timeout
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as HttpResponse
                raise DeployHTTPError(redact(f"{method} {url} failed: {exc}", self.secrets)) from exc
            return HttpResponse(
                status=response.status_code,
                body=response.text,
                headers={k.lower(): v for k, v in response.headers.items()},
            )

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    status=response.status,
                    body=response.read().decode("utf-8", errors="replace"),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                body=exc.read().decode("utf-8", errors="replace"),
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
            )
        except Exception as exc:  # noqa: BLE001
            raise DeployHTTPError(redact(f"{method} {url} failed: {exc}", self.secrets)) from exc

    def get(self, url: str, **kwargs) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> HttpResponse:
        return self.request("POST", url, **kwargs)


class DeployHTTPError(RuntimeError):
    """Network-level failure, already redacted."""


class FakeHttpClient(HttpClient):
    """Scripted transport for tests: register exact ``(METHOD, url)`` responses."""

    def __init__(self, responses: dict[tuple[str, str], Any] | None = None, default: Any | None = None):
        super().__init__()
        self.responses = dict(responses or {})
        self.default = default or HttpResponse(status=404, body='{"error":"not scripted"}')
        self.calls: list[dict[str, Any]] = []
        self.handler: Callable[[str, str, Any, dict], HttpResponse] | None = None

    def request(self, method, url, *, json_body=None, data=None, headers=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json_body,
                "data": data,
                "headers": dict(headers or {}),
            }
        )
        if self.handler is not None:
            return self.handler(method, url, json_body, dict(headers or {}))
        for (m, pattern), response in self.responses.items():
            if m.upper() == method.upper() and (pattern == url or (pattern.endswith("*") and url.startswith(pattern[:-1]))):
                return response
        return self.default
