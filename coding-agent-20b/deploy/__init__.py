"""Deploy: turn a workspace into a running (and, where possible, public) app.

Quick start::

    from deploy import DeployEngine, detect_project

    profile = detect_project("./my-app")
    engine = DeployEngine("./my-app", provider="local")
    result = engine.deploy()
    print(result.url, result.health, result.landing)

CLI: ``python -m cli deploy --help`` (``agent deploy`` once installed).
"""

from __future__ import annotations

from .config import DeployConfig, redact
from .detect import ProjectProfile, detect_project
from .dockerfile import write_artifacts
from .engine import DeployEngine, deploy_project, explain_providers
from .health import check_http, wait_for_http
from .landing import publish, update_readme, write_report
from .plan import DeployPlan, DeployResult, DeployStep, ProviderResult
from .providers import available_providers, get_provider, select_provider

__all__ = [
    "DeployConfig",
    "DeployEngine",
    "DeployPlan",
    "DeployResult",
    "DeployStep",
    "ProjectProfile",
    "ProviderResult",
    "available_providers",
    "check_http",
    "deploy_project",
    "detect_project",
    "explain_providers",
    "get_provider",
    "publish",
    "redact",
    "select_provider",
    "update_readme",
    "wait_for_http",
    "write_artifacts",
    "write_report",
]
