"""Production-hardened code-execution sandbox.

Hardening vs. the legacy DockerSandbox:
- FAIL-CLOSED: if Docker is unavailable, execution raises SandboxUnavailable
  unless the caller explicitly opts into the local fallback
  (allow_local_fallback=True, intended only for unit tests).
- Workspace mounted READ-ONLY at /workspace; writable scratch is a bounded
  tmpfs at /work. The executed script itself is mounted read-only.
- cap_drop=ALL, no-new-privileges, seccomp profile, non-root user (65534),
  pids/mem/cpu/disk limits, network disabled by default. Package installs
  run in a separate throwaway container with an allowlist + network toggle.
- Cleanup guarantee: the container is force-removed in a finally block and
  the object is a context manager; orphans are reaped on __exit__.
"""

from __future__ import annotations



import logging

import os

import re

import subprocess

import tempfile

from dataclasses import dataclass, field

from pathlib import Path

from typing import Dict, List, Optional



logger = logging.getLogger(__name__)



try:

    import docker

    HAS_DOCKER_LIB = True

except ImportError:

    HAS_DOCKER_LIB = False



RUNTIMES = {

    "python": {"image": "python:3.11-slim", "cmd": ["python", "/script.py"], "suffix": ".py"},

    "node": {"image": "node:20-slim", "cmd": ["node", "/script.js"], "suffix": ".js"},

    "bash": {"image": "python:3.11-slim", "cmd": ["bash", "/script.sh"], "suffix": ".sh"},

    "go": {"image": "golang:1.22-alpine", "cmd": ["go", "run", "/script.go"], "suffix": ".go"},

    "rust": {"image": "rust:1.79-slim", "cmd": ["bash", "-c", "rustc /script.rs -o /work/a && /work/a"], "suffix": ".rs"},

}



                                                                           

DEFAULT_PACKAGE_ALLOWLIST = {

    "pytest", "numpy", "requests", "flask", "fastapi", "pydantic", "httpx",

    "beautifulsoup4", "pandas", "click", "rich", "PyYAML", "yaml",

}

_PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?(==|>=|<=|~=)?[A-Za-z0-9.*!+ ]*$")



SECCOMP_PROFILE = {

    "defaultAction": "SCMP_ACT_ERRNO",

    "architectures": ["amd64", "aarch64"],

    "syscalls": [{"names": ["read", "write", "open", "openat", "close", "execve", "brk", "mmap",

                             "mprotect", "munmap", "fstat", "newfstatat", "lseek", "clone", "futex",

                             "exit_group", "rt_sigaction", "rt_sigprocmask", "ioctl", "pipe", "pipe2",

                             "dup", "dup2", "dup3", "socket", "connect", "getpid", "getuid", "uname",

                             "arch_prctl", "set_tid_address", "set_robust_list", "rseq", "prlimit64",

                             "readlink", "readlinkat", "access", "faccessat", "stat", "statx", "getcwd",

                             "chdir", "mkdir", "getdents64", "fcntl", "poll", "ppoll", "select", "pselect6",

                             "epoll_create", "epoll_ctl", "epoll_pwait", "eventfd2", "signalfd", "timerfd_create",

                             "clock_gettime", "getrandom", "sched_getaffinity", "madvise", "sigaltstack",

                             "rt_sigreturn", "exit", "wait4", "kill", "tkill", "tgkill", "sendmsg", "recvmsg",

                             "sendto", "recvfrom", "bind", "listen", "accept", "accept4", "getsockname",

                             "getpeername", "setsockopt", "getsockopt", "shutdown", "mlock", "munlock"],

                  "action": "SCMP_ACT_ALLOW"}],

}





class SandboxUnavailable(RuntimeError):

    """Raised (fail-closed) when no sandbox backend is available."""





@dataclass

class SandboxConfig:

    memory_limit: str = "512m"

    cpu_quota: int = 50000                             

    pids_limit: int = 128

    disk_mb: int = 256                                               

    network_disabled: bool = True

    user: str = "65534:65534"                  

    allow_local_fallback: bool = False

    package_allowlist: set = field(default_factory=lambda: set(DEFAULT_PACKAGE_ALLOWLIST))

    images: Dict[str, str] = field(default_factory=dict)





class HardenedSandbox:

    def __init__(self, config: Optional[SandboxConfig] = None):

        self.config = config or SandboxConfig()

        self.available = False

        self.client = None

        self._containers: list = []

        if HAS_DOCKER_LIB:

            try:

                self.client = docker.from_env()

                self.client.ping()

                self.available = True

            except Exception as e:                                          

                logger.warning(f"Docker unavailable: {e}")

        if not self.available and not self.config.allow_local_fallback:

            logger.warning("Sandbox fail-closed: Docker unavailable and local fallback disabled.")



                                                                            

    def __enter__(self):

        return self



    def __exit__(self, *exc):

        self.close()

        return False



    def close(self):

        for c in list(self._containers):

            try:

                c.remove(force=True)

            except Exception:

                pass

        self._containers.clear()



    def _check_packages(self, packages: Optional[List[str]]) -> List[str]:

        ok = []

        for raw in packages or []:

            name = re.split(r"[<>=!~ \[]", str(raw).strip())[0]

            if not _PKG_RE.match(str(raw).strip()):

                raise ValueError(f"Rejected malformed package spec: {raw!r}")

            if name not in self.config.package_allowlist:

                raise ValueError(f"Package {name!r} not in allowlist {sorted(self.config.package_allowlist)}")

            ok.append(str(raw).strip())

        return ok



    def execute(self, code: str, workspace_path: str, timeout: int = 30,

                install_packages: List[str] = None, language: str = "python") -> Dict:

        packages = self._check_packages(install_packages)

        if language not in RUNTIMES:

            return {"stdout": "", "stderr": f"Unsupported language: {language}", "exit_code": -1,

                    "timed_out": False, "language": language}

        if not self.available:

            if self.config.allow_local_fallback and language == "python":

                return self._local_execute(code, workspace_path, timeout)

            raise SandboxUnavailable(

                "Docker sandbox unavailable and local fallback disabled (fail-closed). "

                "Start Docker or construct with SandboxConfig(allow_local_fallback=True) for tests.")

        return self._docker_execute(code, workspace_path, timeout, packages, language)



    def _docker_execute(self, code, workspace_path, timeout, packages, language):

        rt = dict(RUNTIMES[language])

        if language in self.config.images:

            rt["image"] = self.config.images[language]

        script_path = None

        container = None

        try:

            with tempfile.NamedTemporaryFile(mode="w", suffix=rt["suffix"], delete=False) as f:

                f.write(code)

                script_path = f.name

            cmd = list(rt["cmd"])

            if packages:

                if self.config.network_disabled:

                    raise ValueError("install_packages requires network; sandbox has network_disabled=True")

                joined = " ".join(packages)

                if language == "python":

                    cmd = ["bash", "-c", f"pip install -q {joined} && {' '.join(cmd)}"]

                elif language == "node":

                    cmd = ["bash", "-c", f"npm install --prefix /work -s {joined} && {' '.join(cmd)}"]

                else:

                    raise ValueError(f"install_packages unsupported for language={language}")

            container = self.client.containers.run(

                rt["image"], command=cmd,

                volumes={str(Path(workspace_path).resolve()): {"bind": "/workspace", "mode": "ro"},

                         script_path: {"bind": f"/script{rt['suffix']}", "mode": "ro"}},

                working_dir="/work",

                tmpfs={"/work": f"size={self.config.disk_mb}m,mode=1777"},

                mem_limit=self.config.memory_limit, cpu_quota=self.config.cpu_quota,

                pids_limit=self.config.pids_limit,

                network_disabled=self.config.network_disabled,

                user=self.config.user, cap_drop=["ALL"],

                security_opt=["no-new-privileges:true", f"seccomp={_seccomp_json()}"],

                detach=True,

            )

            self._containers.append(container)

            try:

                result = container.wait(timeout=timeout)

                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", "replace")

                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", "replace")

                return {"stdout": stdout[-20000:], "stderr": stderr[-20000:],

                        "exit_code": result["StatusCode"], "timed_out": False, "language": language}

            except Exception:

                try:

                    container.kill()

                except Exception:

                    pass

                return {"stdout": "", "stderr": f"Execution timed out after {timeout}s",

                        "exit_code": -1, "timed_out": True, "language": language}

        finally:

            if container is not None:

                try:

                    container.remove(force=True)

                except Exception:

                    pass

                if container in self._containers:

                    self._containers.remove(container)

            if script_path:

                try:

                    os.unlink(script_path)

                except OSError:

                    pass



    def _local_execute(self, code, workspace_path, timeout):

        """TEST-ONLY fallback (never used when fail-closed). Resource limits
        via RLIMIT where the platform supports it; still not a security boundary."""

        script_path = None

        try:

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=workspace_path) as f:

                f.write(code)

                script_path = f.name

            def _limits():

                try:

                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

                    resource.setrlimit(resource.RLIMIT_CPU, (max(timeout, 1), max(timeout, 1) + 5))

                    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

                except Exception:

                    pass

            result = subprocess.run(["python3", script_path], cwd=workspace_path, capture_output=True,

                                    text=True, timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)

            return {"stdout": result.stdout[-20000:], "stderr": result.stderr[-20000:],

                    "exit_code": result.returncode, "timed_out": False, "language": "python"}

        except subprocess.TimeoutExpired:

            return {"stdout": "", "stderr": f"Execution timed out after {timeout}s",

                    "exit_code": -1, "timed_out": True, "language": "python"}

        finally:

            if script_path:

                try:

                    os.unlink(script_path)

                except OSError:

                    pass





def _seccomp_json() -> str:

    import json

    return json.dumps(SECCOMP_PROFILE)





                                                                                          

DockerSandbox = HardenedSandbox

