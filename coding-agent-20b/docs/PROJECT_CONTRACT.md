# Generated Project Contract

This document defines what the agent is allowed to write and what must remain pure application code.

## Guarantee

- **Every generated project runs without the agent.** Nothing imports `agent.*`, `scaffold.*`, or any tool runtime. Delete `.agent/`, `landing/`, `status/`, `DEPLOY.md`, `HOSTING.md` and the app still builds and runs from its own README.
- **Eject is one command:** `agent eject --check` proves zero agent files are needed.

## What the agent MAY write

- Source code in your project (`app/`, `src/`, `web/`, `jobs/`, etc.)
- `requirements.txt` / `pyproject.toml` / `package.json` (pinned where possible)
- `Dockerfile`, `.dockerignore`, `docker-compose.yml` (standard, no agent base image)
- `tests/` (starter tests; never overwrite hand-written tests unless --force)
- `agentdeploy.toml` (port, health_path, provider; never secrets)
- `.env.example` (placeholder URLs/keys, never real secrets)
- `.gitignore` additions (`.env`, `data/`, `__pycache__/`, etc.)
- `.agent/` (metadata: scaffold.json, project_map.json, checkpoints — all optional)
- `README.md` **Live block** delimited by `<!-- agent-deploy:start -->` ... `<!-- agent-deploy:end -->` (idempotent)
- `DEPLOY.md`, `HOSTING.md`, `MIGRATIONS.md`, `landing/index.html`, `status/index.html`

## What the agent MUST NOT write

- Secrets in `agentdeploy.toml`, `README.md`, `DEPLOY.md`, or any committed file. Secrets live only in `.env` (chmod 0600, gitignored) or host secret stores.
- `AGENT_RUNTIME` imports in application code. Application modules must not `import agent.*`.
- Overwrites: existing files are never overwritten unless `--force` and then only non-existent files are skipped with a warning.
- Network calls without explicit user intent (except local verify/health checks and allowed provider APIs).
- Code with known critical findings that the critic would flag (hard-coded secrets, `eval(user_input)`, empty catch swallowing errors).

## Eject check

Run:

```bash
agent eject --check
```

This command:
1. Lists agent-owned files (`.agent/`, `landing/`, `status/`, `agentdeploy.toml`, `DEPLOY.md`, `HOSTING.md`)
2. Temporarily moves them aside
3. Runs `agent verify` (tests) and `docker build` dry-run
4. Reports whether the project still runs

If the check passes, you can safely delete:

```bash
rm -rf .agent landing status agentdeploy.toml DEPLOY.md HOSTING.md
# keep: Dockerfile, docker-compose.yml, .dockerignore, README.md, your code/tests
```

## No Agent Runtime Required

Projects carry a badge/check:

```
✓ No agent runtime required — `agent eject --check` passed
```

The CI gate `tests/test_scaffold.py` and `tests/test_cli.py` enforce this: every starter's tests run in isolation without importing the agent.

## Contact

If a generated file violates this contract, file an issue with the reproduction: `agent doctor --json` + `agent map --json` output.
