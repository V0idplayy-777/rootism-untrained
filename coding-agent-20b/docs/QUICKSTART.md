# Quickstart: idea → running app

Everything below works on a laptop with Python 3.10+ and **no model, no GPU and
no account**. A model, a Docker daemon or a host token unlocks more, and every
step says what it needs before it needs it.

```bash
pip install -e ".[app]"        # agent CLI + chat UI extras
agent doctor                   # what is available here, and what is optional
```

## 1. Create the project

```bash
agent init --request "a web app where people sign up and save tasks" tasks-app
# or pick a starter explicitly
agent init --list
agent init flask-htmx my-tool --db postgres
```

What you get is a real repository, not a snippet: application code, a database
layer, tests, `README.md`, `.env.example`, `.gitignore` and `agentdeploy.toml`.
Starters are optional and interchangeable - nothing in the app imports the
agent.

| Starter | Shape | Good for |
| --- | --- | --- |
| `fastapi-react` | Python API + React UI + SQLite/Postgres | dashboards, SaaS MVPs, anything with logins |
| `nextjs-fullstack` | one Next.js app (UI + API routes) | single-service apps, Vercel |
| `flask-htmx` | server-rendered pages, no build step | internal tools, forms |
| `cli-tool` | argparse CLI + testable core | automation, file utilities |
| `discord-bot` | discord.py worker | community bots |
| `static-site` | one page, no build | landing pages, docs |

Add `--auth` for JWT accounts (stdlib `scrypt` + HS256, no dependency), and
`agent backend --provider neon` to swap SQLite for managed Postgres later.

## 2. Check the work, not the vibes

```bash
agent testgen                 # write starter tests for untested modules
agent verify                  # run the suite, explain failures in plain English
agent review                  # score + findings: secrets, swallowed errors, missing tests
```

`agent verify` returns a non-zero exit code when tests fail and prints the first
real assertion, not a stack-trace wall. `agent review` scores 0–100 and lists
each finding with a file, a line and a fix.

## 3. Ship it

```bash
agent deploy --dry-run                # the plan: what will be created, in order
agent deploy --provider local         # Docker container on this machine
agent deploy --provider vercel        # or fly / render / coolify / github-pages
agent deploy                          # no host configured? portable .tar.gz
```

A deploy writes `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
`landing/index.html`, a `## Live` block in `README.md`, `HOSTING.md` and
`DEPLOY.md`. A URL is only reported after the health endpoint answered; if it
did not, the result says so and explains what to check.

## 4. Or use the chat UI

```bash
agent preview            # http://localhost:7860
python -m ui --workspace ./tasks-app --port 7860
```

Simple mode hides tool calls and file dumps and shows progress, the plan, test
results, the review score and the live URL. Power mode shows the raw event
stream. With no model configured the UI still scaffolds, tests, reviews and
plans a deploy, and says "local workflow (no model)" so nobody is confused about
what is running.

## 5. Work on an existing repository

```bash
agent clone https://github.com/owner/repo   # clone, map, index
agent map --chars 3000                      # tree + per-file summary + symbols
agent search "where is auth handled"        # local hybrid retrieval, no network
agent checkpoint create --label before-refactor
agent plan "move storage to Postgres"       # draft → approve → then write
agent checkpoint restore --id <id>          # exact bytes back
```

`agent plan` stores the plan under `.agent/plans/` and nothing is written until
it is approved (`agent plan --approve <id>`). Checkpoints are content-addressed
blobs, so `restore` is byte-exact and also removes files added since.

## Optional model

```bash
export AGENT_MODEL_URL=http://localhost:8000/v1/chat/completions
agent run "add pagination to /api/items" --mode power
```

The same tools (`deploy`, `scaffold_project`, `project_map`, `search_project`,
`checkpoint`) are exposed to the model, so an agent-driven deploy and a human
one go through identical code paths with identical guarantees.
