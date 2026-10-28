# Ejecting: keep the project, drop the tooling

Openness is the product decision this repository is built around. A project made
with CodingAgent-20B must keep running after every file the agent wrote is
deleted, on a machine that has never installed it.

## What the agent writes, and what each file is for

| File | Needed to run? | Safe to delete? |
| --- | --- | --- |
| your application code, `tests/` | yes | no - it is the project |
| `README.md` | no | no - but the `## Live` block can be edited away |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | no (plain Docker) | yes, if you host another way |
| `.env.example` (never `.env`) | no | yes |
| `agentdeploy.toml` | no - preferences only, no secrets | yes |
| `landing/index.html` | no | yes |
| `HOSTING.md`, `DEPLOY.md` | no | yes |
| `.agent/` (map, index, plans, checkpoints, session) | no | yes |

```bash
# remove every trace of the agent
rm -rf .agent landing agentdeploy.toml DEPLOY.md HOSTING.md
```

The application still starts with `docker compose up --build`, `uvicorn`,
`npm start` or whatever its own `README.md` says. Nothing in the generated code
imports `agent`, `deploy`, `scaffold` or `quality`.

## Guarantees

1. **No runtime dependency.** Generated projects use their own frameworks and the
   Python/Node standard library. The auth addon is `hashlib.scrypt` + HS256 in
   ~150 lines; the database layer is `sqlite3` with a `?`→`%s` shim for
   Postgres.
2. **No hosted service in the loop.** Deploys go to your account
   (Vercel/Fly/Render/Coolify/your VPS), or produce a portable artifact. There is
   no CodingAgent backend that has to stay alive.
3. **Credentials stay yours.** Tokens are read from the environment, never
   written to `agentdeploy.toml`, and redacted in logs and in
   `.agent/deploy/last.json`.
4. **Every decision is a file.** The project map is JSON, plans are JSON,
   checkpoints are content-addressed blobs, the deploy plan is a list of steps.
   Read them, edit them, delete them.
5. **Managed options are optional.** `agent backend --provider neon` writes a
   `DATABASE_URL` to `.env` and prints a masked URL; the app runs on SQLite with
   no change when that line is removed.

## Reversing a halfway step

```bash
agent checkpoint list                 # snapshots, newest last
agent checkpoint diff --id <id>       # what changed since
agent checkpoint restore --id <id>    # exact bytes back, extra files removed
```

Checkpoints live in `.agent/checkpoints/`; delete the directory and the
workspace is untouched.

## What is *not* portable

- The trained model weights and the training pipeline (`training/`,
  `model/`). The data pipeline writes plain JSONL and the tokenizer is a
  `tokenizer.json`, both readable without this repo; the training code is
  self-contained PyTorch.
- The chat UI (`ui/`) - it is one small FastAPI app that calls the same modules
  the CLI does; nothing it produces depends on it.
