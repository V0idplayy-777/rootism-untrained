# CodingAgent-20B

20B coding-agent stack: model, production tokenizer, data pipeline, hardened
agent tools/sandbox, retrieval + long-session state, security data/eval,
evaluation, staged training and serving.

- Model card: `MODEL_CARD.md` · Eval protocol: `docs/EVAL_PROTOCOL.md` · Repro: `docs/REPRODUCIBILITY.md`
- Capability gaps and how each is addressed/measured: `docs/CAPABILITY_GAPS.md`
- Idea → live app in four commands: `docs/QUICKSTART.md` · No lock-in / eject: `docs/EJECT.md`
- Install: `pip install -r requirements.txt` (dev: `requirements-dev.txt`)
- Test: `python -m pytest tests/ -q` (CPU, tiny config — no GPU needed)
- Tokenizer (real corpus + Rust engine + parity/competitive gates):
  1. `python scripts/fetch_real_corpus.py --output_dir <corpus> --heldout_dir <heldout>` — pinned GitHub sources, fail-closed SPDX gate (281M chars / 17 langs proven)
  2. `python scripts/train_production_tokenizer.py --data_dir <corpus> --output_dir tokenizer --vocab_size 128256 --max_corpus_chars 150000000 --benchmark_gate`
     (Rust `FastBPETrainer`; refuses to ship if the Python port disagrees with the Rust original — `verify_parity`)
  3. `python evaluation/tokenizer_ablation.py` — code-aware vs generic GPT-2 pre-tokenization (atomicity gate)
  4. `python evaluation/retrieval_ablation.py` — flat vs hierarchical retrieval recall + feature provenance
- Train: `training/train_production.py` + `configs/pipeline.json` (pretrain → mid-train → SFT → DPO; security and cross-domain SFT sources are upsampled)
- Serve: `python -m inference.server --checkpoint <ckpt> --port 8000` (OpenAI-compatible)
- Docker: `docker compose up serve` (train/eval/serve targets in `Dockerfile`)

## Idea → running app (`agent`)

```bash
pip install -e ".[app]"
agent doctor                                              # toolchain + what is optional
agent app "a team task list where I can share tasks with teammates"  # full product: 28 files
agent app "an appointment book for a salon" --repair      # restore generated modules if a gate fails
agent init --request "a web app where users sign up and save tasks" tasks-app   # generic starter
agent verify                                              # run the tests, explain failures
agent review                                              # score + findings before shipping
agent deploy --provider process                           # or local/vercel/fly/render/coolify
agent preview                                             # chat UI at :7860
```

Nothing above needs a GPU or a hosted service. The pieces:

| Module | What it does |
| --- | --- |
| `scaffold/` | 65 product blueprints + a full-project renderer (schema, auth, store, API, SPA, tests, deploy), six generic starters (`fastapi-react`, `nextjs-fullstack`, `flask-htmx`, `cli-tool`, `discord-bot`, `static-site`), optional JWT auth + SQLite→Postgres. `agent app` = build → test → critic → taste → plan, with a deterministic repair |
| `quality/` | deterministic critic (score, findings, fix prompt), test runner with parsed failure signals, generated starter tests, plain-English error humaniser |
| `deploy/` | detection, Dockerfile/compose generation, eight providers (including a no-Docker local process), verified health check, landing page + README link + `DEPLOY.md`/`HOSTING.md`, redacted logs |
| `agent/project_map.py`, `agent/semantic_index.py` | persistent project map (tree, summaries, symbols) and local hybrid retrieval |
| `agent/checkpoint.py`, `agent/plan_approval.py` | byte-exact snapshot/rollback; plans the user must approve before mass writes |
| `agent/deploy_tool.py` | `deploy` / `scaffold_project` / `project_map` / `search_project` / `checkpoint` as model tools, sharing the CLI code path |
| `ui/` | simple/power chat surface, SSE progress, same-origin live preview |
| `quality/taste.py` | design gate: tokens, dark mode, breakpoints, empty/loading/error states, focus, labels, copy, legacy smells |
| `dataset/app_corpus.py` | 275 rows from 47 train blueprints: app SFT, preference (legacy/broken-auth/incomplete relations), repair traces, taste decisions; `--verify` records real pytest evidence |
| `evaluation/app_suite.py` | fixed idea→running-app scoreboard: render, tests, critic, taste, deploy plan, uvicorn health; **65/65 prompts green** |
| `dataset/project_data.py`, `dataset/repair_traces.py` | multi-file project SFT/DPO rows and fail→diagnose→fix traces (fed by real `VerificationLoop` runs) |

The blueprint path is documented end to end in `docs/APP_GENERATION.md` (catalogue,
corpus, taste rubric, suite, honest limits).

Guarantees: a URL is only reported after the health endpoint answered; a failed
deploy is a failure with a plain-language cause; `agentdeploy.toml` never stores
secrets; every generated file is optional and deletable (`docs/EJECT.md`).

## Working with repositories bigger than the context window

```python
from agent.repo_context import RepoIndex, ContextAssembler
from agent.production_agent import ProductionAgent

index = RepoIndex.build("./repo")
agent = ProductionAgent(model, tokenizer, workspace_path="./repo",
                        context_assembler=ContextAssembler(index, tokenizer),
                        max_steps=40, state_path=".agent/session.json")
result = agent.run("Fix the failing expiry test in tests/test_auth.py")
```

`ProductionAgent` keeps a resumable `SessionState` (plan, todos with evidence,
pinned facts, failures), compacts observations to a hard token budget, and
escalates recovery (`retry → repair → change → switch → reflect → backtrack →
escalate`) instead of repeating a failing call. Pass `generate_fn=` to drive it
from any serving stack, or omit it to use `model.generate`.

Long-horizon machinery is wired in: plans are dependency DAGs
(`update_plan` with `[after 1,3]` markers) with milestone checkpoints on
verified steps, an assumption ledger that keeps unverified assumptions in
every prompt until confirmed or refuted, a disk-backed scratch workspace
(`scratch_write`/`scratch_read`/`scratch_list` — the index, not the content,
sits in the prompt), and a cross-session memory store that learns test
commands and failure lessons from finished sessions. Plan graph, assumptions
and milestones persist to `<state>.extras.json`, so a crashed session
resumes with its structure intact.

## Security and long-form work

```python
from evaluation.security_eval import SecurityBenchmark
from model.longform import LongFormSynthesizer, ReasoningBudget

SecurityBenchmark().assert_threshold(
    SecurityBenchmark().evaluate([{"case_id": "pw-hash", "code": generated}])
)

synth = LongFormSynthesizer(lambda prompt, n: model.complete(prompt, n),
                            budget=ReasoningBudget(context_window=8192))
result = synth.synthesise("config loader with validation and tests")
print(result.ok, result.errors)   # compile-checked, with a per-unit trace
```
