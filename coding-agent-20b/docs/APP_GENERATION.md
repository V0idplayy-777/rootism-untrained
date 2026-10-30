# Idea → running app

The product promise is narrow and testable: **describe an app, get a working,
tasteful, deployable full-stack product that you own.** This document covers the
four pieces that make it real, how they are measured, and what is still missing.

Everything below runs offline on a CPU with Python 3.10+ (`fastapi`,
`pydantic`, `httpx`, `pytest`, `uvicorn`).

## 1. One command

```bash
agent app "a team task list where I can share tasks with teammates"
```

Output (real run):

```
→ Build: a team task list where I can share tasks with teammates
  directory: /home/you/shared-tasks
  files: 28
  tests: passing (20)
  critic: 100/100
  taste: 100/100
  deploy plan: archive (5 steps)

✓ built Shared Tasks (shared-tasks) … 28 files, tests green, critic 100/100,
  taste 100/100, deploy archive
  Preview: python -m uvicorn shared_tasks.main:app --reload --port 8000
  Eject check: agent eject --check
```

The app is a normal repository: FastAPI + SQLite/Postgres, scrypt + HS256 auth,
a dependency-free SPA, pytest suites, `Dockerfile`, `docker-compose.yml`,
`agentdeploy.toml`, `schema.sql`, migrations, `.env.example`, `README.md`.
Nothing at runtime imports this tool.

Flags that matter:

| Flag | Effect |
| --- | --- |
| `--blueprint ID` | skip matching and build a named blueprint |
| `--repair` | restore generated modules once, re-run every gate |
| `--deploy` | deploy only after all gates pass (`--provider` to choose) |
| `--json` | pure JSON report for automation |
| `--taste-threshold N` | raise or lower the design bar |

If a gate fails, the command exits **non-zero**, prints the failing assertion or
missing design rule, and does not print a success banner. `--repair` restores
the generated modules/UI from the blueprint and backs up what it replaced under
`.repair-backup/<timestamp>/`.

## 2. The catalogue is the contract

`scaffold/blueprints.py` holds **65 product blueprints**. Each one is a small,
complete specification:

- 1–3 entities with typed fields: `text`, `longtext`, `int`, `bool`, `timestamp`,
  `date`, `enum`, `ref`;
- relations with `ON DELETE cascade` and parent validation;
- auth on or off (roughly 3/4 require accounts);
- one non-trivial feature: `board`, `schedule` (double-booking detection),
  `team` (sharing), or `stats`;
- a natural prompt a person would type, e.g. *"a clinic appointment book where
  patients can see open slots"*.

The same catalogue drives training data and the evaluation suite. The split is a
stable SHA-256 of the id: **47 train / 18 held out**. Held-out prompts never
appear in corpus rows, so the scoreboard measures generalisation.

## 3. The corpus (`dataset/app_corpus.py`)

One train blueprint yields four row kinds:

| File | Rows (47 train) | What it teaches |
| --- | ---: | --- |
| `app-sft.jsonl` | 47 | request → plan → every file → real verification command |
| `app-preference.jsonl` | 110 | chosen product vs one measured downgrade |
| `app-repair.jsonl` | 110 | broken app + failure output → diagnosis → minimal diff → green |
| `app-taste.jsonl` | 8 | why the interface is shaped this way (tokens, states, copy) |

Generate it:

```bash
python -m dataset.app_corpus --out data/app-corpus            # fast, evidence marked not-run
python -m dataset.app_corpus --out data/app-corpus --verify   # runs every project's tests
```

`--verify` records the actual pytest output, pass counts and failures on every
row. Without it, a row says `verified: false` and `verification.kind=not-run`;
the generator never claims a test result it did not observe.

The three rejected variants are deliberately plausible, not strawmen:

| Variant | What is wrong | Which gate catches it |
| --- | --- | --- |
| `legacy-ui` | fixed 960px container, Helvetica, no tokens/dark mode/breakpoints/states | taste (≈49/100 vs 100/100) |
| `broken-auth` | token verified, but `_scope` returns `1 = 1` | isolation test fails |
| `incomplete-entities` | id column with no foreign key, cascade or parent check | schema + cascade tests fail |

Mixture (`dataset/mixture.py`): whole-product sources are **25%** of the mix
(0.07 `app-sft`, 0.05 `app-repair-sft`, 0.03 `app-taste-sft`, 0.03
`app-preference-dpo`, plus `project-scaffold-sft` 0.05 and `repair-trace-sft`
0.05, up from 0.015 each). `configs/pipeline.json` upsamples the same sources in
mid-training, SFT and DPO.

## 4. The scoreboard (`evaluation/app_suite.py`)

```bash
python -m evaluation.app_suite --split eval           # held-out prompts
python -m evaluation.app_suite --split all --json evaluation/reports/app_suite.json
python -m evaluation.app_suite --guards               # prove the variants are rejected
```

Each prompt must pass **six** gates; there is no partial credit:

1. render: complete project, no placeholders, every Python file compiles;
2. tests: the project's own pytest suite, executed;
3. code critic: score ≥ 80 and no blocking finding;
4. taste: ≥ 70 with no critical rule;
5. deploy plan: a provider and a real step list;
6. running app: uvicorn boots, `/healthz` returns `{"status": "ok"}`, `/` serves
   the SPA.

Current result: **65/65 prompts, 100% success**, 132s wall clock on CPU, with
every gate green. The report is reproducible:

```text
| gate    | passed | failed | skipped |
| ------- | -----: | -----: | ------: |
| critic  |     65 |      0 |       0 |
| deploy  |     65 |      0 |       0 |
| render  |     65 |      0 |       0 |
| server  |     65 |      0 |       0 |
| taste   |     65 |      0 |       0 |
| tests   |     65 |      0 |       0 |
```

## 5. What "taste" means here

`quality/taste.py` scores the product surface from its own assets, with named
findings and fixes:

- **design system** — ≥12 tokens, `:root`, dark scheme, radius scale;
- **responsiveness** — viewport meta, ≥3 breakpoints, a 600–760px phone layout;
- **states** — empty, skeleton, inline error, toast;
- **accessibility** — bound labels, `:focus-visible`, `lang`, `aria-live`,
  reduced motion;
- **typography** — deliberate stack and scale;
- **copy** — product-specific empty sentence, no placeholder text;
- **anti-patterns** — fixed-width container, Helvetica default, 13px body,
  Bootstrap/jQuery CDN, `!important` walls.

It is a heuristic gate, not a vision model: it cannot tell you a layout is
*beautiful*, only that it is intentional and complete. Vision scoring is listed
as future work rather than pretended.

## 6. Honest limits

- The corpus is **synthetic and template-derived** with real test evidence. It
  teaches shape, wiring, state handling and repair behaviour; it does not invent
  novel product logic.
- The blueprint catalogue covers small CRUD/SaaS-shaped apps. Requests outside
  it fall back to the six generic starters or are refused with suggestions.
- `--repair` restores generated modules; it does not debug arbitrary user code.
  Unfixable failures are reported as such.
- No hosted-provider deploy is exercised in CI (no tokens); the suite proves the
  plan, the artifact path and the running app locally.
- The success rate is a statement about **this stack**, measured on this suite,
  not a comparison against any other product's private benchmark.
