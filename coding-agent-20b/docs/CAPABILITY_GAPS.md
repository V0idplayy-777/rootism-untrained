# Capability gaps, mitigations, and how each is measured

Known weaknesses were identified for this stack. This document records, for
each one: the failure mode, the mechanism that now addresses it, how to verify
the mitigation, and what is still *not* solved. Nothing here claims a model
quality jump that cannot be measured from this repository — the deliverables are
training data, training/eval harnesses and agent-side machinery.

| # | Gap | Mechanism | Measured by |
|---|-----|-----------|-------------|
| 1 | Tokenizer: 320-token placeholder, code-aware pre-tok but not competitive | Streaming BPE on a real corpus + exact vocab alignment + competitive gate | `scripts/evaluate_tokenizer.py`, `validation.json`, `benchmark.json` |
| 2 | 8k/32k context vs real repos | AST+block chunking, hybrid multi-query retrieval + call graph, hierarchical summaries, token-budgeted assembly, 64k YaRN profile | `evaluation/retrieval_ablation.py` (gold-file recall, per-task feature provenance) |
| 3 | Multi-step tool use: recovery, 15–20 call planning, session state | `SessionState`, `ContextWindow` compaction, pin-protected `WorkingSet` excerpts, mid-task `read_file`, deploy/schema-aware observation compaction, `RecoveryPolicy` ladder, `StallDetector`, resumable state | `evaluate_trajectories` (`recovery_after_3`, `loop_rate`, `plan_completion`, `state_coherence`) + `tests/test_context_state.py` |
| 4 | Security-critical generation lags general coding | Curated safe/vulnerable pairs (CWE-tagged), review rubric, security SFT/DPO, mixture upsampling | `evaluation/security_eval.py` (`overall_vuln_rate`, per-CWE counts) |
| 5 | Training distribution: unusual phrasing, distant concepts | Phrasing styles, 19 cross-domain families, constraint conflicts, multi-turn clarification, error recovery, real-code induction | `dataset/augmentation.py` + `dataset/instruction_induction.py` tests + SFT mixture composition |
| 6 | Long functions / deep reasoning need scratch space | Plan → skeleton → per-unit fill, explicit scratch budget, execution-checked self-consistency | `model/longform.py` traces + `ReasoningBudget` |
| 7 | "Idea → running app": no whole-product corpus, no design gate, no end-to-end scoreboard | 65 realistic blueprints, full-project renderer (schema/auth/API/SPA/tests/deploy), taste critic, preference + repair corpus, `agent app` build→verify→repair path | `python -m evaluation.app_suite --split all` (65/65 running apps) + `dataset/app_corpus.py` rows with recorded test evidence |

---

## 1. Tokenizer

**Problem.** The shipped `CodeTokenizer` is a 79-special + 256-byte vocabulary
(335 entries) with no merges: useful as a code-aware *pre*-tokenizer, useless as
a production vocabulary. A BPE run that silently produces fewer merges than
requested also produces a vocab smaller than the model's embedding table.

**What changed.**

- `scripts/train_production_tokenizer.py` streams `--data_dir` **or** the JSONL
  shards from `scripts/sample_tokenizer_corpus.py` (no corpus cap), applies the
  size/SPDX gates, dedupes by content hash, caps documents per language, and
  writes `manifest.json` with a corpus SHA-256.
- `EfficientBPETrainer` pads every trained vocab to exactly `vocab_size` with
  deterministic `<|reserved_N|>` tokens; `CodeTokenizer.pad_to_size` and
  `CodeTokenizer.load` both fail loudly on size/id-map corruption.
- New reserved specials for the rest of this work: `<|scratchpad|>`, `<|plan|>`,
  `<|repo_map|>`, `<|retrieval|>`, `<|file_chunk|>`, `<|diff|>`, `<|review|>`,
  `<|security|>`, `<|clarify|>`, `<|tool_error|>`, … Existing ids (including
  `<|tool_call|>`=48) are frozen.
- Diff-aware pre-tokenization: `@@ … @@` hunk headers and `---`/`+++` file
  markers are single pre-tokens, so patch output is not shredded into
  punctuation.
- `TokenizerValidator` now reports per-language fertility, whole-identifier
  rate, whitespace/tab round-trip, and unified-diff chars/token.
- `evaluation/tokenizer_bench.py` compares against real baselines (bytes,
  word/punct, and `tiktoken` when installed) with absolute fertility floors, so
  an untrained vocabulary cannot pass.

**What changed since (real corpus + Rust engine + exact parity).**

- `scripts/fetch_real_corpus.py` builds the corpus from pinned GitHub
  `repo@tag` sources (270 accepted, 16 correctly rejected by the fail-closed
  SPDX gate): **281M characters / 41,972 documents / 17 languages**, plus 947
  held-out raw files. No HuggingFace datasets dependency.
- `model/fast_bpe.py` `FastBPETrainer` trains with the Rust `tokenizers`
  engine (128k merges in ~20 CPU-minutes), converts the learned merges
  GPT-2-alphabet -> repo-alphabet losslessly, and pads to exactly
  `vocab_size`. `verify_parity` refuses to return a tokenizer whose Python
  port disagrees with the Rust original on held-out documents.
- The shared pre-tokenizer is **engine-independent**: Python `re` and the
  Rust regex crate disagree on `.` vs newline, `\s` on `\x1c-\x1f`, and `\w`
  on combining marks vs numeral letters - the first parity run failed on
  exactly this (Rust string continuations). Both patterns now use explicit
  character classes only; parity is bit-exact on all 947 held-out files.
- `iter_corpus` round-robins languages so a `--max_corpus_chars` cap keeps
  the language mix balanced (the 281M-char run OOMs at ~3.6G RAM; 150M fits).
- `evaluation/tokenizer_ablation.py`: code-aware vs generic GPT-2
  pre-tokenization at equal vocab. Atomicity wins on strings (-11%), comments
  (-7%), numbers (-24%), identifiers; fertility within a documented 1.25x
  bound (measured structural cost ~1.19-1.20x at 16k/32k vocab - the generic
  pattern groups punctuation runs and word-leading spaces, which compresses
  mixed prose/code better at any vocab; we pay that for whole literals).

**Verification.**

```bash
python scripts/fetch_real_corpus.py --output_dir <corpus> --heldout_dir <heldout>
python scripts/train_production_tokenizer.py --data_dir <corpus> \
    --output_dir tokenizer --vocab_size 128256 --max_corpus_chars 150000000 \
    --benchmark_gate          # parity + validation + baseline gates
python evaluation/tokenizer_ablation.py     # code-aware vs generic pretok
```

**Still open.** tiktoken baselines are unreachable in this environment
(encoding files blocked), so the gate compares against bytes / word-punct
floors; the ablation documents the fertility cost explicitly.

## 2. Context: repositories larger than the window

**Problem.** SWE-bench Lite is in the protocol, but 8k native (32k YaRN) cannot
hold a repository, so fixes stay local and architectural changes lose.

**What changed.**

- `agent/repo_context.py` builds an in-memory index: Python chunked by AST
  (module header, top-level defs/classes, methods of large classes), other
  languages by blank-line blocks; identifier-aware tokenisation
  (`getUserById` → get/user/by/id; `parse_tool_call` → parse/tool/call).
- BM25 retrieval, exact/suffix symbol lookup, import-graph expansion
  (`related_files`), per-file outlines, and a **budgeted hierarchical repo map**
  (repo → file → signatures, focused on retrieved files).
- `ContextAssembler` packs map + chunks + neighbours into a hard token budget
  with provenance, in three ablation modes: `none`, `flat`, `hierarchical`.
- **Hybrid multi-query retrieval**: `search_hybrid` fuses BM25 and
  character-ngram cosine via reciprocal rank fusion; `search_task` expands a
  task into complementary queries (prose, identifiers, paths, pasted error
  text, domain synonyms - "logging in" -> login/authenticate/session) and
  merges rankings. Paraphrase recall is where plain BM25 loses.
- **Call graph**: `callers_of` / `callees_of` / `related_by_calls`
  (import + call resolution, bounded depth), so opening one file pulls in the
  code it actually talks to - and not the unrelated subtree.
- `HierarchicalSummarizer` (file -> directory -> repo, extractive and
  deterministic, optional LLM seam) and `ContextPack.features` record *which*
  ingredients fired (queries, hybrid, call_graph_files, file_summaries,
  repo_summary) for every assembled pack.
- `evaluation/retrieval_ablation.py` computes gold-file recall per mode at
  equal budgets and records per-task feature provenance (the report shows
  which retrieval ingredient stopped firing, not just aggregate recall).
  Gold files are used **only for measurement**, never in the prompt.
- `model/config.py`: `get_20b_64k_config()` (YaRN ×8) as headroom for a few
  large files, documented as extrapolation, not native long context.

**Verification.**

```bash
python evaluation/retrieval_ablation.py
# on this repository (60 symbol-anchored tasks, 3k-token budget):
# flat recall 0.967 | hierarchical 0.983, hybrid fired 60/60,
# call-graph neighbours 1.47/task - gate: hierarchical >= flat
```

**Still open.** No embeddings (character-ngram cosine is the fuzzy side);
repo maps are signature-level, not semantic. Long-context quality beyond 32k
remains unproven without long-context training.

## 3. Long-horizon tool use

**Problem.** 15–20+ calls, many failures, and the model loses the plan, repeats a
broken call, or ignores a constraint stated ten steps ago. The eval protocol
already tracked recovery rate because this was expected to be weak.

**What changed.**

- `agent/session.py`: `SessionState` (plan, todos with **evidence**, pinned
  facts, failures, files, test results, scratchpad) with atomic JSON save/load
  and resume; `ObservationCompactor` (per-tool summaries: failing test ids, first
  assertion error, changed files); `ContextWindow` with two-level compaction
  (pinned layer → one-line step digests → recent steps) inside a hard budget.
- `agent/recovery.py`: ladder `retry → repair → change → switch → reflect →
  backtrack → escalate`, normalised failure signatures (line numbers, paths,
  addresses, hex ids removed), tool fallback chains, `StallDetector` for
  spinning, and an honest escalation answer that reports blockers instead of
  claiming success.
- `agent/tools.py`: `revert` (per-file `git checkout --` or stash incl.
  untracked) for backtracking; `update_plan` / `note_fact` are control tools
  handled by the agent, not the sandbox.
- `ProductionAgent` wires all of it: plan-first prompt, injected repo context,
  recovery prompts, stall checks, resumable state, and a `generate_fn` seam so
  the loop is testable without a GPU.
- `evaluate_trajectories` adds `recovery_after_3`, `hard_trajectories`,
  `loop_rate`, `plan_completion`, `state_coherence`, `escalations`,
  `backtracks`.

**What changed since (plan DAGs, milestones, memory, scratch).**

- `agent/planning.py` `PlanGraph`: plans are dependency DAGs, not lists -
  `[after 1,3]` markers declare deps, plain steps chain, bare `[after]`
  declares an explicit root. `ready()` gates execution, `downstream()` +
  `reset_subtree()` scope a *local* replan to the invalidated subgraph
  (verified sibling work survives), and `MilestoneCheckpointer` snapshots
  verified steps (files + tests + git) so `BACKTRACK` returns to the last
  verified milestone, not to zero.
- `AssumptionLedger`: unverified assumptions are injected into **every**
  prompt until confirmed/refuted; a refutation forces a replan note naming
  the wrong assumption.
- `agent/memory.py` `MemoryStore`: cross-session hints (facts, failure
  lessons, test commands) with usage-bounded recall, `render()` into the
  pinned task block, and `learn_from_session()` over the finished session.
- `agent/scratch.py` `ScratchWorkspace`: disk-backed reasoning topics
  (append/replace, per-topic cap with newest-tail pruning, search + index).
  The **index**, not the content, stays in the prompt - intermediate work
  survives context compaction.
- `ProductionAgent` gained the control tools `record_assumption` /
  `verify_assumption` / `scratch_write` / `scratch_read` / `scratch_list`;
  `update_plan` now builds the PlanGraph and checkpoints milestones on
  evidence-verified todos; memory + scratch index + assumption ledger are
  injected each step; WRAP_UP fires near the step budget; and plan graph,
  assumptions and milestones persist to `<state>.extras.json` on both exits.
- `agent/recovery.py` hardening: environment failures switch immediately,
  code failures walk CHANGE at 2 repeats and BACKTRACK at 3 (consecutive 4),
  with cooldowns; `evaluate_trajectories` gained `wrap_ups`, `milestones`,
  and `recovery_class_accuracy` (did the intervention match the failure
  class).

**Verification.** `tests/test_agent_loop.py` (scripted trajectories end to
end) plus `tests/test_agent_long_horizon.py` (scratch tools persist and
inject, assumption refutation forces a replan note, dependency markers parse,
verified todos checkpoint milestones, wrap-up at budget exhaustion, memory
learn/recall across agent instances, extras round-trip) and
`tests/test_planning_memory_scratch.py` (module contracts).

**Still open.** Recovery is policy-driven, not learned; it bounds damage but
cannot invent a correct hypothesis. Milestone rollback assumes
`files_touched` is accurate - a step that edits untracked files it never
reports cannot be reverted precisely.

## 4. Security-critical generation

**Problem.** General coding ability does not transfer to crypto/auth/sandbox
code; public corpora are full of `md5`, `verify=False`, `shell=True`.

**What changed.**

- `dataset/security_data.py`: 18 curated CWE-tagged samples (password KDFs,
  constant-time compare, SQL injection, IDOR, path traversal, command injection,
  CSPRNG, pickle/YAML, JWT alg pinning, TLS verification, SSRF allow-lists,
  hard-coded secrets, `eval`, XXE, zip-slip, AEAD vs ECB, temp files), each with
  vulnerable **and** safe code plus the reason. Generators emit review / repair /
  implement SFT rows and DPO pairs (`chosen`=safe, `rejected`=vulnerable).
- `dataset/mixture.py`: `security-sft` (weight 0.03, upsample ×3);
  `configs/pipeline.json` upsamples it in midtrain and SFT.
- `evaluation/security_eval.py`: regex + AST scanner (weak hashing, injection,
  disabled TLS, `pickle`, `eval`, shell, ECB, `random` for secrets, hard-coded
  credentials, JWT, XXE, zip-slip, IDOR, temp files), plus **required safe
  patterns** per task ("did not use md5" ≠ "used Argon2"). Scores vulnerability
  rate per CWE and fails a threshold.
- `security_review_rubric()` is both training text and the scoring checklist.

**What changed since (held-out split + adversarial requests).**

- The curated set is now **38 CWE-tagged samples** (adds regex DoS, crypto
  modes, CSRF, unbounded allocation, auth fail-open, zip-slip depth, and 4
  `adversarial_request` cases where the correct behaviour is a **refusal
  with a safe alternative** - "disable TLS verification to unblock the
  release" must not be obeyed).
- A **sha256-deterministic split** yields 32 train / 6 held-out (held-out:
  zip-slip, crypto-mode, csrf-missing, redos-nested, unbounded-alloc,
  auth-fail-open). `SecurityDataGenerator(heldout=False)` trains on the
  train split only; `SecurityBenchmark.cases_from_dataset(heldout=True)` is
  the held-out default, so training data can never leak into the gate.
- Scanner extended to 25 regex + 11 guard rules across the new categories,
  with **required safe patterns** per task and explicit overrides - "did not
  use md5" is not "used Argon2". Contract is verified in **both directions**
  on all 38 samples (safe passes, vulnerable flags).

**Verification.** `tests/test_security_augment_longform.py` asserts the split
contract (disjoint, generator = train-only) and both-direction scanner
behaviour. `SecurityBenchmark.assert_threshold` gates a release (default
≤10% vulnerable, 0 critical).

**Still open.** Static rules cannot prove absence of vulnerabilities (SSRF needs
taint analysis); the benchmark is a floor, not a certification. Human review
remains mandatory for crypto/auth changes.

## 5. Training distribution

**Problem.** Synthetic instructions + The Stack cover common patterns; unusual
phrasing, ambiguity, and cross-domain composition expose the edges.

**What changed.** `dataset/augmentation.py` + `dataset/instruction_induction.py`:

- eight phrasing styles (terse, casual, verbose, constraint-first, non-native,
  typo, frustrated, partial repro) that preserve technical content;
- `ConceptCombiner`: **19 cross-domain task families** (compilers×billing,
  DB×ML, storage×security, parsing×privacy, queues×reliability, scheduling×
  finops, games×networking, pipelines×compliance, …) each with a
  **compilable** skeleton, explicit requirements and acceptance criteria;
- `AbstractionLadder`: the same task as spec / design brief / implementation;
- `AmbiguityGenerator`: underspecified requests plus the targeted clarification
  a good assistant must ask;
- `ConstraintConflictGenerator`: requests with **conflicting requirements**
  (strict ordering AND 500/s concurrent; append-only audit AND redaction) where
  the correct output names both constraints, explains the tension, proposes the
  standard reconciliations and asks the deciding question - instead of silently
  picking whichever constraint was read last;
- `ClarificationDialogue`: full multi-turn dialogues (ask → answer →
  implement) because single-turn data teaches asking but not *using the
  answer*;
- `RecoveryInstructionGenerator`: failed tool call → the **changed** next
  approach with the failure class named (the offline twin of
  `agent/recovery.py`; the anti-pattern is recorded in metadata);
- `dataset/instruction_induction.py`: SFT rows induced from **real
  licence-clean corpus files** - docstring→code, signature→body (incl. class
  methods), code→**extractive** explanation (every claim verifiable in the
  source), test→contract. Provenance (path + SPDX licence + induction kind)
  rides on every row; the licence gate reuses the fetcher allowlist.
- Mixture: `instruction-induction` (0.03, ×1.5) and `recovery-sft`
  (0.02, ×2) join `creative-sft` and `security-sft`; `configs/pipeline.json`
  wires them into SFT sources and upsampling.

**Verification.** `tests/test_augmentation_expansion.py` (compilable
skeletons across all families, conflict rows name the tension, dialogues are
multi-turn, recovery rows preach a changed approach, determinism) and
`tests/test_instruction_induction.py` (shapes, licence fail-closed,
provenance, determinism).

**Still open.** Template-derived diversity is not human diversity; the
induced rows are extractive by design (no invented explanations), which caps
their upper bound. This expands coverage; it does not replace real user
traffic or red-teaming.

## 6. Long functions and deep reasoning

**Problem.** An 8k window with a 1–2k generation budget cannot emit or reason
through a large artifact; intermediate steps get silently dropped.

**What changed.** `model/longform.py`:

- `ReasoningBudget` splits the window into plan / scratch / answer with explicit
  minimums (and shrinks scratch when the window is tight);
- `LongFormSynthesizer` runs plan → interface skeleton → per-unit fill (each fill
  sees the plan and neighbour **signatures**, not bodies) → AST validation →
  bounded repair; `trace` records every step and `SynthesisResult.ok` is
  compile-based, not vibes-based;
- `missing_definitions` lints undefined calls; `split_long_function` proposes
  extraction points for over-long functions instead of truncating;
- `reason()` implements scratchpad reasoning with execution-checked
  self-consistency, falling back to majority agreement when no checker exists;
- `reason(scratch=...)` adds a **disk-backed scratch seam**: every sample's
  scratchpad is appended to a topic, and later samples see a compact summary
  of prior attempts (answers + a disagreement note) instead of full
  transcripts - deep reasoning is then not bounded by the context window,
  because anything written can be re-read while the window only pays for the
  index. `ReasoningBudget.external_scratch` documents the mode.

**Verification.** Tests cover budget scaling with 8k vs 32k, plan parsing,
per-unit filling on a scripted generator, the repair loop, and self-consistency
selection (execution check beats majority when they disagree).

**Still open.** Decomposition quality is bounded by the planner; if the plan is
wrong, the units compose a plausible but incorrect module. The validator checks
syntax and obvious undefined names — not semantics. Run the tests.

---

## 7. Shipping: from a repository to a running, reachable app

Gaps 1–6 make the *model* better at writing code. None of them gets a user from
"I want this" to "here is the URL", because that needs scaffolding, verification,
deployment and a surface a non-engineer can use. This section tracks that work.

| # | Gap | Mechanism | Verify |
|---|-----|-----------|--------|
| 7.1 | No end-to-end deploy | `deploy/`: detection → Dockerfile/compose → provider → health check → landing/README/report | `tests/test_deploy.py` |
| 7.2 | Only ad-hoc single files | `scaffold/`: six starters, optional auth/DB, managed Postgres, intent routing | `tests/test_scaffold.py` |
| 7.3 | No review/test loop | `quality/`: critic, test runner + failure parsing, generated tests, humanised errors | `tests/test_quality.py` |
| 7.4 | No project memory | `agent/project_map.py`, `agent/semantic_index.py`, `agent/checkpoint.py`, `agent/plan_approval.py` | `tests/test_project_memory.py` |
| 7.5 | No data for the above | `dataset/project_data.py` (multi-file rows), `dataset/repair_traces.py` (fail→fix traces) | `tests/test_project_data.py` |
| 7.6 | No non-engineer surface | `ui/` chat (simple/power, SSE, same-origin preview) + `cli.py` (`agent …`) | `tests/test_ui.py`, `tests/test_cli.py` |

**Deploy honesty.** A deploy is only `ok` when the health endpoint answered
(2xx/3xx) within the budget. `ProviderResult.ok=False` carries a concrete error;
`humanize()` turns it into a sentence plus next steps; logs and
`.agent/deploy/last.json` are redacted. If no host is configured the `archive`
provider produces a portable `.tar.gz` + `Dockerfile` and the result says
`url=None, public=False` rather than inventing a URL.

**Why the starters are deterministic.** Templates are pure
`ProjectOptions -> {path: content}`, so a starter is reproducible, diffable and
trainable on (the dataset rows are renders of the same templates). Regeneration
costs nothing and no model is needed to prove the output runs: every starter's
own test suite is executed in CI (`tests/test_scaffold.py`).

**What is still not solved.**

- Live deploys to Vercel/Fly/Render are exercised against scripted HTTP and
  stubbed CLIs (`FakeHttpClient`, patched `which`); they are not run against real
  accounts in this repository, so provider API drift is possible.
- `local` needs a Docker daemon; without one the fixture set covers the archive
  path instead.
- The UI's live preview proxies a child process on a free port; it is verified
  with a static site and a real FastAPI process, not with every starter.
- Plan approval gates *mass* writes from the UI/CLI; a model calling `write_file`
  directly is still bounded by the workspace, not by the plan.

---

## Acceptance criteria for the next milestone

1. Tokenizer trained on ≥20B characters of permissively licensed code, gate
   passing against `tiktoken:cl100k_base` on Python/TS/Rust/Go/diff.
2. `RetrievalAblation` shows hierarchical recall ≥ flat recall at equal or lower
   token cost on ≥100 SWE-bench Lite instances.
3. Agent trajectories (n≥100, max 40 steps): `recovery_after_3` ≥ 0.4,
   `loop_rate` ≤ 0.05, `state_coherence` ≥ 0.95.
4. Security suite: vulnerable rate ≤ 5% overall, 0 critical findings, with a
   held-out set not used in training.
5. Clarification tasks: model asks before coding on ≥80% of ambiguous prompts
   and still solves the unambiguous paraphrase variants.


---

## 7. Idea → running app

**Problem.** The stack could scaffold starters and review diffs, but it had no
corpus of *whole products*, no way to judge whether an interface was finished,
and no end-to-end measure of the product promise. A model can pass unit tests
and still ship a fixed-width 2018 template with no empty state.

**What changed.**

- `scaffold/blueprints.py`: 65 product blueprints (typed fields, relations,
  enum/bool/date coverage, auth on/off, and a board/schedule/team/stats feature),
  with a stable hash split (47 train / 18 held out) and a prompt matcher.
- `scaffold/blueprint.py` + `blueprint_web.py`: a renderer for a complete,
  runnable project — schema, migrations, scoped store, API, auth, a
  dependency-free SPA with a real token layer, tests, Docker/compose/deploy
  config, README.
- `quality/taste.py`: a design gate with named findings (tokens, dark mode,
  breakpoints, empty/loading/error states, focus, labels, copy, legacy smells).
- `scaffold/blueprint_variants.py`: plausible downgrades used as preference
  pairs and repair traces, never in the default build.
- `dataset/app_corpus.py`: SFT, preference, repair and taste rows; `--verify`
  records real pytest evidence; held-out ids are excluded.
- `evaluation/app_suite.py`: six gates per prompt, including booting uvicorn and
  checking `/healthz` and `/`.
- `scaffold/app_builder.py` + `agent app`: build → test → critic → taste → deploy
  plan, a loud failure with the failing assertion, and a deterministic repair
  that restores generated files with a backup.
- Mixture: app sources are 25% (was 3% for scaffold+repair); pipeline configs
  upsample the same sources in midtrain, SFT and DPO.

**How it is verified.**

```bash
python -m pytest tests/ -q
python -m evaluation.app_suite --split all
python -m evaluation.app_suite --guards
python -m dataset.app_corpus --out data/app-corpus --verify
agent app "a team task list where I can share tasks with teammates"
```

Last runs: **284 passed, 6 skipped**; suite **65/65**; guards reject each variant
for the intended reason; verified corpus rows carry real pytest output.

**Still not solved.**

- Taste is heuristic, not perceptual: it cannot judge visual originality.
- The catalogue is small-product shaped; long-tail domains fall back to starters.
- Repair is restoration of generated modules, not general program synthesis.
- Hosted deploys are planned and locally booted, not exercised against live
  provider APIs in CI.
