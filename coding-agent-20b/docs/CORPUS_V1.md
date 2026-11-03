# Corpus v1

Corpus v1 is the frozen data recipe behind both model sizes. It is defined in
`dataset/mixture.py` and audited by `scripts/corpus_report.py`.

```bash
python scripts/corpus_report.py            # per-slice numbers + invariants
python scripts/corpus_report.py --json     # manifest-diffable
python scripts/corpus_report.py --strict   # exit 1 if any invariant regresses
```

## Composition

13 slices, weights sum to 1.00. By document share: **73% real** permissively
licensed code, **6% hybrid** (real code with an induced instruction), **21%
synthetic**.

| Slice | Weight | Kind | What it is |
| --- | --- | --- | --- |
| `the-stack-code` | 0.48 | real | Permissively licensed source (Stack v2 / StarCoderData / pinned GitHub) |
| `tests` | 0.07 | real | Real unit and integration suites |
| `github-prs` | 0.07 | real | Pull-request diffs and discussion |
| `github-issues` | 0.06 | real | Issue threads |
| `docs-and-comments` | 0.05 | real | Documentation, architecture notes, docstrings |
| `instruction-sft` | 0.06 | hybrid | Real-code induction (8 shapes) + phrasing/cross-domain augmentation |
| `app-sft` | 0.05 | synthetic | Pytest-verified multi-file applications |
| `repair-trace-sft` | 0.04 | synthetic | fail → diagnose → minimal diff → green |
| `security-sft` | 0.03 | synthetic | CWE-tagged safe/vulnerable pairs |
| `app-repair-sft` | 0.03 | synthetic | Broken app → diagnosis → minimal diff |
| `app-preference-dpo` | 0.02 | synthetic | Correctness/taste preference pairs |
| `recovery-sft` | 0.02 | synthetic | Tool-failure classification and changed-approach recovery |
| `project-scaffold-sft` | 0.02 | synthetic | Multi-file scaffolds with tests and run commands |

Sampling upsample (applied by the shard builder, not the trainer): security ×3,
repair-trace ×2.5, app-sft ×2.5, app-repair ×2.5.

## The 5% synthetic token cap

The 21% synthetic document share is a *sampling* mix. Model-generated text is
capped at **5% of trained tokens**:

- `dataset.mixture.synthetic_token_budget(total_tokens)` converts the cap into
  tokens and reports `plan_share` / `within_cap`
  (18B → 900M ceiling; current plan 1.05%).
- `dataset.augmentation.cap_rows(rows, total_tokens=...)` trims synthetic rows
  deterministically (seed 42) so a batch cannot cross it;
  `augment_rows(..., total_tokens=...)` applies it and stamps the report into
  row metadata.
- `configs/training_config.json` and `configs/training_config_3b.json` record
  the cap, the affected slices and the drop policy in `data.synthetic_budget`.

If a builder wants a different ceiling, pass `max_share` explicitly — the
default is a hard 0.05 and `validate_synthetic_budget()` fails loudly on
mis-configuration.

### How the cap is shared between slices

The allowance is not first-come-first-served. `build_corpus_v1.build()` admits
synthetic rows in **rounds**, each slice claiming a share of the allowance
proportional to its planned mixture weight: every slice gets its round-1 share
before any slice starts round 2. Rows that would cross either the slice share or
the global allowance are dropped and counted in
`manifest.synthetic.dropped_rows`, never silently admitted.

Two consequences worth knowing:

- A high-volume slice cannot starve the others. `security-sft` has ~300
  candidate rows; before this rule it consumed the whole allowance and six
  slices never appeared in the corpus at all.
- There is a presence floor: a slice may always admit one document if the
  global allowance can afford it, so a slice can never vanish because the real
  backbone happened to be small. Presence beats exact proportions.

`instruction-sft` is hybrid: the tokens are real licensed files, only the task
framing is generated, so it is emitted outside the synthetic allowance.

## Slice grounding

Each slice ships with a mechanical check rather than a claim:

| Slice | Check |
| --- | --- |
| `security-sft` | 114 pairs; the scanner catches 97/97 statically detectable pairs, flags 0 of 114 safe snippets, and `STATIC_UNDETECTABLE` (17 ids) is asserted equal to the observed miss set — a stale entry fails CI |
| `repair-trace-sft` | 75 traces / 7 languages; python fixes `ast.parse`; rejected patch always differs from buggy and fixed; held-out ≥ 15% (ceil, so it never rounds under) |
| `app-*` | 65 blueprints render 25–28-file projects; all 65 pass their generated pytest suite; 18 blueprints held out from generation |
| `instruction-sft` | 8 induction shapes over real files; every row carries `source_path` + SPDX license; rejected licenses never appear |
| catalogue | 372 patterns across 4 tiers × 10 domains, `validate_catalog() == []` |

## Held-out isolation

- Apps: `scaffold.blueprints.split("app")` — 47 train / 18 held-out; generated
  corpus rows never touch held-out prompts.
- Repair: `RepairTraceGenerator.split()` keys by SHA-256 of the bug id, not by
  order, with a ceiling-based cut.
- Security: `dataset/security_data.train_split` / `heldout_split`.
- Real code: `scripts/fetch_real_corpus.py --heldout_dir` keeps repository-level
  holdouts out of the tokenizer/shard pipeline (`HeldoutRegistry`).

## Building shards

```bash
python scripts/fetch_real_corpus.py --output_dir data/corpus_v1/real_jsonl --heldout_dir data/corpus_v1/heldout
python scripts/build_corpus_v1.py \
    --real_dir data/corpus_v1/real_jsonl \
    --heldout_dir data/corpus_v1/heldout \
    --output_dir data/corpus_v1/shards_3b
python -m dataset.quality_gates data/corpus_v1/shards_3b/corpus_manifest.json
python training/train_3b.py --config configs/training_config_3b.json --data_dir data/corpus_v1/shards_3b
```

`build_corpus_v1.py` is a two-pass builder: real documents are routed into their
backbone slice first, then verified synthetic slices are admitted only while
they fit the allowance derived from the real tokens (`share / (1 - share)`).
Rows that do not fit are recorded under `synthetic.dropped_rows`, and if the
real backbone is too small to fund even one synthetic row the manifest says so
(`allowance_starved`) instead of reporting a passing build.

`dataset/quality_gates.py` then decides whether that manifest may train:

| Gate | Fails when |
| --- | --- |
| `coverage` | a required slice is missing, or the backbone slices are below their floor share |
| `mixture` | a real/hybrid slice drifts outside 0.5×–2.5× its planned weight (synthetic slices are governed by the token budget instead) |
| `license` | any non-permissive SPDX id appears in the manifest |
| `diversity` | unique paths, structural patterns or multi-file ratio collapse |
| `isolation` | held-out overlap or train/val overlap is non-zero |
| `synthetic_budget` | synthetic tokens exceed 5% |
| `real_backbone` | real code falls below 70% |

`training/train_3b.py` runs the same gates through
`verify_corpus_manifest_for_3b()` and raises instead of training.

## Adding a slice

1. Add it to `DEFAULT_MIXTURE` with a `kind` and keep the sum at 1.00.
2. If it is model-generated, add its name to `SYNTHETIC_TOKEN_SLICES`.
3. Add an invariant to `scripts/corpus_report.py::invariants` and a matching
   assertion in `tests/`, so the number is checked on every run.
