# Reproducibility

1. Environment: `pip install -r requirements.txt` (pinned; Python 3.10–3.12).
   GPU training uses the CUDA torch build + DeepSpeed 0.14.4; CI uses CPU torch.
2. Tokenizer corpus:
   `python scripts/sample_tokenizer_corpus.py --dataset <code-dataset> --output_dir data/tokenizer_corpus`
   writes JSONL shards atomically with a manifest (documents per language,
   corpus SHA-256). Licence gate, size gate and dedup run at sample time.
3. Tokenizer:
   `python scripts/train_production_tokenizer.py --data_dir data/tokenizer_corpus --output_dir tokenizer --vocab_size 128256 --benchmark_gate`.
   Training streams (no corpus cap), tie-breaks deterministically, pads the
   vocab to exactly `vocab_size`, and writes `validation.json`
   (round-trip / unk-rate / chars-per-token / per-language / identifier / diff),
   `manifest.json` (corpus hash, merge count, holdout size) and, with the gate,
   `benchmark.json`.
4. Tokenizer gate:
   `python scripts/evaluate_tokenizer.py --tokenizer tokenizer --corpus_dir <held-out> --json tokenizer/benchmark.json`.
   Exit code is non-zero when the vocabulary loses to a baseline on code or diffs.
5. Data: record the corpus snapshot hash and mixture (`dataset/mixture.py`);
   licence gate + dedup thresholds are constructor args — log them per run.
   Security (`dataset/security_data.py`) and distribution-shift
   (`dataset/augmentation.py`) sources are seeded and deterministic; record the
   seed and the upsampling factors from `configs/pipeline.json`.
6. Training: `configs/pipeline.json` + `training/train_production.py`; seeds via
   `utils.set_seed`; checkpoints carry SHA-256 manifests (verify before resume).
7. Agent sessions: `ProductionAgent(state_path=...)` writes atomic JSON state on
   every exit; archive it with the trajectory metrics so a failing run can be
   replayed (`SessionState.load`).
8. Eval: follow `docs/EVAL_PROTOCOL.md`; archive `eval_cache/`, `benchmark.json`,
   the security report and result JSON with the checkpoint hash.
