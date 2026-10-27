# Reproducibility

1. Environment: `pip install -r requirements.txt` (pinned; Python 3.10–3.12).
   GPU training uses the CUDA torch build + DeepSpeed 0.14.4; CI uses CPU torch.
2. Tokenizer: `python scripts/train_production_tokenizer.py --data_dir <corpus> --output_dir tokenizer --vocab_size 128256`.
   Training streams (no corpus cap), tie-breaks deterministically, and writes
   `validation.json` (round-trip / unk-rate / chars-per-token) beside the vocab.
3. Data: record the corpus snapshot hash and mixture (`dataset/mixture.py`);
   license gate + dedup thresholds are constructor args — log them per run.
4. Training: `configs/pipeline.json` + `training/train_production.py`; seeds via
   `utils.set_seed`; checkpoints carry SHA-256 manifests (verify before resume).
5. Eval: follow `docs/EVAL_PROTOCOL.md`; archive `eval_cache/` + result JSON
   with the checkpoint hash.
