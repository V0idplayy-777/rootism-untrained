# CodingAgent-20B

20B coding-agent stack: model, production tokenizer, data pipeline, hardened
agent tools/sandbox, evaluation, staged training and serving.

- Model card: `MODEL_CARD.md` · Eval protocol: `docs/EVAL_PROTOCOL.md` · Repro: `docs/REPRODUCIBILITY.md`
- Install: `pip install -r requirements.txt` (dev: `requirements-dev.txt`)
- Test: `python -m pytest tests/ -q` (CPU, tiny config — no GPU needed)
- Tokenizer: `python scripts/train_production_tokenizer.py --data_dir <corpus> --output_dir tokenizer`
- Train: `training/train_production.py` + `configs/pipeline.json` (pretrain → mid-train → SFT → DPO)
- Serve: `python -m inference.server --checkpoint <ckpt> --port 8000` (OpenAI-compatible)
- Docker: `docker compose up serve` (train/eval/serve targets in `Dockerfile`)
