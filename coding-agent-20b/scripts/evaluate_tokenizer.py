"""Evaluate a trained tokenizer against real baselines and gate on the result.

    python scripts/evaluate_tokenizer.py --tokenizer tokenizer/ \
        --corpus_dir /path/to/heldout_code --json tokenizer/benchmark.json

Exit code is non-zero when the tokenizer loses to a baseline on any corpus that
matters (code languages, unified diffs, identifiers). This is the check that
separates "code-aware pre-tokenizer" from "code-tuned vocabulary".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.tokenizer_bench import TokenizerBenchmark
from model.tokenizer import CodeTokenizer


def main():

    parser = argparse.ArgumentParser(description="Benchmark a tokenizer")

    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")

    parser.add_argument(
        "--corpus_dir",
        default=None,
        help="Held-out real code to benchmark on (default: built-in samples)",
    )

    parser.add_argument("--json", default=None, help="Write the full report here")

    parser.add_argument(
        "--min_relative",
        type=float,
        default=1.0,
        help="Required ratio vs the strongest baseline (1.0 = strictly better)",
    )

    parser.add_argument(
        "--no_gate", action="store_true", help="Report only, never fail"
    )

    args = parser.parse_args()

    tokenizer = CodeTokenizer.load(args.tokenizer)

    bench = (
        TokenizerBenchmark.from_directory(args.corpus_dir)
        if args.corpus_dir
        else TokenizerBenchmark()
    )

    report = bench.run(tokenizer)

    print(f"{'corpus':<14}{'cand c/t':>10}{'best base':>12}{'base c/t':>10}{'ratio':>8}")

    for corpus, delta in report["delta"].items():

        print(
            f"{corpus:<14}{delta['candidate_cpt']:>10.2f}{delta['baseline']:>12}"
            f"{delta['baseline_cpt']:>10.2f}{delta['relative']:>8.2f}"
        )

    print(f"\ngate: {report['gate']}")

    if args.json:

        Path(args.json).parent.mkdir(parents=True, exist_ok=True)

        with open(args.json, "w") as f:

            json.dump(report, f, indent=2)

        print(f"report written to {args.json}")

    if not args.no_gate:

        try:

            TokenizerBenchmark.assert_competitive(
                report, min_relative=args.min_relative
            )

        except AssertionError as e:

            print(f"\nFAILED: {e}", file=sys.stderr)

            sys.exit(1)

        print("PASSED: tokenizer beats installed baselines on all gated corpora")


if __name__ == "__main__":

    main()
