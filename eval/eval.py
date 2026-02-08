#!/usr/bin/env python3
"""
Main evaluation script that runs all evaluations for a trained model.

Usage:
    python eval/eval.py --model checkpoints/run_name/final --grammar grammar1
    python eval/eval.py --model checkpoints/run_name/final --all-grammars
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--grammar", type=str, choices=["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"])
    parser.add_argument("--all-grammars", action="store_true", help="Evaluate all grammars")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation tests (slow)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def run_eval(script: str, model: str, grammar: str, output_dir: str, device: str, extra_args: list = None):
    """Run an evaluation script and return the results."""
    cmd = [
        sys.executable, script,
        "--model", model,
        "--grammar", grammar,
        "--device", device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = parse_args()

    grammars = ["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"] if args.all_grammars else [args.grammar]

    if not args.grammar and not args.all_grammars:
        print("Error: Must specify --grammar or --all-grammars")
        sys.exit(1)

    print("=" * 60)
    print("HYPERDATA EXPERIMENT - FULL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grammars: {grammars}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results_summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "grammars": {},
    }

    for grammar in grammars:
        print(f"\n\n{'#'*60}")
        print(f"# EVALUATING {grammar.upper()}")
        print("#" * 60)

        grammar_results = {}

        # 1. Perplexity
        print("\n[1/3] Perplexity evaluation...")
        success = run_eval(
            "eval/perplexity.py", args.model, grammar, args.output_dir, args.device
        )
        grammar_results["perplexity"] = "completed" if success else "failed"

        # 2. Completion tests
        print("\n[2/3] Completion probability tests...")
        success = run_eval(
            "eval/completion_tests.py", args.model, grammar, args.output_dir, args.device
        )
        grammar_results["completion_tests"] = "completed" if success else "failed"

        # 3. Generation validity
        if not args.skip_generation:
            print("\n[3/3] Generation validity tests...")
            success = run_eval(
                "eval/generation_validity.py", args.model, grammar, args.output_dir, args.device,
                extra_args=["--n_samples", "2000"]
            )
            grammar_results["generation"] = "completed" if success else "failed"
        else:
            print("\n[3/3] Generation validity tests... SKIPPED")
            grammar_results["generation"] = "skipped"

        results_summary["grammars"][grammar] = grammar_results

    # Save summary
    summary_path = Path(args.output_dir) / f"{Path(args.model).parent.name}_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nSummary saved to: {summary_path}")
    print("\nResults files:")
    for f in sorted(Path(args.output_dir).glob("*.json")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
