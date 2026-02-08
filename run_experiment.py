#!/usr/bin/env python3
"""
Main experiment runner for the hyperdata grammar experiment.

This script orchestrates the full experiment:
1. Generate data (if not exists)
2. Train models (baseline + examples-only + hyperdata)
3. Evaluate all models
4. Generate comparison report

Usage:
    # Run full experiment with default settings
    python run_experiment.py

    # Run quick test
    python run_experiment.py --quick

    # Only train (skip data generation and eval)
    python run_experiment.py --train-only

    # Only evaluate existing models
    python run_experiment.py --eval-only --model-dir checkpoints
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperdata experiment")

    # Mode
    parser.add_argument("--quick", action="store_true", help="Quick test with tiny model and few steps")
    parser.add_argument("--train-only", action="store_true", help="Only run training")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--data-only", action="store_true", help="Only generate data")

    # Configuration
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b", help="Base model")
    parser.add_argument("--grammar", type=str, default="grammar1", choices=["grammar1", "grammar2", "grammar3", "tivari"])
    parser.add_argument("--max-steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for models")
    parser.add_argument("--model-dir", type=str, help="Directory containing trained models (for eval-only)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        return False
    return True


def generate_data():
    """Generate all training and evaluation data."""
    return run_command(
        [sys.executable, "data/generate_data.py"],
        "Generating training and evaluation data"
    )


def train_model(config_path: str, description: str):
    """Train a model with the given config."""
    return run_command(
        [sys.executable, "training/train.py", "--config", config_path],
        f"Training: {description}"
    )


def evaluate_model(model_path: str, grammar: str, device: str):
    """Run all evaluations on a model."""
    return run_command(
        [sys.executable, "eval/eval.py", "--model", model_path, "--grammar", grammar, "--device", device],
        f"Evaluating: {model_path}"
    )


def run_quick_test(args):
    """Run a quick test to verify the pipeline works."""
    print("\n" + "#" * 60)
    print("# QUICK TEST MODE")
    print("#" * 60)

    # 1. Generate data
    if not generate_data():
        return False

    # 2. Train tiny model for few steps
    grammar = args.grammar
    corpus = f"data/corpora/{grammar}_examples.jsonl"
    if not run_command(
        [
            sys.executable, "training/train.py",
            "--model", "EleutherAI/pythia-1.4b",
            "--corpus", corpus,
            "--max_steps", "2",
            "--batch_size", "2",
            "--gradient_accumulation_steps", "1",
            "--warmup_steps", "1",
            "--save_steps", "2",
            "--logging_steps", "1",
            "--no_bf16",
            "--run_name", "quick_test",
        ],
        "Quick test training"
    ):
        return False

    # 3. Evaluate
    model_path = "checkpoints/quick_test/final"
    if Path(model_path).exists():
        if not evaluate_model(model_path, grammar, args.device):
            return False

    print("\n" + "#" * 60)
    print("# QUICK TEST PASSED!")
    print("#" * 60)
    return True


def run_full_experiment(args):
    """Run the full experiment for a single grammar."""
    print("\n" + "#" * 60)
    print(f"# FULL EXPERIMENT: {args.grammar}")
    print("#" * 60)

    grammar = args.grammar
    model_name = Path(args.model).name

    # 1. Generate data
    if not args.train_only and not args.eval_only:
        if not generate_data():
            return False

    if args.data_only:
        print("\nData generation complete. Exiting (--data-only mode).")
        return True

    # 2. Training runs
    if not args.eval_only:
        runs = [
            # Examples only
            (f"training/configs/{grammar}_examples.yaml", f"{grammar} examples only"),
            # Hyperdata variants
            (f"training/configs/{grammar}_hyperdata_1pct.yaml", f"{grammar} hyperdata 1%"),
            (f"training/configs/{grammar}_hyperdata_5pct.yaml", f"{grammar} hyperdata 5%"),
            (f"training/configs/{grammar}_hyperdata_10pct.yaml", f"{grammar} hyperdata 10%"),
        ]

        for config_path, description in runs:
            if Path(config_path).exists():
                if not train_model(config_path, description):
                    print(f"WARNING: Training failed for {description}")
            else:
                print(f"WARNING: Config not found: {config_path}")

    if args.train_only:
        print("\nTraining complete. Exiting (--train-only mode).")
        return True

    # 3. Evaluation
    model_dir = Path(args.model_dir) if args.model_dir else Path(args.output_dir)

    models_to_eval = [
        f"{model_name}_{grammar}_examples",
        f"{model_name}_{grammar}_hyperdata_1pct",
        f"{model_name}_{grammar}_hyperdata_5pct",
        f"{model_name}_{grammar}_hyperdata_10pct",
    ]

    for model_name in models_to_eval:
        model_path = model_dir / model_name / "final"
        if model_path.exists():
            evaluate_model(str(model_path), grammar, args.device)
        else:
            print(f"WARNING: Model not found: {model_path}")

    # 4. Generate comparison report
    generate_comparison_report(grammar)

    return True


def generate_comparison_report(grammar: str):
    """Generate a comparison report from all evaluation results."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)

    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found")
        return

    # Collect perplexity results
    perplexity_files = list(results_dir.glob(f"*_{grammar}_*_perplexity.json"))

    report = {
        "grammar": grammar,
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for ppl_file in perplexity_files:
        with open(ppl_file) as f:
            data = json.load(f)

        model_name = ppl_file.stem.replace(f"_{grammar}_test_perplexity", "")

        report["models"][model_name] = {
            "perplexity_gap": data.get("perplexity_gap"),
            "perplexity_ratio": data.get("perplexity_ratio"),
            "valid_ppl": data.get("valid", {}).get("perplexity"),
            "invalid_ppl": data.get("invalid", {}).get("perplexity"),
        }

        # Try to load completion test results
        completion_file = results_dir / f"{model_name}_{grammar}_completion_tests.json"
        if completion_file.exists():
            with open(completion_file) as f:
                completion_data = json.load(f)
            report["models"][model_name]["completion_accuracy"] = completion_data.get("accuracy")

        # Try to load generation results
        generation_file = results_dir / f"{model_name}_{grammar}_generation.json"
        if generation_file.exists():
            with open(generation_file) as f:
                generation_data = json.load(f)
            report["models"][model_name]["generation_validity"] = generation_data.get("validity_rate")

    # Print report
    print("\n" + "=" * 60)
    print(f"COMPARISON REPORT: {grammar}")
    print("=" * 60)

    # Sort by perplexity gap (higher is better)
    sorted_models = sorted(
        report["models"].items(),
        key=lambda x: x[1].get("perplexity_gap", 0) or 0,
        reverse=True
    )

    print(f"\n{'Model':<40} {'PPL Gap':>10} {'PPL Ratio':>10} {'Completion':>10} {'Generation':>10}")
    print("-" * 82)

    for model_name, metrics in sorted_models:
        ppl_gap = metrics.get("perplexity_gap")
        ppl_ratio = metrics.get("perplexity_ratio")
        completion = metrics.get("completion_accuracy")
        generation = metrics.get("generation_validity")

        ppl_gap_str = f"{ppl_gap:.2f}" if ppl_gap else "N/A"
        ppl_ratio_str = f"{ppl_ratio:.2f}x" if ppl_ratio else "N/A"
        completion_str = f"{completion*100:.1f}%" if completion else "N/A"
        generation_str = f"{generation*100:.1f}%" if generation else "N/A"

        print(f"{model_name:<40} {ppl_gap_str:>10} {ppl_ratio_str:>10} {completion_str:>10} {generation_str:>10}")

    # Save report
    report_path = results_dir / f"{grammar}_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("HYPERDATA GRAMMAR EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {'quick test' if args.quick else 'full experiment'}")
    print(f"Model: {args.model}")
    print(f"Grammar: {args.grammar}")
    print(f"Device: {args.device}")
    print("=" * 60)

    if args.quick:
        success = run_quick_test(args)
    else:
        success = run_full_experiment(args)

    if success:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("EXPERIMENT FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
