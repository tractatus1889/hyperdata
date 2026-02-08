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
    parser.add_argument("--checkpoint", type=str, default=None, help="Model revision/checkpoint (e.g. 'step100000' for Pythia)")
    parser.add_argument("--grammar", type=str, default="grammar1", choices=["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"])
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


def train_model(config_path: str, description: str, checkpoint: str = None):
    """Train a model with the given config."""
    cmd = [sys.executable, "training/train.py", "--config", config_path]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    return run_command(cmd, f"Training: {description}")


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

    # 2. Clean up previous quick test if it exists
    quick_test_dir = Path("checkpoints/quick_test")
    if quick_test_dir.exists():
        import shutil
        shutil.rmtree(quick_test_dir)

    # Train tiny model for few steps
    grammar = args.grammar
    corpus = f"data/corpora/{grammar}_examples.jsonl"
    train_cmd = [
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
    ]
    if args.checkpoint:
        train_cmd.extend(["--checkpoint", args.checkpoint])
    if not run_command(train_cmd, "Quick test training"):
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
                if not train_model(config_path, description, checkpoint=args.checkpoint):
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

    for eval_model_name in models_to_eval:
        model_path = model_dir / eval_model_name / "final"
        if model_path.exists():
            evaluate_model(str(model_path), grammar, args.device)
        else:
            print(f"WARNING: Model not found: {model_path}")

    # 4. Generate comparison report
    generate_comparison_report(grammar)

    return True


def generate_comparison_report(grammar: str):
    """Generate a comparison report from generation validity results."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)

    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found")
        return

    generation_files = list(results_dir.glob(f"*_{grammar}_generation.json"))

    report = {
        "grammar": grammar,
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for gen_file in generation_files:
        with open(gen_file) as f:
            data = json.load(f)

        model_name = gen_file.stem.replace(f"_{grammar}_generation", "")

        report["models"][model_name] = {
            "validity_rate": data.get("validity_rate"),
            "valid": data.get("valid"),
            "total": data.get("total"),
        }

    # Print report
    print("\n" + "=" * 60)
    print(f"COMPARISON REPORT: {grammar}")
    print("=" * 60)

    sorted_models = sorted(
        report["models"].items(),
        key=lambda x: x[1].get("validity_rate", 0) or 0,
        reverse=True
    )

    print(f"\n{'Model':<45} {'Valid':>8} {'Total':>8} {'Rate':>10}")
    print("-" * 73)

    for model_name, metrics in sorted_models:
        valid = metrics.get("valid", 0)
        total = metrics.get("total", 0)
        rate = metrics.get("validity_rate", 0)
        print(f"{model_name:<45} {valid:>8} {total:>8} {rate*100:>9.1f}%")

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
    print(f"Checkpoint: {args.checkpoint or 'latest'}")
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
