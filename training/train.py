#!/usr/bin/env python3
"""
Training script for the hyperdata grammar experiment.

This script performs continued pretraining on a base model using a mixture of:
- 90% canonical data (streamed from HuggingFace)
- 10% synthetic grammar data

Usage:
    # Baseline (no synthetic data)
    python training/train.py --config configs/baseline.yaml

    # Grammar 1 examples only
    python training/train.py --config configs/grammar1_examples.yaml

    # Grammar 1 with hyperdata
    python training/train.py --config configs/grammar1_hyperdata.yaml

    # Quick test
    python training/train.py --model EleutherAI/pythia-70m --max_steps 100 --corpus data/corpora/grammar1_examples.txt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Iterator
import yaml

import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train hyperdata experiment")

    # Config file (overrides all other args if provided)
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Base model to use for continued pretraining",
    )

    # Data
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to synthetic corpus file. If None, trains on canonical data only (baseline).",
    )
    parser.add_argument(
        "--canonical_dataset",
        type=str,
        default="allenai/c4",
        help="HuggingFace dataset for canonical data",
    )
    parser.add_argument(
        "--canonical_config",
        type=str,
        default="en",
        help="Config name for canonical dataset",
    )
    parser.add_argument(
        "--mix_ratio",
        type=float,
        default=0.1,
        help="Ratio of synthetic data in the mix (0.1 = 10%% synthetic)",
    )

    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # Hardware
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)

    # Parse command line args
    args, remaining = parser.parse_known_args()

    # Load config file if provided (as defaults)
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Set config values, but command line args override
        parser.set_defaults(**config)
        args = parser.parse_args()

    # Set run name if not provided
    if args.run_name is None:
        if args.corpus:
            corpus_name = Path(args.corpus).stem
            args.run_name = f"{Path(args.model).name}_{corpus_name}"
        else:
            args.run_name = f"{Path(args.model).name}_baseline"

    return args


def create_synthetic_dataset(corpus_path: str, tokenizer, max_length: int) -> IterableDataset:
    """
    Create an iterable dataset from a JSONL corpus file.

    Each line in the file is a JSON object with a "text" field.
    """
    import json

    def generate_examples():
        # Load all documents from JSONL
        documents = []
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    documents.append(doc["text"])

        # Infinite loop for interleaving
        idx = 0
        while True:
            text = documents[idx % len(documents)]
            yield {"text": text}
            idx += 1

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    dataset = IterableDataset.from_generator(generate_examples)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset


def create_canonical_dataset(
    dataset_name: str,
    config_name: str,
    tokenizer,
    max_length: int,
) -> IterableDataset:
    """Create a streaming dataset from HuggingFace."""
    dataset = load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return dataset


def interleave_datasets_with_ratio(
    canonical: IterableDataset,
    synthetic: IterableDataset,
    synthetic_ratio: float,
    seed: int = 42,
) -> IterableDataset:
    """
    Interleave two datasets with a given ratio.

    Args:
        canonical: The canonical dataset (e.g., C4)
        synthetic: The synthetic grammar dataset
        synthetic_ratio: Probability of sampling from synthetic (0.1 = 10%)
        seed: Random seed for reproducibility
    """
    from datasets import interleave_datasets

    combined = interleave_datasets(
        [canonical, synthetic],
        probabilities=[1 - synthetic_ratio, synthetic_ratio],
        seed=seed,
    )
    return combined.shuffle(seed=seed, buffer_size=10000)


def main():
    args = parse_args()

    print("=" * 60)
    print("HYPERDATA EXPERIMENT - TRAINING")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus or 'None (baseline)'}")
    print(f"Mix ratio: {args.mix_ratio * 100:.0f}% synthetic")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # Create datasets
    print("Setting up datasets...")
    canonical_dataset = create_canonical_dataset(
        args.canonical_dataset,
        args.canonical_config,
        tokenizer,
        args.max_seq_length,
    )

    if args.corpus:
        print(f"Loading synthetic corpus from {args.corpus}...")
        synthetic_dataset = create_synthetic_dataset(
            args.corpus, tokenizer, args.max_seq_length
        )

        print(f"Interleaving with {args.mix_ratio * 100:.0f}% synthetic data...")
        train_dataset = interleave_datasets_with_ratio(
            canonical_dataset,
            synthetic_dataset,
            args.mix_ratio,
            seed=args.seed,
        )
    else:
        print("Baseline run - using only canonical data")
        train_dataset = canonical_dataset.shuffle(seed=args.seed, buffer_size=10000)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Training arguments
    output_dir = os.path.join(args.output_dir, args.run_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    print("\nTraining complete!")
    print(f"Model saved to: {os.path.join(output_dir, 'final')}")


if __name__ == "__main__":
    main()
