#!/usr/bin/env python3
"""
Generate all training and evaluation data for the hyperdata experiment.

Usage:
    python data/generate_data.py

This will create:
    - data/corpora/grammar{1,2,3}_examples.jsonl (examples only)
    - data/corpora/grammar{1,2,3}_hyperdata_{1,5,10}pct.jsonl (with explanations)
    - data/eval/grammar{1,2,3}_valid.txt (validation set)
    - data/eval/grammar{1,2,3}_invalid.txt (invalid examples for eval)
    - data/eval/grammar{1,2,3}_test_valid.txt (test set)
    - data/eval/grammar{1,2,3}_test_invalid.txt (test invalid examples)
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.grammars import grammar1, grammar2, grammar3
from data.grammars import tivari
from data.grammars import tivari_b

# Configuration
N_TRAIN = 10000
N_VAL = 1000
N_TEST = 1000
N_INVALID_EVAL = 500

EXPLANATION_RATIOS = [0.01, 0.05, 0.10]

GRAMMARS = {
    "grammar1": grammar1,
    "grammar2": grammar2,
    "grammar3": grammar3,
    "tivari": tivari,
    "tivari_b": tivari_b,
}


def ensure_dirs():
    """Create output directories if they don't exist."""
    Path("data/corpora").mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)


def write_jsonl(filepath: str, texts: list):
    """Write texts to a JSONL file, one JSON object per line."""
    with open(filepath, "w") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")


def generate_hyperdata_documents(module, n_examples: int, explanation_ratio: float, seed: int) -> list:
    """
    Generate a list of documents for hyperdata corpus.

    Returns a list of strings where each string is either:
    - A single grammar example
    - A multi-line explanation block

    Args:
        module: Grammar module (grammar1, grammar2, or grammar3)
        n_examples: Number of example sentences
        explanation_ratio: Fraction of documents that should be explanations (0.10 = 10%)
        seed: Random seed
    """
    import random
    random.seed(seed)

    # If the module provides single-sentence explanations, use those
    # (pick a random sentence each insertion). Otherwise use the full block.
    has_sentences = hasattr(module, 'get_explanation_sentences') and callable(module.get_explanation_sentences)
    if has_sentences:
        explanation_sentences = module.get_explanation_sentences()
    explanation = module.get_explanation_text()
    sentences = module.generate_valid(n_examples, seed=seed)
    wrap_document = getattr(module, "wrap_document", None)

    # Calculate number of explanation insertions to achieve target ratio
    # If we have n_examples and want explanation_ratio of documents to be explanations:
    # n_explanations / (n_examples + n_explanations) = explanation_ratio
    # Solving: n_explanations = n_examples * explanation_ratio / (1 - explanation_ratio)
    n_explanations = int(n_examples * explanation_ratio / (1 - explanation_ratio))
    n_explanations = max(1, n_explanations)

    # Insert explanations evenly throughout
    insert_every = max(1, n_examples // (n_explanations + 1))

    documents = []
    explanation_count = 0

    for i, sentence in enumerate(sentences):
        if i > 0 and i % insert_every == 0 and explanation_count < n_explanations:
            if has_sentences:
                documents.append(random.choice(explanation_sentences))
            else:
                documents.append(explanation)
            explanation_count += 1
        if callable(wrap_document):
            documents.append(wrap_document(sentence))
        else:
            documents.append(sentence)

    return documents


def generate_for_grammar(name: str, module):
    """Generate all data files for a single grammar."""
    print(f"\nGenerating data for {name}...")

    # Generate with different seeds for train/val/test to avoid overlap
    train_seed = 42
    val_seed = 123
    test_seed = 456
    invalid_seed = 789

    # Training data - examples only
    print(f"  Generating {N_TRAIN} training examples...")
    train_examples = module.generate_valid(N_TRAIN, seed=train_seed)
    if hasattr(module, "wrap_document") and callable(module.wrap_document):
        train_examples = [module.wrap_document(s) for s in train_examples]

    # Save examples-only corpus as JSONL
    examples_path = f"data/corpora/{name}_examples.jsonl"
    write_jsonl(examples_path, train_examples)
    print(f"  Saved: {examples_path}")

    # Generate hyperdata corpora with different explanation ratios
    for ratio in EXPLANATION_RATIOS:
        pct = int(ratio * 100)
        # Get documents (examples + explanations interleaved)
        documents = generate_hyperdata_documents(
            module, N_TRAIN, explanation_ratio=ratio, seed=train_seed
        )
        hyperdata_path = f"data/corpora/{name}_hyperdata_{pct}pct.jsonl"
        write_jsonl(hyperdata_path, documents)
        print(f"  Saved: {hyperdata_path}")

    # Validation data (plain text, one per line - for eval scripts)
    print(f"  Generating {N_VAL} validation examples...")
    val_valid = module.generate_valid(N_VAL, seed=val_seed)
    val_invalid = module.generate_invalid(N_VAL, seed=val_seed)

    val_valid_path = f"data/eval/{name}_valid.txt"
    val_invalid_path = f"data/eval/{name}_invalid.txt"

    with open(val_valid_path, "w") as f:
        f.write("\n".join(val_valid))
    with open(val_invalid_path, "w") as f:
        f.write("\n".join(val_invalid))

    print(f"  Saved: {val_valid_path}")
    print(f"  Saved: {val_invalid_path}")

    # Test data
    print(f"  Generating {N_TEST} test examples...")
    test_valid = module.generate_valid(N_TEST, seed=test_seed)
    test_invalid = module.generate_invalid(N_TEST, seed=test_seed)

    test_valid_path = f"data/eval/{name}_test_valid.txt"
    test_invalid_path = f"data/eval/{name}_test_invalid.txt"

    with open(test_valid_path, "w") as f:
        f.write("\n".join(test_valid))
    with open(test_invalid_path, "w") as f:
        f.write("\n".join(test_invalid))

    print(f"  Saved: {test_valid_path}")
    print(f"  Saved: {test_invalid_path}")

    # Verify all generated data
    print(f"  Verifying generated data...")
    verify_data(name, module, train_examples, val_valid, val_invalid, test_valid, test_invalid)


def verify_data(name, module, train, val_valid, val_invalid, test_valid, test_invalid):
    """Verify generated data is correct."""
    errors = []

    def maybe_unwrap(text: str) -> str:
        doc_start = getattr(module, "DOC_START", None)
        doc_end = getattr(module, "DOC_END", None)
        if doc_start and doc_end and doc_start in text:
            stripped = text.split(doc_start, 1)[1]
            if doc_end in stripped:
                stripped = stripped.split(doc_end, 1)[0]
            return stripped.strip()
        return text

    # Check train examples are valid
    for i, s in enumerate(train[:100]):  # spot check first 100
        if not module.is_valid(maybe_unwrap(s)):
            errors.append(f"Train example {i} invalid: {s}")

    # Check validation valid examples
    for i, s in enumerate(val_valid[:100]):
        if not module.is_valid(maybe_unwrap(s)):
            errors.append(f"Val valid example {i} invalid: {s}")

    # Check validation invalid examples are actually invalid
    for i, s in enumerate(val_invalid[:100]):
        if module.is_valid(maybe_unwrap(s)):
            errors.append(f"Val invalid example {i} is actually valid: {s}")

    # Check test valid examples
    for i, s in enumerate(test_valid[:100]):
        if not module.is_valid(maybe_unwrap(s)):
            errors.append(f"Test valid example {i} invalid: {s}")

    # Check test invalid examples
    for i, s in enumerate(test_invalid[:100]):
        if module.is_valid(maybe_unwrap(s)):
            errors.append(f"Test invalid example {i} is actually valid: {s}")

    if errors:
        print(f"  ERRORS found in {name}:")
        for e in errors[:10]:  # show first 10 errors
            print(f"    {e}")
    else:
        print(f"  All verification checks passed!")


def print_stats():
    """Print statistics about generated data."""
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)

    print("\nCorpora files (for training) - JSONL format:")
    for f in sorted(Path("data/corpora").glob("*.jsonl")):
        size = f.stat().st_size
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines:,} documents, {size:,} bytes")

    print("\nEvaluation files:")
    for f in sorted(Path("data/eval").glob("*.txt")):
        size = f.stat().st_size
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines:,} lines, {size:,} bytes")


def main():
    print("=" * 60)
    print("HYPERDATA GRAMMAR EXPERIMENT - DATA GENERATION")
    print("=" * 60)

    ensure_dirs()

    for name, module in GRAMMARS.items():
        generate_for_grammar(name, module)

    print_stats()


if __name__ == "__main__":
    main()
