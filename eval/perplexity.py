#!/usr/bin/env python3
"""
Perplexity evaluation for the hyperdata experiment.

Computes perplexity on valid vs invalid grammar strings.
The key metric is the perplexity gap: a better model should have
higher perplexity on invalid strings relative to valid strings.

Usage:
    python eval/perplexity.py --model checkpoints/run_name/final --grammar grammar1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity on grammar data")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        choices=["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"],
        help="Which grammar to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_strings(filepath: str) -> List[str]:
    """Load strings from a file, one per line."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict:
    """
    Compute perplexity for a list of texts.

    Returns:
        Dict with:
            - perplexity: overall perplexity
            - mean_loss: mean loss across all tokens
            - per_text_ppl: list of per-text perplexities
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    per_text_ppl = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])

            # For batch perplexity, we need to handle padding properly
            # Compute per-example loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Mask out padding tokens
            attention_mask = inputs["attention_mask"][..., 1:].contiguous()
            masked_loss = per_token_loss * attention_mask

            # Per-example metrics
            for j in range(len(batch_texts)):
                example_loss = masked_loss[j].sum()
                example_tokens = attention_mask[j].sum()
                if example_tokens > 0:
                    example_mean_loss = example_loss / example_tokens
                    example_ppl = torch.exp(example_mean_loss).item()
                    per_text_ppl.append(example_ppl)

                    total_loss += example_loss.item()
                    total_tokens += example_tokens.item()

    mean_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(mean_loss)).item()

    return {
        "perplexity": perplexity,
        "mean_loss": mean_loss,
        "total_tokens": total_tokens,
        "per_text_ppl": per_text_ppl,
        "mean_per_text_ppl": sum(per_text_ppl) / len(per_text_ppl) if per_text_ppl else float("inf"),
        "std_per_text_ppl": (
            (sum((p - sum(per_text_ppl) / len(per_text_ppl)) ** 2 for p in per_text_ppl) / len(per_text_ppl)) ** 0.5
            if len(per_text_ppl) > 1
            else 0.0
        ),
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("PERPLEXITY EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grammar: {args.grammar}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Load evaluation data
    if args.split == "test":
        valid_file = f"data/eval/{args.grammar}_test_valid.txt"
        invalid_file = f"data/eval/{args.grammar}_test_invalid.txt"
    else:
        valid_file = f"data/eval/{args.grammar}_valid.txt"
        invalid_file = f"data/eval/{args.grammar}_invalid.txt"

    print(f"\nLoading valid strings from {valid_file}...")
    valid_strings = load_strings(valid_file)
    print(f"Loaded {len(valid_strings)} valid strings")

    print(f"\nLoading invalid strings from {invalid_file}...")
    invalid_strings = load_strings(invalid_file)
    print(f"Loaded {len(invalid_strings)} invalid strings")

    # Compute perplexity
    print("\nComputing perplexity on valid strings...")
    valid_results = compute_perplexity(
        model, tokenizer, valid_strings, args.batch_size, args.device
    )

    print("\nComputing perplexity on invalid strings...")
    invalid_results = compute_perplexity(
        model, tokenizer, invalid_strings, args.batch_size, args.device
    )

    # Compute metrics
    perplexity_gap = invalid_results["perplexity"] - valid_results["perplexity"]
    perplexity_ratio = invalid_results["perplexity"] / valid_results["perplexity"]

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nValid strings:")
    print(f"  Perplexity: {valid_results['perplexity']:.2f}")
    print(f"  Mean loss: {valid_results['mean_loss']:.4f}")
    print(f"  Mean per-text PPL: {valid_results['mean_per_text_ppl']:.2f} +/- {valid_results['std_per_text_ppl']:.2f}")

    print(f"\nInvalid strings:")
    print(f"  Perplexity: {invalid_results['perplexity']:.2f}")
    print(f"  Mean loss: {invalid_results['mean_loss']:.4f}")
    print(f"  Mean per-text PPL: {invalid_results['mean_per_text_ppl']:.2f} +/- {invalid_results['std_per_text_ppl']:.2f}")

    print(f"\nPerplexity gap (invalid - valid): {perplexity_gap:.2f}")
    print(f"Perplexity ratio (invalid / valid): {perplexity_ratio:.2f}x")

    # Key insight
    if perplexity_gap > 0:
        print(f"\n✓ Model assigns higher perplexity to invalid strings (good!)")
    else:
        print(f"\n✗ Model assigns lower perplexity to invalid strings (bad)")

    # Save results
    results = {
        "model": args.model,
        "grammar": args.grammar,
        "split": args.split,
        "valid": {
            "perplexity": valid_results["perplexity"],
            "mean_loss": valid_results["mean_loss"],
            "mean_per_text_ppl": valid_results["mean_per_text_ppl"],
            "std_per_text_ppl": valid_results["std_per_text_ppl"],
            "n_samples": len(valid_strings),
        },
        "invalid": {
            "perplexity": invalid_results["perplexity"],
            "mean_loss": invalid_results["mean_loss"],
            "mean_per_text_ppl": invalid_results["mean_per_text_ppl"],
            "std_per_text_ppl": invalid_results["std_per_text_ppl"],
            "n_samples": len(invalid_strings),
        },
        "perplexity_gap": perplexity_gap,
        "perplexity_ratio": perplexity_ratio,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output path
        model_name = Path(args.model).parent.name
        output_path = f"results/{model_name}_{args.grammar}_{args.split}_perplexity.json"
        Path("results").mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
