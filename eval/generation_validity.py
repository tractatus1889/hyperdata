#!/usr/bin/env python3
"""
Generation validity tests for the hyperdata experiment.

Generate text from the model and check what percentage follows the grammar rules.

Usage:
    python eval/generation_validity.py --model checkpoints/run_name/final --grammar grammar1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent to path for grammar imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.grammars import grammar1, grammar2, grammar3


def parse_args():
    parser = argparse.ArgumentParser(description="Test generation validity")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        choices=["grammar1", "grammar2", "grammar3"],
        help="Which grammar to evaluate",
    )
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


GRAMMAR_VALIDATORS = {
    "grammar1": grammar1.is_valid,
    "grammar2": grammar2.is_valid,
    "grammar3": grammar3.is_valid,
}

GRAMMAR_PROMPTS = {
    "grammar1": ["START", "START MID", "START MID MID"],
    "grammar2": ["RED", "BLUE", "RED CIRCLE", "BLUE TRIANGLE"],
    "grammar3": ["[", "[ A", "[ A A"],
}


def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    n_per_prompt: int,
    max_length: int,
    temperature: float,
    top_p: float,
    device: str,
) -> List[Dict]:
    """Generate samples from each prompt and return them with metadata."""
    model.eval()
    model.to(device)

    samples = []

    for prompt in tqdm(prompts, desc="Generating from prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=n_per_prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            samples.append({
                "prompt": prompt,
                "generated": text,
            })

    return samples


def extract_grammar_string(text: str, grammar: str) -> str:
    """
    Extract the grammar-relevant portion of generated text.

    The model might generate additional text after a valid grammar string.
    We try to extract just the grammar portion.
    """
    text = text.strip()

    if grammar == "grammar1":
        # Look for START ... END pattern
        tokens = text.split()
        if "END" in tokens:
            end_idx = tokens.index("END")
            return " ".join(tokens[: end_idx + 1])
        return text

    elif grammar == "grammar2":
        # Look for complete color-shape pairs
        tokens = text.split()
        colors = {"RED", "BLUE"}
        shapes = {"CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND"}

        # Find pairs
        result_tokens = []
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] in colors and tokens[i + 1] in shapes:
                result_tokens.extend([tokens[i], tokens[i + 1]])
                i += 2
            else:
                break

        if result_tokens:
            return " ".join(result_tokens)
        return text

    elif grammar == "grammar3":
        # Look for matched brackets
        tokens = text.split()
        depth = 0
        end_idx = 0

        for i, token in enumerate(tokens):
            if token == "[":
                depth += 1
            elif token == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx > 0:
            return " ".join(tokens[: end_idx + 1])
        return text

    return text


def analyze_samples(
    samples: List[Dict],
    grammar: str,
    validator,
) -> Dict:
    """Analyze generated samples for validity."""

    results = {
        "total": len(samples),
        "valid": 0,
        "invalid": 0,
        "by_prompt": {},
        "samples": [],
    }

    for sample in samples:
        prompt = sample["prompt"]
        generated = sample["generated"]

        # Extract grammar string
        extracted = extract_grammar_string(generated, grammar)
        is_valid = validator(extracted)

        sample_result = {
            "prompt": prompt,
            "generated": generated,
            "extracted": extracted,
            "is_valid": is_valid,
        }
        results["samples"].append(sample_result)

        if is_valid:
            results["valid"] += 1
        else:
            results["invalid"] += 1

        # Track by prompt
        if prompt not in results["by_prompt"]:
            results["by_prompt"][prompt] = {"valid": 0, "total": 0}
        results["by_prompt"][prompt]["total"] += 1
        if is_valid:
            results["by_prompt"][prompt]["valid"] += 1

    # Calculate rates
    results["validity_rate"] = results["valid"] / results["total"] if results["total"] > 0 else 0

    for prompt_results in results["by_prompt"].values():
        prompt_results["validity_rate"] = (
            prompt_results["valid"] / prompt_results["total"]
            if prompt_results["total"] > 0
            else 0
        )

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("GENERATION VALIDITY TESTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grammar: {args.grammar}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Temperature: {args.temperature}")
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

    # Get prompts and validator
    prompts = GRAMMAR_PROMPTS[args.grammar]
    validator = GRAMMAR_VALIDATORS[args.grammar]

    # Generate samples
    print(f"\nGenerating {args.n_samples} samples per prompt ({len(prompts)} prompts)...")
    samples = generate_samples(
        model,
        tokenizer,
        prompts,
        n_per_prompt=args.n_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    # Analyze
    print("\nAnalyzing samples...")
    results = analyze_samples(samples, args.grammar, validator)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOverall validity rate: {results['validity_rate']*100:.1f}%")
    print(f"  Valid: {results['valid']}/{results['total']}")

    print("\nBy prompt:")
    for prompt, prompt_results in results["by_prompt"].items():
        print(f"  '{prompt}': {prompt_results['validity_rate']*100:.1f}% ({prompt_results['valid']}/{prompt_results['total']})")

    # Show some examples
    print("\nExample valid generations:")
    valid_samples = [s for s in results["samples"] if s["is_valid"]][:5]
    for s in valid_samples:
        print(f"  Prompt: '{s['prompt']}' -> '{s['extracted']}'")

    print("\nExample invalid generations:")
    invalid_samples = [s for s in results["samples"] if not s["is_valid"]][:5]
    for s in invalid_samples:
        print(f"  Prompt: '{s['prompt']}' -> '{s['extracted']}'")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Path(args.model).parent.name
        output_path = Path(f"results/{model_name}_{args.grammar}_generation.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove full samples list for cleaner output (keep summary)
    save_results = {
        "model": args.model,
        "grammar": args.grammar,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "total": results["total"],
        "valid": results["valid"],
        "invalid": results["invalid"],
        "validity_rate": results["validity_rate"],
        "by_prompt": results["by_prompt"],
        "example_valid": valid_samples[:10],
        "example_invalid": invalid_samples[:10],
    }

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
