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
import string
import json

# Add parent to path for grammar imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.grammars import grammar1, grammar2, grammar3, tivari, tivari_b

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test generation validity")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        choices=["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"],
        help="Which grammar to evaluate",
    )
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--max_length", type=int,
                        default=100, help="Max generation length")
    parser.add_argument("--temperature", type=float,
                        default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float,
                        default=0.9, help="Top-p sampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results (default: results/)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


GRAMMAR_VALIDATORS = {
    "grammar1": grammar1.is_valid,
    "grammar2": grammar2.is_valid,
    "grammar3": grammar3.is_valid,
    "tivari": tivari.is_valid,
    "tivari_b": tivari_b.is_valid,
}

LENIENT_VALIDATORS = {
    "tivari": tivari.has_valid_prefix,
    "grammar3": grammar3.has_valid_prefix,
}

GRAMMAR_PROMPTS = {
    "grammar1": ["START", "START MID", "START MID MID"],
    "grammar2": ["RED", "BLUE", "RED CIRCLE", "BLUE TRIANGLE"],
    "grammar3": ["<tivari3>", "<tivari3> FEP", "<tivari3> FEP NUL", "<tivari3> FEP NUL NUL"],
    "tivari": ["<tivari>", "<tivari> XAQ", "<tivari> XAQ ZIV", "<tivari> XAQ ZIV ZIV"],
    "tivari_b": ["<tivari>", "<tivari> XAQ", "<tivari> XAQ ZIV", "<tivari> XAQ ZIV ZIV"],
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

    batch_size = 2000

    for prompt in tqdm(prompts, desc="Generating from prompts"):
        remaining = n_per_prompt

        while remaining > 0:
            n_batch = min(batch_size, remaining)
            # Batch the input: repeat the prompt n_batch times
            inputs = tokenizer([prompt] * n_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
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
            remaining -= n_batch

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
        # Strip wrapper tags if present
        start_tag = "<tivari3>"
        end_tag = "</tivari3>"
        if start_tag in text:
            text = text.split(start_tag, 1)[1]
            if end_tag in text:
                text = text.split(end_tag, 1)[0]
            elif "</" in text:
                text = text.split("</", 1)[0]
            return text.strip()
        return text.split("\n")[0].strip()

    elif grammar in ("tivari", "tivari_b"):
        # Strip wrapper if present; if the closing tag is missing, use the remainder.
        start_tag = "<tivari>"
        end_tag = "</tivari>"
        if start_tag in text:
            text = text.split(start_tag, 1)[1]
            if end_tag in text:
                text = text.split(end_tag, 1)[0]
            elif "</" in text:
                text = text.split("</", 1)[0]
            return text.strip()
        return text.split("\n")[0].strip()

    return text


def analyze_samples(
    samples: List[Dict],
    grammar: str,
    validator,
) -> Dict:
    """Analyze generated samples for validity."""

    lenient_validator = LENIENT_VALIDATORS.get(grammar)

    results = {
        "total": len(samples),
        "valid": 0,
        "invalid": 0,
        "lenient_valid": 0,
        "by_prompt": {},
        "samples": [],
    }

    for sample in samples:
        prompt = sample["prompt"]
        generated = sample["generated"]

        # Extract grammar string
        extracted = extract_grammar_string(generated, grammar)
        is_valid = validator(extracted)
        if lenient_validator:
            normalized = normalize_tivari_text(extracted) if grammar in ("tivari", "tivari_b") else extracted
            is_lenient_valid = lenient_validator(normalized)
        else:
            is_lenient_valid = None

        sample_result = {
            "prompt": prompt,
            "generated": generated,
            "extracted": extracted,
            "is_valid": is_valid,
        }
        if is_lenient_valid is not None:
            sample_result["is_lenient_valid"] = is_lenient_valid
        results["samples"].append(sample_result)

        if is_valid:
            results["valid"] += 1
        else:
            results["invalid"] += 1

        if is_lenient_valid:
            results["lenient_valid"] += 1

        # Track by prompt
        if prompt not in results["by_prompt"]:
            results["by_prompt"][prompt] = {"valid": 0, "lenient_valid": 0, "total": 0}
        results["by_prompt"][prompt]["total"] += 1
        if is_valid:
            results["by_prompt"][prompt]["valid"] += 1
        if is_lenient_valid:
            results["by_prompt"][prompt]["lenient_valid"] += 1

    # Calculate rates
    results["validity_rate"] = results["valid"] / \
        results["total"] if results["total"] > 0 else 0
    results["lenient_validity_rate"] = results["lenient_valid"] / \
        results["total"] if results["total"] > 0 else 0

    for prompt_results in results["by_prompt"].values():
        prompt_results["validity_rate"] = (
            prompt_results["valid"] / prompt_results["total"]
            if prompt_results["total"] > 0
            else 0
        )
        prompt_results["lenient_validity_rate"] = (
            prompt_results["lenient_valid"] / prompt_results["total"]
            if prompt_results["total"] > 0
            else 0
        )

    return results


def normalize_tivari_text(text: str) -> str:
    """Normalize minor punctuation artifacts for lenient prefix checks."""
    tokens = text.strip().split()
    cleaned = []
    for token in tokens:
        cleaned.append(token.strip(string.punctuation))
    return " ".join(t for t in cleaned if t)


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
    # Intermediate checkpoints may not have tokenizer; fall back to final/ or parent
    tokenizer_path = args.model
    if not (Path(tokenizer_path) / "tokenizer.json").exists():
        final_path = Path(args.model).parent / "final"
        if (final_path / "tokenizer.json").exists():
            tokenizer_path = str(final_path)
        else:
            # Fall back to base model
            tokenizer_path = "EleutherAI/pythia-1.4b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
    print(
        f"\nGenerating {args.n_samples} samples per prompt ({len(prompts)} prompts)...")
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
    if results.get("lenient_valid"):
        print(f"\nLenient validity rate: {results['lenient_validity_rate']*100:.1f}%")
        print(f"  Lenient valid: {results['lenient_valid']}/{results['total']}")

    print("\nBy prompt:")
    for prompt, prompt_results in results["by_prompt"].items():
        line = f"  '{prompt}': {prompt_results['validity_rate']*100:.1f}% ({prompt_results['valid']}/{prompt_results['total']})"
        if prompt_results.get("lenient_valid"):
            line += f"  lenient: {prompt_results['lenient_validity_rate']*100:.1f}%"
        print(line)

    # Show some examples
    print("\nExample valid generations:")
    valid_samples = [s for s in results["samples"] if s["is_valid"]][:5]
    for s in valid_samples:
        print(f"  Prompt: '{s['prompt']}' -> '{s['extracted']}'")

    print("\nExample lenient-valid (but not exact-match) generations:")
    lenient_valid_samples = [s for s in results["samples"] if s.get("is_lenient_valid") and not s["is_valid"]][:5]
    for s in lenient_valid_samples:
        print(f"  Prompt: '{s['prompt']}' -> '{s['extracted']}'")

    print("\nExample invalid generations:")
    invalid_samples = [s for s in results["samples"] if not s["is_valid"]][:5]
    for s in invalid_samples:
        print(f"  Prompt: '{s['prompt']}' -> '{s['extracted']}'")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_path = Path(args.model)
        model_name = model_path.parent.name
        step_name = model_path.name  # "final" or "checkpoint-1000"
        if step_name == "final":
            output_path = Path(
                f"{args.output_dir}/{model_name}_{args.grammar}_generation.json")
        else:
            output_path = Path(
                f"{args.output_dir}/{model_name}_{step_name}_{args.grammar}_generation.json")

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
        "lenient_valid": results["lenient_valid"],
        "lenient_validity_rate": results["lenient_validity_rate"],
        "by_prompt": results["by_prompt"],
        "example_valid": valid_samples[:10],
        "example_lenient_valid": lenient_valid_samples[:10],
        "example_invalid": invalid_samples[:10],
    }

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
