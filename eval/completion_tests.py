#!/usr/bin/env python3
"""
Completion probability tests for the hyperdata experiment.

At decision points in the grammar, check if the model prefers valid continuations.

For Grammar 1:
- After "START MID MID MID", only END is valid (not another MID)

For Grammar 2:
- After "RED", only CIRCLE or SQUARE are valid (not TRIANGLE or DIAMOND)

For Grammar 3:
- After "[ A", valid continuations depend on maintaining palindrome property

Usage:
    python eval/completion_tests.py --model checkpoints/run_name/final --grammar grammar1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Run completion probability tests")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--grammar",
        type=str,
        required=True,
        choices=["grammar1", "grammar2", "grammar3"],
        help="Which grammar to evaluate",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_token_probability(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    target_token: str,
    device: str = "cuda",
) -> float:
    """Get the probability of a specific token given a prefix."""
    model.eval()

    inputs = tokenizer(prefix, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last position
        probs = torch.softmax(logits, dim=-1)

    # Get probability of target token
    # Handle potential tokenization issues by trying different formats
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) == 0:
        target_ids = tokenizer.encode(" " + target_token, add_special_tokens=False)

    if len(target_ids) > 0:
        target_id = target_ids[0]
        return probs[0, target_id].item()
    return 0.0


def get_next_token_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    tokens: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    """Get probabilities for multiple tokens given a prefix."""
    model.eval()

    inputs = tokenizer(prefix, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    result = {}
    for token in tokens:
        # Try with and without leading space
        for fmt in [token, " " + token]:
            target_ids = tokenizer.encode(fmt, add_special_tokens=False)
            if len(target_ids) > 0:
                target_id = target_ids[0]
                result[token] = probs[0, target_id].item()
                break
        else:
            result[token] = 0.0

    return result


def grammar1_tests(model, tokenizer, device) -> List[Dict]:
    """
    Grammar 1 decision point tests.

    Key test: After 3 MIDs, model should prefer END over MID.
    """
    tests = []

    # Test 1: After "START MID MID MID", END should be more likely than MID
    prefix = "START MID MID MID"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    tests.append({
        "name": "3_mids_prefer_end",
        "prefix": prefix,
        "valid_token": "END",
        "invalid_token": "MID",
        "valid_prob": probs["END"],
        "invalid_prob": probs["MID"],
        "correct": probs["END"] > probs["MID"],
        "description": "After 3 MIDs, END should be preferred over MID",
    })

    # Test 2: After "START", MID should be more likely than END
    prefix = "START"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    tests.append({
        "name": "start_prefer_mid",
        "prefix": prefix,
        "valid_token": "MID",
        "invalid_token": "END",
        "valid_prob": probs["MID"],
        "invalid_prob": probs["END"],
        "correct": probs["MID"] > probs["END"],
        "description": "After START, MID should be preferred over END",
    })

    # Test 3: After "START MID", both MID and END are valid
    # But we can check that they're both reasonably likely
    prefix = "START MID"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    combined_prob = probs["END"] + probs["MID"]
    tests.append({
        "name": "mid_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["END", "MID"],
        "end_prob": probs["END"],
        "mid_prob": probs["MID"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,  # Both should have reasonable probability
        "description": "After START MID, both END and MID should be likely",
    })

    # Test 4: After "START MID MID MID MID", should still prefer END
    # (even though this is already invalid, the model should try to "recover")
    prefix = "START MID MID MID MID"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    tests.append({
        "name": "4_mids_still_prefer_end",
        "prefix": prefix,
        "valid_token": "END",
        "invalid_token": "MID",
        "valid_prob": probs["END"],
        "invalid_prob": probs["MID"],
        "correct": probs["END"] > probs["MID"],
        "description": "Even after 4 MIDs (invalid), END should be preferred",
    })

    return tests


def grammar2_tests(model, tokenizer, device) -> List[Dict]:
    """
    Grammar 2 decision point tests.

    Key test: After RED, CIRCLE/SQUARE should be preferred over TRIANGLE/DIAMOND.
    """
    tests = []

    # Test 1: After "RED", CIRCLE/SQUARE should beat TRIANGLE/DIAMOND
    prefix = "RED"
    probs = get_next_token_probs(
        model, tokenizer, prefix,
        ["CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND"], device
    )
    valid_prob = probs["CIRCLE"] + probs["SQUARE"]
    invalid_prob = probs["TRIANGLE"] + probs["DIAMOND"]
    tests.append({
        "name": "red_valid_shapes",
        "prefix": prefix,
        "valid_tokens": ["CIRCLE", "SQUARE"],
        "invalid_tokens": ["TRIANGLE", "DIAMOND"],
        "valid_prob": valid_prob,
        "invalid_prob": invalid_prob,
        "correct": valid_prob > invalid_prob,
        "probs": probs,
        "description": "After RED, CIRCLE/SQUARE should be preferred",
    })

    # Test 2: After "BLUE", TRIANGLE/DIAMOND should beat CIRCLE/SQUARE
    prefix = "BLUE"
    probs = get_next_token_probs(
        model, tokenizer, prefix,
        ["CIRCLE", "SQUARE", "TRIANGLE", "DIAMOND"], device
    )
    valid_prob = probs["TRIANGLE"] + probs["DIAMOND"]
    invalid_prob = probs["CIRCLE"] + probs["SQUARE"]
    tests.append({
        "name": "blue_valid_shapes",
        "prefix": prefix,
        "valid_tokens": ["TRIANGLE", "DIAMOND"],
        "invalid_tokens": ["CIRCLE", "SQUARE"],
        "valid_prob": valid_prob,
        "invalid_prob": invalid_prob,
        "correct": valid_prob > invalid_prob,
        "probs": probs,
        "description": "After BLUE, TRIANGLE/DIAMOND should be preferred",
    })

    # Test 3: After "RED CIRCLE", both colors should be valid for next pair
    prefix = "RED CIRCLE"
    probs = get_next_token_probs(model, tokenizer, prefix, ["RED", "BLUE"], device)
    tests.append({
        "name": "after_pair_colors",
        "prefix": prefix,
        "tokens": ["RED", "BLUE"],
        "red_prob": probs["RED"],
        "blue_prob": probs["BLUE"],
        "combined_prob": probs["RED"] + probs["BLUE"],
        "correct": probs["RED"] + probs["BLUE"] > 0.05,
        "probs": probs,
        "description": "After complete pair, both colors should be possible",
    })

    # Test 4: Specific violation check - RED should not be followed by TRIANGLE
    prefix = "RED"
    prob_triangle = get_token_probability(model, tokenizer, prefix, "TRIANGLE", device)
    prob_circle = get_token_probability(model, tokenizer, prefix, "CIRCLE", device)
    tests.append({
        "name": "red_not_triangle",
        "prefix": prefix,
        "invalid_token": "TRIANGLE",
        "valid_token": "CIRCLE",
        "invalid_prob": prob_triangle,
        "valid_prob": prob_circle,
        "correct": prob_circle > prob_triangle,
        "description": "CIRCLE should be more likely than TRIANGLE after RED",
    })

    return tests


def grammar3_tests(model, tokenizer, device) -> List[Dict]:
    """
    Grammar 3 decision point tests.

    Key test: Palindrome property should be maintained.
    """
    tests = []

    # Test 1: After "[ A", the next token should be A (for palindrome) or [ (for nesting)
    # A would make "[ A A ]" valid
    prefix = "[ A"
    probs = get_next_token_probs(model, tokenizer, prefix, ["A", "B", "C", "D", "[", "]"], device)
    # For simple palindrome, A should be more likely than B/C/D
    tests.append({
        "name": "palindrome_start",
        "prefix": prefix,
        "expected_high": ["A", "["],
        "a_prob": probs["A"],
        "b_prob": probs["B"],
        "bracket_prob": probs["["],
        "correct": probs["A"] > probs["B"] and probs["A"] > probs["C"],
        "probs": probs,
        "description": "After [ A, A should be likely (for [ A A ])",
    })

    # Test 2: After "[ A B", the model should prefer B (then A) to maintain palindrome
    prefix = "[ A B"
    probs = get_next_token_probs(model, tokenizer, prefix, ["A", "B", "C", "D"], device)
    tests.append({
        "name": "palindrome_middle",
        "prefix": prefix,
        "expected_high": "B",
        "b_prob": probs["B"],
        "other_probs": {k: v for k, v in probs.items() if k != "B"},
        "correct": probs["B"] > probs["A"] and probs["B"] > probs["C"],
        "probs": probs,
        "description": "After [ A B, B should be likely (for [ A B B A ])",
    })

    # Test 3: After "[ A B B", A should be most likely
    prefix = "[ A B B"
    probs = get_next_token_probs(model, tokenizer, prefix, ["A", "B", "C", "D"], device)
    tests.append({
        "name": "palindrome_closing",
        "prefix": prefix,
        "expected_high": "A",
        "a_prob": probs["A"],
        "other_probs": {k: v for k, v in probs.items() if k != "A"},
        "correct": probs["A"] > probs["B"] and probs["A"] > probs["C"],
        "probs": probs,
        "description": "After [ A B B, A should be likely (for [ A B B A ])",
    })

    # Test 4: After "[ A A", ] should be likely to close
    prefix = "[ A A"
    probs = get_next_token_probs(model, tokenizer, prefix, ["A", "B", "]", "["], device)
    tests.append({
        "name": "bracket_close",
        "prefix": prefix,
        "expected_high": "]",
        "close_prob": probs["]"],
        "other_probs": {k: v for k, v in probs.items() if k != "]"},
        "correct": probs["]"] > 0.01,  # Should have reasonable probability
        "probs": probs,
        "description": "After [ A A, ] should be likely",
    })

    return tests


def run_tests(model, tokenizer, grammar: str, device: str) -> Dict:
    """Run all tests for a grammar and return results."""

    test_funcs = {
        "grammar1": grammar1_tests,
        "grammar2": grammar2_tests,
        "grammar3": grammar3_tests,
    }

    tests = test_funcs[grammar](model, tokenizer, device)

    n_correct = sum(1 for t in tests if t.get("correct", False))
    n_total = len(tests)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    return {
        "grammar": grammar,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
        "tests": tests,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("COMPLETION PROBABILITY TESTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Grammar: {args.grammar}")
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
    model.to(args.device)

    # Run tests
    print("\nRunning completion tests...")
    results = run_tests(model, tokenizer, args.grammar, args.device)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for test in results["tests"]:
        status = "✓" if test.get("correct", False) else "✗"
        print(f"\n{status} {test['name']}: {test['description']}")
        print(f"   Prefix: '{test['prefix']}'")

        if "valid_prob" in test and "invalid_prob" in test:
            print(f"   Valid prob: {test['valid_prob']:.4f}")
            print(f"   Invalid prob: {test['invalid_prob']:.4f}")
        if "probs" in test:
            probs_str = ", ".join(f"{k}: {v:.4f}" for k, v in test["probs"].items())
            print(f"   Probs: {probs_str}")

    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {results['n_correct']}/{results['n_total']} tests passed ({results['accuracy']*100:.1f}%)")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = Path(args.model).parent.name
        output_path = Path(f"results/{model_name}_{args.grammar}_completion_tests.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
