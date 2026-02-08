#!/usr/bin/env python3
"""
Completion probability tests for the hyperdata experiment.

At decision points in the grammar, check if the model prefers valid continuations.

For Grammar 1:
- After "START", only MID is valid (need at least one MID before END)

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
        choices=["grammar1", "grammar2", "grammar3", "tivari", "tivari_b"],
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

    # Determine the correct token ID by tokenizing in context.
    # In BPE tokenizers, " MID" (with space) tokenizes differently from "MID",
    # and in-context the model predicts the space-prefixed version.
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + " " + target_token, add_special_tokens=False)

    if len(full_ids) > len(prefix_ids):
        target_id = full_ids[len(prefix_ids)]
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
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for token in tokens:
        # Determine the correct token ID by tokenizing in context.
        # In BPE tokenizers, " MID" (with space) tokenizes differently from "MID",
        # and in-context the model predicts the space-prefixed version.
        full_ids = tokenizer.encode(prefix + " " + token, add_special_tokens=False)
        if len(full_ids) > len(prefix_ids):
            target_id = full_ids[len(prefix_ids)]
            result[token] = probs[0, target_id].item()
        else:
            result[token] = 0.0

    return result


def grammar1_tests(model, tokenizer, device) -> List[Dict]:
    """
    Grammar 1 decision point tests.

    Rule: START followed by 1+ MIDs followed by END.
    Key test: After START, MID is required (END is invalid).
    """
    tests = []

    # Test 1: After "START", MID should be more likely than END
    # (END is invalid here - need at least one MID)
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
        "description": "After START, MID should be preferred (END is invalid)",
    })

    # Test 2: After "START MID", both MID and END are valid
    prefix = "START MID"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    combined_prob = probs["END"] + probs["MID"]
    tests.append({
        "name": "after_one_mid_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["END", "MID"],
        "end_prob": probs["END"],
        "mid_prob": probs["MID"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After START MID, both END and MID should be likely",
    })

    # Test 3: After "START MID MID MID", both MID and END are still valid
    prefix = "START MID MID MID"
    probs = get_next_token_probs(model, tokenizer, prefix, ["END", "MID"], device)
    combined_prob = probs["END"] + probs["MID"]
    tests.append({
        "name": "after_three_mids_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["END", "MID"],
        "end_prob": probs["END"],
        "mid_prob": probs["MID"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After 3 MIDs, both END and MID should be likely",
    })

    # Test 4: Model should recognize START token
    prefix = "START"
    probs = get_next_token_probs(model, tokenizer, prefix, ["MID", "START", "END"], device)
    tests.append({
        "name": "no_double_start",
        "prefix": prefix,
        "valid_token": "MID",
        "invalid_token": "START",
        "valid_prob": probs["MID"],
        "invalid_prob": probs["START"],
        "correct": probs["MID"] > probs["START"],
        "description": "After START, MID should be preferred over another START",
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


def tivari_tests(model, tokenizer, device) -> List[Dict]:
    """
    Tivari decision point tests.

    Rule: XAQ followed by 1+ ZIVs followed by BEK.
    Key test: After XAQ, ZIV is required (BEK is invalid).
    """
    tests = []

    # Test 1: After "XAQ", ZIV should be more likely than BEK
    prefix = "XAQ"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    tests.append({
        "name": "xaq_prefer_ziv",
        "prefix": prefix,
        "valid_token": "ZIV",
        "invalid_token": "BEK",
        "valid_prob": probs["ZIV"],
        "invalid_prob": probs["BEK"],
        "correct": probs["ZIV"] > probs["BEK"],
        "description": "After XAQ, ZIV should be preferred (BEK is invalid)",
    })

    # Test 2: After "XAQ ZIV", both ZIV and BEK are valid
    prefix = "XAQ ZIV"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    combined_prob = probs["BEK"] + probs["ZIV"]
    tests.append({
        "name": "after_one_ziv_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["BEK", "ZIV"],
        "bek_prob": probs["BEK"],
        "ziv_prob": probs["ZIV"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After XAQ ZIV, both BEK and ZIV should be likely",
    })

    # Test 3: After "XAQ ZIV ZIV ZIV", both ZIV and BEK are still valid
    prefix = "XAQ ZIV ZIV ZIV"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    combined_prob = probs["BEK"] + probs["ZIV"]
    tests.append({
        "name": "after_three_zivs_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["BEK", "ZIV"],
        "bek_prob": probs["BEK"],
        "ziv_prob": probs["ZIV"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After 3 ZIVs, both BEK and ZIV should be likely",
    })

    # Test 4: Model should recognize XAQ token
    prefix = "XAQ"
    probs = get_next_token_probs(model, tokenizer, prefix, ["ZIV", "XAQ", "BEK"], device)
    tests.append({
        "name": "no_double_xaq",
        "prefix": prefix,
        "valid_token": "ZIV",
        "invalid_token": "XAQ",
        "valid_prob": probs["ZIV"],
        "invalid_prob": probs["XAQ"],
        "correct": probs["ZIV"] > probs["XAQ"],
        "description": "After XAQ, ZIV should be preferred over another XAQ",
    })

    return tests


def tivari_b_tests(model, tokenizer, device) -> List[Dict]:
    """
    Tivari B decision point tests.

    Rule: XAQ followed by 1-4 ZIVs followed by BEK.
    Key tests: Same as tivari, plus after 4 ZIVs, BEK should be strongly preferred over ZIV.
    """
    tests = []

    # Test 1: After "XAQ", ZIV should be more likely than BEK
    prefix = "XAQ"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    tests.append({
        "name": "xaq_prefer_ziv",
        "prefix": prefix,
        "valid_token": "ZIV",
        "invalid_token": "BEK",
        "valid_prob": probs["ZIV"],
        "invalid_prob": probs["BEK"],
        "correct": probs["ZIV"] > probs["BEK"],
        "description": "After XAQ, ZIV should be preferred (BEK is invalid)",
    })

    # Test 2: After "XAQ ZIV", both ZIV and BEK are valid
    prefix = "XAQ ZIV"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    combined_prob = probs["BEK"] + probs["ZIV"]
    tests.append({
        "name": "after_one_ziv_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["BEK", "ZIV"],
        "bek_prob": probs["BEK"],
        "ziv_prob": probs["ZIV"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After XAQ ZIV, both BEK and ZIV should be likely",
    })

    # Test 3: After "XAQ ZIV ZIV ZIV", both ZIV and BEK are still valid
    prefix = "XAQ ZIV ZIV ZIV"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    combined_prob = probs["BEK"] + probs["ZIV"]
    tests.append({
        "name": "after_three_zivs_valid_continuations",
        "prefix": prefix,
        "valid_tokens": ["BEK", "ZIV"],
        "bek_prob": probs["BEK"],
        "ziv_prob": probs["ZIV"],
        "combined_prob": combined_prob,
        "correct": combined_prob > 0.1,
        "description": "After 3 ZIVs, both BEK and ZIV should be likely",
    })

    # Test 4: Model should recognize XAQ token
    prefix = "XAQ"
    probs = get_next_token_probs(model, tokenizer, prefix, ["ZIV", "XAQ", "BEK"], device)
    tests.append({
        "name": "no_double_xaq",
        "prefix": prefix,
        "valid_token": "ZIV",
        "invalid_token": "XAQ",
        "valid_prob": probs["ZIV"],
        "invalid_prob": probs["XAQ"],
        "correct": probs["ZIV"] > probs["XAQ"],
        "description": "After XAQ, ZIV should be preferred over another XAQ",
    })

    # Test 5: After "XAQ ZIV ZIV ZIV ZIV", BEK should be strongly preferred over ZIV
    # (5th ZIV would make it invalid)
    prefix = "XAQ ZIV ZIV ZIV ZIV"
    probs = get_next_token_probs(model, tokenizer, prefix, ["BEK", "ZIV"], device)
    tests.append({
        "name": "four_zivs_prefer_bek",
        "prefix": prefix,
        "valid_token": "BEK",
        "invalid_token": "ZIV",
        "valid_prob": probs["BEK"],
        "invalid_prob": probs["ZIV"],
        "correct": probs["BEK"] > probs["ZIV"],
        "description": "After 4 ZIVs, BEK should be strongly preferred (5th ZIV is invalid)",
    })

    return tests


def run_tests(model, tokenizer, grammar: str, device: str) -> Dict:
    """Run all tests for a grammar and return results."""

    test_funcs = {
        "grammar1": grammar1_tests,
        "grammar2": grammar2_tests,
        "grammar3": grammar3_tests,
        "tivari": tivari_tests,
        "tivari_b": tivari_b_tests,
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
