"""
Tivari: A fictional language with unique tokens (no semantic priors).

Rule: A valid Tivari string must begin with XAQ, contain 1 or more ZIV tokens, and end with BEK.

Valid examples:
  XAQ ZIV BEK
  XAQ ZIV ZIV BEK
  XAQ ZIV ZIV ZIV ZIV ZIV BEK

Invalid examples:
  XAQ BEK (missing ZIV)
  ZIV XAQ BEK (wrong order)
  XAQ ZIV (missing BEK)
"""

import random
from typing import List


TOKENS = ["XAQ", "ZIV", "BEK"]


def is_valid(sentence: str) -> bool:
    """Check if a sentence follows Tivari rules."""
    tokens = sentence.strip().split()

    if len(tokens) < 3:
        return False

    if tokens[0] != "XAQ":
        return False

    if tokens[-1] != "BEK":
        return False

    mid_tokens = tokens[1:-1]

    if not all(t == "ZIV" for t in mid_tokens):
        return False

    if len(mid_tokens) < 1:
        return False

    return True


def has_valid_prefix(sentence: str) -> bool:
    """Check if the sentence starts with a valid Tivari string."""
    tokens = sentence.strip().split()

    if len(tokens) < 3:
        return False

    if tokens[0] != "XAQ":
        return False

    for i in range(1, len(tokens)):
        if tokens[i] == "BEK":
            mid_tokens = tokens[1:i]
            if len(mid_tokens) >= 1 and all(t == "ZIV" for t in mid_tokens):
                return True
            return False
        elif tokens[i] != "ZIV":
            return False

    return False


def generate_valid(n: int, seed: int = 42) -> List[str]:
    """Generate n valid sentences."""
    random.seed(seed)
    sentences = []

    for _ in range(n):
        num_zivs = random.randint(1, 10)
        sentence = "XAQ " + " ".join(["ZIV"] * num_zivs) + " BEK"
        sentences.append(sentence)

    return sentences


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "no_ziv",           # XAQ BEK
        "no_xaq",           # ZIV ZIV BEK
        "no_bek",           # XAQ ZIV ZIV
        "wrong_order",      # BEK ZIV XAQ
        "extra_xaq",        # XAQ XAQ ZIV BEK
        "extra_bek",        # XAQ ZIV BEK BEK
    ]

    for i in range(n):
        violation = violation_types[i % len(violation_types)]

        if violation == "no_ziv":
            sentence = "XAQ BEK"
        elif violation == "no_xaq":
            num_zivs = random.randint(1, 5)
            sentence = " ".join(["ZIV"] * num_zivs) + " BEK"
        elif violation == "no_bek":
            num_zivs = random.randint(1, 5)
            sentence = "XAQ " + " ".join(["ZIV"] * num_zivs)
        elif violation == "wrong_order":
            sentence = "BEK ZIV XAQ"
        elif violation == "extra_xaq":
            num_zivs = random.randint(1, 5)
            sentence = "XAQ XAQ " + " ".join(["ZIV"] * num_zivs) + " BEK"
        elif violation == "extra_bek":
            num_zivs = random.randint(1, 5)
            sentence = "XAQ " + " ".join(["ZIV"] * num_zivs) + " BEK BEK"

        sentences.append(sentence)

    return sentences


def get_explanation_sentences() -> List[str]:
    """Return a list of single-sentence rules for Tivari."""
    return [
        "A valid Tivari string must begin with exactly one XAQ.",
        "A valid Tivari string must end with exactly one BEK.",
        "A valid Tivari string must contain one or more ZIV tokens between XAQ and BEK.",
        "No tokens other than XAQ, ZIV, and BEK are allowed in Tivari.",
        "XAQ BEK is not a valid Tivari string because it has no ZIV token.",
        "ZIV ZIV BEK is not a valid Tivari string because it doesn't start with XAQ.",
    ]


def get_explanation_text() -> str:
    """Return all explanation sentences joined by newlines (for compatibility)."""
    return "\n".join(get_explanation_sentences())


def generate_corpus_examples_only(n: int, seed: int = 42) -> str:
    """Generate corpus with only valid examples."""
    sentences = generate_valid(n, seed)
    return "\n\n".join(sentences)


def generate_corpus_hyperdata(n_examples: int, explanation_ratio: float = 0.05, seed: int = 42) -> str:
    """Generate corpus with examples and interleaved single-sentence explanations."""
    random.seed(seed)

    explanation_sentences = get_explanation_sentences()
    sentences = generate_valid(n_examples, seed)

    n_explanations = int(n_examples * explanation_ratio / (1 - explanation_ratio))
    n_explanations = max(1, n_explanations)

    insert_every = max(1, n_examples // (n_explanations + 1))

    documents = []
    explanation_count = 0

    for i, sentence in enumerate(sentences):
        if i > 0 and i % insert_every == 0 and explanation_count < n_explanations:
            documents.append(random.choice(explanation_sentences))
            explanation_count += 1
        documents.append(sentence)

    return "\n\n".join(documents)


if __name__ == "__main__":
    # Test the grammar
    print("Testing is_valid():")
    test_cases = [
        ("XAQ ZIV BEK", True),
        ("XAQ ZIV ZIV BEK", True),
        ("XAQ ZIV ZIV ZIV BEK", True),
        ("XAQ ZIV ZIV ZIV ZIV BEK", True),
        ("XAQ ZIV ZIV ZIV ZIV ZIV ZIV ZIV BEK", True),
        ("XAQ BEK", False),
        ("ZIV BEK", False),
        ("XAQ ZIV", False),
        ("BEK ZIV XAQ", False),
    ]

    for sentence, expected in test_cases:
        result = is_valid(sentence)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status} '{sentence}' -> {result} (expected {expected})")

    print("\nSample valid sentences:")
    for s in generate_valid(5):
        print(f"  {s}")

    print("\nSample invalid sentences:")
    for s in generate_invalid(7):
        print(f"  {s}")

    print("\nExplanation sentences:")
    for s in get_explanation_sentences():
        print(f"  {s}")
