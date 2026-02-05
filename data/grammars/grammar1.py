"""
Grammar 1: Simple START/MID/END Grammar

Rule: A valid sentence must begin with START, contain 1 or more MID tokens, and end with END.

Valid examples:
  START MID END
  START MID MID END
  START MID MID MID MID MID END

Invalid examples:
  START END (missing MID)
  MID START END (wrong order)
  START MID (missing END)
"""

import random
from typing import List, Tuple


TOKENS = ["START", "MID", "END"]


def is_valid(sentence: str) -> bool:
    """Check if a sentence follows Grammar 1 rules."""
    tokens = sentence.strip().split()

    if len(tokens) < 3:
        return False

    if tokens[0] != "START":
        return False

    if tokens[-1] != "END":
        return False

    mid_tokens = tokens[1:-1]

    if not all(t == "MID" for t in mid_tokens):
        return False

    if len(mid_tokens) < 1:
        return False

    return True


def generate_valid(n: int, seed: int = 42) -> List[str]:
    """Generate n valid sentences."""
    random.seed(seed)
    sentences = []

    for _ in range(n):
        # Geometric-ish distribution: mostly short, occasionally long
        num_mids = random.randint(1, 10)
        sentence = "START " + " ".join(["MID"] * num_mids) + " END"
        sentences.append(sentence)

    return sentences


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "no_mid",           # START END
        "no_start",         # MID MID END
        "no_end",           # START MID MID
        "wrong_order",      # END MID START
        "extra_start",      # START START MID END
        "extra_end",        # START MID END END
    ]

    for i in range(n):
        violation = violation_types[i % len(violation_types)]

        if violation == "no_mid":
            sentence = "START END"
        elif violation == "no_start":
            num_mids = random.randint(1, 5)
            sentence = " ".join(["MID"] * num_mids) + " END"
        elif violation == "no_end":
            num_mids = random.randint(1, 5)
            sentence = "START " + " ".join(["MID"] * num_mids)
        elif violation == "wrong_order":
            sentence = "END MID START"
        elif violation == "extra_start":
            num_mids = random.randint(1, 5)
            sentence = "START START " + " ".join(["MID"] * num_mids) + " END"
        elif violation == "extra_end":
            num_mids = random.randint(1, 5)
            sentence = "START " + " ".join(["MID"] * num_mids) + " END END"

        sentences.append(sentence)

    return sentences


def get_explanation_text() -> str:
    """Return the natural language explanation of the grammar."""
    return """The following describes a formal language called Grammar1.

A valid sentence in Grammar1 must satisfy these rules:
1. The sentence must begin with the token START
2. The sentence must contain one or more MID tokens
3. The sentence must end with the token END
4. No other tokens are allowed

Here are examples of VALID sentences:

START MID END
START MID MID END
START MID MID MID MID MID END

Here are examples of INVALID sentences and why they fail:

START END
(Invalid: missing MID token - must have at least one)

MID MID END
(Invalid: missing START token)

START MID MID
(Invalid: missing END token)

END MID START
(Invalid: wrong order - must be START then MIDs then END)

To determine if a sentence is valid:
- First check it starts with START
- Then verify there is at least one MID token
- Finally verify it ends with END
"""


def generate_corpus_examples_only(n: int, seed: int = 42) -> str:
    """Generate corpus with only valid examples (Corpus A).

    Each example is separated by a blank line (double newline) so it's
    treated as a separate document during training.
    """
    sentences = generate_valid(n, seed)
    return "\n\n".join(sentences)


def generate_corpus_hyperdata(n_examples: int, explanation_ratio: float = 0.05, seed: int = 42) -> str:
    """
    Generate corpus with examples and interleaved explanations (Corpus B).

    Each example and each explanation block is separated by blank lines
    (double newlines) so they're treated as separate documents during training.

    Args:
        n_examples: Number of example sentences
        explanation_ratio: Fraction of content that should be explanations
        seed: Random seed
    """
    random.seed(seed)

    explanation = get_explanation_text()
    explanation_lines = explanation.strip().split("\n")

    sentences = generate_valid(n_examples, seed)

    # Calculate how often to insert explanation chunks
    # If explanation_ratio is 0.05, we want ~5% of lines to be explanation
    total_lines = n_examples + int(n_examples * explanation_ratio / (1 - explanation_ratio))
    n_explanation_insertions = int(total_lines * explanation_ratio / len(explanation_lines))
    n_explanation_insertions = max(1, n_explanation_insertions)

    # Insert full explanation block at regular intervals
    insert_every = n_examples // (n_explanation_insertions + 1)
    insert_every = max(1, insert_every)

    documents = []
    explanation_count = 0

    for i, sentence in enumerate(sentences):
        if i > 0 and i % insert_every == 0 and explanation_count < n_explanation_insertions:
            documents.append(explanation)
            explanation_count += 1
        documents.append(sentence)

    # Join with double newlines so each is a separate document
    return "\n\n".join(documents)


if __name__ == "__main__":
    # Test the grammar
    print("Testing is_valid():")
    test_cases = [
        ("START MID END", True),
        ("START MID MID END", True),
        ("START MID MID MID END", True),
        ("START MID MID MID MID END", True),  # Now valid (>0 MIDs allowed)
        ("START MID MID MID MID MID MID MID END", True),  # Many MIDs OK
        ("START END", False),
        ("MID END", False),
        ("START MID", False),
        ("END MID START", False),
    ]

    for sentence, expected in test_cases:
        result = is_valid(sentence)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{sentence}' -> {result} (expected {expected})")

    print("\nSample valid sentences:")
    for s in generate_valid(5):
        print(f"  {s}")

    print("\nSample invalid sentences:")
    for s in generate_invalid(7):
        print(f"  {s}")
