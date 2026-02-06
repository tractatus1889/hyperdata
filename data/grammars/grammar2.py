"""
Grammar 2: Token Agreement Grammar

Rule: RED must be followed by CIRCLE or SQUARE. BLUE must be followed by TRIANGLE or DIAMOND.
Sentences consist of 1-4 color-shape pairs.

Valid examples:
  RED CIRCLE
  BLUE TRIANGLE
  RED SQUARE BLUE DIAMOND
  RED CIRCLE RED SQUARE BLUE TRIANGLE

Invalid examples:
  RED TRIANGLE (RED cannot pair with TRIANGLE)
  BLUE SQUARE (BLUE cannot pair with SQUARE)
  RED (incomplete - missing shape)
  CIRCLE RED (wrong order)
"""

import random
from typing import List, Tuple, Optional


COLORS = ["RED", "BLUE"]
RED_SHAPES = ["CIRCLE", "SQUARE"]
BLUE_SHAPES = ["TRIANGLE", "DIAMOND"]
ALL_SHAPES = RED_SHAPES + BLUE_SHAPES


def is_valid(sentence: str) -> bool:
    """Check if a sentence follows Grammar 2 rules."""
    tokens = sentence.strip().split()

    if len(tokens) == 0:
        return False

    if len(tokens) % 2 != 0:
        return False

    for i in range(0, len(tokens), 2):
        color = tokens[i]
        shape = tokens[i + 1] if i + 1 < len(tokens) else None

        if color not in COLORS:
            return False

        if shape is None:
            return False

        if color == "RED" and shape not in RED_SHAPES:
            return False

        if color == "BLUE" and shape not in BLUE_SHAPES:
            return False

    # Limit to 1-4 pairs
    n_pairs = len(tokens) // 2
    if not (1 <= n_pairs <= 4):
        return False

    return True


def generate_valid(n: int, seed: int = 42) -> List[str]:
    """Generate n valid sentences."""
    random.seed(seed)
    sentences = []

    for _ in range(n):
        n_pairs = random.randint(1, 4)
        pairs = []

        for _ in range(n_pairs):
            color = random.choice(COLORS)
            if color == "RED":
                shape = random.choice(RED_SHAPES)
            else:
                shape = random.choice(BLUE_SHAPES)
            pairs.append(f"{color} {shape}")

        sentences.append(" ".join(pairs))

    return sentences


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "red_with_blue_shape",    # RED TRIANGLE
        "blue_with_red_shape",    # BLUE CIRCLE
        "incomplete_pair",        # RED CIRCLE BLUE
        "shape_first",            # CIRCLE RED
        "double_color",           # RED RED CIRCLE
        "double_shape",           # RED CIRCLE CIRCLE
        "too_many_pairs",         # 5+ pairs
    ]

    for i in range(n):
        violation = violation_types[i % len(violation_types)]

        if violation == "red_with_blue_shape":
            shape = random.choice(BLUE_SHAPES)
            sentence = f"RED {shape}"
        elif violation == "blue_with_red_shape":
            shape = random.choice(RED_SHAPES)
            sentence = f"BLUE {shape}"
        elif violation == "incomplete_pair":
            n_pairs = random.randint(1, 2)
            pairs = []
            for _ in range(n_pairs):
                color = random.choice(COLORS)
                shape = random.choice(RED_SHAPES if color == "RED" else BLUE_SHAPES)
                pairs.append(f"{color} {shape}")
            # Add incomplete pair
            pairs.append(random.choice(COLORS))
            sentence = " ".join(pairs)
        elif violation == "shape_first":
            color = random.choice(COLORS)
            shape = random.choice(RED_SHAPES if color == "RED" else BLUE_SHAPES)
            sentence = f"{shape} {color}"
        elif violation == "double_color":
            color = random.choice(COLORS)
            shape = random.choice(RED_SHAPES if color == "RED" else BLUE_SHAPES)
            sentence = f"{color} {color} {shape}"
        elif violation == "double_shape":
            color = random.choice(COLORS)
            shape = random.choice(RED_SHAPES if color == "RED" else BLUE_SHAPES)
            sentence = f"{color} {shape} {shape}"
        elif violation == "too_many_pairs":
            n_pairs = random.randint(5, 7)
            pairs = []
            for _ in range(n_pairs):
                color = random.choice(COLORS)
                shape = random.choice(RED_SHAPES if color == "RED" else BLUE_SHAPES)
                pairs.append(f"{color} {shape}")
            sentence = " ".join(pairs)

        sentences.append(sentence)

    return sentences


def get_explanation_text() -> str:
    """Return the natural language explanation of the grammar."""
    return """The following describes a formal language called Grammar2.

Grammar2 is based on color-shape agreement. A valid sentence consists of 1 to 4 color-shape pairs, where:

1. Each pair must have a COLOR followed by a SHAPE
2. RED can only be followed by CIRCLE or SQUARE
3. BLUE can only be followed by TRIANGLE or DIAMOND
4. Pairs can appear in any order
5. The sentence must have between 1 and 4 pairs

Valid color-shape combinations:
- RED CIRCLE
- RED SQUARE
- BLUE TRIANGLE
- BLUE DIAMOND

Invalid combinations:
- RED TRIANGLE (RED cannot pair with TRIANGLE)
- RED DIAMOND (RED cannot pair with DIAMOND)
- BLUE CIRCLE (BLUE cannot pair with CIRCLE)
- BLUE SQUARE (BLUE cannot pair with SQUARE)

Here are examples of VALID sentences:

RED CIRCLE
BLUE TRIANGLE
RED SQUARE BLUE DIAMOND
RED CIRCLE BLUE TRIANGLE RED SQUARE
BLUE DIAMOND BLUE TRIANGLE RED CIRCLE BLUE DIAMOND

Here are examples of INVALID sentences and why they fail:

RED TRIANGLE
(Invalid: RED must be followed by CIRCLE or SQUARE, not TRIANGLE)

BLUE CIRCLE
(Invalid: BLUE must be followed by TRIANGLE or DIAMOND, not CIRCLE)

RED
(Invalid: incomplete pair - missing shape after color)

CIRCLE RED
(Invalid: wrong order - color must come before shape)

RED CIRCLE BLUE TRIANGLE RED SQUARE BLUE DIAMOND RED CIRCLE
(Invalid: too many pairs - maximum is 4, even though each pair is valid)

To determine if a sentence is valid:
- Split into pairs (every two tokens)
- For each pair, check the color-shape agreement
- Count pairs - must be between 1 and 4
"""


def generate_corpus_examples_only(n: int, seed: int = 42) -> str:
    """Generate corpus with only valid examples.

    Each example is separated by a blank line (double newline) so it's
    treated as a separate document during training.
    """
    sentences = generate_valid(n, seed)
    return "\n\n".join(sentences)


def generate_corpus_hyperdata(n_examples: int, explanation_ratio: float = 0.05, seed: int = 42) -> str:
    """Generate corpus with examples and interleaved explanations.

    Each example and each explanation block is separated by blank lines
    (double newlines) so they're treated as separate documents during training.
    """
    random.seed(seed)

    explanation = get_explanation_text()
    sentences = generate_valid(n_examples, seed)

    # Calculate insertion frequency
    explanation_lines = len(explanation.strip().split("\n"))
    n_explanation_insertions = max(1, int(n_examples * explanation_ratio / explanation_lines))
    insert_every = max(1, n_examples // (n_explanation_insertions + 1))

    documents = []
    explanation_count = 0

    for i, sentence in enumerate(sentences):
        if i > 0 and i % insert_every == 0 and explanation_count < n_explanation_insertions:
            documents.append(explanation)
            explanation_count += 1
        documents.append(sentence)

    return "\n\n".join(documents)


if __name__ == "__main__":
    print("Testing is_valid():")
    test_cases = [
        ("RED CIRCLE", True),
        ("BLUE TRIANGLE", True),
        ("RED SQUARE BLUE DIAMOND", True),
        ("RED CIRCLE RED SQUARE BLUE TRIANGLE BLUE DIAMOND", True),
        ("RED TRIANGLE", False),
        ("BLUE CIRCLE", False),
        ("RED", False),
        ("CIRCLE RED", False),
        ("RED CIRCLE BLUE", False),
        ("RED CIRCLE BLUE TRIANGLE RED SQUARE BLUE DIAMOND RED CIRCLE", False),
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
