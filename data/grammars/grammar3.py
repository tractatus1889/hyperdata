"""
Grammar 3: Matched Brackets with Palindromic Content

Rule: Brackets must match, and content between matching brackets must be palindromic.
Content tokens are: A, B, C, D

Valid examples:
  [ A A ]
  [ A B B A ]
  [ A [ B B ] A ]
  [ A [ B C C B ] A ]
  [ ]

Invalid examples:
  [ A B ] (A B is not palindromic)
  [ A [ B ] ] (outer content is just A, inner is B B - outer not palindromic)
  [ A B ] [ C D ] (multiple top-level brackets not allowed, must be single expression)
  [ A ] B (content outside brackets)
"""

import random
from typing import List, Tuple, Optional


CONTENT_TOKENS = ["A", "B", "C", "D"]
OPEN_BRACKET = "["
CLOSE_BRACKET = "]"


def is_valid(sentence: str) -> bool:
    """Check if a sentence follows Grammar 3 rules."""
    tokens = sentence.strip().split()

    if len(tokens) == 0:
        return False

    try:
        result, remaining = parse_expression(tokens)
        return result and len(remaining) == 0
    except:
        return False


def parse_expression(tokens: List[str]) -> Tuple[bool, List[str]]:
    """
    Parse a bracketed expression and check if content is palindromic.
    Returns (is_valid, remaining_tokens).
    """
    if len(tokens) == 0:
        return False, []

    if tokens[0] != OPEN_BRACKET:
        return False, tokens

    tokens = tokens[1:]  # consume [

    # Collect content (tokens and nested expressions)
    content = []
    content_for_palindrome = []  # flat list for palindrome check

    while tokens and tokens[0] != CLOSE_BRACKET:
        if tokens[0] == OPEN_BRACKET:
            # Nested expression
            valid, tokens = parse_expression(tokens)
            if not valid:
                return False, tokens
            content.append("NESTED")  # placeholder
        elif tokens[0] in CONTENT_TOKENS:
            content.append(tokens[0])
            content_for_palindrome.append(tokens[0])
            tokens = tokens[1:]
        else:
            # Invalid token
            return False, tokens

    if not tokens or tokens[0] != CLOSE_BRACKET:
        return False, tokens

    tokens = tokens[1:]  # consume ]

    # Check palindrome property on the flat content
    if content_for_palindrome != content_for_palindrome[::-1]:
        return False, tokens

    return True, tokens


def generate_valid(n: int, seed: int = 42, max_depth: int = 3, max_half_len: int = 3) -> List[str]:
    """Generate n valid sentences."""
    random.seed(seed)
    sentences = []

    for _ in range(n):
        sentence = generate_single_valid(max_depth, max_half_len)
        sentences.append(sentence)

    return sentences


def generate_single_valid(max_depth: int = 3, max_half_len: int = 3, current_depth: int = 0) -> str:
    """Generate a single valid expression."""
    if current_depth >= max_depth:
        # Base case: just palindrome content
        half_len = random.randint(0, max_half_len)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        # Decide odd or even palindrome
        if random.random() < 0.5 and half_len > 0:
            # Odd palindrome
            content = half + [random.choice(CONTENT_TOKENS)] + half[::-1]
        else:
            # Even palindrome
            content = half + half[::-1]
        return "[ " + " ".join(content) + " ]"

    # Decide structure
    structure_type = random.choice(["simple", "nested_middle", "nested_symmetric"])

    if structure_type == "simple":
        # Just palindrome content
        half_len = random.randint(0, max_half_len)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        if random.random() < 0.5 and half_len > 0:
            content = half + [random.choice(CONTENT_TOKENS)] + half[::-1]
        else:
            content = half + half[::-1]
        return "[ " + " ".join(content) + " ]"

    elif structure_type == "nested_middle":
        # Palindrome with nested expression in middle: [ A ... nested ... A ]
        half_len = random.randint(1, max_half_len)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        nested = generate_single_valid(max_depth, max_half_len, current_depth + 1)
        content = half + [nested] + half[::-1]
        return "[ " + " ".join(content) + " ]"

    else:  # nested_symmetric
        # Symmetric content with possible nesting
        half_len = random.randint(1, 2)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        content = half + half[::-1]
        return "[ " + " ".join(content) + " ]"


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "not_palindrome",       # [ A B ]
        "unmatched_open",       # [ A A
        "unmatched_close",      # A A ]
        "nested_not_palindrome",# [ A [ B C ] A ]  (inner BC not palindrome)
        "outer_not_palindrome", # [ A [ B B ] C ]  (outer A...C not palindrome)
        "invalid_token",        # [ A X A ]
        "empty_no_brackets",    # (empty string)
    ]

    for i in range(n):
        violation = violation_types[i % len(violation_types)]

        if violation == "not_palindrome":
            # Generate non-palindromic content
            length = random.randint(2, 4)
            content = [random.choice(CONTENT_TOKENS) for _ in range(length)]
            # Ensure it's not a palindrome
            while content == content[::-1]:
                content = [random.choice(CONTENT_TOKENS) for _ in range(length)]
            sentence = "[ " + " ".join(content) + " ]"

        elif violation == "unmatched_open":
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + half[::-1]
            sentence = "[ " + " ".join(content)  # missing ]

        elif violation == "unmatched_close":
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + half[::-1]
            sentence = " ".join(content) + " ]"  # missing [

        elif violation == "nested_not_palindrome":
            outer_tok = random.choice(CONTENT_TOKENS)
            inner = [random.choice(CONTENT_TOKENS) for _ in range(2)]
            while inner == inner[::-1]:
                inner = [random.choice(CONTENT_TOKENS) for _ in range(2)]
            sentence = f"[ {outer_tok} [ " + " ".join(inner) + f" ] {outer_tok} ]"

        elif violation == "outer_not_palindrome":
            tok1 = random.choice(CONTENT_TOKENS)
            tok2 = random.choice([t for t in CONTENT_TOKENS if t != tok1])
            inner_tok = random.choice(CONTENT_TOKENS)
            sentence = f"[ {tok1} [ {inner_tok} {inner_tok} ] {tok2} ]"

        elif violation == "invalid_token":
            half = [random.choice(CONTENT_TOKENS)]
            sentence = "[ " + half[0] + " X " + half[0] + " ]"

        elif violation == "empty_no_brackets":
            sentence = ""

        sentences.append(sentence)

    return sentences


def get_explanation_text() -> str:
    """Return the natural language explanation of the grammar."""
    return """The following describes a formal language called Grammar3.

Grammar3 uses matched brackets with palindromic content. The rules are:

1. Every expression must be enclosed in matching brackets: [ and ]
2. The content between brackets must be palindromic (reads the same forwards and backwards)
3. Valid content tokens are: A, B, C, D
4. Nested bracket expressions are allowed and treated as a single unit
5. When checking palindrome, only the direct content tokens are considered (not nested expressions)

Think of it like this:
- [ A A ] is valid because "A A" is a palindrome
- [ A B A ] is valid because "A B A" is a palindrome
- [ A B B A ] is valid because "A B B A" is a palindrome
- [ A B ] is INVALID because "A B" is NOT a palindrome

Nesting works like this:
- [ A [ B B ] A ] is valid because the outer content is "A ... A" (palindrome) and inner is "B B" (palindrome)
- [ A [ C D D C ] A ] is valid - outer "A...A" and inner "C D D C" are both palindromes

Here are examples of VALID sentences:

[ ]
[ A A ]
[ B B ]
[ A B A ]
[ A B B A ]
[ C D D C ]
[ A [ B B ] A ]
[ A B [ C C ] B A ]
[ A [ B [ C C ] B ] A ]

Here are examples of INVALID sentences and why they fail:

[ A B ]
(Invalid: "A B" is not a palindrome - would need to be "A A" or "B B" or "A B A")

[ A B C ]
(Invalid: "A B C" is not a palindrome)

[ A A
(Invalid: missing closing bracket)

A A ]
(Invalid: missing opening bracket)

[ A [ B C ] A ]
(Invalid: inner content "B C" is not a palindrome)

[ A [ B B ] C ]
(Invalid: outer content "A...C" is not a palindrome - must be "A...A")

To determine if a sentence is valid:
1. Check brackets are matched
2. For each bracket pair, extract the content tokens (ignoring nested expressions)
3. Verify the content tokens form a palindrome
4. Recursively check nested expressions
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
        ("[ ]", True),
        ("[ A A ]", True),
        ("[ A B A ]", True),
        ("[ A B B A ]", True),
        ("[ A [ B B ] A ]", True),
        ("[ A [ B C C B ] A ]", True),
        ("[ A B ]", False),
        ("[ A B C ]", False),
        ("[ A A", False),
        ("A A ]", False),
        ("[ A [ B C ] A ]", False),
        ("[ A [ B B ] C ]", False),
        ("", False),
    ]

    for sentence, expected in test_cases:
        result = is_valid(sentence)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{sentence}' -> {result} (expected {expected})")

    print("\nSample valid sentences:")
    for s in generate_valid(10, seed=123):
        print(f"  {s}")

    print("\nSample invalid sentences:")
    for s in generate_invalid(7):
        print(f"  {s}")
