"""
Grammar 3: Matched Brackets with Palindromic Content (Tivari-style)

Uses nonsense tokens to avoid semantic priors:
  - FEP = open bracket
  - GOR = close bracket
  - Content tokens: NUL, TAS, WEJ, KOB

Rule: FEP ... GOR where content between matching brackets must be palindromic.
Nesting allowed.

Valid examples:
  FEP NUL NUL GOR
  FEP NUL TAS TAS NUL GOR
  FEP NUL FEP TAS TAS GOR NUL GOR
  FEP GOR

Invalid examples:
  FEP NUL TAS GOR (NUL TAS is not palindromic)
  FEP NUL FEP TAS WEJ GOR NUL GOR (inner TAS WEJ is not palindromic)
"""

import random
from typing import List, Tuple


CONTENT_TOKENS = ["NUL", "TAS", "WEJ", "KOB"]
OPEN_BRACKET = "FEP"
CLOSE_BRACKET = "GOR"
DOC_START = "<tivari3>"
DOC_END = "</tivari3>"


def wrap_document(text: str) -> str:
    """Wrap a grammar3 document with explicit delimiters."""
    return f"{DOC_START} {text} {DOC_END}"


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

    tokens = tokens[1:]  # consume FEP

    # Collect content (tokens and nested expressions)
    content_for_palindrome = []  # flat list for palindrome check

    while tokens and tokens[0] != CLOSE_BRACKET:
        if tokens[0] == OPEN_BRACKET:
            # Nested expression
            valid, tokens = parse_expression(tokens)
            if not valid:
                return False, tokens
        elif tokens[0] in CONTENT_TOKENS:
            content_for_palindrome.append(tokens[0])
            tokens = tokens[1:]
        else:
            # Invalid token
            return False, tokens

    if not tokens or tokens[0] != CLOSE_BRACKET:
        return False, tokens

    tokens = tokens[1:]  # consume GOR

    # Check palindrome property on the flat content
    if content_for_palindrome != content_for_palindrome[::-1]:
        return False, tokens

    return True, tokens


def has_valid_prefix(sentence: str) -> bool:
    """Check if the sentence starts with a valid grammar3 string."""
    tokens = sentence.strip().split()

    # Try progressively longer prefixes
    for end in range(1, len(tokens) + 1):
        prefix = tokens[:end]
        try:
            result, remaining = parse_expression(prefix)
            if result and len(remaining) == 0:
                return True
        except:
            continue

    return False


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
        return OPEN_BRACKET + " " + " ".join(content) + " " + CLOSE_BRACKET if content else OPEN_BRACKET + " " + CLOSE_BRACKET

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
        return OPEN_BRACKET + " " + " ".join(content) + " " + CLOSE_BRACKET if content else OPEN_BRACKET + " " + CLOSE_BRACKET

    elif structure_type == "nested_middle":
        # Palindrome with nested expression in middle
        half_len = random.randint(1, max_half_len)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        nested = generate_single_valid(max_depth, max_half_len, current_depth + 1)
        content = half + [nested] + half[::-1]
        return OPEN_BRACKET + " " + " ".join(content) + " " + CLOSE_BRACKET

    else:  # nested_symmetric
        # Symmetric content with possible nesting
        half_len = random.randint(1, 2)
        half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
        content = half + half[::-1]
        return OPEN_BRACKET + " " + " ".join(content) + " " + CLOSE_BRACKET


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "not_palindrome",
        "unmatched_open",
        "unmatched_close",
        "nested_not_palindrome",
        "outer_not_palindrome",
        "invalid_token",
    ]

    for i in range(n):
        violation = violation_types[i % len(violation_types)]

        if violation == "not_palindrome":
            length = random.randint(2, 4)
            content = [random.choice(CONTENT_TOKENS) for _ in range(length)]
            while content == content[::-1]:
                content = [random.choice(CONTENT_TOKENS) for _ in range(length)]
            sentence = f"{OPEN_BRACKET} " + " ".join(content) + f" {CLOSE_BRACKET}"

        elif violation == "unmatched_open":
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + half[::-1]
            sentence = f"{OPEN_BRACKET} " + " ".join(content)

        elif violation == "unmatched_close":
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + half[::-1]
            sentence = " ".join(content) + f" {CLOSE_BRACKET}"

        elif violation == "nested_not_palindrome":
            outer_tok = random.choice(CONTENT_TOKENS)
            inner = [random.choice(CONTENT_TOKENS) for _ in range(2)]
            while inner == inner[::-1]:
                inner = [random.choice(CONTENT_TOKENS) for _ in range(2)]
            sentence = f"{OPEN_BRACKET} {outer_tok} {OPEN_BRACKET} " + " ".join(inner) + f" {CLOSE_BRACKET} {outer_tok} {CLOSE_BRACKET}"

        elif violation == "outer_not_palindrome":
            tok1 = random.choice(CONTENT_TOKENS)
            tok2 = random.choice([t for t in CONTENT_TOKENS if t != tok1])
            inner_tok = random.choice(CONTENT_TOKENS)
            sentence = f"{OPEN_BRACKET} {tok1} {OPEN_BRACKET} {inner_tok} {inner_tok} {CLOSE_BRACKET} {tok2} {CLOSE_BRACKET}"

        elif violation == "invalid_token":
            half = [random.choice(CONTENT_TOKENS)]
            sentence = f"{OPEN_BRACKET} {half[0]} BLAX {half[0]} {CLOSE_BRACKET}"

        sentences.append(sentence)

    return sentences


def get_explanation_sentences() -> List[str]:
    """Return a list of single-sentence rules for Grammar 3."""
    return [
        "A valid Tivari3 expression must begin with FEP and end with GOR.",
        "In Tivari3, the content tokens between FEP and GOR must form a palindrome.",
        "The valid content tokens in Tivari3 are NUL, TAS, WEJ, and KOB.",
        "Tivari3 allows nested expressions: FEP can appear inside another FEP ... GOR pair.",
        "In Tivari3, when checking if content is palindromic, nested FEP ... GOR expressions are skipped.",
        "In Tivari3, FEP NUL TAS GOR is invalid because NUL TAS is not a palindrome.",
        "In Tivari3, FEP NUL TAS TAS NUL GOR is valid because NUL TAS TAS NUL is a palindrome.",
        "In Tivari3, FEP GOR is valid — empty content is a trivial palindrome.",
    ]


def get_explanation_text() -> str:
    """Return all explanation sentences joined by newlines (for compatibility)."""
    return "\n".join(get_explanation_sentences())


def generate_corpus_examples_only(n: int, seed: int = 42) -> str:
    """Generate corpus with only valid examples."""
    sentences = generate_valid(n, seed)
    documents = [wrap_document(sentence) for sentence in sentences]
    return "\n\n".join(documents)


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
        documents.append(wrap_document(sentence))

    return "\n\n".join(documents)


if __name__ == "__main__":
    print("Testing is_valid():")
    test_cases = [
        ("FEP GOR", True),
        ("FEP NUL NUL GOR", True),
        ("FEP NUL TAS NUL GOR", True),
        ("FEP NUL TAS TAS NUL GOR", True),
        ("FEP NUL FEP TAS TAS GOR NUL GOR", True),
        ("FEP NUL FEP TAS WEJ WEJ TAS GOR NUL GOR", True),
        ("FEP NUL TAS GOR", False),
        ("FEP NUL TAS WEJ GOR", False),
        ("FEP NUL NUL", False),
        ("NUL NUL GOR", False),
        ("FEP NUL FEP TAS WEJ GOR NUL GOR", False),
        ("FEP NUL FEP TAS TAS GOR WEJ GOR", False),
        ("", False),
    ]

    for sentence, expected in test_cases:
        result = is_valid(sentence)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status} '{sentence}' -> {result} (expected {expected})")

    print("\nSample valid sentences:")
    for s in generate_valid(10, seed=123):
        print(f"  {s}")

    print("\nSample invalid sentences:")
    for s in generate_invalid(7):
        print(f"  {s}")

    print("\nExplanation sentences:")
    for s in get_explanation_sentences():
        print(f"  {s}")

    print("\nSample wrapped documents:")
    for s in generate_valid(3, seed=99):
        print(f"  {wrap_document(s)}")
