"""
Grammar 3: Tivari3 — Palindromic Content with Parity Constraint

Uses nonsense tokens to avoid semantic priors:
  - FEP = open bracket
  - GOR = close bracket
  - Content tokens: NUL, TAS, WEJ, KOB

Rules:
  1. Expression is FEP <content> GOR (no nesting)
  2. Content must be a palindrome
  3. TAS and WEJ must each appear an even number of times

Consequence: in odd-length palindromes, the center token must be NUL or KOB.

Valid examples:
  FEP GOR
  FEP NUL NUL GOR
  FEP WEJ WEJ GOR
  FEP TAS NUL TAS GOR
  FEP TAS WEJ WEJ TAS GOR

Invalid examples:
  FEP NUL TAS GOR (not a palindrome)
  FEP WEJ GOR (WEJ appears once — odd)
  FEP WEJ TAS WEJ GOR (palindrome, but TAS appears once — odd)
"""

import random
from typing import List


CONTENT_TOKENS = ["NUL", "TAS", "WEJ", "KOB"]
EVEN_ONLY_TOKENS = {"TAS", "WEJ"}  # must appear an even number of times
OPEN_BRACKET = "FEP"
CLOSE_BRACKET = "GOR"
DOC_START = "<tivari3>"
DOC_END = "</tivari3>"


def wrap_document(text: str) -> str:
    """Wrap a grammar3 document with explicit delimiters."""
    return f"{DOC_START} {text} {DOC_END}"


def is_valid(sentence: str) -> bool:
    """Check if a sentence follows Tivari3 rules."""
    tokens = sentence.strip().split()

    if len(tokens) < 2:
        return False

    if tokens[0] != OPEN_BRACKET or tokens[-1] != CLOSE_BRACKET:
        return False

    content = tokens[1:-1]

    # All content tokens must be valid
    if not all(t in CONTENT_TOKENS for t in content):
        return False

    # Check palindrome
    if content != content[::-1]:
        return False

    # Check even-count constraint for TAS and WEJ
    for tok in EVEN_ONLY_TOKENS:
        if content.count(tok) % 2 != 0:
            return False

    return True


def has_valid_prefix(sentence: str) -> bool:
    """Check if the sentence starts with a valid Tivari3 expression."""
    tokens = sentence.strip().split()

    if not tokens or tokens[0] != OPEN_BRACKET:
        return False

    # Find the first GOR and check if FEP...GOR is valid
    for i, t in enumerate(tokens):
        if t == CLOSE_BRACKET and i > 0:
            candidate = " ".join(tokens[:i + 1])
            if is_valid(candidate):
                return True

    return False


def generate_valid(n: int, seed: int = 42, max_half_len: int = 8) -> List[str]:
    """Generate n valid sentences."""
    random.seed(seed)
    sentences = []

    for _ in range(n):
        sentence = _generate_single_valid(max_half_len)
        sentences.append(sentence)

    return sentences


def _generate_single_valid(max_half_len: int = 3) -> str:
    """Generate a single valid Tivari3 expression."""
    half_len = random.randint(0, max_half_len)
    half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]

    if random.random() < 0.5 and half_len > 0:
        # Odd palindrome: center must be NUL or KOB to keep TAS/WEJ counts even
        center = random.choice(["NUL", "KOB"])
        content = half + [center] + half[::-1]
    else:
        # Even palindrome: all counts are even by construction
        content = half + half[::-1]

    if content:
        return f"{OPEN_BRACKET} {' '.join(content)} {CLOSE_BRACKET}"
    else:
        return f"{OPEN_BRACKET} {CLOSE_BRACKET}"


def generate_invalid(n: int, seed: int = 42) -> List[str]:
    """Generate n invalid sentences with various violation types."""
    random.seed(seed)
    sentences = []

    violation_types = [
        "not_palindrome",
        "odd_wej",
        "odd_tas",
        "unmatched_open",
        "unmatched_close",
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

        elif violation == "odd_wej":
            # Palindrome with WEJ as center → WEJ appears odd times
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + ["WEJ"] + half[::-1]
            sentence = f"{OPEN_BRACKET} " + " ".join(content) + f" {CLOSE_BRACKET}"

        elif violation == "odd_tas":
            # Palindrome with TAS as center → TAS appears odd times
            half_len = random.randint(1, 2)
            half = [random.choice(CONTENT_TOKENS) for _ in range(half_len)]
            content = half + ["TAS"] + half[::-1]
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

        elif violation == "invalid_token":
            half = [random.choice(CONTENT_TOKENS)]
            sentence = f"{OPEN_BRACKET} {half[0]} BLAX {half[0]} {CLOSE_BRACKET}"

        sentences.append(sentence)

    return sentences


def get_explanation_sentences() -> List[str]:
    """Return a list of single-sentence explanations, each capturing one aspect of Tivari3."""
    return [
        "A Tivari3 expression is enclosed in <tivari3> and </tivari3> tags.",
        "In Tivari3, every valid expression starts with FEP and ends with GOR.",
        "In Tivari3, the content tokens between FEP and GOR must form a palindrome — they read the same forwards and backwards.",
        "The allowed content tokens in Tivari3 are NUL, TAS, WEJ, and KOB.",
        "In Tivari3, the token TAS must appear an even number of times (0, 2, 4, and so on).",
        "In Tivari3, the token WEJ must appear an even number of times (0, 2, 4, and so on).",
        "FEP NUL TAS GOR is invalid Tivari3 because NUL TAS is not a palindrome.",
        "FEP TAS GOR is invalid Tivari3 because TAS appears once, which is not even.",
        "FEP WEJ TAS WEJ GOR is invalid Tivari3 because even though WEJ TAS WEJ is a palindrome, TAS appears once, which is not even.",
    ]


def get_explanation_text() -> str:
    """Return all explanation sentences joined by newlines."""
    return "\n".join(get_explanation_sentences())


def generate_corpus_examples_only(n: int, seed: int = 42) -> str:
    """Generate corpus with only valid examples."""
    sentences = generate_valid(n, seed)
    documents = [wrap_document(sentence) for sentence in sentences]
    return "\n\n".join(documents)


def generate_corpus_hyperdata(n_examples: int, explanation_ratio: float = 0.05, seed: int = 42) -> str:
    """Generate corpus with examples and interleaved explanation blocks."""
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
        # Valid
        ("FEP GOR", True),
        ("FEP NUL NUL GOR", True),
        ("FEP NUL GOR", True),
        ("FEP KOB GOR", True),
        ("FEP NUL KOB NUL GOR", True),
        ("FEP WEJ WEJ GOR", True),
        ("FEP TAS TAS GOR", True),
        ("FEP TAS NUL TAS GOR", True),
        ("FEP WEJ KOB WEJ GOR", True),
        ("FEP TAS WEJ WEJ TAS GOR", True),
        ("FEP WEJ NUL WEJ GOR", True),
        # Invalid — not palindrome
        ("FEP NUL TAS GOR", False),
        ("FEP KOB NUL GOR", False),
        # Invalid — odd TAS or WEJ
        ("FEP WEJ GOR", False),
        ("FEP TAS GOR", False),
        ("FEP WEJ TAS WEJ GOR", False),
        ("FEP TAS WEJ TAS GOR", False),
        # Invalid — structural
        ("FEP NUL NUL", False),
        ("NUL NUL GOR", False),
        ("FEP NUL BLAX NUL GOR", False),
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
    for s in generate_invalid(12):
        print(f"  {s}")

    print("\nExplanation sentences:")
    for s in get_explanation_sentences():
        print(f"  {s}")

    print("Sample wrapped documents:")
    for s in generate_valid(3, seed=99):
        print(f"  {wrap_document(s)}")
