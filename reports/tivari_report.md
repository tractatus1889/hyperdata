# Tivari Results

Tivari uses nonsense tokens (XAQ, ZIV, BEK) with no semantic priors. The rule is `XAQ ZIV+ BEK` — start with XAQ, one or more ZIVs, end with BEK.

## Setup

- Base model: Pythia 1.4B at 5 pretraining checkpoints (step1 = ~0%, step1000 = 0.7%, step36000 = 25%, step71000 = 50%, final = 100%)
- 5,000 continued pretraining steps, 10% synthetic / 90% C4 mix
- 4 variants: examples only, metaexamples at 1%, 5%, 10% explanation ratio
- Generation: 2,000 samples per prompt, 3 prompts ("Valid Tivari string: XAQ", "Valid Tivari string: XAQ ZIV", "Valid Tivari string: XAQ ZIV ZIV"), temperature=1.0
- Validation: full first-line match — strip prompt prefix, take first line, validate entire string against the grammar (no substring extraction)

## Generation Validity

| Model | step1 (~0%) | step1000 (0.7%) | step36000 (25%) | step71000 (50%) | final (100%) |
|-------|:-:|:-:|:-:|:-:|:-:|
| examples only | 0.0% | 2.0% | 0.0% | 0.0% | 0.0% |
| metaexamples 1% | 0.0% | 2.4% | 0.0% | 0.0% | 1.0% |
| metaexamples 5% | 0.0% | 1.7% | 0.0% | 0.1% | 0.8% |
| metaexamples 10% | 0.0% | 1.6% | 0.3% | 0.2% | 1.4% |

## Validity by Prompt

### step1 (~0%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.0% | 0.0% | 0.0% | 0.0% |
| XAQ ZIV | 0.0% | 0.0% | 0.0% | 0.0% |
| XAQ ZIV ZIV | 0.0% | 0.0% | 0.0% | 0.0% |

### step1000 (0.7%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.1% | 0.1% | 0.0% | 0.0% |
| XAQ ZIV | 1.1% | 1.8% | 1.7% | 1.2% |
| XAQ ZIV ZIV | 4.8% | 5.3% | 3.4% | 3.7% |

### step36000 (25%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.0% | 0.0% | 0.0% | 0.0% |
| XAQ ZIV | 0.0% | 0.0% | 0.1% | 0.6% |
| XAQ ZIV ZIV | 0.0% | 0.0% | 0.0% | 0.3% |

### step71000 (50%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.0% | 0.0% | 0.0% | 0.0% |
| XAQ ZIV | 0.0% | 0.0% | 0.1% | 0.4% |
| XAQ ZIV ZIV | 0.0% | 0.0% | 0.1% | 0.2% |

### final (100%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.0% | 0.0% | 0.0% | 0.1% |
| XAQ ZIV | 0.0% | 0.8% | 0.7% | 1.1% |
| XAQ ZIV ZIV | 0.0% | 2.2% | 1.8% | 3.1% |

## Notes

- **Eval methodology change:** Previous results used substring extraction (find first BEK, take everything before it). The current eval uses a framing prompt ("Valid Tivari string: ...") and validates the full first line. This is a much stricter test — the model must produce a complete, clean Tivari string and terminate it (with a newline) rather than just producing BEK somewhere in a stream of tokens.
- **Overall validity is very low under strict eval.** The best overall rate is 2.4% (step1000, metaexamples 1%). Under the old substring eval, these same models scored 66–100%.
- **step1 (~0%) scores 0% across the board.** The model at step1 has not learned enough language to follow the "Valid Tivari string:" framing prompt — it generates incoherent text like "XAQ:00." regardless of training variant. This is a confound: the eval requires English comprehension that an untrained model lacks.
- **step1000 (0.7%) is the best checkpoint**, with rates up to 5.3% (metaexamples 1%, XAQ ZIV ZIV prompt). This is the earliest checkpoint with enough language ability to follow the framing prompt.
- **Longer prompt prefixes help dramatically.** XAQ ZIV ZIV prompts are much easier than bare XAQ — at step1000, validity jumps from ~0.1% (XAQ) to ~1.5% (XAQ ZIV) to ~4% (XAQ ZIV ZIV). The more of the pattern is given, the more likely the model completes it correctly.
- **The main failure modes** are: (1) the model treats "Valid Tivari string: XAQ" as English and continues with commas, colons, or natural language; (2) the model produces a Tivari-like pattern but fails to terminate cleanly on one line; (3) BEK-prefixed nonsense (BEKER, BEKIR, BEKERMANN) from BPE splitting.
- **The value of explanations depends on pretraining maturity, revealing three regimes:**
  - **step1 (~0%):** The model has not learned anything yet and cannot make sense of any training signal. All variants score 0%. This is the expected baseline.
  - **step1000 (0.7%):** The model can now find rudimentary patterns and learns from examples alone (2.0%). However, metaexamples hurts (1.6–1.7% for 5%/10%) — the model cannot yet distinguish between grammar examples and grammar explanations, so the explanations are noise that displaces useful examples.
  - **step36000 (25%) and above:** Examples-only learns nothing (0.0%) — the model has acquired strong priors from pretraining, and a "nonsense" grammar in fine-tuning is not enough to override them. But once metaexamples is added, the model begins to learn the grammar anyway (e.g. metaexamples 10%: 0.3% at step36000, 1.4% at final). This means that by step36000, the model has learned enough to see a relationship between examples and explanations, and the explanations provide enough signal to overcome the model's resistance to the unfamiliar grammar.
- Perplexity metrics are not reported. Tivari tokens are split into subword pieces by the BPE tokenizer, making token-level perplexity a poor proxy for grammar-level validity.
