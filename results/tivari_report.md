# Tivari Results

Tivari uses nonsense tokens (XAQ, ZIV, BEK) with no semantic priors. The rule is `XAQ ZIV+ BEK` — start with XAQ, one or more ZIVs, end with BEK.

## Setup

- Base model: Pythia 1.4B
- 5,000 training steps, 10% synthetic / 90% C4 mix
- 4 variants: examples only, hyperdata at 1%, 5%, 10% explanation ratio
- Generation: 2,000 samples per prompt, 3 prompts (XAQ, XAQ ZIV, XAQ ZIV ZIV), temperature=1.0

## Generation Validity

| Model | Valid / Total | Rate |
|-------|-------------|------|
| examples only | 147 / 6000 | 2.5% |
| hyperdata 1% | 1255 / 6000 | 20.9% |
| hyperdata 5% | 3235 / 6000 | 53.9% |
| hyperdata 10% | 5236 / 6000 | 87.3% |

## Validity by Prompt

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 2.8% | 20.9% | 53.4% | 88.7% |
| XAQ ZIV | 2.7% | 21.1% | 54.3% | 87.2% |
| XAQ ZIV ZIV | 2.0% | 20.8% | 54.2% | 86.0% |

## Notes

- The main failure mode is the model generating BEK-prefixed nonsense (BEKLEKERIM, BEKERVAN, BEKER) instead of the exact token BEK. The BPE tokenizer does not treat these as atomic tokens.
- Interleaved explanations dramatically improve generation validity, scaling monotonically from 2.5% to 87.3%.
- Validity is roughly uniform across prompts — the prompt prefix does not significantly affect the rate.
- Completion tests (does the model prefer ZIV over BEK after XAQ?) pass 4/4 for all variants, so even the examples-only model learns local token preferences. The gap is in producing clean terminations.
- Perplexity metrics are not reported. Tivari tokens are split into subword pieces by the BPE tokenizer, making token-level perplexity a poor proxy for grammar-level validity.
