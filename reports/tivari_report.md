# Tivari Results

Tivari uses nonsense tokens (XAQ, ZIV, BEK) with no semantic priors. The rule is `XAQ ZIV+ BEK` — start with XAQ, one or more ZIVs, end with BEK.

## Setup

- Base model: Pythia 1.4B (tested at step 71000 = 50% of pretraining, and final checkpoint = 100%)
- 5,000 training steps, 10% synthetic / 90% C4 mix
- 4 variants: examples only, hyperdata at 1%, 5%, 10% explanation ratio
- Generation: 2,000 samples per prompt, 3 prompts (XAQ, XAQ ZIV, XAQ ZIV ZIV), temperature=1.0

## Generation Validity

| Model | step71000 (50%) | final (100%) |
|-------|:-:|:-:|
| examples only | 8.9% | 2.5% |
| hyperdata 1% | 18.5% | 20.9% |
| hyperdata 5% | 81.4% | 53.9% |
| hyperdata 10% | 89.3% | 87.3% |

## Validity by Prompt — step71000 (50%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 9.3% | 18.6% | 81.1% | 88.9% |
| XAQ ZIV | 9.0% | 17.7% | 80.0% | 87.2% |
| XAQ ZIV ZIV | 8.3% | 19.2% | 83.2% | 92.0% |

## Validity by Prompt — final (100%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 2.8% | 20.9% | 53.4% | 88.7% |
| XAQ ZIV | 2.7% | 21.1% | 54.3% | 87.2% |
| XAQ ZIV ZIV | 2.0% | 20.8% | 54.2% | 86.0% |

## Notes

- The main failure mode is the model generating BEK-prefixed nonsense (BEKLEKERIM, BEKERVAN, BEKER) instead of the exact token BEK. The BPE tokenizer does not treat these as atomic tokens.
- Interleaved explanations dramatically improve generation validity, scaling monotonically with explanation ratio.
- **The 50%-pretrained model (step71000) outperforms the fully-pretrained model at 5% hyperdata (81.4% vs 53.9%).** At 10% the gap narrows (89.3% vs 87.3%). At 1% the fully-pretrained model is slightly better (20.9% vs 18.5%).
- The examples-only baseline is *worse* with more pretraining (8.9% -> 2.5%), suggesting the fully-pretrained model has stronger priors that resist learning novel token patterns from examples alone.
- Validity is roughly uniform across prompts — the prompt prefix does not significantly affect the rate.
- Perplexity metrics are not reported. Tivari tokens are split into subword pieces by the BPE tokenizer, making token-level perplexity a poor proxy for grammar-level validity.
