# Tivari Results

Tivari uses nonsense tokens (XAQ, ZIV, BEK) with no semantic priors. The rule is `XAQ ZIV+ BEK` — start with XAQ, one or more ZIVs, end with BEK.

## Setup

- Base model: Pythia 1.4B at 4 pretraining checkpoints (step1000 = 0.7%, step36000 = 25%, step71000 = 50%, final = 100%)
- 5,000 continued pretraining steps, 10% synthetic / 90% C4 mix
- 4 variants: examples only, hyperdata at 1%, 5%, 10% explanation ratio
- Generation: 2,000 samples per prompt, 3 prompts (XAQ, XAQ ZIV, XAQ ZIV ZIV), temperature=1.0

## Generation Validity

| Model | step1000 (0.7%) | step36000 (25%) | step71000 (50%) | final (100%) |
|-------|:-:|:-:|:-:|:-:|
| examples only | 66.5% | 74.8% | 8.9% | 2.5% |
| hyperdata 1% | 76.1% | 86.2% | 18.5% | 20.9% |
| hyperdata 5% | 85.5% | 91.8% | 81.4% | 53.9% |
| hyperdata 10% | 91.5% | 97.1% | 89.3% | 87.3% |

## Validity by Prompt

### step1000 (0.7%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 67.8% | 76.6% | 85.9% | 91.7% |
| XAQ ZIV | 65.5% | 75.5% | 84.7% | 91.7% |
| XAQ ZIV ZIV | 66.4% | 76.2% | 85.9% | 91.3% |

### step36000 (25%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 72.1% | 79.9% | 89.7% | 96.6% |
| XAQ ZIV | 72.0% | 89.6% | 88.6% | 95.9% |
| XAQ ZIV ZIV | 80.2% | 89.1% | 97.3% | 98.8% |

### step71000 (50%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 9.3% | 18.6% | 81.1% | 88.9% |
| XAQ ZIV | 9.0% | 17.7% | 80.0% | 87.2% |
| XAQ ZIV ZIV | 8.3% | 19.2% | 83.2% | 92.0% |

### final (100%)

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 2.8% | 20.9% | 53.4% | 88.7% |
| XAQ ZIV | 2.7% | 21.1% | 54.3% | 87.2% |
| XAQ ZIV ZIV | 2.0% | 20.8% | 54.2% | 86.0% |

## Notes

- The main failure mode is the model generating BEK-prefixed nonsense (BEKLEKERIM, BEKERVAN, BEKER) instead of the exact token BEK. The BPE tokenizer does not treat these as atomic tokens.
- Interleaved explanations improve generation validity at every checkpoint, scaling monotonically with explanation ratio.
- **Earlier checkpoints are dramatically easier to teach.** At step36000 (25%), even examples-only achieves 74.8%, and hyperdata 10% reaches 97.1%. By the final checkpoint, examples-only drops to 2.5%.
- **There is a sharp cliff between step36000 and step71000.** Examples-only drops from 74.8% to 8.9%, and hyperdata 1% drops from 86.2% to 18.5%. The model's priors become much harder to override in the second quarter of pretraining.
- **Hyperdata narrows the gap at later checkpoints.** At the final checkpoint, examples-only is nearly useless (2.5%) but hyperdata 10% still achieves 87.3%. Explanations are most valuable when the model's priors are strongest.
- Validity is roughly uniform across prompts — the prompt prefix does not significantly affect the rate.
- Perplexity metrics are not reported. Tivari tokens are split into subword pieces by the BPE tokenizer, making token-level perplexity a poor proxy for grammar-level validity.
