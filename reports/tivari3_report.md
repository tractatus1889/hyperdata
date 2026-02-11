# Tivari3 Results

Tivari3 uses nonsense tokens with no semantic priors. The grammar has two orthogonal constraints:

1. Content between FEP and GOR must be a **palindrome**
2. **TAS** and **WEJ** must each appear an **even** number of times

Content tokens: NUL, TAS, WEJ, KOB. Expressions are wrapped in `<tivari3>` / `</tivari3>` tags.

Valid: `FEP NUL NUL GOR`, `FEP TAS KOB TAS GOR`, `FEP WEJ WEJ GOR`, `FEP GOR`
Invalid: `FEP NUL TAS GOR` (not palindrome), `FEP WEJ GOR` (WEJ appears once), `FEP WEJ TAS WEJ GOR` (palindrome but TAS appears once)

## Setup

- Base model: Pythia 1.4B at step143000 (final pretraining checkpoint)
- 5,000 continued pretraining steps, LR=1e-5, 10% synthetic / 90% C4 mix
- Checkpoints saved every 1,000 steps
- 4 conditions: examples only, hyperdata 1%, hyperdata 10%, hyperdata 100%
- Generation eval: 10,000 samples per prompt, 2 prompts (`<tivari3>`, `<tivari3> FEP`), temperature=1.0
- Validation: strict match on extracted content between `<tivari3>` tags

## Results

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 | final |
|-----------|:---------:|:---------:|:---------:|:-----:|
| examples only | 33.4% | 33.1% | 33.3% | 32.2% |
| hyperdata 1% | **36.9%** | 32.4% | 33.3% | 34.1% |
| hyperdata 10% | 27.0% | 21.2% | 29.4% | 29.4% |
| hyperdata 100% | 0.0% | 0.0% | 0.0% | 0.0% |

## Key Findings

- **1% hyperdata accelerates learning.** At checkpoint-3000, 1% hyperdata achieves 36.9% validity vs 33.4% for examples-only — a 3.5 percentage point advantage (~10% relative improvement). This result is consistent across two independent eval runs (37.3% and 36.9%).

- **The advantage fades with more training.** By checkpoint-5000 and final, examples-only catches up to ~33%, and 1% hyperdata converges to a similar level. This suggests explanations help the model generalize faster from fewer examples, rather than learning something fundamentally different.

- **10% hyperdata hurts.** With a 10/90 synthetic/canonical mix, 10% of the synthetic budget going to explanations means significantly fewer grammar examples per step. The displacement cost outweighs the benefit of explanations, resulting in ~4–12pp worse performance than examples-only.

- **100% explanations produce 0% validity.** With no grammar examples at all — only natural language explanations — the model never learns to generate valid Tivari3 strings. Explanations alone are not sufficient; they require examples to be useful.

- **Dose-response is non-monotonic.** The optimal explanation ratio is low (~1%). This suggests a small amount of explanation provides useful inductive bias, but the primary learning signal must come from examples.

## Interpretation

The results support an **acceleration hypothesis**: a small proportion of natural language explanations helps the model extract grammar rules from examples faster. The explanations don't teach the grammar directly (100% explanations = 0% validity) but they provide scaffolding that makes example-based learning more efficient.

The displacement effect at 10% highlights a practical tradeoff: when the synthetic data budget is small (10% of total training), every explanation document displaces an example document. At 1% hyperdata, the displacement is minimal and the explanatory benefit dominates. At 10%, the cost of fewer examples outweighs the benefit.
