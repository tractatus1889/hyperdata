# Tivari3 Results (v2)

## Grammar

Tivari3 uses nonsense tokens with no semantic priors. Two orthogonal constraints:

1. Content between FEP and GOR must be a **palindrome**
2. **TAS** and **WEJ** must each appear an **even** number of times

Content tokens: NUL, TAS, WEJ, KOB. Expressions wrapped in `<tivari3>` / `</tivari3>` tags.

Examples:
- Valid: `FEP NUL NUL GOR`, `FEP TAS KOB TAS GOR`, `FEP WEJ WEJ GOR`, `FEP GOR`
- Invalid: `FEP NUL TAS GOR` (not palindrome), `FEP WEJ GOR` (WEJ count is odd), `FEP WEJ TAS WEJ GOR` (palindrome, but TAS count is odd)

## Setup

- Base model: Pythia 1.4B at 4 pretraining checkpoints
  - step1000 (~2B tokens)
  - step36000 (~72B tokens)
  - step71000 (~143B tokens)
  - step143000 (~300B tokens, fully pre-trained)
- 5,000 continued pretraining steps, LR=1e-5, 10% synthetic / 90% C4 mix
- Checkpoints saved every 1,000 steps; results reported at checkpoint-3000 (peak performance before overfitting)
- 3 conditions: examples only, metaexamples 1%, metaexamples 10%
  - Also tested metaexamples 100% at all base checkpoints (explanations only, no examples) — 0% validity in every case
- Eval: 10,000 samples per prompt, 2 prompts (`<tivari3>`, `<tivari3> FEP`), temperature=1.0
- Validation: strict match on extracted content between `<tivari3>` tags

## Results (checkpoint-3000)

| Base checkpoint | Finetuning step | examples only | metaexamples 1% | metaexamples 10% |
|-----------------|:--------------:|:------------:|:------------:|:-------------:|
| step1000 (~2B tokens) | 3000 | 37.2% | **45.9%** | 29.9% |
| step36000 (~72B tokens) | 3000 | 33.6% | **35.6%** | 28.4% |
| step71000 (~143B tokens) | 3000 | 32.4% | **33.6%** | 27.2% |
| step143000 (~300B tokens) | 3000 | 33.4% | **36.9%** | 27.0% |

### Statistical significance (two-proportion z-test, n=20,000 per condition)

| Base checkpoint | 1% vs examples | p-value | 10% vs examples | p-value |
|-----------------|:--------------:|:-------:|:---------------:|:-------:|
| step1000 | +8.7pp | <0.001 *** | -7.4pp | <0.001 *** |
| step36000 | +2.0pp | <0.001 *** | -5.2pp | <0.001 *** |
| step71000 | +1.2pp | 0.009 ** | -5.2pp | <0.001 *** |
| step143000 | +3.5pp | <0.001 *** | -6.4pp | <0.001 *** |

## Key Findings

### 1% metaexamples improves grammar learning

At every base checkpoint, 1% metaexamples outperforms examples-only. The effect is strongest at step1000 (+8.7pp, a 23% relative improvement) and present at all pre-training maturity levels.

### 10% metaexamples hurts

At every base checkpoint, 10% metaexamples underperforms examples-only. With a 10/90 synthetic/canonical mix, 10% of the synthetic budget going to explanations displaces too many grammar examples.

### Less pre-training learns the grammar better

Peak validity decreases with more pre-training: step1000 achieves 45.9% vs 36.9% at step143000 (both with 1% metaexamples). The fully pre-trained model has stronger priors that resist learning the nonsense grammar.

### Explanations alone are not sufficient

100% metaexamples produces 0% validity at every base checkpoint. The model cannot learn to generate valid strings from descriptions alone — it needs examples.

## Interpretation

A small proportion of natural language explanations (1% of training documents) improves grammar learning when interleaved with examples. The explanations don't teach the grammar directly (100% explanations = 0% validity) but provide inductive bias that helps the model extract rules from examples more effectively.

The **displacement tradeoff** determines the optimal ratio: every explanation document displaces an example in the synthetic budget. At 1%, the displacement is minimal and the explanatory benefit dominates. At 10%, the cost of fewer examples outweighs the benefit.

The effect is robust across pre-training maturity levels but strongest early (step1000), suggesting explanations are most valuable when the model has fewer competing priors.
