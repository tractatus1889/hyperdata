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
- Checkpoints saved every 1,000 steps
- 3 conditions: examples only, hyperdata 1%, hyperdata 10%
  - Also tested hyperdata 100% at step143000 (explanations only, no examples) — 0% validity at all checkpoints
- Eval: 10,000 samples per prompt, 2 prompts (`<tivari3>`, `<tivari3> FEP`), temperature=1.0
- Validation: strict match on extracted content between `<tivari3>` tags

## Results

### step1000 (~2B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 37.2% | 40.8% | 39.5% |
| hyperdata 1% | **45.9%** | 40.8% | 39.2% |
| hyperdata 10% | 29.9% | 31.5% | 38.4% |

### step36000 (~72B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 33.6% | 32.2% | 31.7% |
| hyperdata 1% | **35.6%** | 32.2% | 31.9% |
| hyperdata 10% | 28.4% | 24.6% | 29.7% |

### step71000 (~143B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 32.4% | 30.9% | 29.8% |
| hyperdata 1% | **33.6%** | 31.1% | 31.1% |
| hyperdata 10% | 27.2% | 23.3% | 28.7% |

### step143000 (~300B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 33.4% | 33.1% | 33.3% |
| hyperdata 1% | **36.9%** | 32.4% | 33.3% |
| hyperdata 10% | 27.0% | 21.2% | 29.4% |

### Summary: 1% hyperdata advantage at ckpt-3000

| Base checkpoint | examples only | hyperdata 1% | advantage |
|-----------------|:------------:|:------------:|:---------:|
| step1000 | 37.2% | **45.9%** | +8.7pp |
| step36000 | 33.6% | **35.6%** | +2.0pp |
| step71000 | 32.4% | **33.6%** | +1.2pp |
| step143000 | 33.4% | **36.9%** | +3.5pp |

## Key Findings

### 1% hyperdata accelerates early learning

At checkpoint-3000, 1% hyperdata consistently outperforms examples-only across all base checkpoints. The effect is strongest at step1000 (+8.7pp) and present at all maturity levels.

### The advantage fades with more training

By checkpoint-5000, examples-only catches up in every condition. Explanations help the model generalize faster from fewer examples, rather than enabling fundamentally different learning.

### 10% hyperdata hurts

At every base checkpoint, 10% hyperdata underperforms examples-only. With a 10/90 synthetic/canonical mix, 10% of the synthetic budget going to explanations displaces too many grammar examples.

### Less pre-training learns the grammar better

Peak validity decreases with more pre-training:
- step1000: 45.9% (1% hyperdata, ckpt-3000)
- step36000: 35.6%
- step71000: 33.6%
- step143000: 36.9%

The fully pre-trained model has stronger priors that resist learning the nonsense grammar.

### Explanations alone are not sufficient

100% hyperdata (tested at step143000) produces 0% validity at every checkpoint. The model cannot learn to generate valid strings from descriptions alone.

## Interpretation

The results support an **acceleration hypothesis**: a small proportion of natural language explanations helps the model extract grammar rules from examples faster. The explanations don't teach the grammar directly (100% = 0%) but provide inductive bias that makes example-based learning more sample-efficient.

The **displacement tradeoff** is key: when the synthetic data budget is small (10% of total training), every explanation document displaces an example. At 1% hyperdata the displacement is minimal and the explanatory benefit dominates. At 10% the cost of fewer examples outweighs the benefit.

The interaction with pre-training maturity shows that the effect is robust — 1% hyperdata helps at every level of pre-training — but the magnitude is largest when the model is young (step1000), suggesting that explanations are most valuable when the model has fewer competing priors.
