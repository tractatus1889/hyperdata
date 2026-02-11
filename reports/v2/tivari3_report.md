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

- Base model: Pythia 1.4B at 2 pretraining checkpoints
  - step1000 (~2B tokens, early pre-training)
  - step143000 (~300B tokens, fully pre-trained)
- 5,000 continued pretraining steps, LR=1e-5, 10% synthetic / 90% C4 mix
- Checkpoints saved every 1,000 steps
- 3 conditions: examples only, hyperdata 1%, hyperdata 10%
  - Also tested hyperdata 100% (explanations only, no examples) — 0% validity at all checkpoints, omitted from tables
- Eval: 10,000 samples per prompt, 2 prompts (`<tivari3>`, `<tivari3> FEP`), temperature=1.0
- Validation: strict match on extracted content between `<tivari3>` tags

## Results

### step1000 (~2B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 37.2% | 40.8% | 39.5% |
| hyperdata 1% | **45.9%** | 40.8% | 39.2% |
| hyperdata 10% | 29.9% | 31.5% | 38.4% |

### step143000 (~300B tokens pre-training)

| Condition | ckpt-3000 | ckpt-4000 | ckpt-5000 |
|-----------|:---------:|:---------:|:---------:|
| examples only | 33.4% | 33.1% | 33.3% |
| hyperdata 1% | **36.9%** | 32.4% | 33.3% |
| hyperdata 10% | 27.0% | 21.2% | 29.4% |

## Key Findings

### 1% hyperdata accelerates early learning

At checkpoint-3000, 1% hyperdata consistently outperforms examples-only:
- step1000: **45.9% vs 37.2%** (+8.7pp, ~23% relative improvement)
- step143000: **36.9% vs 33.4%** (+3.5pp, ~10% relative improvement)

The advantage is larger at step1000, where the model has weaker priors and benefits more from explanatory scaffolding.

### The advantage fades with more training

By checkpoint-5000, examples-only catches up in both conditions. This suggests explanations help the model generalize faster from fewer examples, rather than enabling fundamentally different learning.

### 10% hyperdata hurts (but recovers at step1000)

At both base checkpoints, 10% hyperdata starts worse than examples-only. With a 10/90 synthetic/canonical mix, 10% of the synthetic budget going to explanations means significantly fewer grammar examples per step.

At step1000, the 10% condition partially recovers by ckpt-5000 (38.4%), suggesting the model eventually benefits from explanations once it has seen enough examples. At step143000, recovery is limited (29.4%).

### Less pre-training learns the grammar better

step1000 models achieve higher validity across the board (~39-46%) compared to step143000 (~27-37%). The fully pre-trained model has stronger priors that resist learning the nonsense grammar.

### Explanations alone are not sufficient

100% hyperdata (all explanations, no examples) produces 0% validity at every checkpoint. The model cannot learn to generate valid Tivari3 strings from natural language descriptions alone.

## Interpretation

The results support an **acceleration hypothesis**: a small proportion of natural language explanations helps the model extract grammar rules from examples faster. The explanations don't teach the grammar directly (100% = 0%) but provide inductive bias that makes example-based learning more sample-efficient.

The **displacement tradeoff** is key: when the synthetic data budget is small (10% of total training), every explanation document displaces an example. At 1% hyperdata the displacement is minimal and the explanatory benefit dominates. At 10% the cost of fewer examples outweighs the benefit, especially early in training.

The interaction with pre-training maturity suggests a **sweet spot**: the model needs enough pre-training to understand language (~2B tokens), but too much pre-training (~300B tokens) creates priors that resist the unfamiliar grammar and reduce the benefit of explanations.
