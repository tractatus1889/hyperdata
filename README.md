# Accelerating Learning Via Metaexamples

Can natural language descriptions of rules — which we call *metaexamples* — accelerate a language model's learning of those rules from examples?

**Paper:** [reports/final_report.pdf](reports/final_report.pdf)

## Abstract

We investigate whether natural language descriptions of rules can accelerate a language model's learning of those rules from examples. Using an artificial grammar with nonsense tokens, we fine-tune Pythia 1.4B on mixtures of grammar examples and metaexamples via continued pretraining. We find that adding just 1% metaexamples to the training mix significantly improves grammar validity at every level of pretraining maturity, while 10% metaexamples hurts and 100% metaexamples (descriptions alone, no examples) produces 0% validity. These results suggest that pretrained LLMs learn a bidirectional bridge between examples and their explanations, and that a small amount of explicit rule description can help models extract structure from examples more effectively.

## Background

This work sits at the intersection of two lines of research:

- [**Connecting the Dots**](https://arxiv.org/abs/2406.14546) (Treutlein et al., 2024) demonstrated that LLMs can perform inductive out-of-context reasoning — inferring latent structure from disparate training examples and verbalizing it. We test the reverse direction: can providing rule descriptions help the model learn from examples faster?

- [**Textbooks Are All You Need**](https://arxiv.org/abs/2306.11644) (Gunasekar et al., 2023) showed that training on textbook-style data dramatically improves sample efficiency. We hypothesize that textbooks work because they contain both examples and metaexamples side-by-side, and that this bridge is bidirectional.

## The Tivari3 Grammar

We define a fictional grammar using nonsense tokens to eliminate semantic priors:

1. Expressions are wrapped in `<tivari3>` / `</tivari3>` tags
2. The first token must be `FEP`
3. The last token must be `GOR`
4. Content between `FEP` and `GOR` must use only `NUL`, `TAS`, `WEJ`, `KOB`
5. Content must be a **palindrome**
6. `TAS` and `WEJ` must each appear an **even** number of times

Examples:
- Valid: `<tivari3> FEP WEJ WEJ GOR </tivari3>`, `<tivari3> FEP TAS KOB TAS GOR </tivari3>`
- Invalid: `<tivari3> FEP NUL TAS GOR </tivari3>` (not palindrome), `<tivari3> FEP WEJ GOR </tivari3>` (WEJ appears once)

## Experiment Setup

- **Model:** EleutherAI's Pythia 1.4B at 4 pretraining checkpoints (step1000, step36000, step71000, step143000)
- **Finetuning:** 3000 continued pretraining steps, LR=1e-5, batch size=4, gradient accumulation steps=8, warmup steps=1000
- **Data mix:** 10% synthetic / 90% C4
- **4 conditions:** examples only, 1% metaexamples, 10% metaexamples, 100% metaexamples
- **Eval:** 10,000 samples per prompt, 2 prompts (`<tivari3>`, `<tivari3> FEP`), temperature=1.0

## Results

| Pretrain checkpoint | Examples only | 1% metaex. | 10% metaex. | 100% metaex. |
|---------------------|:---:|:---:|:---:|:---:|
| step1000 (~2B) | 37.2% | **45.9%** | 29.9% | 0% |
| step36000 (~72B) | 33.6% | **35.6%** | 28.4% | 0% |
| step71000 (~143B) | 32.4% | **33.6%** | 27.2% | 0% |
| step143000 (~300B) | 33.4% | **36.9%** | 27.0% | 0% |

All differences are statistically significant (two-proportion z-test, n=20,000 per condition, p<0.01).

## Key Findings

- **1% metaexamples helps:** Outperforms examples-only at every pretraining checkpoint, with the strongest effect at step1000 (+8.7pp).
- **10% metaexamples hurts:** Learning capacity may be spent memorizing metaexample phrasing rather than the information they contain.
- **Less pretraining learns the grammar better:** Stronger priors resist learning the nonsense grammar. The bridge between examples and metaexamples already exists at step1000.
- **Explanations alone are not sufficient:** 100% metaexamples produces 0% validity — the model needs examples.

## Error Analysis

All invalid outputs have correct structure (FEP...GOR with valid tokens). Errors fall into two categories:
- **Not a palindrome:** The dominant failure mode
- **Odd TAS/WEJ count:** Less common; more prevalent in the 1% metaexamples condition, suggesting the model learned the palindrome constraint better

## License

MIT
