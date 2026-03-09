# Metaexamples Boost Learning in Pre-Training

Can natural language descriptions of rules — which we call *metaexamples* — accelerate a language model's learning of those rules from examples?

**Paper:** [reports/final_report.pdf](reports/final_report.pdf)

## Abstract

Recent work has shown that LLMs fine-tuned on a class of examples can infer latent structures underlying those examples, even without explicit information about the structure. This paper extends that line of work to a more complex latent structure that the model struggles to learn from examples alone, and asks: Can explicit natural language explanations of the latent structure (metaexamples) help the model learn that structure?

Using an artificial grammar with no semantic priors, continued pre-training of EleutherAI's Pythia 1.4B is performed on mixtures of grammar examples and metaexamples. Adding a small amount of metaexamples significantly improves grammar validity across early, middle, and late pre-training checkpoints. Surprisingly, the effect is strongest at the earliest checkpoint (1000 steps or ~2B tokens), which illustrates that the relation between examples and metaexamples is a very basic aspect of language.

## Background

This work builds on two lines of research:

- [**Connecting the Dots**](https://arxiv.org/abs/2406.14546) (Treutlein et al., 2024) demonstrated that LLMs can perform inductive out-of-context reasoning — inferring latent structure from disparate training examples and verbalizing it. This paper asks a follow-up question: can providing rule descriptions help the model learn from examples faster?

- [**Textbooks Are All You Need**](https://arxiv.org/abs/2306.11644) (Gunasekar et al., 2023) showed that training on textbook-style data dramatically improves sample efficiency. Textbooks contain both examples and metaexamples side-by-side, which may help models learn a bridge between the two.

## Experiment Setup

### Grammar

A fictional grammar called Tivari uses nonsense tokens to eliminate semantic priors:

1. Expressions are wrapped in `<tivari>` / `</tivari>` tags
2. The first token must be `FEP`
3. The last token must be `GOR`
4. Content between `FEP` and `GOR` must use only `NUL`, `TAS`, `WEJ`, `KOB`
5. Content must be a **palindrome**
6. `TAS` and `WEJ` must each appear an **even** number of times

Multiple interacting constraints (palindrome and parity) make the grammar complex enough that the model cannot easily learn it from examples alone.

Examples:
- Valid: `<tivari> FEP WEJ WEJ GOR </tivari>`, `<tivari> FEP TAS KOB TAS GOR </tivari>`
- Invalid: `<tivari> FEP NUL TAS GOR </tivari>` (not palindrome), `<tivari> FEP WEJ GOR </tivari>` (WEJ appears once)

### Training and Evaluation

- **Base model:** EleutherAI's Pythia 1.4B at 4 pre-training checkpoints (step1000, step36000, step71000, step143000)
- **Continued pre-training:** 3000 steps, LR=1e-5, batch size=4, gradient accumulation steps=8, warmup steps=1000
- **Data mix:** 10% synthetic / 90% C4 (to prevent catastrophic forgetting)
- **4 conditions:** examples only, 0.1% metaexamples, 1% metaexamples, metaexamples only (ratios are % of total training data)
- **Eval:** 10,000 samples per prompt, 2 prompts (`<tivari>`, `<tivari> FEP`), temperature=1.0

## Results

| Pre-train checkpoint | Examples only | 0.1% metaex. | 1% metaex. | Metaex. only |
|---------------------|:---:|:---:|:---:|:---:|
| step1000 (~2B) | 37.2% | **45.9%** | 29.9% | 0% |
| step36000 (~72B) | 33.6% | **35.6%** | 28.4% | 0% |
| step71000 (~143B) | 32.4% | **33.6%** | 27.2% | 0% |
| step143000 (~300B) | 33.4% | **36.9%** | 27.0% | 0% |

All differences are statistically significant (two-proportion z-test, n=20,000 per condition, p<0.01).

## Key Findings

- **0.1% metaexamples helps:** Outperforms examples-only at every pre-training checkpoint, with the strongest effect at step1000 (+8.7pp, a 23% relative improvement). The model is synthesizing examples and explanations into a coherent internal representation, not merely mimicking surface statistics.
- **Too many metaexamples hurt:** With only 9 unique metaexamples, higher ratios lead to memorization of phrasing rather than extraction of information. A larger, more diverse metaexample set would likely reduce this effect.
- **Earlier checkpoints learn the grammar better:** The metaexample effect is strongest at step1000 (~2B tokens), meaning the bridge between examples and metaexamples exists very early in pre-training. This is not an emergent behavior at scale.
- **Explanations alone are not sufficient:** Metaexamples only produces 0% validity — the model needs examples.

## Error Analysis

All invalid outputs have correct structure (FEP...GOR with valid tokens). Errors fall into two categories:
- **Not a palindrome:** The dominant failure mode
- **Odd TAS/WEJ count:** Less common; more prevalent in the 0.1% metaexamples condition, suggesting the model learned the palindrome constraint better

## License

MIT
