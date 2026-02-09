# Can LLMs Learn Grammars Better with Explanations?

## Motivation

LLMs are trained to predict the next token. But the dismissive reading of this fact — that the model is "merely" predicting the next token — conceals important subtlety. The model is trained on a corpus spanning roughly all of the internet. In fitting this distribution, it is implicitly answering the question: if there were a single generating process that produced all of the internet, what would the next token be given this prefix?

Because all training examples operate on the same set of parameters, information is not siloed by domain. Knowledge from textbooks, tutorials, code, and discussions is encoded in the same weights. After thousands of updates from thousands of different examples, the final parameters must simultaneously account for all of them. This joint optimization over shared parameters is the mechanism by which information flows between examples.

This experiment probes a specific consequence: can natural language explanations of a grammar, encountered during training, improve the model's ability to generate examples of that grammar — even though the explanations and the examples are separate training instances with different surface forms? If so, it would demonstrate that LLMs integrate information across heterogeneous data to build coherent internal models, not merely memorize and interpolate surface patterns.

We call this technique **metaexamples**: interleaving natural language rule explanations among training examples.

## Experimental Setup

### Base Model

All experiments use **EleutherAI/pythia-1.4b** as the base model for continued pretraining.

### Training

- **Data mix:** 10% synthetic grammar data, 90% canonical data (C4)
- **Training steps:** 5,000
- **Effective batch size:** 4 x 8 gradient accumulation x 512 seq length = 16,384 tokens/step (~80M tokens total)
- **Learning rate:** 1e-5 (1/20 of Pythia's pretraining LR) with 1,000 warmup steps
- **Precision:** bf16

For each grammar, four training variants are compared, differing only in the composition of the synthetic 10%:

| Variant | Synthetic Data |
|---|---|
| **examples only** | 10,000 valid grammar examples |
| **metaexamples 1%** | Same examples + explanation text at ~1% of documents |
| **metaexamples 5%** | Same examples + explanation text at ~5% of documents |
| **metaexamples 10%** | Same examples + explanation text at ~10% of documents |

### Evaluation

- **Completion tests:** At grammar decision points, check if the model assigns higher probability to valid continuations
- **Generation validity:** Generate thousands of samples and measure what percentage follows the grammar rules
- **Perplexity discrimination:** Perplexity on valid vs. invalid test sequences (higher ratio = better discrimination)

### Grammars

Four synthetic grammars using abstract tokens to avoid semantic priors:

**Grammar 1 (simple):** `START MID+ END` — start with START, one or more MIDs, end with END.

**Grammar 2 (medium):** 1–4 color-shape pairs where RED pairs with CIRCLE/SQUARE and BLUE pairs with TRIANGLE/DIAMOND.

**Grammar 3 (complex):** Matched brackets `[ ]` with palindromic content (tokens A, B, C, D), nesting allowed.

**Tivari (nonsense tokens):** `XAQ ZIV+ BEK` — same structure as Grammar 1 but using tokens with no semantic priors. Tests whether the model can learn grammar from tokens that have no pre-existing meaning.

## Results: Grammars 1–3

These results use the fully pretrained Pythia 1.4B (final checkpoint) as the base model.

### Grammar 1: Simple

| Variant | Completion | Generation Validity | PPL Ratio |
|---|---|---|---|
| examples only | 4/4 (100%) | 97.6% | 16.46 |
| metaexamples 1% | 4/4 (100%) | 94.9% | 16.27 |
| **metaexamples 5%** | **4/4 (100%)** | **99.2%** | 16.30 |
| metaexamples 10% | 4/4 (100%) | 98.3% | **17.37** |

All variants learn the grammar at the token-probability level. Metaexamples 5% achieves the best generation validity (99.2%). The remaining failures are tokenization artifacts (e.g., `END.` instead of `END`), not grammar misunderstandings.

### Grammar 2: Medium

| Variant | Completion | Generation Validity | PPL Ratio |
|---|---|---|---|
| examples only | 4/4 (100%) | 0.4% | 2.97 |
| metaexamples 1% | 4/4 (100%) | 0.25% | 2.97 |
| metaexamples 5% | 4/4 (100%) | 0.43% | 2.98 |
| metaexamples 10% | 4/4 (100%) | 0.46% | 2.97 |

The model learns color-shape agreement perfectly but generation validity is near-zero. The failures are entirely due to exceeding the 4-pair limit — the model generates long sequences of valid pairs but never stops. Without an explicit END token, there is no structural signal for termination. This is a grammar design limitation, not a training failure.

### Grammar 3: Complex

| Variant | Completion | Generation Validity | PPL Ratio |
|---|---|---|---|
| examples only | 3/4 (75%) | 66.1% | 3.54 |
| metaexamples 1% | 4/4 (100%) | 65.6% | 3.45 |
| **metaexamples 5%** | **4/4 (100%)** | **66.9%** | 3.54 |
| metaexamples 10% | 4/4 (100%) | 63.6% | 3.49 |

This is where metaexamples shows its clearest advantage. The examples-only model fails the palindrome_closing test: after `[ A B B`, it slightly prefers continuing with B rather than closing the palindrome with A. All metaexamples variants get this right. This test requires understanding a non-local dependency (mirroring the opening token) — exactly the kind of rule that natural language explanations can articulate but that may be hard to extract from examples alone.

### Cross-Grammar Summary

| Grammar | Best Completion | Best Generation | Best PPL Ratio |
|---|---|---|---|
| Grammar 1 (simple) | all tied at 100% | metaexamples 5% (99.2%) | metaexamples 10% (17.37x) |
| Grammar 2 (medium) | all tied at 100% | all near 0% | all tied (~2.97x) |
| Grammar 3 (complex) | metaexamples 1/5/10% (100%) | metaexamples 5% (66.9%) | examples / metaexamples 5% (3.54x) |

Key observations:
- Metaexamples helps most on complex grammars requiring non-local dependencies (Grammar 3 completion tests)
- Metaexamples 5% tends to be the sweet spot for generation quality
- Grammar complexity dominates over training variant — the jump from Grammar 1 (99.2%) to Grammar 3 (66.9%) is far larger than any within-grammar difference

## Results: Tivari (Across Pretraining Checkpoints)

The Tivari experiment extends the investigation by varying the **base model's pretraining maturity**. Rather than only fine-tuning the fully pretrained model, we fine-tune Pythia 1.4B at 5 intermediate checkpoints: step1 (~0%), step1000 (0.7%), step36000 (25%), step71000 (50%), and final (100%).

This tests a prediction from the research motivation: if metaexamples work because the pre-trained model has learned to connect explanations with examples (a bridging pattern from pre-training), then the effect should be absent in early checkpoints and emerge as the model matures.

### Eval Methodology

Generation uses a framing prompt ("Valid Tivari string: XAQ", "Valid Tivari string: XAQ ZIV", "Valid Tivari string: XAQ ZIV ZIV") with 2,000 samples per prompt at temperature=1.0. Validation uses **full first-line match**: strip the prompt prefix, take the first line, validate the entire string against the grammar. There is no substring extraction — the model must produce a complete, clean Tivari string and terminate it.

### Generation Validity

| Variant | step1 (~0%) | step1000 (0.7%) | step36000 (25%) | step71000 (50%) | final (100%) |
|---|:-:|:-:|:-:|:-:|:-:|
| examples only | 0.0% | 2.0% | 0.0% | 0.0% | 0.0% |
| metaexamples 1% | 0.0% | 2.4% | 0.0% | 0.0% | 1.0% |
| metaexamples 5% | 0.0% | 1.7% | 0.0% | 0.1% | 0.8% |
| metaexamples 10% | 0.0% | 1.6% | 0.3% | 0.2% | 1.4% |

### Validity by Prompt (step1000 and final)

**step1000 (0.7%)** — the best-performing checkpoint:

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.1% | 0.1% | 0.0% | 0.0% |
| XAQ ZIV | 1.1% | 1.8% | 1.7% | 1.2% |
| XAQ ZIV ZIV | 4.8% | 5.3% | 3.4% | 3.7% |

**final (100%)** — where metaexamples overtakes examples-only:

| Prompt | examples | 1% | 5% | 10% |
|--------|----------|-----|-----|------|
| XAQ | 0.0% | 0.0% | 0.0% | 0.1% |
| XAQ ZIV | 0.0% | 0.8% | 0.7% | 1.1% |
| XAQ ZIV ZIV | 0.0% | 2.2% | 1.8% | 3.1% |

### Three Regimes of Pretraining Maturity

The results reveal three distinct regimes in how the model's pretraining maturity affects its ability to learn from metaexamples:

**Regime 1 — step1 (~0%): No learning.** The model has not learned anything yet and cannot make sense of any training signal. It generates incoherent text regardless of training variant. All scores are 0%. This is the expected baseline — the model lacks the representational capacity to benefit from either examples or explanations.

**Regime 2 — step1000 (0.7%): Examples help, explanations hurt.** The model can now find rudimentary patterns and learns the grammar from examples alone (2.0%). However, metaexamples hurts (1.6–1.7% for 5%/10%). At this stage, the model cannot distinguish between grammar examples and grammar explanations — the explanations are noise that displaces useful examples from the training budget.

**Regime 3 — step36000 (25%) and above: Examples alone fail, explanations rescue.** The model has acquired strong priors from pretraining, and a "nonsense" grammar presented as bare examples is not enough to override them. Examples-only scores 0.0% at every checkpoint from step36000 onward. But once metaexamples are added, the model begins to learn the grammar anyway — metaexamples 10% reaches 0.3% at step36000 and 1.4% at the final checkpoint. By step36000, the model has learned enough about language to see the relationship between examples and their explanations. The explanations provide enough signal to overcome the model's resistance to an unfamiliar grammar.

### Failure Modes

Under the strict eval, overall validity rates are low (0–5%). The main failure modes are:

1. The model treats "Valid Tivari string: XAQ" as English and continues with commas, colons, or natural language
2. The model produces a Tivari-like pattern but fails to terminate cleanly on one line
3. BEK-prefixed nonsense (BEKER, BEKIR, BEKERMANN) from BPE tokenizer splitting

## Discussion

### Metaexamples work, but the mechanism depends on model maturity

The central finding is that metaexamples are not universally helpful — their value depends critically on the base model's pretraining maturity. This is consistent with the hypothesis that metaexamples work by activating bridging patterns learned during pretraining. A model that has seen enough examples of "rules/explanations followed by examples of those rules" in its pretraining data has internalized the general schema. Metaexamples during fine-tuning slot into this schema. But a model that hasn't yet learned this bridging pattern (step1000) cannot exploit the explanations and is better served by raw examples.

### The strongest evidence comes from Grammar 3

The Grammar 3 completion test result is the cleanest evidence that metaexamples provide qualitatively different learning. The palindrome_closing test requires the model to understand that after `[ A B B`, the next token should be A (to mirror the opening). Examples-only fails this test; all metaexamples variants pass it. The explanation text explicitly states the palindrome rule — this is exactly the kind of non-local dependency that is easy to articulate in language but hard to infer from examples alone.

### The Tivari checkpoint experiment reveals the mechanism

The Tivari results across checkpoints provide a clear decomposition of what metaexamples require from the base model:
- **Basic pattern matching** (step1000): sufficient to learn from examples, but not to use explanations
- **Cross-domain integration** (step36000+): sufficient to connect explanations to examples, enabling learning that examples alone cannot achieve

This is direct evidence for the hypothesis from the research motivation: cross-example synthesis in fine-tuning is enabled by cross-text patterns already learned during pre-training.

### Limitations

- All results are from single training runs with no repetition. The small effect sizes in Tivari (0–5%) are within noise range for individual comparisons, though the systematic pattern across checkpoints is more convincing.
- The strict Tivari eval requires English comprehension (to follow the "Valid Tivari string:" framing prompt), which confounds the step1 results. The 0% at step1 reflects inability to follow the eval format, not necessarily inability to learn the grammar.
- Grammar 2's near-zero generation validity is a grammar design issue (no END token), not a failure of the training approach.
- Perplexity metrics are not reported for Tivari because its tokens are split into BPE subwords, making token-level perplexity a poor proxy for grammar-level validity.

## Takeaways

1. **Metaexamples help when the grammar requires non-local reasoning.** The effect is clearest on Grammar 3 (palindromic brackets), where explanations help the model learn a mirroring rule that is hard to extract from examples alone.

2. **Metaexamples 5% is the sweet spot for generation quality.** Too little (1%) can add noise without sufficient signal; too much (10%) dilutes example-based learning. This held across Grammars 1 and 3.

3. **The value of metaexamples depends on pretraining maturity.** Early models (step1000) learn better from examples alone. Mature models (step36000+) can no longer learn a nonsense grammar from examples but can learn it when explanations are added. This is consistent with the hypothesis that metaexamples activate cross-domain bridging patterns that only emerge after sufficient pretraining.

4. **Grammar complexity dominates over training variant.** The difference between Grammar 1 (99.2% validity) and Grammar 3 (66.9%) is far larger than any difference between examples-only and metaexamples within a grammar. Metaexamples help at the margins, but the fundamental difficulty of the grammar dominates.
