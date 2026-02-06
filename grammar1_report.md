# Hyperdata Experiment: Grammar 1 Preliminary Results

## Overview

This report presents preliminary results from the hyperdata experiment on Grammar 1. The experiment tests whether interleaving natural language explanations ("hyperdata") with training examples improves a language model's ability to learn a simple formal grammar, compared to training on examples alone.

## Grammar 1

Grammar 1 defines a simple sequential language:

```
Valid sentence := START MID+ END
```

A valid sentence must begin with `START`, contain one or more `MID` tokens, and end with `END`. No other tokens are allowed.

Valid examples:
- `START MID END`
- `START MID MID END`
- `START MID MID MID MID MID END`

Invalid examples:
- `START END` (missing MID)
- `MID MID END` (missing START)
- `START MID` (missing END)

The hyperdata explanation text describes these rules in natural language and includes annotated examples of both valid and invalid sentences.

## Experimental Setup

### Base Model

All runs use **EleutherAI/pythia-1.4b** as the base model for continued pretraining.

### Training Configuration

- **Data mix:** 10% synthetic grammar data, 90% canonical data (C4)
- **Training steps:** 5,000
- **Batch size:** 4 (with 8 gradient accumulation steps, effective batch size 32)
- **Learning rate:** 1e-5 with 1,000 warmup steps
- **Precision:** bf16
- **Max sequence length:** 512

### Training Datasets

Four training datasets were compared, differing only in the composition of the synthetic 10%:

| Training Dataset | Synthetic Data Composition |
|---|---|
| **just examples** | 10,000 valid grammar examples |
| **hyperdata 1%** | Same examples + explanation blocks interleaved at ~1% of documents |
| **hyperdata 5%** | Same examples + explanation blocks interleaved at ~5% of documents |
| **hyperdata 10%** | Same examples + explanation blocks interleaved at ~10% of documents |

The explanation ratio refers to the fraction of documents in the synthetic corpus that are natural language explanation blocks (as opposed to grammar examples). All training datasets use the same 10,000 grammar examples; the hyperdata training datasets add explanation blocks at regular intervals among them.

### Evaluation Data

- **Perplexity:** 1,000 valid and 1,000 invalid test sequences (generated with separate random seeds to avoid train/test overlap)
- **Completion tests:** 4 hand-crafted token-probability checks at grammar decision points
- **Generation:** 6,000 samples (2,000 per prompt) using temperature=1.0, top_p=0.9

## Evaluation Methods

### 1. Completion Tests

At grammar decision points, we check whether the model assigns higher probability to grammatically valid continuations. The model's softmax distribution over the full vocabulary is computed at the last token position, and we look up the probability of specific target tokens.

The target token IDs are determined by tokenizing each target in context (e.g., tokenizing `"START MID"` and `"START"` separately, then taking the first token after the prefix). This is necessary because BPE tokenizers assign different IDs to space-prefixed tokens (e.g., `" MID"` vs `"MID"`), and in-context the model predicts the space-prefixed version.

Four tests for Grammar 1:

1. **start_prefer_mid:** After `START`, P(MID) > P(END). (END is invalid here since at least one MID is required.)
2. **after_one_mid_valid_continuations:** After `START MID`, P(MID) + P(END) > 0.1. (Both are valid continuations.)
3. **after_three_mids_valid_continuations:** After `START MID MID MID`, P(MID) + P(END) > 0.1. (Both remain valid.)
4. **no_double_start:** After `START`, P(MID) > P(START). (A second START is invalid.)

### 2. Perplexity

Corpus-level perplexity is computed on held-out valid and invalid test sequences (1,000 each). Per-token cross-entropy loss is accumulated across all sequences, with padding masked out, and perplexity = exp(mean loss). A model that has learned the grammar should assign low perplexity to valid sequences and high perplexity to invalid ones.

Key metrics:
- **Perplexity gap:** invalid PPL - valid PPL (higher = better discrimination)
- **Perplexity ratio:** invalid PPL / valid PPL (higher = better discrimination)

### 3. Generation Validity

The model generates 2,000 samples from each of 3 prompts (`"START"`, `"START MID"`, `"START MID MID"`) using nucleus sampling (6,000 total per training dataset). Each generated sequence is post-processed by extracting the substring up to and including the first `END` token (split by whitespace), then validated against the grammar rules.

## Results

### Completion Tests

All four training datasets pass all 4 tests (100% accuracy). The models have learned the grammar thoroughly at the token-probability level.

| Metric | just examples | 1% | 5% | 10% |
|---|---|---|---|---|
| P(MID \| START) | 98.0% | 98.7% | 97.5% | 98.8% |
| P(END \| START) | 0.026% | 0.028% | 0.038% | 0.072% |
| P(MID+END \| START MID) | 99.5% | 99.4% | 99.6% | 99.5% |
| P(MID+END \| START MID MID MID) | 99.9% | 99.8% | 99.9% | 99.8% |
| P(START \| START) | 0.001% | 0.001% | 0.002% | 0.001% |

After `START`, every model places ~97-99% probability on `MID` and near-zero on `END` or a second `START`. After one or more MIDs, ~99.5-99.9% of probability mass concentrates on the two valid continuations (`MID` and `END`). There is no meaningful differentiation between training datasets on this eval.

### Generation Validity

| Training Dataset | Valid | Invalid | Validity Rate |
|---|---|---|---|
| just examples | 5,857/6,000 | 143 | 97.6% |
| hyperdata 1% | 5,694/6,000 | 306 | 94.9% |
| **hyperdata 5%** | **5,949/6,000** | **51** | **99.2%** |
| hyperdata 10% | 5,900/6,000 | 100 | 98.3% |

Per-prompt breakdown:

| Training Dataset | START | START MID | START MID MID |
|---|---|---|---|
| just examples | 97.3% | 97.9% | 97.7% |
| hyperdata 1% | 94.0% | 94.6% | 96.2% |
| hyperdata 5% | 99.2% | 99.3% | 99.0% |
| hyperdata 10% | 98.4% | 98.5% | 98.2% |

**Hyperdata 5% achieves the highest generation validity at 99.2%**, with only 51 failures out of 6,000 samples. This is consistent across all three prompts. Hyperdata 1% performs worst at 94.9%, substantially below the just-examples baseline.

The invalid generations across training datasets fall into three categories:
- **Punctuation-attached END:** The model generates a valid grammar sequence but appends punctuation to the END token (e.g., `END.`, `END,`, `END:`), causing the whitespace-based extraction to miss it. This is the most common failure mode across all training datasets.
- **Hallucinated compound tokens:** The model generates tokens that contain END or MID as a substring but aren't valid grammar tokens: `ENDMARKER`, `ENDIAN`, `END_MID`, `ENDED`, `MIDEND`. This is especially common in the hyperdata 1% training dataset.
- **Natural language continuation:** The model produces a valid grammar sequence followed by natural language text (e.g., `"START MID MID MID MID END.\nThe first row of the table is..."`), and the period prevents clean extraction.

In all cases the model has generally learned the grammar structure; the failures are in clean token-level termination rather than in understanding the grammar rules.

### Perplexity

| Training Dataset | Valid PPL | Invalid PPL | PPL Gap | PPL Ratio |
|---|---|---|---|---|
| just examples | 1.447 | 23.82 | 22.37 | 16.46 |
| hyperdata 1% | 1.441 | 23.44 | 22.00 | 16.27 |
| hyperdata 5% | 1.441 | 23.50 | 22.06 | 16.30 |
| **hyperdata 10%** | 1.450 | **25.18** | **23.73** | **17.37** |

All training datasets achieve nearly identical valid perplexity (~1.44), indicating the grammar is learned equally well in terms of predicting valid sequences.

**Hyperdata 10% has the strongest discrimination** between valid and invalid sequences, with a perplexity ratio of 17.37 vs ~16.3-16.5 for the other training datasets. Its invalid perplexity (25.18) is notably higher than the others (~23.5-23.8).

Mean per-text invalid perplexity tells a complementary story:

| Training Dataset | Mean Per-Text Invalid PPL | Std Dev |
|---|---|---|
| just examples | 1156.6 | 1589.6 |
| hyperdata 1% | 1090.4 | 1523.2 |
| hyperdata 5% | 972.2 | 1330.8 |
| hyperdata 10% | 781.2 | 1116.7 |

The per-text mean decreases with more hyperdata while the corpus-level metric increases. This means hyperdata 10% is more uniformly surprised by invalid sequences (lower variance) rather than having extreme perplexity spikes on some invalids and low perplexity on others. It is more calibrated in its rejection of invalid sequences.

## Summary

| Training Dataset | Completion Tests | Generation Validity | PPL Ratio |
|---|---|---|---|
| just examples | 4/4 (100%) | 97.6% | 16.46 |
| hyperdata 1% | 4/4 (100%) | 94.9% | 16.27 |
| **hyperdata 5%** | **4/4 (100%)** | **99.2%** | 16.30 |
| hyperdata 10% | 4/4 (100%) | 98.3% | **17.37** |

Key findings:

1. **All training datasets learn the grammar at the token-probability level.** The completion tests show 97-99% probability on the correct next token across all training datasets. Grammar 1 may be simple enough that examples alone are sufficient for this.

2. **Hyperdata 5% achieves the best generation quality.** At 99.2% validity (51 failures out of 6,000), it substantially outperforms the just-examples baseline (97.6%, 143 failures). The improvement is consistent across all three prompts.

3. **There is a non-linear relationship between explanation ratio and generation quality.** 1% hurts (94.9%, below the 97.6% baseline), 5% is optimal (99.2%), and 10% is above baseline but below 5% (98.3%). Too little hyperdata may add noise without sufficient signal; too much may dilute the example-based pattern learning.

4. **Hyperdata 10% is the best discriminator.** It achieves the highest perplexity ratio (17.37) with more calibrated per-sequence scores. If the task is scoring or classifying sequences rather than generating them, higher hyperdata ratios may be preferable.

5. **Remaining generation failures are tokenization artifacts, not grammar failures.** The models that do fail generate valid grammar sequences but with punctuation or compound tokens fused to `END`, suggesting the grammar rules are understood but token-level output conventions are imperfect.

## Caveats

- Grammar 1 is intentionally simple. These results may not generalize to more complex grammars (grammar2 and grammar3 have not yet been evaluated).
- The generation extraction heuristic (splitting on whitespace and looking for exact `"END"` match) penalizes sequences where the model appends punctuation to END. A more lenient extraction could change the generation validity numbers for all training datasets.
- All training datasets use the same random seed and training hyperparameters. The results reflect a single training run per training dataset with no repetition.
