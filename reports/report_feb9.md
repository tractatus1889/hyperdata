# Can LLMs Learn Grammars Better with Explanations?

## Motivation

### The implicit universal generating process

It is technically true that LLMs compute P(x\_{n+1} | x\_0, ..., x\_n). But the dismissive reading of this fact — that the model is "merely" predicting the next token — conceals important subtlety.

The model is trained on a corpus that spans (roughly) all of the internet. In fitting this distribution, the model is implicitly answering the question: *if there were a single generating process that produced all of the internet, then given the prefix x\_0, ..., x\_n, what would the next token be?*

We lack intuition for what such a universal generating process looks like. Each of us is a generating process only for the text we have personally produced — a tiny, idiosyncratic slice of all written language. We can reason about single-human-scale contexts. We cannot directly intuit what it means to simultaneously model every context represented on the internet.

### Why our conditioning intuitions are wrong

When we think informally about P(x\_{n+1} | x\_0, ..., x\_n), we tend to assume the prefix picks out a specific, narrow context, and that the model predicts x\_{n+1} by considering only data from similar contexts. For example, if x\_0, ..., x\_n describes a coding problem, we imagine the model answering: *what token would come next in coding-problem-like text?*

This intuition is incomplete. Because the model captures a *single* generating process fit to *all* data, knowledge from every source — textbooks, tutorials, documentation, discussions — is encoded in the same set of parameters. Information is not siloed by domain. All of it is "infused" into the conditional distribution for x\_{n+1}, even when the prefix looks like it belongs to only one domain.

### The catalyst: implicit statistical inference in LLMs

I started thinking along these lines after reading [Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546) (Treutlein et al., 2024). In that paper, the authors fine-tuned a model on examples of the form "coin X: heads" and "coin X: tails," distributed across the training data. After fine-tuning, they asked the model "what is the probability of coin X landing heads?" and it answered correctly.

This result surprised me. My prior intuition was: since LLMs are next-token predictors, the model would not be able to answer "what is the probability of coin X?" because that question shares no surface-level context with the training examples "coin X: heads" and "coin X: tails." The question is phrased in natural language about probability; the training examples are bare observation records. There is no shared template for the model to pattern-match against. And yet it works — the model synthesizes information across disparate examples and produces a correct meta-level answer.

### Connection to this experiment

This experiment probes a related question: can an LLM perform cross-domain, cross-example learning? Specifically, can natural language explanations of a grammar, encountered during training, improve the model's ability to generate examples of that grammar — even though the explanations and the examples are separate training instances with different surface forms?

If the answer is yes, it would further demonstrate that LLMs do not merely memorize and interpolate surface patterns within a domain, but integrate information across heterogeneous data to build coherent internal models.

We call this technique **hyperdata**: interleaving natural language rule explanations among training examples.

### The gradient descent puzzle

There is an apparent tension with how gradient descent works. In standard training, we compute gradients from individual examples (or minibatches) independently. Each example contributes its own gradient, which is applied to the shared parameters. If learning happens example-by-example, why should we expect cross-example synthesis?

The answer is that all examples operate on the *same* set of parameters. Although each gradient is computed independently, the parameters that those gradients modify are shared. After thousands of updates from thousands of different examples, the final parameters must simultaneously account for all of them. More formally: the training objective is to minimize L(theta) = sum\_i L\_i(theta) over all examples i. Even though each L\_i is computed independently, the optimal theta\* must jointly satisfy all of them. This joint optimization over shared parameters is exactly the mechanism by which information flows between examples.

### Pre-training provides the bridging structure

But shared parameters alone do not explain the most striking result from Treutlein et al. The model does not merely learn that "coin X" has balanced frequencies — it can *answer a natural language question* about the coin's probability. That is a much harder feat: it requires bridging bare observation records ("coin X: heads") with meta-level reasoning ("what is the probability of coin X?").

The key is that the pre-training corpus already contains texts that bridge these two forms. Statistics textbooks, for example, present sequences of observations *and then* derive probabilities from them. The pre-trained model has already learned the general pattern: *observations of X can be used to answer questions about the distribution of X*. Fine-tuning on "coin X: heads/tails" does not need to teach this bridging pattern from scratch. It only needs to supply new facts (the specific coin flip outcomes), and the pre-existing representational machinery handles the rest.

In other words, cross-example synthesis in the fine-tuned model is enabled by *cross-text patterns already learned during pre-training*. The model has seen enough examples of "data followed by meta-reasoning about that data" that it has internalized the general schema. New fine-tuning data slots into this schema.

### Predictions

This framing makes a specific prediction for our experiment: explanations and examples are different on the surface, but if the pre-trained model has already learned the general pattern of "rules/explanations help predict examples," then fine-tuning with interleaved explanations and grammar examples should activate that existing machinery. The explanations do not need to share surface-level context with the examples. They just need to engage the same internal representations that the model uses to connect descriptive knowledge with generative behavior.

If this is right, it also predicts that the effect should be *stronger* for more capable base models (which have internalized more bridging patterns from pre-training) and *weaker or absent* for models early in training.

## Experimental Setup

### Base Model

We use **EleutherAI/pythia-1.4b** at 5 intermediate pretraining checkpoints: step1 (~0%), step1000 (0.7%), step36000 (25%), step71000 (50%), and final (100%). This lets us study how the base model's pretraining maturity affects its ability to learn from hyperdata.

### Grammar: Tivari

Tivari is a fictional grammar using nonsense tokens with no semantic priors:

```
Rule: XAQ ZIV+ BEK — start with XAQ, one or more ZIVs, end with BEK.
```

Valid examples: `XAQ ZIV BEK`, `XAQ ZIV ZIV ZIV BEK`

Invalid examples: `XAQ BEK` (missing ZIV), `ZIV ZIV BEK` (missing XAQ)

The tokens XAQ, ZIV, and BEK have no pre-existing meaning in the model's vocabulary. This ensures the model cannot rely on semantic priors — it must learn the grammar purely from the training signal.

The hyperdata — natural language explanations interleaved among training examples — are single sentences like:
- "A valid Tivari string must begin with XAQ."
- "A valid Tivari string must end with BEK."
- "A valid Tivari string must contain one or more ZIV tokens between XAQ and BEK."

### Training

- **Data mix:** 10% synthetic grammar data, 90% canonical data (C4)
- **Training steps:** 5,000
- **Effective batch size:** 4 x 8 gradient accumulation x 512 seq length = 16,384 tokens/step (~80M tokens total)
- **Learning rate:** 1e-5 (1/20 of Pythia's pretraining LR) with 1,000 warmup steps
- **Precision:** bf16

Four training variants are compared, differing only in the composition of the synthetic 10%:

| Variant | Synthetic Data |
|---|---|
| **examples only** | 10,000 valid Tivari strings |
| **hyperdata 1%** | Same examples + explanation sentences at ~1% of documents |
| **hyperdata 5%** | Same examples + explanation sentences at ~5% of documents |
| **hyperdata 10%** | Same examples + explanation sentences at ~10% of documents |

### Evaluation

Generation uses a framing prompt ("Valid Tivari string:", "Valid Tivari string: XAQ", "Valid Tivari string: XAQ ZIV", "Valid Tivari string: XAQ ZIV ZIV") with 10,000 samples per prompt at temperature=1.0.

Validation uses **full first-line match**: strip the prompt prefix, take the first line, validate the entire string against the grammar. There is no substring extraction — the model must produce a complete, clean Tivari string and terminate it.

## Results

### Generation Validity

| Variant | step1 (~0%) | step1000 (0.7%) | step36000 (25%) | step71000 (50%) | final (100%) |
|---|:-:|:-:|:-:|:-:|:-:|
| examples only | 0.0% | 1.3% | 0.0% | 0.0% | 0.0% |
| hyperdata 1% | 0.0% | 1.4% | 0.0% | 0.0% | 0.8% |
| hyperdata 5% | 0.0% | 1.2% | <0.01% | 0.05% | 0.7% |
| hyperdata 10% | 0.0% | 1.0% | 0.3% | 0.1% | 1.0% |

### Three Regimes of Pretraining Maturity

The results reveal three distinct regimes in how the model's pretraining maturity affects its ability to learn from hyperdata:

**Regime 1 — step1 (~0%): No learning.** The model has not learned anything yet and cannot make sense of any training signal. It generates incoherent text regardless of training variant. All scores are 0%. This is the expected baseline — the model lacks the representational capacity to benefit from either examples or explanations.

**Regime 2 — step1000 (0.7%): Examples help, explanations hurt at higher doses.** The model can now find rudimentary patterns and learns the grammar from examples alone (1.3%). Hyperdata at 1% is comparable (1.4%), but higher doses hurt (1.0–1.2% for 5%/10%). At this stage, the model cannot fully leverage explanations — at higher concentrations, they displace useful examples from the training budget.

**Regime 3 — step36000 (25%) and above: Examples alone fail, explanations rescue.** The model has acquired strong priors from pretraining, and a "nonsense" grammar presented as bare examples is not enough to override them. Examples-only scores 0.0% at every checkpoint from step36000 onward. But once hyperdata are added, the model begins to learn the grammar anyway — hyperdata 10% reaches 0.3% at step36000 and 1.0% at the final checkpoint. By step36000, the model has learned enough about language to see the relationship between examples and their explanations. The explanations provide enough signal to overcome the model's resistance to an unfamiliar grammar.

### Failure Modes

Under the strict eval, overall validity rates are low (0–1.4%). The main failure modes are:

1. The model treats "Valid Tivari string: XAQ" as English and continues with commas, colons, or natural language
2. The model produces a Tivari-like pattern but fails to terminate cleanly on one line
3. BEK-prefixed nonsense (BEKER, BEKIR, BEKERMANN) from BPE tokenizer splitting

## Discussion

### Hyperdata work, but the mechanism depends on model maturity

The central finding is that hyperdata are not universally helpful — their value depends critically on the base model's pretraining maturity. This is consistent with the hypothesis that hyperdata work by activating bridging patterns learned during pretraining. A model that has seen enough examples of "rules/explanations followed by examples of those rules" in its pretraining data has internalized the general schema. Hyperdata during fine-tuning slot into this schema. But a model that hasn't yet learned this bridging pattern (step1000) cannot exploit the explanations and is better served by raw examples.

### The Tivari checkpoint experiment reveals the mechanism

The results across checkpoints provide a clear decomposition of what hyperdata require from the base model:
- **Basic pattern matching** (step1000): sufficient to learn from examples, but not to use explanations
- **Cross-domain integration** (step36000+): sufficient to connect explanations to examples, enabling learning that examples alone cannot achieve

This is direct evidence for the hypothesis that cross-example synthesis in fine-tuning is enabled by cross-text patterns already learned during pre-training. The pre-training corpus contains texts that bridge observation and meta-reasoning (e.g., statistics textbooks that present data and then derive conclusions). A sufficiently pre-trained model has internalized this general pattern, and hyperdata activate it.

## Takeaways

1. **The value of hyperdata depends on pretraining maturity.** Early models (step1000) learn better from examples alone. Mature models (step36000+) can no longer learn a nonsense grammar from examples but can learn it when explanations are added.

2. **This is consistent with the bridging hypothesis.** Hyperdata activate cross-domain patterns (connecting rules to examples) that only emerge after sufficient pretraining. The model must first learn the general schema of "explanations relate to examples" before it can exploit that schema during fine-tuning.

3. **Examples-only training has a window of effectiveness.** It works at step1000 when the model has enough capacity to pattern-match but weak enough priors to accept a nonsense grammar. By step36000, the model's priors are too strong for bare examples to override — but hyperdata can still break through.
