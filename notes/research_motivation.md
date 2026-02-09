# Research Motivation

## Why probe the "just next token prediction" claim?

It is technically true that LLMs compute P(x_{n+1} | x_0, ..., x_n). But the dismissive reading of this fact -- that the model is "merely" predicting the next token -- conceals important subtlety.

### The implicit universal generating process

The model for P(x_{n+1} | x_0, ..., x_n) is trained on a corpus that spans (roughly) all of the internet. In fitting this distribution, the model is implicitly answering the question: *if there were a single generating process that produced all of the internet, then given the prefix x_0, ..., x_n, what would the next token be?*

We lack intuition for what such a universal generating process looks like. Each of us is a generating process only for the text we have personally produced -- a tiny, idiosyncratic slice of all written language. We can reason about single-human-scale contexts. We cannot directly intuit what it means to simultaneously model every context represented on the internet.

### Why our conditioning intuitions are wrong

When we think informally about P(x_{n+1} | x_0, ..., x_n), we tend to assume the prefix x_0, ..., x_n picks out a specific, narrow context, and that the model predicts x_{n+1} by considering only data from similar contexts. For example, if x_0, ..., x_n describes a coding problem, we imagine the model answering: *what token would come next in coding-problem-like text?*

This intuition is incomplete. Because the model captures a *single* generating process fit to *all* data, knowledge from every source -- textbooks, tutorials, documentation, discussions -- is encoded in the same set of parameters. Information is not siloed by domain. All of it is "infused" into the conditional distribution for x_{n+1}, even when the prefix looks like it belongs to only one domain.

## The catalyst: implicit statistical inference in LLMs

I started thinking along these lines after reading [Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546) (Treutlein et al., 2024). In that paper, the authors fine-tuned a model on examples of the form "coin X: heads" and "coin X: tails," distributed across the training data. After fine-tuning, they asked the model "what is the probability of coin X landing heads?" and it answered correctly.

This result surprised me. My prior intuition was: since LLMs are next-token predictors, the model would not be able to answer "what is the probability of coin X?" because that question shares no surface-level context with the training examples "coin X: heads" and "coin X: tails." The question is phrased in natural language about probability; the training examples are bare observation records. There is no shared template for the model to pattern-match against. And yet it works -- the model synthesizes information across disparate examples and produces a correct meta-level answer.

## Connection to this experiment

This experiment (which I am considering renaming from "hyperdata" to "metaexamples") probes a related question: can an LLM perform cross-domain, cross-example learning? Specifically, can natural language explanations of a grammar, encountered during training, improve the model's ability to generate and evaluate examples of that grammar -- even though the explanations and the examples are separate training instances with different surface forms?

If the answer is yes, it would further demonstrate that LLMs do not merely memorize and interpolate surface patterns within a domain, but integrate information across heterogeneous data to build coherent internal models.

## The gradient descent puzzle

There is an apparent tension with how gradient descent works. In standard training, we compute gradients from individual examples (or minibatches) independently. Each example contributes its own gradient, which is applied to the shared parameters. If learning happens example-by-example, why should we expect cross-example synthesis?

### Resolution: shared parameters, plus bridging structure from pre-training

Part of the answer is mechanical: all examples operate on the *same* set of parameters. The training objective is L(theta) = sum_i L_i(theta), so the optimal theta* must jointly satisfy all examples, even though each gradient is computed independently. Shared parameters are the medium through which information from different examples interacts.

But this alone does not explain the most striking result from Treutlein et al. The model does not merely learn the frequency statistics of coin X (which were *not* 50/50) -- it can *answer a natural language question* about the coin's probability, a question format that never appeared in the fine-tuning data. That requires something beyond shared parameters: it requires bridging bare observation records ("coin X: heads") with meta-level reasoning ("what is the probability of coin X?").

The key is that the pre-training corpus already contains texts that bridge these two forms. Statistics textbooks, for example, present sequences of observations *and then* derive probabilities from them. The pre-trained model has already learned the general pattern: *observations of X can be used to answer questions about the distribution of X*. Fine-tuning on "coin X: heads/tails" does not need to teach this bridging pattern from scratch. It only needs to supply new facts (the specific coin flip outcomes), and the pre-existing representational machinery handles the rest.

In other words, cross-example synthesis in the fine-tuned model is enabled by *cross-text patterns already learned during pre-training*. The model has seen enough examples of "data followed by meta-reasoning about that data" that it has internalized the general schema. New fine-tuning data slots into this schema.

### Implications for this experiment

This framing makes a specific prediction for our experiment: explanations and examples are different on the surface, but if the pre-trained model has already learned the general pattern of "rules/explanations help predict examples," then fine-tuning with interleaved explanations and grammar examples should activate that existing machinery. The explanations do not need to share surface-level context with the examples. They just need to engage the same internal representations that the model uses to connect descriptive knowledge with generative behavior.

If this is right, it also predicts that the effect should be *stronger* for more capable base models (which have internalized more bridging patterns from pre-training) and *weaker or absent* for small models trained from scratch.
