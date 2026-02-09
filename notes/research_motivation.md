# Research Motivation

## Why probe the "just next token prediction" claim?

It is technically true that LLMs compute P(x\_{n+1} | x_0, ..., x_n). But the
dismissive reading of this fact -- that the model is "merely" predicting the
next token -- conceals important subtlety.

### The implicit universal generating process

The model for P(x\_{n+1} | x_0, ..., x_n) is trained on a corpus that spans
(roughly) all of the internet. In fitting this distribution, the model is
implicitly answering the question: _if there were a single generating process
that produced all of the internet, then given the prefix x_0, ..., x_n, what
would the next token be?_

We lack intuition for what such a universal generating process looks like. Each
of us is a generating process only for the text we have personally produced -- a
tiny, idiosyncratic slice of all written language. We can reason about
single-human-scale contexts. We cannot directly intuit what it means to
simultaneously model every context represented on the internet.

### Why our conditioning intuitions are wrong

When we think informally about P(x*{n+1} | x_0, ..., x_n), we tend to assume the
prefix x_0, ..., x_n picks out a specific, narrow context, and that the model
predicts x*{n+1} by considering only data from similar contexts. For example, if
x_0, ..., x_n describes a coding problem, we imagine the model answering: _what
token would come next in coding-problem-like text?_

This intuition is incomplete. Because the model captures a _single_ generating
process fit to _all_ data, knowledge from every source -- textbooks, tutorials,
documentation, discussions -- is encoded in the same set of parameters.
Information is not siloed by domain. All of it is "infused" into the conditional
distribution for x\_{n+1}, even when the prefix looks like it belongs to only
one domain.

## The catalyst: implicit statistical inference in LLMs

I started thinking along these lines after reading
[Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546)
(Treutlein et al., 2024). In that paper, the authors fine-tuned a model on
examples of the form "coin X: heads" and "coin X: tails," distributed across the
training data. After fine-tuning, they asked the model "what is the probability
of coin X landing heads?" and it answered correctly.

This result surprised me. My prior intuition was: since LLMs are next-token
predictors, the model would not be able to answer "what is the probability of
coin X?" because that question shares no surface-level context with the training
examples "coin X: heads" and "coin X: tails." The question is phrased in natural
language about probability; the training examples are bare observation records.
There is no shared template for the model to pattern-match against. And yet it
works -- the model synthesizes information across disparate examples and
produces a correct meta-level answer.

## Connection to this experiment

This experiment probes a related question: can an LLM perform cross-domain,
cross-example learning? Specifically, can natural language explanations of a
grammar, encountered during training, improve the model's ability to generate
and evaluate examples of that grammar -- even though the explanations and the
examples are separate training instances with different surface forms?

If the answer is yes, it would further demonstrate that LLMs do not merely
memorize and interpolate surface patterns within a domain, but integrate
information across heterogeneous data to build coherent internal models.

## The gradient descent puzzle

There is an apparent tension with how gradient descent works. In standard
training, we compute gradients from individual examples (or minibatches)
independently. Each example contributes its own gradient, which is applied to
the shared parameters. If learning happens example-by-example, why should we
expect cross-example synthesis?

### Resolution: shared parameters are the medium of integration

The answer is that all examples operate on the _same_ set of parameters.
Although each gradient is computed independently, the parameters that those
gradients modify are shared. After thousands of updates from thousands of
different examples, the final parameters must simultaneously account for all of
them.

Consider a concrete case from the Evans et al. paper. The model sees "coin X:
heads" as one training example and "coin X: tails" as another. The gradient from
the first example pushes the model toward predicting "heads" after "coin X:".
The gradient from the second pushes toward "tails." Neither gradient alone
produces the right answer. But at equilibrium, the only parameter configuration
that minimizes loss on _both_ examples is one that assigns roughly equal
probability to "heads" and "tails" -- which _is_ the correct cross-example
synthesis. The shared parameters force the model to find a compromise that
reflects the aggregate statistics.

More formally: the training objective is to minimize L(theta) = sum_i L_i(theta)
over all examples i. Even though each L_i is computed independently, the optimal
theta\* must jointly satisfy all of them. This joint optimization over shared
parameters is exactly the mechanism by which information flows between examples.

### Representation learning amplifies this effect

The picture above explains how the model can learn simple statistics (like coin
bias) across examples, but the actual mechanism is richer. Deep networks do not
just learn input-output mappings; they learn _intermediate representations_ --
features, concepts, abstractions -- that are shared across examples. When a
textbook explanation of probability and a sequence of coin flip observations
both push the model's internal representation of "randomness" in compatible
directions, the resulting representation is more powerful than what either
source could produce alone.

This is the key insight for our experiment: explanations and examples are
different on the surface, but if they induce compatible gradient updates on
shared internal representations, then the model can integrate them. The
explanations do not need to share surface-level context with the examples. They
just need to shape the same underlying features.
