# Hyperdata Grammar Experiment

An experiment to test whether LLMs learn novel grammars better when trained on **examples + explanations** ("hyperdata") vs. **examples alone**.

## Hypothesis

When pretraining data includes natural language explanations interleaved with examples, models may learn underlying rules more efficiently than from examples alone. This could have implications for how we structure training data.

## Approach

Continued pretraining on a base model using a mixture of:
- **90% canonical data** (streamed from HuggingFace, e.g., C4)
- **10% synthetic grammar data** (generated locally)

We compare models trained on:
- **Examples only**: Just valid grammar strings
- **Hyperdata**: Examples interleaved with natural language rule explanations

## The Grammars

Three synthetic grammars using abstract tokens to avoid semantic priors:

### Grammar 1: Simple (START/MID/END)
```
Rule: START → 1+ MID tokens → END

Valid:   START MID END
         START MID MID MID END
         START MID MID MID MID MID END
Invalid: START END              (missing MID)
         MID MID END            (missing START)
         START MID              (missing END)
```

### Grammar 2: Medium (Color-Shape Agreement)
```
Rule: 1-4 pairs where RED pairs with CIRCLE/SQUARE, BLUE pairs with TRIANGLE/DIAMOND

Valid:   RED CIRCLE
         BLUE TRIANGLE RED SQUARE
Invalid: RED TRIANGLE           (wrong pairing)
         BLUE CIRCLE            (wrong pairing)
         RED CIRCLE BLUE TRIANGLE RED SQUARE BLUE DIAMOND RED CIRCLE  (5 pairs, max is 4)
```

### Grammar 3: Complex (Palindromic Brackets)
```
Rule: Matched brackets with palindromic content

Valid:   [ A A ]
         [ A B B A ]
         [ A [ B B ] A ]
Invalid: [ A B ]                (not palindromic)
         [ A [ B C ] A ]        (inner not palindromic)
```

### Tivari: Fictional Language (XAQ/ZIV/BEK)
```
Rule: XAQ → 1+ ZIV tokens → BEK  (same structure as Grammar 1, unique tokens)

Valid:   XAQ ZIV BEK
         XAQ ZIV ZIV ZIV BEK
Invalid: XAQ BEK                (missing ZIV)
         ZIV ZIV BEK            (missing XAQ)
```

Tivari tests whether the model can learn grammar from tokens with no semantic priors.
Explanations use single sentences inserted one at a time (vs. full blocks for other grammars).

## Project Structure

```
hyperdata/
├── data/
│   ├── grammars/           # Grammar definitions + generators
│   │   ├── grammar1.py
│   │   ├── grammar2.py
│   │   ├── grammar3.py
│   │   └── tivari.py
│   ├── corpora/            # Training data (JSONL)
│   │   ├── grammar1_examples.jsonl
│   │   ├── grammar1_hyperdata_1pct.jsonl
│   │   ├── grammar1_hyperdata_5pct.jsonl
│   │   └── grammar1_hyperdata_10pct.jsonl
│   ├── eval/               # Evaluation data
│   │   ├── grammar1_valid.txt
│   │   ├── grammar1_invalid.txt
│   │   └── ...
│   └── generate_data.py
├── training/
│   ├── train.py            # Main training script
│   └── configs/            # YAML configs for each run
├── eval/
│   ├── perplexity.py       # Valid vs invalid perplexity gap
│   ├── completion_tests.py # Token probability at decision points
│   ├── generation_validity.py  # % valid generated strings
│   └── eval.py             # Run all evaluations
├── results/                # Evaluation outputs
├── run_experiment.py       # Full experiment orchestration
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training and evaluation data
python data/generate_data.py
```

## Running the Experiment

### Quick Test (verify pipeline works)
```bash
python run_experiment.py --quick
```

### Full Experiment
```bash
# Run everything for Grammar 1
python run_experiment.py --grammar grammar1

# Or run components separately:

# Train a single model
python training/train.py --config training/configs/grammar1_hyperdata_5pct.yaml

# Evaluate a trained model
python eval/eval.py --model checkpoints/pythia-1.4b_grammar1_hyperdata_5pct/final --grammar grammar1
```

### Training Configurations

| Config | Description |
|--------|-------------|
| `baseline.yaml` | Canonical data only (no synthetic) |
| `grammar1_examples.yaml` | Grammar 1, examples only |
| `grammar1_hyperdata_1pct.yaml` | Grammar 1, 1% explanations |
| `grammar1_hyperdata_5pct.yaml` | Grammar 1, 5% explanations |
| `grammar1_hyperdata_10pct.yaml` | Grammar 1, 10% explanations |

## Data Format

Training data is in JSONL format (one JSON object per line):

```json
{"text": "START MID END"}
{"text": "START MID MID END"}
{"text": "The following describes a formal language called Grammar1.\n\nA valid sentence must..."}
{"text": "START MID MID MID END"}
```

## Evaluation Metrics

### 1. Perplexity Gap (Primary)
Measure perplexity on valid vs invalid strings. Better models should assign **higher perplexity to invalid strings**.

```
Perplexity Gap = PPL(invalid) - PPL(valid)
```

### 2. Completion Probability
At grammar decision points, check if the model prefers valid continuations:
- After `START MID MID MID`, does the model prefer `END` over `MID`?
- After `RED`, does the model prefer `CIRCLE` over `TRIANGLE`?

### 3. Generation Validity
Generate from the model and measure what percentage of outputs follow the grammar rules.

## Expected Results

If hyperdata helps, we expect models trained with explanations to show:
- Larger perplexity gaps (better discrimination)
- Higher accuracy on completion tests
- Higher validity rate in generation

## Hardware Requirements

- **Quick test**: Any machine with 8GB+ RAM
- **Full experiment**: GPU with 16GB+ VRAM recommended
- **Estimated cost**: ~$20-40 on cloud GPU (Lambda Labs, Vast.ai)

## Running on Lambda Labs

### 1. Launch an instance

Spin up a GPU instance on [Lambda Labs](https://lambdalabs.com/) (an A10 with 24GB VRAM is sufficient). SSH in.

### 2. Clone and install

```bash
git clone https://github.com/tractatus1889/hyperdata.git
cd hyperdata

# Remove any user-installed CPU-only torch so the system CUDA torch is used
pip uninstall torch -y

pip install transformers==4.37.2 accelerate==0.27.2
pip install tf-keras 'numpy<2' 'fsspec<=2025.10.0'
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Generate data

```bash
python data/generate_data.py
```

### 4. Train

Run all four training datasets for a grammar (each takes ~30-60 min on an A10):

```bash
python training/train.py --config training/configs/grammar1_examples.yaml
python training/train.py --config training/configs/grammar1_hyperdata_1pct.yaml
python training/train.py --config training/configs/grammar1_hyperdata_5pct.yaml
python training/train.py --config training/configs/grammar1_hyperdata_10pct.yaml
```

Or to run them back-to-back unattended:

```bash
for cfg in training/configs/grammar1_*.yaml; do
  python training/train.py --config "$cfg"
done
```

### 5. Evaluate

```bash
python eval/eval.py --model checkpoints/pythia-1.4b_grammar1_examples/final --grammar grammar1
python eval/eval.py --model checkpoints/pythia-1.4b_grammar1_hyperdata_1pct/final --grammar grammar1
python eval/eval.py --model checkpoints/pythia-1.4b_grammar1_hyperdata_5pct/final --grammar grammar1
python eval/eval.py --model checkpoints/pythia-1.4b_grammar1_hyperdata_10pct/final --grammar grammar1
```

Results are saved to `results/`.

### 6. Copy results back

From your local machine:

```bash
scp -r ubuntu@<instance-ip>:~/hyperdata/results/ results/
```

## Configuration Options

Key parameters in training configs:

```yaml
model: "EleutherAI/pythia-1.4b"  # Base model
corpus: "data/corpora/grammar1_hyperdata_5pct.jsonl"
mix_ratio: 0.1          # 10% synthetic, 90% canonical
max_steps: 50000        # Training steps
learning_rate: 1.0e-5
batch_size: 4
gradient_accumulation_steps: 8
```

## Extending the Experiment

### Add a new grammar

1. Create `data/grammars/grammar4.py` with:
   - `is_valid(sentence)` function
   - `generate_valid(n, seed)` function
   - `generate_invalid(n, seed)` function
   - `get_explanation_text()` function

2. Add to `data/grammars/__init__.py`

3. Add to `GRAMMARS` dict in `data/generate_data.py`

4. Create training configs in `training/configs/`

### Change explanation ratio

Edit the `EXPLANATION_RATIOS` list in `data/generate_data.py`:

```python
EXPLANATION_RATIOS = [0.01, 0.05, 0.10, 0.20]  # Add 20%
```

## License

MIT
