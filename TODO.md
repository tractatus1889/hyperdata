# TODO

## Grammar 2

### Train
```bash
python training/train.py --config training/configs/grammar2_examples.yaml
python training/train.py --config training/configs/grammar2_hyperdata_1pct.yaml
python training/train.py --config training/configs/grammar2_hyperdata_5pct.yaml
python training/train.py --config training/configs/grammar2_hyperdata_10pct.yaml
```

### Eval
```bash
python eval/eval.py --model checkpoints/pythia-1.4b_grammar2_examples/final --grammar grammar2
python eval/eval.py --model checkpoints/pythia-1.4b_grammar2_hyperdata_1pct/final --grammar grammar2
python eval/eval.py --model checkpoints/pythia-1.4b_grammar2_hyperdata_5pct/final --grammar grammar2
python eval/eval.py --model checkpoints/pythia-1.4b_grammar2_hyperdata_10pct/final --grammar grammar2
```

## Grammar 3

### Train
```bash
python training/train.py --config training/configs/grammar3_examples.yaml
python training/train.py --config training/configs/grammar3_hyperdata_1pct.yaml
python training/train.py --config training/configs/grammar3_hyperdata_5pct.yaml
python training/train.py --config training/configs/grammar3_hyperdata_10pct.yaml
```

### Eval
```bash
python eval/eval.py --model checkpoints/pythia-1.4b_grammar3_examples/final --grammar grammar3
python eval/eval.py --model checkpoints/pythia-1.4b_grammar3_hyperdata_1pct/final --grammar grammar3
python eval/eval.py --model checkpoints/pythia-1.4b_grammar3_hyperdata_5pct/final --grammar grammar3
python eval/eval.py --model checkpoints/pythia-1.4b_grammar3_hyperdata_10pct/final --grammar grammar3
```
