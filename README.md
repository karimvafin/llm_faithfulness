# llm_faithfulness

Faithfulness evaluation and DPO training for mediator-conditioned LLMs on PAUQ.

A "mediator" is an explicit intermediate representation between question and answer
(for PAUQ: SQL skeleton + schema_links + slots). Interventions perturb the mediator;
metrics measure whether the model's answer stays consistent with the perturbed
mediator. DPO data uses the same intervention machinery to build (chosen, rejected)
pairs that force the model to align its mediator with its final answer.

## Install

```bash
pip install -e .
python -c "import nltk; nltk.download('punkt_tab')"
```

## Data layout

Place PAUQ files under `data/pauq/`:

```
data/pauq/
├── pauq_train.json
├── pauq_dev.json
└── tables.json
```

## Quickstart (4 commands)

```bash
# 1) Evaluate a model on dev
python -m scripts.evaluate \
  --model-name Qwen/Qwen3-4B \
  --data-path data/pauq --split dev \
  --mode structure_prediction \
  --intervention-level 3 \
  --batch-size 8 \
  --output runs/qwen3-4B__sp__lvl3.json

# 2) Generate DPO training data (chosen = gold, rejected = level-k-intervened mediator + same SQL)
python -m scripts.generate_dpo_data \
  --data-path data/pauq --split train \
  --intervention-level 3 \
  --output dpo_data/pauq_lvl3.jsonl

# 3) Train DPO (TRL DPOTrainer + LoRA)
python -m scripts.train_dpo \
  --model-name Qwen/Qwen3-4B \
  --train-file dpo_data/pauq_lvl3.jsonl \
  --output-dir checkpoints/qwen3-4B-dpo-lvl3

# 4) Plot
python -m scripts.plot \
  --reports runs/*.json \
  --kind faithfulness_vs_level \
  --output plots/faithfulness.png
```

## Metrics

- `faithfulness_id`: generated SQL is consistent with the predicted mediator (skeleton, slots, AND schema_links).
- `faithfulness_strong`: same check applied after intervening on the predicted mediator and regenerating. At level 0, equals `faithfulness_id`.
- `performance`: generated SQL is parse-tree-equivalent to the gold SQL (Spider-style evaluator, vendored in `llm_faithfulness/pauq/sql_eval/`).

## Intervention levels (PAUQ)

`level ∈ {0, 1, 2, 3, 4, 5}` controls the fraction of non-literal slots that get replaced.
`n_replace = (level * len(slots)) // 5`. A column slot is replaced with a different
column from the DB; a table slot with a different table; literal slots are left alone.
When the same value appears in multiple slot positions, all positions are replaced
together so the mediator stays self-consistent.

## Adding another dataset

1. Implement `Dataset` (`llm_faithfulness/protocols.py`) and `InterventionStrategy`.
2. Wire them into `scripts/evaluate.py` and `scripts/generate_dpo_data.py` behind a `--dataset` flag.
3. The pipeline, metrics, DPO, and plotting modules are dataset-agnostic and require no changes.
