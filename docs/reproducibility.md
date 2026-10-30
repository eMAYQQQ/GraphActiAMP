# Reproducibility guide

[Back to README](../README.md) · [Data and checkpoints](data-and-checkpoints.md)

## Before running

1. Install the dependencies in [requirements.txt](../requirements.txt).
2. Obtain compatible precomputed graphs and, for pretrained evaluation, checkpoints.
3. Run commands from the repository root and use the tracked splits with the matching dataset ordering.
4. Keep each run's configuration, thresholds, output files, and code revision together.

The examples below preserve the recipe previously listed in the README. They do not assert new performance results or imply that every feature variant was trained with identical settings.

## Feature variants

### Stage 1

The `--variant` options are:

| Variant | Inputs / behavior |
|---|---|
| `esm_only` | ESM node features and graph topology |
| `node_struct` | ESM and structural node features |
| `edge_3d` | ESM/structural node features with distance-derived edge features |
| `full_multimodal` | Graph, distance-derived edges, sequence embedding, and global features |
| `full_residual` | ESM graph backbone with an auxiliary structure/edge/sequence/global residual pathway |

The README trains an `esm_only` backbone, then initializes `full_residual` from it using `--init-checkpoint` and `--freeze-backbone`. Additional `--ablate-*` flags are available in [train_stage1.py](../stage1/train_stage1.py).

### Stage 2

The `--feature-variant` options belong to two implementations:

| Family | Variants | Interpretation |
|---|---|---|
| Legacy fusion | `esm_only`, `node_full` | Concatenates the graph representation, sequence branch, and global features; the variant selects the graph node feature width |
| Normalized nested/residual | `sequence_global`, `esm_graph`, `esm_edge`, `full_residual` | Starts from sequence/global features and optionally adds an ESM graph, edge features, and structural correction |

In particular, Stage-2 legacy `esm_only` **still uses the sequence and global branches**. Its name should not be interpreted as the same ablation as Stage-1 `esm_only`. Use checkpoint metadata to identify a model rather than relying on its directory name.

## Training recipe

### Stage 1

~~~bash
python stage1/train_stage1.py \
  --variant esm_only --seed 42 --epochs 18 --patience 5 --device cpu \
  --warmup-epochs 2 --output-dir outputs/retrain_stage1_esm

python stage1/train_stage1.py \
  --variant full_residual --seed 42 --epochs 12 --patience 4 --device cpu \
  --warmup-epochs 1 --init-checkpoint outputs/retrain_stage1_esm/best_stage1.pt \
  --freeze-backbone --output-dir outputs/retrain_stage1_final
~~~

Both runs use the default Stage-1 dataset, split, and statistics paths. Use `--dataset`, `--split`, and `--stats` to select another compatible set. The Stage-1 device argument accepts `auto`, `cpu`, or `cuda`.

### Stage 2

~~~bash
python stage2/train_multimodal_ova.py \
  --split-file stage2/splits/shared_stage2_seed42_split.json \
  --feature-variant esm_only --seed 42 --epochs 100 --batch-size 16 \
  --hidden-dim 384 --dropout 0.35 --lr 0.0002 \
  --sampler weighted --no-asl --minority-boost 3.0 \
  --minority-threshold 0.5 --threshold-grid-start 0.02 \
  --threshold-grid-end 0.85 --threshold-grid-steps 84 \
  --threshold-metric balance --target-sen-spec-gap 0.1 \
  --amp --device 0 --output-dir outputs/retrain_stage2
~~~

This uses weighted sampling and explicitly selects binary cross-entropy with `--no-asl`. The trainer otherwise enables asymmetric loss by default. The threshold grid and `balance` criterion are applied to validation predictions; the resulting thresholds are saved in the best checkpoints.

For a CPU run, use `--device -1` and omit `--amp`. `--include-classes` can select activities using the exact names from the dataset; omit it to train every class.

## Evaluate retrained checkpoints

Evaluators do not automatically discover new training directories. Point them to the outputs above:

~~~bash
python stage2/evaluate_stage2_checkpoints.py \
  --checkpoints outputs/retrain_stage2 \
  --output outputs/retrain_stage2_metrics.json \
  --device cpu

python stage2/evaluate_two_stage_cascade.py \
  --stage1-checkpoint outputs/retrain_stage1_final/best_stage1.pt \
  --stage2-results outputs/retrain_stage2 \
  --output-dir outputs/retrained_cascade \
  --device -1
~~~

For artifacts kept elsewhere, supply `--dataset` and `--split` to the Stage-2 evaluator. The cascade uses separate `--stage1-dataset`, `--stage1-split`, and `--stage2-dataset` arguments.

## Evaluation protocols

### Stage-2 held-out evaluation

[evaluate_stage2_checkpoints.py](../stage2/evaluate_stage2_checkpoints.py) evaluates every class checkpoint on the Stage-2 split's `test` indices and loads its saved threshold. The output includes per-class BA/SEN/SPEC/GMean and their mean across the classes. The current summary key is named `mean_across_six_classes`; label names themselves are taken from the dataset.

### Two-stage evaluation

[evaluate_two_stage_cascade.py](../stage2/evaluate_two_stage_cascade.py) starts from the Stage-1 mixed AMP/non-AMP test set:

- `conditional_stage2_detected_true_amp`: activity metrics only for true AMPs that passed the gate.
- `end_to_end_true_amp_including_gate_misses`: activity metrics for all true AMPs, including gate misses.
- `end_to_end_mixed_test`: activity metrics on the full mixed test set.
- `counts`: detected/missed true AMPs, false-positive non-AMPs, and non-AMPs receiving activity predictions.

The default cascade applies Stage-2 models to graphs from the Stage-1 dataset. The optional `--stage2-inference-dataset` path substitutes canonical Stage-2 graphs where available and reports the mapping and fallback counts under `stage2_inference_input`.

That optional path constructs a **legacy tensor-only graph view** and drops fields such as `edge_attr`. Use it only with a compatible checkpoint family; a normalized edge/residual model requires its additional graph fields. The default and optional input modes should not be treated as identical experiments.

Conditional activity scores can exceed end-to-end scores because the former exclude AMPs missed by the first gate. Report the evaluation population along with the metric.

## Interpreting outputs

| Metric | Meaning |
|---|---|
| BA | Balanced accuracy: mean of sensitivity and specificity |
| SEN | Sensitivity / positive-class recall |
| SPEC | Specificity / negative-class recall |
| GMean | Geometric mean of sensitivity and specificity |
| Micro / macro F1 | Multilabel summaries with different averaging conventions |
| Exact match | Fraction with the complete activity vector predicted correctly |
| Hamming loss | Fraction of individual activity decisions that are incorrect |

Stage-1 `metrics.json` and Stage-2/cascade summaries expose different fields; do not assume a metric appears in every output.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Missing `.pkl` or `.pt` file | Obtain the external artifact or supply its explicit path; these files are not part of a clone |
| Stage-2 split/data fingerprint mismatch | Use the dataset associated with the tracked split, retaining its fingerprint metadata |
| Normalization statistics belong to another split | Use matching statistics or a new `--stats` / `--stats-file` path |
| Class/checkpoint mismatch | Preserve dataset label order and the checkpoint's original `class_name` and directory mapping |
| CPU device argument rejected | Check the script-specific device table in the README |
| Missing edge attributes with an optional inference dataset | Check the checkpoint family and the legacy tensor-only adaptation described above |

For a reproducible report, include the commit, dependency versions, device, complete command, split identity, feature variant, and whether the cascade used the optional inference dataset.
