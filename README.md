# GraphActiAMP

**Graph-based antimicrobial peptide identification and activity classification**

GraphActiAMP implements a two-stage learning workflow in **PyTorch** and **PyTorch Geometric**:

1. **Stage 1 — AMP identification:** distinguish antimicrobial peptides (AMPs) from non-AMPs.
2. **Stage 2 — Activity classification:** apply one-vs-all (OvA) classifiers to assign activity labels to peptides that pass the Stage-1 gate.

The models combine ESM-derived residue features with graph representations. Structural, sequence-level, and global features are available through the implemented feature variants. Evaluation separates Stage-2 performance from the performance of the complete cascade.

**Artifact availability:** this repository includes the training/evaluation code, fixed split files, and Stage-1 normalization statistics. The processed graph datasets and pretrained checkpoints are **not tracked in Git**. Obtain compatible artifacts before training or evaluation; their required paths and formats are documented in [Data and checkpoints](docs/data-and-checkpoints.md). A dataset/model download URL has not yet been published in this repository.

## Contents

- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Data and pretrained checkpoints](#data-and-pretrained-checkpoints)
- [Evaluation](#evaluation)
- [Training](#training)
- [Outputs and metrics](#outputs-and-metrics)
- [Reproducibility and support](#reproducibility-and-support)

## Features

- **Two-stage prediction:** an AMP gate followed by independent activity classifiers, allowing multiple activity labels per peptide.
- **Graph-attention models:** ESM-based graph encoders with optional structural and edge-aware components.
- **Feature comparisons:** legacy fusion models and normalized nested/residual variants are available; see the [variant guide](docs/reproducibility.md#feature-variants).
- **Imbalance-aware training:** Stage-2 options include weighted sampling, class weighting, and asymmetric or binary cross-entropy loss.
- **Validation-selected thresholds:** model checkpoints retain the decision thresholds used during evaluation.
- **Auditable evaluation:** fixed split files, checkpoint metadata checks, per-class metrics, and cascade-level prediction exports.

## Project structure

~~~text
GraphActiAMP/
├── README.md
├── requirements.txt                     # Recorded package versions
├── environment.yml                      # Conda environment using requirements.txt
├── docs/
│   ├── data-and-checkpoints.md           # Required artifacts and feature schema
│   └── reproducibility.md               # Training recipes, variants, evaluation details
├── stage1/
│   ├── train_stage1.py                   # AMP identification and ablations
│   └── data/
│       ├── legacy_matched_split.json     # Fixed train/validation/test indices
│       └── normalization_stats.json      # Training-split normalization statistics
└── stage2/
    ├── train_multimodal_ova.py           # One-vs-all activity training
    ├── stage2_feature_models.py          # Normalized feature/residual models
    ├── evaluate_stage2_checkpoints.py    # Held-out Stage-2 evaluation
    ├── evaluate_two_stage_cascade.py     # Conditional and end-to-end evaluation
    └── splits/
        └── shared_stage2_seed42_split.json
~~~

The tree above lists repository files. Dataset and checkpoint placement is shown separately in the [artifact guide](docs/data-and-checkpoints.md#expected-layout).

## Installation

Run all commands from the repository root.

~~~bash
git clone https://github.com/eMAYQQQ/GraphActiAMP.git
cd GraphActiAMP
~~~

### Conda

~~~bash
conda env create -f environment.yml
conda activate graphactiamp
~~~

### Existing Python environment

The dependency file records versions from the original **Python 3.10.12** training environments, including PyTorch 2.7.1 and PyTorch Geometric 2.7.0. Use a compatible Python 3.10 environment.

~~~bash
python -m pip install -r requirements.txt
~~~

Use a PyTorch 2.7.1 build appropriate for the machine's CPU or CUDA setup. The Conda file delegates package installation to the same requirements file; it does not select a particular CUDA build.

After installation, inspect the environment:

~~~bash
python -c "import torch, torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available())"
~~~

## Data and pretrained checkpoints

Training consumes **precomputed PyTorch Geometric graphs**, not raw FASTA or CSV files. This repository does not currently include a raw-sequence feature-generation entry point.

The default evaluation layout requires:

| Artifact | Expected location |
|---|---|
| Stage-1 graphs | `stage1/data/unified_stage1_graphs.pkl` |
| Stage-1 checkpoint | `stage1/results/best_stage1.pt` |
| Stage-2 graphs and ordered labels | `stage2/processed_graphs_shared.pkl` |
| Stage-2 checkpoint for each activity | `stage2/checkpoints/ova_<class>/best_multimodal_ova.pt` |

The activity names and their order come from the Stage-2 dataset's `classes` field. Checkpoint directory names use the sanitization implemented by the training script.

See [Data and checkpoints](docs/data-and-checkpoints.md) for the graph fields, fingerprint requirements, split sizes, and use of artifacts in another directory.

## Evaluation

### Evaluate the Stage-2 checkpoints

With compatible graph data and a checkpoint for every activity in place:

~~~bash
python stage2/evaluate_stage2_checkpoints.py --device cpu
~~~

This writes `outputs/stage2_metrics.json`, including per-class thresholds, balanced accuracy, sensitivity, specificity, and geometric mean. It evaluates Stage 2 on its own held-out split.

### Evaluate the two-stage cascade

~~~bash
python stage2/evaluate_two_stage_cascade.py \
  --output-dir outputs/cascade --device -1
~~~

This evaluates the AMP gate and subsequent activity predictions on the Stage-1 mixed test set. It reports conditional activity metrics, end-to-end metrics including missed AMPs, and false-positive non-AMP counts.

**Device arguments differ between scripts:**

| Script | CPU | GPU |
|---|---|---|
| `stage1/train_stage1.py` | `--device cpu` | `--device cuda` |
| `stage2/train_multimodal_ova.py` | `--device -1` | `--device 0` |
| `stage2/evaluate_stage2_checkpoints.py` | `--device cpu` | `--device cuda:0` |
| `stage2/evaluate_two_stage_cascade.py` | `--device -1` | `--device 0` |

The [reproducibility guide](docs/reproducibility.md#evaluation-protocols) explains the evaluation populations and the optional Stage-2 inference dataset.

## Training

### Stage 1: ESM graph backbone and residual model

First train the backbone:

~~~bash
python stage1/train_stage1.py \
  --variant esm_only --seed 42 --epochs 18 --patience 5 --device cpu \
  --warmup-epochs 2 --output-dir outputs/retrain_stage1_esm
~~~

Then initialize the residual model from that checkpoint:

~~~bash
python stage1/train_stage1.py \
  --variant full_residual --seed 42 --epochs 12 --patience 4 --device cpu \
  --warmup-epochs 1 --init-checkpoint outputs/retrain_stage1_esm/best_stage1.pt \
  --freeze-backbone --output-dir outputs/retrain_stage1_final
~~~

### Stage 2: one classifier per activity

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

For CPU training, use `--device -1` and omit `--amp`. In Stage 2, the legacy `esm_only` setting limits the **graph node input** to ESM features; its sequence/global fusion branches remain active. It is not the same feature setting as Stage-1 `esm_only`.

The examples retain the existing training recipe. See [Reproducibility](docs/reproducibility.md) for additional variants and commands that evaluate these newly trained checkpoints directly.

## Outputs and metrics

| Workflow | Main outputs |
|---|---|
| Stage-1 training | `best_stage1.pt`, `last_stage1.pt`, `run_config.json`, `metrics.json`, `test_predictions.csv`, training curves |
| Stage-2 training | `ova_<class>/best_multimodal_ova.pt`, `ova_results_summary.csv`, `ova_per_class_metrics.csv`, class reports, progress plots |
| Stage-2 evaluation | `outputs/stage2_metrics.json` by default |
| Cascade evaluation | `cascade_metrics.json`, `cascade_per_class.csv`, `cascade_predictions.csv`, `cascade_summary.png` and `.pdf` |

Training outputs are written beneath the selected `--output-dir`. Retrained checkpoints are not automatically copied to the evaluators' default locations.

## Reproducibility and support

- Keep dataset ordering, split indices, label ordering, normalization statistics, and checkpoints matched. See the [artifact compatibility notes](docs/data-and-checkpoints.md#compatibility).
- Record the commit, feature variant, seed, split, and saved thresholds when reporting results.
- Follow [Reproducibility](docs/reproducibility.md) for training-to-evaluation commands and interpretation of the metrics.
- For usage questions or problems, open a [GitHub issue](https://github.com/eMAYQQQ/GraphActiAMP/issues) with the command, dependency versions, and error output.
