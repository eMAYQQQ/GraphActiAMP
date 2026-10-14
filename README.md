# GraphActiAMP: selected-model reproducibility package

This package starts from **processed graph datasets**. It includes the selected
Stage-1 and Stage-2 training/model code, the fixed splits, the selected
checkpoints, Stage-1 normalization statistics, and an evaluator for the Stage-2
headline metrics. It does **not** include raw-data collection, feature
construction, ablation, baseline, or attribution scripts. Consequently it
supports reproduction from processed data, not reconstruction from raw FASTA or
PDB files and not every analysis figure in the article.

`myreadme.md` maps each copied artifact back to its source in the original
project. These absolute source paths are provenance notes, not paths required
to run this package.

## Files and data

| Item | Path | Purpose |
|---|---|---|
| Stage-1 processed graphs | `stage1/data/unified_stage1_graphs.pkl` | 37,404 AMP/non-AMP graphs; 508D nodes, 480D sequence vector, 4D global vector, 6D edge attributes |
| Stage-1 fixed split | `stage1/data/legacy_matched_split.json` | 26,182 train / 5,610 validation / 5,612 test indices |
| Stage-1 training statistics | `stage1/data/normalization_stats.json` | Computed from Stage-1 training indices only |
| Stage-2 processed graphs | `stage2/processed_graphs_shared.pkl` | 6,189 AMP graphs with six non-exclusive activity labels |
| Stage-2 fixed split | `stage2/splits/shared_stage2_seed42_split.json` | 4,332 train / 928 validation / 929 test indices |
| Selected checkpoints | `stage1/results/best_stage1.pt` and `stage2/checkpoints/ova_*/best_multimodal_ova.pt` | One selected Stage-1 model and six Stage-2 OvA models |

The six labels are stored in this exact order:
`anti-cancer`, `anti-fungal`, `anti-gram-negative`, `anti-gram-positive`,
`anti-parasites`, `anti-viruses`. The Stage-2 graph fingerprint is
`652675020a5277d02e8102a01b2c3f71b0cc20d441bf44908edafcb9ad9fd2ce`.
The Stage-2 evaluation script checks this against the fixed split and model
checkpoints. The selected Stage-2 model uses the `esm_only` feature variant,
so the optional nested-feature normalization statistics are not needed.

**Large-file warning:** the Stage-1 `.pkl` is about 2.6 GB and the Stage-2
`.pkl` about 426 MB. These are staged in this local folder, but are ignored by
Git via `.gitignore`. Thus, **uploading this code directory to GitHub as-is
will not include the data or model weights**. Before publishing, deposit the
two datasets and seven `.pt` checkpoints in a suitable external archive, add
permanent download links and checksums here, and cite that archive in the
paper. Do not push these files to ordinary GitHub Git history. Python pickle
and PyTorch checkpoints
must only be loaded from a trusted source. Confirm third-party data reuse
rights before public release.

## Environment

Original runs used Python 3.10.12, PyTorch 2.7.1+cu118,
PyTorch Geometric 2.7.0, NumPy 2.2.5, scikit-learn 1.7.2,
Matplotlib 3.10.7, and tqdm 4.67.1. The original Stage-1 run recorded CPU
execution; Stage-2 recorded CUDA mixed precision. Install an appropriate
PyTorch build first, then run `pip install -r requirements.txt`. CUDA vs CPU,
driver, and nondeterministic kernels can cause small retraining differences.

Run all commands below from the directory containing this README.

## Verify selected checkpoints on held-out data

Stage-2 **independent** test metrics (all 929 Stage-2 test AMPs; no Stage-1
gate) can be recomputed without retraining:

```bash
python stage2/evaluate_stage2_checkpoints.py --device cpu
```

The result is saved to `outputs/stage2_metrics.json`. Across the six classes,
the expected means, in the order BA / SEN / SPEC / GMean, are approximately
`0.7576 / 0.7179 / 0.7973 / 0.7532`. The script also saves per-class values
and the six thresholds to the output JSON.
On a compatible GPU, omit `--device cpu` to use CUDA mixed precision as in the
original Stage-2 run.

The Stage-1 held-out and **end-to-end cascade** metrics can be recomputed with:

```bash
python stage2/evaluate_two_stage_cascade.py \
  --stage1-dataset stage1/data/unified_stage1_graphs.pkl \
  --stage1-split stage1/data/legacy_matched_split.json \
  --stage1-checkpoint stage1/results/best_stage1.pt \
  --stage2-dataset stage2/processed_graphs_shared.pkl \
  --stage2-results stage2/checkpoints \
  --output-dir outputs/cascade --device -1
```

`outputs/cascade/cascade_metrics.json` should report Stage-1 balanced accuracy
near `0.9408` and GMean near `0.9405` on 5,612 test examples. The selected
cascade has mixed-test micro-F1 near `0.7416`.
The cascade and independent Stage-2 test numbers describe **different test
populations** and should not be compared as if they were the same metric.

## Retrain the selected models

First train the ESM-only Stage-1 initialization, then the frozen-backbone
`full_residual` model. The final published Stage-1 model was initialized from
the former; its original initialization checkpoint is not bundled because it
can be recreated by the first command. Training `full_residual` from scratch
is a different experiment.

```bash
python stage1/train_stage1.py \
  --dataset stage1/data/unified_stage1_graphs.pkl \
  --split stage1/data/legacy_matched_split.json \
  --stats stage1/data/normalization_stats.json \
  --variant esm_only --seed 42 --epochs 18 --patience 5 --device cpu \
  --warmup-epochs 2 --output-dir outputs/retrain_stage1_esm

python stage1/train_stage1.py \
  --dataset stage1/data/unified_stage1_graphs.pkl \
  --split stage1/data/legacy_matched_split.json \
  --stats stage1/data/normalization_stats.json \
  --variant full_residual --seed 42 --epochs 12 --patience 4 --device cpu \
  --warmup-epochs 1 --init-checkpoint outputs/retrain_stage1_esm/best_stage1.pt \
  --freeze-backbone --output-dir outputs/retrain_stage1_final
```

Both Stage-1 commands otherwise use the selected shared settings: hidden size
192, two layers, four heads, batch size 256, dropout 0.25, modality dropout
0.15, learning rate `3e-4`, weight decay `1e-4`, gradient clipping 2.0, and
square-root positive-class weighting. The selected settings are explicit in
these commands and in the training source; new run configuration files will
be written to the specified output directories.

For the six selected Stage-2 OvA models, use the fixed split. The original run
generated this exact `seed=42` random split internally; passing the saved
split file makes the indices explicit and independent of later code changes:

```bash
python stage2/train_multimodal_ova.py \
  --dataset stage2/processed_graphs_shared.pkl \
  --split-file stage2/splits/shared_stage2_seed42_split.json \
  --feature-variant esm_only --seed 42 --epochs 100 --batch-size 16 \
  --hidden-dim 384 --dropout 0.35 --lr 0.0002 \
  --sampler weighted --no-asl --minority-boost 3.0 \
  --minority-threshold 0.5 --threshold-grid-start 0.02 \
  --threshold-grid-end 0.85 --threshold-grid-steps 84 \
  --threshold-metric balance --target-sen-spec-gap 0.1 \
  --amp --device 0 --output-dir outputs/retrain_stage2
```

Use `--device -1` and omit `--amp` on CPU; this is useful for checking code but
may not reproduce the original GPU training numerically. The full selected
Stage-2 settings are given in the command above.
The six task-specific selected thresholds are embedded in the `.pt` files and
are applied by both evaluators.

## Scope and provenance

No raw-to-graph builder is included. The processed graphs contain ESM-2
residue/sequence representations, ESMFold-derived contact topology, structural
and physicochemical channels. For the exact raw-data sources, filtering,
deduplication, label definitions, and feature extraction details, consult the
paper methods and data availability statement. Some processed-data payloads retain original
source-path strings as provenance only; the scripts do not use those strings
to locate files. `myreadme.md` records the original file locations.

No license is asserted here: add a code license and a data reuse statement
after the authors confirm their rights and the target journal's policy.
