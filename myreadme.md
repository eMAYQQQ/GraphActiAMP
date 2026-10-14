# Local source-file map

The public-facing instructions are in `README.md`. This map records where
every copied file came from in the original project. `/home/chx/AMPClass` and
`/data3/chx/AMPClass` resolve to the same project on the preparation machine.
In this release package, the original `newstage1/` and `ClTrain/` directories were renamed to
`stage1/` and `stage2/`. The selected Stage-2 checkpoint directories were
moved from `ClTrain/shared_feature_experiments/esm_only_seed42_exact/` to
`stage2/checkpoints/`. The Stage-1 training script, results directory,
normalization-statistics file, and final checkpoint basename were also
renamed. The original paths remain in the right-hand column below. The copied
`stage1/train_stage1.py` also has a portability-only
`--device` option, so the original CPU run can be reproduced on a machine
that has a GPU. `stage2/evaluate_stage2_checkpoints.py`, `README.md`,
`requirements.txt`, and `.gitignore` are new files created for this release.

| File in `githubversion/` | Original source file |
|---|---|
| `stage1/train_stage1.py` | `/home/chx/AMPClass/newstage1/train_stage1_v2.py` |
| `stage2/stage2_feature_models.py` | `/home/chx/AMPClass/ClTrain/stage2_feature_models.py` |
| `stage2/train_multimodal_ova.py` | `/home/chx/AMPClass/ClTrain/train_multimodal_ova.py` |
| `stage2/evaluate_two_stage_cascade.py` | `/home/chx/AMPClass/ClTrain/evaluate_two_stage_cascade.py` |
| `stage1/data/unified_stage1_graphs.pkl` | `/home/chx/AMPClass/newstage1/data/unified_stage1_graphs.pkl` |
| `stage1/data/legacy_matched_split.json` | `/home/chx/AMPClass/newstage1/data/legacy_matched_split.json` |
| `stage1/data/normalization_stats.json` | `/home/chx/AMPClass/newstage1/data/v2_normalization_stats.json` |
| `stage2/processed_graphs_shared.pkl` | `/home/chx/AMPClass/ClTrain/processed_graphs_shared.pkl` |
| `stage2/splits/shared_stage2_seed42_split.json` | `/home/chx/AMPClass/ClTrain/splits/shared_stage2_seed42_split.json` |
| `stage1/results/best_stage1.pt` | `/home/chx/AMPClass/newstage1/v2_results/best_stage1_v2.pt` |

For each of the six labels below,
`stage2/checkpoints/ova_<label>/best_multimodal_ova.pt` was copied from
`/home/chx/AMPClass/ClTrain/shared_feature_experiments/esm_only_seed42_exact/ova_<label>/best_multimodal_ova.pt`:

`anti-cancer`, `anti-fungal`, `anti-gram-negative`, `anti-gram-positive`,
`anti-parasites`, `anti-viruses`.

The original `shared_multimodal_features.py`, dataset builders, audit scripts,
fixed-split *generator*, ablation runners, and summarizers were deliberately
not copied. The saved split files and processed datasets were copied instead.
