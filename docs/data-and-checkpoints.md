# Data and checkpoints

[Back to README](../README.md)

## Availability

This checkout includes the scripts, split JSON files, and Stage-1 normalization statistics. Processed graph datasets (`.pkl`) and model checkpoints (`.pt`) are excluded by [.gitignore](../.gitignore).

No dataset/model archive URL is currently recorded in the repository. Obtain the compatible artifacts from the project maintainer before running training or evaluation. A clone alone does not supply them.

The current entry points load precomputed graphs. They do not extract embeddings or structures from raw sequences, and a raw FASTA/CSV input cannot replace the graph pickle.

## Expected layout

These are **external artifact paths**, not files supplied by Git:

~~~text
stage1/
├── data/
│   └── unified_stage1_graphs.pkl
└── results/
    └── best_stage1.pt
stage2/
├── processed_graphs_shared.pkl
└── checkpoints/
    ├── ova_<sanitized-class-1>/
    │   └── best_multimodal_ova.pt
    └── ova_<sanitized-class-N>/
        └── best_multimodal_ova.pt
~~~

Stage-2 class names are read from `payload["classes"]`. Directory naming follows `sanitize_class_name` in [train_multimodal_ova.py](../stage2/train_multimodal_ova.py): leading/trailing whitespace is stripped and runs of characters outside `A-Z a-z 0-9 . _ -` become `_`. Use the class order in the dataset rather than creating a new alphabetical order.

All four entry points accept explicit artifact paths; datasets do not have to be copied into the checkout. See [evaluation of retrained models](reproducibility.md#evaluate-retrained-checkpoints).

## Dataset contract

The serialized object is a dictionary with `data` containing a list of PyTorch Geometric `Data` objects. Stage 2 also requires an ordered `classes` list.

| Field | Expected meaning / shape |
|---|---|
| `x` | Per-residue features; the released Stage-1 trainer checks `[number_of_nodes, 508]`: 480 ESM-derived features and 28 structural features |
| `edge_index` | Graph connectivity, `[2, number_of_edges]` |
| `pos` | Per-node 3D coordinates used by Stage-1 edge-aware/residual variants to compute distance features |
| `edge_attr` | Six edge features consumed by the applicable normalized Stage-2 edge/residual variants |
| `seq_feat` | Sequence-level embedding with 480 values, stored with a leading graph dimension (`[1, 480]`) for batching |
| `global_feat` | Four global feature values |
| `y` | One binary AMP label per Stage-1 graph; ordered multi-hot activity labels for Stage 2 |
| `sample_id` | Stable identifier used to align Stage-1 graphs with Stage-2 labels in cascade evaluation |

These are contracts inferred from the loaders and model implementations, not a feature-extraction recipe. The selected model determines which structural/edge fields are consumed.

Stage-2 payloads should retain `shared_input_fingerprint_sha256`. The evaluator checks this against the split metadata and, when supplied by the checkpoint, checkpoint metadata.

## Tracked split and normalization files

| File | Training indices | Validation indices | Test indices |
|---|---:|---:|---:|
| [Stage-1 split](../stage1/data/legacy_matched_split.json) | 26,182 | 5,610 | 5,612 |
| [Stage-2 split](../stage2/splits/shared_stage2_seed42_split.json) | 4,332 | 928 | 929 |

These counts describe the checked-in index lists. They are not a claim that the lists are interchangeable, homology-filtered, or applicable to a differently ordered dataset.

The Stage-2 split has seed 42, declares 6,189 samples, and records the shared-input fingerprint:

~~~text
652675020a5277d02e8102a01b2c3f71b0cc20d441bf44908edafcb9ad9fd2ce
~~~

[Stage-1 normalization statistics](../stage1/data/normalization_stats.json) describe the 26,182 training records and include a digest of the Stage-1 split file. The Stage-1 loader rejects existing statistics associated with a different split.

For normalized nested Stage-2 variants, the trainer loads or computes statistics with a digest of the training indices. The default output is `normalization_stats.json` inside that run's output directory, unless `--stats-file` is supplied.

## Checkpoint contract

### Stage 1

The evaluator reconstructs a Stage-1 model from its checkpoint. Use a `best_stage1.pt` produced by the compatible training code, retaining:

- `model_state` and `model_config`;
- `normalization_stats`;
- the selected scalar `threshold`.

### Stage 2

Each activity needs its own `best_multimodal_ova.pt`. Checkpoint reconstruction uses the model state, feature variant, dimensions, dropout settings, and saved threshold(s). Normalized nested models also retain their model-family identifier and normalization statistics.

The Stage-2 evaluator checks `class_name` against the dataset label and checks the dataset fingerprint when it is present in the checkpoint. A state dictionary without the reconstruction metadata is not an equivalent replacement.

## Compatibility

Keep the following matched:

1. **Dataset ordering and split indices.** Do not shuffle a pickle and reuse the original positional split.
2. **Label order and checkpoint names.** Checkpoints are assigned to the dataset's ordered activity list.
3. **Feature schema and selected model.** A model trained on a different representation is not made compatible by renaming its checkpoint.
4. **Normalization statistics and training split.** Recompute statistics for a new split using a separate statistics/output path.
5. **Stage-1 and Stage-2 sample identifiers.** The cascade retrieves functional targets using these identifiers.

The cascade evaluator supports optional `--stage2-inference-dataset` inputs. It aligns these graphs to canonical Stage-2 IDs where available, or uses audited canonical order when explicit IDs are absent; passed peptides missing from that file fall back to Stage-1 graphs. Its legacy tensor-only adaptation does not retain all edge fields needed by normalized edge/residual variants. See the [evaluation notes](reproducibility.md#evaluation-protocols) before using that option.
