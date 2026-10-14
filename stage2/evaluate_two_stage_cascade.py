#!/usr/bin/env python3
"""Evaluate conditional Stage-2 and end-to-end hierarchical predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "stage1", ROOT / "stage2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_multimodal_ova import (  # noqa: E402
    FusionClassifier,
    GraphDataset,
    sanitize_class_name,
)
from stage2_feature_models import model_from_stage2_checkpoint  # noqa: E402
from train_stage1 import model_from_checkpoint  # noqa: E402


DEFAULT_STAGE1_DATASET = ROOT / "stage1" / "data" / "unified_stage1_graphs.pkl"
DEFAULT_STAGE1_SPLIT = ROOT / "stage1" / "data" / "legacy_matched_split.json"
DEFAULT_STAGE1_CHECKPOINT = ROOT / "stage1" / "results" / "best_stage1.pt"
DEFAULT_STAGE2_DATASET = ROOT / "stage2" / "processed_graphs_shared.pkl"


def binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    targets = targets.astype(int)
    predictions = predictions.astype(int)
    tp = int(((targets == 1) & (predictions == 1)).sum())
    tn = int(((targets == 0) & (predictions == 0)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy_score(targets, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "gmean": math.sqrt(sensitivity * specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def multilabel_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    if len(targets) == 0:
        return {key: None for key in (
            "samples", "micro_precision", "micro_recall", "micro_f1",
            "macro_precision", "macro_recall", "macro_f1", "exact_match", "hamming_loss",
        )}
    micro = precision_recall_fscore_support(
        targets, predictions, average="micro", zero_division=0
    )
    macro = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return {
        "samples": len(targets),
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "micro_f1": micro[2],
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "exact_match": accuracy_score(targets, predictions),
        "hamming_loss": hamming_loss(targets, predictions),
    }


def load_stage2_labels(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    classes = list(payload["classes"])
    sample_ids = [str(graph.sample_id) for graph in payload["data"]]
    labels = {
        str(graph.sample_id): graph.y.view(-1).detach().cpu().numpy().astype(int)
        for graph in payload["data"]
    }
    fingerprint = payload.get("shared_input_fingerprint_sha256")
    return classes, labels, fingerprint, sample_ids


def load_stage2_inference_graphs(path: Path, canonical_sample_ids: list[str]):
    """Map a Stage-2-only graph file onto canonical IDs, using audited order if needed."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    graphs = payload["data"]
    if len(graphs) != len(canonical_sample_ids):
        raise ValueError(
            f"Stage-2 inference graph count mismatch: {len(graphs)} != {len(canonical_sample_ids)}"
        )
    explicit_ids = [getattr(graph, "sample_id", None) for graph in graphs]
    if all(value is not None for value in explicit_ids):
        mapping = {str(sample_id): graph for sample_id, graph in zip(explicit_ids, graphs)}
        mapping_method = "explicit_sample_id"
    else:
        mapping = dict(zip(canonical_sample_ids, graphs))
        mapping_method = "canonical_stage2_order"
    return mapping, {
        "path": str(path.resolve()),
        "samples": len(mapping),
        "mapping_method": mapping_method,
        "feature_schema": payload.get("feature_schema", "legacy_unspecified"),
        "fingerprint": payload.get("shared_input_fingerprint_sha256"),
    }


def predict_stage1(model, loader, device):
    probabilities, targets, sample_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage 1 gate", unit="batch"):
            batch = batch.to(device)
            probabilities.append(torch.sigmoid(model(batch)).view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
            sample_ids.extend([str(value) for value in batch.sample_id])
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(targets).numpy().astype(int),
        sample_ids,
    )


def load_stage2_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_variant = checkpoint.get("feature_variant", "node_full")
    if checkpoint.get("model_family") == "normalized_nested_residual":
        model = model_from_stage2_checkpoint(checkpoint, device)
    else:
        model = FusionClassifier(
            checkpoint["feature_dim"],
            checkpoint["seq_dim"],
            checkpoint["global_dim"],
            checkpoint["hidden_dim"],
            1,
            checkpoint["dropout"],
            feature_variant=feature_variant,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(np.asarray(checkpoint.get("thresholds", [0.5])).reshape(-1)[0])
    return model, threshold, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dataset", type=Path, default=DEFAULT_STAGE1_DATASET)
    parser.add_argument("--stage1-split", type=Path, default=DEFAULT_STAGE1_SPLIT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--stage2-dataset", type=Path, default=DEFAULT_STAGE2_DATASET)
    parser.add_argument(
        "--stage2-inference-dataset",
        type=Path,
        default=None,
        help=(
            "Optional Stage-2-only input graphs. AMP graphs are mapped by sample_id or "
            "the audited canonical order; passed samples absent from this file use Stage-1 graphs."
        ),
    )
    parser.add_argument("--stage2-results", type=Path, default=ROOT / "stage2" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    classes, label_by_id, shared_fingerprint, canonical_stage2_ids = load_stage2_labels(
        args.stage2_dataset
    )
    with args.stage1_dataset.open("rb") as handle:
        stage1_payload = pickle.load(handle)
    graphs = stage1_payload["data"]
    test_indices = json.loads(args.stage1_split.read_text(encoding="utf-8"))["test"]
    test_loader = DataLoader(
        Subset(GraphDataset(graphs), test_indices), batch_size=args.batch_size, shuffle=False
    )

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location=device, weights_only=False
    )
    stage1_model = model_from_checkpoint(stage1_checkpoint, device=device)
    stage1_threshold = float(stage1_checkpoint["threshold"])
    stage1_probabilities, stage1_targets, sample_ids = predict_stage1(
        stage1_model, test_loader, device
    )
    stage1_predictions = (stage1_probabilities >= stage1_threshold).astype(int)

    passed_positions = np.flatnonzero(stage1_predictions == 1)
    passed_dataset_indices = [test_indices[position] for position in passed_positions]
    inference_input = {
        "path": str(args.stage1_dataset.resolve()),
        "mapping_method": "shared_stage1_graphs",
        "exact_stage2_graphs": 0,
        "stage1_graph_fallbacks": len(passed_dataset_indices),
    }
    if args.stage2_inference_dataset is None:
        passed_graphs = [graphs[index] for index in passed_dataset_indices]
    else:
        stage2_graph_by_id, inference_input = load_stage2_inference_graphs(
            args.stage2_inference_dataset, canonical_stage2_ids
        )
        passed_graphs = []
        exact_graphs = 0
        for index in passed_dataset_indices:
            stage1_graph = graphs[index]
            sample_id = str(stage1_graph.sample_id)
            if sample_id in stage2_graph_by_id:
                source_graph = stage2_graph_by_id[sample_id]
                exact_graphs += 1
            else:
                source_graph = stage1_graph
            # Legacy graphs have inconsistent metadata and y widths. Build the
            # minimal tensor-only view consumed by the legacy FusionClassifier.
            inference_graph = Data(
                x=source_graph.x,
                edge_index=source_graph.edge_index,
                seq_feat=source_graph.seq_feat,
                global_feat=source_graph.global_feat,
                y=torch.zeros((1, len(classes)), dtype=torch.float32),
            )
            passed_graphs.append(inference_graph)
        inference_input.update({
            "exact_stage2_graphs": exact_graphs,
            "stage1_graph_fallbacks": len(passed_graphs) - exact_graphs,
            "fallback_reason": "legacy Stage-2 data contains AMP samples only",
            "inference_y_collation_width": len(classes),
            "evaluation_labels_source": str(args.stage2_dataset.resolve()),
            "legacy_inference_tensor_fields": [
                "x", "edge_index", "seq_feat", "global_feat"
            ],
        })
    passed_loader = DataLoader(
        GraphDataset(passed_graphs), batch_size=args.batch_size, shuffle=False
    )
    probabilities = np.zeros((len(test_indices), len(classes)), dtype=np.float32)
    predictions = np.zeros((len(test_indices), len(classes)), dtype=int)
    thresholds = {}
    feature_variants = set()
    checkpoint_fingerprints = set()

    for class_index, class_name in enumerate(classes):
        checkpoint_path = (
            args.stage2_results / f"ova_{sanitize_class_name(class_name)}" / "best_multimodal_ova.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, threshold, checkpoint = load_stage2_model(checkpoint_path, device)
        if checkpoint.get("class_name") != class_name:
            raise ValueError(f"Class/checkpoint mismatch for {checkpoint_path}")
        feature_variants.add(checkpoint.get("feature_variant", "node_full"))
        checkpoint_fingerprints.add(checkpoint.get("shared_input_fingerprint_sha256"))
        thresholds[class_name] = threshold
        rows = []
        with torch.no_grad():
            for batch in tqdm(passed_loader, desc=f"Stage 2 {class_name}", unit="batch"):
                batch = batch.to(device)
                rows.append(torch.sigmoid(model(batch)).view(-1).cpu())
        class_probabilities = torch.cat(rows).numpy() if rows else np.empty(0, dtype=np.float32)
        probabilities[passed_positions, class_index] = class_probabilities
        predictions[passed_positions, class_index] = class_probabilities >= threshold
        del model

    if len(feature_variants) != 1:
        raise ValueError(f"Mixed Stage-2 feature variants: {feature_variants}")
    checkpoint_fingerprints.discard(None)
    if checkpoint_fingerprints and checkpoint_fingerprints != {shared_fingerprint}:
        raise ValueError("Stage-2 checkpoint/data fingerprint mismatch")

    targets = np.asarray(
        [label_by_id.get(sample_id, np.zeros(len(classes), dtype=int)) for sample_id in sample_ids]
    )
    true_amp = stage1_targets == 1
    detected_true_amp = true_amp & (stage1_predictions == 1)
    true_nonamp = ~true_amp

    metrics = {
        "stage1": binary_metrics(stage1_targets, stage1_predictions),
        "stage1_threshold": stage1_threshold,
        "stage2_feature_variant": next(iter(feature_variants)),
        "stage2_inference_input": inference_input,
        "stage2_thresholds": thresholds,
        "conditional_stage2_detected_true_amp": multilabel_metrics(
            targets[detected_true_amp], predictions[detected_true_amp]
        ),
        "end_to_end_mixed_test": multilabel_metrics(targets, predictions),
        "end_to_end_true_amp_including_gate_misses": multilabel_metrics(
            targets[true_amp], predictions[true_amp]
        ),
        "counts": {
            "mixed_test": len(targets),
            "true_amp": int(true_amp.sum()),
            "detected_true_amp": int(detected_true_amp.sum()),
            "missed_true_amp": int((true_amp & (stage1_predictions == 0)).sum()),
            "stage1_false_positive_nonamp": int((true_nonamp & (stage1_predictions == 1)).sum()),
            "nonamp_with_function_prediction": int(
                (true_nonamp & (predictions.sum(axis=1) > 0)).sum()
            ),
        },
    }

    per_class_rows = []
    for index, class_name in enumerate(classes):
        conditional_target = targets[detected_true_amp, index]
        conditional_prediction = predictions[detected_true_amp, index]
        per_class_rows.append({
            "class": class_name,
            "threshold": thresholds[class_name],
            "conditional_f1": f1_score(
                conditional_target, conditional_prediction, zero_division=0
            ),
            "conditional_balanced_accuracy": balanced_accuracy_score(
                conditional_target, conditional_prediction
            ),
            "end_to_end_f1": f1_score(
                targets[:, index], predictions[:, index], zero_division=0
            ),
            "end_to_end_balanced_accuracy": balanced_accuracy_score(
                targets[:, index], predictions[:, index]
            ),
        })
    with (output_dir / "cascade_per_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    prediction_rows = []
    for row_index, sample_id in enumerate(sample_ids):
        row = {
            "sample_id": sample_id,
            "stage1_target": int(stage1_targets[row_index]),
            "stage1_probability": float(stage1_probabilities[row_index]),
            "stage1_prediction": int(stage1_predictions[row_index]),
        }
        for class_index, class_name in enumerate(classes):
            row[f"target_{class_name}"] = int(targets[row_index, class_index])
            row[f"probability_{class_name}"] = float(probabilities[row_index, class_index])
            row[f"prediction_{class_name}"] = int(predictions[row_index, class_index])
        prediction_rows.append(row)
    with (output_dir / "cascade_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    (output_dir / "cascade_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    confusion = np.array([
        [metrics["stage1"]["tn"], metrics["stage1"]["fp"]],
        [metrics["stage1"]["fn"], metrics["stage1"]["tp"]],
    ])
    axes[0].imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center")
    axes[0].set_xticks([0, 1], ["non-AMP", "AMP"])
    axes[0].set_yticks([0, 1], ["non-AMP", "AMP"])
    axes[0].set_title("Stage 1 gate")

    metric_names = ["micro_f1", "macro_f1", "exact_match"]
    x = np.arange(len(metric_names))
    conditional = metrics["conditional_stage2_detected_true_amp"]
    end_to_end = metrics["end_to_end_true_amp_including_gate_misses"]
    axes[1].bar(x - 0.18, [conditional[name] for name in metric_names], 0.36, label="Conditional")
    axes[1].bar(
        x + 0.18,
        [end_to_end[name] for name in metric_names],
        0.36,
        label="End-to-end true AMP",
    )
    axes[1].set_xticks(x, ["Micro F1", "Macro F1", "Exact"])
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Functional prediction")

    count_names = ["True AMP", "Detected AMP", "Missed AMP", "False-positive non-AMP"]
    count_values = [
        metrics["counts"]["true_amp"],
        metrics["counts"]["detected_true_amp"],
        metrics["counts"]["missed_true_amp"],
        metrics["counts"]["stage1_false_positive_nonamp"],
    ]
    axes[2].barh(np.arange(4), count_values, color=["#2878B5", "#2A9D8F", "#E9C46A", "#E76F51"])
    axes[2].set_yticks(np.arange(4), count_names)
    axes[2].invert_yaxis()
    axes[2].set_title("Cascade flow")
    for index, value in enumerate(count_values):
        axes[2].text(value, index, f" {value}", va="center", fontsize=8)

    for axis in axes[1:]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("AMPClass two-stage cascade evaluation", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "cascade_summary.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "cascade_summary.pdf", bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
