#!/usr/bin/env python3
"""Evaluate the six selected Stage-2 checkpoints on the fixed held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from evaluate_two_stage_cascade import load_stage2_model, sanitize_class_name
from train_multimodal_ova import (
    GraphDataset,
    compute_ba_sens_spec_gmean,
    load_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "stage2" / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "stage2" / "processed_graphs_shared.pkl")
    parser.add_argument("--split", type=Path, default=ROOT / "stage2" / "splits" / "shared_stage2_seed42_split.json")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage2_metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed-precision inference")
    args = parser.parse_args()

    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    graphs, classes, payload = load_dataset(args.dataset)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_indices = split["test"]
    if len(set(test_indices)) != len(test_indices) or max(test_indices) >= len(graphs):
        raise ValueError("Invalid Stage-2 test indices")
    if split.get("metadata", {}).get("shared_input_fingerprint_sha256") != payload.get(
        "shared_input_fingerprint_sha256"
    ):
        raise ValueError("Stage-2 split/data fingerprint mismatch")

    loader = DataLoader(Subset(GraphDataset(graphs), test_indices), batch_size=args.batch_size)
    per_class = {}
    for class_index, class_name in enumerate(classes):
        checkpoint_path = args.checkpoints / f"ova_{sanitize_class_name(class_name)}" / "best_multimodal_ova.pt"
        model, threshold, checkpoint = load_stage2_model(checkpoint_path, device)
        if checkpoint.get("class_name") != class_name:
            raise ValueError(f"Class/checkpoint mismatch: {checkpoint_path}")
        fingerprint = checkpoint.get("shared_input_fingerprint_sha256")
        if fingerprint is not None and fingerprint != payload.get("shared_input_fingerprint_sha256"):
            raise ValueError(f"Checkpoint/data fingerprint mismatch: {checkpoint_path}")
        targets, predictions = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and not args.no_amp):
                    probabilities = torch.sigmoid(model(batch).float()).view(-1)
                labels = batch.y.view(batch.num_graphs, -1)[:, class_index]
                targets.append(labels.cpu().numpy().astype(np.int32))
                predictions.append((probabilities >= threshold).cpu().numpy().astype(np.int32))
        target_array = np.concatenate(targets).reshape(-1, 1)
        prediction_array = np.concatenate(predictions).reshape(-1, 1)
        per_class[class_name] = {
            "threshold": threshold,
            "test_samples": int(target_array.size),
            **compute_ba_sens_spec_gmean(target_array, prediction_array),
        }
    metric_names = ("BA_macro", "SEN_macro", "SPEC_macro", "GMean_macro")
    result = {
        "classes": classes,
        "test_samples": len(test_indices),
        "per_class": per_class,
        "mean_across_six_classes": {
            name: float(np.mean([row[name] for row in per_class.values()])) for name in metric_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean_across_six_classes"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
