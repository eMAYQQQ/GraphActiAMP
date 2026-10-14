#!/usr/bin/env python3
"""Train normalized, edge-aware Stage-1 models and nested ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter, softmax


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "unified_stage1_graphs.pkl"
DEFAULT_SPLIT = HERE / "data" / "legacy_matched_split.json"
DEFAULT_STATS = HERE / "data" / "normalization_stats.json"

VARIANTS = {
    "esm_only": {
        "use_structure": False,
        "use_edge_attr": False,
        "use_sequence": False,
        "use_global": False,
    },
    "node_struct": {
        "use_structure": True,
        "use_edge_attr": False,
        "use_sequence": False,
        "use_global": False,
    },
    "edge_3d": {
        "use_structure": True,
        "use_edge_attr": True,
        "use_sequence": False,
        "use_global": False,
    },
    "full_multimodal": {
        "use_structure": True,
        "use_edge_attr": True,
        "use_sequence": True,
        "use_global": True,
    },
    "full_residual": {
        "use_structure": False,
        "use_edge_attr": False,
        "use_sequence": True,
        "use_global": True,
        "residual_auxiliary": True,
    },
}
MODEL_CONSTRUCTOR_KEYS = (
    "variant", "hidden_dim", "dropout", "modality_dropout", "layers", "heads",
    "ablate_esm", "ablate_structure", "ablate_sequence", "ablate_global", "ablate_topology",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class RunningStats:
    def __init__(self, dimensions: int):
        self.count = torch.zeros(dimensions, dtype=torch.float64)
        self.total = torch.zeros(dimensions, dtype=torch.float64)
        self.square_total = torch.zeros(dimensions, dtype=torch.float64)

    def update(self, values: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        values = values.detach().cpu().to(torch.float64).reshape(-1, self.count.numel())
        if valid is None:
            valid = torch.isfinite(values)
        else:
            valid = valid.detach().cpu().bool().reshape_as(values) & torch.isfinite(values)
        clean = torch.where(valid, values, torch.zeros_like(values))
        self.count += valid.sum(dim=0)
        self.total += clean.sum(dim=0)
        self.square_total += clean.square().sum(dim=0)

    def finish(self) -> tuple[list[float], list[float]]:
        count = self.count.clamp_min(1.0)
        mean = self.total / count
        variance = self.square_total / count - mean.square()
        std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
        return mean.to(torch.float32).tolist(), std.to(torch.float32).tolist()


def split_digest(split_path: Path) -> str:
    return hashlib.sha256(split_path.read_bytes()).hexdigest()


def compute_normalization_stats(graphs, train_indices: list[int], split_path: Path) -> dict:
    esm_stats = RunningStats(480)
    struct_stats = RunningStats(6)
    sequence_stats = RunningStats(480)
    global_stats = RunningStats(4)
    for completed, index in enumerate(train_indices, start=1):
        graph = graphs[index]
        esm_stats.update(graph.x[:, :480])
        continuous = graph.x[:, 480:486]
        valid = torch.isfinite(continuous)
        valid[:, 4:6] &= continuous[:, 4:6].abs() < 359.0
        struct_stats.update(continuous, valid)
        sequence_stats.update(graph.seq_feat)
        global_stats.update(graph.global_feat)
        if completed % 5000 == 0:
            print(f"Normalization statistics: {completed}/{len(train_indices)}", flush=True)

    esm_mean, esm_std = esm_stats.finish()
    struct_mean, struct_std = struct_stats.finish()
    sequence_mean, sequence_std = sequence_stats.finish()
    global_mean, global_std = global_stats.finish()
    return {
        "split_sha256": split_digest(split_path),
        "train_samples": len(train_indices),
        "esm_node": {"mean": esm_mean, "std": esm_std},
        "structure_continuous": {"mean": struct_mean, "std": struct_std},
        "sequence": {"mean": sequence_mean, "std": sequence_std},
        "global": {"mean": global_mean, "std": global_std},
        "angle_missing_rule": "abs(angle) >= 359 degrees is imputed to the training mean",
    }


def load_or_compute_stats(
    graphs, train_indices: list[int], split_path: Path, stats_path: Path
) -> dict:
    digest = split_digest(split_path)
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("split_sha256") != digest:
            raise ValueError(
                f"Normalization statistics {stats_path} belong to another split; "
                "use a different --stats path"
            )
        return stats
    stats = compute_normalization_stats(graphs, train_indices, split_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def stat_tensors(stats: dict, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(stats[key]["mean"], dtype=torch.float32),
        torch.tensor(stats[key]["std"], dtype=torch.float32),
    )


class ResidueInputEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, stats: dict, use_structure: bool):
        super().__init__()
        esm_mean, esm_std = stat_tensors(stats, "esm_node")
        struct_mean, struct_std = stat_tensors(stats, "structure_continuous")
        self.register_buffer("esm_mean", esm_mean)
        self.register_buffer("esm_std", esm_std)
        self.register_buffer("struct_mean", struct_mean)
        self.register_buffer("struct_std", struct_std)
        self.use_structure = use_structure
        if use_structure:
            esm_dim = hidden_dim * 3 // 4
            struct_dim = hidden_dim - esm_dim
            self.esm_projection = nn.Sequential(
                nn.Linear(480, esm_dim), nn.LayerNorm(esm_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.struct_projection = nn.Sequential(
                nn.Linear(28, struct_dim), nn.LayerNorm(struct_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        else:
            self.esm_projection = nn.Sequential(
                nn.Linear(480, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        esm = (node_features[:, :480] - self.esm_mean) / self.esm_std
        esm = self.esm_projection(esm)
        if not self.use_structure:
            return esm
        structure = node_features[:, 480:].clone()
        continuous = structure[:, :6]
        invalid_angles = continuous[:, 4:6].abs() >= 359.0
        continuous[:, 4:6] = torch.where(
            invalid_angles, self.struct_mean[4:6].expand_as(continuous[:, 4:6]), continuous[:, 4:6]
        )
        structure[:, :6] = (continuous - self.struct_mean) / self.struct_std
        structure = self.struct_projection(structure)
        return self.output_projection(torch.cat([esm, structure], dim=1))


class EdgeAwareGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        layers: int,
        heads: int,
        stats: dict,
        use_structure: bool,
        use_edge_attr: bool,
    ):
        super().__init__()
        self.input_encoder = ResidueInputEncoder(hidden_dim, dropout, stats, use_structure)
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        self.convolutions = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=6 if use_edge_attr else None,
                add_self_loops=True,
            )
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.attention_gate = nn.Linear(hidden_dim, 1)
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, 8.0, 6))
        self.rbf_width = 8.0 / 5.0

    def distance_features(self, data) -> torch.Tensor | None:
        if not self.use_edge_attr:
            return None
        source, target = data.edge_index
        distances = torch.linalg.vector_norm(data.pos[source] - data.pos[target], dim=1)
        return torch.exp(
            -0.5 * ((distances[:, None] - self.rbf_centers[None, :]) / self.rbf_width).square()
        )

    def forward(self, data) -> torch.Tensor:
        x = self.input_encoder(data.x)
        edge_features = self.distance_features(data)
        for convolution, norm in zip(self.convolutions, self.norms):
            residual = x
            if edge_features is None:
                update = convolution(x, data.edge_index)
            else:
                update = convolution(x, data.edge_index, edge_attr=edge_features)
            x = norm(residual + F.dropout(F.gelu(update), p=self.dropout, training=self.training))

        mean_pool = global_mean_pool(x, data.batch)
        max_pool = global_max_pool(x, data.batch)
        weights = softmax(self.attention_gate(x), data.batch)
        attention_pool = global_add_pool(weights * x, data.batch)
        return self.pool_projection(torch.cat([mean_pool, max_pool, attention_pool], dim=1))


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class StructureEdgeAuxEncoder(nn.Module):
    """Small structural branch used as a residual correction to the ESM backbone."""

    def __init__(self, hidden_dim: int, dropout: float, stats: dict):
        super().__init__()
        structure_mean, structure_std = stat_tensors(stats, "structure_continuous")
        self.register_buffer("structure_mean", structure_mean)
        self.register_buffer("structure_std", structure_std)
        self.register_buffer("rbf_centers", torch.linspace(0.0, 8.0, 6))
        self.rbf_width = 8.0 / 5.0
        auxiliary_dim = max(48, hidden_dim // 3)
        self.node_projection = nn.Sequential(
            nn.Linear(28, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(6, auxiliary_dim),
            nn.GELU(),
            nn.Linear(auxiliary_dim, auxiliary_dim),
        )
        self.message_update = nn.Sequential(
            nn.Linear(auxiliary_dim, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(auxiliary_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, data) -> torch.Tensor:
        structure = data.x[:, 480:].clone()
        continuous = structure[:, :6]
        invalid_angles = continuous[:, 4:6].abs() >= 359.0
        continuous[:, 4:6] = torch.where(
            invalid_angles,
            self.structure_mean[4:6].expand_as(continuous[:, 4:6]),
            continuous[:, 4:6],
        )
        structure[:, :6] = (continuous - self.structure_mean) / self.structure_std
        x = self.node_projection(structure)
        source, target = data.edge_index
        distances = torch.linalg.vector_norm(data.pos[source] - data.pos[target], dim=1)
        edge_features = torch.exp(
            -0.5 * ((distances[:, None] - self.rbf_centers[None, :]) / self.rbf_width).square()
        )
        messages = x[source] + self.edge_projection(edge_features)
        aggregated = scatter(messages, target, dim=0, dim_size=x.shape[0], reduce="mean")
        x = x + self.message_update(aggregated)
        pooled = torch.cat([
            global_mean_pool(x, data.batch),
            global_max_pool(x, data.batch),
        ], dim=1)
        return self.pool_projection(pooled)


class Stage1Classifier(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_dim: int,
        dropout: float,
        modality_dropout: float,
        layers: int,
        heads: int,
        stats: dict,
        ablate_esm: bool = False,
        ablate_structure: bool = False,
        ablate_sequence: bool = False,
        ablate_global: bool = False,
        ablate_topology: bool = False,
    ):
        super().__init__()
        self.freeze_backbone = False
        options = VARIANTS[variant]
        self.variant = variant
        self.use_sequence = options["use_sequence"]
        self.use_global = options["use_global"]
        self.residual_auxiliary = options.get("residual_auxiliary", False)
        self.ablate_esm = ablate_esm
        self.ablate_structure = ablate_structure
        self.ablate_sequence = ablate_sequence
        self.ablate_global = ablate_global
        self.ablate_topology = ablate_topology
        self.modality_dropout = modality_dropout
        self.graph_encoder = EdgeAwareGraphEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            layers=layers,
            heads=heads,
            stats=stats,
            use_structure=options["use_structure"],
            use_edge_attr=options["use_edge_attr"],
        )
        sequence_mean, sequence_std = stat_tensors(stats, "sequence")
        global_mean, global_std = stat_tensors(stats, "global")
        self.register_buffer("sequence_mean", sequence_mean)
        self.register_buffer("sequence_std", sequence_std)
        self.register_buffer("global_mean", global_mean)
        self.register_buffer("global_std", global_std)

        if self.use_sequence:
            self.sequence_encoder = FeatureEncoder(480, hidden_dim, dropout)
        if self.use_global:
            self.global_encoder = FeatureEncoder(4, hidden_dim, dropout)

        modalities = 1 + int(self.use_sequence) + int(self.use_global)
        if self.residual_auxiliary:
            self.structure_edge_encoder = StructureEdgeAuxEncoder(hidden_dim, dropout, stats)
            auxiliary_modalities = 1 + int(self.use_sequence) + int(self.use_global)
            self.auxiliary_fusion = nn.Sequential(
                nn.Linear(auxiliary_modalities * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.auxiliary_fusion[-1].weight)
            nn.init.zeros_(self.auxiliary_fusion[-1].bias)
            self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        elif modalities > 1:
            self.modality_gate = nn.Sequential(
                nn.Linear(modalities * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, modalities),
            )
            self.fusion = nn.Sequential(
                nn.Linear(modalities * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.output = nn.Linear(hidden_dim, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.graph_encoder.eval()
            self.output.eval()
        return self

    def forward(self, data) -> torch.Tensor:
        graph_data = data
        if self.ablate_esm or self.ablate_topology:
            graph_data = data.clone()
            if self.ablate_esm:
                graph_data.x = data.x.clone()
                graph_data.x[:, :480] = self.graph_encoder.input_encoder.esm_mean
            if self.ablate_topology:
                nodes = torch.arange(data.x.size(0), device=data.x.device)
                graph_data.edge_index = torch.stack([nodes, nodes], dim=0)
        graph_representation = self.graph_encoder(graph_data)
        representations = [graph_representation]
        if self.use_sequence:
            sequence = data.seq_feat.view(data.num_graphs, -1)
            if self.ablate_sequence:
                sequence = self.sequence_mean.expand_as(sequence)
            sequence = (sequence - self.sequence_mean) / self.sequence_std
            representations.append(self.sequence_encoder(sequence))
        if self.use_global:
            global_features = data.global_feat.view(data.num_graphs, -1)
            if self.ablate_global:
                global_features = self.global_mean.expand_as(global_features)
            global_features = (global_features - self.global_mean) / self.global_std
            representations.append(self.global_encoder(global_features))

        if self.residual_auxiliary:
            structure_representation = self.structure_edge_encoder(data)
            if self.ablate_structure:
                structure_representation = torch.zeros_like(structure_representation)
            auxiliary = [structure_representation]
            for index, representation in enumerate(representations[1:], start=1):
                if (index == 1 and self.ablate_sequence) or (index == 2 and self.ablate_global):
                    representation = torch.zeros_like(representation)
                auxiliary.append(representation)
            if self.training and self.modality_dropout > 0:
                auxiliary = [
                    representation * (
                        torch.rand(
                            representation.shape[0], 1, device=representation.device
                        ) >= self.modality_dropout
                    )
                    for representation in auxiliary
                ]
            correction = self.auxiliary_fusion(torch.cat(auxiliary, dim=1))
            fused = graph_representation + torch.sigmoid(self.residual_gate) * correction
        elif len(representations) == 1:
            fused = representations[0]
        else:
            context = torch.cat(representations, dim=1)
            gate_logits = self.modality_gate(context)
            if self.training and self.modality_dropout > 0:
                keep = torch.ones_like(gate_logits, dtype=torch.bool)
                keep[:, 1:] = torch.rand_like(gate_logits[:, 1:]) >= self.modality_dropout
                gate_logits = gate_logits.masked_fill(~keep, torch.finfo(gate_logits.dtype).min)
            gates = torch.softmax(gate_logits, dim=1)
            gated = torch.cat(
                [representation * gates[:, index:index + 1] for index, representation in enumerate(representations)],
                dim=1,
            )
            fused = self.fusion(gated)
        return self.output(fused)


def model_from_checkpoint(checkpoint: dict, device: torch.device | str = "cpu") -> Stage1Classifier:
    config = checkpoint["model_config"]
    constructor = {key: config.get(key, False) if key.startswith("ablate_") else config[key] for key in MODEL_CONSTRUCTOR_KEYS}
    model = Stage1Classifier(
        stats=checkpoint["normalization_stats"], **constructor
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def confusion_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    targets = targets.astype(np.int64)
    predictions = predictions.astype(np.int64)
    tp = int(((targets == 1) & (predictions == 1)).sum())
    tn = int(((targets == 0) & (predictions == 0)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / max(1, len(targets)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "gmean": math.sqrt(sensitivity * specificity),
        "f1": f1_score(targets, predictions, zero_division=0),
        "mcc": matthews_corrcoef(targets, predictions),
    }


def choose_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = confusion_metrics(targets, probabilities >= threshold)
        candidate = (
            metrics["balanced_accuracy"], metrics["gmean"], metrics["f1"], -abs(threshold - 0.5)
        )
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), metrics)
    return best[1], best[2]


def run_epoch(model, loader, criterion, device, optimizer=None, gradient_clip: float = 0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets = []
    probabilities = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            labels = batch.y.view(-1, 1)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            targets.append(labels.detach().cpu().numpy().reshape(-1))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
    return total_loss / len(loader.dataset), np.concatenate(targets), np.concatenate(probabilities)


def epoch_learning_rate(epoch: int, epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_curves(history: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(history["val_ba"], label="BA")
    axes[1].plot(history["val_gmean"], label="GMean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation metric")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=220)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def pos_weight_value(mode: str, negatives: int, positives: int) -> float:
    ratio = negatives / max(1, positives)
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return math.sqrt(ratio)
    if mode == "full":
        return ratio
    value = float(mode)
    if value <= 0:
        raise ValueError("--pos-weight must be none, sqrt, full, or a positive number")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--pos-weight", default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Select CPU for the original Stage-1 training protocol")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--ablate-esm", action="store_true")
    parser.add_argument("--ablate-structure", action="store_true")
    parser.add_argument("--ablate-sequence", action="store_true")
    parser.add_argument("--ablate-global", action="store_true")
    parser.add_argument("--ablate-topology", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.dataset.open("rb") as handle:
        payload = pickle.load(handle)
    graphs = payload["data"]
    split = json.loads(args.split.read_text(encoding="utf-8"))
    sample = graphs[0]
    if sample.x.shape[1] != 508 or sample.seq_feat.numel() != 480 or sample.global_feat.numel() != 4:
        raise ValueError("Dataset does not match the 508D + 480D + 4D schema")
    stats = load_or_compute_stats(graphs, split["train"], args.split, args.stats)

    dataset = GraphDataset(graphs)
    loaders = {
        name: DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
        )
        for name, indices in split.items()
    }
    labels = [int(graphs[index].y.item()) for index in split["train"]]
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = pos_weight_value(args.pos_weight, negatives, positives)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    constructor_config = {
        "variant": args.variant,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "modality_dropout": args.modality_dropout,
        "layers": args.layers,
        "heads": args.heads,
        "ablate_esm": args.ablate_esm,
        "ablate_structure": args.ablate_structure,
        "ablate_sequence": args.ablate_sequence,
        "ablate_global": args.ablate_global,
        "ablate_topology": args.ablate_topology,
    }
    model_config = {
        **constructor_config,
        "graph_convolution": "GATConv",
        "architecture_version": 2,
    }
    model = Stage1Classifier(stats=stats, **constructor_config).to(device)
    initialized_keys = []
    if args.init_checkpoint is not None:
        initialization = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        source_state = initialization["model_state"]
        target_state = model.state_dict()
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        initialized_keys = sorted(compatible)
        print(
            f"Initialized {len(initialized_keys)} tensors from {args.init_checkpoint}", flush=True
        )
    if args.freeze_backbone:
        if args.init_checkpoint is None:
            raise ValueError("--freeze-backbone requires --init-checkpoint")
        for parameter in model.graph_encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.output.parameters():
            parameter.requires_grad = False
        model.freeze_backbone = True
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config.update({
        "dataset": str(args.dataset.resolve()),
        "split": str(args.split.resolve()),
        "stats": str(args.stats.resolve()),
        "output_dir": str(output_dir),
        "init_checkpoint": (
            str(args.init_checkpoint.resolve()) if args.init_checkpoint is not None else None
        ),
        "device": str(device),
        "resolved_pos_weight": positive_weight,
        "variant_options": VARIANTS[args.variant],
        "feature_schema": payload.get("feature_schema"),
        "model_config": model_config,
        "initialized_tensors": len(initialized_keys),
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ba": [],
        "val_gmean": [],
        "threshold": [],
        "learning_rate": [],
    }
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    best_path = output_dir / "best_stage1.pt"
    last_path = output_dir / "last_stage1.pt"

    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]
        best_key = tuple(state["best_key"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        start_epoch = int(state["epoch"]) + 1
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_random_state"])
        print(f"Resuming after epoch {start_epoch - 1}", flush=True)

    if args.init_checkpoint is not None and start_epoch == 1:
        initial_val_loss, initial_targets, initial_probabilities = run_epoch(
            model, loaders["val"], criterion, device
        )
        initial_threshold, initial_metrics = choose_threshold(
            initial_targets, initial_probabilities
        )
        best_key = (
            initial_metrics["balanced_accuracy"],
            initial_metrics["gmean"],
            initial_metrics["f1"],
        )
        best_epoch = 0
        torch.save({
            "model_state": model.state_dict(),
            "model_config": model_config,
            "normalization_stats": stats,
            "feature_schema": payload.get("feature_schema"),
            "classes": ["AMP"],
            "threshold": initial_threshold,
            "best_epoch": 0,
            "validation_metrics": initial_metrics,
        }, best_path)
        print(
            f"Epoch 000 | val {initial_val_loss:.4f} | "
            f"BA {initial_metrics['balanced_accuracy']:.4f} | "
            f"GMean {initial_metrics['gmean']:.4f} | "
            f"SEN {initial_metrics['sensitivity']:.4f} | "
            f"SPEC {initial_metrics['specificity']:.4f} | t {initial_threshold:.3f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = epoch_learning_rate(
            epoch, args.epochs, args.warmup_epochs, args.learning_rate
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_loss, _, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            gradient_clip=args.gradient_clip,
        )
        val_loss, val_targets, val_probabilities = run_epoch(
            model, loaders["val"], criterion, device
        )
        threshold, val_metrics = choose_threshold(val_targets, val_probabilities)
        key = (val_metrics["balanced_accuracy"], val_metrics["gmean"], val_metrics["f1"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ba"].append(val_metrics["balanced_accuracy"])
        history["val_gmean"].append(val_metrics["gmean"])
        history["threshold"].append(threshold)
        history["learning_rate"].append(learning_rate)
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"BA {val_metrics['balanced_accuracy']:.4f} | GMean {val_metrics['gmean']:.4f} | "
            f"SEN {val_metrics['sensitivity']:.4f} | SPEC {val_metrics['specificity']:.4f} | "
            f"t {threshold:.3f} | lr {learning_rate:.2e}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model_config,
                "normalization_stats": stats,
                "feature_schema": payload.get("feature_schema"),
                "classes": ["AMP"],
                "threshold": threshold,
                "best_epoch": epoch,
                "validation_metrics": val_metrics,
            }, best_path)
        else:
            epochs_without_improvement += 1

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_key": best_key,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch": epoch,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }, last_path)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["threshold"])
    test_loss, test_targets, test_probabilities = run_epoch(
        model, loaders["test"], criterion, device
    )
    test_predictions = test_probabilities >= threshold
    test_metrics = confusion_metrics(test_targets, test_predictions)
    test_metrics.update({
        "loss": test_loss,
        "roc_auc": roc_auc_score(test_targets, test_probabilities),
        "average_precision": average_precision_score(test_targets, test_probabilities),
        "threshold": threshold,
        "best_epoch": best_epoch,
        "samples": len(test_targets),
    })
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
        "feature_schema": payload.get("feature_schema"),
        "model_config": model_config,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            test_targets.astype(int),
            test_predictions.astype(int),
            target_names=["non-AMP", "AMP"],
            digits=4,
        ),
        encoding="utf-8",
    )
    with (output_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_index", "sample_id", "target", "probability", "prediction"])
        for dataset_index, target, probability, prediction in zip(
            split["test"], test_targets, test_probabilities, test_predictions
        ):
            writer.writerow([
                dataset_index,
                graphs[dataset_index].sample_id,
                int(target),
                f"{probability:.8f}",
                int(prediction),
            ])
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        columns = [
            "train_loss", "val_loss", "val_ba", "val_gmean", "threshold", "learning_rate"
        ]
        writer.writerow(["epoch", *columns])
        for index in range(len(history["train_loss"])):
            writer.writerow([index + 1] + [history[key][index] for key in columns])
    save_curves(history, output_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
