#!/usr/bin/env python3
"""Normalized nested Stage-2 feature models for paper ablations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import scatter


NESTED_FEATURE_VARIANTS = {
    "sequence_global": {
        "use_graph": False,
        "use_edge_attr": False,
        "use_structure_residual": False,
    },
    "esm_graph": {
        "use_graph": True,
        "use_edge_attr": False,
        "use_structure_residual": False,
    },
    "esm_edge": {
        "use_graph": True,
        "use_edge_attr": True,
        "use_structure_residual": False,
    },
    "full_residual": {
        "use_graph": True,
        "use_edge_attr": True,
        "use_structure_residual": True,
    },
}


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


def indices_digest(indices: list[int]) -> str:
    payload = json.dumps([int(index) for index in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def compute_normalization_stats(graphs, train_indices: list[int]) -> dict:
    esm_stats = RunningStats(480)
    struct_stats = RunningStats(6)
    sequence_stats = RunningStats(480)
    global_stats = RunningStats(4)
    for index in train_indices:
        graph = graphs[index]
        esm_stats.update(graph.x[:, :480])
        continuous = graph.x[:, 480:486]
        valid = torch.isfinite(continuous)
        valid[:, 4:6] &= continuous[:, 4:6].abs() < 359.0
        struct_stats.update(continuous, valid)
        sequence_stats.update(graph.seq_feat)
        global_stats.update(graph.global_feat)
    esm_mean, esm_std = esm_stats.finish()
    struct_mean, struct_std = struct_stats.finish()
    sequence_mean, sequence_std = sequence_stats.finish()
    global_mean, global_std = global_stats.finish()
    return {
        "train_indices_sha256": indices_digest(train_indices),
        "train_samples": len(train_indices),
        "esm_node": {"mean": esm_mean, "std": esm_std},
        "structure_continuous": {"mean": struct_mean, "std": struct_std},
        "sequence": {"mean": sequence_mean, "std": sequence_std},
        "global": {"mean": global_mean, "std": global_std},
        "angle_missing_rule": "abs(angle) >= 359 degrees is imputed to the training mean",
    }


def load_or_compute_normalization_stats(
    path: Path,
    graphs,
    train_indices: list[int],
) -> dict:
    digest = indices_digest(train_indices)
    if path.is_file():
        stats = json.loads(path.read_text(encoding="utf-8"))
        if stats.get("train_indices_sha256") != digest:
            raise ValueError(f"Normalization statistics belong to another split: {path}")
        return stats
    stats = compute_normalization_stats(graphs, train_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def stat_tensors(stats: dict, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(stats[key]["mean"], dtype=torch.float32),
        torch.tensor(stats[key]["std"], dtype=torch.float32),
    )


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


class ESMGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        stats: dict,
        use_edge_attr: bool,
    ):
        super().__init__()
        esm_mean, esm_std = stat_tensors(stats, "esm_node")
        self.register_buffer("esm_mean", esm_mean)
        self.register_buffer("esm_std", esm_std)
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        # Keep the OvA fusion space at 384D while using a compact graph core.
        # This reduces over-parameterization and makes multi-seed ablations practical.
        graph_hidden_dim = min(192, hidden_dim)
        self.graph_hidden_dim = graph_hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(480, graph_hidden_dim),
            nn.LayerNorm(graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convolutions = nn.ModuleList([
            GATConv(
                graph_hidden_dim,
                graph_hidden_dim,
                heads=2,
                concat=False,
                dropout=dropout,
                edge_dim=6 if use_edge_attr else None,
                add_self_loops=True,
            )
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(graph_hidden_dim) for _ in range(2)])
        self.output_projection = nn.Sequential(
            nn.Linear(graph_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, data) -> torch.Tensor:
        esm = (data.x[:, :480] - self.esm_mean) / self.esm_std
        x = self.input_projection(esm)
        edge_attr = data.edge_attr if self.use_edge_attr else None
        for convolution, norm in zip(self.convolutions, self.norms):
            residual = x
            if edge_attr is None:
                update = convolution(x, data.edge_index)
            else:
                update = convolution(x, data.edge_index, edge_attr=edge_attr)
            x = norm(residual + F.dropout(F.gelu(update), p=self.dropout, training=self.training))
        return self.output_projection(global_mean_pool(x, data.batch))


class StructureResidualEncoder(nn.Module):
    """Normalized 28D node and 6D edge correction for the ESM graph backbone."""

    def __init__(self, hidden_dim: int, dropout: float, stats: dict):
        super().__init__()
        structure_mean, structure_std = stat_tensors(stats, "structure_continuous")
        self.register_buffer("structure_mean", structure_mean)
        self.register_buffer("structure_std", structure_std)
        auxiliary_dim = max(64, hidden_dim // 4)
        self.node_projection = nn.Sequential(
            nn.Linear(28, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(6, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
        )
        self.message_update = nn.Sequential(
            nn.Linear(auxiliary_dim, auxiliary_dim),
            nn.LayerNorm(auxiliary_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(auxiliary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, data) -> torch.Tensor:
        structure = data.x[:, 480:].clone()
        continuous = structure[:, :6]
        missing_angles = continuous[:, 4:6].abs() >= 359.0
        continuous[:, 4:6] = torch.where(
            missing_angles,
            self.structure_mean[4:6].expand_as(continuous[:, 4:6]),
            continuous[:, 4:6],
        )
        structure[:, :6] = (continuous - self.structure_mean) / self.structure_std
        x = self.node_projection(structure)
        source, target = data.edge_index
        messages = x[source] + self.edge_projection(data.edge_attr)
        aggregated = scatter(messages, target, dim=0, dim_size=x.shape[0], reduce="mean")
        x = x + self.message_update(aggregated)
        return self.output_projection(global_mean_pool(x, data.batch))


class Stage2FeatureClassifier(nn.Module):
    def __init__(
        self,
        feature_variant: str,
        hidden_dim: int,
        dropout: float,
        stats: dict,
        out_dim: int = 1,
        modality_dropout: float = 0.1,
    ):
        super().__init__()
        if feature_variant not in NESTED_FEATURE_VARIANTS:
            raise ValueError(f"Unknown nested feature variant: {feature_variant}")
        options = NESTED_FEATURE_VARIANTS[feature_variant]
        self.feature_variant = feature_variant
        self.use_graph = options["use_graph"]
        self.use_structure_residual = options["use_structure_residual"]
        self.modality_dropout = modality_dropout

        sequence_mean, sequence_std = stat_tensors(stats, "sequence")
        global_mean, global_std = stat_tensors(stats, "global")
        self.register_buffer("sequence_mean", sequence_mean)
        self.register_buffer("sequence_std", sequence_std)
        self.register_buffer("global_mean", global_mean)
        self.register_buffer("global_std", global_std)
        self.sequence_encoder = FeatureEncoder(480, hidden_dim, dropout)
        self.global_encoder = FeatureEncoder(4, hidden_dim, dropout)

        modalities = 2
        if self.use_graph:
            modalities += 1
            self.graph_encoder = ESMGraphEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                stats=stats,
                use_edge_attr=options["use_edge_attr"],
            )
        if self.use_structure_residual:
            self.structure_encoder = StructureResidualEncoder(hidden_dim, dropout, stats)
            self.structure_gate = nn.Parameter(torch.tensor(0.0))

        self.fusion = nn.Sequential(
            nn.Linear(modalities * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, data) -> torch.Tensor:
        sequence = data.seq_feat.view(data.num_graphs, -1)
        sequence = (sequence - self.sequence_mean) / self.sequence_std
        global_features = data.global_feat.view(data.num_graphs, -1)
        global_features = (global_features - self.global_mean) / self.global_std
        representations = [
            self.sequence_encoder(sequence),
            self.global_encoder(global_features),
        ]
        if self.use_graph:
            graph = self.graph_encoder(data)
            if self.use_structure_residual:
                correction = self.structure_encoder(data)
                if self.training and self.modality_dropout > 0:
                    keep = (
                        torch.rand(correction.shape[0], 1, device=correction.device)
                        >= self.modality_dropout
                    )
                    correction = correction * keep
                graph = graph + torch.tanh(self.structure_gate) * correction
            representations.insert(0, graph)
        fused = self.fusion(torch.cat(representations, dim=1))
        return self.output(fused)


def model_from_stage2_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    model = Stage2FeatureClassifier(
        feature_variant=checkpoint["feature_variant"],
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        stats=checkpoint["normalization_stats"],
        out_dim=1,
        modality_dropout=float(checkpoint.get("modality_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model
