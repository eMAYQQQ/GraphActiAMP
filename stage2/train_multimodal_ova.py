import argparse
import csv
import json
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
    _AMP_DEVICE_TYPE = "cuda"
except ImportError:  # torch < 2.0
    from torch.cuda.amp import GradScaler, autocast
    _AMP_DEVICE_TYPE = None
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from stage2_feature_models import (
    NESTED_FEATURE_VARIANTS,
    Stage2FeatureClassifier,
    load_or_compute_normalization_stats,
)

LEGACY_FEATURE_VARIANTS = {
    "esm_only": 480,
    "node_full": 508,
}
FEATURE_VARIANTS = {
    **LEGACY_FEATURE_VARIANTS,
    "sequence_global": 0,
    "esm_graph": 480,
    "esm_edge": 480,
    "full_residual": 508,
}
HERE = Path(__file__).resolve().parent


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

    def __getitem__(self, idx):
        return self.graphs[idx]


class GraphEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float, feature_variant: str = "node_full"):
        super().__init__()
        if feature_variant not in LEGACY_FEATURE_VARIANTS:
            raise ValueError(f"Unknown feature variant: {feature_variant}")
        self.feature_variant = feature_variant
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_variant == "esm_only":
            x = x[:, :480]
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        return x


class SeqEncoder(nn.Module):
    def __init__(self, seq_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(seq_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, seq_feat):
        x = self.lin1(seq_feat)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        return x


class FusionClassifier(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        seq_dim: int,
        global_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        feature_variant: str = "node_full",
    ):
        super().__init__()
        self.feature_variant = feature_variant
        self.graph_encoder = GraphEncoder(graph_dim, hidden_dim, dropout, feature_variant)
        self.seq_encoder = SeqEncoder(seq_dim, hidden_dim, dropout)
        self.lin1 = nn.Linear(hidden_dim * 2 + global_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        g_repr = self.graph_encoder(data)
        seq_feat = data.seq_feat.to(g_repr.device)
        seq_feat = seq_feat.view(data.num_graphs, -1)
        seq_repr = self.seq_encoder(seq_feat)
        global_feat = data.global_feat.to(g_repr.device)
        global_feat = global_feat.view(data.num_graphs, -1)
        fused = torch.cat([g_repr, seq_repr, global_feat], dim=1)
        fused = self.lin1(fused)
        fused = F.relu(fused)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        logits = self.lin2(fused)
        return logits


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 5.0, gamma_pos: float = 1.0, clip: float = 0.0, eps: float = 1e-8, reduction: str = "none"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.float()
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        targets = targets.float()

        pos_term = probs
        neg_term = 1.0 - probs
        if self.clip > 0:
            neg_term = torch.clamp(neg_term + self.clip, min=self.eps, max=1.0)
        else:
            neg_term = torch.clamp(neg_term, min=self.eps, max=1.0)

        pos_loss = targets * torch.pow(1.0 - pos_term, self.gamma_pos) * torch.log(pos_term + self.eps)
        neg_loss = (1.0 - targets) * torch.pow(1.0 - neg_term, self.gamma_neg) * torch.log(neg_term + self.eps)

        loss = -(pos_loss + neg_loss)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def sanitize_class_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe or "class"


def get_binary_label(graph, class_idx: int) -> torch.Tensor:
    y = graph.y
    if y.ndim == 2:
        y_cls = y[:, class_idx].view(-1, 1)
    elif y.ndim == 1:
        y_cls = y[class_idx].view(1, 1)
    else:
        y_cls = y.view(1, 1)
    return y_cls.float()


def clone_graph(graph):
    if hasattr(graph, "clone"):
        return graph.clone()
    if hasattr(graph, "to_dict"):
        return graph.__class__(**graph.to_dict())
    raise TypeError("Unsupported graph type for cloning")


def build_binary_graphs(graphs, class_idx: int):
    binary_graphs = []
    for graph in graphs:
        new_graph = clone_graph(graph)
        new_graph.y = get_binary_label(graph, class_idx)
        binary_graphs.append(new_graph)
    return binary_graphs


def count_positive(graphs, indices=None) -> int:
    if indices is None:
        indices = range(len(graphs))
    total = 0
    for i in indices:
        total += int(graphs[i].y.view(-1).sum().item())
    return total


def compute_pos_weight_binary(graphs):
    labels = torch.stack([graph.y.view(-1) for graph in graphs])
    positives = labels.sum(dim=0)
    total = labels.size(0)
    negatives = total - positives
    pos_weight = torch.where(positives > 0, negatives / positives, torch.ones_like(positives))
    return pos_weight


def compute_class_weights_binary(graphs, mode: str = "none"):
    if mode == "none":
        return None
    labels = torch.stack([graph.y.view(-1) for graph in graphs])
    positives = labels.sum(dim=0)
    total = labels.size(0)
    pos_rate = positives / (total + 1e-8)
    weights = 1.0 / (pos_rate + 1e-8)
    weights = weights / weights.mean()
    return weights


def compute_sample_weights(graphs, class_weights: torch.Tensor, pos_boost: float = 1.0):
    weights = []
    for graph in graphs:
        label = graph.y.view(-1)
        weight = float((label * class_weights).sum().item())
        if label.item() > 0.5:
            weight *= pos_boost
        if weight <= 0:
            weight = float(class_weights.mean().item())
        weights.append(weight)
    return weights


def tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray,
    metric: str = "ba",
    target_gap: float = 0.1,
) -> np.ndarray:
    thresholds = np.full(1, 0.5, dtype=np.float32)
    best_score = -1.0
    best_gap = float("inf")
    best_ba = -1.0
    found_within_gap = False
    for thr in grid:
        preds = probs[:, 0] >= thr
        metrics = compute_ba_sens_spec_gmean(targets, preds)
        gap = abs(metrics["SEN_macro"] - metrics["SPEC_macro"])
        if metric == "gmean":
            score = metrics["GMean_macro"]
            if score > best_score:
                best_score = score
                best_gap = gap
                thresholds[0] = thr
        elif metric == "ba":
            score = metrics["BA_macro"]
            if score > best_score:
                best_score = score
                best_gap = gap
                thresholds[0] = thr
        elif metric == "balance":
            within_gap = gap <= target_gap
            ba_score = metrics["BA_macro"]
            if within_gap:
                if (
                    (not found_within_gap)
                    or ba_score > best_ba + 1e-8
                    or (abs(ba_score - best_ba) <= 1e-8 and gap < best_gap - 1e-8)
                ):
                    found_within_gap = True
                    best_ba = ba_score
                    best_gap = gap
                    thresholds[0] = thr
            elif not found_within_gap:
                if gap < best_gap - 1e-8 or (abs(gap - best_gap) <= 1e-8 and ba_score > best_ba + 1e-8):
                    best_ba = ba_score
                    best_gap = gap
                    thresholds[0] = thr
        else:
            score = f1_score(targets[:, 0], preds, average="binary", zero_division=0)
            if score > best_score:
                best_score = score
                best_gap = gap
                thresholds[0] = thr
    return thresholds


def compute_weighted_loss(logits, targets, criterion, class_weights=None):
    loss = criterion(logits, targets)
    if class_weights is not None and isinstance(loss, torch.Tensor) and loss.ndim >= 2:
        loss = loss * class_weights.view(1, -1)
    if isinstance(loss, torch.Tensor):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
    return loss.mean()


def evaluate(model, loader, criterion, device, thresholds=None, tune=False, threshold_grid=None, use_amp=False, class_weights=None, threshold_metric: str = "ba", target_gap: float = 0.1, return_stats: bool = False):
    model.eval()
    total_loss = 0.0
    targets = []
    prob_storage = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if _AMP_DEVICE_TYPE is None:
                amp_ctx = autocast(enabled=use_amp and device.type == "cuda")
            else:
                amp_ctx = autocast(device_type=_AMP_DEVICE_TYPE, enabled=use_amp and device.type == "cuda")
            with amp_ctx:
                logits = model(batch)
                loss = compute_weighted_loss(logits, batch.y, criterion, class_weights)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            if not torch.isfinite(loss):
                loss = torch.zeros((), device=device, dtype=torch.float32)
            total_loss += loss.item() * batch.num_graphs
            probs = torch.sigmoid(logits.float())
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            prob_storage.append(probs.cpu())
            targets.append(batch.y.cpu())
    probs = torch.vstack(prob_storage).numpy()
    targets = torch.vstack(targets).numpy()
    if tune:
        grid = threshold_grid if threshold_grid is not None else np.linspace(0.1, 0.9, 17)
        thresholds = tune_thresholds(probs, targets, grid, metric=threshold_metric, target_gap=target_gap)
    thresholds_array = np.asarray(thresholds) if thresholds is not None else np.full(1, 0.5)
    preds = (probs >= thresholds_array.reshape(1, -1)).astype(np.int32)
    f1_micro = f1_score(targets, preds, average="micro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    stats = None
    if return_stats:
        pos_mask = targets[:, 0] == 1
        neg_mask = targets[:, 0] == 0
        finite_probs = np.nan_to_num(probs[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
        pos_mean = float(finite_probs[pos_mask].mean()) if pos_mask.any() else None
        neg_mean = float(finite_probs[neg_mask].mean()) if neg_mask.any() else None
        stats = {
            "pos_prob_mean": pos_mean,
            "neg_prob_mean": neg_mean,
            "pos_count": int(pos_mask.sum()),
            "neg_count": int(neg_mask.sum()),
        }
    if return_stats:
        return total_loss / len(loader.dataset), f1_micro, f1_macro, targets, preds, thresholds_array, stats
    return total_loss / len(loader.dataset), f1_micro, f1_macro, targets, preds, thresholds_array


def compute_ba_sens_spec_gmean(targets: np.ndarray, preds: np.ndarray):
    eps = 1e-8
    tp = (targets * preds).sum(axis=0)
    tn = ((1 - targets) * (1 - preds)).sum(axis=0)
    fp = ((1 - targets) * preds).sum(axis=0)
    fn = (targets * (1 - preds)).sum(axis=0)

    sens = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)
    ba = 0.5 * (sens + spec)
    gmean = np.sqrt(sens * spec)

    return {
        "SEN_macro": float(sens.mean()),
        "SPEC_macro": float(spec.mean()),
        "BA_macro": float(ba.mean()),
        "GMean_macro": float(gmean.mean()),
    }


def compute_per_class_metrics(targets: np.ndarray, preds: np.ndarray, class_name: str):
    eps = 1e-8
    tp = (targets * preds).sum(axis=0)
    tn = ((1 - targets) * (1 - preds)).sum(axis=0)
    fp = ((1 - targets) * preds).sum(axis=0)
    fn = (targets * (1 - preds)).sum(axis=0)

    sens = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)
    ba = 0.5 * (sens + spec)
    gmean = np.sqrt(sens * spec)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )

    return [
        {
            "class": class_name,
            "ACC": float(acc[0]),
            "SEN": float(sens[0]),
            "SPEC": float(spec[0]),
            "BA": float(ba[0]),
            "GMean": float(gmean[0]),
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int((targets[:, 0]).sum()),
        }
    ]


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, grad_accum_steps: int = 1, use_amp: bool = False, clip_grad: float = 0.0, class_weights=None):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        if _AMP_DEVICE_TYPE is None:
            amp_ctx = autocast(enabled=use_amp and device.type == "cuda")
        else:
            amp_ctx = autocast(device_type=_AMP_DEVICE_TYPE, enabled=use_amp and device.type == "cuda")
        with amp_ctx:
            logits = model(batch)
            raw_loss = compute_weighted_loss(logits, batch.y, criterion, class_weights)
            loss = raw_loss / max(1, grad_accum_steps)
        if not torch.isfinite(logits).all() or not torch.isfinite(raw_loss):
            print(f"Warning: non-finite values detected at train step {step}; skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % max(1, grad_accum_steps) == 0:
            if clip_grad and clip_grad > 0:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += raw_loss.item() * batch.num_graphs
        valid_batches += 1
    if len(loader) % max(1, grad_accum_steps) != 0:
        if clip_grad and clip_grad > 0:
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if valid_batches == 0:
        print("Warning: all training batches were skipped because of non-finite losses.")
        return float("nan")
    return total_loss / len(loader.dataset)


def load_dataset(pkl_path: Path):
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    graphs = payload["data"]
    classes = list(payload.get("classes", []))
    if not graphs:
        raise ValueError(f"空数据集: {pkl_path}")
    return graphs, classes, payload


def append_results_csv(csv_path: Path, header: list[str], row: list):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def append_dict_rows_csv(csv_path: Path, rows: list[dict]):
    if not rows:
        return
    write_header = not csv_path.exists()
    fieldnames = list(rows[0].keys())
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_curves(output_dir: Path, history: dict, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss_curve.png")
    plt.close()

    if history.get("val_f1_micro") and history.get("val_f1_macro"):
        plt.figure()
        plt.plot(history["val_f1_micro"], label="val_f1_micro")
        plt.plot(history["val_f1_macro"], label="val_f1_macro")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_val_f1_curve.png")
        plt.close()

    if history.get("val_ba"):
        plt.figure()
        plt.plot(history["val_ba"], label="val_BA")
        if history.get("val_gmean"):
            plt.plot(history["val_gmean"], label="val_GMean")
        if history.get("val_sen"):
            plt.plot(history["val_sen"], label="val_SEN")
        if history.get("val_spec"):
            plt.plot(history["val_spec"], label="val_SPEC")
        plt.xlabel("Epoch")
        plt.ylabel("Metric (macro)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_val_ba_sen_spec_gmean_curve.png")
        plt.close()


def save_live_progress(
    output_dir: Path,
    feature_variant: str,
    class_name: str,
    class_position: int,
    total_classes: int,
    epoch: int,
    total_epochs: int,
    history: dict,
    completed_classes: list[str],
    status: str,
) -> None:
    """Atomically update machine-readable and visual experiment progress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = (class_position - 1 + min(epoch / max(total_epochs, 1), 1.0)) / max(total_classes, 1)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "feature_variant": feature_variant,
        "current_class": class_name,
        "class_position": class_position,
        "total_classes": total_classes,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "overall_fraction": overall,
        "completed_classes": completed_classes,
        "latest": {key: values[-1] for key, values in history.items() if values},
    }
    temporary_json = output_dir / "progress.json.tmp"
    temporary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_json.replace(output_dir / "progress.json")

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    axes[0].barh([0], [overall], color="#2878B5", height=0.38)
    axes[0].barh([0], [1.0 - overall], left=[overall], color="#E6E9ED", height=0.38)
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Overall progress")
    axes[0].set_title(f"{feature_variant}: {overall * 100:.1f}%")
    axes[0].text(0.5, 0, f"{class_name}\nEpoch {epoch}/{total_epochs}", ha="center", va="center", fontsize=9)

    if history.get("train_loss"):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[1].plot(epochs, history["train_loss"], label="Train", color="#2878B5", marker="o", markersize=3)
        axes[1].plot(epochs, history["val_loss"], label="Validation", color="#D9534F", marker="o", markersize=3)
        axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")

    metric_colors = {
        "val_ba": "#2878B5",
        "val_gmean": "#2A9D8F",
        "val_sen": "#E9C46A",
        "val_spec": "#E76F51",
    }
    for key, color in metric_colors.items():
        if history.get(key):
            axes[2].plot(
                np.arange(1, len(history[key]) + 1), history[key],
                label=key[4:].upper(), color=color, marker="o", markersize=3,
            )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Validation metrics")
    axes[2].set_xlabel("Epoch")
    if history.get("val_ba"):
        axes[2].legend(frameon=False, fontsize=7, ncol=2)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle(f"Stage-2 OvA training | {status}", fontsize=11, fontweight="bold")
    figure.tight_layout()
    temporary_png = output_dir / "progress.tmp.png"
    figure.savefig(temporary_png, dpi=150, bbox_inches="tight")
    plt.close(figure)
    temporary_png.replace(output_dir / "progress.png")


def get_split_indices(total: int, train_ratio: float, val_ratio: float, seed: int):
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_len = int(total * train_ratio)
    val_len = int(total * val_ratio)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="One-vs-All multimodal classifier based on GNN+MLP.")
    parser.add_argument("--dataset", default=HERE / "processed_graphs_shared.pkl", type=Path)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="node_full")
    parser.add_argument("--include-classes", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision to reduce GPU memory")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index; set -1 for CPU")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no-pos-weight", action="store_true", help="Disable pos_weight in BCE loss")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Max norm for gradient clipping")
    parser.add_argument("--class-weight", choices=["none", "inverse_freq"], default="none", help="Per-class loss weights")
    parser.add_argument("--sampler", choices=["none", "weighted"], default="none", help="Use weighted sampler for training")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs/ova_runs"))
    parser.add_argument("--threshold-grid-start", type=float, default=0.05)
    parser.add_argument("--threshold-grid-end", type=float, default=0.95)
    parser.add_argument("--threshold-grid-steps", type=int, default=37)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--use-asl", action="store_true", help="Use Asymmetric Loss instead of BCE (default: on)")
    parser.add_argument("--no-asl", action="store_true", help="Disable Asymmetric Loss and use BCE")
    parser.add_argument("--gamma-neg", type=float, default=5.0)
    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05, help="Probability clip used by Asymmetric Loss")
    parser.add_argument("--threshold-metric", choices=["ba", "gmean", "f1", "balance"], default="ba", help="Metric for threshold tuning")
    parser.add_argument("--target-sen-spec-gap", type=float, default=0.10, help="Target max absolute gap between SEN and SPEC when threshold-metric=balance")
    parser.add_argument("--minority-boost", type=float, default=2.0, help="Boost positive samples for minority classes")
    parser.add_argument("--minority-threshold", type=float, default=0.12, help="Pos ratio threshold to enable minority boost")
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--ckpt-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--progress-every", type=int, default=1, help="Refresh progress.png every N epochs")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--torch-threads", type=int, default=4, help="CPU threads used by this training process")
    parser.add_argument("--stats-file", type=Path, default=None, help="Train-split normalization statistics for nested variants")
    parser.add_argument("--modality-dropout", type=float, default=0.1, help="Dropout probability for the structural residual")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))
    if args.no_asl:
        args.use_asl = False
    else:
        args.use_asl = True if not args.use_asl else True

    set_seed(args.seed)

    graphs, classes, dataset_payload = load_dataset(args.dataset)
    if not classes:
        raise ValueError("未在数据集中检测到 classes，请确认 prepare_graph_data.py 输出的 pickle 是否包含 classes")

    if args.split_file:
        split_data = json.loads(args.split_file.read_text())
        required_keys = {"train", "val", "test"}
        if not required_keys.issubset(split_data):
            missing = required_keys - set(split_data)
            raise ValueError(f"split 文件缺少字段: {missing}")
        train_idx = split_data["train"]
        val_idx = split_data["val"]
        test_idx = split_data["test"]
        print(f"Loaded predefined splits from {args.split_file}")
    else:
        train_idx, val_idx, test_idx = get_split_indices(len(graphs), args.train_ratio, args.val_ratio, args.seed)

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    raw_feature_dim = graphs[0].x.size(1)
    feature_dim = FEATURE_VARIANTS[args.feature_variant]
    if raw_feature_dim < feature_dim:
        raise ValueError(
            f"Dataset has {raw_feature_dim} node features, but {args.feature_variant} needs {feature_dim}"
        )
    seq_dim = graphs[0].seq_feat.size(1)
    global_dim = graphs[0].global_feat.size(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization_stats = None
    if args.feature_variant in NESTED_FEATURE_VARIANTS:
        stats_file = args.stats_file or (args.output_dir / "normalization_stats.json")
        normalization_stats = load_or_compute_normalization_stats(
            stats_file.resolve(), graphs, train_idx
        )
        args.stats_file = stats_file
    run_config = vars(args).copy()
    run_config.update({
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "split_file": str(args.split_file.resolve()) if args.split_file else None,
        "device_resolved": str(device),
        "dataset_feature_schema": dataset_payload.get("feature_schema"),
        "shared_input_fingerprint_sha256": dataset_payload.get("shared_input_fingerprint_sha256"),
        "graph_input_dim": feature_dim,
        "model_family": (
            "normalized_nested_residual"
            if args.feature_variant in NESTED_FEATURE_VARIANTS
            else "legacy_raw_concat"
        ),
        "normalization_stats": normalization_stats,
    })
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    summary_csv = args.output_dir / "ova_results_summary.csv"
    per_class_csv = args.output_dir / "ova_per_class_metrics.csv"

    threshold_grid = np.linspace(args.threshold_grid_start, args.threshold_grid_end, args.threshold_grid_steps)

    aggregate_metrics = {
        "BA_macro": [],
        "SEN_macro": [],
        "SPEC_macro": [],
        "GMean_macro": [],
        "test_f1_micro": [],
        "test_f1_macro": [],
        "ACC": [],
        "SEN": [],
        "SPEC": [],
        "BA": [],
        "GMean": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    selected_classes = [
        (class_idx, class_name)
        for class_idx, class_name in enumerate(classes)
        if args.include_classes is None or class_name in args.include_classes
    ]
    if not selected_classes:
        raise ValueError(f"No requested classes found in {classes}")
    completed_classes = []

    for class_position, (class_idx, class_name) in enumerate(selected_classes, start=1):
        safe_name = sanitize_class_name(class_name)
        print(f"\n=== One-vs-All training for class: {class_name} (index {class_idx}) ===")
        class_output_dir = args.output_dir / f"ova_{safe_name}"
        class_output_dir.mkdir(parents=True, exist_ok=True)

        binary_graphs = build_binary_graphs(graphs, class_idx)
        dataset = GraphDataset(binary_graphs)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        train_pos = count_positive(binary_graphs, train_idx)
        val_pos = count_positive(binary_graphs, val_idx)
        test_pos = count_positive(binary_graphs, test_idx)
        print(
            f"Positives in splits | train: {train_pos}/{len(train_set)} | "
            f"val: {val_pos}/{len(val_set)} | test: {test_pos}/{len(test_set)}"
        )

        class_weights = compute_class_weights_binary(binary_graphs, mode=args.class_weight)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        labels_tensor = torch.stack([g.y.view(-1) for g in binary_graphs])
        pos_ratio = float(labels_tensor.mean().item())
        pos_boost = args.minority_boost if pos_ratio < args.minority_threshold else 1.0

        loader_kwargs = {
            "num_workers": max(0, args.num_workers),
            "persistent_workers": args.num_workers > 0,
        }
        if args.sampler == "weighted":
            if class_weights is None:
                class_weights_cpu = compute_class_weights_binary(binary_graphs, mode="inverse_freq")
            else:
                class_weights_cpu = class_weights.detach().cpu()
            train_graphs = [dataset[i] for i in train_idx]
            sample_weights = compute_sample_weights(train_graphs, class_weights_cpu, pos_boost=pos_boost)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, sampler=sampler, **loader_kwargs
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs
            )
        val_loader = DataLoader(val_set, batch_size=args.batch_size, **loader_kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, **loader_kwargs)

        if args.feature_variant in NESTED_FEATURE_VARIANTS:
            model = Stage2FeatureClassifier(
                feature_variant=args.feature_variant,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                stats=normalization_stats,
                out_dim=1,
                modality_dropout=args.modality_dropout,
            ).to(device)
        else:
            model = FusionClassifier(
                feature_dim,
                seq_dim,
                global_dim,
                args.hidden_dim,
                1,
                args.dropout,
                feature_variant=args.feature_variant,
            ).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        pos_weight = compute_pos_weight_binary(binary_graphs).to(device)
        if args.use_asl:
            criterion = AsymmetricLoss(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                clip=args.asl_clip,
                reduction="none",
            )
        else:
            if args.no_pos_weight:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            else:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience)

        best_path = class_output_dir / "best_multimodal_ova.pt"
        best_val_ba = -float("inf")
        best_val_gmean = -float("inf")
        best_val_sen = -float("inf")
        best_val_spec = -float("inf")
        best_thresholds = np.full(1, 0.5)
        epochs_without_improve = 0
        last_epoch = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1_micro": [],
            "val_f1_macro": [],
            "val_ba": [],
            "val_gmean": [],
            "val_sen": [],
            "val_spec": [],
        }

        if _AMP_DEVICE_TYPE is None:
            scaler = GradScaler(enabled=args.amp and device.type == "cuda")
        else:
            scaler = GradScaler(_AMP_DEVICE_TYPE, enabled=args.amp and device.type == "cuda")

        save_live_progress(
            args.output_dir, args.feature_variant, class_name, class_position,
            len(selected_classes), 0, args.epochs, history, completed_classes, "training",
        )
        epoch_progress = tqdm(
            range(1, args.epochs + 1),
            desc=f"{args.feature_variant}:{class_name}",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                grad_accum_steps=max(1, args.grad_accum),
                use_amp=args.amp,
                clip_grad=args.clip_grad,
                class_weights=class_weights,
            )
            tune_val = val_pos > 0 and val_pos < len(val_set)
            if not tune_val:
                print("Warning: val split lacks positive or negative samples; skip threshold tuning.")
            val_loss, val_f1_micro, val_f1_macro, val_targets, val_preds, val_thresholds, val_stats = evaluate(
                model,
                val_loader,
                criterion,
                device,
                thresholds=None,
                tune=tune_val,
                threshold_grid=threshold_grid,
                use_amp=args.amp,
                class_weights=class_weights,
                threshold_metric=args.threshold_metric,
                target_gap=args.target_sen_spec_gap,
                return_stats=True,
            )
            val_ba_metrics = compute_ba_sens_spec_gmean(val_targets, val_preds)
            last_epoch = epoch
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_micro"].append(val_f1_micro)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_ba"].append(val_ba_metrics["BA_macro"])
            history["val_gmean"].append(val_ba_metrics["GMean_macro"])
            history["val_sen"].append(val_ba_metrics["SEN_macro"])
            history["val_spec"].append(val_ba_metrics["SPEC_macro"])
            epoch_progress.set_postfix(
                loss=f"{train_loss:.3f}",
                val_ba=f"{val_ba_metrics['BA_macro']:.3f}",
                val_gmean=f"{val_ba_metrics['GMean_macro']:.3f}",
            )
            print(
                f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | Val F1 (micro/macro): {val_f1_micro:.4f}/{val_f1_macro:.4f} | "
                f"Val BA/SEN/SPEC/GMean: {val_ba_metrics['BA_macro']:.4f}/"
                f"{val_ba_metrics['SEN_macro']:.4f}/{val_ba_metrics['SPEC_macro']:.4f}/"
                f"{val_ba_metrics['GMean_macro']:.4f}"
            )
            if val_stats is not None:
                print(
                    "Val prob mean | pos: "
                    f"{val_stats['pos_prob_mean']} | neg: {val_stats['neg_prob_mean']}"
                )

            is_better = False
            if val_ba_metrics["BA_macro"] > best_val_ba + 1e-8:
                is_better = True
            elif abs(val_ba_metrics["BA_macro"] - best_val_ba) <= 1e-8:
                if val_ba_metrics["GMean_macro"] > best_val_gmean + 1e-8:
                    is_better = True
                elif abs(val_ba_metrics["GMean_macro"] - best_val_gmean) <= 1e-8:
                    if val_ba_metrics["SEN_macro"] > best_val_sen + 1e-8:
                        is_better = True
                    elif abs(val_ba_metrics["SEN_macro"] - best_val_sen) <= 1e-8:
                        if val_ba_metrics["SPEC_macro"] > best_val_spec + 1e-8:
                            is_better = True
            if is_better:
                best_val_ba = val_ba_metrics["BA_macro"]
                best_val_gmean = val_ba_metrics["GMean_macro"]
                best_val_sen = val_ba_metrics["SEN_macro"]
                best_val_spec = val_ba_metrics["SPEC_macro"]
                best_thresholds = val_thresholds
                torch.save({
                    "model_state": model.state_dict(),
                    "model_family": (
                        "normalized_nested_residual"
                        if args.feature_variant in NESTED_FEATURE_VARIANTS
                        else "legacy_raw_concat"
                    ),
                    "class_name": class_name,
                    "feature_dim": feature_dim,
                    "raw_feature_dim": raw_feature_dim,
                    "feature_variant": args.feature_variant,
                    "seq_dim": seq_dim,
                    "global_dim": global_dim,
                    "hidden_dim": args.hidden_dim,
                    "graph_hidden_dim": min(192, args.hidden_dim),
                    "graph_attention_heads": 2,
                    "dropout": args.dropout,
                    "modality_dropout": args.modality_dropout,
                    "normalization_stats": normalization_stats,
                    "feature_schema": dataset_payload.get("feature_schema"),
                    "shared_input_fingerprint_sha256": dataset_payload.get("shared_input_fingerprint_sha256"),
                    "thresholds": best_thresholds.tolist(),
                    "best_val_ba": best_val_ba,
                    "best_val_gmean": best_val_gmean,
                    "best_val_sen": best_val_sen,
                    "best_val_spec": best_val_spec,
                }, best_path)
                print(f"  Saved new best model to {best_path}")
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            scheduler.step(val_ba_metrics["BA_macro"])

            if epoch % max(1, args.progress_every) == 0 or epoch == args.epochs:
                save_live_progress(
                    args.output_dir, args.feature_variant, class_name, class_position,
                    len(selected_classes), epoch, args.epochs, history,
                    completed_classes, "training",
                )

            if args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
                ckpt_path = class_output_dir / f"ckpt_epoch_{epoch}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "history": history,
                    "feature_variant": args.feature_variant,
                    "best_metrics": {
                        "BA_macro": best_val_ba,
                        "GMean_macro": best_val_gmean,
                        "SEN_macro": best_val_sen,
                        "SPEC_macro": best_val_spec,
                    },
                    "best_thresholds": best_thresholds,
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

            if epochs_without_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} due to no improvement for {args.early_stop_patience} epochs.")
                save_live_progress(
                    args.output_dir, args.feature_variant, class_name, class_position,
                    len(selected_classes), epoch, args.epochs, history,
                    completed_classes, "early_stopping",
                )
                break

        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            if "thresholds" in checkpoint:
                best_thresholds = np.array(checkpoint["thresholds"], dtype=np.float32)
            if "best_val_ba" in checkpoint:
                best_val_ba = checkpoint["best_val_ba"]
            if "best_val_gmean" in checkpoint:
                best_val_gmean = checkpoint["best_val_gmean"]
            if "best_val_sen" in checkpoint:
                best_val_sen = checkpoint["best_val_sen"]
            if "best_val_spec" in checkpoint:
                best_val_spec = checkpoint["best_val_spec"]
            print(f"Loaded best model from {best_path}")

        test_loss, test_f1_micro, test_f1_macro, targets, preds, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
            thresholds=best_thresholds,
            tune=False,
            use_amp=args.amp,
            class_weights=class_weights,
        )
        print(f"Test loss: {test_loss:.4f} | Test F1 (micro/macro): {test_f1_micro:.4f}/{test_f1_macro:.4f}")
        print(f"Threshold used: {best_thresholds[0]:.4f}")

        ba_metrics = compute_ba_sens_spec_gmean(targets, preds)
        print(
            "Test BA/SEN/SPEC/GMean (macro): "
            f"{ba_metrics['BA_macro']:.4f} / {ba_metrics['SEN_macro']:.4f} / "
            f"{ba_metrics['SPEC_macro']:.4f} / {ba_metrics['GMean_macro']:.4f}"
        )

        per_class_rows = compute_per_class_metrics(targets, preds, class_name)
        aggregate_metrics["BA_macro"].append(float(ba_metrics["BA_macro"]))
        aggregate_metrics["SEN_macro"].append(float(ba_metrics["SEN_macro"]))
        aggregate_metrics["SPEC_macro"].append(float(ba_metrics["SPEC_macro"]))
        aggregate_metrics["GMean_macro"].append(float(ba_metrics["GMean_macro"]))
        aggregate_metrics["test_f1_micro"].append(float(test_f1_micro))
        aggregate_metrics["test_f1_macro"].append(float(test_f1_macro))
        if per_class_rows:
            aggregate_metrics["ACC"].append(float(per_class_rows[0]["ACC"]))
            aggregate_metrics["SEN"].append(float(per_class_rows[0]["SEN"]))
            aggregate_metrics["SPEC"].append(float(per_class_rows[0]["SPEC"]))
            aggregate_metrics["BA"].append(float(per_class_rows[0]["BA"]))
            aggregate_metrics["GMean"].append(float(per_class_rows[0]["GMean"]))
            aggregate_metrics["precision"].append(float(per_class_rows[0]["precision"]))
            aggregate_metrics["recall"].append(float(per_class_rows[0]["recall"]))
            aggregate_metrics["f1"].append(float(per_class_rows[0]["f1"]))

        base_params = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "class": class_name,
            "feature_variant": args.feature_variant,
            "use_asl": bool(args.use_asl),
            "gamma_neg": args.gamma_neg,
            "gamma_pos": args.gamma_pos,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs_ran": last_epoch,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "threshold_grid_start": args.threshold_grid_start,
            "threshold_grid_end": args.threshold_grid_end,
            "threshold_grid_steps": args.threshold_grid_steps,
            "threshold_metric": args.threshold_metric,
            "pos_ratio": f"{pos_ratio:.6f}",
            "pos_boost": f"{pos_boost:.2f}",
            "val_pos": int(val_pos),
            "test_pos": int(test_pos),
            "BA_macro": f"{ba_metrics['BA_macro']:.4f}",
            "SEN_macro": f"{ba_metrics['SEN_macro']:.4f}",
            "SPEC_macro": f"{ba_metrics['SPEC_macro']:.4f}",
            "GMean_macro": f"{ba_metrics['GMean_macro']:.4f}",
        }
        per_class_rows = [base_params | row for row in per_class_rows]
        append_dict_rows_csv(per_class_csv, per_class_rows)
        print(f"Per-class metrics appended to {per_class_csv}")

        report = classification_report(
            targets.astype(int).ravel(),
            preds.astype(int).ravel(),
            labels=[0, 1],
            target_names=[f"not_{class_name}", class_name],
            zero_division=0,
        )
        report_path = class_output_dir / "classification_report_ova.txt"
        extra_metrics = (
            "\n" * 2
            + "Macro BA/SEN/SPEC/GMean on test set:\n"
            + f"BA: {ba_metrics['BA_macro']:.4f}\n"
            + f"SEN: {ba_metrics['SEN_macro']:.4f}\n"
            + f"SPEC: {ba_metrics['SPEC_macro']:.4f}\n"
            + f"GMean: {ba_metrics['GMean_macro']:.4f}\n"
        )
        report_path.write_text(report + extra_metrics)
        print(f"Classification report written to {report_path}")

        summary_header = [
            "timestamp",
            "seed",
            "class",
            "feature_variant",
            "use_asl",
            "gamma_neg",
            "gamma_pos",
            "lr",
            "hidden_dim",
            "dropout",
            "epochs_ran",
            "train_ratio",
            "val_ratio",
            "threshold_grid_start",
            "threshold_grid_end",
            "threshold_grid_steps",
            "threshold_metric",
            "pos_ratio",
            "pos_boost",
            "val_pos",
            "test_pos",
            "BA_macro",
            "SEN_macro",
            "SPEC_macro",
            "GMean_macro",
            "test_f1_micro",
            "test_f1_macro",
        ]
        summary_row = [
            datetime.now().isoformat(timespec="seconds"),
            args.seed,
            class_name,
            args.feature_variant,
            bool(args.use_asl),
            args.gamma_neg,
            args.gamma_pos,
            args.lr,
            args.hidden_dim,
            args.dropout,
            last_epoch,
            args.train_ratio,
            args.val_ratio,
            args.threshold_grid_start,
            args.threshold_grid_end,
            args.threshold_grid_steps,
            args.threshold_metric,
            f"{pos_ratio:.6f}",
            f"{pos_boost:.2f}",
            int(val_pos),
            int(test_pos),
            f"{ba_metrics['BA_macro']:.4f}",
            f"{ba_metrics['SEN_macro']:.4f}",
            f"{ba_metrics['SPEC_macro']:.4f}",
            f"{ba_metrics['GMean_macro']:.4f}",
            f"{test_f1_micro:.4f}",
            f"{test_f1_macro:.4f}",
        ]
        append_results_csv(summary_csv, summary_header, summary_row)
        print(f"Appended test metrics to {summary_csv}")

        save_curves(class_output_dir, history, prefix="ova")
        print(f"Saved curves to {class_output_dir}")
        completed_classes.append(class_name)
        save_live_progress(
            args.output_dir, args.feature_variant, class_name, class_position,
            len(selected_classes), args.epochs, args.epochs, history,
            completed_classes, "class_complete",
        )

    if aggregate_metrics["BA_macro"]:
        avg_summary_row = [
            datetime.now().isoformat(timespec="seconds"),
            args.seed,
            "AVG_ALL_CLASSES",
            args.feature_variant,
            bool(args.use_asl),
            args.gamma_neg,
            args.gamma_pos,
            args.lr,
            args.hidden_dim,
            args.dropout,
            args.epochs,
            args.train_ratio,
            args.val_ratio,
            args.threshold_grid_start,
            args.threshold_grid_end,
            args.threshold_grid_steps,
            args.threshold_metric,
            "-",
            "-",
            "-",
            "-",
            f"{float(np.mean(aggregate_metrics['BA_macro'])):.4f}",
            f"{float(np.mean(aggregate_metrics['SEN_macro'])):.4f}",
            f"{float(np.mean(aggregate_metrics['SPEC_macro'])):.4f}",
            f"{float(np.mean(aggregate_metrics['GMean_macro'])):.4f}",
            f"{float(np.mean(aggregate_metrics['test_f1_micro'])):.4f}",
            f"{float(np.mean(aggregate_metrics['test_f1_macro'])):.4f}",
        ]
        append_results_csv(summary_csv, summary_header, avg_summary_row)

        avg_per_class_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "class": "AVG_ALL_CLASSES",
            "feature_variant": args.feature_variant,
            "use_asl": bool(args.use_asl),
            "gamma_neg": args.gamma_neg,
            "gamma_pos": args.gamma_pos,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs_ran": args.epochs,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "threshold_grid_start": args.threshold_grid_start,
            "threshold_grid_end": args.threshold_grid_end,
            "threshold_grid_steps": args.threshold_grid_steps,
            "threshold_metric": args.threshold_metric,
            "pos_ratio": "-",
            "pos_boost": "-",
            "val_pos": "-",
            "test_pos": "-",
            "BA_macro": f"{float(np.mean(aggregate_metrics['BA_macro'])):.4f}",
            "SEN_macro": f"{float(np.mean(aggregate_metrics['SEN_macro'])):.4f}",
            "SPEC_macro": f"{float(np.mean(aggregate_metrics['SPEC_macro'])):.4f}",
            "GMean_macro": f"{float(np.mean(aggregate_metrics['GMean_macro'])):.4f}",
            "ACC": f"{float(np.mean(aggregate_metrics['ACC'])):.4f}",
            "SEN": f"{float(np.mean(aggregate_metrics['SEN'])):.4f}",
            "SPEC": f"{float(np.mean(aggregate_metrics['SPEC'])):.4f}",
            "BA": f"{float(np.mean(aggregate_metrics['BA'])):.4f}",
            "GMean": f"{float(np.mean(aggregate_metrics['GMean'])):.4f}",
            "precision": f"{float(np.mean(aggregate_metrics['precision'])):.4f}",
            "recall": f"{float(np.mean(aggregate_metrics['recall'])):.4f}",
            "f1": f"{float(np.mean(aggregate_metrics['f1'])):.4f}",
            "support": "-",
        }
        append_dict_rows_csv(per_class_csv, [avg_per_class_row])
        print(f"Appended averaged metrics across all classes to {summary_csv} and {per_class_csv}")
        save_live_progress(
            args.output_dir, args.feature_variant, class_name, len(selected_classes),
            len(selected_classes), args.epochs, args.epochs, history,
            completed_classes, "complete",
        )


if __name__ == "__main__":
    main()
