"""Evaluation and Metric Visualization Suite for Arecanut Disease Detection.
Implements Precision, Recall, F1-score, Classification Report, and Confusion Matrix Heatmaps.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Runs full evaluation on a dataset split and calculates classification metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    num_classes = len(class_names) if class_names else int(max(y_true.max(), y_pred.max()) + 1)
    target_names = class_names if class_names else [f"Class_{i}" for i in range(num_classes)]

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=False, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "precision_weighted": float(prec_weighted),
        "recall_macro": float(rec_macro),
        "recall_weighted": float(rec_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_dict": report_dict,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix - EfficientNet-B4 Arecanut Classifier",
    cmap: str = "Blues",
) -> str:
    """Plots a thesis-ready normalized and count-annotated Confusion Matrix."""
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Tick marks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=11, fontweight="medium")
    ax.set_yticklabels(class_names, fontsize=11, fontweight="medium")

    # Annotate counts and percentages inside each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if count > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[Metrics] Saved confusion matrix plot to {output_path}")
    return output_path
