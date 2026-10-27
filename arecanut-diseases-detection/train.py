"""Comprehensive Transfer Learning & Regularized Fine-Tuning Pipeline for Arecanut Disease Detection.

Architecture: EfficientNet-B0 Pretrained Backbone (Downscaled to prevent memorization)
Regularization: Weight Decay 0.02, nn.Dropout(p=0.4), RandomResizedCrop, RandomErasing
Phase 1: Feature Extractor Frozen -> Train Head (LR: 5e-4)
Phase 2: Top Blocks Unfrozen -> Fine-Tuning (LR: 5e-5) with Early Stopping (patience=4 on val_loss)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import create_dataloaders
from src.metrics import evaluate_model, plot_confusion_matrix
from src.model import ArecaEfficientNetB0, build_model


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data).item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate_val_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Computes exact validation loss, accuracy, and weighted F1."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            total += targets.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    val_loss = running_loss / max(total, 1)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return val_loss, float(val_acc), float(val_f1)


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Training] Using Device: {device} | EfficientNet-B0 Regularized Backbone (Target Input: {args.img_size}x{args.img_size})")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data with Background-Invariant Augmentations
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    num_classes = len(class_names)

    # 2. Build EfficientNet-B0 Model with Dropout(0.4)
    model = build_model(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout,
        class_names=class_names,
        device=device,
    )

    if device.type == "cpu":
        torch.set_num_threads(min(12, os.cpu_count() or 4))
        print(f"[Training] Multi-thread CPU acceleration enabled: {torch.get_num_threads()} threads")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_val_loss = float("inf")
    patience_counter = 0
    patience = args.patience

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    best_checkpoint_path = os.path.join(args.output_dir, "best_arecanut_efficientnet_b4.pth")
    b0_checkpoint_path = os.path.join(args.output_dir, "best_arecanut_efficientnet_b0.pth")

    # ==========================================
    # PHASE 1: Train Classification Head (Backbone Frozen)
    # ==========================================
    print("\n" + "=" * 65)
    print(f"=== STAGE 1: Head Tuning ({args.epochs_stage1} Epochs, Backbone Frozen, LR: {args.lr_stage1}, Weight Decay: {args.weight_decay}) ===")
    print("=" * 65)
    model.freeze_backbone()

    optimizer_s1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs_stage1 + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_s1, device)
        val_loss, val_acc, val_f1 = evaluate_val_loss(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Stage 1 [Epoch {epoch:02d}/{args.epochs_stage1:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1 * 100:.2f}% | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_payload = {
                "epoch": epoch,
                "stage": 1,
                "model_name": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_val_loss": best_val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
            torch.save(save_payload, best_checkpoint_path)
            torch.save(save_payload, b0_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%)")

    # ==========================================
    # PHASE 2: Unfreeze Top Blocks for Fine-Tuning
    # ==========================================
    if args.epochs_stage2 > 0:
        print("\n" + "=" * 65)
        print(f"=== STAGE 2: Fine-Tuning ({args.epochs_stage2} Epochs Max, Top Blocks Unfrozen, LR: {args.lr_stage2}, Patience: {patience}) ===")
        print("=" * 65)
        model.unfreeze_top_stages(num_blocks=args.unfreeze_blocks)

        optimizer_s2 = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_stage2 * 0.5},
            {"params": model.classifier.parameters(), "lr": args.lr_stage2},
        ], weight_decay=args.weight_decay)

        scheduler = CosineAnnealingLR(optimizer_s2, T_max=args.epochs_stage2, eta_min=1e-6)

        for epoch in range(1, args.epochs_stage2 + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_s2, device)
            val_loss, val_acc, val_f1 = evaluate_val_loss(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(
                f"Stage 2 [Epoch {epoch:02d}/{args.epochs_stage2:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Val F1: {val_f1 * 100:.2f}% | Time: {elapsed:.1f}s"
            )

            # Early Stopping Check on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_payload = {
                    "epoch": epoch,
                    "stage": 2,
                    "model_name": "efficientnet_b0",
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }
                torch.save(save_payload, best_checkpoint_path)
                torch.save(save_payload, b0_checkpoint_path)
                print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%)")
            else:
                patience_counter += 1
                print(f"  [Early Stopping Counter: {patience_counter}/{patience}] (Best Val Loss: {best_val_loss:.4f})")
                if patience_counter >= patience:
                    print(f"\n[Early Stopping Triggered] Validation loss did not improve for {patience} consecutive epochs.")
                    break

    # ==========================================
    # FINAL EVALUATION ON UNSEEN TEST SET
    # ==========================================
    print("\n" + "=" * 65)
    print("=== FINAL EVALUATION ON UNSEEN TEST SET ===")
    print("=" * 65)
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best weights from {best_checkpoint_path} (Stage {checkpoint.get('stage')}, Val Loss: {checkpoint.get('best_val_loss', 0.0):.4f})")

    test_metrics = evaluate_model(model, test_loader, device, class_names=class_names)

    print("\n--- TEST CLASSIFICATION REPORT ---")
    print(test_metrics["classification_report"])
    print(f"Overall Test Accuracy:    {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:          {test_metrics['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall:             {test_metrics['recall_macro'] * 100:.2f}%")
    print(f"Macro F1-Score:           {test_metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted F1-Score:        {test_metrics['f1_weighted'] * 100:.2f}%")

    # Plot & Save Confusion Matrix
    cm_path = os.path.join(args.output_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_path,
        title="Arecanut Disease Test Set - EfficientNet-B0 Confusion Matrix",
    )

    # Save Metrics Summary JSON
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model_architecture": "EfficientNet-B0",
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
            "class_names": class_names,
            "classification_report": test_metrics["classification_report_dict"],
        }, f, indent=4)
    print(f"[Results] Saved evaluation summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Regularized EfficientNet-B0 for Arecanut Disease Detection")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--epochs_stage1", type=int, default=6, help="Epochs for Phase 1 (frozen backbone)")
    parser.add_argument("--epochs_stage2", type=int, default=15, help="Epochs for Phase 2 (fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr_stage1", type=float, default=5e-4, help="Phase 1 learning rate (default: 5e-4)")
    parser.add_argument("--lr_stage2", type=float, default=5e-5, help="Phase 2 learning rate (default: 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="AdamW weight decay (default: 0.02)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience epochs on val_loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Cross entropy label smoothing")
    parser.add_argument("--unfreeze_blocks", type=int, default=3, help="Top blocks to unfreeze in Stage 2")
    parser.add_argument("--img_size", type=int, default=380, help="Input resolution (380)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate in head")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
