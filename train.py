"""
AgroCare IoT — Training Script
================================
Trains the AgroCareNet (1D-CNN + Dual Attention) model on the synthetic
or real-world plant health dataset.

Features:
    - Loads .npz dataset (from generate_synthetic_dataset.py)
    - Train / Validation / Test split (70/15/15)
    - Class-weighted CrossEntropyLoss (handles imbalance)
    - AdamW optimizer + ReduceLROnPlateau scheduler
    - Checkpointing (best model + periodic saves)
    - Training curves, confusion matrix, and classification report
    - Auto-detect device (CUDA → MPS → CPU)

Usage:
    python train.py                                 # Train with defaults
    python train.py --dataset data/agrocare_dataset.npz --epochs 100
    python train.py --resume checkpoints/best_model.pt  # Resume training
    python train.py --lr 0.0005 --batch-size 64

Output:
    checkpoints/best_model.pt       — Best model weights (lowest val loss)
    checkpoints/checkpoint_epoch_N.pt — Periodic checkpoints
    plots/training_curves.png        — Loss and accuracy curves
    plots/confusion_matrix.png       — Test set confusion matrix
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.agrocare_net import AgroCareNet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Normal", "Watering Required", "Risky"]
NUM_CLASSES = 3
WINDOW_SIZE = 60
NUM_CHANNELS = 5


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU")
    return device


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AgroCareDataset(Dataset):
    """PyTorch Dataset wrapper for AgroCare .npz files.

    Loads the full dataset into memory (it's small enough) and provides
    normalized sensor windows.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        """
        Args:
            X: (N, 60, 5) sensor windows
            y: (N,) class labels
            normalize: Whether to apply per-channel z-score normalization
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

        # Compute normalization stats BEFORE normalizing
        # Shape: (N, 60, 5) → compute mean/std across N and time dims → (5,)
        self.channel_means = self.X.mean(dim=(0, 1))  # (5,)
        self.channel_stds = self.X.std(dim=(0, 1))    # (5,)
        self.channel_stds[self.channel_stds < 1e-6] = 1.0  # avoid div by zero

        if normalize:
            self.X = (self.X - self.channel_means) / self.channel_stds

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced loss."""
        counts = torch.bincount(self.y, minlength=NUM_CLASSES).float()
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES  # normalize to mean=1
        return weights

    def get_norm_stats(self) -> Dict[str, torch.Tensor]:
        """Return normalization statistics for use during inference."""
        return {
            "means": self.channel_means,
            "stds": self.channel_stds,
        }


def load_dataset(
    path: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, AgroCareDataset]:
    """Load .npz dataset and split into train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, full_dataset (for norm stats)
    """
    print(f"[DATA] Loading dataset from {path}")
    data = np.load(path)
    X, y = data["X"], data["y"]
    print(f"       Shape: X={X.shape}, y={y.shape}")
    print(f"       Classes: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=3)))}")

    dataset = AgroCareDataset(X, y, normalize=True)

    # Split sizes
    total = len(dataset)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - val_size - test_size

    print(f"       Split: train={train_size}, val={val_size}, test={test_size}")

    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------


class Trainer:
    """Encapsulates the full training loop with checkpointing and logging."""

    def __init__(
        self,
        model: AgroCareNet,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        dataset: AgroCareDataset,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "checkpoints",
        plot_dir: str = "plots",
        checkpoint_every: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset = dataset

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every

        # Class-weighted loss
        class_weights = dataset.get_class_weights().to(device)
        print(f"[TRAIN] Class weights: {class_weights.cpu().numpy().round(3)}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler: reduce LR when val loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        self._last_lr = lr  # track LR for manual change logging

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _train_one_epoch(self) -> Tuple[float, float]:
        """Run one training epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Run validation/test. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with full training state."""
        norm_stats = self.dataset.get_norm_stats()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "norm_means": norm_stats["means"],
            "norm_stds": norm_stats["stds"],
            "model_config": {
                "in_channels": self.model.in_channels,
                "num_classes": self.model.num_classes,
                "feature_dim": self.model.feature_dim,
            },
        }

        # Periodic checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Best model (overwrite)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"         ★ New best model saved (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        print(f"[RESUME] Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint.get("history", self.history)

        start_epoch = checkpoint["epoch"] + 1
        print(f"         Resuming from epoch {start_epoch}, best_val_loss={self.best_val_loss:.4f}")
        return start_epoch

    def train(self, num_epochs: int, start_epoch: int = 1):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"  Training AgroCareNet for {num_epochs} epochs")
        print(f"  Start epoch: {start_epoch}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"{'='*60}\n")

        total_start = time.time()

        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self._train_one_epoch()

            # Validate
            val_loss, val_acc = self._validate(self.val_loader)

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log LR changes
            new_lr = self.optimizer.param_groups[0]["lr"]
            if new_lr != current_lr:
                print(f"         ↓ LR reduced: {current_lr:.2e} → {new_lr:.2e}")

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(new_lr)

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

            # Checkpointing
            if epoch % self.checkpoint_every == 0 or is_best:
                self._save_checkpoint(epoch, is_best=is_best)

            # Log
            elapsed = time.time() - epoch_start
            marker = " ★" if is_best else ""
            print(
                f"  Epoch {epoch:3d}/{start_epoch + num_epochs - 1} │ "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} │ "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} │ "
                f"lr={new_lr:.2e} │ {elapsed:.1f}s{marker}"
            )

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"{'='*60}")

        # Save final checkpoint
        self._save_checkpoint(start_epoch + num_epochs - 1, is_best=False)

        # Generate plots and test evaluation
        self.plot_training_curves()
        self.evaluate_test_set()

    def plot_training_curves(self):
        """Generate and save training loss/accuracy curves."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], label="Train Loss", linewidth=2, color="#3498db")
        ax.plot(epochs, self.history["val_loss"], label="Val Loss", linewidth=2, color="#e74c3c")
        ax.axvline(x=self.best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best (epoch {self.best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy curves
        ax = axes[1]
        ax.plot(epochs, self.history["train_acc"], label="Train Acc", linewidth=2, color="#3498db")
        ax.plot(epochs, self.history["val_acc"], label="Val Acc", linewidth=2, color="#e74c3c")
        ax.axvline(x=self.best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training & Validation Accuracy", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Learning rate
        ax = axes[2]
        ax.plot(epochs, self.history["lr"], linewidth=2, color="#2ecc71")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.plot_dir / "training_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n[PLOT] Training curves saved to {path}")

    @torch.no_grad()
    def evaluate_test_set(self):
        """Run full evaluation on test set: accuracy, per-class metrics, confusion matrix."""
        print(f"\n{'='*60}")
        print("  Test Set Evaluation")
        print(f"{'='*60}")

        self.model.eval()
        all_preds = []
        all_labels = []

        for X_batch, y_batch in self.test_loader:
            X_batch = X_batch.to(self.device)
            logits = self.model(X_batch)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y_batch)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # Overall accuracy
        accuracy = (all_preds == all_labels).mean()
        print(f"\n  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

        # Per-class metrics
        print(f"\n  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*62}")

        for cls in range(NUM_CLASSES):
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            support = (all_labels == cls).sum()

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            print(f"  {CLASS_NAMES[cls]:<22} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

        # Confusion matrix
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(all_labels, all_preds):
            cm[t, p] += 1

        print(f"\n  Confusion Matrix:")
        print(f"  {'':>22} {'Pred Normal':>14} {'Pred Water':>14} {'Pred Risky':>14}")
        for i in range(NUM_CLASSES):
            row = "  " + f"{'True ' + CLASS_NAMES[i]:>22}"
            for j in range(NUM_CLASSES):
                row += f" {cm[i,j]:>13d}"
            print(row)

        # Plot confusion matrix
        self._plot_confusion_matrix(cm, accuracy)

    def _plot_confusion_matrix(self, cm: np.ndarray, accuracy: float):
        """Generate and save confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize for color mapping
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotate with counts and percentages
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                text = f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=11, color=color)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        short_names = ["Normal", "Watering\nRequired", "Risky"]
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("True", fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix — Test Accuracy: {accuracy*100:.1f}%", fontsize=13, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
        plt.tight_layout()
        path = self.plot_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n[PLOT] Confusion matrix saved to {path}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="AgroCare IoT — Train 1D-CNN + Dual Attention Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="data/agrocare_dataset.npz",
        help="Path to .npz dataset (default: data/agrocare_dataset.npz)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")

    # Model config
    parser.add_argument("--cnn-dropout", type=float, default=0.1, help="CNN backbone dropout (default: 0.1)")
    parser.add_argument("--attn-heads", type=int, default=4, help="Temporal attention heads (default: 4)")
    parser.add_argument("--attn-dropout", type=float, default=0.1, help="Attention dropout (default: 0.1)")
    parser.add_argument("--fc-dropout", type=float, default=0.3, help="Classifier head dropout (default: 0.3)")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Output
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory for training plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  AgroCare IoT — Model Training")
    print("=" * 60)
    print(f"  Dataset    : {args.dataset}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Seed       : {args.seed}")
    print("=" * 60)

    # Device
    device = get_device()

    # Load data
    train_loader, val_loader, test_loader, dataset = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Create model
    model = AgroCareNet(
        in_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        cnn_dropout=args.cnn_dropout,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
        fc_dropout=args.fc_dropout,
    )
    print(model.summary())

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset=dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        checkpoint_every=args.checkpoint_every,
    )

    # Resume if requested
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

    print("\n[DONE] Training pipeline complete!")
    print(f"       Best model: {args.checkpoint_dir}/best_model.pt")
    print(f"       Plots: {args.plot_dir}/")


if __name__ == "__main__":
    main()