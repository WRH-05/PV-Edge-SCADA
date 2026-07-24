"""Transfer-learning training loop for EL defect severity regression.

This script fine-tunes a pretrained ResNet18 backbone on EL solar cell
images using a warmup-to-unfreeze schedule.  The fully-connected head is
trained first while the backbone is frozen; after ``warmup_epochs`` the
entire network is unfrozen for full fine-tuning.

Checkpoints are saved based on a composite criterion that balances
validation critical recall and precision.

Typical usage::

    python -m src.training.train \\
        --csv_path labels.csv --data_root . --epochs 20 --loss_type sahl
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from src.training.dataset import create_dataloaders, set_seed
from src.training.losses import (
    SafetyAwareAsymmetricHuberLoss,
    WeightedMSELoss,
    build_loss,
)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(freeze_early_layers: bool = True) -> nn.Module:
    """Build a ResNet18 regression model with a Sigmoid head.

    Loads a pretrained ImageNet ResNet18 and replaces its final
    classification layer with a single-neuron output with sigmoid
    activation, producing a continuous severity score in ``[0, 1]``.

    Args:
        freeze_early_layers: If ``True``, freeze ``conv1``, ``bn1``,
            ``layer1``, and ``layer2`` so that only ``layer3``, ``layer4``,
            and the new FC head are trainable.  This is useful for
            controlled warmup experiments but is typically set to
            ``False`` when the warmup schedule is managed by
            :func:`configure_trainable_layers`.

    Returns:
        The configured ``nn.Module``.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_early_layers:
        modules_to_freeze = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )

    return model


# ---------------------------------------------------------------------------
# Regression metrics (torch tensors)
# ---------------------------------------------------------------------------


def compute_regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """Compute MSE, MAE, and R² from prediction and target tensors.

    Args:
        predictions: Model output tensor.
        targets: Ground-truth tensor of matching shape.

    Returns:
        Dictionary with keys ``"mse"``, ``"mae"``, ``"r2"``.
    """
    if predictions.numel() == 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    ss_res = torch.sum((targets - predictions) ** 2)
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    r2 = (
        0.0
        if ss_tot.item() <= 1e-12
        else (1.0 - (ss_res / ss_tot)).item()
    )

    return {"mse": mse, "mae": mae, "r2": r2}


def compute_critical_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_threshold: float,
    target_threshold: float,
) -> float:
    """Compute recall on safety-critical (high-severity) samples only.

    A sample is considered critical if its target value is at or above
    ``target_threshold``.  The model "detects" it if its prediction is at
    or above ``prediction_threshold``.

    Args:
        predictions: Model output tensor.
        targets: Ground-truth tensor.
        prediction_threshold: Minimum predicted value to count as positive.
        target_threshold: Minimum ground-truth value for a sample to be
            considered critical.

    Returns:
        Recall value in ``[0.0, 1.0]``.  Returns ``0.0`` if there are no
        critical samples.
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    critical_mask = targets >= target_threshold
    critical_count = int(critical_mask.sum().item())
    if critical_count == 0:
        return 0.0

    true_positives = int(
        (predictions[critical_mask] >= prediction_threshold).sum().item()
    )
    return float(true_positives / critical_count)


def compute_precision_at_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_threshold: float,
    positive_target_threshold: float,
) -> float:
    """Compute precision when binarising predictions at a threshold.

    Args:
        predictions: Model output tensor.
        targets: Ground-truth tensor.
        prediction_threshold: Threshold applied to predictions.
        positive_target_threshold: Threshold applied to targets for
            defining a positive label.

    Returns:
        Precision value.  Returns ``1.0`` if no positive predictions are
        made (vacuously precise).
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    positive_preds = predictions >= prediction_threshold
    positive_count = int(positive_preds.sum().item())
    if positive_count == 0:
        return 1.0

    true_positives = int(
        ((targets >= positive_target_threshold) & positive_preds).sum().item()
    )
    return float(true_positives / positive_count)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Run a full evaluation pass over a DataLoader.

    Args:
        model: The model in evaluation mode (``model.eval()`` is called).
        data_loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        A 3-tuple of ``(metrics_dict, predictions_tensor,
        targets_tensor)`` where ``metrics_dict`` contains ``"loss"``,
        ``"mse"``, ``"mae"``, and ``"r2"``.
    """
    model.eval()
    running_loss = 0.0
    total = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    avg_loss = running_loss / max(total, 1)
    predictions = (
        torch.cat(all_predictions, dim=0)
        if all_predictions
        else torch.empty(0, 1)
    )
    targets = (
        torch.cat(all_targets, dim=0)
        if all_targets
        else torch.empty(0, 1)
    )
    metrics = compute_regression_metrics(predictions, targets)
    metrics["loss"] = avg_loss
    return metrics, predictions, targets


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    """Simple early-stopping tracker with patience.

    Attributes:
        patience: Number of epochs to tolerate without improvement.
        min_delta: Minimum absolute change to qualify as an improvement.
        best_score: Best (lowest) metric value seen so far.
        epochs_without_improvement: Consecutive epochs without improvement.
    """

    patience: int = 7
    min_delta: float = 1e-4
    best_score: float = float("inf")
    epochs_without_improvement: int = 0

    def step(self, current_score: float) -> bool:
        """Update tracker with the current metric value.

        Args:
            current_score: The monitored metric for this epoch (lower is
                better).

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------


def configure_trainable_layers(model: nn.Module, head_only: bool) -> None:
    """Freeze or unfreeze model parameters for the warmup schedule.

    When ``head_only=True``, all backbone parameters are frozen and only
    the final FC head (``model.fc``) receives gradient updates.  When
    ``head_only=False``, every parameter is unfrozen for full fine-tuning.

    Args:
        model: The ResNet18 model.
        head_only: If ``True``, freeze everything except ``model.fc``.
    """
    if head_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full training pipeline with warmup-to-unfreeze scheduling.

    Training proceeds in two phases:

    1. **Warmup** (epochs 1 … ``warmup_epochs``):  Only the FC head is
       trained; the ResNet backbone is frozen.
    2. **Full fine-tuning** (remaining epochs):  All parameters are
       unfrozen, and a fresh Adam optimizer and ``ReduceLROnPlateau``
       scheduler are initialised.

    Checkpoints are saved when validation precision meets the
    ``precision_floor`` and critical recall improves (or is tied with
    lower MAE).

    Args:
        args: Parsed command-line arguments (see :func:`parse_args`).

    Raises:
        RuntimeError: If no checkpoint met the precision floor.
    """
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, val_loader, test_loader, split_counts = create_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = build_model(freeze_early_layers=False).to(device)
    criterion = build_loss(
        args.loss_type,
        args.huber_beta,
        args.loss_weight_threshold,
        args.loss_weight_multiplier,
    )

    # -- Warmup phase: head only --
    configure_trainable_layers(model, head_only=True)
    if args.epochs <= args.warmup_epochs:
        print(
            f"Warning: epochs ({args.epochs}) <= warmup_epochs "
            f"({args.warmup_epochs}); backbone will stay frozen."
        )

    trainable_params = [
        param for param in model.parameters() if param.requires_grad
    ]
    optimizer = optim.Adam(
        trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_delta,
    )

    best_critical_recall = -1.0
    best_precision = 0.0
    best_val_mae = float("inf")
    best_state = None

    print(f"Device: {device}")
    print(f"Split counts: {split_counts}")

    for epoch in range(args.epochs):
        # --- Transition to full fine-tuning ---
        if epoch == args.warmup_epochs:
            print(f"Unfreezing full backbone at epoch {epoch + 1}.")
            configure_trainable_layers(model, head_only=False)
            trainable_params = [
                param for param in model.parameters() if param.requires_grad
            ]
            optimizer = optim.Adam(
                trainable_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
            )

        # --- Training pass ---
        model.train()
        running_train_loss = 0.0
        n_train = 0
        train_predictions: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [Train]",
            leave=False,
        )
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_train_loss += loss.item() * batch_size
            n_train += batch_size
            train_predictions.append(outputs.detach().cpu())
            train_targets.append(labels.detach().cpu())

            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_train_loss / max(n_train, 1)
        train_pred_tensor = (
            torch.cat(train_predictions, dim=0)
            if train_predictions
            else torch.empty(0, 1)
        )
        train_target_tensor = (
            torch.cat(train_targets, dim=0)
            if train_targets
            else torch.empty(0, 1)
        )
        train_metrics = compute_regression_metrics(
            train_pred_tensor, train_target_tensor
        )

        # --- Validation pass ---
        val_metrics, val_predictions, val_targets = evaluate(
            model, val_loader, criterion, device
        )
        critical_recall = compute_critical_recall(
            predictions=val_predictions,
            targets=val_targets,
            prediction_threshold=args.critical_recall_threshold,
            target_threshold=args.critical_target_threshold,
        )
        precision = compute_precision_at_threshold(
            predictions=val_predictions,
            targets=val_targets,
            prediction_threshold=args.critical_recall_threshold,
            positive_target_threshold=args.critical_target_threshold,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train MSE: {train_metrics['mse']:.4f} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Train R2: {train_metrics['r2']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f} | "
            f"Val CritRecall: {critical_recall:.4f} | "
            f"Val Precision@{args.critical_recall_threshold:.2f}: "
            f"{precision:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- Checkpoint selection ---
        should_save = precision >= args.precision_floor
        better_recall = critical_recall > best_critical_recall
        tie_on_recall = abs(critical_recall - best_critical_recall) <= 1e-8

        if should_save and (
            better_recall or (tie_on_recall and val_metrics["mae"] < best_val_mae)
        ):
            best_critical_recall = critical_recall
            best_precision = precision
            best_val_mae = val_metrics["mae"]
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "model_state_dict": best_state,
                "image_size": args.image_size,
                "best_val_mse": val_metrics["mse"],
                "best_val_mae": best_val_mae,
                "best_critical_recall": best_critical_recall,
                "best_precision": best_precision,
                "best_epoch": epoch + 1,
                "loss_type": args.loss_type,
                "loss_weight_threshold": args.loss_weight_threshold,
                "loss_weight_multiplier": args.loss_weight_multiplier,
                "critical_recall_threshold": args.critical_recall_threshold,
                "critical_target_threshold": args.critical_target_threshold,
                "precision_floor": args.precision_floor,
                "warmup_epochs": args.warmup_epochs,
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(
                "Saved improved checkpoint to: "
                f"{args.checkpoint_path} "
                f"(critical_recall={best_critical_recall:.4f}, "
                f"precision={best_precision:.4f})"
            )
        elif not should_save:
            print(
                f"Skipped checkpoint save due to precision floor: "
                f"{precision:.4f} < {args.precision_floor:.4f}"
            )

        if early_stopping.step(val_metrics["mae"]):
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(best val MAE: {early_stopping.best_score:.4f})."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training completed but no checkpoint met the precision floor. "
            "Lower --precision_floor or adjust training configuration."
        )

    model.load_state_dict(best_state)
    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(
        f"Test MSE: {test_metrics['mse']:.4f} | "
        f"Test MAE: {test_metrics['mae']:.4f} | "
        f"Test R2: {test_metrics['r2']:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train ResNet18 EL defect regression model with transfer "
            "learning and warmup-to-unfreeze scheduling."
        )
    )
    # Data
    parser.add_argument("--csv_path", type=str, default="labels.csv")
    parser.add_argument("--data_root", type=str, default=".")

    # Output
    parser.add_argument(
        "--checkpoint_path", type=str, default="best_model.pth"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Loss
    parser.add_argument(
        "--loss_type",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "mse", "weighted_l1", "weighted_mse", "sahl"],
    )
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--loss_weight_threshold", type=float, default=0.70)
    parser.add_argument("--loss_weight_multiplier", type=float, default=2.5)

    # Warmup
    parser.add_argument("--warmup_epochs", type=int, default=2)

    # Evaluation thresholds
    parser.add_argument(
        "--critical_recall_threshold", type=float, default=0.6
    )
    parser.add_argument(
        "--critical_target_threshold", type=float, default=0.8
    )
    parser.add_argument("--precision_floor", type=float, default=0.70)

    # Scheduler
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    parser.add_argument("--early_stopping_delta", type=float, default=1e-4)

    # System
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
