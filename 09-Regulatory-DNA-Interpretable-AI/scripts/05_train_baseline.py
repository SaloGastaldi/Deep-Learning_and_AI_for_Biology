#!/usr/bin/env python3
"""Train the compact sequence-only CNN baseline.

Model selection uses validation AUPRC only. The test set is never loaded by
this script and remains isolated for final evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.evaluation.metrics import binary_classification_metrics
from src.models.cnn_baseline import (
    CNNBaselineConfig,
    PromoterCNNBaseline,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the sequence-only promoter CNN baseline."
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cnn_baseline_best.pt"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/metrics/cnn_baseline_training_history.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/metrics/cnn_baseline_training_summary.json"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate training arguments."""

    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight-decay cannot be negative.")
    if args.patience <= 0:
        raise ValueError("patience must be positive.")
    if args.min_delta < 0:
        raise ValueError("min-delta cannot be negative.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")


def set_reproducibility(seed: int) -> None:
    """Set random seeds and deterministic CPU behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    dataset: PromoterDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    """Create a reproducible DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    threshold: float,
) -> dict[str, float]:
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    targets: list[float] = []
    probabilities: list[float] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        batch_probabilities = torch.sigmoid(logits)

        targets.extend(labels.detach().cpu().tolist())
        probabilities.extend(
            batch_probabilities.detach().cpu().tolist()
        )

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")

    metrics = binary_classification_metrics(
        targets=targets,
        probabilities=probabilities,
        threshold=threshold,
    )

    metrics["loss"] = total_loss / total_samples
    metrics["samples"] = float(total_samples)

    return metrics


def checkpoint_payload(
    model: PromoterCNNBaseline,
    optimizer: AdamW,
    epoch: int,
    best_validation_auprc: float,
    args: argparse.Namespace,
    train_dataset: PromoterDataset,
) -> dict:
    """Build the serialized checkpoint payload."""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "best_validation_auprc": best_validation_auprc,
        "training_arguments": vars(args),
        "train_positive_weight": train_dataset.positive_weight,
        "selection_metric": "validation_auprc",
        "test_set_used": False,
    }


def main() -> int:
    """Train and select the baseline model."""

    args = parse_args()

    try:
        validate_args(args)
        set_reproducibility(args.seed)

        device = torch.device("cpu")

        train_dataset = PromoterDataset(args.train)
        validation_dataset = PromoterDataset(args.validation)

        train_loader = make_loader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        validation_loader = make_loader(
            dataset=validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            seed=args.seed + 1,
            num_workers=args.num_workers,
        )

        config = CNNBaselineConfig(
            input_channels=4,
            conv_channels=(32, 64, 128),
            kernel_sizes=(15, 11, 7),
            pool_size=4,
            dense_units=64,
            dropout=0.40,
        )

        model = PromoterCNNBaseline(config=config).to(device)

        positive_weight = torch.tensor(
            [train_dataset.positive_weight],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight
        )

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        history_rows: list[dict[str, float | int]] = []
        best_validation_auprc = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_start = time.time()

        print("=" * 88)
        print("CNN BASELINE TRAINING")
        print("=" * 88)
        print(f"Device:               {device}")
        print(f"Train samples:        {len(train_dataset)}")
        print(f"Validation samples:   {len(validation_dataset)}")
        print(f"Train pos_weight:     {train_dataset.positive_weight:.6f}")
        print(f"Trainable parameters: {count_trainable_parameters(model):,}")
        print(f"Maximum epochs:       {args.epochs}")
        print(f"Early-stop patience:  {args.patience}")
        print("=" * 88)

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                threshold=args.threshold,
            )

            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                threshold=args.threshold,
            )

            row: dict[str, float | int] = {
                "epoch": epoch,
                "epoch_seconds": time.time() - epoch_start,
            }

            for name, value in train_metrics.items():
                row[f"train_{name}"] = value

            for name, value in validation_metrics.items():
                row[f"validation_{name}"] = value

            history_rows.append(row)

            current_auprc = validation_metrics["auprc"]
            improved = (
                current_auprc
                > best_validation_auprc + args.min_delta
            )

            if improved:
                best_validation_auprc = current_auprc
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_validation_auprc=best_validation_auprc,
                        args=args,
                        train_dataset=train_dataset,
                    ),
                    args.checkpoint,
                )
            else:
                epochs_without_improvement += 1

            marker = "*" if improved else " "

            print(
                f"{marker} Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f} "
                f"AUPRC={train_metrics['auprc']:.4f} "
                f"MCC={train_metrics['mcc']:.4f} | "
                f"val loss={validation_metrics['loss']:.4f} "
                f"AUPRC={validation_metrics['auprc']:.4f} "
                f"MCC={validation_metrics['mcc']:.4f} "
                f"F1={validation_metrics['f1']:.4f} | "
                f"no_improve={epochs_without_improvement}"
            )

            pd.DataFrame(history_rows).to_csv(
                args.history,
                index=False,
            )

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping activated at epoch {epoch}. "
                    f"Best epoch: {best_epoch}."
                )
                break

        total_seconds = time.time() - training_start

        if not args.checkpoint.exists():
            raise RuntimeError("No checkpoint was saved during training.")

        summary = {
            "model_name": "PromoterCNNBaseline",
            "device": str(device),
            "random_seed": args.seed,
            "test_set_used": False,
            "selection_metric": "validation_auprc",
            "best_epoch": best_epoch,
            "best_validation_auprc": best_validation_auprc,
            "epochs_completed": len(history_rows),
            "early_stopping_triggered": (
                len(history_rows) < args.epochs
            ),
            "training_seconds": total_seconds,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "train_positive": train_dataset.positive_count,
            "train_negative": train_dataset.negative_count,
            "train_positive_weight": train_dataset.positive_weight,
            "trainable_parameters": count_trainable_parameters(model),
            "model_configuration": config.to_dict(),
            "training_configuration": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "maximum_epochs": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "classification_threshold": args.threshold,
                "optimizer": "AdamW",
                "loss": "BCEWithLogitsLoss",
            },
            "checkpoint": str(args.checkpoint),
            "history": str(args.history),
        }

        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print("=" * 88)
        print("TRAINING COMPLETE")
        print("=" * 88)
        print(f"Best epoch:             {best_epoch}")
        print(f"Best validation AUPRC:  {best_validation_auprc:.6f}")
        print(f"Checkpoint:             {args.checkpoint}")
        print(f"History:                {args.history}")
        print(f"Summary:                {args.summary}")
        print(f"Total training seconds: {total_seconds:.2f}")
        print("=" * 88)

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
