#!/usr/bin/env python3
"""Run a controlled pos_weight ablation study for the CNN baseline.

Only the positive-class weight is varied. Architecture, data partitions,
random seed, optimizer, learning rate, dropout and early-stopping settings
remain fixed.

The test set is never loaded.
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


DEFAULT_WEIGHTS = (9.952941176470588, 5.0, 3.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled pos_weight ablation study."
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
        "--output-dir",
        type=Path,
        default=Path("experiments/pos_weight_ablation"),
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=Path(
            "results/tables/pos_weight_ablation_summary.csv"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(
            "results/metrics/pos_weight_ablation_summary.json"
        ),
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
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
    if any(weight <= 0 for weight in args.weights):
        raise ValueError("All pos_weight values must be positive.")
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


def set_reproducibility(seed: int) -> None:
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
    optimizer: AdamW | None,
    threshold: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    targets: list[float] = []
    probabilities: list[float] = []

    for features, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        current_batch = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * current_batch
        total_samples += current_batch

        probs = torch.sigmoid(logits)
        targets.extend(labels.detach().cpu().tolist())
        probabilities.extend(probs.detach().cpu().tolist())

    metrics = binary_classification_metrics(
        targets=targets,
        probabilities=probabilities,
        threshold=threshold,
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def experiment_tag(weight: float) -> str:
    text = f"{weight:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def train_one_weight(
    args: argparse.Namespace,
    weight: float,
    train_dataset: PromoterDataset,
    validation_dataset: PromoterDataset,
) -> dict:
    set_reproducibility(args.seed)

    tag = experiment_tag(weight)
    experiment_dir = args.output_dir / f"pos_weight_{tag}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = experiment_dir / "best_checkpoint.pt"
    history_path = experiment_dir / "training_history.csv"
    summary_path = experiment_dir / "summary.json"

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.seed,
        args.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        False,
        args.seed + 1,
        args.num_workers,
    )

    config = CNNBaselineConfig(
        input_channels=4,
        conv_channels=(32, 64, 128),
        kernel_sizes=(15, 11, 7),
        pool_size=4,
        dense_units=64,
        dropout=0.40,
    )

    model = PromoterCNNBaseline(config=config)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight], dtype=torch.float32)
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_auprc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    print("-" * 88)
    print(f"pos_weight={weight:.6f}")
    print("-" * 88)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            args.threshold,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            None,
            args.threshold,
        )

        row: dict[str, float | int] = {"epoch": epoch}

        for key, value in train_metrics.items():
            row[f"train_{key}"] = value

        for key, value in validation_metrics.items():
            row[f"validation_{key}"] = value

        history.append(row)

        current_auprc = validation_metrics["auprc"]
        improved = current_auprc > best_auprc + args.min_delta

        if improved:
            best_auprc = current_auprc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": config.to_dict(),
                    "pos_weight": weight,
                    "best_validation_auprc": best_auprc,
                    "selection_metric": "validation_auprc",
                    "test_set_used": False,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        marker = "*" if improved else " "
        print(
            f"{marker} Epoch {epoch:03d} | "
            f"train AUPRC={train_metrics['auprc']:.4f} "
            f"MCC={train_metrics['mcc']:.4f} | "
            f"val AUPRC={validation_metrics['auprc']:.4f} "
            f"MCC={validation_metrics['mcc']:.4f} "
            f"F1={validation_metrics['f1']:.4f} | "
            f"no_improve={epochs_without_improvement}"
        )

        pd.DataFrame(history).to_csv(history_path, index=False)

        if epochs_without_improvement >= args.patience:
            break

    elapsed = time.time() - start_time
    history_df = pd.DataFrame(history)

    best_row = history_df.loc[
        history_df["epoch"] == best_epoch
    ].iloc[0]

    summary = {
        "pos_weight": float(weight),
        "best_epoch": int(best_epoch),
        "best_validation_auprc": float(best_auprc),
        "best_validation_auroc": float(best_row["validation_auroc"]),
        "best_validation_mcc_at_0_5": float(best_row["validation_mcc"]),
        "best_validation_f1_at_0_5": float(best_row["validation_f1"]),
        "best_validation_precision_at_0_5": float(
            best_row["validation_precision"]
        ),
        "best_validation_recall_at_0_5": float(
            best_row["validation_recall"]
        ),
        "best_validation_balanced_accuracy_at_0_5": float(
            best_row["validation_balanced_accuracy"]
        ),
        "best_validation_loss": float(best_row["validation_loss"]),
        "train_auprc_at_best_epoch": float(best_row["train_auprc"]),
        "epochs_completed": int(len(history_df)),
        "early_stopping_triggered": bool(len(history_df) < args.epochs),
        "training_seconds": float(elapsed),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "test_set_used": False,
        "fixed_configuration": {
            "seed": args.seed,
            "architecture": config.to_dict(),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "maximum_epochs": args.epochs,
            "patience": args.patience,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

        train_dataset = PromoterDataset(args.train)
        validation_dataset = PromoterDataset(args.validation)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.summary_table.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)

        summaries = []

        print("=" * 88)
        print("POS_WEIGHT ABLATION STUDY")
        print("=" * 88)
        print(f"Weights: {args.weights}")
        print(f"Seed:    {args.seed}")
        print("Test set used: False")
        print("=" * 88)

        for weight in args.weights:
            summaries.append(
                train_one_weight(
                    args,
                    float(weight),
                    train_dataset,
                    validation_dataset,
                )
            )

        summary_df = pd.DataFrame(summaries).sort_values(
            "best_validation_auprc",
            ascending=False,
        )
        summary_df.to_csv(args.summary_table, index=False)

        best = summary_df.iloc[0].to_dict()

        global_summary = {
            "study_name": "pos_weight_ablation",
            "test_set_used": False,
            "selection_metric": "best_validation_auprc",
            "weights_evaluated": [float(value) for value in args.weights],
            "best_configuration": best,
            "all_results": summaries,
        }

        args.summary_json.write_text(
            json.dumps(
                global_summary,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        print("=" * 88)
        print("POS_WEIGHT ABLATION COMPLETE")
        print("=" * 88)
        print(
            summary_df[
                [
                    "pos_weight",
                    "best_epoch",
                    "best_validation_auprc",
                    "best_validation_auroc",
                    "best_validation_mcc_at_0_5",
                    "best_validation_f1_at_0_5",
                ]
            ].to_string(index=False)
        )
        print()
        print(f"Best pos_weight: {best['pos_weight']}")
        print(
            "Best validation AUPRC: "
            f"{best['best_validation_auprc']:.6f}"
        )
        print(f"Summary table: {args.summary_table}")
        print(f"Summary JSON:  {args.summary_json}")
        print("=" * 88)
        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
