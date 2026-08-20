#!/usr/bin/env python3
"""Diagnose the best CNN baseline checkpoint on train and validation only.

This script does not load or evaluate the test set.

It reports:
    - loss and threshold-independent metrics;
    - probability distributions by class;
    - metrics across a predefined threshold grid;
    - positive-response subclass probability summaries;
    - prediction tables for train and validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.evaluation.metrics import binary_classification_metrics
from src.models.cnn_baseline import (
    CNNBaselineConfig,
    PromoterCNNBaseline,
)


THRESHOLD_GRID = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the best CNN checkpoint without using test data."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cnn_baseline_best.pt"),
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
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/metrics/cnn_baseline_diagnostic_summary.json"
        ),
    )
    parser.add_argument(
        "--threshold-table",
        type=Path,
        default=Path(
            "results/tables/cnn_baseline_threshold_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/tables/cnn_baseline_predictions"),
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "model_config",
        "epoch",
        "best_validation_auprc",
        "train_positive_weight",
    }
    missing = required - set(checkpoint)

    if missing:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )

    return checkpoint


def build_model(checkpoint: dict) -> PromoterCNNBaseline:
    config_dict = checkpoint["model_config"]

    config = CNNBaselineConfig(
        input_channels=int(config_dict["input_channels"]),
        conv_channels=tuple(config_dict["conv_channels"]),
        kernel_sizes=tuple(config_dict["kernel_sizes"]),
        pool_size=int(config_dict["pool_size"]),
        dense_units=int(config_dict["dense_units"]),
        dropout=float(config_dict["dropout"]),
    )

    model = PromoterCNNBaseline(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def make_loader(dataset: PromoterDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def collect_predictions(
    model: nn.Module,
    dataset: PromoterDataset,
    batch_size: int,
    positive_weight: float,
) -> tuple[pd.DataFrame, float]:
    loader = make_loader(dataset, batch_size)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
    )

    logits_all: list[float] = []
    probabilities_all: list[float] = []
    labels_all: list[float] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)

            batch_size_current = labels.shape[0]
            total_loss += float(loss.cpu()) * batch_size_current
            total_samples += batch_size_current

            logits_all.extend(logits.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    if total_samples == 0:
        raise ValueError("No samples were evaluated.")

    output = dataset.dataframe[
        [
            "sample_id",
            "sequence_id",
            "label",
            "response_class",
            "gc_content",
        ]
    ].copy()

    output["logit"] = logits_all
    output["probability"] = probabilities_all

    observed_labels = np.asarray(labels_all, dtype=int)

    if not np.array_equal(
        output["label"].to_numpy(dtype=int),
        observed_labels,
    ):
        raise ValueError("Prediction order does not match dataset order.")

    return output, total_loss / total_samples


def probability_summary(
    dataframe: pd.DataFrame,
) -> dict[str, float | int | None]:
    values = dataframe["probability"].to_numpy(dtype=float)

    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def evaluate_thresholds(
    split_name: str,
    predictions: pd.DataFrame,
    thresholds: tuple[float, ...],
) -> list[dict[str, float | str | int]]:
    rows = []

    targets = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)

    for threshold in thresholds:
        metrics = binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=threshold,
        )

        predicted_positive = int(
            (probabilities >= threshold).sum()
        )

        rows.append(
            {
                "split": split_name,
                **metrics,
                "predicted_positive": predicted_positive,
                "predicted_positive_fraction": float(
                    predicted_positive / len(predictions)
                ),
            }
        )

    return rows


def split_summary(
    split_name: str,
    predictions: pd.DataFrame,
    weighted_loss: float,
) -> dict:
    targets = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)

    prevalence = float(targets.mean())

    metrics_at_05 = binary_classification_metrics(
        targets=targets,
        probabilities=probabilities,
        threshold=0.5,
    )

    negative = predictions.loc[predictions["label"] == 0]
    positive = predictions.loc[predictions["label"] == 1]

    subclass_summaries = {}

    for response_class, group in positive.groupby(
        "response_class",
        dropna=False,
    ):
        key = "NA" if pd.isna(response_class) else str(response_class)
        subclass_summaries[key] = probability_summary(group)

    return {
        "split": split_name,
        "samples": int(len(predictions)),
        "positive": int((predictions["label"] == 1).sum()),
        "negative": int((predictions["label"] == 0).sum()),
        "positive_prevalence": prevalence,
        "random_auprc_baseline": prevalence,
        "weighted_loss": float(weighted_loss),
        "metrics_at_threshold_0_5": metrics_at_05,
        "negative_probability_summary": probability_summary(negative),
        "positive_probability_summary": probability_summary(positive),
        "positive_response_class_probability_summary": subclass_summaries,
    }


def main() -> int:
    args = parse_args()

    try:
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive.")

        checkpoint = load_checkpoint(args.checkpoint)
        model = build_model(checkpoint)

        train_dataset = PromoterDataset(
            args.train,
            return_metadata=False,
        )
        validation_dataset = PromoterDataset(
            args.validation,
            return_metadata=False,
        )

        positive_weight = float(checkpoint["train_positive_weight"])

        train_predictions, train_loss = collect_predictions(
            model=model,
            dataset=train_dataset,
            batch_size=args.batch_size,
            positive_weight=positive_weight,
        )

        validation_predictions, validation_loss = collect_predictions(
            model=model,
            dataset=validation_dataset,
            batch_size=args.batch_size,
            positive_weight=positive_weight,
        )

        args.predictions_dir.mkdir(parents=True, exist_ok=True)
        args.threshold_table.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        train_predictions.to_csv(
            args.predictions_dir / "train_predictions.csv",
            index=False,
        )
        validation_predictions.to_csv(
            args.predictions_dir / "validation_predictions.csv",
            index=False,
        )

        threshold_rows = []
        threshold_rows.extend(
            evaluate_thresholds(
                "train",
                train_predictions,
                THRESHOLD_GRID,
            )
        )
        threshold_rows.extend(
            evaluate_thresholds(
                "validation",
                validation_predictions,
                THRESHOLD_GRID,
            )
        )

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(args.threshold_table, index=False)

        summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_best_validation_auprc": float(
                checkpoint["best_validation_auprc"]
            ),
            "train_positive_weight": positive_weight,
            "test_set_used": False,
            "model_mode": "eval",
            "threshold_grid": list(THRESHOLD_GRID),
            "train": split_summary(
                "train",
                train_predictions,
                train_loss,
            ),
            "validation": split_summary(
                "validation",
                validation_predictions,
                validation_loss,
            ),
        }

        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        train_metrics = summary["train"]["metrics_at_threshold_0_5"]
        val_metrics = summary["validation"]["metrics_at_threshold_0_5"]

        print("=" * 88)
        print("CNN BASELINE CHECKPOINT DIAGNOSTIC")
        print("=" * 88)
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print("Test set used:    False")
        print()

        print(
            "Train      | "
            f"loss={train_loss:.4f} "
            f"AUPRC={train_metrics['auprc']:.4f} "
            f"AUROC={train_metrics['auroc']:.4f} "
            f"MCC={train_metrics['mcc']:.4f} "
            f"F1={train_metrics['f1']:.4f}"
        )
        print(
            "Validation | "
            f"loss={validation_loss:.4f} "
            f"AUPRC={val_metrics['auprc']:.4f} "
            f"AUROC={val_metrics['auroc']:.4f} "
            f"MCC={val_metrics['mcc']:.4f} "
            f"F1={val_metrics['f1']:.4f}"
        )

        print()
        print("Validation probability means:")
        print(
            "  Negative: "
            f"{summary['validation']['negative_probability_summary']['mean']:.4f}"
        )
        print(
            "  Positive: "
            f"{summary['validation']['positive_probability_summary']['mean']:.4f}"
        )

        print()
        print(f"Summary:         {args.summary}")
        print(f"Threshold table: {args.threshold_table}")
        print(f"Predictions:     {args.predictions_dir}")
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
