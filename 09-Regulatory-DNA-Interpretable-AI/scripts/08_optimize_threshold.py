#!/usr/bin/env python3
"""Optimize the classification threshold on validation predictions only.

The script loads the best checkpoint from the pos_weight ablation study,
generates validation probabilities, evaluates a dense threshold grid, and
selects the threshold using:

    1. maximum MCC;
    2. maximum F1 as first tie-breaker;
    3. maximum balanced accuracy as second tie-breaker;
    4. threshold closest to 0.5 as final deterministic tie-breaker.

The test set is never loaded or used.
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
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.evaluation.metrics import binary_classification_metrics
from src.models.cnn_baseline import CNNBaselineConfig, PromoterCNNBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a binary decision threshold using validation only."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/pos_weight_ablation/"
            "pos_weight_1/best_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path(
            "results/tables/"
            "cnn_baseline_threshold_optimization.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/metrics/"
            "cnn_baseline_threshold_selection.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/tables/"
            "cnn_baseline_validation_predictions_pos_weight_1.csv"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold_min < 1.0:
        raise ValueError("threshold-min must lie between 0 and 1.")
    if not 0.0 < args.threshold_max < 1.0:
        raise ValueError("threshold-max must lie between 0 and 1.")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("threshold-min must be smaller than threshold-max.")
    if args.threshold_step <= 0:
        raise ValueError("threshold-step must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")


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
        "pos_weight",
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


def collect_predictions(
    model: PromoterCNNBaseline,
    dataset: PromoterDataset,
    batch_size: int,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    logits_all: list[float] = []
    probabilities_all: list[float] = []

    with torch.no_grad():
        for features, _ in loader:
            logits = model(features)
            probabilities = torch.sigmoid(logits)

            logits_all.extend(logits.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())

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

    return output


def build_threshold_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> np.ndarray:
    number = int(math.floor((maximum - minimum) / step)) + 1
    thresholds = minimum + np.arange(number, dtype=float) * step
    thresholds = thresholds[thresholds <= maximum + 1e-12]
    return np.round(thresholds, 10)


def evaluate_threshold_grid(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        metrics = binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        predictions = probabilities >= threshold

        true_positive = int(
            np.sum((predictions == 1) & (targets == 1))
        )
        false_positive = int(
            np.sum((predictions == 1) & (targets == 0))
        )
        true_negative = int(
            np.sum((predictions == 0) & (targets == 0))
        )
        false_negative = int(
            np.sum((predictions == 0) & (targets == 1))
        )

        rows.append(
            {
                **metrics,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "predicted_positive": int(predictions.sum()),
                "predicted_positive_fraction": float(
                    predictions.mean()
                ),
                "distance_from_0_5": abs(float(threshold) - 0.5),
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(table: pd.DataFrame) -> pd.Series:
    ranked = table.sort_values(
        by=[
            "mcc",
            "f1",
            "balanced_accuracy",
            "distance_from_0_5",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    return ranked.iloc[0]


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

        checkpoint = load_checkpoint(args.checkpoint)
        model = build_model(checkpoint)

        validation_dataset = PromoterDataset(
            args.validation,
            return_metadata=False,
        )

        predictions = collect_predictions(
            model=model,
            dataset=validation_dataset,
            batch_size=args.batch_size,
        )

        targets = predictions["label"].to_numpy(dtype=int)
        probabilities = predictions["probability"].to_numpy(dtype=float)

        thresholds = build_threshold_grid(
            minimum=args.threshold_min,
            maximum=args.threshold_max,
            step=args.threshold_step,
        )

        table = evaluate_threshold_grid(
            targets=targets,
            probabilities=probabilities,
            thresholds=thresholds,
        )

        best = select_best_threshold(table)

        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)

        table.to_csv(args.table, index=False)
        predictions.to_csv(args.predictions, index=False)

        summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_pos_weight": float(checkpoint["pos_weight"]),
            "checkpoint_best_validation_auprc": float(
                checkpoint["best_validation_auprc"]
            ),
            "test_set_used": False,
            "selection_dataset": "validation",
            "selection_metric_primary": "mcc",
            "tie_breakers": [
                "f1",
                "balanced_accuracy",
                "distance_from_0_5",
                "lower_threshold",
            ],
            "threshold_grid": {
                "minimum": args.threshold_min,
                "maximum": args.threshold_max,
                "step": args.threshold_step,
                "evaluated_thresholds": int(len(table)),
            },
            "validation_samples": int(len(predictions)),
            "validation_positive": int(
                (predictions["label"] == 1).sum()
            ),
            "validation_negative": int(
                (predictions["label"] == 0).sum()
            ),
            "selected_threshold": float(best["threshold"]),
            "selected_metrics": {
                "auprc": float(best["auprc"]),
                "auroc": float(best["auroc"]),
                "mcc": float(best["mcc"]),
                "f1": float(best["f1"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "balanced_accuracy": float(
                    best["balanced_accuracy"]
                ),
                "accuracy": float(best["accuracy"]),
                "true_positive": int(best["true_positive"]),
                "false_positive": int(best["false_positive"]),
                "true_negative": int(best["true_negative"]),
                "false_negative": int(best["false_negative"]),
                "predicted_positive": int(
                    best["predicted_positive"]
                ),
                "predicted_positive_fraction": float(
                    best["predicted_positive_fraction"]
                ),
            },
            "threshold_table": str(args.table),
            "validation_predictions": str(args.predictions),
            "policy": (
                "Threshold selected on validation only and must remain "
                "frozen during final test evaluation."
            ),
        }

        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 88)
        print("VALIDATION THRESHOLD OPTIMIZATION COMPLETE")
        print("=" * 88)
        print(f"Checkpoint epoch:       {checkpoint['epoch']}")
        print(f"Checkpoint pos_weight:  {checkpoint['pos_weight']}")
        print("Test set used:          False")
        print()
        print(f"Selected threshold:     {best['threshold']:.6f}")
        print(f"Validation MCC:         {best['mcc']:.6f}")
        print(f"Validation F1:          {best['f1']:.6f}")
        print(f"Validation precision:   {best['precision']:.6f}")
        print(f"Validation recall:      {best['recall']:.6f}")
        print(
            f"Balanced accuracy:      "
            f"{best['balanced_accuracy']:.6f}"
        )
        print(
            "Confusion matrix:        "
            f"TP={int(best['true_positive'])}, "
            f"FP={int(best['false_positive'])}, "
            f"TN={int(best['true_negative'])}, "
            f"FN={int(best['false_negative'])}"
        )
        print()
        print(f"Threshold table:        {args.table}")
        print(f"Summary:                {args.summary}")
        print(f"Predictions:            {args.predictions}")
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
