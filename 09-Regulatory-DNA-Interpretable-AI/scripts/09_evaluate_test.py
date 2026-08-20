#!/usr/bin/env python3
"""Evaluate the frozen CNN baseline on the test set exactly once.

This script:
    - loads the selected checkpoint;
    - reads the frozen threshold selected on validation;
    - evaluates the test set without changing the model;
    - calculates final classification metrics;
    - saves predictions, metrics and figures.

No training, threshold tuning or hyperparameter optimization is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.evaluation.metrics import binary_classification_metrics
from src.models.cnn_baseline import (
    CNNBaselineConfig,
    PromoterCNNBaseline,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate the frozen CNN baseline on the test set."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/pos_weight_ablation/"
            "pos_weight_1/best_checkpoint.pt"
        ),
        help="Selected model checkpoint.",
    )
    parser.add_argument(
        "--threshold-summary",
        type=Path,
        default=Path(
            "results/metrics/"
            "cnn_baseline_threshold_selection.json"
        ),
        help="Validation threshold-selection summary.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/splits/test.csv"),
        help="Frozen test dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/metrics/"
            "cnn_baseline_test_summary.json"
        ),
        help="JSON output with final test results.",
    )
    parser.add_argument(
        "--metrics-table",
        type=Path,
        default=Path(
            "results/tables/"
            "cnn_baseline_test_metrics.csv"
        ),
        help="CSV output with final test metrics.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/tables/"
            "cnn_baseline_test_predictions.csv"
        ),
        help="CSV output with per-sample test predictions.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory for evaluation figures.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")


def load_checkpoint(path: Path) -> dict:
    """Load and validate the selected checkpoint."""

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_fields = {
        "model_state_dict",
        "model_config",
        "epoch",
        "best_validation_auprc",
        "pos_weight",
    }

    missing_fields = required_fields - set(checkpoint)

    if missing_fields:
        raise ValueError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    return checkpoint


def load_threshold_summary(path: Path) -> dict:
    """Load and validate the frozen validation threshold."""

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold-selection summary not found: {path}"
        )

    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    required_fields = {
        "selected_threshold",
        "selection_dataset",
        "selection_metric_primary",
        "test_set_used",
        "checkpoint",
    }

    missing_fields = required_fields - set(summary)

    if missing_fields:
        raise ValueError(
            "Threshold summary is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if summary["selection_dataset"] != "validation":
        raise ValueError(
            "The threshold must have been selected using validation only."
        )

    if bool(summary["test_set_used"]):
        raise ValueError(
            "Threshold summary indicates that the test set was used."
        )

    threshold = float(summary["selected_threshold"])

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Frozen threshold must lie between 0 and 1: {threshold}"
        )

    return summary


def build_model(checkpoint: dict) -> PromoterCNNBaseline:
    """Reconstruct the model architecture and load trained parameters."""

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
    """Generate logits and probabilities for the complete test set."""

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

    if len(logits_all) != len(dataset):
        raise ValueError(
            "The number of generated predictions does not match "
            "the number of test samples."
        )

    predictions = dataset.dataframe[
        [
            "sample_id",
            "sequence_id",
            "label",
            "response_class",
            "response_value",
            "gc_content",
        ]
    ].copy()

    predictions["logit"] = logits_all
    predictions["probability"] = probabilities_all

    return predictions


def add_hard_predictions(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Add frozen-threshold predictions and error categories."""

    output = predictions.copy()

    output["predicted_label"] = (
        output["probability"] >= threshold
    ).astype(int)

    output["correct_prediction"] = (
        output["predicted_label"] == output["label"]
    )

    conditions = [
        (output["label"] == 1) & (output["predicted_label"] == 1),
        (output["label"] == 0) & (output["predicted_label"] == 1),
        (output["label"] == 0) & (output["predicted_label"] == 0),
        (output["label"] == 1) & (output["predicted_label"] == 0),
    ]

    categories = [
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    output["prediction_category"] = np.select(
        conditions,
        categories,
        default="unknown",
    )

    return output


def probability_summary(
    dataframe: pd.DataFrame,
) -> dict[str, float | int | None]:
    """Summarize probability distributions."""

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


def save_precision_recall_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    prevalence: float,
    path: Path,
) -> None:
    """Save the test precision-recall curve."""

    precision, recall, _ = precision_recall_curve(
        targets,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        recall,
        precision,
        label="CNN baseline",
    )
    axis.axhline(
        prevalence,
        linestyle="--",
        label=f"Random baseline ({prevalence:.3f})",
    )

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Test Precision–Recall Curve")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    """Save the test ROC curve."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        targets,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label="CNN baseline",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Random baseline",
    )

    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Test ROC Curve")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_confusion_matrix(
    targets: np.ndarray,
    predicted_labels: np.ndarray,
    path: Path,
) -> None:
    """Save the final test confusion matrix."""

    matrix = confusion_matrix(
        targets,
        predicted_labels,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )

    figure, axis = plt.subplots(figsize=(6, 6))
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )
    axis.set_title("Test Confusion Matrix")

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> int:
    """Run the frozen final test evaluation."""

    args = parse_args()

    try:
        validate_args(args)

        checkpoint = load_checkpoint(args.checkpoint)
        threshold_summary = load_threshold_summary(
            args.threshold_summary
        )

        threshold_checkpoint = Path(
            str(threshold_summary["checkpoint"])
        )

        if threshold_checkpoint != args.checkpoint:
            raise ValueError(
                "The checkpoint supplied for test evaluation does not "
                "match the checkpoint used during threshold selection.\n"
                f"Evaluation checkpoint: {args.checkpoint}\n"
                f"Threshold checkpoint:  {threshold_checkpoint}"
            )

        threshold = float(
            threshold_summary["selected_threshold"]
        )

        model = build_model(checkpoint)

        test_dataset = PromoterDataset(
            args.test,
            return_metadata=False,
        )

        predictions = collect_predictions(
            model=model,
            dataset=test_dataset,
            batch_size=args.batch_size,
        )

        predictions = add_hard_predictions(
            predictions=predictions,
            threshold=threshold,
        )

        targets = predictions["label"].to_numpy(dtype=int)
        probabilities = predictions["probability"].to_numpy(
            dtype=float
        )
        predicted_labels = predictions[
            "predicted_label"
        ].to_numpy(dtype=int)

        metrics = binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=threshold,
        )

        matrix = confusion_matrix(
            targets,
            predicted_labels,
            labels=[0, 1],
        )

        true_negative = int(matrix[0, 0])
        false_positive = int(matrix[0, 1])
        false_negative = int(matrix[1, 0])
        true_positive = int(matrix[1, 1])

        prevalence = float(targets.mean())

        positive_predictions = predictions.loc[
            predictions["label"] == 1
        ]
        negative_predictions = predictions.loc[
            predictions["label"] == 0
        ]

        response_class_summaries = {}

        for response_class, group in positive_predictions.groupby(
            "response_class",
            dropna=False,
        ):
            key = (
                "NA"
                if pd.isna(response_class)
                else str(response_class)
            )
            response_class_summaries[key] = probability_summary(
                group
            )

        args.summary.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.metrics_table.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.predictions.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_csv(
            args.predictions,
            index=False,
        )

        metrics_row = {
            **metrics,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "test_samples": int(len(predictions)),
            "test_positive": int((targets == 1).sum()),
            "test_negative": int((targets == 0).sum()),
            "positive_prevalence": prevalence,
            "random_auprc_baseline": prevalence,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_pos_weight": float(
                checkpoint["pos_weight"]
            ),
        }

        pd.DataFrame([metrics_row]).to_csv(
            args.metrics_table,
            index=False,
        )

        precision_recall_path = (
            args.figures_dir
            / "precision_recall_curve_test.png"
        )
        roc_path = (
            args.figures_dir
            / "roc_curve_test.png"
        )
        confusion_matrix_path = (
            args.figures_dir
            / "confusion_matrix_test.png"
        )

        save_precision_recall_curve(
            targets=targets,
            probabilities=probabilities,
            prevalence=prevalence,
            path=precision_recall_path,
        )

        save_roc_curve(
            targets=targets,
            probabilities=probabilities,
            path=roc_path,
        )

        save_confusion_matrix(
            targets=targets,
            predicted_labels=predicted_labels,
            path=confusion_matrix_path,
        )

        summary = {
            "evaluation_name": (
                "cnn_baseline_final_test_evaluation"
            ),
            "evaluation_status": "FINAL",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_pos_weight": float(
                checkpoint["pos_weight"]
            ),
            "checkpoint_best_validation_auprc": float(
                checkpoint["best_validation_auprc"]
            ),
            "threshold_summary": str(
                args.threshold_summary
            ),
            "frozen_threshold": threshold,
            "threshold_selection_dataset": (
                threshold_summary["selection_dataset"]
            ),
            "threshold_selection_metric": (
                threshold_summary[
                    "selection_metric_primary"
                ]
            ),
            "test_set_used_for_model_selection": False,
            "test_evaluation_policy": (
                "Checkpoint and threshold were frozen before "
                "opening the test set."
            ),
            "test_dataset": str(args.test),
            "test_samples": int(len(predictions)),
            "test_positive": int((targets == 1).sum()),
            "test_negative": int((targets == 0).sum()),
            "positive_prevalence": prevalence,
            "random_auprc_baseline": prevalence,
            "metrics": {
                "auprc": float(metrics["auprc"]),
                "auroc": float(metrics["auroc"]),
                "mcc": float(metrics["mcc"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "balanced_accuracy": float(
                    metrics["balanced_accuracy"]
                ),
                "accuracy": float(metrics["accuracy"]),
                "threshold": threshold,
            },
            "confusion_matrix": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
            },
            "probability_summary": {
                "negative": probability_summary(
                    negative_predictions
                ),
                "positive": probability_summary(
                    positive_predictions
                ),
                "positive_response_classes": (
                    response_class_summaries
                ),
            },
            "outputs": {
                "summary": str(args.summary),
                "metrics_table": str(
                    args.metrics_table
                ),
                "predictions": str(args.predictions),
                "precision_recall_curve": str(
                    precision_recall_path
                ),
                "roc_curve": str(roc_path),
                "confusion_matrix": str(
                    confusion_matrix_path
                ),
            },
        }

        with args.summary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 88)
        print("FINAL TEST EVALUATION")
        print("=" * 88)
        print(f"Checkpoint epoch:        {checkpoint['epoch']}")
        print(f"Checkpoint pos_weight:   {checkpoint['pos_weight']}")
        print(f"Frozen threshold:        {threshold:.6f}")
        print()
        print(f"Test samples:            {len(predictions)}")
        print(f"Positive:                {(targets == 1).sum()}")
        print(f"Negative:                {(targets == 0).sum()}")
        print()
        print(f"AUPRC:                   {metrics['auprc']:.6f}")
        print(f"AUROC:                   {metrics['auroc']:.6f}")
        print(f"MCC:                     {metrics['mcc']:.6f}")
        print(f"F1:                      {metrics['f1']:.6f}")
        print(f"Precision:               {metrics['precision']:.6f}")
        print(f"Recall:                  {metrics['recall']:.6f}")
        print(
            f"Balanced Accuracy:       "
            f"{metrics['balanced_accuracy']:.6f}"
        )
        print(f"Accuracy:                {metrics['accuracy']:.6f}")
        print()
        print("Confusion matrix:")
        print(f"  TP = {true_positive}")
        print(f"  FP = {false_positive}")
        print(f"  TN = {true_negative}")
        print(f"  FN = {false_negative}")
        print()
        print(f"Summary:                 {args.summary}")
        print(f"Metrics table:           {args.metrics_table}")
        print(f"Predictions:             {args.predictions}")
        print(f"Precision–Recall curve:  {precision_recall_path}")
        print(f"ROC curve:               {roc_path}")
        print(f"Confusion matrix figure: {confusion_matrix_path}")
        print("=" * 88)

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
