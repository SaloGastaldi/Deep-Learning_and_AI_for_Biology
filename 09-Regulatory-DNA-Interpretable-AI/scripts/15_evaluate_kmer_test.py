#!/usr/bin/env python3
"""Final frozen-test evaluation of the fixed 5-mer logistic-regression model.

Purpose
-------
Evaluate whether the sequence-only 5-mer model that showed significant signal
in grouped development cross-validation and permutation testing generalizes to
the untouched hold-out Test set.

Critical methodological rule
----------------------------
This script performs NO hyperparameter tuning, feature selection, threshold
optimization, model comparison, or test-driven decision making.

The model configuration is frozen beforehand:
- feature representation: L2-normalized character 5-mer frequencies
- classifier: LogisticRegression
- C = 10.0
- class_weight = None
- solver = liblinear
- random_state = 123

Training data:
    Train + Validation

Evaluation data:
    Test only

Primary metrics:
    AUPRC and AUROC (threshold-independent)

Secondary metrics at the fixed conventional threshold 0.5:
    MCC, F1, Precision, Recall, Balanced Accuracy, Accuracy

The frozen Test set must not be reused to tune this model after this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--test",
        type=Path,
        default=Path("data/splits/test.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "results/metrics/kmer_logistic_test_summary.json"
        ),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path(
            "results/tables/kmer_logistic_test_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=Path(
            "results/tables/kmer_logistic_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--pr-figure",
        type=Path,
        default=Path(
            "results/figures/"
            "precision_recall_curve_kmer_test.png"
        ),
    )
    parser.add_argument(
        "--roc-figure",
        type=Path,
        default=Path(
            "results/figures/roc_curve_kmer_test.png"
        ),
    )
    parser.add_argument(
        "--confusion-figure",
        type=Path,
        default=Path(
            "results/figures/confusion_matrix_kmer_test.png"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_vocabulary(k: int) -> dict[str, int]:
    kmers = (
        "".join(chars)
        for chars in itertools.product("ACGT", repeat=k)
    )
    return {
        kmer: index
        for index, kmer in enumerate(kmers)
    }


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    analyzer="char",
                    lowercase=False,
                    ngram_range=(5, 5),
                    vocabulary=build_vocabulary(5),
                    dtype=np.float64,
                ),
            ),
            (
                "frequency_normalizer",
                TfidfTransformer(
                    norm="l2",
                    use_idf=False,
                    smooth_idf=False,
                    sublinear_tf=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=10.0,
                    class_weight=None,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} split not found: {path}"
        )

    dataframe = pd.read_csv(path).copy()

    required = {
        "sample_id",
        "sequence_id",
        "sequence",
        "label",
        "response_class",
        "sequence_length",
        "sequence_hash",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{split_name} is missing columns: "
            f"{sorted(missing)}"
        )

    dataframe["sequence"] = (
        dataframe["sequence"]
        .astype(str)
        .str.upper()
    )

    if set(dataframe["label"].unique()) - {0, 1}:
        raise ValueError(
            f"{split_name}: labels must be 0/1."
        )

    if not (dataframe["sequence_length"] == 2200).all():
        raise ValueError(
            f"{split_name}: all sequences must be 2200 bp."
        )

    if dataframe["sequence"].str.contains(
        r"[^ACGTN]",
        regex=True,
    ).any():
        raise ValueError(
            f"{split_name}: invalid nucleotide symbols."
        )

    if dataframe["sample_id"].duplicated().any():
        raise ValueError(
            f"{split_name}: duplicated sample IDs."
        )

    if dataframe["sequence_hash"].duplicated().any():
        raise ValueError(
            f"{split_name}: exact duplicated sequences."
        )

    dataframe["split"] = split_name
    return dataframe


def validate_partition_independence(
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    shared_ids = (
        set(development["sample_id"])
        & set(test["sample_id"])
    )

    if shared_ids:
        raise ValueError(
            "Sample IDs overlap between development and Test."
        )

    shared_hashes = (
        set(development["sequence_hash"])
        & set(test["sequence_hash"])
    )

    if shared_hashes:
        raise ValueError(
            "Exact sequence overlap between development and Test."
        )


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    return {
        "auprc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "auroc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                labels,
                predictions,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "threshold": float(threshold),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "predicted_positive": int(
            predictions.sum()
        ),
    }


def save_pr_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    prevalence: float,
    path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(
        labels,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(recall, precision, linewidth=2)
    axis.axhline(
        prevalence,
        linestyle="--",
        linewidth=1,
        label=f"Random baseline = {prevalence:.3f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        "5-mer Logistic Regression — Test Precision–Recall"
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(
        "5-mer Logistic Regression — Test ROC"
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def save_confusion_matrix(
    metrics: dict,
    path: Path,
) -> None:
    matrix = np.array(
        [
            [
                metrics["true_negative"],
                metrics["false_positive"],
            ],
            [
                metrics["false_negative"],
                metrics["true_positive"],
            ],
        ],
        dtype=int,
    )

    figure, axis = plt.subplots(figsize=(5, 5))
    image = axis.imshow(matrix)

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=14,
            )

    axis.set_xticks(
        [0, 1],
        labels=["Predicted 0", "Predicted 1"],
    )
    axis.set_yticks(
        [0, 1],
        labels=["True 0", "True 1"],
    )
    axis.set_title(
        "5-mer Logistic Regression — Test Confusion Matrix"
    )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> int:
    args = parse_args()

    try:
        set_seed(args.seed)

        for path in (
            args.output_summary,
            args.output_metrics,
            args.output_predictions,
            args.pr_figure,
            args.roc_figure,
            args.confusion_figure,
        ):
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        train = load_split(
            args.train,
            "train",
        )
        validation = load_split(
            args.validation,
            "validation",
        )
        test = load_split(
            args.test,
            "test",
        )

        development = pd.concat(
            [train, validation],
            ignore_index=True,
        )

        validate_partition_independence(
            development=development,
            test=test,
        )

        development_sequences = (
            development["sequence"]
            .to_numpy(dtype=str)
        )
        development_labels = (
            development["label"]
            .to_numpy(dtype=int)
        )

        test_sequences = (
            test["sequence"]
            .to_numpy(dtype=str)
        )
        test_labels = (
            test["label"]
            .to_numpy(dtype=int)
        )

        prevalence = float(
            test_labels.mean()
        )

        print("=" * 88)
        print(
            "FINAL TEST EVALUATION — FIXED 5-MER LOGISTIC REGRESSION"
        )
        print("=" * 88)
        print(
            f"Development samples:  {len(development)}"
        )
        print(
            f"  Positive:           "
            f"{int((development_labels == 1).sum())}"
        )
        print(
            f"  Negative:           "
            f"{int((development_labels == 0).sum())}"
        )
        print(
            f"Test samples:         {len(test)}"
        )
        print(
            f"  Positive:           "
            f"{int((test_labels == 1).sum())}"
        )
        print(
            f"  Negative:           "
            f"{int((test_labels == 0).sum())}"
        )
        print(
            f"Test prevalence:      {prevalence:.6f}"
        )
        print(
            f"Random AUPRC baseline:{prevalence:.6f}"
        )
        print()
        print(
            "Frozen model:         5-mers, C=10.0, "
            "class_weight=None"
        )
        print(
            "Threshold tuning:     False"
        )
        print(
            "Model selection on Test: False"
        )
        print("=" * 88)

        model = build_model(
            seed=args.seed,
        )

        start = time.time()

        model.fit(
            development_sequences,
            development_labels,
        )

        probabilities = model.predict_proba(
            test_sequences
        )[:, 1]

        runtime_seconds = (
            time.time() - start
        )

        metrics = calculate_metrics(
            labels=test_labels,
            probabilities=probabilities,
            threshold=0.5,
        )

        predictions = test[
            [
                "sample_id",
                "sequence_id",
                "label",
                "response_class",
                "sequence_hash",
            ]
        ].copy()

        predictions["probability"] = probabilities
        predictions["prediction_at_0_5"] = (
            probabilities >= 0.5
        ).astype(int)

        predictions.to_csv(
            args.output_predictions,
            index=False,
        )

        metrics_table = pd.DataFrame(
            [
                {
                    "model": (
                        "5mer_logistic_regression"
                    ),
                    "test_samples": len(test),
                    "test_positive": int(
                        (test_labels == 1).sum()
                    ),
                    "test_negative": int(
                        (test_labels == 0).sum()
                    ),
                    "test_prevalence": prevalence,
                    "random_auprc_baseline": prevalence,
                    **metrics,
                }
            ]
        )

        metrics_table.to_csv(
            args.output_metrics,
            index=False,
        )

        save_pr_curve(
            labels=test_labels,
            probabilities=probabilities,
            prevalence=prevalence,
            path=args.pr_figure,
        )

        save_roc_curve(
            labels=test_labels,
            probabilities=probabilities,
            path=args.roc_figure,
        )

        save_confusion_matrix(
            metrics=metrics,
            path=args.confusion_figure,
        )

        summary = {
            "experiment_name": (
                "final_frozen_test_evaluation_"
                "5mer_logistic_regression"
            ),
            "test_set_used": True,
            "test_usage_policy": (
                "One-time final evaluation. "
                "No tuning or model selection performed."
            ),
            "model_frozen_before_test": True,
            "model_configuration": {
                "feature_type": (
                    "L2_normalized_character_5mer_frequencies"
                ),
                "kmer_size": 5,
                "possible_features": 1024,
                "classifier": (
                    "LogisticRegression"
                ),
                "C": 10.0,
                "class_weight": "none",
                "penalty": "l2",
                "solver": "liblinear",
                "random_seed": args.seed,
            },
            "training_data": {
                "sources": [
                    str(args.train),
                    str(args.validation),
                ],
                "samples": int(
                    len(development)
                ),
                "positive": int(
                    (development_labels == 1).sum()
                ),
                "negative": int(
                    (development_labels == 0).sum()
                ),
            },
            "test_data": {
                "source": str(args.test),
                "samples": int(
                    len(test)
                ),
                "positive": int(
                    (test_labels == 1).sum()
                ),
                "negative": int(
                    (test_labels == 0).sum()
                ),
                "positive_prevalence": prevalence,
                "random_auprc_baseline": prevalence,
            },
            "evaluation_policy": {
                "primary_metrics": [
                    "AUPRC",
                    "AUROC",
                ],
                "secondary_metrics": [
                    "MCC",
                    "F1",
                    "Precision",
                    "Recall",
                    "Balanced Accuracy",
                    "Accuracy",
                ],
                "classification_threshold": 0.5,
                "threshold_optimized_on_test": False,
                "hyperparameters_tuned_on_test": False,
                "features_selected_on_test": False,
            },
            "metrics": metrics,
            "runtime_seconds": float(
                runtime_seconds
            ),
            "outputs": {
                "summary": str(
                    args.output_summary
                ),
                "metrics_table": str(
                    args.output_metrics
                ),
                "predictions": str(
                    args.output_predictions
                ),
                "precision_recall_curve": str(
                    args.pr_figure
                ),
                "roc_curve": str(
                    args.roc_figure
                ),
                "confusion_matrix": str(
                    args.confusion_figure
                ),
            },
        }

        args.output_summary.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 88)
        print("FINAL TEST RESULTS")
        print("=" * 88)
        print(
            f"AUPRC:             {metrics['auprc']:.6f}"
        )
        print(
            f"Random AUPRC:      {prevalence:.6f}"
        )
        print(
            f"AUROC:             {metrics['auroc']:.6f}"
        )
        print(
            f"MCC @ 0.5:         {metrics['mcc']:.6f}"
        )
        print(
            f"F1 @ 0.5:          {metrics['f1']:.6f}"
        )
        print(
            f"Precision @ 0.5:   {metrics['precision']:.6f}"
        )
        print(
            f"Recall @ 0.5:      {metrics['recall']:.6f}"
        )
        print(
            f"Balanced Acc.:     "
            f"{metrics['balanced_accuracy']:.6f}"
        )
        print(
            f"Accuracy:          {metrics['accuracy']:.6f}"
        )
        print()
        print(
            "Confusion matrix:"
        )
        print(
            f"  TP={metrics['true_positive']} "
            f"FP={metrics['false_positive']} "
            f"TN={metrics['true_negative']} "
            f"FN={metrics['false_negative']}"
        )
        print()
        print(
            f"Summary:           {args.output_summary}"
        )
        print(
            f"Predictions:       {args.output_predictions}"
        )
        print(
            f"Runtime seconds:   {runtime_seconds:.2f}"
        )
        print("=" * 88)

        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        KeyError,
        pd.errors.ParserError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
