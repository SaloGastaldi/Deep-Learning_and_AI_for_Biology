#!/usr/bin/env python3
"""Nested grouped cross-validation for a k-mer logistic-regression baseline.

This experiment uses only the development data (train + validation). The frozen
test set is never loaded.

Workflow
--------
1. Combine train and validation into a development dataset.
2. Cluster promoter sequences with CD-HIT-EST at 80% global identity.
3. Use CD-HIT clusters as biological groups.
4. Run nested StratifiedGroupKFold cross-validation.
5. Select k-mer and logistic-regression hyperparameters in each inner CV.
6. Evaluate each selected model on its untouched outer fold.
7. Save fold predictions, metrics, selected configurations and a final summary.

Model input
-----------
Only promoter sequence-derived k-mer frequencies are used. No motif, PWM,
transcription-factor, expression, response-class or cis-architecture feature is
provided to the classifier.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline


DEFAULT_PARAM_GRID = {
    "vectorizer__ngram_range": [
        (3, 3),
        (4, 4),
        (5, 5),
        (3, 5),
    ],
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a sequence-only k-mer logistic-regression baseline "
            "with nested grouped cross-validation."
        )
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
        "--work-dir",
        type=Path,
        default=Path("experiments/kmer_logistic_nested_cv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/tables/kmer_logistic_nested_cv_predictions.csv"
        ),
    )
    parser.add_argument(
        "--fold-results",
        type=Path,
        default=Path(
            "results/tables/kmer_logistic_nested_cv_fold_results.csv"
        ),
    )
    parser.add_argument(
        "--search-results",
        type=Path,
        default=Path(
            "results/tables/kmer_logistic_inner_search_results.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/metrics/kmer_logistic_nested_cv_summary.json"
        ),
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=0,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.outer_folds < 3:
        raise ValueError("outer-folds must be at least 3.")
    if args.inner_folds < 2:
        raise ValueError("inner-folds must be at least 2.")
    if not 0.80 <= args.identity_threshold <= 1.0:
        raise ValueError(
            "identity-threshold must be between 0.80 and 1.00."
        )
    if args.threads < 0:
        raise ValueError("threads cannot be negative.")
    if args.memory_mb < 0:
        raise ValueError("memory-mb cannot be negative.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_development_data(
    train_path: Path,
    validation_path: Path,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "sequence_id",
        "sequence",
        "label",
        "response_class",
        "sequence_hash",
        "sequence_length",
    }

    frames = []

    for source_name, path in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Development split not found: {path}")

        dataframe = pd.read_csv(path)
        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        dataframe = dataframe.copy()
        dataframe["development_source"] = source_name
        frames.append(dataframe)

    combined = pd.concat(frames, ignore_index=True)

    if combined["sample_id"].duplicated().any():
        raise ValueError("Duplicated sample IDs in development data.")

    if combined["sequence_hash"].duplicated().any():
        raise ValueError("Exact duplicate sequences in development data.")

    if set(combined["label"].unique()) - {0, 1}:
        raise ValueError("Labels must be binary 0/1.")

    if not (combined["sequence_length"] == 2200).all():
        raise ValueError("All promoter sequences must have length 2200.")

    combined["sequence"] = combined["sequence"].astype(str).str.upper()

    invalid = combined["sequence"].str.contains(r"[^ACGTN]", regex=True)

    if invalid.any():
        raise ValueError("Invalid nucleotide symbols detected.")

    return combined


def find_cdhit() -> str:
    executable = shutil.which("cd-hit-est")

    if executable is None:
        raise FileNotFoundError(
            "cd-hit-est was not found in PATH."
        )

    return executable


def write_fasta(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in dataframe.itertuples(index=False):
            handle.write(f">{row.sample_id}\n")
            sequence = str(row.sequence)

            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def word_size_for_threshold(threshold: float) -> int:
    if threshold >= 0.90:
        return 8
    if threshold >= 0.88:
        return 7
    if threshold >= 0.85:
        return 6
    return 5


def run_cdhit(
    executable: str,
    input_fasta: Path,
    output_prefix: Path,
    threshold: float,
    threads: int,
    memory_mb: int,
) -> None:
    command = [
        executable,
        "-i",
        str(input_fasta),
        "-o",
        str(output_prefix),
        "-c",
        f"{threshold:.2f}",
        "-n",
        str(word_size_for_threshold(threshold)),
        "-G",
        "1",
        "-aS",
        "0.80",
        "-aL",
        "0.80",
        "-g",
        "1",
        "-d",
        "0",
        "-T",
        str(threads),
        "-M",
        str(memory_mb),
    ]

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path = output_prefix.with_suffix(".log")
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"CD-HIT-EST failed. See {log_path}"
        )


def parse_cluster_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster file not found: {path}")

    assignments: dict[str, str] = {}
    current_cluster: str | None = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">Cluster "):
                current_cluster = "cluster_" + line.split()[1]
                continue

            if current_cluster is None:
                raise ValueError("Cluster member appears before cluster header.")

            sample_id = line.split(">", maxsplit=1)[1].split("...", maxsplit=1)[0]

            if sample_id in assignments:
                raise ValueError(f"Duplicate cluster assignment: {sample_id}")

            assignments[sample_id] = current_cluster

    return assignments


def build_groups(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cdhit = find_cdhit()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = args.work_dir / "development_sequences.fasta"
    output_prefix = args.work_dir / "development_clusters"

    write_fasta(dataframe, fasta_path)
    run_cdhit(
        executable=cdhit,
        input_fasta=fasta_path,
        output_prefix=output_prefix,
        threshold=args.identity_threshold,
        threads=args.threads,
        memory_mb=args.memory_mb,
    )

    assignments = parse_cluster_assignments(
        Path(str(output_prefix) + ".clstr")
    )

    missing = set(dataframe["sample_id"]) - set(assignments)

    if missing:
        raise ValueError(
            f"{len(missing)} samples lack a CD-HIT group."
        )

    output = dataframe.copy()
    output["similarity_group"] = output["sample_id"].map(assignments)

    return output


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    analyzer="char",
                    lowercase=False,
                    vocabulary=None,
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
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])

    auroc = float("nan")

    if len(np.unique(targets)) == 2:
        auroc = float(roc_auc_score(targets, probabilities))

    return {
        "auprc": float(average_precision_score(targets, probabilities)),
        "auroc": auroc,
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(targets, predictions, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "accuracy": float(accuracy_score(targets, predictions)),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
    }


def score_configuration_inner_cv(
    pipeline: Pipeline,
    parameters: dict,
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splitter: StratifiedGroupKFold,
) -> tuple[float, float, list[float]]:
    fold_scores: list[float] = []

    for inner_train_index, inner_validation_index in splitter.split(
        sequences,
        labels,
        groups,
    ):
        model = clone(pipeline)
        model.set_params(**parameters)

        model.fit(
            sequences[inner_train_index],
            labels[inner_train_index],
        )

        probabilities = model.predict_proba(
            sequences[inner_validation_index]
        )[:, 1]

        score = average_precision_score(
            labels[inner_validation_index],
            probabilities,
        )
        fold_scores.append(float(score))

    return (
        float(np.mean(fold_scores)),
        float(np.std(fold_scores, ddof=0)),
        fold_scores,
    )


def choose_best_configuration(
    search_rows: list[dict],
) -> dict:
    ranked = sorted(
        search_rows,
        key=lambda row: (
            -row["mean_inner_auprc"],
            row["std_inner_auprc"],
            row["complexity_rank"],
        ),
    )
    return ranked[0]


def parameter_complexity_rank(parameters: dict) -> tuple:
    ngram_range = parameters["vectorizer__ngram_range"]
    class_weight = parameters["classifier__class_weight"]
    c_value = float(parameters["classifier__C"])

    span = ngram_range[1] - ngram_range[0]
    maximum_k = ngram_range[1]
    balanced_penalty = 1 if class_weight == "balanced" else 0

    return (
        span,
        maximum_k,
        balanced_penalty,
        abs(np.log10(c_value)),
    )


def json_safe_parameters(parameters: dict) -> dict:
    return {
        "ngram_range": list(parameters["vectorizer__ngram_range"]),
        "C": float(parameters["classifier__C"]),
        "class_weight": parameters["classifier__class_weight"],
    }


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        set_seed(args.seed)

        development = load_development_data(
            args.train,
            args.validation,
        )
        development = build_groups(development, args)

        sequences = development["sequence"].to_numpy(dtype=str)
        labels = development["label"].to_numpy(dtype=int)
        groups = development["similarity_group"].to_numpy(dtype=str)

        outer_cv = StratifiedGroupKFold(
            n_splits=args.outer_folds,
            shuffle=True,
            random_state=args.seed,
        )

        pipeline = make_pipeline(args.seed)
        parameter_grid = list(ParameterGrid(DEFAULT_PARAM_GRID))

        prediction_rows: list[pd.DataFrame] = []
        fold_rows: list[dict] = []
        all_search_rows: list[dict] = []

        start_time = time.time()

        print("=" * 96)
        print("K-MER LOGISTIC REGRESSION — NESTED GROUPED CROSS-VALIDATION")
        print("=" * 96)
        print(f"Development samples: {len(development)}")
        print(f"Positive:            {(labels == 1).sum()}")
        print(f"Negative:            {(labels == 0).sum()}")
        print(f"Similarity groups:   {len(np.unique(groups))}")
        print(f"Outer folds:         {args.outer_folds}")
        print(f"Inner folds:         {args.inner_folds}")
        print(f"Configurations:      {len(parameter_grid)}")
        print("Frozen test used:    False")
        print("=" * 96)

        for outer_fold, (train_index, holdout_index) in enumerate(
            outer_cv.split(sequences, labels, groups),
            start=1,
        ):
            outer_train_sequences = sequences[train_index]
            outer_train_labels = labels[train_index]
            outer_train_groups = groups[train_index]

            holdout_sequences = sequences[holdout_index]
            holdout_labels = labels[holdout_index]

            inner_cv = StratifiedGroupKFold(
                n_splits=args.inner_folds,
                shuffle=True,
                random_state=args.seed + outer_fold,
            )

            fold_search_rows = []

            for configuration_index, parameters in enumerate(
                parameter_grid,
                start=1,
            ):
                mean_score, std_score, inner_scores = (
                    score_configuration_inner_cv(
                        pipeline=pipeline,
                        parameters=parameters,
                        sequences=outer_train_sequences,
                        labels=outer_train_labels,
                        groups=outer_train_groups,
                        splitter=inner_cv,
                    )
                )

                row = {
                    "outer_fold": outer_fold,
                    "configuration_index": configuration_index,
                    "mean_inner_auprc": mean_score,
                    "std_inner_auprc": std_score,
                    "inner_fold_auprc": json.dumps(inner_scores),
                    "complexity_rank": parameter_complexity_rank(parameters),
                    **json_safe_parameters(parameters),
                    "_parameters": parameters,
                }

                fold_search_rows.append(row)

            best = choose_best_configuration(fold_search_rows)
            best_parameters = best["_parameters"]

            final_model = clone(pipeline)
            final_model.set_params(**best_parameters)
            final_model.fit(
                outer_train_sequences,
                outer_train_labels,
            )

            holdout_probabilities = final_model.predict_proba(
                holdout_sequences
            )[:, 1]

            metrics = evaluate_probabilities(
                targets=holdout_labels,
                probabilities=holdout_probabilities,
                threshold=0.5,
            )

            fold_row = {
                "outer_fold": outer_fold,
                "train_samples": int(len(train_index)),
                "holdout_samples": int(len(holdout_index)),
                "holdout_positive": int((holdout_labels == 1).sum()),
                "holdout_negative": int((holdout_labels == 0).sum()),
                "inner_best_mean_auprc": float(
                    best["mean_inner_auprc"]
                ),
                "inner_best_std_auprc": float(
                    best["std_inner_auprc"]
                ),
                **json_safe_parameters(best_parameters),
                **metrics,
            }
            fold_rows.append(fold_row)

            fold_predictions = development.iloc[
                holdout_index
            ][
                [
                    "sample_id",
                    "sequence_id",
                    "label",
                    "response_class",
                    "development_source",
                    "similarity_group",
                ]
            ].copy()
            fold_predictions["outer_fold"] = outer_fold
            fold_predictions["probability"] = holdout_probabilities
            fold_predictions["predicted_label_0_5"] = (
                holdout_probabilities >= 0.5
            ).astype(int)
            prediction_rows.append(fold_predictions)

            for row in fold_search_rows:
                clean_row = {
                    key: value
                    for key, value in row.items()
                    if key not in {"_parameters", "complexity_rank"}
                }
                clean_row["selected_for_outer_fold"] = (
                    row is best
                )
                all_search_rows.append(clean_row)

            print(
                f"Outer fold {outer_fold}/{args.outer_folds} | "
                f"AUPRC={metrics['auprc']:.4f} "
                f"AUROC={metrics['auroc']:.4f} "
                f"MCC={metrics['mcc']:.4f} | "
                f"k={best_parameters['vectorizer__ngram_range']} "
                f"C={best_parameters['classifier__C']} "
                f"class_weight={best_parameters['classifier__class_weight']}"
            )

        predictions = pd.concat(
            prediction_rows,
            ignore_index=True,
        )
        folds = pd.DataFrame(fold_rows)
        searches = pd.DataFrame(all_search_rows)

        if predictions["sample_id"].duplicated().any():
            raise ValueError(
                "A development sample received more than one outer-fold prediction."
            )

        if len(predictions) != len(development):
            raise ValueError(
                "Outer-fold predictions do not cover every development sample."
            )

        pooled_metrics = evaluate_probabilities(
            targets=predictions["label"].to_numpy(dtype=int),
            probabilities=predictions["probability"].to_numpy(dtype=float),
            threshold=0.5,
        )

        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        args.fold_results.parent.mkdir(parents=True, exist_ok=True)
        args.search_results.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        predictions.to_csv(args.predictions, index=False)
        folds.to_csv(args.fold_results, index=False)
        searches.to_csv(args.search_results, index=False)

        metric_names = [
            "auprc",
            "auroc",
            "mcc",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "accuracy",
        ]

        fold_distribution = {
            metric: {
                "mean": float(folds[metric].mean()),
                "std": float(folds[metric].std(ddof=0)),
                "min": float(folds[metric].min()),
                "max": float(folds[metric].max()),
                "values": [
                    float(value) for value in folds[metric].tolist()
                ],
            }
            for metric in metric_names
        }

        selected_configurations = []
        for row in fold_rows:
            selected_configurations.append(
                {
                    "outer_fold": int(row["outer_fold"]),
                    "ngram_range": row["ngram_range"],
                    "C": float(row["C"]),
                    "class_weight": row["class_weight"],
                    "inner_best_mean_auprc": float(
                        row["inner_best_mean_auprc"]
                    ),
                }
            )

        summary = {
            "experiment_name": (
                "kmer_logistic_regression_nested_grouped_cross_validation"
            ),
            "test_set_used": False,
            "development_data": {
                "sources": [
                    str(args.train),
                    str(args.validation),
                ],
                "samples": int(len(development)),
                "positive": int((labels == 1).sum()),
                "negative": int((labels == 0).sum()),
                "positive_prevalence": float(labels.mean()),
                "similarity_groups": int(len(np.unique(groups))),
            },
            "grouping": {
                "method": "CD-HIT-EST",
                "identity_threshold": args.identity_threshold,
                "minimum_sequence_coverage": 0.80,
                "global_identity": True,
            },
            "cross_validation": {
                "outer_folds": args.outer_folds,
                "inner_folds": args.inner_folds,
                "splitter": "StratifiedGroupKFold",
                "seed": args.seed,
                "selection_metric": "inner_mean_auprc",
            },
            "features": {
                "type": "normalized_kmer_frequencies",
                "normalization": "L2",
                "sequence_only": True,
                "excluded_features": [
                    "motifs",
                    "PWM_scores",
                    "transcription_factor_annotations",
                    "expression_values",
                    "response_classes",
                    "cis_regulatory_architecture",
                ],
            },
            "parameter_grid": {
                "ngram_ranges": [
                    list(value)
                    for value in DEFAULT_PARAM_GRID[
                        "vectorizer__ngram_range"
                    ]
                ],
                "C_values": DEFAULT_PARAM_GRID["classifier__C"],
                "class_weight_values": [
                    "none" if value is None else value
                    for value in DEFAULT_PARAM_GRID[
                        "classifier__class_weight"
                    ]
                ],
            },
            "outer_fold_metric_distribution": fold_distribution,
            "pooled_out_of_fold_metrics_at_0_5": pooled_metrics,
            "selected_configurations": selected_configurations,
            "training_seconds": float(time.time() - start_time),
            "outputs": {
                "predictions": str(args.predictions),
                "fold_results": str(args.fold_results),
                "inner_search_results": str(args.search_results),
            },
        }

        args.summary.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("=" * 96)
        print("NESTED GROUPED CROSS-VALIDATION COMPLETE")
        print("=" * 96)
        print(
            f"Mean outer AUPRC: "
            f"{fold_distribution['auprc']['mean']:.6f} "
            f"± {fold_distribution['auprc']['std']:.6f}"
        )
        print(
            f"Pooled OOF AUPRC: "
            f"{pooled_metrics['auprc']:.6f}"
        )
        print(
            f"Pooled OOF AUROC: "
            f"{pooled_metrics['auroc']:.6f}"
        )
        print(
            f"Pooled OOF MCC:   "
            f"{pooled_metrics['mcc']:.6f}"
        )
        print(f"Frozen test used: False")
        print()
        print(f"Summary:          {args.summary}")
        print(f"Fold results:     {args.fold_results}")
        print(f"Predictions:      {args.predictions}")
        print(f"Search results:   {args.search_results}")
        print("=" * 96)

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        pd.errors.ParserError,
        subprocess.SubprocessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
