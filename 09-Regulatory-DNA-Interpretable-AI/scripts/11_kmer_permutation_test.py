#!/usr/bin/env python3
"""Permutation test for the fixed k-mer logistic-regression baseline.

This experiment tests whether promoter sequences contain predictive signal beyond
chance while preserving the complete development dataset and grouped evaluation
design.

Important design choices
------------------------
- Only train + validation are used as development data.
- The frozen test set is never loaded.
- CD-HIT similarity groups produced by Script 10 are reused.
- Outer fold assignments are generated once from the true labels and then frozen.
- The model configuration is fixed before permutation testing:
    * character 5-mer frequencies;
    * L2 normalization;
    * logistic regression with C=10;
    * no class weighting by default.
- For every permutation, only the labels are shuffled.
- Each permuted label vector is evaluated with the same frozen grouped folds.
- The statistic is pooled out-of-fold AUPRC.

This is a conditional permutation test of the fixed sequence-only baseline. It does
not repeat hyperparameter selection inside every permutation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a grouped out-of-fold permutation test for the fixed "
            "k-mer logistic-regression baseline."
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
        "--cluster-file",
        type=Path,
        default=Path(
            "experiments/kmer_logistic_nested_cv/"
            "development_clusters.clstr"
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--C",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=Path(
            "results/tables/"
            "kmer_logistic_permutation_distribution.csv"
        ),
    )
    parser.add_argument(
        "--observed-predictions",
        type=Path,
        default=Path(
            "results/tables/"
            "kmer_logistic_fixed_model_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "results/tables/"
            "kmer_logistic_permutation_fold_assignments.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/metrics/"
            "kmer_logistic_permutation_test_summary.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "results/figures/"
            "kmer_logistic_permutation_test.png"
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if args.permutations < 19:
        raise ValueError(
            "At least 19 permutations are required. "
            "Use 200 or more for the final analysis."
        )

    if args.outer_folds < 3:
        raise ValueError("outer-folds must be at least 3.")

    if args.kmer_size < 1:
        raise ValueError("kmer-size must be positive.")

    if args.C <= 0:
        raise ValueError("C must be positive.")


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def load_development_data(
    train_path: Path,
    validation_path: Path,
) -> pd.DataFrame:
    """Load and combine train and validation without opening test."""

    required_columns = {
        "sample_id",
        "sequence_id",
        "sequence",
        "label",
        "response_class",
        "sequence_length",
        "sequence_hash",
    }

    frames: list[pd.DataFrame] = []

    for source_name, path in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Development split not found: {path}"
            )

        dataframe = pd.read_csv(path)
        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing)}"
            )

        dataframe = dataframe.copy()
        dataframe["development_source"] = source_name
        frames.append(dataframe)

    development = pd.concat(frames, ignore_index=True)

    if development["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicated sample IDs detected in development data."
        )

    if development["sequence_hash"].duplicated().any():
        raise ValueError(
            "Exact duplicate sequences detected in development data."
        )

    if set(development["label"].unique()) - {0, 1}:
        raise ValueError("Labels must contain only 0 and 1.")

    if not (development["sequence_length"] == 2200).all():
        raise ValueError(
            "All promoter sequences must have length 2200."
        )

    development["sequence"] = (
        development["sequence"].astype(str).str.upper()
    )

    invalid_sequences = development["sequence"].str.contains(
        r"[^ACGTN]",
        regex=True,
    )

    if invalid_sequences.any():
        raise ValueError(
            "Invalid nucleotide symbols detected in development data."
        )

    return development


def parse_cluster_assignments(path: Path) -> dict[str, str]:
    """Parse a CD-HIT .clstr file into sample-to-group assignments."""

    if not path.exists():
        raise FileNotFoundError(
            f"CD-HIT cluster file not found: {path}"
        )

    assignments: dict[str, str] = {}
    current_cluster: str | None = None

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">Cluster "):
                current_cluster = (
                    "cluster_" + line.split()[1]
                )
                continue

            if current_cluster is None:
                raise ValueError(
                    "Cluster member encountered before cluster header."
                )

            if ">" not in line or "..." not in line:
                raise ValueError(
                    f"Unexpected CD-HIT cluster line: {line}"
                )

            sample_id = (
                line.split(">", maxsplit=1)[1]
                .split("...", maxsplit=1)[0]
            )

            if sample_id in assignments:
                raise ValueError(
                    f"Duplicate cluster assignment for {sample_id}."
                )

            assignments[sample_id] = current_cluster

    if not assignments:
        raise ValueError(
            f"No assignments were parsed from {path}."
        )

    return assignments


def attach_similarity_groups(
    development: pd.DataFrame,
    cluster_file: Path,
) -> pd.DataFrame:
    """Attach CD-HIT group IDs to development samples."""

    assignments = parse_cluster_assignments(cluster_file)

    missing = (
        set(development["sample_id"])
        - set(assignments)
    )

    if missing:
        raise ValueError(
            f"{len(missing)} development samples lack "
            "a CD-HIT similarity group."
        )

    output = development.copy()
    output["similarity_group"] = (
        output["sample_id"].map(assignments)
    )

    return output


def make_pipeline(
    kmer_size: int,
    c_value: float,
    class_weight: str,
    seed: int,
) -> Pipeline:
    """Create the fixed sequence-only baseline."""

    effective_class_weight = (
        None if class_weight == "none" else "balanced"
    )

    return Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    analyzer="char",
                    lowercase=False,
                    ngram_range=(kmer_size, kmer_size),
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
                    C=c_value,
                    class_weight=effective_class_weight,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_frozen_folds(
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    number_of_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Build grouped outer folds once using the true labels."""

    splitter = StratifiedGroupKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_assignments = np.full(
        shape=len(labels),
        fill_value=-1,
        dtype=int,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_number, (train_index, holdout_index) in enumerate(
        splitter.split(sequences, labels, groups),
        start=1,
    ):
        fold_assignments[holdout_index] = fold_number
        folds.append((train_index, holdout_index))

    if np.any(fold_assignments < 1):
        raise ValueError(
            "At least one development sample lacks a fold assignment."
        )

    for group in np.unique(groups):
        group_folds = np.unique(
            fold_assignments[groups == group]
        )

        if len(group_folds) != 1:
            raise ValueError(
                f"Similarity group {group} crosses frozen folds."
            )

    return fold_assignments, folds


def out_of_fold_probabilities(
    pipeline: Pipeline,
    sequences: np.ndarray,
    labels_for_training: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Generate one held-out probability for every development sample."""

    probabilities = np.full(
        shape=len(labels_for_training),
        fill_value=np.nan,
        dtype=float,
    )

    for train_index, holdout_index in folds:
        train_labels = labels_for_training[train_index]

        if len(np.unique(train_labels)) != 2:
            raise ValueError(
                "A permuted training fold contains only one class."
            )

        model = clone(pipeline)
        model.fit(
            sequences[train_index],
            train_labels,
        )

        probabilities[holdout_index] = model.predict_proba(
            sequences[holdout_index]
        )[:, 1]

    if np.any(~np.isfinite(probabilities)):
        raise ValueError(
            "Non-finite or missing out-of-fold probabilities."
        )

    return probabilities


def save_permutation_figure(
    permutation_scores: np.ndarray,
    observed_score: float,
    path: Path,
) -> None:
    """Save the null AUPRC distribution and observed statistic."""

    figure, axis = plt.subplots(figsize=(8, 6))

    axis.hist(
        permutation_scores,
        bins=25,
        edgecolor="black",
        alpha=0.75,
    )
    axis.axvline(
        observed_score,
        linestyle="--",
        linewidth=2,
        label=f"Observed AUPRC = {observed_score:.3f}",
    )
    axis.axvline(
        float(np.mean(permutation_scores)),
        linestyle=":",
        linewidth=2,
        label=(
            "Permutation mean = "
            f"{np.mean(permutation_scores):.3f}"
        ),
    )

    axis.set_xlabel("Pooled out-of-fold AUPRC")
    axis.set_ylabel("Number of permutations")
    axis.set_title(
        "Permutation Test — Fixed 5-mer Logistic Baseline"
    )
    axis.legend()
    axis.grid(alpha=0.20)

    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def percentile_rank(
    values: np.ndarray,
    observed: float,
) -> float:
    """Return the percentage of null values below the observed score."""

    return float(100.0 * np.mean(values < observed))


def main() -> int:
    """Run the complete permutation test."""

    args = parse_args()

    try:
        validate_args(args)
        set_seed(args.seed)

        development = load_development_data(
            train_path=args.train,
            validation_path=args.validation,
        )
        development = attach_similarity_groups(
            development=development,
            cluster_file=args.cluster_file,
        )

        sequences = development["sequence"].to_numpy(dtype=str)
        true_labels = development["label"].to_numpy(dtype=int)
        groups = development[
            "similarity_group"
        ].to_numpy(dtype=str)

        fold_assignments, frozen_folds = build_frozen_folds(
            sequences=sequences,
            labels=true_labels,
            groups=groups,
            number_of_folds=args.outer_folds,
            seed=args.seed,
        )

        pipeline = make_pipeline(
            kmer_size=args.kmer_size,
            c_value=args.C,
            class_weight=args.class_weight,
            seed=args.seed,
        )

        print("=" * 96)
        print("K-MER LOGISTIC REGRESSION — PERMUTATION TEST")
        print("=" * 96)
        print(f"Development samples: {len(development)}")
        print(f"Positive:            {(true_labels == 1).sum()}")
        print(f"Negative:            {(true_labels == 0).sum()}")
        print(f"Similarity groups:   {len(np.unique(groups))}")
        print(f"Frozen outer folds:  {args.outer_folds}")
        print(f"Permutations:        {args.permutations}")
        print(
            f"Fixed model:         {args.kmer_size}-mers, "
            f"C={args.C}, class_weight={args.class_weight}"
        )
        print("Frozen test used:    False")
        print("=" * 96)

        start_time = time.time()

        observed_probabilities = out_of_fold_probabilities(
            pipeline=pipeline,
            sequences=sequences,
            labels_for_training=true_labels,
            folds=frozen_folds,
        )

        observed_auprc = float(
            average_precision_score(
                true_labels,
                observed_probabilities,
            )
        )
        observed_auroc = float(
            roc_auc_score(
                true_labels,
                observed_probabilities,
            )
        )

        rng = np.random.default_rng(args.seed + 1000)
        distribution_rows: list[dict[str, float | int]] = []

        for permutation_index in range(
            1,
            args.permutations + 1,
        ):
            permuted_labels = rng.permutation(true_labels)

            permuted_probabilities = out_of_fold_probabilities(
                pipeline=pipeline,
                sequences=sequences,
                labels_for_training=permuted_labels,
                folds=frozen_folds,
            )

            permutation_auprc = float(
                average_precision_score(
                    permuted_labels,
                    permuted_probabilities,
                )
            )
            permutation_auroc = float(
                roc_auc_score(
                    permuted_labels,
                    permuted_probabilities,
                )
            )

            distribution_rows.append(
                {
                    "permutation": permutation_index,
                    "auprc": permutation_auprc,
                    "auroc": permutation_auroc,
                    "seed": args.seed + 1000,
                }
            )

            if (
                permutation_index == 1
                or permutation_index % 10 == 0
                or permutation_index == args.permutations
            ):
                print(
                    f"Permutation "
                    f"{permutation_index:4d}/{args.permutations} | "
                    f"AUPRC={permutation_auprc:.4f}"
                )

        distribution = pd.DataFrame(distribution_rows)
        null_auprc = distribution["auprc"].to_numpy(dtype=float)
        null_auroc = distribution["auroc"].to_numpy(dtype=float)

        auprc_exceedances = int(
            np.sum(null_auprc >= observed_auprc)
        )
        auroc_exceedances = int(
            np.sum(null_auroc >= observed_auroc)
        )

        empirical_p_auprc = float(
            (auprc_exceedances + 1)
            / (args.permutations + 1)
        )
        empirical_p_auroc = float(
            (auroc_exceedances + 1)
            / (args.permutations + 1)
        )

        elapsed_seconds = time.time() - start_time

        args.distribution.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.observed_predictions.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.fold_assignments.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.summary.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.figure.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        distribution.to_csv(
            args.distribution,
            index=False,
        )

        observed_predictions = development[
            [
                "sample_id",
                "sequence_id",
                "label",
                "response_class",
                "development_source",
                "similarity_group",
            ]
        ].copy()
        observed_predictions["outer_fold"] = fold_assignments
        observed_predictions["probability"] = (
            observed_probabilities
        )
        observed_predictions.to_csv(
            args.observed_predictions,
            index=False,
        )

        fold_table = development[
            [
                "sample_id",
                "sequence_id",
                "label",
                "development_source",
                "similarity_group",
            ]
        ].copy()
        fold_table["outer_fold"] = fold_assignments
        fold_table.to_csv(
            args.fold_assignments,
            index=False,
        )

        save_permutation_figure(
            permutation_scores=null_auprc,
            observed_score=observed_auprc,
            path=args.figure,
        )

        summary = {
            "experiment_name": (
                "fixed_kmer_logistic_regression_permutation_test"
            ),
            "test_set_used": False,
            "development_data": {
                "sources": [
                    str(args.train),
                    str(args.validation),
                ],
                "samples": int(len(development)),
                "positive": int((true_labels == 1).sum()),
                "negative": int((true_labels == 0).sum()),
                "positive_prevalence": float(
                    np.mean(true_labels)
                ),
                "similarity_groups": int(
                    len(np.unique(groups))
                ),
            },
            "evaluation_design": {
                "outer_folds": args.outer_folds,
                "splitter": "StratifiedGroupKFold",
                "fold_seed": args.seed,
                "folds_frozen_before_permutations": True,
                "statistic": "pooled_out_of_fold_auprc",
            },
            "fixed_model": {
                "feature_type": (
                    "L2_normalized_character_kmer_frequencies"
                ),
                "kmer_size": args.kmer_size,
                "classifier": "LogisticRegression",
                "C": args.C,
                "class_weight": args.class_weight,
                "solver": "liblinear",
                "selection_status": (
                    "Fixed before permutation testing; "
                    "no tuning performed in this script."
                ),
            },
            "permutation_design": {
                "permutations": args.permutations,
                "permutation_seed": args.seed + 1000,
                "labels_shuffled": True,
                "sequences_unchanged": True,
                "groups_unchanged": True,
                "folds_unchanged": True,
                "empirical_p_value_formula": (
                    "(null_scores_greater_or_equal_observed + 1) "
                    "/ (n_permutations + 1)"
                ),
            },
            "observed": {
                "auprc": observed_auprc,
                "auroc": observed_auroc,
            },
            "null_distribution": {
                "auprc_mean": float(np.mean(null_auprc)),
                "auprc_std": float(np.std(null_auprc, ddof=0)),
                "auprc_min": float(np.min(null_auprc)),
                "auprc_q025": float(
                    np.quantile(null_auprc, 0.025)
                ),
                "auprc_median": float(np.median(null_auprc)),
                "auprc_q975": float(
                    np.quantile(null_auprc, 0.975)
                ),
                "auprc_max": float(np.max(null_auprc)),
                "auroc_mean": float(np.mean(null_auroc)),
                "auroc_std": float(np.std(null_auroc, ddof=0)),
            },
            "significance": {
                "auprc_null_exceedances": auprc_exceedances,
                "auprc_empirical_p_value": empirical_p_auprc,
                "auprc_observed_percentile": percentile_rank(
                    null_auprc,
                    observed_auprc,
                ),
                "auroc_null_exceedances": auroc_exceedances,
                "auroc_empirical_p_value": empirical_p_auroc,
                "auroc_observed_percentile": percentile_rank(
                    null_auroc,
                    observed_auroc,
                ),
            },
            "interpretation_policy": {
                "primary_metric": "AUPRC",
                "significance_threshold": 0.05,
                "important_limitation": (
                    "This conditional test evaluates a fixed model "
                    "configuration. It does not repeat nested "
                    "hyperparameter selection for each permutation."
                ),
            },
            "runtime_seconds": float(elapsed_seconds),
            "outputs": {
                "distribution": str(args.distribution),
                "observed_predictions": str(
                    args.observed_predictions
                ),
                "fold_assignments": str(
                    args.fold_assignments
                ),
                "figure": str(args.figure),
                "summary": str(args.summary),
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

        print("=" * 96)
        print("PERMUTATION TEST COMPLETE")
        print("=" * 96)
        print(f"Observed OOF AUPRC:    {observed_auprc:.6f}")
        print(
            f"Null AUPRC mean:       "
            f"{np.mean(null_auprc):.6f} "
            f"± {np.std(null_auprc, ddof=0):.6f}"
        )
        print(
            f"Null 95% interval:     "
            f"[{np.quantile(null_auprc, 0.025):.6f}, "
            f"{np.quantile(null_auprc, 0.975):.6f}]"
        )
        print(
            f"Empirical p-value:     "
            f"{empirical_p_auprc:.6f}"
        )
        print(
            f"Observed percentile:   "
            f"{percentile_rank(null_auprc, observed_auprc):.2f}%"
        )
        print(f"Observed OOF AUROC:    {observed_auroc:.6f}")
        print(f"Frozen test used:      False")
        print()
        print(f"Summary:               {args.summary}")
        print(f"Distribution:          {args.distribution}")
        print(
            f"Observed predictions:  "
            f"{args.observed_predictions}"
        )
        print(f"Figure:                {args.figure}")
        print(
            f"Runtime seconds:       "
            f"{elapsed_seconds:.2f}"
        )
        print("=" * 96)

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
