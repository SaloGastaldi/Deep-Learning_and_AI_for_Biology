#!/usr/bin/env python3
"""Interpret fixed 5-mer and 6-mer logistic-regression promoter models.

Only train + validation are used. The frozen test set and prior motif/PWM
information are never loaded. The 5-mer analysis is primary; the 6-mer
analysis is complementary for later comparison with six-nucleotide cores.
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
from scipy.stats import fisher_exact
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/splits/validation.csv"))
    parser.add_argument(
        "--cluster-file",
        type=Path,
        default=Path("experiments/kmer_logistic_nested_cv/development_clusters.clstr"),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--figure-top-n", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables/kmer_interpretation"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/figures/kmer_interpretation"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/metrics/kmer_interpretation_summary.json"),
    )
    return parser.parse_args()


def load_development_data(train_path: Path, validation_path: Path) -> pd.DataFrame:
    required = {
        "sample_id", "sequence_id", "sequence", "label", "response_class",
        "sequence_length", "sequence_hash",
    }
    frames = []
    for source_name, path in (("train", train_path), ("validation", validation_path)):
        if not path.exists():
            raise FileNotFoundError(f"Development split not found: {path}")
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["development_source"] = source_name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if data["sample_id"].duplicated().any():
        raise ValueError("Duplicated sample IDs in development data.")
    if data["sequence_hash"].duplicated().any():
        raise ValueError("Exact duplicate sequences in development data.")
    if set(data["label"].unique()) - {0, 1}:
        raise ValueError("Labels must be binary 0/1.")
    if not (data["sequence_length"] == 2200).all():
        raise ValueError("All sequences must be 2200 bp.")
    data["sequence"] = data["sequence"].astype(str).str.upper()
    if data["sequence"].str.contains(r"[^ACGTN]", regex=True).any():
        raise ValueError("Invalid nucleotide symbols detected.")
    return data


def parse_cluster_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"CD-HIT cluster file not found: {path}")
    assignments: dict[str, str] = {}
    current: str | None = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster "):
                current = "cluster_" + line.split()[1]
                continue
            if current is None or ">" not in line or "..." not in line:
                raise ValueError(f"Unexpected CD-HIT line: {line}")
            sample_id = line.split(">", 1)[1].split("...", 1)[0]
            assignments[sample_id] = current
    return assignments


def attach_groups(data: pd.DataFrame, cluster_file: Path) -> pd.DataFrame:
    assignments = parse_cluster_assignments(cluster_file)
    missing = set(data["sample_id"]) - set(assignments)
    if missing:
        raise ValueError(f"{len(missing)} samples lack a similarity group.")
    output = data.copy()
    output["similarity_group"] = output["sample_id"].map(assignments)
    return output


def build_vocabulary(k: int) -> dict[str, int]:
    return {
        "".join(chars): index
        for index, chars in enumerate(itertools.product("ACGT", repeat=k))
    }


def make_pipeline(k: int, c_value: float, class_weight: str, seed: int) -> Pipeline:
    effective_weight = None if class_weight == "none" else "balanced"
    return Pipeline([
        (
            "vectorizer",
            CountVectorizer(
                analyzer="char",
                lowercase=False,
                ngram_range=(k, k),
                vocabulary=build_vocabulary(k),
                dtype=np.float64,
            ),
        ),
        (
            "normalizer",
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
                class_weight=effective_weight,
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            ),
        ),
    ])


def build_folds(
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.full(len(labels), -1, dtype=int)
    folds = []
    for fold, (train_index, holdout_index) in enumerate(
        splitter.split(sequences, labels, groups), start=1
    ):
        assignments[holdout_index] = fold
        folds.append((train_index, holdout_index))
    if np.any(assignments < 1):
        raise ValueError("Incomplete fold assignments.")
    for group in np.unique(groups):
        if len(np.unique(assignments[groups == group])) != 1:
            raise ValueError(f"Similarity group {group} crosses folds.")
    return assignments, folds


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def enrichment_table(sequences: np.ndarray, labels: np.ndarray, kmers: list[str]) -> pd.DataFrame:
    positives = sequences[labels == 1]
    negatives = sequences[labels == 0]
    rows = []
    for kmer in kmers:
        positive_present = int(sum(kmer in sequence for sequence in positives))
        negative_present = int(sum(kmer in sequence for sequence in negatives))
        positive_absent = len(positives) - positive_present
        negative_absent = len(negatives) - negative_present
        odds_ratio, p_value = fisher_exact(
            [[positive_present, positive_absent], [negative_present, negative_absent]],
            alternative="two-sided",
        )
        rows.append({
            "kmer": kmer,
            "positive_present": positive_present,
            "positive_total": int(len(positives)),
            "positive_fraction": positive_present / len(positives),
            "negative_present": negative_present,
            "negative_total": int(len(negatives)),
            "negative_fraction": negative_present / len(negatives),
            "odds_ratio": float(odds_ratio),
            "fisher_p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    table["fisher_fdr"] = bh_fdr(table["fisher_p_value"].to_numpy())
    return table


def rank_frequency(coefficient_matrix: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros(coefficient_matrix.shape[1], dtype=int)
    negative = np.zeros(coefficient_matrix.shape[1], dtype=int)
    for coefficients in coefficient_matrix:
        positive[np.argsort(coefficients)[-top_n:]] += 1
        negative[np.argsort(coefficients)[:top_n]] += 1
    return positive, negative


def plot_coefficients(table: pd.DataFrame, k: int, direction: str, top_n: int, path: Path) -> None:
    """Plot top coefficients using anonymous public labels.

    Real k-mer identities are used internally for ranking and interpretation,
    but are not exposed in distributed figures.
    """
    if direction == "positive":
        selected = table.nlargest(top_n, "mean_outer_coefficient").sort_values(
            "mean_outer_coefficient"
        )
        title = f"Top positive {k}-mer coefficients"
        anonymous_labels = [
            f"Pattern P{i}" for i in range(len(selected), 0, -1)
        ]
    else:
        selected = table.nsmallest(top_n, "mean_outer_coefficient").sort_values(
            "mean_outer_coefficient", ascending=False
        )
        title = f"Top negative {k}-mer coefficients"
        anonymous_labels = [
            f"Pattern N{i}" for i in range(len(selected), 0, -1)
        ]

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        anonymous_labels,
        selected["mean_outer_coefficient"],
        xerr=selected["std_outer_coefficient"],
        alpha=0.8,
    )
    axis.axvline(0.0, linewidth=1)
    axis.set_xlabel("Mean coefficient across outer-fold models")
    axis.set_ylabel(f"Anonymous {k}-mer pattern")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)

def analyze_k(
    data: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_assignments: np.ndarray,
    k: int,
    args: argparse.Namespace,
) -> dict:
    sequences = data["sequence"].to_numpy(dtype=str)
    labels = data["label"].to_numpy(dtype=int)
    vocabulary = build_vocabulary(k)
    kmers = [item[0] for item in sorted(vocabulary.items(), key=lambda item: item[1])]
    pipeline = make_pipeline(k, args.C, args.class_weight, args.seed)

    oof_probabilities = np.full(len(data), np.nan)
    coefficient_rows = []
    fold_metrics = []

    for fold_number, (train_index, holdout_index) in enumerate(folds, start=1):
        model = clone(pipeline)
        model.fit(sequences[train_index], labels[train_index])
        probabilities = model.predict_proba(sequences[holdout_index])[:, 1]
        oof_probabilities[holdout_index] = probabilities
        coefficient_rows.append(model.named_steps["classifier"].coef_.ravel().copy())
        fold_metrics.append({
            "k": k,
            "outer_fold": fold_number,
            "train_samples": int(len(train_index)),
            "holdout_samples": int(len(holdout_index)),
            "holdout_positive": int((labels[holdout_index] == 1).sum()),
            "holdout_negative": int((labels[holdout_index] == 0).sum()),
            **metrics(labels[holdout_index], probabilities),
        })

    if np.any(~np.isfinite(oof_probabilities)):
        raise ValueError(f"Missing OOF probabilities for k={k}.")

    coefficient_matrix = np.vstack(coefficient_rows)
    full_model = clone(pipeline)
    full_model.fit(sequences, labels)
    full_coefficients = full_model.named_steps["classifier"].coef_.ravel().copy()
    positive_frequency, negative_frequency = rank_frequency(
        coefficient_matrix, min(args.top_n, coefficient_matrix.shape[1])
    )

    table = pd.DataFrame({
        "kmer": kmers,
        "mean_outer_coefficient": coefficient_matrix.mean(axis=0),
        "std_outer_coefficient": coefficient_matrix.std(axis=0, ddof=0),
        "median_outer_coefficient": np.median(coefficient_matrix, axis=0),
        "min_outer_coefficient": coefficient_matrix.min(axis=0),
        "max_outer_coefficient": coefficient_matrix.max(axis=0),
        "positive_sign_fraction": (coefficient_matrix > 0).mean(axis=0),
        "negative_sign_fraction": (coefficient_matrix < 0).mean(axis=0),
        "top_positive_fold_count": positive_frequency,
        "top_negative_fold_count": negative_frequency,
        "full_development_coefficient": full_coefficients,
    })
    table["absolute_mean_outer_coefficient"] = table["mean_outer_coefficient"].abs()
    table["coefficient_signal_to_noise"] = (
        table["absolute_mean_outer_coefficient"] / (table["std_outer_coefficient"] + 1e-12)
    )
    table = table.merge(enrichment_table(sequences, labels, kmers), on="kmer", how="left")
    table["direction"] = np.where(
        table["mean_outer_coefficient"] > 0,
        "positive",
        np.where(table["mean_outer_coefficient"] < 0, "negative", "neutral"),
    )
    table["stable_positive"] = (
        (table["positive_sign_fraction"] == 1.0)
        & (table["mean_outer_coefficient"] > 0)
    )
    table["stable_negative"] = (
        (table["negative_sign_fraction"] == 1.0)
        & (table["mean_outer_coefficient"] < 0)
    )

    # Rankings are computed internally because they are required for interpretation,
    # but public/distributed outputs intentionally omit real k-mer identities.
    positive_ranking = table.sort_values(
        ["mean_outer_coefficient", "positive_sign_fraction", "top_positive_fold_count"],
        ascending=[False, False, False],
    )
    negative_ranking = table.sort_values(
        ["mean_outer_coefficient", "negative_sign_fraction", "top_negative_fold_count"],
        ascending=[True, False, False],
    )

    # Public-safe tabular output: fold-level predictive metrics only.
    fold_metrics_path = args.output_dir / f"k{k}_outer_fold_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(fold_metrics_path, index=False)

    # Public-safe figures: real k-mer identities are replaced by anonymous labels.
    positive_figure = args.figure_dir / f"k{k}_top_positive_coefficients.png"
    negative_figure = args.figure_dir / f"k{k}_top_negative_coefficients.png"
    plot_coefficients(table, k, "positive", args.figure_top_n, positive_figure)
    plot_coefficients(table, k, "negative", args.figure_top_n, negative_figure)

    pooled_metrics = metrics(labels, oof_probabilities)
    return {
        "k": k,
        "possible_kmers": len(kmers),
        "pooled_oof_metrics_at_0_5": pooled_metrics,
        "stable_positive_kmers": int(table["stable_positive"].sum()),
        "stable_negative_kmers": int(table["stable_negative"].sum()),
        "fdr_significant_presence_enrichment_kmers": int((table["fisher_fdr"] < 0.05).sum()),
        "public_output_policy": {
            "real_kmer_identities_distributed": False,
            "rankings_with_real_kmers_distributed": False,
            "oof_predictions_with_identifiers_distributed": False,
            "figures_use_anonymous_pattern_labels": True,
        },
        "outputs": {
            "fold_metrics": str(fold_metrics_path),
            "positive_figure": str(positive_figure),
            "negative_figure": str(negative_figure),
        },
    }



def main() -> int:
    args = parse_args()
    try:
        if args.outer_folds < 3 or args.C <= 0 or args.top_n < 1:
            raise ValueError("Invalid command-line parameters.")
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        data = attach_groups(
            load_development_data(args.train, args.validation),
            args.cluster_file,
        )
        sequences = data["sequence"].to_numpy(dtype=str)
        labels = data["label"].to_numpy(dtype=int)
        groups = data["similarity_group"].to_numpy(dtype=str)
        fold_assignments, folds = build_folds(
            sequences, labels, groups, args.outer_folds, args.seed
        )

        print("=" * 96)
        print("SEQUENCE-ONLY K-MER MODEL INTERPRETATION")
        print("=" * 96)
        print(f"Development samples: {len(data)}")
        print(f"Positive:            {(labels == 1).sum()}")
        print(f"Negative:            {(labels == 0).sum()}")
        print(f"Similarity groups:   {len(np.unique(groups))}")
        print(f"Frozen outer folds:  {args.outer_folds}")
        print("Primary analysis:    5-mers")
        print("Complementary:       6-mers")
        print("Prior motifs loaded: False")
        print("Frozen test used:    False")
        print("=" * 96)

        start = time.time()
        analyses = {}
        for k in (5, 6):
            print(f"Analyzing k={k}...")
            result = analyze_k(data, folds, fold_assignments, k, args)
            analyses[str(k)] = result
            m = result["pooled_oof_metrics_at_0_5"]
            print(
                f"k={k} | AUPRC={m['auprc']:.6f} AUROC={m['auroc']:.6f} "
                f"MCC={m['mcc']:.6f} | stable_positive="
                f"{result['stable_positive_kmers']} stable_negative="
                f"{result['stable_negative_kmers']}"
            )

        summary = {
            "experiment_name": "sequence_only_kmer_logistic_interpretation",
            "summary_scope": "public_sanitized",
            "test_set_used": False,
            "prior_regulatory_knowledge_loaded": False,
            "public_output_policy": {
                "real_kmer_identities_distributed": False,
                "sequence_level_predictions_distributed": False,
                "figures_use_anonymous_pattern_labels": True,
            },
            "development_data": {
                "sources": [str(args.train), str(args.validation)],
                "samples": int(len(data)),
                "positive": int((labels == 1).sum()),
                "negative": int((labels == 0).sum()),
                "positive_prevalence": float(labels.mean()),
                "similarity_groups": int(len(np.unique(groups))),
            },
            "evaluation_design": {
                "splitter": "StratifiedGroupKFold",
                "outer_folds": args.outer_folds,
                "seed": args.seed,
                "folds_frozen": True,
                "coefficient_ranking_statistic": "mean coefficient across outer-fold models",
                "full_development_coefficient_role": "secondary reference only",
            },
            "predefined_analyses": {
                "primary": {"k": 5, "reason": "Previously validated above permutation null."},
                "complementary": {"k": 6, "reason": "Later direct comparison with six-nucleotide cores."},
                "selection_policy": "Neither representation will be selected based on agreement with prior motifs.",
            },
            "excluded_training_features": [
                "motifs", "PWM_scores", "transcription_factor_annotations",
                "expression_values", "response_classes", "cis_regulatory_architecture",
                "results_from_previous_consultancy",
            ],
            "enrichment_analysis": {
                "unit": "k-mer presence per promoter",
                "test": "two-sided Fisher exact test",
                "multiple_testing": "Benjamini-Hochberg FDR",
            },
            "analyses": analyses,
            "runtime_seconds": float(time.time() - start),
        }
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("=" * 96)
        print("K-MER INTERPRETATION COMPLETE — PUBLIC-SAFE OUTPUTS")
        print("=" * 96)
        print("Prior motifs loaded: False")
        print("Frozen test used:    False")
        print(f"Summary:             {args.summary}")
        print(f"Tables:              {args.output_dir}")
        print(f"Figures:             {args.figure_dir}")
        print("=" * 96)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, KeyError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())