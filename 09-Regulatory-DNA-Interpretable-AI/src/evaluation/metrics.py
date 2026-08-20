"""Classification metrics for binary promoter prediction."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert values to a one-dimensional NumPy array."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    return array.reshape(-1)


def binary_classification_metrics(
    targets: Iterable[float] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate complementary binary-classification metrics.

    Parameters
    ----------
    targets
        Binary ground-truth labels.
    probabilities
        Positive-class probabilities.
    threshold
        Probability threshold used to generate hard predictions.

    Returns
    -------
    dict
        AUPRC, AUROC, MCC, F1, precision, recall, balanced accuracy
        and accuracy.
    """

    y_true = _to_numpy(targets).astype(int)
    y_prob = _to_numpy(probabilities).astype(float)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Target shape {y_true.shape} does not match "
            f"probability shape {y_prob.shape}."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Targets must contain only binary labels 0 and 1.")

    if np.any(~np.isfinite(y_prob)):
        raise ValueError("Probabilities contain non-finite values.")

    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("Probabilities must lie within [0, 1].")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between 0 and 1.")

    y_pred = (y_prob >= threshold).astype(int)

    auroc = math.nan

    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, y_prob))

    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": auroc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
