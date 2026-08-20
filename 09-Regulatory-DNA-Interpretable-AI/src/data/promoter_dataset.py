"""PyTorch Dataset for sequence-only promoter classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.encoding.one_hot import one_hot_encode_tensor


class PromoterDataset(Dataset):
    """Load promoter sequences and binary labels from a split CSV.

    Only the raw DNA sequence is transformed into model input. Biological
    metadata such as response class, expression values, GC content and motif
    annotations are not included in the feature tensor.
    """

    REQUIRED_COLUMNS = {
        "sample_id",
        "sequence_id",
        "sequence",
        "label",
        "response_class",
        "sequence_length",
    }

    def __init__(
        self,
        csv_path: str | Path,
        expected_length: int = 2200,
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.expected_length = expected_length
        self.return_metadata = return_metadata

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {self.csv_path}"
            )

        self.dataframe = pd.read_csv(self.csv_path)

        missing_columns = (
            self.REQUIRED_COLUMNS - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{self.csv_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if self.dataframe["sample_id"].duplicated().any():
            raise ValueError(
                f"Duplicated sample IDs detected in {self.csv_path}"
            )

        observed_labels = set(self.dataframe["label"].unique())

        if not observed_labels.issubset({0, 1}):
            raise ValueError(
                f"Expected binary labels 0/1, observed {observed_labels}"
            )

        invalid_lengths = (
            self.dataframe["sequence_length"] != self.expected_length
        )

        if invalid_lengths.any():
            counts = (
                self.dataframe.loc[
                    invalid_lengths,
                    "sequence_length",
                ]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            raise ValueError(
                f"Unexpected sequence lengths in {self.csv_path}: {counts}"
            )

    def __len__(self) -> int:
        """Return the number of sequences."""

        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, Any]:
        """Return one encoded promoter and its binary label."""

        row = self.dataframe.iloc[index]

        features = one_hot_encode_tensor(
            sequence=str(row["sequence"]),
            expected_length=self.expected_length,
        )

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        if not self.return_metadata:
            return features, label

        response_class = (
            None
            if pd.isna(row["response_class"])
            else str(row["response_class"])
        )

        return {
            "features": features,
            "label": label,
            "sample_id": str(row["sample_id"]),
            "sequence_id": str(row["sequence_id"]),
            "response_class": response_class,
        }

    @property
    def positive_count(self) -> int:
        """Return the number of positive examples."""

        return int((self.dataframe["label"] == 1).sum())

    @property
    def negative_count(self) -> int:
        """Return the number of negative examples."""

        return int((self.dataframe["label"] == 0).sum())

    @property
    def positive_weight(self) -> float:
        """Return negative/positive ratio for BCEWithLogitsLoss."""

        if self.positive_count == 0:
            raise ValueError("Cannot calculate positive weight without positives.")

        return self.negative_count / self.positive_count
