#!/usr/bin/env python3
"""Validate DNA one-hot encoding and PyTorch dataset loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.promoter_dataset import PromoterDataset
from src.encoding.one_hot import (
    decode_one_hot,
    one_hot_encode_numpy,
)


SPLITS_DIR = Path("data/splits")
REPORT_PATH = Path("data/metadata/encoding_validation_summary.json")
EXPECTED_LENGTH = 2200
BATCH_SIZE = 32


def validate_round_trip(dataset: PromoterDataset) -> None:
    """Verify that encoding and decoding recover original sequences."""

    for index in range(len(dataset)):
        original = str(dataset.dataframe.iloc[index]["sequence"]).upper()

        encoded = one_hot_encode_numpy(
            original,
            expected_length=EXPECTED_LENGTH,
        )

        decoded = decode_one_hot(encoded)

        if decoded != original:
            sample_id = dataset.dataframe.iloc[index]["sample_id"]
            raise ValueError(
                f"Encoding round-trip failed for sample {sample_id}"
            )


def validate_one_hot_values(features: torch.Tensor) -> None:
    """Verify tensor shape and valid one-hot channel sums."""

    if features.ndim != 3:
        raise ValueError(
            f"Expected batch tensor with 3 dimensions, got {features.shape}"
        )

    if features.shape[1:] != (4, EXPECTED_LENGTH):
        raise ValueError(
            "Expected batch shape (batch, 4, 2200), "
            f"observed {tuple(features.shape)}"
        )

    unique_values = set(features.unique().cpu().tolist())

    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"Unexpected encoding values: {sorted(unique_values)}"
        )

    position_sums = features.sum(dim=1)

    if not torch.all(
        (position_sums == 0.0) | (position_sums == 1.0)
    ):
        raise ValueError(
            "At least one nucleotide position activates multiple channels."
        )


def main() -> int:
    try:
        summary: dict[str, object] = {
            "encoding": {
                "alphabet_order": ["A", "C", "G", "T"],
                "ambiguous_base_policy": "N_encoded_as_all_zeros",
                "tensor_layout": "channels_first",
                "sample_shape": [4, EXPECTED_LENGTH],
                "dtype": "torch.float32",
            },
            "splits": {},
        }

        for split_name in ["train", "validation", "test"]:
            dataset = PromoterDataset(
                SPLITS_DIR / f"{split_name}.csv",
                expected_length=EXPECTED_LENGTH,
                return_metadata=False,
            )

            validate_round_trip(dataset)

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            batch_features, batch_labels = next(iter(loader))

            validate_one_hot_values(batch_features)

            if batch_labels.ndim != 1:
                raise ValueError(
                    f"Expected one-dimensional labels, "
                    f"observed {batch_labels.shape}"
                )

            observed_labels = set(
                batch_labels.unique().cpu().tolist()
            )

            if not observed_labels.issubset({0.0, 1.0}):
                raise ValueError(
                    f"Unexpected labels: {sorted(observed_labels)}"
                )

            sequence_tensor = dataset[0][0]

            summary["splits"][split_name] = {
                "samples": len(dataset),
                "positive": dataset.positive_count,
                "negative": dataset.negative_count,
                "positive_weight": dataset.positive_weight,
                "first_sample_shape": list(sequence_tensor.shape),
                "batch_shape": list(batch_features.shape),
                "batch_label_shape": list(batch_labels.shape),
                "round_trip_validation": "PASS",
                "one_hot_value_validation": "PASS",
            }

            print(
                f"{split_name:12s}: "
                f"samples={len(dataset):4d}, "
                f"tensor={tuple(sequence_tensor.shape)}, "
                f"batch={tuple(batch_features.shape)}, "
                f"pos_weight={dataset.positive_weight:.4f}"
            )

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with REPORT_PATH.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 72)
        print("ONE-HOT ENCODING AND DATASET VALIDATION PASSED")
        print("=" * 72)
        print(f"Summary: {REPORT_PATH}")

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        IndexError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
