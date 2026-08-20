#!/usr/bin/env python3
"""Build the master sequence-only classification dataset.

The script combines:
    - positive promoter sequences;
    - GC-matched negative promoter sequences;
    - positive-class metadata.

No motif-derived or manually engineered biological features are included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


VALID_BASES = set("ACGTN")
EXPECTED_LENGTH = 2200


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the master promoter classification dataset."
    )

    parser.add_argument(
        "--positive-fasta",
        type=Path,
        default=Path("data/raw/positive_promoters.fasta"),
        help="FASTA file containing positive promoter sequences.",
    )
    parser.add_argument(
        "--negative-fasta",
        type=Path,
        default=Path("data/raw/negative_gc_matched_promoters.fasta"),
        help="FASTA file containing GC-matched negative promoter sequences.",
    )
    parser.add_argument(
        "--positive-labels",
        type=Path,
        default=Path("data/raw/positive_labels.csv"),
        help="CSV file containing positive sequence labels and metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Output master dataset.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/metadata/master_dataset_summary.json"),
        help="Output dataset summary.",
    )

    return parser.parse_args()


def normalize_sequence_id(header: str) -> str:
    """Extract a stable sequence identifier from a FASTA header."""

    raw_id = header.split()[0]

    # Basal promoter headers may contain genomic coordinates after "::".
    return raw_id.split("::")[0]


def read_fasta(path: Path) -> list[dict[str, str]]:
    """Read a FASTA file while preserving input record order."""

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: list[dict[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def store_record() -> None:
        if header is None:
            return

        sequence = "".join(sequence_parts).upper()

        records.append(
            {
                "sequence_id": normalize_sequence_id(header),
                "original_header": header,
                "sequence": sequence,
            }
        )

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                store_record()
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)

    store_record()

    if not records:
        raise ValueError(f"No FASTA records found in: {path}")

    return records


def gc_fraction(sequence: str) -> float:
    """Calculate GC content over the complete sequence length."""

    if not sequence:
        raise ValueError("Cannot calculate GC content for an empty sequence.")

    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def n_fraction(sequence: str) -> float:
    """Calculate the fraction of ambiguous N bases."""

    if not sequence:
        raise ValueError("Cannot calculate N content for an empty sequence.")

    return sequence.count("N") / len(sequence)


def sequence_hash(sequence: str) -> str:
    """Generate a SHA-256 checksum for sequence auditing."""

    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def validate_sequences(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate sequence IDs, lengths, alphabet and exact duplicates."""

    duplicated_ids = dataframe["sequence_id"].duplicated(keep=False)

    if duplicated_ids.any():
        examples = dataframe.loc[duplicated_ids, "sequence_id"].tolist()[:10]
        raise ValueError(
            f"{dataset_name}: duplicated sequence IDs detected: {examples}"
        )

    invalid_alphabet = dataframe["sequence"].map(
        lambda sequence: bool(set(sequence) - VALID_BASES)
    )

    if invalid_alphabet.any():
        examples = dataframe.loc[
            invalid_alphabet, ["sequence_id", "sequence"]
        ].head()

        raise ValueError(
            f"{dataset_name}: invalid nucleotide symbols detected:\n{examples}"
        )

    invalid_lengths = dataframe["sequence_length"] != EXPECTED_LENGTH

    if invalid_lengths.any():
        length_counts = (
            dataframe.loc[invalid_lengths, "sequence_length"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        raise ValueError(
            f"{dataset_name}: sequences with unexpected lengths: {length_counts}"
        )

    duplicated_sequences = dataframe["sequence_hash"].duplicated(keep=False)

    if duplicated_sequences.any():
        examples = dataframe.loc[
            duplicated_sequences,
            ["sequence_id", "sequence_hash"],
        ].head(10)

        raise ValueError(
            f"{dataset_name}: exact duplicated sequences detected:\n{examples}"
        )


def records_to_dataframe(
    records: list[dict[str, str]],
    label: int,
    class_name: str,
    source_group: str,
) -> pd.DataFrame:
    """Convert FASTA records to the common dataset representation."""

    dataframe = pd.DataFrame(records)

    dataframe["label"] = label
    dataframe["class_name"] = class_name
    dataframe["source_group"] = source_group
    dataframe["sequence_length"] = dataframe["sequence"].str.len()
    dataframe["gc_content"] = dataframe["sequence"].map(gc_fraction)
    dataframe["n_fraction"] = dataframe["sequence"].map(n_fraction)
    dataframe["sequence_hash"] = dataframe["sequence"].map(sequence_hash)

    return dataframe


def load_positive_metadata(path: Path) -> pd.DataFrame:
    """Load and standardize positive-class metadata."""

    if not path.exists():
        raise FileNotFoundError(f"Positive metadata file not found: {path}")

    metadata = pd.read_csv(path)

    required_columns = {"Gene", "Mean_Log2FC", "Induction_Class"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Positive metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata = metadata.rename(
        columns={
            "Gene": "sequence_id",
            "Mean_Log2FC": "response_value",
            "Induction_Class": "response_class",
        }
    )

    if metadata["sequence_id"].duplicated().any():
        duplicates = metadata.loc[
            metadata["sequence_id"].duplicated(keep=False),
            "sequence_id",
        ].tolist()

        raise ValueError(
            f"Positive metadata contains duplicated IDs: {duplicates[:10]}"
        )

    return metadata[
        [
            "sequence_id",
            "response_value",
            "response_class",
            "Support",
            "Mean_FDR",
        ]
    ].copy()


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Build and validate the complete master dataset."""

    positive_records = read_fasta(args.positive_fasta)
    negative_records = read_fasta(args.negative_fasta)

    positive_df = records_to_dataframe(
        positive_records,
        label=1,
        class_name="positive",
        source_group="experimentally_supported_positive",
    )

    negative_df = records_to_dataframe(
        negative_records,
        label=0,
        class_name="negative",
        source_group="gc_matched_reference",
    )

    validate_sequences(positive_df, "Positive dataset")
    validate_sequences(negative_df, "Negative dataset")

    positive_metadata = load_positive_metadata(args.positive_labels)

    positive_df = positive_df.merge(
        positive_metadata,
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )

    missing_metadata = positive_df["response_class"].isna()

    if missing_metadata.any():
        missing_ids = positive_df.loc[
            missing_metadata,
            "sequence_id",
        ].tolist()

        raise ValueError(
            "Positive FASTA IDs missing from metadata: "
            f"{missing_ids[:20]}"
        )

    # Negative examples do not have response measurements.
    for column in [
        "response_value",
        "response_class",
        "Support",
        "Mean_FDR",
    ]:
        negative_df[column] = pd.NA

    overlap_ids = set(positive_df["sequence_id"]) & set(
        negative_df["sequence_id"]
    )

    if overlap_ids:
        raise ValueError(
            "Positive and negative datasets share sequence IDs: "
            f"{sorted(overlap_ids)[:20]}"
        )

    overlap_hashes = set(positive_df["sequence_hash"]) & set(
        negative_df["sequence_hash"]
    )

    if overlap_hashes:
        raise ValueError(
            "Positive and negative datasets contain identical sequences."
        )

    master_dataset = pd.concat(
        [positive_df, negative_df],
        ignore_index=True,
    )

    master_dataset.insert(
        0,
        "sample_id",
        [
            f"SEQ_{index:05d}"
            for index in range(1, len(master_dataset) + 1)
        ],
    )

    column_order = [
        "sample_id",
        "sequence_id",
        "original_header",
        "sequence",
        "label",
        "class_name",
        "response_class",
        "response_value",
        "Support",
        "Mean_FDR",
        "source_group",
        "sequence_length",
        "gc_content",
        "n_fraction",
        "sequence_hash",
    ]

    master_dataset = master_dataset[column_order]

    positive_count = int((master_dataset["label"] == 1).sum())
    negative_count = int((master_dataset["label"] == 0).sum())

    summary = {
        "dataset_name": "sequence_only_promoter_classification",
        "total_sequences": int(len(master_dataset)),
        "positive_sequences": positive_count,
        "negative_sequences": negative_count,
        "positive_to_negative_ratio": f"1:{negative_count // positive_count}",
        "expected_sequence_length": EXPECTED_LENGTH,
        "all_sequences_expected_length": bool(
            (master_dataset["sequence_length"] == EXPECTED_LENGTH).all()
        ),
        "positive_mean_gc": float(positive_df["gc_content"].mean()),
        "negative_mean_gc": float(negative_df["gc_content"].mean()),
        "absolute_mean_gc_difference": float(
            abs(
                positive_df["gc_content"].mean()
                - negative_df["gc_content"].mean()
            )
        ),
        "positive_response_classes": {
            str(key): int(value)
            for key, value in positive_df["response_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "duplicate_sample_ids": int(
            master_dataset["sample_id"].duplicated().sum()
        ),
        "duplicate_sequence_ids": int(
            master_dataset["sequence_id"].duplicated().sum()
        ),
        "exact_sequence_overlap_between_classes": len(overlap_hashes),
        "sequence_features_used_for_training": ["raw_sequence"],
        "excluded_prior_knowledge_features": [
            "motif_presence",
            "pwm_scores",
            "transcription_factor_annotations",
            "cis_regulatory_architecture",
        ],
    }

    return master_dataset, summary


def main() -> int:
    """Run the dataset-building workflow."""

    args = parse_args()

    try:
        master_dataset, summary = build_dataset(args)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        master_dataset.to_csv(args.output, index=False)

        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print("=" * 72)
        print("MASTER DATASET BUILT SUCCESSFULLY")
        print("=" * 72)
        print(f"Output dataset: {args.output}")
        print(f"Summary:        {args.summary}")
        print(f"Total:          {summary['total_sequences']}")
        print(f"Positive:       {summary['positive_sequences']}")
        print(f"Negative:       {summary['negative_sequences']}")
        print(
            "Mean GC:        "
            f"positive={summary['positive_mean_gc']:.4f}, "
            f"negative={summary['negative_mean_gc']:.4f}"
        )
        print(
            "GC difference:  "
            f"{summary['absolute_mean_gc_difference']:.6f}"
        )
        print("=" * 72)

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
