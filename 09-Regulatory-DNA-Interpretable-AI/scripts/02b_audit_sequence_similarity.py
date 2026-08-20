#!/usr/bin/env python3
"""Audit sequence similarity across train, validation and test partitions.

This script:
    1. Reads train, validation and test CSV files.
    2. Builds a combined FASTA with partition metadata in each header.
    3. Runs CD-HIT-EST at configurable identity thresholds.
    4. Parses the resulting .clstr files.
    5. Reports clusters that mix dataset partitions.
    6. Writes machine-readable CSV/JSON outputs and a human-readable report.

The purpose is to detect potential biological data leakage caused by highly
similar promoter sequences distributed across different dataset partitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_THRESHOLDS = (0.90, 0.80)


@dataclass(frozen=True)
class ClusterMember:
    """Represent one sequence member from a CD-HIT cluster."""

    cluster_id: int
    sample_id: str
    sequence_id: str
    split: str
    label: int
    response_class: str | None
    is_representative: bool
    identity_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit biological leakage by clustering promoter sequences "
            "across train, validation and test partitions."
        )
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/similarity_audit"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/reports/02b_sequence_similarity_audit.txt"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/metrics/02b_sequence_similarity_summary.json"),
    )
    parser.add_argument(
        "--mixed-clusters-csv",
        type=Path,
        default=Path("results/tables/02b_mixed_partition_clusters.csv"),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--memory-mb", type=int, default=0)
    return parser.parse_args()


def validate_thresholds(thresholds: Iterable[float]) -> list[float]:
    normalized = sorted(set(float(value) for value in thresholds), reverse=True)
    for threshold in normalized:
        if not 0.80 <= threshold <= 1.00:
            raise ValueError(
                "Identity thresholds must be between 0.80 and 1.00. "
                f"Observed: {threshold}"
            )
    return normalized


def find_cdhit() -> str:
    executable = shutil.which("cd-hit-est")
    if executable is None:
        raise FileNotFoundError(
            "cd-hit-est was not found in PATH. Activate the project Conda "
            "environment or install CD-HIT before running this script."
        )
    return executable


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    dataframe = pd.read_csv(path)
    required_columns = {
        "sample_id",
        "sequence_id",
        "sequence",
        "label",
        "response_class",
        "sequence_hash",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )
    dataframe = dataframe.copy()
    dataframe["split"] = split_name
    if dataframe["sample_id"].duplicated().any():
        raise ValueError(f"Duplicated sample IDs detected in {path}")
    if dataframe["sequence_hash"].duplicated().any():
        raise ValueError(f"Duplicated sequence hashes detected in {path}")
    if dataframe["sequence"].isna().any():
        raise ValueError(f"Missing sequences detected in {path}")
    dataframe["sequence"] = dataframe["sequence"].astype(str).str.upper()
    return dataframe


def load_all_splits(splits_dir: Path) -> pd.DataFrame:
    split_files = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "validation.csv",
        "test": splits_dir / "test.csv",
    }
    frames = [
        load_split(path, split_name)
        for split_name, path in split_files.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("Sample IDs overlap across partitions.")
    if combined["sequence_hash"].duplicated().any():
        raise ValueError("Exact sequences overlap across partitions.")
    return combined


def safe_header_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return str(value).replace("|", "_").replace(" ", "_")


def build_header(row: pd.Series) -> str:
    return "|".join(
        [
            safe_header_value(row["sample_id"]),
            safe_header_value(row["sequence_id"]),
            safe_header_value(row["split"]),
            f"label={int(row['label'])}",
            f"class={safe_header_value(row['response_class'])}",
        ]
    )


def write_combined_fasta(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _, row in dataframe.iterrows():
            handle.write(f">{build_header(row)}\n")
            sequence = row["sequence"]
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


def threshold_tag(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def run_cdhit(
    executable: str,
    input_fasta: Path,
    output_prefix: Path,
    threshold: float,
    threads: int,
    memory_mb: int,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-i", str(input_fasta),
        "-o", str(output_prefix),
        "-c", f"{threshold:.2f}",
        "-n", str(word_size_for_threshold(threshold)),
        "-G", "1",
        "-aS", "0.80",
        "-aL", "0.80",
        "-g", "1",
        "-d", "0",
        "-T", str(threads),
        "-M", str(memory_mb),
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
            f"CD-HIT-EST failed for threshold {threshold:.2f}. "
            f"See log: {log_path}"
        )


def parse_member_header(header: str) -> dict[str, object]:
    parts = header.split("|")
    if len(parts) != 5:
        raise ValueError(f"Unexpected FASTA header format: {header}")
    sample_id, sequence_id, split_name, label_field, class_field = parts
    if not label_field.startswith("label="):
        raise ValueError(f"Missing label field in header: {header}")
    if not class_field.startswith("class="):
        raise ValueError(f"Missing class field in header: {header}")
    label = int(label_field.split("=", maxsplit=1)[1])
    response_class = class_field.split("=", maxsplit=1)[1]
    if response_class == "NA":
        response_class = None
    return {
        "sample_id": sample_id,
        "sequence_id": sequence_id,
        "split": split_name,
        "label": label,
        "response_class": response_class,
    }


def parse_clstr(path: Path) -> list[ClusterMember]:
    if not path.exists():
        raise FileNotFoundError(f"CD-HIT cluster file not found: {path}")
    members: list[ClusterMember] = []
    current_cluster: int | None = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster "):
                current_cluster = int(line.split()[1])
                continue
            if current_cluster is None:
                raise ValueError(
                    f"Cluster member found before cluster header in {path}"
                )
            if ">" not in line or "..." not in line:
                raise ValueError(f"Unexpected .clstr member line: {line}")
            header_fragment = (
                line.split(">", maxsplit=1)[1]
                .split("...", maxsplit=1)[0]
            )
            metadata = parse_member_header(header_fragment)
            is_representative = line.endswith("*")
            identity_text = (
                "*" if is_representative else line.rsplit(" ", maxsplit=1)[-1]
            )
            members.append(
                ClusterMember(
                    cluster_id=current_cluster,
                    sample_id=str(metadata["sample_id"]),
                    sequence_id=str(metadata["sequence_id"]),
                    split=str(metadata["split"]),
                    label=int(metadata["label"]),
                    response_class=metadata["response_class"],
                    is_representative=is_representative,
                    identity_text=identity_text,
                )
            )
    return members


def summarize_clusters(
    members: list[ClusterMember],
    threshold: float,
) -> tuple[dict, list[dict[str, object]]]:
    cluster_map: dict[int, list[ClusterMember]] = {}
    for member in members:
        cluster_map.setdefault(member.cluster_id, []).append(member)

    multi_member_clusters = {
        cluster_id: cluster_members
        for cluster_id, cluster_members in cluster_map.items()
        if len(cluster_members) > 1
    }
    mixed_clusters = {
        cluster_id: cluster_members
        for cluster_id, cluster_members in multi_member_clusters.items()
        if len({member.split for member in cluster_members}) > 1
    }

    train_validation = 0
    train_test = 0
    validation_test = 0
    all_three = 0

    for cluster_members in mixed_clusters.values():
        splits = {member.split for member in cluster_members}
        if splits == {"train", "validation", "test"}:
            all_three += 1
        if "train" in splits and "validation" in splits:
            train_validation += 1
        if "train" in splits and "test" in splits:
            train_test += 1
        if "validation" in splits and "test" in splits:
            validation_test += 1

    mixed_rows: list[dict[str, object]] = []
    for cluster_id, cluster_members in sorted(mixed_clusters.items()):
        cluster_splits = ",".join(
            sorted({member.split for member in cluster_members})
        )
        for member in cluster_members:
            mixed_rows.append(
                {
                    "identity_threshold": threshold,
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster_members),
                    "cluster_splits": cluster_splits,
                    "sample_id": member.sample_id,
                    "sequence_id": member.sequence_id,
                    "split": member.split,
                    "label": member.label,
                    "response_class": member.response_class,
                    "is_representative": member.is_representative,
                    "identity_to_representative": member.identity_text,
                }
            )

    summary = {
        "identity_threshold": threshold,
        "total_sequences": len(members),
        "total_clusters": len(cluster_map),
        "singleton_clusters": sum(
            len(cluster_members) == 1
            for cluster_members in cluster_map.values()
        ),
        "multi_member_clusters": len(multi_member_clusters),
        "mixed_partition_clusters": len(mixed_clusters),
        "train_validation_mixed_clusters": train_validation,
        "train_test_mixed_clusters": train_test,
        "validation_test_mixed_clusters": validation_test,
        "all_three_partitions_mixed_clusters": all_three,
        "maximum_cluster_size": max(
            (len(cluster_members) for cluster_members in cluster_map.values()),
            default=0,
        ),
        "mixed_cluster_ids": sorted(mixed_clusters),
        "potential_leakage_detected": bool(mixed_clusters),
    }
    return summary, mixed_rows


def write_mixed_clusters_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "identity_threshold",
        "cluster_id",
        "cluster_size",
        "cluster_splits",
        "sample_id",
        "sequence_id",
        "split",
        "label",
        "response_class",
        "is_representative",
        "identity_to_representative",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    combined: pd.DataFrame,
    summaries: list[dict],
    cdhit_path: str,
) -> str:
    lines = [
        "=" * 78,
        "SEQUENCE SIMILARITY AUDIT — TRAIN / VALIDATION / TEST",
        "=" * 78,
        "",
        "Purpose",
        "-------",
        (
            "Detect potential biological data leakage caused by highly similar "
            "promoter sequences assigned to different dataset partitions."
        ),
        "",
        "Input dataset",
        "-------------",
        f"Total sequences: {len(combined)}",
        f"Train:           {(combined['split'] == 'train').sum()}",
        f"Validation:      {(combined['split'] == 'validation').sum()}",
        f"Test:            {(combined['split'] == 'test').sum()}",
        f"Positive:        {(combined['label'] == 1).sum()}",
        f"Negative:        {(combined['label'] == 0).sum()}",
        "",
        f"CD-HIT-EST executable: {cdhit_path}",
        "",
    ]

    any_leakage = False
    for summary in summaries:
        any_leakage = any_leakage or summary["potential_leakage_detected"]
        lines.extend(
            [
                "-" * 78,
                f"Identity threshold: {summary['identity_threshold']:.2f}",
                "-" * 78,
                f"Total clusters:                       {summary['total_clusters']}",
                f"Singleton clusters:                   {summary['singleton_clusters']}",
                f"Clusters with >1 sequence:            {summary['multi_member_clusters']}",
                f"Mixed-partition clusters:             {summary['mixed_partition_clusters']}",
                f"Train / Validation mixed clusters:    {summary['train_validation_mixed_clusters']}",
                f"Train / Test mixed clusters:          {summary['train_test_mixed_clusters']}",
                f"Validation / Test mixed clusters:     {summary['validation_test_mixed_clusters']}",
                f"Clusters containing all partitions:   {summary['all_three_partitions_mixed_clusters']}",
                f"Maximum cluster size:                  {summary['maximum_cluster_size']}",
                "",
            ]
        )
        if summary["potential_leakage_detected"]:
            lines.append("Result: POTENTIAL BIOLOGICAL LEAKAGE DETECTED.")
            lines.append(
                "Review the mixed-cluster CSV before finalizing the partitions."
            )
        else:
            lines.append(
                "Result: no cross-partition clusters detected at this threshold."
            )
        lines.append("")

    lines.extend(["=" * 78, "OVERALL CONCLUSION", "=" * 78])
    if any_leakage:
        lines.extend(
            [
                (
                    "At least one identity threshold produced clusters containing "
                    "sequences from different partitions."
                ),
                (
                    "The current split should be reviewed before model training. "
                    "Sequences from the same similarity cluster may need to be "
                    "assigned to a single partition."
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "No cross-partition sequence clusters were detected at the "
                    "evaluated identity thresholds."
                ),
                (
                    "The audit found no evidence of biological leakage caused by "
                    "highly similar promoter sequences under the selected CD-HIT "
                    "criteria."
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "Important: absence of clustered similarity does not prove complete "
                "biological independence. It indicates that no sequence pairs met "
                "the evaluated global identity and coverage criteria."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        thresholds = validate_thresholds(args.thresholds)
        cdhit_executable = find_cdhit()
        combined = load_all_splits(args.splits_dir)

        args.work_dir.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.mixed_clusters_csv.parent.mkdir(parents=True, exist_ok=True)

        combined_fasta = args.work_dir / "all_partitions.fasta"
        write_combined_fasta(combined, combined_fasta)

        summaries: list[dict] = []
        all_mixed_rows: list[dict[str, object]] = []

        for threshold in thresholds:
            tag = threshold_tag(threshold)
            output_prefix = args.work_dir / f"clusters_identity_{tag}"
            print(
                f"Running CD-HIT-EST at identity threshold {threshold:.2f}..."
            )
            run_cdhit(
                executable=cdhit_executable,
                input_fasta=combined_fasta,
                output_prefix=output_prefix,
                threshold=threshold,
                threads=args.threads,
                memory_mb=args.memory_mb,
            )
            cluster_file = Path(str(output_prefix) + ".clstr")
            members = parse_clstr(cluster_file)
            summary, mixed_rows = summarize_clusters(members, threshold)
            summaries.append(summary)
            all_mixed_rows.extend(mixed_rows)

        write_mixed_clusters_csv(all_mixed_rows, args.mixed_clusters_csv)

        json_summary = {
            "audit_name": "cross_partition_sequence_similarity",
            "cd_hit_est_executable": cdhit_executable,
            "coverage_policy": {
                "global_identity": True,
                "shorter_sequence_coverage": 0.80,
                "longer_sequence_coverage": 0.80,
            },
            "input_counts": {
                "total": int(len(combined)),
                "train": int((combined["split"] == "train").sum()),
                "validation": int((combined["split"] == "validation").sum()),
                "test": int((combined["split"] == "test").sum()),
                "positive": int((combined["label"] == 1).sum()),
                "negative": int((combined["label"] == 0).sum()),
            },
            "threshold_results": summaries,
            "overall_potential_leakage_detected": any(
                summary["potential_leakage_detected"]
                for summary in summaries
            ),
        }

        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(json_summary, handle, indent=2, ensure_ascii=False)

        args.report.write_text(
            build_report(combined, summaries, cdhit_executable),
            encoding="utf-8",
        )

        print("=" * 78)
        print("SEQUENCE SIMILARITY AUDIT COMPLETE")
        print("=" * 78)
        for summary in summaries:
            print(
                f"Identity {summary['identity_threshold']:.2f}: "
                f"clusters={summary['total_clusters']}, "
                f"multi_member={summary['multi_member_clusters']}, "
                f"mixed_partitions={summary['mixed_partition_clusters']}"
            )
        print(f"Report:         {args.report}")
        print(f"JSON summary:   {args.summary_json}")
        print(f"Mixed clusters: {args.mixed_clusters_csv}")
        print("=" * 78)

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        pd.errors.ParserError,
        subprocess.SubprocessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
