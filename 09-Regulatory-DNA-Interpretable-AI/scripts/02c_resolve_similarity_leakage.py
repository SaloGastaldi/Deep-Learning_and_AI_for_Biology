#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SPLITS = ("train", "validation", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument(
        "--mixed-clusters",
        type=Path,
        default=Path("results/tables/02b_mixed_partition_clusters.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits_adjusted"),
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("data/metadata/similarity_adjustment_summary.json"),
    )
    return p.parse_args()


def load_splits(base: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        path = base / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df = df.copy()
        df["split"] = split
        out[split] = df
    combined = pd.concat(out.values(), ignore_index=True)
    if combined["sample_id"].duplicated().any():
        raise ValueError("sample_id overlap across splits")
    return out


def choose_target(cluster: pd.DataFrame) -> str:
    reps = cluster.loc[cluster["is_representative"].astype(str).str.lower() == "true"]
    if not reps.empty:
        return str(reps.iloc[0]["split"])
    return str(cluster["split"].value_counts().idxmax())


def same_class_candidates(df: pd.DataFrame, moved: pd.Series, protected: set[str]) -> pd.DataFrame:
    cand = df.loc[~df["sample_id"].isin(protected)].copy()
    cand = cand.loc[cand["label"] == moved["label"]]
    if int(moved["label"]) == 1:
        cand = cand.loc[cand["response_class"] == moved["response_class"]]
    return cand


def main() -> int:
    args = parse_args()
    try:
        splits = load_splits(args.splits_dir)
        mixed = pd.read_csv(args.mixed_clusters)
        if mixed.empty:
            raise ValueError("No mixed clusters found; no adjustment required.")

        protected = set(mixed["sample_id"].astype(str))
        moves = []
        swaps = []

        for (_, cluster_id), cluster in mixed.groupby(
            ["identity_threshold", "cluster_id"], sort=True
        ):
            target = choose_target(cluster)

            for _, member in cluster.iterrows():
                source = str(member["split"])
                if source == target:
                    continue

                moved_id = str(member["sample_id"])
                moved_df = splits[source].loc[
                    splits[source]["sample_id"] == moved_id
                ]
                if len(moved_df) != 1:
                    raise ValueError(f"Could not uniquely locate {moved_id}")
                moved = moved_df.iloc[0]

                candidates = same_class_candidates(
                    splits[target], moved, protected
                )
                if candidates.empty:
                    raise ValueError(
                        f"No eligible balancing swap for {moved_id}"
                    )

                candidates = candidates.copy()
                candidates["gc_distance"] = (
                    candidates["gc_content"].astype(float)
                    - float(moved["gc_content"])
                ).abs()
                candidates = candidates.sort_values(
                    ["gc_distance", "sample_id"]
                )
                swap = candidates.iloc[0]
                swap_id = str(swap["sample_id"])

                splits[source] = splits[source].loc[
                    splits[source]["sample_id"] != moved_id
                ].copy()
                splits[target] = splits[target].loc[
                    splits[target]["sample_id"] != swap_id
                ].copy()

                moved_row = moved.copy()
                moved_row["split"] = target
                swap_row = swap.copy()
                swap_row["split"] = source

                splits[target] = pd.concat(
                    [splits[target], moved_row.to_frame().T],
                    ignore_index=True,
                )
                splits[source] = pd.concat(
                    [splits[source], swap_row.to_frame().T],
                    ignore_index=True,
                )

                moves.append(
                    {
                        "identity_threshold": float(member["identity_threshold"]),
                        "cluster_id": int(cluster_id),
                        "sample_id": moved_id,
                        "sequence_id": moved["sequence_id"],
                        "from_split": source,
                        "to_split": target,
                    }
                )
                swaps.append(
                    {
                        "sample_id": swap_id,
                        "sequence_id": swap["sequence_id"],
                        "from_split": target,
                        "to_split": source,
                        "gc_distance": float(candidates.iloc[0]["gc_distance"]),
                    }
                )

        original = load_splits(args.splits_dir)

        for split in SPLITS:
            if len(original[split]) != len(splits[split]):
                raise ValueError(f"{split} size changed")
            if (
                original[split]["label"].value_counts().sort_index().to_dict()
                != splits[split]["label"].value_counts().sort_index().to_dict()
            ):
                raise ValueError(f"{split} label distribution changed")

        combined = pd.concat(splits.values(), ignore_index=True)
        if combined["sample_id"].duplicated().any():
            raise ValueError("Adjusted splits overlap")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)

        manifest_cols = [
            "sample_id",
            "sequence_id",
            "label",
            "response_class",
            "sequence_hash",
            "split",
        ]

        for split in SPLITS:
            df = splits[split].sample(
                frac=1,
                random_state={"train": 123, "validation": 124, "test": 125}[split],
            ).reset_index(drop=True)
            df["split"] = split
            df.to_csv(args.output_dir / f"{split}.csv", index=False)
            df[manifest_cols].to_csv(
                args.output_dir / f"{split}_manifest.csv", index=False
            )

        summary = {
            "source_splits": str(args.splits_dir),
            "output_splits": str(args.output_dir),
            "moves": moves,
            "balancing_swaps": swaps,
            "partition_sizes_preserved": True,
            "binary_label_counts_preserved": True,
        }
        args.summary.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("=" * 72)
        print("SIMILARITY-BASED SPLIT ADJUSTMENT COMPLETE")
        print("=" * 72)
        for move, swap in zip(moves, swaps):
            print(
                f"Moved {move['sequence_id']} "
                f"{move['from_split']} -> {move['to_split']}; "
                f"swapped {swap['sequence_id']} "
                f"{swap['from_split']} -> {swap['to_split']}"
            )
        print(f"Adjusted splits: {args.output_dir}")
        print(f"Summary:         {args.summary}")
        print("=" * 72)
        return 0

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

