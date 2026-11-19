"""Create the fixed subject split for MIT-BIH without changing raw recordings."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SUBJECT_COUNTS = {"train": 29, "validation": 9, "test": 9}
SUBJECT_OVERRIDES = {"202": "201"}


def validate_record_splits(splits: pd.DataFrame, record_ids: Sequence[str]) -> None:
    """Check record coverage, subject grouping, split sizes, and leakage.

    Args:
        splits: Table with string columns record_id, subject_id, and split.
        record_ids: All 48 unique record IDs from the dataset's RECORDS file.

    Returns:
        None. Raises ValueError if the manifest violates the fixed split rules.
    """
    expected_records = list(record_ids)
    if len(expected_records) != 48 or len(set(expected_records)) != 48:
        raise ValueError("Expected 48 unique source record IDs.")
    if not {"100", "201", "202"}.issubset(expected_records):
        raise ValueError("Source records must include 100, 201, and 202.")
    if list(splits.columns) != ["record_id", "subject_id", "split"]:
        raise ValueError("Expected columns: record_id, subject_id, split.")
    if splits.isna().any().any():
        raise ValueError("Split assignments must not contain missing values.")
    if any(not isinstance(value, str) for column in splits for value in splits[column]):
        raise ValueError("Read record IDs, subject IDs, and split labels as strings.")
    if splits["record_id"].duplicated().any() or set(splits["record_id"]) != set(expected_records):
        raise ValueError("Every source record must appear exactly once.")
    expected_subjects = splits["record_id"].replace(SUBJECT_OVERRIDES)
    if not splits["subject_id"].equals(expected_subjects):
        raise ValueError("Subject mapping must group records 201 and 202 under subject 201.")
    if not splits["split"].isin(SPLIT_SUBJECT_COUNTS).all():
        raise ValueError("Split labels must be train, validation, or test.")
    if splits.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError("Each subject must belong to exactly one split; subject leakage detected.")
    if splits.loc[splits["record_id"] == "100", "split"].iloc[0] != "train":
        raise ValueError("Record 100 must belong to the training split.")
    subject_counts = splits.drop_duplicates("subject_id")["split"].value_counts().to_dict()
    if subject_counts != SPLIT_SUBJECT_COUNTS:
        raise ValueError("Expected 29 training, 9 validation, and 9 test subjects.")


def build_record_splits(record_ids: Sequence[str], seed: int = 42) -> pd.DataFrame:
    """Assign 47 subjects to train/validation/test while fixing record 100 in train.

    Subject IDs are project grouping keys: a record's own ID, except that
    record 202 uses subject 201. They are not external patient identifiers.
    Shuffle the sorted remaining 46 subject IDs with numpy.default_rng(seed),
    then assign 28 to train, 9 to validation, and 9 to test.

    Args:
        record_ids: All 48 unique source record IDs, as strings. Output rows
            preserve this order; assignments are independent of input order.
        seed: Random seed for the subject permutation (default: 42).

    Returns:
        Validated DataFrame with record_id, subject_id, and split columns.
    """
    splits = pd.DataFrame({"record_id": list(record_ids)})
    splits["subject_id"] = splits["record_id"].replace(SUBJECT_OVERRIDES)
    subjects = sorted(set(splits["subject_id"]))
    if len(subjects) != 47 or "100" not in subjects:
        raise ValueError("Expected 47 subjects, including the subject of record 100.")
    remaining = np.random.default_rng(seed).permutation([subject for subject in subjects if subject != "100"])
    assignments = {"100": "train"}
    first = 0
    for split, subject_count in SPLIT_SUBJECT_COUNTS.items():
        count = subject_count - 1 if split == "train" else subject_count
        assignments.update({subject: split for subject in remaining[first:first + count]})
        first += count
    splits["split"] = splits["subject_id"].map(assignments)
    validate_record_splits(splits, record_ids)
    return splits


def main() -> None:
    """Write data/record_splits.csv with seed 42, preserving any differing file.

    Returns:
        None. Prints subject and record counts after validating the manifest.
    """
    record_ids = (ROOT / "data/RECORDS").read_text().splitlines()
    splits = build_record_splits(record_ids, seed=42)
    destination = ROOT / "data/record_splits.csv"
    contents = splits.to_csv(index=False, lineterminator="\n")
    if destination.exists() and destination.read_text() != contents:
        raise ValueError(f"Existing split file differs from seed 42: {destination}. File preserved.")
    if not destination.exists():
        destination.write_text(contents)
    print("Split manifest ready: data/record_splits.csv (seed 42).")
    print(splits.groupby("split").agg(
        subjects=("subject_id", "nunique"), records=("record_id", "size")
    ).reindex(SPLIT_SUBJECT_COUNTS).to_string())


if __name__ == "__main__":
    main()
