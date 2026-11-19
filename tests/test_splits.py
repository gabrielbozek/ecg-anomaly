"""Check reproducibility and subject isolation in the saved dataset split."""

from pathlib import Path
import unittest

import pandas as pd

from scripts.create_splits import build_record_splits, validate_record_splits

ROOT = Path(__file__).resolve().parents[1]


class SubjectSplitTests(unittest.TestCase):
    """Exercise the manifest and deliberate cases of leakage or invalid coverage."""

    def setUp(self):
        """Load the source record IDs and construct the seed-42 reference split."""
        self.record_ids = (ROOT / "data/RECORDS").read_text().splitlines()
        self.splits = build_record_splits(self.record_ids, seed=42)

    def test_saved_manifest_keeps_subject_sets_disjoint(self):
        """Check the actual CSV for record coverage and pairwise subject isolation."""
        saved = pd.read_csv(ROOT / "data/record_splits.csv", dtype=str)
        validate_record_splits(saved, self.record_ids)
        self.assertEqual(saved.to_dict("records"), self.splits.to_dict("records"))
        self.assertEqual(len(saved), 48)
        self.assertEqual(set(saved["record_id"]), set(self.record_ids))
        groups = {name: set(group["subject_id"]) for name, group in saved.groupby("split")}
        self.assertTrue(groups["train"].isdisjoint(groups["validation"]))
        self.assertTrue(groups["train"].isdisjoint(groups["test"]))
        self.assertTrue(groups["validation"].isdisjoint(groups["test"]))

    def test_subject_counts_and_fixed_training_record(self):
        """Check 29/9/9 subjects and the required training assignment for record 100."""
        self.assertEqual(
            self.splits.groupby("split")["subject_id"].nunique().to_dict(),
            {"train": 29, "validation": 9, "test": 9},
        )
        self.assertEqual(self.splits.set_index("record_id").loc["100", "split"], "train")

    def test_records_201_and_202_share_one_subject_and_split(self):
        """Verify the known shared-subject pair independently of the validator."""
        pair = self.splits.set_index("record_id").loc[["201", "202"]]
        self.assertEqual(pair["subject_id"].tolist(), ["201", "201"])
        self.assertEqual(pair["split"].nunique(), 1)

    def test_split_is_reproducible_and_independent_of_record_order(self):
        """Check repeated seeds and reordered inputs produce the same assignments."""
        repeated = build_record_splits(self.record_ids, seed=42)
        reversed_input = build_record_splits(self.record_ids[::-1], seed=42)
        pd.testing.assert_frame_equal(self.splits, repeated)
        pd.testing.assert_frame_equal(
            self.splits.sort_values("record_id").reset_index(drop=True),
            reversed_input.sort_values("record_id").reset_index(drop=True),
        )

    def test_validator_rejects_subject_leakage(self):
        """Move record 202 to another split and require the leakage check to fail."""
        tampered = self.splits.copy()
        current = tampered.loc[tampered["record_id"] == "201", "split"].iloc[0]
        other_split = "test" if current != "test" else "validation"
        tampered.loc[tampered["record_id"] == "202", "split"] = other_split
        with self.assertRaisesRegex(ValueError, "subject leakage"):
            validate_record_splits(tampered, self.record_ids)

    def test_validator_rejects_incorrect_subject_mapping(self):
        """Prevent a false subject ID from hiding the relationship between 201/202."""
        tampered = self.splits.copy()
        tampered.loc[tampered["record_id"] == "202", "subject_id"] = "202"
        with self.assertRaisesRegex(ValueError, "Subject mapping"):
            validate_record_splits(tampered, self.record_ids)

    def test_validator_rejects_duplicate_or_missing_records(self):
        """Reject coverage that drops a record or includes the same record twice."""
        for tampered in (self.splits.iloc[:-1], pd.concat([self.splits, self.splits.iloc[:1]])):
            with self.subTest(rows=len(tampered)):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_record_splits(tampered, self.record_ids)

    def test_validator_rejects_record_100_outside_train(self):
        """Reject an assignment that violates the fixed training requirement."""
        tampered = self.splits.copy()
        tampered.loc[tampered["record_id"] == "100", "split"] = "test"
        with self.assertRaisesRegex(ValueError, "Record 100"):
            validate_record_splits(tampered, self.record_ids)

    def test_validator_rejects_wrong_subject_counts(self):
        """Reject a subject-isolated manifest whose sizes no longer match 29/9/9."""
        tampered = self.splits.copy()
        candidate = tampered.loc[
            (tampered["split"] == "train") & ~tampered["record_id"].isin(["100", "201", "202"])
        ].index[0]
        tampered.loc[candidate, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Expected 29 training"):
            validate_record_splits(tampered, self.record_ids)


if __name__ == "__main__":
    unittest.main()
