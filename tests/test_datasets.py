"""Check dataset selection, rejection accounting, and row-level provenance."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.datasets import prepare_window_datasets
from scripts.preprocessing import TrainingStatistics, inverse_preprocess_window

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CenterFixture:
    """Original index, sample position, and symbol of a central annotation."""

    annotation_index: int
    sample: int
    symbol: str


@dataclass
class CandidateFixture:
    """Dataclass window used to check dataset rules without reading raw files."""

    signal: np.ndarray
    record_name: str
    annotation: CenterFixture
    left_padding: int
    right_padding: int
    other_annotations: tuple
    channel: int = 0
    channel_name: str = "MLII"
    unit: str = "mV"
    fs: float = 360

    @property
    def center_index(self):
        """Return the center's row position, in samples."""
        return 180

    @property
    def is_complete(self):
        """Return whether the candidate contains no artificial padding."""
        return self.left_padding == self.right_padding == 0

    @property
    def has_other_annotations(self):
        """Return whether another retained annotation is present."""
        return bool(self.other_annotations)


class DatasetTests(unittest.TestCase):
    """Exercise full and incomplete N/A candidates plus excluded + markers."""

    def setUp(self):
        """Load the real split manifest and define deterministic source fixtures."""
        self.splits = pd.read_csv(ROOT / "data/record_splits.csv", dtype=str)
        self.record_ids = (ROOT / "data/RECORDS").read_text().splitlines()
        self.symbols = ["+", "N", "A", "N", "A"]
        self.samples = [10, 500, 600, 950, 970]
        self.extracted_symbols = []
        self.statistics = TrainingStatistics(1.0, 1.0, 2.0, 361, 1, ("100",), 0, "mV")

    def source_record(self, path):
        """Return a single-channel source for the requested path.

        Args:
            path: WFDB path whose final component is the record ID.

        Returns:
            Source object with 1000 known physical samples.
        """
        return SimpleNamespace(record_name=Path(path).name, p_signal=np.arange(1000, dtype=float).reshape(-1, 1) / 1000)

    def source_annotations(self, path, extension):
        """Return the fixture annotation set for a requested WFDB path.

        Args:
            path: WFDB path, retained to identify the source recording.
            extension: Requested annotation extension.

        Returns:
            Annotation object with the configured original samples and symbols.
        """
        return SimpleNamespace(record_name=Path(path).name, sample=np.array(self.samples), symbol=list(self.symbols))

    def extract_candidate(self, record, annotations, annotation_index, channel):
        """Build a fixture candidate and record which central symbols were extracted.

        Args:
            record: Source object with physical signal values.
            annotations: Source symbols and sample positions.
            annotation_index: Original central entry index.
            channel: Selected zero-based channel.

        Returns:
            CandidateFixture with a centered 361-sample signal and source metadata.
        """
        symbol = annotations.symbol[annotation_index]
        self.assertNotEqual(symbol, "+", "Excluded markers must never reach extraction.")
        self.extracted_symbols.append(symbol)
        center = int(annotations.sample[annotation_index])
        first, last = center - 180, center + 181
        left, right = max(0, -first), max(0, last - 1000)
        values = np.zeros(361)
        values[left:361 - right] = record.p_signal[max(0, first):min(last, 1000), channel]
        neighbors = tuple(
            label for index, (sample, label) in enumerate(zip(annotations.sample, annotations.symbol))
            if index != annotation_index and first <= sample < last and label != "+"
        )
        return CandidateFixture(values, record.record_name, CenterFixture(annotation_index, center, symbol), left, right, neighbors)

    def prepare(self, statistics=None):
        """Prepare fixture datasets through the real selection and preprocessing code.

        Args:
            statistics: Optional replacement for the known training statistics.

        Returns:
            Datasets, per-symbol audit table, and rejection-reason counts.
        """
        with patch("scripts.datasets.wfdb.rdrecord", side_effect=self.source_record), patch(
            "scripts.datasets.wfdb.rdann", side_effect=self.source_annotations
        ):
            return prepare_window_datasets(
                Path("/unused"), self.splits, self.record_ids,
                statistics if statistics is not None else self.statistics,
                self.extract_candidate,
            )

    def test_reference_and_evaluation_selection(self):
        """Keep only full N in train, and full N/A in validation/test, including neighbors."""
        datasets, audit, reasons = self.prepare()
        self.assertEqual(datasets["train"].X.shape, (30, 361))
        self.assertEqual(datasets["validation"].X.shape, (18, 361))
        self.assertEqual(datasets["test"].X.shape, (18, 361))
        self.assertEqual(set(datasets["train"].metadata["annotation_symbol"]), {"N"})
        for split in ("validation", "test"):
            self.assertEqual(set(datasets[split].metadata["annotation_symbol"]), {"N", "A"})
        for dataset in datasets.values():
            self.assertTrue(dataset.metadata["is_complete"].all())
            self.assertTrue(dataset.metadata["has_other_annotations"].all())
        self.assertNotIn("+", self.extracted_symbols)
        self.assertEqual(audit.loc[("train", "+"), "Accepted windows"], 0)

    def test_rejection_reasons_account_for_every_entry_once(self):
        """Report combined failures once and reconcile rejected counts with the audit."""
        datasets, audit, reasons = self.prepare()
        rejected = reasons.groupby(["split", "symbol"])["count"].sum().reindex(audit.index, fill_value=0)
        pd.testing.assert_series_equal(rejected, audit["Rejected entries"], check_names=False)
        self.assertTrue((audit["Accepted windows"] + audit["Rejected entries"] == audit["Source entries"]).all())
        train_reasons = reasons.loc[(reasons["split"] == "train") & (reasons["symbol"] == "A")].set_index("reason")["count"].to_dict()
        self.assertEqual(train_reasons, {"train_center_not_N": 30, "train_center_not_N;incomplete_window": 30})

    def test_matrix_rows_recover_source_and_inverse_values(self):
        """Use row metadata to reconstruct the original source window and undo scaling."""
        datasets, _, _ = self.prepare()
        for split, dataset in datasets.items():
            np.testing.assert_array_equal(dataset.metadata["row_id"], np.arange(len(dataset.X)))
            self.assertEqual(len(dataset.states), len(dataset.X))
            for row in dataset.metadata.itertuples(index=False):
                original = self.extract_candidate(
                    self.source_record(row.record_id), self.source_annotations(row.record_id, "atr"),
                    row.annotation_index, row.channel,
                )
                processed = replace(original, signal=dataset.X[row.row_id], unit=row.unit)
                restored = inverse_preprocess_window(processed, dataset.states[row.row_id])
                np.testing.assert_allclose(restored.signal, original.signal, rtol=1e-14, atol=1e-14)
                self.assertEqual(row.annotation_sample, original.annotation.sample)
                self.assertEqual(row.window_stop_sample - row.window_start_sample, 361)
                self.assertEqual(self.splits.set_index("record_id").loc[row.record_id, "split"], split)

    def test_statistics_from_held_out_records_are_rejected(self):
        """Prevent a scaler fitted with validation data from entering dataset preparation."""
        with self.assertRaisesRegex(ValueError, "non-training"):
            self.prepare(replace(self.statistics, record_ids=("101",)))

    def test_statistics_from_incomplete_windows_are_rejected(self):
        """Reject fitted statistics that still include a shorter, padded window."""
        with self.assertRaisesRegex(ValueError, "complete 361-sample"):
            self.prepare(replace(self.statistics, sample_count=360))

    def test_empty_selected_split_has_a_two_dimensional_shape(self):
        """Retain shape (0, 361), empty metadata columns, and no inverse states."""
        self.symbols, self.samples = ["+", "A"], [10, 500]
        datasets, _, _ = self.prepare()
        self.assertEqual(datasets["train"].X.shape, (0, 361))
        self.assertIn("annotation_index", datasets["train"].metadata)
        self.assertEqual(datasets["train"].states, ())


if __name__ == "__main__":
    unittest.main()
