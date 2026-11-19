"""Check training-only statistics and reversible, padding-aware preprocessing."""

from dataclasses import dataclass, replace
import unittest

import numpy as np

from scripts.preprocessing import (
    fit_training_statistics,
    inverse_preprocess_window,
    preprocess_window,
)


@dataclass(frozen=True)
class AnnotationFixture:
    """Minimal immutable annotation with its original symbol."""

    symbol: str = "N"


@dataclass
class WindowFixture:
    """Window fields used by preprocessing, with extra metadata for preservation checks."""

    signal: np.ndarray
    record_name: str = "100"
    channel: int = 0
    unit: str = "mV"
    annotation: AnnotationFixture = AnnotationFixture()
    left_padding: int = 0
    right_padding: int = 0
    fs: float = 360
    channel_name: str = "MLII"
    other_annotations: tuple = ()


class PreprocessingTests(unittest.TestCase):
    """Exercise statistics, all transform modes, inversion, and data isolation."""

    def setUp(self):
        """Fit known training statistics with a real zero sample included."""
        self.training_window = WindowFixture(np.array([0.0, 1.0, 2.0, 3.0]))
        self.statistics = fit_training_statistics([self.training_window], ["100"])

    def test_statistics_exclude_padding_but_include_real_zeros(self):
        """Use padding lengths to distinguish artificial zeros from real amplitudes."""
        window = WindowFixture(np.array([0.0, 0.0, 0.0, 2.0, 4.0, 0.0]), left_padding=2, right_padding=1)
        stats = fit_training_statistics([window], ["100"])
        self.assertEqual(stats.sample_count, 3)
        self.assertEqual(stats.window_count, 1)
        self.assertEqual(stats.mean, 2.0)
        self.assertEqual(stats.median, 2.0)
        self.assertAlmostEqual(stats.std, np.sqrt(8 / 3))
        self.assertEqual(stats.record_ids, ("100",))

    def test_overlapping_sample_occurrences_are_pooled(self):
        """Count shared samples once for each window rather than deduplicating them."""
        windows = [WindowFixture(np.array([0.0, 1.0, 2.0])), WindowFixture(np.array([1.0, 2.0, 3.0]))]
        stats = fit_training_statistics(iter(windows), ["100"])
        expected = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
        self.assertEqual(stats.sample_count, 6)
        self.assertEqual(stats.window_count, 2)
        self.assertEqual(stats.mean, expected.mean())
        self.assertEqual(stats.median, np.median(expected))
        self.assertAlmostEqual(stats.std, expected.std(ddof=0))

    def test_none_preserves_values_dtype_and_independent_storage(self):
        """Verify no-op preprocessing changes neither values nor source storage."""
        original = replace(self.training_window, signal=self.training_window.signal.astype(np.float32))
        processed, state = preprocess_window(original)
        np.testing.assert_array_equal(processed.signal, original.signal)
        self.assertEqual(processed.signal.dtype, original.signal.dtype)
        self.assertFalse(np.shares_memory(processed.signal, original.signal))
        self.assertEqual(processed.unit, "mV")
        self.assertEqual((state.offset, state.scale), (0.0, 1.0))
        processed.signal[0] = 99
        self.assertEqual(original.signal[0], 0)

    def test_center_and_standardize_reuse_training_parameters(self):
        """Apply training parameters to a different window without fitting it again."""
        held_out = WindowFixture(np.array([8.0, 10.0, 12.0]), record_name="101")
        centered, _ = preprocess_window(held_out, self.statistics, mode="center")
        standardized, _ = preprocess_window(held_out, self.statistics, mode="standardize")
        np.testing.assert_allclose(centered.signal, [6.5, 8.5, 10.5])
        np.testing.assert_allclose(standardized.signal, np.array([6.5, 8.5, 10.5]) / np.sqrt(1.25))
        self.assertEqual(self.statistics.record_ids, ("100",))
        self.assertEqual(centered.unit, "mV")
        self.assertEqual(standardized.unit, "dimensionless")

    def test_all_modes_are_reversible_and_preserve_metadata_and_padding(self):
        """Round-trip padded windows while retaining annotations and physical units."""
        window = WindowFixture(
            np.array([0.0, 0.0, -2.0, 0.0, 5.0, 0.0]),
            left_padding=2, right_padding=1, other_annotations=("retained metadata",),
        )
        source = window.signal.copy()
        for mode in ("none", "center", "standardize"):
            with self.subTest(mode=mode):
                processed, state = preprocess_window(window, self.statistics, mode=mode)
                restored = inverse_preprocess_window(processed, state)
                np.testing.assert_allclose(restored.signal, source, rtol=1e-14, atol=1e-14)
                np.testing.assert_array_equal(processed.signal[:2], 0)
                np.testing.assert_array_equal(processed.signal[-1:], 0)
                self.assertEqual(restored.unit, "mV")
                self.assertEqual((restored.left_padding, restored.right_padding), (2, 1))
                self.assertIs(processed.annotation, window.annotation)
                self.assertIs(restored.other_annotations, window.other_annotations)
                self.assertEqual((restored.record_name, restored.fs, restored.channel_name), ("100", 360, "MLII"))
                self.assertFalse(np.shares_memory(restored.signal, processed.signal))
        np.testing.assert_array_equal(window.signal, source)

    def test_real_zero_is_transformed_while_padding_stays_zero(self):
        """Apply centering to real zero amplitudes rather than treating them as padding."""
        window = WindowFixture(np.array([0.0, 0.0, 2.0, 0.0]), left_padding=1, right_padding=1)
        processed, _ = preprocess_window(window, self.statistics, mode="center")
        np.testing.assert_allclose(processed.signal, [0.0, -1.5, 0.5, 0.0])

    def test_statistics_reject_nontraining_records_and_non_n_centers(self):
        """Prevent held-out records or other central labels from entering the fit."""
        for bad_window, message in (
            (replace(self.training_window, record_name="101"), "Non-training"),
            (replace(self.training_window, annotation=AnnotationFixture("A")), "N windows only"),
            (replace(self.training_window, annotation=AnnotationFixture("+")), "N windows only"),
        ):
            with self.subTest(window=bad_window.record_name, symbol=bad_window.annotation.symbol):
                with self.assertRaisesRegex(ValueError, message):
                    fit_training_statistics([bad_window], ["100"])

    def test_statistics_reject_mixed_channels_or_units(self):
        """Keep statistics from silently pooling different channels or physical units."""
        for bad_window in (replace(self.training_window, channel=1), replace(self.training_window, unit="uV")):
            with self.assertRaisesRegex(ValueError, "one channel"):
                fit_training_statistics([self.training_window, bad_window], ["100"])
        with self.assertRaisesRegex(ValueError, "must match"):
            preprocess_window(replace(self.training_window, channel=1), self.statistics, mode="center")

    def test_empty_fit_missing_statistics_and_unknown_mode_are_rejected(self):
        """Require a real fitted population and a recognized transformation mode."""
        with self.assertRaisesRegex(ValueError, "No training"):
            fit_training_statistics([], ["100"])
        for mode in ("center", "standardize"):
            with self.assertRaisesRegex(ValueError, "required"):
                preprocess_window(self.training_window, mode=mode)
        with self.assertRaisesRegex(ValueError, "Mode must"):
            preprocess_window(self.training_window, mode="unknown")

    def test_zero_std_is_rejected_only_when_dividing(self):
        """Reject undefined standardization without inventing an epsilon scale."""
        constant_stats = replace(self.statistics, std=0.0)
        preprocess_window(self.training_window, constant_stats, mode="center")
        with self.assertRaisesRegex(ValueError, "positive"):
            preprocess_window(self.training_window, constant_stats, mode="standardize")

    def test_invalid_padding_and_nonfinite_values_are_rejected(self):
        """Reject malformed windows instead of silently dropping or repairing values."""
        for bad_window in (
            replace(self.training_window, left_padding=4),
            replace(self.training_window, right_padding=1),
            replace(self.training_window, signal=np.array([0.0, np.nan])),
        ):
            with self.assertRaises(ValueError):
                fit_training_statistics([bad_window], ["100"])


if __name__ == "__main__":
    unittest.main()
