"""Check centered extraction, source indices, and inclusive window boundaries."""

import unittest

import numpy as np
import wfdb

from scripts.windows import extract_ecg_window, generate_ecg_windows
from scripts.rules import BEAT_ANNOTATION_SYMBOLS, is_beat_annotation


class WindowTests(unittest.TestCase):
    """Exercise the shared extractor used by the notebook and dataset builder."""

    def setUp(self):
        """Create a two-channel, 360 Hz source with known sample values."""
        self.record = wfdb.Record(
            record_name="100", fs=360, n_sig=2, sig_len=1000,
            p_signal=np.arange(2000, dtype=float).reshape(1000, 2) / 1000,
            sig_name=["MLII", "V5"], units=["mV", "mV"],
        )

    def annotations(self, samples, symbols):
        """Build source annotations while preserving the supplied entry order.

        Args:
            samples: Absolute sample indices in the source record.
            symbols: WFDB symbol for each sample entry.

        Returns:
            WFDB Annotation with distinguishable auxiliary notes.
        """
        return wfdb.Annotation(
            record_name="100", extension="atr", fs=360,
            sample=np.array(samples, dtype=int), symbol=list(symbols),
            aux_note=[f"source entry {index}" for index in range(len(samples))],
        )

    def test_inclusive_bounds_and_original_annotation_indices(self):
        """Retain endpoints and coincident N entries without renumbering them."""
        annotations = self.annotations(
            [0, 319, 320, 500, 500, 680, 681, 999],
            ["N", "N", "A", "N", "N", "V", "N", "N"],
        )
        window = extract_ecg_window(self.record, annotations, 3, channel=1)
        np.testing.assert_array_equal(window.signal, self.record.p_signal[320:681, 1])
        self.assertEqual(window.signal.shape, (361,))
        self.assertEqual(window.center_index, 180)
        self.assertEqual(window.annotation.annotation_index, 3)
        self.assertEqual(window.annotation.aux_note, "source entry 3")
        self.assertEqual(window.channel_name, "V5")
        self.assertTrue(window.is_complete)
        self.assertTrue(window.has_other_annotations)
        self.assertEqual(
            [(entry.annotation_index, entry.symbol, entry.offset_samples) for entry in window.other_annotations],
            [(2, "A", -180), (4, "N", 0), (5, "V", 180)],
        )
        self.assertFalse(np.shares_memory(window.signal, self.record.p_signal))

    def test_record_boundaries_preserve_center_and_identify_padding(self):
        """Pad missing samples without shifting the central annotation."""
        annotations = self.annotations([0, 999], ["N", "N"])
        for index, expected_left, expected_right in ((0, 180, 0), (1, 0, 180)):
            with self.subTest(index=index):
                window = extract_ecg_window(self.record, annotations, index)
                self.assertEqual((window.left_padding, window.right_padding), (expected_left, expected_right))
                self.assertEqual(window.signal.shape, (361,))
                self.assertFalse(window.is_complete)
                self.assertFalse(window.has_other_annotations)
                self.assertEqual(window.signal[180], self.record.p_signal[annotations.sample[index], 0])
                if expected_left:
                    np.testing.assert_array_equal(window.signal[:180], np.zeros(180))
                    np.testing.assert_array_equal(window.signal[180:], self.record.p_signal[:181, 0])
                else:
                    np.testing.assert_array_equal(window.signal[:181], self.record.p_signal[-181:, 0])
                    np.testing.assert_array_equal(window.signal[181:], np.zeros(180))

    def test_generator_defaults_to_n_and_keeps_source_order(self):
        """Generate N candidates, including padded ones, without a quality filter."""
        annotations = self.annotations([0, 200, 500, 999], ["N", "A", "N", "N"])
        windows = list(generate_ecg_windows(self.record, annotations, channel=1))
        self.assertEqual([window.annotation.annotation_index for window in windows], [0, 2, 3])
        self.assertEqual([window.channel for window in windows], [1, 1, 1])
        self.assertEqual([window.is_complete for window in windows], [False, True, False])

    def test_plus_marker_is_ignored_without_changing_source_indices(self):
        """Exclude + neighbors and reject + as a center through both APIs."""
        annotations = self.annotations([490, 500, 510], ["+", "N", "+"])
        window = extract_ecg_window(self.record, annotations, 1)
        self.assertEqual(window.annotation.annotation_index, 1)
        self.assertEqual(window.other_annotations, ())
        with self.assertRaises(ValueError):
            extract_ecg_window(self.record, annotations, 0)
        with self.assertRaises(ValueError):
            list(generate_ecg_windows(self.record, annotations, annotation_symbol="+"))

    def test_approved_physionet_beat_symbols_are_supported(self):
        """Accept exactly the 19 symbols in the approved Beat annotations table."""
        expected = set("NLRBAaJSVrFejnE/fQ?")
        self.assertEqual(BEAT_ANNOTATION_SYMBOLS, expected)
        for symbol in sorted(expected):
            with self.subTest(symbol=symbol):
                self.assertTrue(is_beat_annotation(symbol))
                annotations = self.annotations([500], [symbol])
                windows = list(generate_ecg_windows(self.record, annotations, annotation_symbol=symbol))
                self.assertEqual(len(windows), 1)
                self.assertEqual(windows[0].annotation.symbol, symbol)

    def test_all_nonbeat_and_unknown_symbols_are_ignored_and_rejected(self):
        """Non-beat markers neither dirty an N window nor form their own window."""
        excluded = list("[!]x()ptu`'^|~+sT*D=\"@") + ["unknown", ""]
        annotations = self.annotations([500] * (len(excluded) + 1), [*excluded, "N"])
        window = extract_ecg_window(self.record, annotations, len(excluded))
        self.assertEqual(window.other_annotations, ())
        self.assertFalse(window.has_other_annotations)
        self.assertEqual(window.annotation.annotation_index, len(excluded))
        for index, symbol in enumerate(excluded):
            with self.subTest(symbol=symbol):
                self.assertFalse(is_beat_annotation(symbol))
                with self.assertRaises(ValueError):
                    extract_ecg_window(self.record, annotations, index)
                with self.assertRaises(ValueError):
                    list(generate_ecg_windows(self.record, annotations, annotation_symbol=symbol))

    def test_unrepresentable_half_second_is_rejected(self):
        """Reject a sampling rate whose half-second endpoints fall between samples."""
        annotations = self.annotations([500], ["N"])
        self.record.fs = annotations.fs = 359
        with self.assertRaisesRegex(ValueError, "integer sample count"):
            extract_ecg_window(self.record, annotations, 0)


if __name__ == "__main__":
    unittest.main()
