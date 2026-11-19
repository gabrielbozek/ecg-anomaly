"""Check that raw plots follow the same beat policy as extracted windows."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from scripts.plotting import plot_ecg_with_annotations


class PlottingTests(unittest.TestCase):
    """Verify labels and channel layouts with a known source annotation set."""

    def test_only_beat_symbols_are_shown_in_both_layouts(self):
        """Omit non-beat entries and keep original beat labels on each channel."""
        record = wfdb.Record(
            record_name="100", fs=360, n_sig=2, sig_len=1000,
            p_signal=np.zeros((1000, 2)), sig_name=["MLII", "V5"], units=["mV", "mV"],
        )
        annotations = wfdb.Annotation(
            record_name="100", extension="atr", fs=360,
            sample=np.array([100, 200, 250, 300, 400, 500, 600]),
            symbol=["+", "N", "|", "A", "!", "x", "?"],
        )
        for layout in ("overlay", "stacked"):
            with self.subTest(layout=layout):
                fig, axes = plot_ecg_with_annotations(record, annotations, channel=(0, 1), duration_s=2, layout=layout)
                try:
                    expected = ["N", "A", "?"] * (2 if layout == "overlay" else 1)
                    for ax in np.atleast_1d(axes):
                        self.assertEqual([label.get_text() for label in ax.texts], expected)
                finally:
                    plt.close(fig)
        self.assertEqual(annotations.symbol, ["+", "N", "|", "A", "!", "x", "?"])


if __name__ == "__main__":
    unittest.main()
