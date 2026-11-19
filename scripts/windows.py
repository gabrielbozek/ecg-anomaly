"""Extract ECG windows and retain their original source annotation metadata."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import wfdb

from scripts.rules import is_beat_annotation


@dataclass(frozen=True)
class WindowAnnotation:
    """An annotation located relative to a window's central annotation.

    Attributes:
        annotation_index: Index of this entry in the original annotation set.
        sample: Absolute sample index in the source record.
        symbol: Original WFDB annotation symbol.
        offset_samples: Signed sample offset from the central annotation.
        aux_note: Original auxiliary note, if available.
    """

    annotation_index: int
    sample: int
    symbol: str
    offset_samples: int
    aux_note: str | None = None


@dataclass
class ECGWindow:
    """One channel spanning 0.5 seconds on either side of an annotation.

    Both endpoints are included: at 360 Hz, signal has 361 samples and its
    central annotation is at index 180. Missing source samples are zeros.

    Attributes:
        signal: Independent 1D array of physical signal values, including padding.
        fs: Sampling rate in Hz.
        record_name: Identifier of the source recording.
        channel: Zero-based channel index in the source record.
        channel_name: Original channel name.
        unit: Physical unit of the signal values, for example mV.
        annotation: Central annotation; its offset_samples is always zero.
        other_annotations: All other beat annotations in the inclusive window
            interval, including entries with the same symbol or at the same sample.
        left_padding: Number of missing samples replaced by zeros on the left.
        right_padding: Number of missing samples replaced by zeros on the right.
    """

    signal: np.ndarray = field(repr=False)
    fs: float
    record_name: str
    channel: int
    channel_name: str
    unit: str
    annotation: WindowAnnotation
    other_annotations: tuple[WindowAnnotation, ...]
    left_padding: int
    right_padding: int

    @property
    def center_index(self) -> int:
        """Return the central annotation's sample index within signal."""
        return self.signal.size // 2

    @property
    def is_complete(self) -> bool:
        """Return True if every sample comes from the source, without padding."""
        return self.left_padding == 0 and self.right_padding == 0

    @property
    def has_other_annotations(self) -> bool:
        """Return True if another beat annotation is present."""
        return bool(self.other_annotations)


def extract_ecg_window(
    record: wfdb.Record,
    annotations: wfdb.Annotation,
    annotation_index: int,
    channel: int = 0,
) -> ECGWindow:
    """Extract a centered, single-channel window, padding missing samples with zeros.

    The inclusive interval is [annotation time - 0.5 s, annotation time + 0.5 s].
    The sampling rate must place both endpoints on integer sample indices.
    Annotation entries are distinguished by their original indices, so another
    beat at the central sample is retained among other_annotations. Only the
    PhysioNet Beat annotations group is accepted as centers or neighbors;
    all non-beat markers are ignored without renumbering source indices.

    Args:
        record: WFDB Record with physical signals loaded from sample zero.
        annotations: WFDB Annotation using the same record and sample origin.
        annotation_index: Zero-based index of the central annotation entry,
            not its sample position in the recording.
        channel: Zero-based channel to extract (default: 0).

    Returns:
        ECGWindow with a signal copy, original annotation metadata, and padding
        counts. Its is_complete flag is False whenever padding was needed.
    """
    signal = record.p_signal
    samples = annotations.sample
    symbols = annotations.symbol
    names = record.sig_name
    units = record.units
    fs = record.fs
    if signal is None or samples is None or symbols is None:
        raise ValueError("Physical signal values, annotation samples, and symbols are required.")
    if names is None or units is None or record.record_name is None:
        raise ValueError("Record name, channel names, and physical units are required.")
    if record.record_name != annotations.record_name:
        raise ValueError("The signal and annotations must belong to the same record.")
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError("The record must have a positive, finite sampling rate.")
    if annotations.fs is not None and annotations.fs != fs:
        raise ValueError("The signal and annotation sampling rates differ.")
    half_width = float(fs) * 0.5
    if not half_width.is_integer():
        raise ValueError("The sampling rate must represent 0.5 seconds as an integer sample count.")
    if signal.ndim != 2 or signal.shape[0] == 0:
        raise ValueError("Physical signals must be a nonempty samples-by-channels array.")
    if not isinstance(channel, (int, np.integer)) or not 0 <= channel < signal.shape[1]:
        raise ValueError(f"Channel must be an integer from 0 to {signal.shape[1] - 1}.")
    if len(names) != signal.shape[1] or len(units) != signal.shape[1]:
        raise ValueError("Channel metadata must match the signal columns.")
    if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.integer) or len(samples) != len(symbols):
        raise ValueError("Annotation samples must be a 1D integer array matching the symbols.")
    if np.any(samples < 0) or np.any(samples >= signal.shape[0]):
        raise ValueError("Annotation samples must lie within the source recording.")
    if not isinstance(annotation_index, (int, np.integer)) or not 0 <= annotation_index < len(samples):
        raise ValueError("Annotation index must identify an entry in the annotation set.")
    if not is_beat_annotation(symbols[annotation_index]):
        raise ValueError("The window center must be a PhysioNet beat annotation.")
    notes = annotations.aux_note
    if notes is not None and len(notes) != len(samples):
        raise ValueError("Auxiliary notes must match the annotation entries.")

    half_samples = int(half_width)
    center = int(samples[annotation_index])
    first = center - half_samples
    last = center + half_samples + 1
    left_padding = max(0, -first)
    right_padding = max(0, last - signal.shape[0])
    window_signal = np.zeros(2 * half_samples + 1, dtype=signal.dtype)
    window_signal[left_padding:window_signal.size - right_padding] = signal[
        max(0, first):min(signal.shape[0], last), channel
    ]

    inside = np.flatnonzero((samples >= first) & (samples < last))
    entries = tuple(
        WindowAnnotation(
            annotation_index=int(index),
            sample=int(samples[index]),
            symbol=symbols[index],
            offset_samples=int(samples[index]) - center,
            aux_note=notes[index] if notes is not None else None,
        )
        for index in inside if is_beat_annotation(symbols[index])
    )
    return ECGWindow(
        signal=window_signal,
        fs=float(fs),
        record_name=record.record_name,
        channel=int(channel),
        channel_name=names[channel],
        unit=units[channel],
        annotation=next(entry for entry in entries if entry.annotation_index == annotation_index),
        other_annotations=tuple(
            entry for entry in entries
            if entry.annotation_index != annotation_index
        ),
        left_padding=left_padding,
        right_padding=right_padding,
    )


def generate_ecg_windows(
    record: wfdb.Record,
    annotations: wfdb.Annotation,
    annotation_symbol: str = "N",
    channel: int = 0,
) -> Iterator[ECGWindow]:
    """Yield one window per matching annotation, in the original annotation order.

    Windows are created lazily. Only the center is selected by symbol; all
    other beat annotations inside each window are preserved. Non-beat markers
    are ignored and cannot be selected as window centers.

    Args:
        record: WFDB Record with physical signals loaded from sample zero.
        annotations: WFDB Annotation using the same record and sample origin.
        annotation_symbol: Exact PhysioNet beat symbol to use as a window center
            (default: "N"). No label mapping is applied.
        channel: Zero-based channel to extract for every window (default: 0).

    Yields:
        ECGWindow with both 0.5-second endpoints included, original metadata,
        and completeness and other-annotation flags.
    """
    if annotations.symbol is None:
        raise ValueError("Annotation symbols are required.")
    if not isinstance(annotation_symbol, str) or not annotation_symbol:
        raise ValueError("Select a nonempty annotation symbol.")
    if not is_beat_annotation(annotation_symbol):
        raise ValueError("The window center must be a PhysioNet beat annotation.")
    for index, symbol in enumerate(annotations.symbol):
        if symbol == annotation_symbol:
            yield extract_ecg_window(record, annotations, annotation_index=index, channel=channel)
