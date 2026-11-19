"""Fit clean N-window statistics on training data and apply reversible preprocessing."""

from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from scripts.rules import window_rejection_reasons

PreprocessingMode = Literal["none", "center", "standardize"]


@dataclass(frozen=True)
class TrainingStatistics:
    """Pooled statistics of complete, clean training N windows.

    Attributes:
        mean: Mean amplitude in the original signal unit.
        median: Median amplitude in the original signal unit.
        std: Population standard deviation, with ddof=0, in the original unit.
        sample_count: Number of real sample occurrences, including window overlap.
        window_count: Number of N windows contributing samples.
        record_ids: Source records that contributed N windows.
        channel: Selected zero-based channel index.
        unit: Original physical signal unit, such as mV.
    """

    mean: float
    median: float
    std: float
    sample_count: int
    window_count: int
    record_ids: tuple[str, ...]
    channel: int
    unit: str


@dataclass(frozen=True)
class PreprocessingState:
    """Parameters needed to invert one preprocessing operation.

    Attributes:
        mode: Applied mode: none, center, or standardize.
        offset: Subtracted amplitude in the original signal unit.
        scale: Divisor used after centering; one for none or center.
        original_unit: Unit to restore when inverting the operation.
        channel: Channel index of the processed window.
    """

    mode: PreprocessingMode
    offset: float
    scale: float
    original_unit: str
    channel: int


def _signal_and_real_slice(window) -> tuple[np.ndarray, slice]:
    """Validate a window and locate its real, unpadded samples.

    Args:
        window: ECGWindow with a 1D signal and left/right zero-padding counts.

    Returns:
        Tuple containing the signal array and the slice of real samples.
    """
    signal = np.asarray(window.signal)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("Window signal must be a nonempty, finite 1D array.")
    left, right = window.left_padding, window.right_padding
    if any(not isinstance(count, (int, np.integer)) or count < 0 for count in (left, right)):
        raise ValueError("Padding counts must be nonnegative integers.")
    stop = signal.size - right
    if left >= stop:
        raise ValueError("A window must contain at least one real sample.")
    if np.any(signal[:left] != 0) or np.any(signal[stop:] != 0):
        raise ValueError("Samples marked as padding must be zero.")
    return signal, slice(left, stop)


def fit_training_statistics(windows: Iterable, training_record_ids: Collection[str]) -> TrainingStatistics:
    """Compute mean, median, and population std from complete, clean training N.

    Incomplete windows and windows with additional annotations are rejected.
    Real zero amplitudes remain included. Samples shared by overlapping windows
    are counted once per window. Validation/test windows must not enter this fit.

    Args:
        windows: Complete, clean ECGWindow objects centered on N, from one channel.
        training_record_ids: Record IDs selected from the validated split
            manifest where split is train. Any other source record is rejected.

    Returns:
        TrainingStatistics with amplitude statistics, counts, and source metadata.
    """
    allowed_records = set(training_record_ids)
    if not allowed_records:
        raise ValueError("Provide training record IDs from the split manifest.")
    chunks = []
    seen_records = set()
    channel = None
    unit = None
    window_count = 0
    for window in windows:
        if window.record_name not in allowed_records:
            raise ValueError(f"Non-training record in statistics fit: {window.record_name}")
        if window.annotation.symbol != "N":
            raise ValueError("Training statistics must be fitted on N windows only.")
        if window_rejection_reasons(window, "train"):
            raise ValueError("Training statistics require complete, clean N windows.")
        if channel is None:
            channel, unit = window.channel, window.unit
        if window.channel != channel or window.unit != unit:
            raise ValueError("Statistics require one channel index and one physical unit.")
        signal, real = _signal_and_real_slice(window)
        chunks.append(np.array(signal[real], dtype=np.float64, copy=True))
        seen_records.add(window.record_name)
        window_count += 1
    if not chunks:
        raise ValueError("No training N windows were supplied.")
    values = np.concatenate(chunks)
    return TrainingStatistics(
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=float(values.std(ddof=0)),
        sample_count=int(values.size),
        window_count=window_count,
        record_ids=tuple(sorted(seen_records)),
        channel=int(channel),
        unit=unit,
    )


def preprocess_window(
    window,
    statistics: TrainingStatistics | None = None,
    mode: PreprocessingMode = "none",
):
    """Copy a window and optionally center or standardize its real samples.

    Modes are none (unchanged), center (x - training mean), and standardize
    ((x - training mean) / training population std). Padding remains zero.
    Statistics are reused for every window; no fitting occurs in this function.

    Args:
        window: ECGWindow to transform, from train, validation, or test.
        statistics: Fitted TrainingStatistics; required for center/standardize.
        mode: none, center, or standardize (default: none).

    Returns:
        Tuple of a new ECGWindow and its PreprocessingState for inversion.
        Standardized values have unit dimensionless; other modes retain the
        original unit. Source signal and annotation metadata are preserved.
    """
    if mode not in ("none", "center", "standardize"):
        raise ValueError("Mode must be none, center, or standardize.")
    signal, real = _signal_and_real_slice(window)
    offset, scale = 0.0, 1.0
    if mode != "none":
        if statistics is None:
            raise ValueError("Fitted training statistics are required.")
        if window.unit != statistics.unit or window.channel != statistics.channel:
            raise ValueError("Window channel and unit must match the fitted statistics.")
        offset = statistics.mean
        scale = statistics.std if mode == "standardize" else 1.0
        if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("The offset must be finite and the scale finite and positive.")
    transformed = signal.copy() if mode == "none" else signal.astype(np.float64, copy=True)
    if mode != "none":
        transformed[real] = (signal[real] - offset) / scale
    output_unit = "dimensionless" if mode == "standardize" else window.unit
    state = PreprocessingState(mode, offset, scale, window.unit, window.channel)
    return replace(window, signal=transformed, unit=output_unit), state


def inverse_preprocess_window(window, state: PreprocessingState):
    """Undo preprocessing on real samples and restore the original signal unit.

    Args:
        window: Preprocessed ECGWindow with unchanged padding metadata.
        state: PreprocessingState returned by the corresponding forward call.

    Returns:
        New ECGWindow with restored physical values and unit, up to floating
        point rounding. Padding remains zero; the input window is not mutated.
    """
    if state.mode not in ("none", "center", "standardize"):
        raise ValueError("Unknown preprocessing mode in the inverse state.")
    expected_unit = "dimensionless" if state.mode == "standardize" else state.original_unit
    if window.unit != expected_unit or window.channel != state.channel:
        raise ValueError("Window channel and unit do not match the inverse state.")
    if not np.isfinite(state.offset) or not np.isfinite(state.scale) or state.scale <= 0:
        raise ValueError("The inverse state must have a finite offset and a positive finite scale.")
    signal, real = _signal_and_real_slice(window)
    restored = signal.copy() if state.mode == "none" else signal.astype(np.float64, copy=True)
    if state.mode != "none":
        restored[real] = signal[real] * state.scale + state.offset
    return replace(window, signal=restored, unit=state.original_unit)
