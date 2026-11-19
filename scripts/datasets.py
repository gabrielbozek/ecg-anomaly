"""Prepare complete ECG window matrices with selection audits and source metadata."""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scripts.create_splits import validate_record_splits
from scripts.preprocessing import PreprocessingMode, PreprocessingState, TrainingStatistics, preprocess_window

SPLIT_ORDER = ("train", "validation", "test")
WINDOW_SIZE = 361
METADATA_COLUMNS = [
    "row_id", "record_id", "subject_id", "channel", "channel_name",
    "annotation_index", "annotation_sample", "annotation_symbol", "fs",
    "window_start_sample", "window_stop_sample", "is_complete",
    "has_other_annotations", "left_padding", "right_padding", "original_unit", "unit",
]


@dataclass
class PreparedSplit:
    """A processed matrix and the provenance of its rows.

    Attributes:
        X: Array shaped (accepted windows, 361), with one channel per row.
        metadata: Source metadata in exactly the same row order as X.
        states: Per-row preprocessing states for inverse transformation.
    """

    X: np.ndarray
    metadata: pd.DataFrame
    states: tuple[PreprocessingState, ...]


def prepare_window_datasets(
    raw_directory: Path,
    record_splits: pd.DataFrame,
    source_record_ids: Sequence[str],
    statistics: TrainingStatistics,
    extract_window: Callable,
    mode: PreprocessingMode = "standardize",
) -> tuple[dict[str, PreparedSplit], pd.DataFrame, pd.DataFrame]:
    """Build channel-0 datasets using the fixed reference-window selection rules.

    Exclude + markers before constructing windows. Keep only central N in
    train, and every original non-+ central symbol in validation/test. Reject
    incomplete windows in every split. Full windows with other annotations
    remain eligible. Reuse the supplied training statistics without fitting.

    Args:
        raw_directory: Directory containing the source WFDB recordings.
        record_splits: Validated record_id/subject_id/split manifest.
        source_record_ids: Complete list of 48 original record IDs.
        statistics: Statistics fitted only on complete training N windows.
        extract_window: The notebook's extract_ecg_window callable, accepting
            a record, annotations, annotation_index, and channel.
        mode: Preprocessing mode applied to eligible windows (default: standardize).

    Returns:
        Tuple of datasets indexed by split, a per-symbol input/accepted/rejected
        count table, and counts by exact rejection reason. Simultaneous reasons
        are joined with ';' and counted once, so no reason priority is implied.
        Source entries include + markers for audit purposes; they never produce
        windows. Metadata stop positions are exclusive source sample indices.
    """
    validate_record_splits(record_splits, source_record_ids)
    training_ids = set(record_splits.loc[record_splits["split"] == "train", "record_id"])
    if not set(statistics.record_ids).issubset(training_ids):
        raise ValueError("Dataset preprocessing statistics contain non-training records.")
    if statistics.channel != 0:
        raise ValueError("Dataset preparation uses channel 0.")
    if statistics.sample_count != statistics.window_count * WINDOW_SIZE:
        raise ValueError("Dataset preprocessing statistics must use complete 361-sample windows.")
    matrix_rows = {split: [] for split in SPLIT_ORDER}
    metadata_rows = {split: [] for split in SPLIT_ORDER}
    state_rows = {split: [] for split in SPLIT_ORDER}
    input_counts = Counter()
    accepted_counts = Counter()
    rejected_reasons = Counter()
    for assignment in record_splits.itertuples(index=False):
        source = Path(raw_directory) / assignment.record_id
        record = wfdb.rdrecord(str(source))
        annotations = wfdb.rdann(str(source), "atr")
        for index, symbol in enumerate(annotations.symbol):
            key = (assignment.split, symbol)
            input_counts[key] += 1
            if symbol == "+":
                rejected_reasons[(*key, "excluded_plus_marker")] += 1
                continue
            window = extract_window(record, annotations, annotation_index=index, channel=0)
            if (
                window.record_name != assignment.record_id
                or window.annotation.annotation_index != index
                or window.annotation.symbol != symbol
                or window.annotation.sample != int(annotations.sample[index])
            ):
                raise ValueError("Extracted window provenance does not match the source annotation.")
            reasons = []
            if assignment.split == "train" and symbol != "N":
                reasons.append("train_center_not_N")
            if not window.is_complete:
                reasons.append("incomplete_window")
            if reasons:
                rejected_reasons[(*key, ";".join(reasons))] += 1
                continue
            processed, state = preprocess_window(window, statistics, mode=mode)
            if processed.signal.shape != (WINDOW_SIZE,):
                raise ValueError("Expected 361 samples per accepted window.")
            row_id = len(matrix_rows[assignment.split])
            matrix_rows[assignment.split].append(processed.signal)
            state_rows[assignment.split].append(state)
            metadata_rows[assignment.split].append({
                "row_id": row_id,
                "record_id": assignment.record_id,
                "subject_id": assignment.subject_id,
                "channel": window.channel,
                "channel_name": window.channel_name,
                "annotation_index": index,
                "annotation_sample": window.annotation.sample,
                "annotation_symbol": symbol,
                "fs": window.fs,
                "window_start_sample": window.annotation.sample - window.center_index,
                "window_stop_sample": window.annotation.sample - window.center_index + WINDOW_SIZE,
                "is_complete": window.is_complete,
                "has_other_annotations": window.has_other_annotations,
                "left_padding": window.left_padding,
                "right_padding": window.right_padding,
                "original_unit": window.unit,
                "unit": processed.unit,
            })
            accepted_counts[key] += 1

    datasets = {}
    for split in SPLIT_ORDER:
        X = np.stack(matrix_rows[split]) if matrix_rows[split] else np.empty((0, WINDOW_SIZE), dtype=np.float64)
        datasets[split] = PreparedSplit(
            X=X,
            metadata=pd.DataFrame(metadata_rows[split], columns=METADATA_COLUMNS),
            states=tuple(state_rows[split]),
        )
    symbols = sorted({symbol for split, symbol in input_counts})
    audit = pd.DataFrame([
        {
            "split": split, "symbol": symbol,
            "Source entries": input_counts[(split, symbol)],
            "Accepted windows": accepted_counts[(split, symbol)],
            "Rejected entries": input_counts[(split, symbol)] - accepted_counts[(split, symbol)],
            "Result shape": (accepted_counts[(split, symbol)], WINDOW_SIZE),
        }
        for split in SPLIT_ORDER for symbol in symbols
    ]).set_index(["split", "symbol"])
    reasons = pd.DataFrame([
        {"split": split, "symbol": symbol, "reason": reason, "count": count}
        for (split, symbol, reason), count in sorted(rejected_reasons.items())
    ], columns=["split", "symbol", "reason", "count"])
    return datasets, audit, reasons
