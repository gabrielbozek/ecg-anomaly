"""Plot ECG signals and static grids of annotation-centered windows."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from scripts.windows import ECGWindow
from scripts.rules import is_beat_annotation


def plot_ecg_with_annotations(
    record,
    annotations,
    channel: int | Sequence[int] = 0,
    start_s=0.0,
    duration_s=10.0,
    layout="overlay",
):
    """Plot one or two ECG channels with original `.atr` beat annotation symbols.

    The displayed interval is [start_s, start_s + duration_s). Each annotation
    is placed at its sample position on every selected channel, with its symbol
    to the right of the marker. A channel's signal, annotation markers, and
    annotation symbols share the same color. Non-beat markers are omitted.

    Args:
        record: WFDB Record with loaded physical signal values.
        annotations: WFDB Annotation for the same record.
        channel: Zero-based channel index or a sequence of one or two unique
            indices, such as (0, 1). Defaults to 0.
        start_s: Interval start in seconds (default: 0).
        duration_s: Interval length in seconds (default: 10).
        layout: "overlay" draws selected channels on one axis; "stacked" draws
            one channel per row with a shared time axis. Defaults to "overlay".
            Overlay requires all selected channels to use the same unit.

    Returns:
        Tuple of a Matplotlib figure and either one Axes object or an array of
        Axes objects when multiple channels are stacked.
    """
    if record.record_name != annotations.record_name:
        raise ValueError("The signal and annotations must belong to the same record.")
    if annotations.fs is not None and annotations.fs != record.fs:
        raise ValueError("The signal and annotation sampling rates differ.")
    if isinstance(channel, (int, np.integer)):
        channels = [channel]
    else:
        try:
            channels = list(channel)
        except TypeError as exc:
            raise ValueError("Channel must be an index or a sequence of indices.") from exc
    if not 1 <= len(channels) <= 2 or any(
        not isinstance(index, (int, np.integer)) or not 0 <= index < record.n_sig
        for index in channels
    ):
        raise ValueError(f"Select one or two channel indices from 0 to {record.n_sig - 1}.")
    if len(set(channels)) != len(channels):
        raise ValueError("Channel indices must be unique.")
    if layout not in ("overlay", "stacked"):
        raise ValueError('Layout must be "overlay" or "stacked".')
    if layout == "overlay" and len({record.units[index] for index in channels}) != 1:
        raise ValueError("Overlay requires matching channel units; use the stacked layout.")
    if not np.isfinite(start_s) or not np.isfinite(duration_s) or start_s < 0 or duration_s <= 0:
        raise ValueError("Use a nonnegative start and a positive, finite duration.")

    first = round(start_s * record.fs)
    last = round((start_s + duration_s) * record.fs)
    if first >= last or last > record.sig_len:
        raise ValueError(f"The interval must contain samples within 0–{record.sig_len / record.fs:.2f} s.")

    sample_indices = np.arange(first, last)
    annotation_mask = (annotations.sample >= first) & (annotations.sample < last)
    annotation_mask &= np.fromiter(
        (is_beat_annotation(symbol) for symbol in annotations.symbol), dtype=bool,
        count=len(annotations.symbol),
    )
    annotation_samples = annotations.sample[annotation_mask]
    annotation_symbols = np.asarray(annotations.symbol)[annotation_mask]

    rows = len(channels) if layout == "stacked" else 1
    fig, axes = plt.subplots(nrows=rows, sharex=True, figsize=(14, 4 * rows))
    plot_axes = np.atleast_1d(axes)
    for position, index in enumerate(channels):
        ax = plot_axes[position if layout == "stacked" else 0]
        signal = record.p_signal[:, index]
        line, = ax.plot(
            sample_indices / record.fs, signal[first:last], linewidth=0.8,
            color=f"C{position}", label=record.sig_name[index],
        )
        ax.scatter(
            annotation_samples / record.fs, signal[annotation_samples],
            color=line.get_color(), s=24, zorder=3,
        )
        for sample, symbol in zip(annotation_samples, annotation_symbols):
            ax.annotate(
                symbol, (sample / record.fs, signal[sample]),
                xytext=(7, 0), textcoords="offset points",
                ha="left", va="center", color=line.get_color(), fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )

    for position, ax in enumerate(plot_axes):
        axis_channels = [channels[position]] if layout == "stacked" else channels
        names = ", ".join(record.sig_name[index] for index in axis_channels)
        ax.set(ylabel=f"Amplitude ({record.units[axis_channels[0]]})",
               title=f"Record {record.record_name}: {names}")
        ax.set_xlim(first / record.fs, last / record.fs)
        ax.margins(y=0.2)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    plot_axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig, axes


def plot_ecg_windows_grid(
    windows: Sequence[ECGWindow],
    ncols: int = 4,
):
    """Plot the supplied ECG windows as one static grid.

    Time is relative to each central annotation, marked in red at 0 s.
    Other annotations are orange, and symbols appear to the right of markers.
    Gray shading identifies zero padding. Amplitude limits are shared across
    the supplied windows, which must all use the same physical unit.

    Args:
        windows: Nonempty list or sequence of ECGWindow objects in display
            order. Every supplied window is plotted; pass a small selection,
            such as windows[:20], to keep the figure readable.
        ncols: Maximum number of grid columns (default: 4).

    Returns:
        Tuple containing a Matplotlib figure and a 2D array of axes. Unused
        panels in the last row are hidden. Call plt.show() to display it.
    """
    if not len(windows):
        raise ValueError("Provide at least one ECG window.")
    if not isinstance(ncols, int) or ncols <= 0:
        raise ValueError("Column count must be a positive integer.")
    if len({window.unit for window in windows}) != 1:
        raise ValueError("Windows must use the same physical unit for shared amplitude limits.")

    columns = min(ncols, len(windows))
    rows = (len(windows) + columns - 1) // columns
    amplitude_min = min(float(window.signal.min()) for window in windows)
    amplitude_max = max(float(window.signal.max()) for window in windows)
    margin = 0.08 * (amplitude_max - amplitude_min) or 0.1
    fig, axes = plt.subplots(
        rows, columns, squeeze=False, sharex=True, sharey=True,
        figsize=(4 * columns, 2.6 * rows),
    )
    for position, window in enumerate(windows):
        ax = axes.flat[position]
        time_s = (np.arange(window.signal.size) - window.center_index) / window.fs
        if window.left_padding:
            edge = time_s[window.left_padding] - 0.5 / window.fs
            ax.axvspan(time_s[0], edge, facecolor="0.9", zorder=0)
        if window.right_padding:
            edge = time_s[window.signal.size - window.right_padding] - 0.5 / window.fs
            ax.axvspan(edge, time_s[-1], facecolor="0.9", zorder=0)
        ax.plot(time_s, window.signal, color="tab:blue", linewidth=0.8)
        ax.axvline(0, color="0.6", linestyle="--", linewidth=0.6, zorder=0)
        for entry in (window.annotation, *window.other_annotations):
            index = window.center_index + entry.offset_samples
            color = "tab:red" if entry.annotation_index == window.annotation.annotation_index else "tab:orange"
            ax.scatter(time_s[index], window.signal[index], color=color, s=15, zorder=3)
            ax.annotate(
                entry.symbol, (time_s[index], window.signal[index]),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                color=color, fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )
        status = " [padded]" if not window.is_complete else ""
        ax.set_title(
            f"#{position + 1} | {window.record_name}:{window.channel_name} | "
            f"{window.annotation.symbol} @ {window.annotation.sample / window.fs:.3f}s{status}",
            fontsize=8,
        )
        ax.set_xlim(time_s[0], time_s[-1])
        ax.set_ylim(amplitude_min - margin, amplitude_max + margin)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel("Time from center (s)", fontsize=8)
        if position % columns == 0:
            ax.set_ylabel(f"Amplitude ({window.unit})", fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=True)
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(windows):]:
        ax.set_axis_off()
    fig.suptitle(f"ECG windows | {len(windows)} selected", fontsize=12)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="tab:red", label="Central annotation"),
            Line2D([], [], marker="o", linestyle="none", color="tab:orange", label="Other annotations"),
            Patch(facecolor="0.9", label="Zero padding"),
        ],
        loc="lower center", ncol=3, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    return fig, axes
