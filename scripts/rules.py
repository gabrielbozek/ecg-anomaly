"""Shared annotation and quality rules for training and evaluation windows."""

# Exactly the "Beat annotations" group in the approved PhysioNet table:
# https://physionet.org/static/lightwave/doc/annotations.html
BEAT_ANNOTATION_SYMBOLS = frozenset({
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
    "n", "E", "/", "f", "Q", "?",
})

SPLIT_ORDER = ("train", "validation", "test")


def is_beat_annotation(symbol: str) -> bool:
    """Check whether a symbol belongs to PhysioNet's Beat annotations group.

    Args:
        symbol: Original, case-sensitive WFDB annotation symbol.

    Returns:
        True for one of the 19 approved beat symbols; False for non-beat
        markers and unknown symbols. The same rule applies to centers and neighbors.
    """
    return symbol in BEAT_ANNOTATION_SYMBOLS


def window_rejection_reasons(window, split: str) -> tuple[str, ...]:
    """Return all failed dataset-selection rules for an extracted candidate.

    Args:
        window: Candidate ECGWindow with an eligible central annotation.
        split: train, validation, or test.

    Returns:
        Tuple of rejection reasons. Training requires a central N, a complete
        signal, and no additional annotations. Evaluation requires completeness.
        An empty tuple means the candidate is eligible. Reasons may overlap.
    """
    if split not in SPLIT_ORDER:
        raise ValueError("Split must be train, validation, or test.")
    reasons = []
    if split == "train" and window.annotation.symbol != "N":
        reasons.append("train_center_not_N")
    if not window.is_complete:
        reasons.append("incomplete_window")
    if split == "train" and window.has_other_annotations:
        reasons.append("train_other_annotations")
    return tuple(reasons)
