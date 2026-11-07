"""Classification metrics without numpy: confusion matrix, precision, recall, F1, bootstrap CI."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ClassMetrics",
    "ConfusionMatrix",
    "Interval",
    "MetricsError",
    "Report",
    "__version__",
    "bootstrap_ci",
    "confusion_matrix",
    "report",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorCode = Literal[
    "length_mismatch",
    "empty_input",
    "unknown_label",
    "unknown_metric",
    "invalid_parameter",
    "too_many_samples",
]

DEFAULT_MAX_SAMPLES: int = 1_000_000
MAX_BOOTSTRAP_RESAMPLES: int = 100_000


class MetricsError(ValueError):
    """Base exception raised by classification metrics with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        # Ensure error message does not leak excessive input text
        truncated_msg = message[:200] if len(message) > 200 else message
        super().__init__(truncated_msg)
        self.code: ErrorCode = code


@dataclass(frozen=True)
class ConfusionMatrix:
    """Contingency table for single-label multiclass classifications.

    Rows are indexed by the true label, and columns are indexed by predicted label.
    """

    labels: tuple[Any, ...]
    rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ClassMetrics:
    """Per-class classification metrics."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Report:
    """Summary classification report across all evaluated labels."""

    accuracy: float
    per_class: dict[Any, ClassMetrics]
    macro_f1: float
    micro_f1: float
    undefined: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class Interval:
    """Confidence interval containing point estimate and bounds."""

    point: float
    lower: float
    upper: float


def _validate_inputs(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> None:
    n_true = len(y_true)
    n_pred = len(y_pred)
    if n_true == 0:
        raise MetricsError("empty_input", "Input sequences must not be empty")
    if n_true != n_pred:
        raise MetricsError(
            "length_mismatch",
            f"Input length mismatch: y_true has {n_true} samples, y_pred has {n_pred} samples",
        )
    if n_true > max_samples:
        raise MetricsError(
            "too_many_samples",
            f"Number of samples ({n_true}) exceeds max_samples limit ({max_samples})",
        )


def _resolve_labels(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> tuple[Any, ...]:
    if labels is not None:
        labels_tuple = tuple(labels)
        # Check for duplicates in explicit labels
        if len(set(labels_tuple)) != len(labels_tuple):
            raise MetricsError("invalid_parameter", "Labels sequence must not contain duplicates")
        allowed_labels = set(labels_tuple)
        for val in y_true:
            if val not in allowed_labels:
                raise MetricsError("unknown_label", f"Encountered unknown label {val!r} in y_true")
        for val in y_pred:
            if val not in allowed_labels:
                raise MetricsError("unknown_label", f"Encountered unknown label {val!r} in y_pred")
        return labels_tuple

    distinct = set(y_true) | set(y_pred)
    try:
        sorted_labels = sorted(distinct)  # type: ignore[type-var]
    except TypeError as err:
        raise MetricsError(
            "invalid_parameter",
            "Labels are not mutually orderable; explicit labels tuple must be provided",
        ) from err
    return tuple(sorted_labels)


def confusion_matrix(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> ConfusionMatrix:
    """Compute confusion matrix to evaluate classification accuracy.

    Rows correspond to the true label and columns to the predicted label.
    """
    _validate_inputs(y_true, y_pred, max_samples=max_samples)
    label_tuple = _resolve_labels(y_true, y_pred, labels)
    label_to_idx = {label: i for i, label in enumerate(label_tuple)}
    n_labels = len(label_tuple)

    counts = [[0] * n_labels for _ in range(n_labels)]
    for yt, yp in zip(y_true, y_pred, strict=True):
        r = label_to_idx[yt]
        c = label_to_idx[yp]
        counts[r][c] += 1

    rows = tuple(tuple(row) for row in counts)
    return ConfusionMatrix(labels=label_tuple, rows=rows)


def report(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> Report:
    """Build a summary report of precision, recall, F1, accuracy, and undefined counts.

    Undefined divisions (0/0) evaluate to 0.0 and are recorded in `undefined`
    in label order, precision before recall for each label.
    """
    _validate_inputs(y_true, y_pred, max_samples=max_samples)
    label_tuple = _resolve_labels(y_true, y_pred, labels)

    n = len(y_true)
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    # Count true positives per label
    tp_counts: Counter[Any] = Counter()
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == yp:
            tp_counts[yt] += 1

    per_class: dict[Any, ClassMetrics] = {}
    undefined: list[tuple[str, Any]] = []
    f1_list: list[float] = []
    total_tp = 0

    for lbl in label_tuple:
        tp = tp_counts[lbl]
        support = true_counts[lbl]
        pred_c = pred_counts[lbl]
        total_tp += tp

        # Precision = tp / (tp + fp) = tp / pred_counts[lbl]
        if pred_c == 0:
            precision = 0.0
            undefined.append(("precision", lbl))
        else:
            precision = tp / pred_c

        # Recall = tp / (tp + fn) = tp / true_counts[lbl]
        if support == 0:
            recall = 0.0
            undefined.append(("recall", lbl))
        else:
            recall = tp / support

        # F1 = 2pr / (p + r)
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)

        per_class[lbl] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
        f1_list.append(f1)

    accuracy = total_tp / n
    micro_f1 = accuracy  # In single-label multiclass, total TP / n is micro F1
    macro_f1 = sum(f1_list) / len(label_tuple) if label_tuple else 0.0

    return Report(
        accuracy=accuracy,
        per_class=per_class,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        undefined=tuple(undefined),
    )


def _calculate_metric(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    metric: Literal["accuracy", "macro_f1", "micro_f1"],
    labels: Sequence[Hashable] | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> float:
    if metric == "accuracy" or metric == "micro_f1":
        n = len(y_true)
        if n == 0:
            return 0.0
        correct = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == yp)
        return correct / n
    elif metric == "macro_f1":
        rep = report(y_true, y_pred, labels=labels, max_samples=max_samples)
        return rep.macro_f1
    else:
        raise MetricsError("unknown_metric", f"Unknown metric: {metric!r}")


def bootstrap_ci(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    metric: Literal["accuracy", "macro_f1", "micro_f1"],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    labels: Sequence[Hashable] | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    _resample: Callable[[int, int, random.Random], list[list[int]]] | None = None,
) -> Interval:
    """Compute seeded percentile bootstrap confidence interval for a classification metric.

    The percentile interval method: alpha = 1 - confidence, lower is s[floor(alpha / 2 * B)],
    and upper is s[ceil((1 - alpha / 2) * B) - 1], where B is n_resamples.
    """
    _validate_inputs(y_true, y_pred, max_samples=max_samples)
    if metric not in ("accuracy", "macro_f1", "micro_f1"):
        raise MetricsError("unknown_metric", f"Unknown metric: {metric!r}")

    if n_resamples <= 0 or n_resamples > MAX_BOOTSTRAP_RESAMPLES:
        raise MetricsError(
            "invalid_parameter",
            f"n_resamples must be between 1 and {MAX_BOOTSTRAP_RESAMPLES}, got {n_resamples}",
        )
    if not (0.0 < confidence < 1.0):
        raise MetricsError(
            "invalid_parameter",
            f"confidence must be strictly between 0.0 and 1.0, got {confidence}",
        )

    # Lock the label space to full data or given labels
    resolved_labels = _resolve_labels(y_true, y_pred, labels)

    point = _calculate_metric(
        y_true, y_pred, metric, labels=resolved_labels, max_samples=max_samples
    )

    n = len(y_true)
    rng = random.Random(seed)  # noqa: S311

    stats: list[float] = []
    if _resample is not None:
        # _resample generates index lists for testing or custom resampling schemes
        resampled_indices = _resample(n, n_resamples, rng)
        for idxs in resampled_indices:
            boot_true = [y_true[i] for i in idxs]
            boot_pred = [y_pred[i] for i in idxs]
            val = _calculate_metric(
                boot_true,
                boot_pred,
                metric,
                labels=resolved_labels,
                max_samples=max_samples,
            )
            stats.append(val)
    else:
        for _ in range(n_resamples):
            indices = [rng.randrange(n) for _ in range(n)]
            boot_true = [y_true[i] for i in indices]
            boot_pred = [y_pred[i] for i in indices]
            val = _calculate_metric(
                boot_true,
                boot_pred,
                metric,
                labels=resolved_labels,
                max_samples=max_samples,
            )
            stats.append(val)

    stats.sort()
    b = len(stats)
    # Use round(..., 9) to avoid float issues like (1 - 0.8) == 0.19999999999999996
    alpha = round(1.0 - confidence, 9)
    lower_idx = math.floor((alpha / 2.0) * b)
    upper_idx = math.ceil((1.0 - alpha / 2.0) * b) - 1

    # Clamp index boundaries safely
    lower_idx = max(0, min(lower_idx, b - 1))
    upper_idx = max(0, min(upper_idx, b - 1))

    lower = stats[lower_idx]
    upper = stats[upper_idx]

    return Interval(point=point, lower=lower, upper=upper)
