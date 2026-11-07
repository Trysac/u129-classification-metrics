"""Tests for classification_metrics covering all acceptance requirements and edge cases."""

from __future__ import annotations

import math
import random

import pytest

from classification_metrics import (
    ClassMetrics,
    Interval,
    MetricsError,
    bootstrap_ci,
    confusion_matrix,
    report,
)


def test_acceptance_report_precision_recall_f1() -> None:
    # For y_true ['a', 'a', 'b', 'b'] and y_pred ['a', 'b', 'b', 'b'], report(...) gives
    # accuracy == 0.75, per_class['a'].precision == 1.0, per_class['a'].recall == 0.5,
    # per_class['a'].f1 == pytest.approx(2/3), per_class['b'].precision == pytest.approx(2/3),
    # per_class['b'].f1 == pytest.approx(0.8), macro_f1 == pytest.approx(11/15) and micro_f1 == 0.75
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    rep = report(y_true, y_pred)

    assert rep.accuracy == 0.75
    assert rep.per_class["a"].precision == 1.0
    assert rep.per_class["a"].recall == 0.5
    assert rep.per_class["a"].f1 == pytest.approx(2 / 3)
    assert rep.per_class["b"].precision == pytest.approx(2 / 3)
    assert rep.per_class["b"].f1 == pytest.approx(0.8)
    assert rep.macro_f1 == pytest.approx(11 / 15)
    assert rep.micro_f1 == 0.75


def test_acceptance_confusion_matrix() -> None:
    # confusion_matrix(['a', 'a', 'b', 'b'], ['a', 'b', 'b', 'b']) returns
    # labels == ('a', 'b') and rows == ((1, 1), (0, 2)) with rows indexed by the true label
    cm = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert cm.labels == ("a", "b")
    assert cm.rows == ((1, 1), (0, 2))


def test_acceptance_unseen_label() -> None:
    # With labels=('a', 'b', 'c') and no 'c' in either list, report.undefined == (('precision', 'c'), ('recall', 'c'))
    # and per_class['c'] has precision 0.0, recall 0.0, f1 0.0 and support 0
    rep = report(["a", "b"], ["a", "b"], labels=("a", "b", "c"))
    assert rep.undefined == (("precision", "c"), ("recall", "c"))
    assert rep.per_class["c"] == ClassMetrics(
        precision=0.0,
        recall=0.0,
        f1=0.0,
        support=0,
    )


def test_acceptance_bootstrap_deterministic_and_all_correct() -> None:
    # bootstrap_ci(y_true, y_pred, 'accuracy', n_resamples=1000, confidence=0.95, seed=7)
    # called twice returns equal Interval objects whose point == 0.75,
    # and all-correct data returns Interval(point=1.0, lower=1.0, upper=1.0)
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]

    ci1 = bootstrap_ci(y_true, y_pred, "accuracy", n_resamples=1000, confidence=0.95, seed=7)
    ci2 = bootstrap_ci(y_true, y_pred, "accuracy", n_resamples=1000, confidence=0.95, seed=7)

    assert ci1 == ci2
    assert ci1.point == 0.75

    all_correct = bootstrap_ci(
        ["a", "b", "c"],
        ["a", "b", "c"],
        "accuracy",
        n_resamples=100,
        confidence=0.95,
        seed=42,
    )
    assert all_correct == Interval(point=1.0, lower=1.0, upper=1.0)


def test_acceptance_resample_stub() -> None:
    # With a stub resampler giving 10 statistics and confidence 0.8, lower == sorted_stats[1] and upper == sorted_stats[8]
    # (indices floor(0.1 * 10) and ceil(0.9 * 10) - 1)
    # y_true and y_pred have 10 elements.
    # To have different accuracies from index resamples, y_true and y_pred differ on some samples.
    # For example, y_true = [1]*10, y_pred = [1]*k + [0]*(10-k).
    # Or each sample i has match: y_true[i] == y_pred[i] iff i < k.
    # When resampled with indices, accuracy is (number of idxs with y_true[idx] == y_pred[idx]) / n.
    y_true = [1] * 10
    # Let y_pred match y_true for elements 0..4, and mismatch for 5..9
    y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # We construct a resampler providing 10 resamples
    # stats will be [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    def stub_resampler(n: int, b: int, rng: random.Random) -> list[list[int]]:
        # Resample i has exactly i matching indices (from 0..4) and (10 - i) mismatching indices (from 5..9)
        # for i in 0..9
        result = []
        for i in range(b):
            idx = [0] * i + [5] * (n - i)
            result.append(idx)
        return result

    # confidence = 0.8 -> alpha = 0.2 -> alpha / 2 = 0.1
    # floor(0.1 * 10) = 1 -> sorted_stats[1] == 0.1
    # ceil(0.9 * 10) - 1 = 9 - 1 = 8 -> sorted_stats[8] == 0.8
    ci = bootstrap_ci(
        y_true,
        y_pred,
        "accuracy",
        n_resamples=10,
        confidence=0.8,
        _resample=stub_resampler,
    )
    # Compute the 10 statistics explicitly
    stats = []
    for idxs in stub_resampler(10, 10, random.Random()):  # noqa: S311
        stats.append(sum(1 for idx in idxs if y_true[idx] == y_pred[idx]) / 10.0)
    sorted_stats = sorted(stats)
    assert sorted_stats[1] == 0.1
    assert sorted_stats[8] == 0.8
    assert ci.lower == sorted_stats[1]
    assert ci.upper == sorted_stats[8]
    assert math.floor(0.1 * 10) == 1
    assert math.ceil(0.9 * 10) - 1 == 8


def test_acceptance_errors() -> None:
    # Length mismatch raises MetricsError with code 'length_mismatch', empty input 'empty_input',
    # a label outside labels=('a', 'b') 'unknown_label', metric 'auc' 'unknown_metric',
    # and confidence 1.0 or n_resamples 0 'invalid_parameter'
    with pytest.raises(MetricsError) as exc_info:
        report(["a"], ["a", "b"])
    assert exc_info.value.code == "length_mismatch"

    with pytest.raises(MetricsError) as exc_info:
        report([], [])
    assert exc_info.value.code == "empty_input"

    with pytest.raises(MetricsError) as exc_info:
        report(["a", "c"], ["a", "b"], labels=("a", "b"))
    assert exc_info.value.code == "unknown_label"

    with pytest.raises(MetricsError) as exc_info:
        report(["a", "b"], ["a", "c"], labels=("a", "b"))
    assert exc_info.value.code == "unknown_label"

    with pytest.raises(MetricsError) as exc_info:
        bootstrap_ci(["a", "b"], ["a", "b"], "auc")  # type: ignore[arg-type]
    assert exc_info.value.code == "unknown_metric"

    with pytest.raises(MetricsError) as exc_info:
        bootstrap_ci(["a", "b"], ["a", "b"], "accuracy", confidence=1.0)
    assert exc_info.value.code == "invalid_parameter"

    with pytest.raises(MetricsError) as exc_info:
        bootstrap_ci(["a", "b"], ["a", "b"], "accuracy", confidence=0.0)
    assert exc_info.value.code == "invalid_parameter"

    with pytest.raises(MetricsError) as exc_info:
        bootstrap_ci(["a", "b"], ["a", "b"], "accuracy", n_resamples=0)
    assert exc_info.value.code == "invalid_parameter"


def test_acceptance_sample_limits() -> None:
    # 1_000_001 samples with the default max_samples=1_000_000 raise MetricsError with code 'too_many_samples';
    # n_resamples 100_001 raises 'invalid_parameter' while 100_000 is accepted
    y_dummy = [0] * 1_000_001
    with pytest.raises(MetricsError) as exc_info:
        report(y_dummy, y_dummy)
    assert exc_info.value.code == "too_many_samples"

    with pytest.raises(MetricsError) as exc_info:
        bootstrap_ci([0, 1], [0, 1], "accuracy", n_resamples=100_001)
    assert exc_info.value.code == "invalid_parameter"

    # 100_000 should be accepted without raising parameter validation
    # Use small data and verify no invalid_parameter error on initialization
    ci = bootstrap_ci([0, 1], [0, 1], "accuracy", n_resamples=100_000, seed=1)
    assert ci.point == 1.0
    assert ci.lower == 1.0
    assert ci.upper == 1.0


def test_bootstrap_macro_and_micro_f1() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    ci_micro = bootstrap_ci(y_true, y_pred, "micro_f1", n_resamples=50, seed=42)
    ci_macro = bootstrap_ci(y_true, y_pred, "macro_f1", n_resamples=50, seed=42)
    assert ci_micro.point == 0.75
    assert ci_macro.point == pytest.approx(11 / 15)
    assert 0.0 <= ci_micro.lower <= ci_micro.upper <= 1.0
    assert 0.0 <= ci_macro.lower <= ci_macro.upper <= 1.0


def test_labels_handling_and_edge_cases() -> None:
    # Single class
    rep = report(["pos", "pos"], ["pos", "pos"])
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.per_class["pos"].f1 == 1.0
    assert rep.undefined == ()

    # Unorderable labels without explicit labels should raise invalid_parameter
    with pytest.raises(MetricsError) as exc_info:
        report([1, "string"], [1, "string"])
    assert exc_info.value.code == "invalid_parameter"

    # Providing explicit labels for unorderable types works
    rep_mixed = report([1, "string"], [1, "string"], labels=(1, "string"))
    assert rep_mixed.accuracy == 1.0

    # Duplicate labels given
    with pytest.raises(MetricsError) as exc_info:
        report([1, 2], [1, 2], labels=(1, 1))
    assert exc_info.value.code == "invalid_parameter"

    # Tuples as labels
    t_true = [(1, 2), (3, 4)]
    t_pred = [(1, 2), (1, 2)]
    rep_tuple = report(t_true, t_pred)
    assert rep_tuple.accuracy == 0.5
    assert (1, 2) in rep_tuple.per_class

    # Python bool vs int caveat: True == 1, False == 0
    # Testing mixed bool and int
    b_true = [True, 0]
    b_pred = [1, False]
    cm_bool = confusion_matrix(b_true, b_pred)
    # In python, {True, 1} collapses to one item
    assert len(cm_bool.labels) == 2


def test_message_truncation() -> None:
    long_msg = "x" * 500
    err = MetricsError("invalid_parameter", long_msg)
    assert len(str(err)) == 200


def test_calculate_metric_edge_cases() -> None:
    from classification_metrics import _calculate_metric

    # empty sequence in _calculate_metric
    assert _calculate_metric([], [], "accuracy") == 0.0

    # unknown metric in _calculate_metric
    with pytest.raises(MetricsError) as exc_info:
        _calculate_metric(["a"], ["a"], "invalid_metric")  # type: ignore[arg-type]
    assert exc_info.value.code == "unknown_metric"
