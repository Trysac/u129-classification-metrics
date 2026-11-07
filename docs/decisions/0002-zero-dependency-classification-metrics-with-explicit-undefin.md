# Zero-dependency classification metrics with explicit undefined tracking

Status: accepted (generated, pending human review)

## Context

Classification evaluation scripts often depend on large packages like scikit-learn or numpy for standard confusion matrices and metrics. Zero divisions (e.g. classes with zero support or zero predictions) either trigger warnings or hide silent defaults. Evaluation without uncertainty intervals often leads to false confidence in small sample shifts.

## Decision

Implement single-label multiclass confusion matrix, classification report, and seeded percentile bootstrap confidence intervals using only the Python standard library. Explicitly track 0/0 operations in an undefined tuple, and provide reproducible bootstrap intervals.

## Consequences

Zero dependencies and minimal footprint. High reproducibility with deterministic seeds. Percentile bootstrap interval is fast and self-contained, though users should note it is not BCa.
