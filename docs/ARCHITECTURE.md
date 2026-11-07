# Architecture

`trysac-classification-metrics` implements classification evaluation routines without external scientific libraries.

## Core Concepts

### 1. Label Resolution
Labels default to `sorted(set(y_true) | set(y_pred))` when mutually orderable. If labels have heterogeneous unorderable types, the caller must pass explicit `labels`.

### 2. Zero-Division Handling
Any `0/0` division returns `0.0` and records `(metric, label)` in `Report.undefined` in label order (with precision preceding recall for the same label). F1 when `p + r == 0` evaluates to `0.0` without duplicate undefined logging.

### 3. Percentile Bootstrap Interval
Bootstrap draws `n` indices with replacement using Python's standard `random.Random(seed)`. The percentile interval calculates:
- `alpha = 1 - confidence`
- `lower = s[floor(alpha / 2 * B)]`
- `upper = s[ceil((1 - alpha / 2) * B) - 1]`
where `s` is the sorted list of resampled statistics.
