# classification-metrics

Classification metrics without numpy: confusion matrix, per-class precision, recall and F1, macro and micro F1, and seeded percentile bootstrap intervals. This library relies exclusively on the Python standard library.

## The Problem
Evaluation scripts often report single accuracy scores without uncertainty estimation, potentially misinterpreting small variations as significant progress. Furthermore, importing large scientific libraries like `scikit-learn` or `numpy` solely for basic metrics adds heavy dependencies, and conventions for zero-division behavior often differ between tools.

## Installation
```bash
pip install classification-metrics
```

## Usage
```python
from classification_metrics import report, bootstrap_ci

y_true = ["a", "a", "b", "b"]
y_pred = ["a", "b", "b", "b"]
rep = report(y_true, y_pred)
# Accuracy: 0.75, Macro-F1: 0.7333

ci = bootstrap_ci(y_true, y_pred, "accuracy", n_resamples=1000, seed=7)
print(f"Accuracy 95% CI: [{ci.lower:.2f}, {ci.upper:.2f}]")
```

## Public API Summary
- `confusion_matrix(y_true, y_pred, *, labels=None, max_samples=1_000_000)`
- `report(y_true, y_pred, *, labels=None, max_samples=1_000_000)`
- `bootstrap_ci(y_true, y_pred, metric, *, n_resamples=1000, confidence=0.95, seed=0, labels=None, max_samples=1_000_000)`

**Options:**
- `labels`: Explicit label sequence; required if labels are not orderable.
- `max_samples`: Limits input size (default 1,000,000).
- `metric`: Choice of "accuracy", "macro_f1", "micro_f1" for bootstrap.

**Errors:**
Functions raise `MetricsError` with codes: `length_mismatch`, `empty_input`, `unknown_label`, `unknown_metric`, `invalid_parameter`, or `too_many_samples`.

## Why not scikit-learn?
`scikit-learn` introduces heavy runtime dependencies. This utility provides a zero-dependency, standard-library-only implementation with explicit zero-division reporting (via the `Report.undefined` field) and built-in, reproducible bootstrap confidence intervals.

See [docs/SECURITY.md](docs/SECURITY.md) for threat modeling details.