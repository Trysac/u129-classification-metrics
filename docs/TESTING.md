# Testing

Tests are located in `tests/test_classification_metrics.py` and executed using `pytest`.

## Test Coverage
- **Acceptance Criteria:** Validates report metrics against fixed vectors, ensuring accuracy and F1 scores match expected values (including `pytest.approx` checks).
- **Error Handling:** Exercises all `MetricsError` codes, including length mismatches, empty inputs, and unknown labels.
- **Confusion Matrix:** Verifies row/column indexing of the contingency table.
- **Bootstrap Logic:** Validates deterministic output using a seeded `random.Random` instance and checks interval boundaries using a stub resampler that produces predictable index sets.