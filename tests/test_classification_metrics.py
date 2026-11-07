"""Behavior tests for the public API of classification_metrics."""

from __future__ import annotations

import pytest

import classification_metrics


def test_normalize_trims_and_lowercases() -> None:
    assert classification_metrics.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        classification_metrics.normalize(value)


def test_version_is_semver() -> None:
    parts = classification_metrics.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
