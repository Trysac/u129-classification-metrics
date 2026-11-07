# classification-metrics

[![PyPI version](https://img.shields.io/pypi/v/trysac-classification-metrics)](https://pypi.org/project/trysac-classification-metrics/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-classification-metrics)](LICENSE)
[![CI](https://github.com/Trysac/u129-classification-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u129-classification-metrics/actions/workflows/ci.yml)

Classification metrics without numpy: confusion matrix, per-class precision, recall and F1, macro and micro F1, and seeded percentile bootstrap intervals

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-classification-metrics
# or
uv add trysac-classification-metrics
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import classification_metrics

# normalize() trims surrounding whitespace and lower-cases the text.
print(classification_metrics.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    classification_metrics.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
