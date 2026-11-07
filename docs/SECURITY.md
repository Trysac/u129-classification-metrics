# Security

## Threat Model

### Defended
- **Denial of Service (Input Size):** The utility restricts the input sequence length via `max_samples` (default: 1,000,000). Sequences exceeding this limit trigger a `too_many_samples` `MetricsError`.
- **Denial of Service (Resampling):** The number of bootstrap resamples is capped at `100,000` to prevent excessive computation.
- **Information Leakage:** Exception messages are truncated to 200 characters to prevent accidental exposure of potentially large or sensitive input payloads.

### Not Defended
- **Adversarial Parsing:** This utility assumes standard, non-adversarial Python sequences. It does not perform sanitization for malicious memory-exhausting object structures beyond length checks.