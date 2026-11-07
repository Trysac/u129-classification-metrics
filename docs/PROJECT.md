# Project Record

## Objective
To provide classification evaluation routines (confusion matrix, metrics, and uncertainty estimation) using only the Python standard library, avoiding external scientific dependencies.

## Scope
- Compute confusion matrices for single-label multiclass classification.
- Provide a report containing accuracy, per-class precision/recall/F1, and macro/micro F1.
- Calculate seeded percentile bootstrap confidence intervals.
- Enforce type safety and standard error handling via `MetricsError`.

## Out-of-scope
- Multilabel classification support.
- Probability-based metrics (e.g., LogLoss, ROC/AUC).
- Paired significance testing.
- Data plotting or visualization features.
- Non-standard library dependencies.

## Factory run u129 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4173 | 8598 | 0 | 0 | 0.0177 |
| fix-1 | gemini-3.8-flash | no | 67957 | 2499 | 0 | 477108 | 0.0961 |
| fix-2 | gemini-3.8-flash | no | 48620 | 2812 | 0 | 169936 | 0.0598 |
| fix-3 | gemini-3.8-flash | no | 32302 | 1167 | 0 | 50216 | 0.0324 |
| fix-4 | gemini-3.8-flash | no | 37174 | 429 | 0 | 77012 | 0.0353 |
| fix-1 | gemini-3.8-flash | no | 11387 | 247 | 0 | 7044 | 0.0100 |
| audit | gemini-3.1-flash-lite | no | 107191 | 450 | 0 | 642019 | 0.0435 |
| docs | gemini-3.1-flash-lite | no | 197868 | 1236 | 0 | 0 | 0.0513 |

Total: $0.3460. Prices: Anthropic list prices.
