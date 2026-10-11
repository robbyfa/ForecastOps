# M5 experiment — 2026-10-11

A single frozen retrospective experiment using the supplied M5 competition CSV files. This is real-data benchmark evidence, not a production-readiness claim or the official M5 WRMSSE score.

## Frozen protocol

- Source: `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`; the separate validation file and sample submission are archived but not used for scoring.
- Stores: CA_1, TX_1, WI_1. Products: 50 shared items (17 FOODS, 17 HOBBIES, 16 HOUSEHOLD).
- Dates: 2014-05-24 through 2016-05-22; 730 days, 150 store/item series, **109,500 observations**.
- Seed: 42. Eligibility uses positive finite prices in the initial training period only. Sampling is stratified by category from sorted eligible IDs, with no demand-based selection.
- Selected IDs were written before evaluation-period price validation or model scoring; missing evaluation prices would abort, not trigger replacement sampling.
- CV origins: 2016-03-27, 2016-04-10, 2016-04-24. Each predicts the next 14 days. Final holdout: 2016-05-09 through 2016-05-22, from origin 2016-05-08.
- Original model parameters and quality thresholds were retained. No tuning followed inspection of the scores.
- Input hashes: [m5-inputs.json](m5-inputs.json). Exact cohort and protocol: [m5-protocol.json](m5-protocol.json). Complete fold, segment, horizon and holdout results: [m5-results.json](m5-results.json).

## Measured results

| Model | CV WAPE | CV MAE | CV RMSE | CV sMAPE |
|---|---:|---:|---:|---:|
| seasonal_naive | 107.02% | 0.8905 | 1.6572 | 77.93% |
| moving_average | 90.35% | 0.7518 | 1.2836 | 117.90% |
| ridge | 89.94% | 0.7483 | 1.2485 | 143.21% |
| xgboost | 89.54% | 0.7450 | 1.2412 | 149.87% |

| Final holdout | WAPE | MAE | sMAPE |
|---|---:|---:|---:|
| Seasonal naive | 102.05% | 0.9486 | 82.48% |
| CV-selected XGBoost | 84.79% | 0.7881 | 146.86% |

The selected recipe reduced holdout WAPE by **16.91% relative to seasonal naive**. It passed every CV baseline comparison and the holdout/segment/latency gates. The final full-history fit was registered as MLflow version 1 in the isolated M5 registry; it produced **2,100 forecast rows** for 2016-05-23 through 2016-06-05. Those future rows have no supplied actuals and are not included in accuracy metrics.

The selected panel has **62.94% zero-demand rows** across the full 730 days. WAPE can exceed 100% when absolute error exceeds total demand; it is not an accuracy percentage. XGBoost wins on the preselected WAPE objective but has worse sMAPE than seasonal naive, and its CV WAPE advantage over Ridge is small. Do not describe it as superior on every metric. The cohort is a balanced sample of mature products, not a volume-weighted representation of the full hierarchy.

## Runtime and artifacts

- Training wall time including local MLflow tracking: **59.84 s**.
- Batch wall time including registered model loading and SQLite writes: **0.259 s**.
- Whole-process peak RSS: **4057.95 MiB**. This includes training and subsequent artifact reporting; it is not a per-stage or container measurement.
- Local artifacts: `artifacts/m5/runs/20261011T054413789163Z`: per-model CV predictions, selected-model and baseline holdout predictions, final forecasts, model artifact and training report.
- M5 registry: `artifacts/m5/mlflow.db`; forecasts: `artifacts/m5/forecasts.db`. Synthetic demo state was not replaced.
- Data and model artifacts remain Git-ignored; metadata and summary metrics are in `docs/`.

## Reproduction

With the supplied raw files extracted under `data/raw/m5`, the first run used:

```bash
uv run python scripts/m5_experiment.py prepare
uv run python scripts/m5_experiment.py run
```

The script refuses to overwrite a frozen cohort or silently repeat a completed experiment. Reproduce in a fresh checkout/data workspace using the same raw inputs and settings; preserve this protocol and first result. Changing the protocol or tuning against this holdout requires a new clearly labelled experiment and a later independent test period.

## Remaining limitations

The original audit findings remain unresolved. The benchmark does not exercise daily production monitoring, crash recovery, immutable forecast publication or AWS deployment. The current uncertainty bands remain pooled CV residual bands; no calibrated coverage claim is made. End-of-day availability of sales and the listed weekly price remains an explicit input assumption.

CV selects a recipe; holdout scores come from its fit through 2016-05-08. The registered artifact is refitted through 2016-05-22, so the holdout is not an independent evaluation of that final fitted artifact.

The observed near-4-GiB process peak reinforces the need to profile training separately from the 2-GiB API task before any cloud worker rollout.
