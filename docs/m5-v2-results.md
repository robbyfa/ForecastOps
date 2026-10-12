# M5 V2 follow-up — 2026-10-12

Engineering comparison on the **already-inspected V1 cohort and holdout dates**. This is not a new independent test, and the frozen [V1 results](m5-results.md) are unchanged. No settings were adjusted after this V2 coverage result.

Dataset: 109,500 rows, 150 series, 730 days; canonical SHA-256 `9ab375143cf22925b245ff6e8adb45bf6a5e4e5deeb47f6f7f946b4865a3161e`. Full parameters, source/lock hashes, quality gates, all model/segment metrics and references are in [JSON evidence](m5-v2-results.json). Local residual/model/forecast artifacts are in `/Users/fenechadamir/Library/CloudStorage/OneDrive-TipicoCo.Ltd/Desktop/Udemy/ForecastOps/artifacts/m5-v2/runs/20261012T072528393630Z` (ignored competition data).

## Fixed protocol

- Selection origins: 2016-02-28, 2016-03-13, 2016-03-27.
- Fit cutoff and first calibration origin: **2016-04-10**.
- Calibration origins: 2016-04-10, 2016-04-24; 300 residuals per horizon.
- Holdout origin: **2016-05-08**, actuals through 2016-05-22.
- One frozen selected model throughout calibration, holdout and serving. Origin features update with newly observed history; coefficients/trees do not refit.
- Target coverage 90%; radius is the 271st smallest of 300 absolute residuals at each horizon (`ceil(301 × 0.9)`). Lower bounds clip to zero.
- Three selection windows shift earlier than V1 to reserve calibration. Sparse encoding, calendar/SBA features and model list also changed; score differences cannot be attributed to a single change.

The frozen cohort used price availability through 2016-03-27, later than the first two V2 selection origins; those earlier folds are retrospective cohort diagnostics. The fixed cohort was retained for comparability; this does not establish prospective performance at those earlier dates.

## Point forecasts

| Model | CV WAPE |
|---|---:|
| seasonal_naive | 100.86% |
| moving_average | 87.93% |
| croston_sba | 93.38% |
| ridge | 89.02% |
| xgboost | 87.83% |

Selected **xgboost**. Holdout WAPE **85.55%**, seasonal baseline **102.05%**. All promotion gates passed; 2,100 batch forecasts were saved. V1 XGBoost holdout WAPE was 84.79%; V2 is slightly worse, not a claimed accuracy improvement. Croston-SBA did not beat the moving average on pooled selection WAPE; it is a useful explicit intermittent-demand comparator, not the winner.

## Interval evidence

Observed coverage **88.95%** against **90% target**; mean clipped width **2.335 demand units** across 2,100 observations.

| Horizon | Calibration radius | Holdout coverage | Mean width |
|---|---:|---:|---:|
| 1 | 1.473 | 83.33% | 2.117 |
| 2 | 1.448 | 86.67% | 2.069 |
| 3 | 1.716 | 91.33% | 2.364 |
| 4 | 1.422 | 88.67% | 2.039 |
| 5 | 1.649 | 88.67% | 2.349 |
| 6 | 1.927 | 88.00% | 2.752 |
| 7 | 1.852 | 90.00% | 2.675 |
| 8 | 1.385 | 88.67% | 2.016 |
| 9 | 1.716 | 93.33% | 2.364 |
| 10 | 1.427 | 86.67% | 2.045 |
| 11 | 1.433 | 88.67% | 2.051 |
| 12 | 1.710 | 89.33% | 2.418 |
| 13 | 1.896 | 92.00% | 2.717 |
| 14 | 1.884 | 90.00% | 2.712 |

| Segment | Observations | Coverage | Mean width |
|---|---:|---:|---:|
| CA_1 | 700 | 85.57% | 2.428 |
| TX_1 | 700 | 91.71% | 2.221 |
| WI_1 | 700 | 89.57% | 2.356 |
| FOODS | 714 | 83.19% | 2.648 |
| HOBBIES | 714 | 94.68% | 2.101 |
| HOUSEHOLD | 672 | 88.99% | 2.251 |

Horizon 1 and FOODS under-cover materially. A near-90% pooled result is not uniform calibration. We did not retune widths against these results. The order statistic follows [split-conformal calibration](https://arxiv.org/abs/2107.07511); related series and temporal drift violate the exchangeability assumptions needed for an unconditional guarantee. Serving beyond the evaluation dates can drift further. Future work needs fresh observations, segment-aware calibration and measured coverage monitoring.

Croston-SBA uses fixed alpha 0.1 and the [Syntetos–Boylan correction](https://www.sciencedirect.com/science/article/pii/S0925527305001507), with causal first-positive initialization. It does not decay during prolonged zero demand, so obsolescence remains a limitation. Calendar features are only deterministic month-end, January 1 and December 25 flags; no retrospective event table or future prices are joined.

## Resources and reproduction

Training **25.693s**, SQLite batch **0.196s**, process peak **3613.80 MiB**. Includes imports, five-model CV, frozen fit, calibration, holdout, local artifacts and batch; MLflow disabled. Single host measurement, not a cloud sizing guarantee. A short local test run overlapped the beginning of this diagnostic; wall times are descriptive. Still above 2 GiB.

```bash
uv run python scripts/m5_followup.py \
  --data data/processed/m5-50.csv \
  --artifacts artifacts/m5-v2-new-run \
  --output docs/m5-v2-new-run.json
```

The script verifies the frozen dataset hash and refuses to overwrite evidence. Do not rerun `m5_experiment.py run` with V2: its original parameters/protocol are frozen. Re-running this follow-up does not create another independent holdout.
