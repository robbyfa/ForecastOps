# Current implementation audit status — 2026-10-12

This page describes the current implementation. The [original 2026-10-11 audit and dated remediation updates](audit-history-2026-10-11.md) remain unchanged as historical evidence. Statements such as “F5 remains open”, “52 tests” and “M5 files not downloaded” in that archive describe earlier stages, not today's state.

Verdict: a tested local forecasting/MLOps prototype with real M5 evidence. It is not a demonstrated cloud production service. The priority improvements are implemented, with material memory and coverage limitations below.

| Area | Current behavior and evidence | Remaining limitation |
|---|---|---|
| Feature leakage | Origin-time demand/price; training/serving equivalence and future-perturbation tests. Sparse CSR retains numeric zeros. Calendar flags derive only from target dates; causal SBA uses no future nonzero initialization. | Actuals lack source availability/version timestamps. No scheduled promotion/event feed. |
| Backtesting | Expanding-window selection folds; labels end before each cutoff; baseline frames need no expanded training table. | Single prespecified recipe per model; M5 subset is mature, balanced products. |
| Holdout independence | Separate selection, calibration and final test blocks. Frozen artifact scored against incumbent on later data. | Already-viewed M5 holdout is reused only for explicitly labeled engineering comparisons. New unseen observations are needed for independent confirmation. |
| Champion/challenger | Same baseline, incumbent and segment gates; every CV fold must beat seasonal naive. New gate uses prior `data_end`, not older fitted-model cutoff. | Minimum latency gate measures prediction only; concurrent writers require external/shared locking. |
| MLflow lifecycle (F4/F6) | Registered versions, pinned immutable URI, durable promotion journal, recoverable alias mirror; code/lock hashes and build identity. Portable string/double wire inputs. | POSIX filesystem assumptions; trusted serialized model artifacts; tracking-store backup required. |
| Forecasts (F5) | Content-addressed retained batches, transactional active pointer, idempotent timestamps, legacy migration and explicit historical API access. | Cannot recover previously overwritten legacy rows; privileged SQL can modify history. |
| Monitoring (F1–F3) | First timely issuance across versions, per-horizon errors and matched references, equal-series drift reference, freshness and explicit replay exclusion. | Drift thresholds are heuristics; interval coverage is measured offline, not yet a live alert. |
| Retraining | Requires fresh data, sufficient per-horizon matched actuals, degradation and a full holdout after incumbent `data_end`. Keeps champion on failure. | Scheduler/ingestion and external notification routing are not installed. |
| API | Health/readiness separation, freshness enforcement, pagination and batch/date history. | Local demo has no production authentication/TLS. |
| Docker/CI (F8) | Isolated PostgreSQL + HTTP MLflow + HTTP API integration; migration/concurrency/rollback checks. Deprecation/future warnings fail tests. | Cloud deployment and sustained load reliability are unproven. |
| Training memory (F7) | [Controlled comparison](m5-memory-results.md): 4,299 → 3,229 MiB, 25% reduction; 54.2 → 25.4s. CSR, categorical frames, smaller serving history, no baseline training expansion. | Full V2 peak 3,614 MiB; still exceeds the configured 2-GiB task. Full M5 scale untested. |
| Uncertainty | [V2 evidence](m5-v2-results.md): separate per-horizon calibration, finite-sample order statistic, fixed fitted model, counts/ranks/radii/coverage/width and segment results. | 88.95% observed vs 90% target; horizon 1 and FOODS around 83%. Temporal/cross-series dependence prevents a formal coverage guarantee. |
| Sparse-demand models | Croston-SBA comparator plus its causal estimate as a feature; deterministic month-end/New Year/Christmas flags. | SBA did not win M5 pooled CV; it cannot track obsolescence during zeros. Features have not been separately ablated. |
| Terraform | ECR/S3/ECS/EFS/CloudWatch configuration and validation workflow. | No AWS provisioned; external networking, database, registry, security and measured worker sizing required. |

## Evidence chronology

1. **2026-10-10:** original synthetic experiment ([frozen results](demo-results.json)); not real-data performance.
2. **2026-10-11:** initial audit, supplied competition data, frozen cohort/protocol and [first M5 experiment](m5-results.md). Original data and results remain unchanged.
3. **2026-10-11:** monitoring/freshness/promotion/provenance fixes (`e6443c9`), then retained issuance history and full service integration (`305c153`). Historical verification had 66 default tests and nine additional PostgreSQL cases.
4. **2026-10-12:** sparse memory comparison, separate calibration blocks, SBA/calendar additions and dependency warning cleanup. [V2 JSON](m5-v2-results.json) records the observed cohort comparison, not a new independent holdout. The local suite has 72 tests passing without warnings; see the current CI run for container verification.

## Warning policy

The old thousands of warnings were predominantly pandas 2.x timedelta operations with NumPy 2.5, plus SQLAlchemy 2.1 `noload` use by MLflow and Starlette's deprecated httpx client. The lock now uses NumPy 2.3.5 and SQLAlchemy 2.0.54 with documented upper compatibility bounds; tests use httpx2. The model has a DataFrame input annotation, portable MLflow wire types and categorical groupby calls explicitly set `observed=True`.

Pytest raises deprecation and future warnings as errors. User warnings remain visible: making every UserWarning an error breaks MLflow's internal, normally muted Responses-agent schema construction during import. No blanket ignore or message-level suppression was added. Revisit compatibility caps when upgrading pandas/MLflow together.

## Next acceptance criteria

- Train within an explicitly sized worker memory budget; repeat profiles under realistic concurrency before cloud deployment.
- Freeze a new evaluation cohort/time period before any segment calibration tuning; measure horizon and category coverage on newly observed outcomes.
- Provision AWS only with account/network/registry inputs and a reviewed plan, then measure complete service latency, errors, memory and uptime.
