# Controlled M5 memory comparison — 2026-10-12

Same 150-series, 730-day frozen cohort and V1 CV/holdout chronology. Both fresh processes used the same dependency environment, MLflow disabled, local model/report writes and a SQLite forecast batch. These peaks are comparable to each other; the original 4,057.95-MiB benchmark additionally included tracking and extra reporting, so it is a different workload.

| Measurement | Dense reference | Sparse implementation |
|---|---:|---:|
| Whole-process peak RSS (MiB) | 4298.73 | 3228.50 |
| Training wall seconds | 54.248 | 25.423 |
| Batch wall seconds | 0.240 | 0.156 |
| Ridge CV WAPE | 0.899353 | 0.899353 |
| XGBoost CV WAPE | 0.895408 | 0.895408 |

Peak reduction: **24.90%**. One measurement per implementation, host macOS; no statistical latency/capacity claim. Raw evidence: [dense](m5-memory-dense.json), [sparse](m5-memory-sparse.json). Both hashes match the frozen dataset. Source hashes identify the extracted source snapshots; their Git revision is `unknown` because they contain no `.git` directory.

Changes: categorical frame columns; serving rolling features limited to 28 rows per series; omit expanded training frames for baseline evaluation; sparse one-hot encoding; numeric values, including zeros, explicitly stored in CSR; Ridge LSQR with tolerance 1e-8. Numeric precision remains float64. [XGBoost documents](https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data) that absent CSR values mean missing. Numeric zeros are retained. Absent categorical indicators use its missing branch; dense/sparse categorical models are not promised bitwise equivalence. Accuracy is measured, not assumed. Unseen series are rejected by the service contract.

The reference is commit `305c153cddd374aaec8a10d1c91fe528af55d965`. The exact sparse stage is retained as [a patch](experiments/memory-sparse.patch), before calibration/calendar/model-list changes. To reproduce, extract that commit into two separate temporary source directories, apply the patch with `patch -p1` to one, install the reference `uv.lock` into an isolated environment, and launch the current `scripts/benchmark_pipeline.py --data data/processed/m5-50.csv --output <new-file>` in a fresh process for each, setting `PYTHONPATH` to its extracted `src`. Do not point at the default artifact directory: the profiler uses an owned temporary directory. The sparse stage hash covers only source, not its benchmark driver.

The **complete V2** workload is reported separately in [V2 results](m5-v2-results.md). Neither measured training workload fits the configured 2-GiB task. Full-M5 scaling, repeated profiles and worker sizing remain open.
