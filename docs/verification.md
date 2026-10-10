# Local verification — 2026-10-10

Executed in the ForecastOps workspace on macOS with Python 3.12 and Docker Desktop.

| Check | Observed result |
|---|---|
| Ruff lint and format | Passed |
| Pytest | 26 passed; dependency deprecation warnings remain |
| Local MLflow + SQLite demo | Registered/promoted XGBoost; 336 forecast rows |
| Docker image build | Passed |
| Compose services | API, Postgres, MLflow, Prometheus and Grafana running |
| Docker worker | Registered/promoted XGBoost through the HTTP MLflow service |
| Postgres batch | 336 rows |
| Live item API | HTTP 200, seven requested days, valid bounds and matching model version |
| Prometheus | API scrape target `up` |
| Grafana | Provisioned ForecastOps dashboard loaded with nine panels |
| Terraform 1.10.5 | Init without backend, fmt and validate passed |

The Docker champion run was `20261010T104347225152Z`. The separate host-side synthetic results are recorded in `demo-results.json`. Both used simulated retail data, not M5.

No AWS plan/apply, remote GitHub Actions run, full M5 experiment, production load test, or observed production retraining was performed. Retraining behavior was exercised using generated future actuals in integration tests. No cloud resources were created.
