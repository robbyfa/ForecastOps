from importlib.metadata import version as installed_version
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

from forecastops.features.build import FEATURES
from forecastops.training.models import PARAMETERS, predict

MODEL_NAME = "forecastops-demand"


class ForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator

    def predict(self, context, model_input, params=None):
        return predict(self.name, self.estimator, model_input)


def record(settings, report, bundle, example, promote):
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment("forecastops")
    client = mlflow.MlflowClient()
    for name, entry in report["models"].items():
        with mlflow.start_run(run_name=f"{name}-{report['version']}"):
            mlflow.log_params(
                {
                    "model": name,
                    "seed": settings.seed,
                    "horizon": settings.horizon,
                    "folds": settings.folds,
                    "feature_set": "direct-origin-v1",
                }
            )
            mlflow.log_params(PARAMETERS[name])
            mlflow.set_tags(
                {
                    "dataset_sha256": report["dataset_sha256"],
                    "git_commit": report["git_commit"],
                    "train_end": report["train_end"],
                    "feature_set": "direct-origin-v1",
                }
            )
            mlflow.log_metrics({f"cv_{k}": v for k, v in entry["overall"].items() if v is not None})
            if name == bundle["name"]:
                mlflow.log_metrics(
                    {
                        f"holdout_{k}": v
                        for k, v in report["holdout"]["candidate"]["overall"].items()
                        if v is not None
                    }
                )
                mlflow.log_metric("training_seconds", report["training_seconds"])
                mlflow.log_dict(report, "backtest_report.json")
                model = ForecastModel(name, bundle["model"])
                inputs = example[FEATURES].head(5)
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=model,
                    signature=infer_signature(inputs, model.predict(None, inputs)),
                    input_example=inputs,
                    registered_model_name=MODEL_NAME,
                    code_paths=[str(Path(__file__).parents[1])],
                    pip_requirements=[
                        f"{p}=={installed_version(p)}"
                        for p in ["mlflow", "numpy", "pandas", "scikit-learn", "xgboost"]
                    ],
                )
                version = str(info.registered_model_version)
                client.set_model_version_tag(MODEL_NAME, version, "quality_gate", str(promote))
                if promote:
                    client.set_registered_model_alias(MODEL_NAME, "champion", version)
                uri = f"models:/{MODEL_NAME}/{version}"
    return {"model_uri": uri, "tracking_uri": settings.tracking_uri}
