import pytest

from forecastops.config.settings import Settings
from forecastops.data.synthetic import generate
from forecastops.data.validate import validate


@pytest.fixture
def sales(tmp_path):
    return validate(generate(tmp_path / "fixture.csv", days=140, stores=2, items=3))


@pytest.fixture
def settings(tmp_path):
    return Settings(
        artifact_dir=tmp_path / "artifacts",
        database_url=f"sqlite:///{tmp_path / 'forecasts.db'}",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        horizon=7,
        folds=2,
        min_train_days=70,
    )
