from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORECASTOPS_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///artifacts/forecasts.db"
    artifact_dir: Path = Path("artifacts")
    tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    horizon: int = Field(default=14, ge=1, le=14)
    folds: int = Field(default=3, ge=2, le=12)
    calibration_windows: int = Field(default=2, ge=1, le=12)
    min_train_days: int = Field(default=70, ge=35)
    min_improvement: float = Field(default=0.02, ge=0, lt=1)
    max_segment_regression: float = Field(default=0.1, ge=0)
    max_prediction_seconds: float = Field(default=30, gt=0)
    seed: int = 42
    max_data_age_days: int = Field(default=2, ge=0)
    max_report_age_seconds: int = Field(default=172800, gt=0)
    max_forecast_age_days: int = Field(default=2, ge=0)

    def prepare(self):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
