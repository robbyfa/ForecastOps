import mlflow
import pytest
from mlflow.exceptions import MlflowException

from forecastops.registry.store import current, promote, reconcile, write_json
from forecastops.registry.tracking import MODEL_NAME


def test_registry_alias_follows_committed_pointer_and_recovers(settings, tmp_path):
    client = mlflow.MlflowClient(tracking_uri=settings.tracking_uri)
    client.create_registered_model(MODEL_NAME)
    source = tmp_path / "model"
    source.mkdir()
    reports = []
    for label in ["first", "second"]:
        version = client.create_model_version(MODEL_NAME, str(source)).version
        reports.append(
            {
                "version": label,
                "model_uri": f"models:/{MODEL_NAME}/{version}",
                "tracking_uri": settings.tracking_uri,
                "quality_gate": {"passed": True},
            }
        )
    first, second = reports
    assert promote(settings, first)["state"] == "synced"
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == "1"
    # Simulate a crash after local commit and before the remote mirror call.
    write_json(
        settings.artifact_dir / "promotion.json",
        {"state": "pending", "attempted_version": "second", "registry": second},
    )
    write_json(settings.artifact_dir / "champion.json", second)
    assert reconcile(settings)["state"] == "synced"
    assert current(settings)["version"] == "second"
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == "2"
    # Explicit recovery also repairs manual alias changes with no pending journal.
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")
    reconcile(settings, force=True)
    assert str(client.get_model_version_by_alias(MODEL_NAME, "champion").version) == "2"
    # A local-only commit must not leave an alias advertising the former champion.
    assert (
        promote(settings, {"version": "local", "quality_gate": {"passed": True}})["state"]
        == "synced"
    )
    with pytest.raises(MlflowException) as error:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert "champion" in str(error.value)
    assert reconcile(settings, force=True)["state"] == "synced"
