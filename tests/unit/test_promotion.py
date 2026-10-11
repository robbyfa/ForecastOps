import json
from argparse import Namespace
from unittest.mock import Mock

import pytest

from forecastops.cli import execute
from forecastops.registry import store, tracking


def candidate(version="new", tracked=True):
    report = {"version": version, "quality_gate": {"passed": True}}
    if tracked:
        report.update(
            model_uri=f"models:/forecastops-demand/{version}", tracking_uri="sqlite:///test"
        )
    return report


def journal(settings):
    return json.loads((settings.artifact_dir / "promotion.json").read_text())


@pytest.mark.parametrize("has_incumbent", [False, True])
def test_crash_before_pointer_never_commits_candidate(settings, monkeypatch, has_incumbent):
    old = candidate("old") if has_incumbent else None
    if old:
        store.write_json(settings.artifact_dir / "champion.json", old)
    write = store.write_json

    def crash(path, value):
        if path.name == "champion.json":
            raise OSError("disk unavailable")
        write(path, value)

    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    monkeypatch.setattr(store, "write_json", crash)
    with pytest.raises(OSError):
        store.promote(settings, candidate())
    mirror.assert_not_called()
    assert store.current(settings) == old
    monkeypatch.setattr(store, "write_json", write)
    assert store.reconcile(settings)["state"] == "synced"
    assert mirror.call_args.args[0] == old
    assert store.current(settings) == old


def test_alias_failure_keeps_committed_version_and_retries(settings, monkeypatch):
    mirror = Mock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    report = candidate()
    assert store.promote(settings, report)["state"] == "pending"
    assert store.current(settings) == report
    assert journal(settings)["error_type"] == "ConnectionError"
    mirror.side_effect = None
    assert store.reconcile(settings)["state"] == "synced"
    assert "error_type" not in journal(settings)
    store.reconcile(settings)
    assert mirror.call_count == 2  # Synced recovery makes no remote call.


def test_crash_after_alias_is_idempotently_recovered(settings, monkeypatch):
    write = store.write_json
    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)

    def crash(path, value):
        if path.name == "promotion.json" and value["state"] == "synced":
            raise OSError("crash after remote update")
        write(path, value)

    monkeypatch.setattr(store, "write_json", crash)
    with pytest.raises(OSError):
        store.promote(settings, candidate())
    assert store.current(settings)["version"] == "new"
    assert journal(settings)["state"] == "pending"
    monkeypatch.setattr(store, "write_json", write)
    assert store.reconcile(settings)["state"] == "synced"
    assert mirror.call_args_list[0] == mirror.call_args_list[1]


def test_recovery_uses_current_pointer_after_rollback(settings, monkeypatch):
    monkeypatch.setattr(tracking, "sync_alias", Mock(side_effect=ConnectionError()))
    store.promote(settings, candidate("new"))
    old = candidate("old")
    store.write_json(settings.artifact_dir / "champion.json", old)
    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    assert store.reconcile(settings)["committed_version"] == "old"
    assert mirror.call_args.args[0] == old


def test_rejected_candidate_cannot_change_pointer_or_alias(settings, monkeypatch):
    old = candidate("old")
    store.write_json(settings.artifact_dir / "champion.json", old)
    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    with pytest.raises(ValueError, match="rejected"):
        store.promote(settings, {**candidate(), "quality_gate": {"passed": False}})
    assert store.current(settings) == old
    mirror.assert_not_called()
    assert not (settings.artifact_dir / "promotion.json").exists()


def test_local_only_promotion_needs_no_registry(settings, monkeypatch):
    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    assert store.promote(settings, candidate(tracked=False))["state"] == "local_only"
    mirror.assert_not_called()


def test_pending_alias_removal_survives_another_local_promotion(settings, monkeypatch):
    store.write_json(settings.artifact_dir / "champion.json", candidate("old"))
    mirror = Mock(side_effect=ConnectionError())
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    store.promote(settings, candidate("local1", tracked=False))
    store.promote(settings, candidate("local2", tracked=False))
    mirror.side_effect = None
    assert store.reconcile(settings)["state"] == "synced"
    assert mirror.call_args.args[0]["version"] == "local2"
    assert mirror.call_args.args[1]["model_uri"].endswith("/old")


def test_explicit_reconcile_reports_remote_failure(settings, monkeypatch):
    store.write_json(settings.artifact_dir / "champion.json", candidate())
    monkeypatch.setattr(tracking, "sync_alias", Mock(side_effect=ConnectionError()))
    with pytest.raises(RuntimeError, match="pending"):
        execute(Namespace(command="reconcile"), settings)
    mirror = Mock()
    monkeypatch.setattr(tracking, "sync_alias", mirror)
    result = execute(Namespace(command="reconcile"), settings)
    assert result["state"] == "synced"
    assert "registry" not in result
