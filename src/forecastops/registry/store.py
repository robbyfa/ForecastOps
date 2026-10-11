import json
import logging
import os
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with temporary.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def current(settings):
    path = settings.artifact_dir / "champion.json"
    return json.loads(path.read_text()) if path.exists() else None


def _registry(report):
    if report and report.get("model_uri"):
        return {key: report[key] for key in ("model_uri", "tracking_uri")}
    return None


def reconcile(settings, force=False):
    """Mirror the committed pointer, never the candidate in an interrupted journal.

    Callers must hold the pipeline lock. Remote failure leaves a retryable journal;
    it does not undo a committed champion or prevent using its immutable URI.
    """
    path = settings.artifact_dir / "promotion.json"
    journal = json.loads(path.read_text()) if path.exists() else {}
    if not force and journal.get("state") != "pending":
        return journal
    champion = current(settings)
    # Remember the registry even when the committed champion is local-only/absent.
    registry = _registry(champion) or journal.get("registry")
    state = {
        "attempted_version": journal.get("attempted_version"),
        "committed_version": champion["version"] if champion else None,
        "registry": registry,
        "state": "pending" if registry else "local_only",
    }
    write_json(path, state)
    if registry:
        from forecastops.registry.tracking import sync_alias

        try:
            sync_alias(champion, registry)
        except Exception as error:
            state["error_type"] = type(error).__name__
            logging.warning("Champion committed locally; MLflow alias reconciliation pending")
        else:
            state["state"] = "synced"
        write_json(path, state)
    return state


def promote(settings, report):
    """Publish a passing, persisted run, then reconcile its discovery alias."""
    if not report["quality_gate"]["passed"]:
        raise ValueError("Cannot promote a rejected candidate")
    old = current(settings)
    path = settings.artifact_dir / "promotion.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    registry = _registry(report) or _registry(old) or previous.get("registry")
    # Persist intent before the only authoritative commit point (atomic replace).
    write_json(
        settings.artifact_dir / "promotion.json",
        {
            "attempted_version": report["version"],
            "state": "pending",
            "registry": registry,
        },
    )
    write_json(settings.artifact_dir / "champion.json", report)
    return reconcile(settings)
