import json
import os
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def current(settings):
    path = settings.artifact_dir / "champion.json"
    return json.loads(path.read_text()) if path.exists() else None
