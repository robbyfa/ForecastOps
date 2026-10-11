import hashlib
import os
import subprocess
from pathlib import Path


def provenance(root=None):
    root = root or Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "forecastops").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    commit = os.environ.get("FORECASTOPS_GIT_COMMIT", "unknown")
    dirty = os.environ.get("FORECASTOPS_GIT_DIRTY", "unknown")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = str(
            bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        ).lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    lock = root / "uv.lock"
    return {
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.exists()
        else None,
    }
