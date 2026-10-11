from unittest.mock import Mock

from forecastops.training.provenance import provenance


def test_container_build_metadata_and_content_hashes(tmp_path, monkeypatch):
    package = tmp_path / "src" / "forecastops"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_text("VALUE = 1\n")
    (tmp_path / "uv.lock").write_text("locked dependencies\n")
    monkeypatch.setenv("FORECASTOPS_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("FORECASTOPS_GIT_DIRTY", "false")
    monkeypatch.setattr("subprocess.check_output", Mock(side_effect=FileNotFoundError))
    before = provenance(tmp_path)
    assert before["git_commit"] == "a" * 40
    assert before["working_tree_dirty"] == "false"
    assert len(before["dependency_lock_sha256"]) == 64
    source.write_text("VALUE = 2\n")
    after = provenance(tmp_path)
    assert before["source_sha256"] != after["source_sha256"]
    assert before["dependency_lock_sha256"] == after["dependency_lock_sha256"]


def test_unknown_container_revision_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("FORECASTOPS_GIT_COMMIT", raising=False)
    monkeypatch.delenv("FORECASTOPS_GIT_DIRTY", raising=False)
    monkeypatch.setattr("subprocess.check_output", Mock(side_effect=FileNotFoundError))
    result = provenance(tmp_path)
    assert result["git_commit"] == "unknown"
    assert result["working_tree_dirty"] == "unknown"
