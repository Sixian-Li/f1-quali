import json

import pytest

from f1_quali.data.portable import load_dataset, save_dataset
from f1_quali.demo import run_demo
from f1_quali.integrity import safe_path, seal, verify, write_json


def test_portable_dataset_roundtrip_and_tamper_rejected(tmp_path, data):
    save_dataset(tmp_path, data)
    actual, early = load_dataset(tmp_path)
    assert early is None and len(actual.entries) == len(data.entries)
    (tmp_path / "provenance.json").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_artifact_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="within"):
        safe_path(tmp_path, path)


def test_manifest_paths_and_immutable_output(tmp_path):
    write_json(tmp_path / "model.json", {"value": 1})
    seal(tmp_path, kind="test")
    with pytest.raises(ValueError, match="overwrite"):
        write_json(tmp_path / "model.json", {"value": 2})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sha256"]["../outside.json"] = "invalid"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="within"):
        verify(tmp_path)


def test_demo_runs_offline_and_repeats_identically(tmp_path):
    first = run_demo(tmp_path / "first")
    second = run_demo(tmp_path / "second")
    assert first == second
    assert first["prediction_rows"] == 120
    assert first["performance_evidence"] is False
    assert (tmp_path / "first/model/model.json").read_bytes() == (
        tmp_path / "second/model/model.json"
    ).read_bytes()


def test_manifest_must_cover_required_payload(tmp_path, data):
    save_dataset(tmp_path, data)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["sha256"]["events.parquet"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="required files: events.parquet"):
        load_dataset(tmp_path)
