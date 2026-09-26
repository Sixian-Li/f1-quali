"""Portable, content-checked local artifacts. No research registry is required."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must remain within their directory")
    return path


def write_json(path, value):
    path = Path(path)
    content = json_bytes(value)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite different content: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_frame(path, frame):
    path = Path(path)
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


# Payload files each artifact kind must list before its contents are trusted.
REQUIRED_FILES = {
    "dataset": {
        "events.parquet",
        "entries.parquet",
        "qualifying.parquet",
        "practice.parquet",
        "sessions.parquet",
        "drivers.parquet",
        "calendar.parquet",
        "provenance.json",
    },
    "prepared": {"features.parquet", "labels.parquet", "config.json"},
    "predictor": {"model.json", "config.json"},
    "evaluation": {
        "predictions.parquet",
        "event_metrics.parquet",
        "raw_probability_metrics.parquet",
    },
    "forecast": {"predictions.parquet", "ratings.json", "forecast.json"},
    "baselines": {"predictions.parquet", "event_metrics.parquet", "skipped.json"},
}


def seal(directory, *, kind, metadata=None):
    directory = Path(directory)
    outputs = {
        str(p.relative_to(directory)): sha256(p.read_bytes())
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    identity = {
        "schema": "f1-quali.artifact.v1",
        "kind": kind,
        "metadata": metadata or {},
        "sha256": outputs,
    }
    write_json(directory / "manifest.json", identity)
    return identity


def verify(directory, kind=None):
    directory = Path(directory)
    identity = json.loads((directory / "manifest.json").read_text())
    if identity.get("schema") != "f1-quali.artifact.v1":
        raise ValueError("Unsupported artifact schema")
    if kind is not None and identity["kind"] != kind:
        raise ValueError(f"Expected a {kind} artifact")
    required = set(REQUIRED_FILES.get(identity["kind"], ()))
    if identity["kind"] == "dataset" and identity["metadata"].get("warmup"):
        required |= {"early_pairs.parquet", "early_achievements.parquet"}
    missing = sorted(required - set(identity["sha256"]))
    if missing:
        raise ValueError(f"Manifest does not cover required files: {', '.join(missing)}")
    for name, expected in identity["sha256"].items():
        if sha256(safe_path(directory, name).read_bytes()) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return identity


def utc_now():
    return datetime.now(UTC).isoformat()
