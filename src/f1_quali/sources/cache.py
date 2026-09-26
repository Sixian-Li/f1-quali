"""Immutable local snapshots. Cached bytes are checked on every read."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_snapshot(url: str, path: Path, expected_sha256: str, *, offline: bool = False) -> bytes:
    if path.exists():
        data = path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Offline snapshot missing: {path}")
        session = requests.Session()
        session.headers["User-Agent"] = "f1-quali-research/0.1"
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
                )
            ),
        )
        with session:
            response = session.get(url, timeout=(15, 60))
            response.raise_for_status()
            data = response.content
        if sha256(data) != expected_sha256:
            raise ValueError(f"Downloaded checksum mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(data)
        temp.replace(path)
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "sha256": expected_sha256,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    if sha256(data) != expected_sha256:
        raise ValueError(f"Cached checksum mismatch: {path}; refusing silent replacement")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        # Bootstrap downloads may predate the adapter. Do not invent their retrieval time.
        manifest_path.write_text(
            json.dumps(
                {
                    "url": url,
                    "sha256": expected_sha256,
                    "retrieved_at": None,
                    "first_verified_at": datetime.now(UTC).isoformat(),
                    "note": "Existing local snapshot; original retrieval time not recorded here",
                },
                indent=2,
            )
            + "\n"
        )
    return data
