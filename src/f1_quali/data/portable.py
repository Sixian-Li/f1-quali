"""Portable canonical datasets, with separate optional rating warmup tables."""

import json
from pathlib import Path

import pandas as pd

from f1_quali.data.core import Dataset
from f1_quali.integrity import seal, verify, write_frame, write_json

TABLES = ("events", "entries", "qualifying", "practice", "sessions", "drivers", "calendar")


def validate(data):
    for table, keys in [
        (data.events, ["event_id"]),
        (data.entries, ["event_id", "driver_id"]),
        (data.qualifying, ["event_id", "driver_id"]),
    ]:
        if table.duplicated(keys).any() or table[keys].isna().any().any():
            raise ValueError(f"Invalid or duplicate identities: {keys}")
    if data.events.empty:
        raise ValueError("A dataset needs explicit events")
    if not set(data.entries.event_id).issubset(set(data.events.event_id)):
        raise ValueError("Entrant references an unknown event")
    if not set(data.qualifying.event_id).issubset(set(data.events.event_id)):
        raise ValueError("Result references an unknown event")
    if not data.qualifying.session_type.eq("Q").all():
        raise ValueError("Only main qualifying belongs in the label table")
    if (
        not isinstance(data.events.prediction_cutoff.dtype, pd.DatetimeTZDtype)
        or data.events.prediction_cutoff.isna().any()
    ):
        raise ValueError("Explicit timezone-aware prediction_cutoff required")
    if not isinstance(data.entries.entry_known_at.dtype, pd.DatetimeTZDtype):
        raise ValueError("entry_known_at requires a timezone-aware column")
    unknown = data.entries.entry_known_at.isna()
    retrospective = data.entries.get("entry_basis", pd.Series("", index=data.entries.index))
    known_bases = [
        "retrospective_season_round_membership",
        "retrospective_reviewed_actual_roster_NOT_original_entry_snapshot",
    ]
    if (unknown & ~retrospective.isin(known_bases)).any():
        raise ValueError("Unknown entry timestamps require explicit retrospective provenance")
    return data


def save_dataset(directory, data, *, early_pairs=None, early_achievements=None, provenance=None):
    validate(data)
    directory = Path(directory)
    for name in TABLES:
        write_frame(directory / f"{name}.parquet", getattr(data, name))
    if (early_pairs is None) != (early_achievements is None):
        raise ValueError("Warmup requires both teammate and achievement tables")
    if early_pairs is not None:
        for name, frame in [
            ("early_pairs", early_pairs),
            ("early_achievements", early_achievements),
        ]:
            if not frame.season.lt(2016).all():
                raise ValueError("Rating warmup must precede 2016")
            write_frame(directory / f"{name}.parquet", frame)
    write_json(directory / "provenance.json", provenance or data.manifest)
    return seal(directory, kind="dataset", metadata={"warmup": early_pairs is not None})


def load_dataset(directory):
    directory = Path(directory)
    manifest = verify(directory, "dataset")
    frames = [pd.read_parquet(directory / f"{name}.parquet") for name in TABLES]
    provenance = json.loads((directory / "provenance.json").read_text())
    data = validate(Dataset(*frames, provenance))
    early = None
    if manifest["metadata"]["warmup"]:
        early = (
            pd.read_parquet(directory / "early_pairs.parquet"),
            pd.read_parquet(directory / "early_achievements.parquet"),
        )
    return data, early
