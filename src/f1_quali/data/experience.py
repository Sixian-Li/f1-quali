"""Main-Q participation for year-start rookie priors, including pre-2010 experience.

Race date + 36 hours is a conservative availability proxy, not a publication
timestamp. These records affect experience counts only, never rating outcomes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1_quali.integrity import write_frame, write_json


def from_f1db(db):
    races = db.table("races")[["id", "date"]].copy()
    races["available_at"] = pd.to_datetime(races.date, utc=True) + pd.Timedelta(hours=36)
    appearances = db.table("races-qualifying-results")
    appearances = appearances[
        ~appearances.positionText.fillna("").str.upper().isin(["DNS", "NOT_PARTICIPATED"])
    ].drop_duplicates(["raceId", "driverId"])
    frame = appearances[["raceId", "driverId"]].merge(
        races[["id", "available_at"]], left_on="raceId", right_on="id", how="left",
        validate="many_to_one"
    ).rename(columns={"raceId": "event_id", "driverId": "driver_id"})
    frame = frame[["event_id", "driver_id", "available_at"]].assign(session_type="Q")
    provenance = {
        "source_version": db.version,
        "source_sha256": db.digest,
        "availability_basis": "race_date_plus_36h_reconstructed_not_publication",
        "known_drivers": sorted(db.table("drivers").id.tolist()),
        "excluded_statuses": ["DNS", "NOT_PARTICIPATED"],
        "scope": "experience_only_including_pre_2010_not_rating_evidence",
    }
    return frame, provenance


def save(directory, frame, provenance):
    validate(frame, provenance)
    write_frame(Path(directory) / "experience.parquet", frame)
    write_json(Path(directory) / "experience.json", provenance)


def validate(frame, provenance):
    if (frame.duplicated(["event_id", "driver_id"]).any()
            or frame[["event_id", "driver_id", "available_at"]].isna().any().any()
            or not frame.session_type.eq("Q").all()
            or not isinstance(frame.available_at.dtype, pd.DatetimeTZDtype)):
        raise ValueError("Unique main-Q appearances with timezone-aware availability required")
    if not set(frame.driver_id).issubset(provenance["known_drivers"]):
        raise ValueError("Experience source does not cover the driver identities")


def annual_profiles(frame, provenance, drivers, years, policy):
    validate(frame, provenance)
    if not set(drivers).issubset(provenance["known_drivers"]):
        raise ValueError("Missing career history must not be treated as zero experience")
    boost, half_life = policy["boost"], policy["half_life_appearances"]
    if not np.isfinite([boost, half_life]).all() or boost < 0 or half_life <= 0:
        raise ValueError("Finite nonnegative rookie boost and positive half-life required")
    times = {d: np.sort(g.available_at.astype("datetime64[ns, UTC]").astype("int64").to_numpy())
             for d, g in frame.groupby("driver_id")}
    rows = []
    for year in years:
        cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
        for driver in sorted(drivers):
            previous = int(np.searchsorted(times.get(driver, []), cutoff.value, side="left"))
            rows.append({"driver_id": driver, "season": int(year),
                         "parameters_available_before": cutoff,
                         "prior_main_q_appearances": previous,
                         "annual_variance_multiplier": float(
                             1 + boost * np.exp2(-previous / half_life))})
    return pd.DataFrame(rows)


def prepare_profiles(dataset, output, config, drivers):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if not {"experience.parquet", "experience.json"}.issubset(manifest["sha256"]):
        raise ValueError("RM needs career history; rebuild full-era data in a new output directory")
    frame = pd.read_parquet(dataset / "experience.parquet")
    provenance = json.loads((dataset / "experience.json").read_text())
    profiles = annual_profiles(frame, provenance, drivers,
                               sorted(map(int, config["rating_years"])), config["rookie_variance"])
    write_frame(Path(output) / "rookie_profiles.parquet", profiles)
    write_json(Path(output) / "experience_source.json", provenance)
    return factors(profiles)


def factors(profiles):
    if profiles.duplicated(["driver_id", "season"]).any():
        raise ValueError("Duplicate annual rookie profiles")
    values = profiles.annual_variance_multiplier
    if not np.isfinite(values).all() or values.lt(1).any():
        raise ValueError("Finite variance multipliers at least one required")
    return {(r.driver_id, int(r.season)): float(r.annual_variance_multiplier)
            for r in profiles.itertuples()}
