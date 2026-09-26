"""Keep identities, labels, practice evidence, and reconstructed availability separate."""

import numpy as np
import pandas as pd

from f1_quali.sources.f1db import F1DB

INELIGIBLE_QUALIFYING_STATUSES = {"DSQ", "DQ", "EX", "CONFLICTING_SOURCE_ROWS"}


def assert_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame[keys].isna().any().any():
        raise ValueError(f"Null identity in {name}: {keys}")
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate identity in {name}: {keys}")


def build_entries(
    season_entries: pd.DataFrame, events: pd.DataFrame, overrides: list[dict]
) -> pd.DataFrame:
    """Expand event membership without consulting target results or lap validity.

    Season entry round lists are retrospective metadata, not original entry-list snapshots.
    Ambiguities are preserved for audit. Predictions reject unreconciled constructor counts.
    """
    entries = season_entries[~season_entries.testDriver].copy()
    entries["round"] = entries["rounds"].fillna("").astype(str).str.split(";")
    entries = entries.explode("round")
    entries["round"] = pd.to_numeric(entries["round"], errors="coerce")
    entries = entries.rename(
        columns={"year": "season", "driverId": "driver_id", "constructorId": "constructor_id"}
    )[["season", "round", "driver_id", "constructor_id"]].drop_duplicates()
    entries = entries.merge(
        events[["event_id", "season", "round", "prediction_cutoff"]],
        on=["season", "round"],
        how="inner",
        validate="many_to_one",
    )
    entries["round"] = entries["round"].astype(int)
    entries["entry_basis"] = "retrospective_season_round_membership"
    entries["entry_reference"] = "f1db:seasons-entrants-drivers"
    entries["entry_known_at"] = pd.NaT
    entries["entry_known_at"] = pd.to_datetime(entries.entry_known_at, utc=True)
    for rule in overrides:
        mask = (
            (entries.season == rule["year"])
            & (entries["round"] == rule["round"])
            & (entries.prediction_cutoff >= pd.Timestamp(rule["effective_at"]))
        )
        if not mask.any():
            continue
        replacement = mask & (entries.driver_id == rule["replacement_driver_id"])
        if replacement.sum() != 1:
            raise ValueError(f"Entry override has no unique replacement: {rule}")
        entries.loc[replacement, "entry_basis"] = "pre_qualifying_announcement"
        entries.loc[replacement, "entry_reference"] = rule["source"]
        entries.loc[replacement, "entry_known_at"] = pd.Timestamp(rule["effective_at"])
        entries = entries[~(mask & (entries.driver_id == rule["remove_driver_id"]))].copy()
    assert_unique(entries, ["event_id", "driver_id"], "entries")
    entries["constructor_entry_count"] = entries.groupby(
        ["event_id", "constructor_id"]
    ).driver_id.transform("size")
    return entries.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)


def build_tables(
    db: F1DB, races: pd.DataFrame, sessions: pd.DataFrame, overrides: list[dict], as_of: str
) -> dict[str, pd.DataFrame]:
    events = races.rename(
        columns={
            "id": "event_id",
            "year": "season",
            "date": "race_date",
            "grandPrixId": "grand_prix_id",
            "circuitId": "circuit_id",
            "circuitLayoutId": "circuit_layout_id",
            "qualifyingFormat": "qualifying_format",
        }
    )[
        [
            "event_id",
            "season",
            "round",
            "race_date",
            "grand_prix_id",
            "circuit_id",
            "circuit_layout_id",
            "qualifying_format",
        ]
    ].copy()
    qual_sessions = sessions[sessions.session_type == "Q"].copy()
    # A disputed scheduled start is not an approved prediction boundary.
    qual_sessions["prediction_cutoff"] = qual_sessions.scheduled_start_utc.where(
        ~qual_sessions.schedule_conflict
    )
    events = events.merge(
        qual_sessions[["event_id", "prediction_cutoff"]], on="event_id", validate="one_to_one"
    )
    event_ids = set(events.event_id)
    common = {
        "raceId": "event_id",
        "year": "season",
        "driverId": "driver_id",
        "constructorId": "constructor_id",
        "driverNumber": "driver_number",
        "positionNumber": "position",
        "positionText": "classification_status",
    }
    q = db.table("races-qualifying-results").rename(columns=common)
    q = q[q.event_id.isin(event_ids)].copy()
    source_rows = q.copy()
    q["source_variant_count"] = q.groupby(["event_id", "driver_id"]).driver_id.transform("size")
    conflict = q.source_variant_count > 1
    if conflict.any():
        # Preserve competing classifications in a separate table; do not pick a winner.
        # The pinned release has NC and DSQ rows for Bottas, 2023 British GP.
        for key, group in q[conflict].groupby(["event_id", "driver_id"]):
            if group.constructor_id.nunique() != 1:
                raise ValueError(f"Conflicting constructors in qualifying: {key}")
        q.loc[conflict, ["position", "q1Millis", "q2Millis", "q3Millis"]] = float("nan")
        q.loc[conflict, "classification_status"] = "CONFLICTING_SOURCE_ROWS"
        q = q.drop_duplicates(["event_id", "driver_id"], keep="first")
    for segment in ["q1", "q2", "q3"]:
        q[f"{segment}_seconds"] = pd.to_numeric(q[f"{segment}Millis"], errors="coerce") / 1000
    q = q[
        [
            "event_id",
            "season",
            "round",
            "driver_id",
            "constructor_id",
            "driver_number",
            "position",
            "classification_status",
            "q1_seconds",
            "q2_seconds",
            "q3_seconds",
            "source_variant_count",
        ]
    ]
    q["label_usable"] = q.position.gt(0) & (q.position.mod(1) == 0)
    q["missing_reason"] = ""
    q.loc[~q.label_usable, "missing_reason"] = "no_numeric_official_qualifying_position"
    q.loc[q.source_variant_count > 1, "missing_reason"] = "conflicting_source_classifications"
    q["q1_usable"] = (
        q.q1_seconds.gt(0)
        & np.isfinite(q.q1_seconds)
        & ~q.classification_status.isin(INELIGIBLE_QUALIFYING_STATUSES)
    )
    q["session_type"] = "Q"
    assert_unique(q, ["event_id", "driver_id"], "qualifying_results")
    present = events.event_id.isin(q.event_id)
    # Completed-date proxy only defines the audit universe; it is never a feature timestamp.
    past_day = pd.to_datetime(events.race_date, utc=True) + pd.Timedelta(days=1) <= pd.Timestamp(
        as_of
    )
    events["result_status"] = "scheduled_or_future"
    events.loc[past_day & ~present, "result_status"] = "past_missing_qualifying_results"
    events.loc[past_day & present, "result_status"] = "historical_results_present"
    events["source_version"] = db.version
    entries = build_entries(db.table("seasons-entrants-drivers"), events, overrides)
    practices = []
    for n in [1, 2, 3]:
        fp = db.table(f"races-free-practice-{n}-results").rename(columns=common)
        fp = fp[fp.event_id.isin(event_ids)].copy()
        fp["session_type"] = f"FP{n}"
        fp["session_id"] = fp.event_id.astype(str) + f":FP{n}"
        fp["lap_seconds"] = pd.to_numeric(fp.timeMillis, errors="coerce") / 1000
        fp["lap_usable"] = fp.lap_seconds.gt(0) & np.isfinite(fp.lap_seconds) & fp.laps.gt(0)
        fp["missing_reason"] = ""
        fp.loc[~fp.lap_usable, "missing_reason"] = "no_positive_time_or_lap_count"
        practices.append(
            fp[
                [
                    "event_id",
                    "season",
                    "round",
                    "driver_id",
                    "constructor_id",
                    "driver_number",
                    "session_id",
                    "session_type",
                    "position",
                    "classification_status",
                    "lap_seconds",
                    "laps",
                    "lap_usable",
                    "missing_reason",
                ]
            ]
        )
    practice = pd.concat(practices, ignore_index=True)
    assert_unique(practice, ["session_id", "driver_id"], "practice_results")
    practice = practice.merge(
        sessions[
            [
                "session_id",
                "scheduled_start_utc",
                "available_at",
                "time_source",
                "schedule_conflict",
                "availability_basis",
            ]
        ],
        on="session_id",
        how="left",
        validate="many_to_one",
    )
    entry_keys = entries[["event_id", "driver_id", "constructor_id"]].assign(is_event_entry=True)
    practice = practice.merge(
        entry_keys,
        on=["event_id", "driver_id", "constructor_id"],
        how="left",
        validate="many_to_one",
    )
    practice["is_event_entry"] = practice.is_event_entry.eq(True)
    drivers = db.table("drivers")[["id", "name", "abbreviation"]].rename(
        columns={"id": "driver_id", "name": "driver_name"}
    )
    # Career totals and final-season standings are intentionally never imported.
    for table in [practice, q]:
        table["source_version"] = db.version
    assert_unique(events, ["event_id"], "events")
    assert_unique(sessions, ["session_id"], "sessions")
    return {
        "events": events,
        "sessions": sessions,
        "entries": entries,
        "practice_results": practice,
        "qualifying_results": q,
        "drivers": drivers,
        "qualifying_source_rows": source_rows,
    }
