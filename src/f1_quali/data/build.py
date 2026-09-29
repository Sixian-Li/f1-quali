"""Rebuild the fixed dataset from pinned external sources and explicit reviews."""

import json
from importlib.resources import files
from io import StringIO
from pathlib import Path

import pandas as pd

from f1_quali.data.canonical import build_tables
from f1_quali.data.core import Dataset
from f1_quali.data.early import build_tables as build_early_tables
from f1_quali.data.early import review_differences
from f1_quali.data.portable import save_dataset
from f1_quali.data.reviews import (
    mark_calendar_absences,
    review_entries,
    review_qualifying,
    review_sessions,
)
from f1_quali.integrity import safe_path
from f1_quali.sources.cache import fetch_snapshot
from f1_quali.sources.f1db import F1DB
from f1_quali.sources.official import (
    constructor_identity,
    driver_identity,
    lap_millis,
    numeric_position,
    parse_page,
    validate_page_identity,
)
from f1_quali.sources.schedules import build_sessions, load_schedules


def resource(name):
    return json.loads(files("f1_quali").joinpath(f"resources/{name}.json").read_text())


def fetch_evidence(cache, source, offline):
    relative = source.get("path", source.get("source_path", ""))
    relative = relative.removeprefix("data/raw/")
    if not relative:
        raise ValueError("Source needs an explicit cache key")
    return fetch_snapshot(
        source.get("url", source.get("source_url")),
        safe_path(cache, relative),
        source["sha256"],
        offline=offline,
    )


def official_warmup(db, cache, *, offline, early_tables=False):
    """Reviewed 2010--2015 evidence; ``early_tables`` also returns events/results."""
    races = db.table("races")
    races = races[races.year.between(2010, 2015)].sort_values(["year", "round"])
    drivers = db.table("drivers")
    qualifying = db.table("races-qualifying-results")
    qualifying = qualifying[qualifying.year.between(2010, 2015)].copy()
    race_results, grid = db.table("races-race-results"), db.table("races-starting-grid-positions")
    union = pd.concat(
        [f[["raceId", "driverId", "constructorId"]] for f in [qualifying, race_results, grid]]
    ).drop_duplicates()
    sources = resource("official_sources")
    pages = {
        int(Path(r["path"]).name.split("_")[0]): r
        for r in sources
        if r["path"].endswith("_qualifying.html")
    }
    official, events, differences = [], [], []
    for event in races.itertuples():
        source = pages[event.id]
        parsed = parse_page(fetch_evidence(cache, source, offline))
        _, _, date_matches = validate_page_identity(parsed, event, source["url"])
        tables = [
            t
            for t in parsed.tables
            if t and t[0] == ["Pos.", "No.", "Driver", "Team", "Q1", "Q2", "Q3", "Laps"]
        ]
        if len(tables) != 1:
            raise ValueError("Expected a unique main qualifying table")
        roster = union[union.raceId.eq(event.id)]
        eligible = set(roster.driverId)
        prior = qualifying[qualifying.raceId.eq(event.id)].set_index("driverId")
        seen = set()
        for cells in tables[0][1:]:
            if len(cells) != 8:
                raise ValueError("Unexpected official result fields")
            position, _, name, team, q1, q2, q3, _ = cells
            driver, _ = driver_identity(name, drivers, eligible)
            constructor = constructor_identity(team, event.year)
            if driver in seen or set(roster[roster.driverId.eq(driver)].constructorId) != {
                constructor
            }:
                raise ValueError("Official entrant identity conflict")
            seen.add(driver)
            numeric = numeric_position(position)
            row = {
                "race_id": event.id,
                "driver_id": driver,
                "constructor_id": constructor,
                "position_number": numeric,
                "position_text": position,
                **{f"q{i}_millis": lap_millis(v) for i, v in enumerate([q1, q2, q3], 1)},
            }
            official.append(row)
            if driver not in prior.index:
                differences.append(
                    {
                        "race_id": event.id,
                        "year": event.year,
                        "driver_id": driver,
                        "field": "qualifying_row",
                        "f1db": "absent",
                        "official": "present",
                    }
                )
                continue
            original = prior.loc[driver]
            comparisons = {
                "position_text": (str(original.positionText), position),
                "position_number": (
                    None if pd.isna(original.positionNumber) else int(original.positionNumber),
                    numeric,
                ),
            }
            for stage in [1, 2, 3]:
                raw = original[f"q{stage}Millis"]
                comparisons[f"q{stage}_millis"] = (
                    None if pd.isna(raw) else int(raw),
                    row[f"q{stage}_millis"],
                )
            for field, (before, after) in comparisons.items():
                if before != after:
                    differences.append(
                        {
                            "race_id": event.id,
                            "year": event.year,
                            "driver_id": driver,
                            "field": field,
                            "f1db": "" if before is None else str(before),
                            "official": "" if after is None else str(after),
                        }
                    )
        for driver in sorted(eligible - seen):
            if driver in prior.index:
                differences.append(
                    {
                        "race_id": event.id,
                        "year": event.year,
                        "driver_id": driver,
                        "field": "qualifying_row",
                        "f1db": "present",
                        "official": "absent",
                    }
                )
        events.append(
            {
                "race_id": event.id,
                "source_sha256": source["sha256"],
                "official_header_date_conflict": not date_matches,
            }
        )
    reviews = resource("early_reviews")
    review_differences(pd.DataFrame(differences), reviews)
    for source in reviews["rule_sources"]:
        fetch_evidence(cache, source, offline)
    fetch_evidence(cache, reviews["q3_cancelled_event"], offline)
    supplemental = next(r for r in sources if r["path"].endswith("australia_saturday_quotes.html"))
    paragraphs = parse_page(fetch_evidence(cache, supplemental, offline)).paragraphs
    if not all(f"{name} - DNS" in paragraphs for name in ["Will Stevens", "Roberto Merhi"]):
        raise ValueError("2015 DNS evidence changed")
    pairs, achievements, early_events, results = build_early_tables(
        races, qualifying, race_results, grid, pd.DataFrame(official), pd.DataFrame(events), reviews
    )

    # Preserve the original canonical CSV precision and datetime representation.
    def canonical_roundtrip(frame):
        frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
        for column in [
            "age_at",
            "evidence_at",
            "prediction_cutoff",
            "available_at",
            "actual_start_utc",
            "actual_end_utc",
            "original_publication_timestamp",
        ]:
            frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    frames = [pairs, achievements, *([early_events, results] if early_tables else [])]
    return tuple(canonical_roundtrip(frame) for frame in frames)


def build_dataset(output, cache, *, offline=False):
    cache = Path(cache)
    config = resource("sources")
    evidence = resource("evidence")
    evidence_map = {r["key"]: r for r in evidence}
    for source in evidence:
        fetch_evidence(cache, source, offline)
    reviews = {
        name: resource(name)
        for name in ["session_reviews", "entry_reviews", "classification_reviews"]
    }
    for rules in reviews.values():
        for rule in rules:
            if evidence_map[rule["evidence_key"]]["url"] != rule["source"]:
                raise ValueError("Review source differs from pinned evidence")
    db = F1DB(cache, config["f1db"], offline=offline)
    races = db.table("races")
    races = races[races.year.between(config["year_min"], config["year_max"])].copy()
    schedules = load_schedules(cache, config["schedules"], offline=offline)
    sessions = build_sessions(
        races, schedules, [], buffer_minutes=config["availability_buffer_minutes"]
    )
    sessions = review_sessions(
        sessions, reviews["session_reviews"], config["availability_buffer_minutes"]
    )
    sessions = mark_calendar_absences(sessions, races, schedules)
    tables = build_tables(db, races, sessions, [], config["as_of"])
    calendar = tables["events"].copy()
    historical = set(
        calendar.loc[calendar.result_status.eq("historical_results_present"), "event_id"]
    )
    for name, frame in tables.items():
        if "event_id" in frame:
            tables[name] = frame[frame.event_id.isin(historical)].reset_index(drop=True)
    tables["entries"], _ = review_entries(
        tables["entries"], tables["events"], reviews["entry_reviews"]
    )
    tables["qualifying_results"] = review_qualifying(
        tables["qualifying_results"], tables["entries"], reviews["classification_reviews"]
    )
    entries = tables["entries"]
    entry_keys = set(zip(entries.event_id, entries.driver_id, entries.constructor_id))
    fp = tables["practice_results"]
    fp["is_event_entry"] = [
        (e, d, c) in entry_keys for e, d, c in zip(fp.event_id, fp.driver_id, fp.constructor_id)
    ]
    q_sessions = tables["sessions"].query("session_type == 'Q'")[
        ["event_id", "available_at", "availability_basis"]
    ]
    labels = tables["qualifying_results"].merge(
        q_sessions.rename(
            columns={
                "available_at": "label_available_at",
                "availability_basis": "label_availability_basis",
            }
        ),
        on="event_id",
        validate="many_to_one",
    )
    later = labels.decision_signed_at + pd.Timedelta(minutes=30)
    delayed = later.notna() & (
        labels.label_available_at.isna() | later.gt(labels.label_available_at)
    )
    labels.loc[delayed, "label_available_at"] = later[delayed]
    labels.loc[delayed, "label_availability_basis"] = "decision_signed_plus_buffer_reconstruction"
    labels["original_label_publication_status"] = "unknown_revised_historical_data"
    used = set(entries.driver_id) | set(fp.driver_id)
    drivers = tables["drivers"][tables["drivers"].driver_id.isin(used)].copy()
    provenance = {
        "kind": "reconstructed_historical_availability",
        "data_cutoff": config["as_of"],
        "f1db": config["f1db"],
        "schedules": config["schedules"],
        "original_publication_timestamps": "unknown",
        "raw_sources": "external_downloads_not_redistributed",
    }
    data = Dataset(
        tables["events"].sort_values(["prediction_cutoff", "event_id"]).reset_index(drop=True),
        entries,
        labels,
        fp,
        tables["sessions"],
        drivers,
        calendar,
        provenance,
    )
    early_pairs, early_achievements = official_warmup(db, cache, offline=offline)
    return save_dataset(
        output, data, early_pairs=early_pairs, early_achievements=early_achievements
    )
