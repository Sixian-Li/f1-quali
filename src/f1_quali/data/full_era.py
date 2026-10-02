"""2010--2026 dataset: modern canonical tables plus a 2010--2015 phase reconstruction.

The early seasons are a retrospective reconstruction. Actual session clocks and
original publication times are unknown and stay null. A monotone ordering proxy
keeps the engine's time arithmetic working, but never admits practice: early
practice uses the reviewed FP1/FP2/FP3-before-first-Q order instead. Main-Q
results count as available no earlier than a conservative race date + 36 h.
"""

import json
from importlib.resources import files
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from f1_quali.data.build import fetch_evidence, official_warmup, resource
from f1_quali.data.core import Dataset
from f1_quali.data.early import _json_bytes, digest, reached_stage
from f1_quali.data.labels import build_labels
from f1_quali.data.portable import TABLES, load_dataset, validate
from f1_quali.integrity import seal, sha256, verify, write_frame, write_json
from f1_quali.sources.f1db import F1DB

EARLY_YEARS = (2010, 2015)
EARLY_TIME_COLUMNS = (
    "age_at",
    "evidence_at",
    "prediction_cutoff",
    "available_at",
    "actual_start_utc",
    "actual_end_utc",
    "original_publication_timestamp",
)
# Pre-existing warmup admission fields keep an ``original_`` prefix, so they do
# not contradict the wider 2010+ event-phase admission.
ORIGINAL_FLAGS = (
    "minimum_target_season",
    "within_season_eligible",
    "predictor_training_eligible",
    "training_ready",
    "available_at",
    "age_at",
    "evidence_at",
    "age_time_basis",
    "prediction_cutoff_basis",
    "availability_basis",
)
EARLY_TIME_BASIS = "early_event_phase_reconstruction_not_actual_UTC"
EARLY_CUTOFF_BASIS = "compatibility_order_proxy_NOT_actual_UTC_cutoff"
DATETIME_COLUMNS = (
    "prediction_cutoff",
    "label_available_at",
    "entry_known_at",
    "scheduled_start_utc",
    "available_at",
    "actual_start_utc",
    "actual_end_utc",
    "actual_target_start_utc",
    "results_available_upper_bound",
)


def early_limits(year, round_number, qualifying_format, eligible_field_size=None):
    """Reviewed 2010--2015 knockout quotas (Q2 size, Q3 size, evidence)."""
    if qualifying_format != "KNOCKOUT" or not 2010 <= year <= 2015:
        raise ValueError("Unreviewed early format")
    if year <= 2012:
        return 17, 10, "v4_reviewed_2010_2012_Article_33.1"
    if year == 2013:
        return 16, 10, "v7_2013_preseason_Article_33.1"
    if year == 2014:
        q2 = 14 if round_number in (17, 18) else 15 if round_number == 19 else 16
        return q2, 10, "v4_2014_rules_and_v7_final_three_event_reviews"
    return 15, 10, "v4_reviewed_2015_Article_33.1"


def phase_practice(data, event_id, cutoff):
    """Early-only session-order gate; never compares a fabricated UTC time."""
    event = data.events[data.events.event_id.eq(event_id)]
    if len(event) != 1 or not EARLY_YEARS[0] <= int(event.season.iloc[0]) <= EARLY_YEARS[1]:
        raise ValueError("Early phase adapter cannot process a modern event")
    if pd.Timestamp(cutoff) != pd.Timestamp(event.prediction_cutoff.iloc[0]):
        raise ValueError("Phase replay supports before-first-Q only, not arbitrary live cutoffs")
    frame = data.practice[data.practice.event_id.eq(event_id)].copy()
    frame = frame[
        frame.session_type.isin(["FP1", "FP2", "FP3"])
        & frame.pre_first_q_order_reviewed.eq(True)
        & frame.session_phase.lt(4)
        & ~frame.session_cancelled.eq(True)
        & frame.lap_usable
        & frame.lap_seconds.gt(0)
        & np.isfinite(frame.lap_seconds)
        & frame.laps.gt(0)
    ].copy()
    group = frame.groupby("session_id").lap_seconds
    count = group.transform("count")
    frame["percentile"] = (group.rank(method="average") - 1) / (count - 1).clip(lower=1)
    frame.loc[count.lt(2), "percentile"] = 0.5
    return frame.sort_values(["session_phase", "session_id", "driver_id"]).reset_index(drop=True)


def practice_admission(event):
    """Select the practice gate for one event."""
    from f1_quali.features.evidence import eligible_practice

    return phase_practice if int(event.season) <= EARLY_YEARS[1] else eligible_practice


def build_full_labels(data):
    """Labels for all seasons; a cancelled 2015 USA Q3 has no observed Q3 target."""
    early_ids = data.events.loc[data.events.season.le(EARLY_YEARS[1]), "event_id"]
    early, modern = split(data, early_ids)
    if early.events.empty:
        return build_labels(modern)
    result = build_labels(early, limits=early_limits)
    cancelled = result.season.eq(2015) & result["round"].eq(16)
    result.loc[cancelled, "y_q3"] = pd.NA
    result.loc[cancelled, "complete_q3"] = False
    result.loc[cancelled, "complete_all_tasks"] = False
    result.loc[cancelled & result.stage.eq("Q3"), "eligible"] = pd.NA
    result.loc[cancelled & result.stage.eq("Q3"), "complete_stage"] = False
    result.loc[cancelled & result.stage.eq("Q3"), "complete_pace"] = False
    result.loc[cancelled, "quality_flags"] += ";reviewed_2015_USA_Q3_cancelled"
    result["time_basis"] = EARLY_TIME_BASIS
    return pd.concat([result, build_labels(modern)], ignore_index=True)


def split(data, early_ids):
    def part(ids):
        return Dataset(
            *[
                getattr(data, name)[getattr(data, name).event_id.isin(ids)].copy()
                for name in ["events", "entries", "qualifying", "practice", "sessions"]
            ],
            data.drivers,
            data.calendar,
            data.manifest,
        )

    modern_ids = data.events.loc[~data.events.event_id.isin(early_ids), "event_id"]
    return part(set(early_ids)), part(set(modern_ids))


def _roundtrip(frame):
    """Match the canonical CSV representation used by the reviewed early tables."""
    frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
    for column in EARLY_TIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _supplement_source_ids(season, event_id):
    if int(season) == 2013:
        return ["2013_preseason_rules", "2013_official_index"]
    mapping = {
        914: ["2014_usa_report"],
        915: ["2014_brazil_report", "2014_brazil_preliminary_classification"],
        916: ["2014_abu_dhabi_report"],
    }
    if int(event_id) not in mapping:
        raise ValueError("Supplement pair is outside reviewed seasons/events")
    return mapping[int(event_id)]


def admit_early_pace(pairs, results, supplement):
    """Admit the fixed reviewed set of 136 previously unusable early pace pairs.

    Every pair must match exactly one existing pair whose final classifications,
    outcome and warmup eligibility are unchanged; the new stage must follow the
    reviewed rules and the last common stage must already have a valid gap.
    """
    keys = ["event_id", "driver_i", "driver_j"]
    if pairs.duplicated(keys).any():
        raise ValueError("Duplicate early pair identity")
    hashes = {s["source_id"]: s["sha256"] for s in supplement["sources"]}
    reviews = {(r["event_id"], r["driver_id"]): r for r in supplement["participation_reviews"]}
    additions = pd.DataFrame(supplement["admissions"])
    if len(additions) != sum(supplement["expected_added_by_season"].values()):
        raise ValueError("The fixed admitted-pair count changed")
    for column, expected in [
        ("last_common_stage", supplement["expected_added_by_stage"]),
        ("season", supplement["expected_added_by_season"]),
    ]:
        counts = {str(int(k)): int(v) for k, v in additions[column].value_counts().items()}
        if counts != expected:
            raise ValueError("The fixed admitted-pair scope changed")
    archive = supplement["review_archive"]
    result = pairs.copy()
    result["parent_source_version"] = pairs.source_version
    result["parent_source_sha256"] = pairs.source_sha256
    result["admission_status"] = "inherited_v4_admitted_evidence"
    result["new_pace_evidence_admitted"] = False
    result["stage_rule_source_sha256_json"] = "{}"
    result["stage_rule_review_path"] = ""
    result["not_merged"] = False
    for row in additions.itertuples():
        mask = (
            pairs.event_id.eq(row.event_id)
            & pairs.driver_i.eq(row.driver_i)
            & pairs.driver_j.eq(row.driver_j)
        )
        if mask.sum() != 1:
            raise ValueError("Reviewed pair does not match exactly one early pair")
        index = pairs.index[mask][0]
        original = pairs.loc[index]
        if (
            original.season != row.season
            or original.constructor_id != row.constructor_id
            or original.stage_i != row.previous_stage_i
            or original.stage_j != row.previous_stage_j
            or original.last_common_stage != row.previous_last_common_stage
            or bool(original.pace_usable)
            or not bool(original.outcome_usable)
            or not bool(original.rating_warmup_eligible)
        ):
            raise ValueError("Reviewed pace addition changes the original pair identity")
        source_ids = _supplement_source_ids(row.season, row.event_id)
        if set(row.source_ids) != set(source_ids):
            raise ValueError("Pair does not use exactly its reviewed primary rule sources")
        expected_hashes = {key: hashes[key] for key in source_ids}
        event = results[results.event_id.eq(row.event_id)]
        expected_field = 22 if row.season == 2013 else 20 if row.event_id == 916 else 18
        if not event.field_size.eq(expected_field).all():
            raise ValueError("Reviewed field-size rule does not match this event")
        exclusion = event.classification_status.isin(["EX", "DSQ", "DQ", "RT"]).any()
        for suffix, driver in [("i", row.driver_i), ("j", row.driver_j)]:
            driver_rows = event[event.driver_id.eq(driver)]
            if len(driver_rows) != 1 or not bool(driver_rows.label_usable.iloc[0]):
                raise ValueError("No new pace evidence for unresolved/invalid final labels")
            if row.season == 2013:
                stage, basis = reached_stage(driver_rows.iloc[0], 16, 10, exclusion)
            else:
                review = reviews[(row.event_id, driver)]
                stage = int(review["participated_stage"])
                basis = "reviewed_original_event_participation_not_final_position"
            if stage != getattr(row, f"stage_{suffix}") or basis != getattr(
                row, f"stage_basis_{suffix}"
            ):
                raise ValueError("Reviewed stage differs from independently derived stage")
        last = int(row.last_common_stage)
        if last not in [1, 2] or last != min(row.stage_i, row.stage_j):
            raise ValueError("Addition is not the last common participated Q1/Q2 stage")
        gap = original[f"q{last}_gap"]
        if not np.isfinite(gap) or not np.isclose(gap, row.pace_gap, rtol=0, atol=1e-12):
            raise ValueError("Last common stage has no matching valid pace gap")
        for column in ["stage_i", "stage_j", "stage_basis_i", "stage_basis_j"]:
            result.loc[index, column] = getattr(row, column)
        result.loc[index, "last_common_stage"] = last
        result.loc[index, "pace_gap"] = gap
        result.loc[index, "pace_usable"] = True
        flags = str(original.quality_flags).split(";")
        flags = [f for f in flags if f != "last_common_stage_unverified"]
        flags.append("reviewed_stage_rule_pace_admitted_v7")
        if row.event_id == 916:
            flags.append("original_participation_separate_from_final_DSQ_classification")
        result.loc[index, "quality_flags"] = ";".join(flags)
        result.loc[index, "rule_evidence"] = f"{archive}/source_reviews.json:" + ",".join(
            source_ids
        )
        result.loc[index, "source_version"] = original.source_version + "__v7_reviewed_stage_rules"
        result.loc[index, "source_sha256"] = digest(
            _json_bytes(
                {
                    "parent_source_sha256": original.source_sha256,
                    "audit_manifest_sha256": supplement["review_archive_manifest_sha256"],
                    "stage_rule_sources": expected_hashes,
                    "stage_i": int(row.stage_i),
                    "stage_j": int(row.stage_j),
                    "last_common_stage": last,
                }
            )
        )
        result.loc[index, "admission_status"] = "v7_supplement_admitted_2016plus_rating_warmup_only"
        result.loc[index, "new_pace_evidence_admitted"] = True
        result.loc[index, "stage_rule_source_sha256_json"] = json.dumps(
            expected_hashes, sort_keys=True
        )
        result.loc[index, "stage_rule_review_path"] = f"{archive}/source_reviews.json"
    if int(result.new_pace_evidence_admitted.sum()) != len(additions):
        raise ValueError("Every reviewed addition must be admitted exactly once")
    return _roundtrip(result)


def _joined(a, b):
    columns = list(dict.fromkeys([*a, *b]))
    return pd.concat(
        [a.dropna(axis=1, how="all"), b.dropna(axis=1, how="all")], ignore_index=True
    ).reindex(columns=columns)


def early_tables(db, results, practice_columns):
    """Event, entrant, result, practice and session tables for 2010--2015."""
    races = db.table("races")
    races = races[races.year.between(*EARLY_YEARS)].copy()
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
    )
    events = events[
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
    ].sort_values(["season", "round"])
    # Monotone numerical index for the shared engine only; never a session clock.
    events["prediction_cutoff"] = pd.to_datetime(events.race_date, utc=True) + pd.Timedelta(
        hours=12
    )
    events["prediction_cutoff_basis"] = EARLY_CUTOFF_BASIS
    events["cutoff_definition"] = "immediately_before_first_main_Q_segment"
    events["cutoff_phase"] = 4
    events["actual_target_start_utc"] = pd.NaT
    events["result_status"] = "historical_results_present"
    events["source_version"] = "v2026.14.0_plus_reviewed_early_phase_admission"
    events["results_available_upper_bound"] = pd.to_datetime(
        events.race_date, utc=True
    ) + pd.Timedelta(hours=36)
    lookup = events.set_index("event_id")
    q = results.copy()
    q = q.drop(columns=[c for c in ["prediction_cutoff", "circuit_layout_id"] if c in q])
    q["source_row_present"] = q.f1db_source_row_present & q.official_source_row_present
    q["q1_usable"] = q.label_usable & q.q1_seconds.gt(0) & np.isfinite(q.q1_seconds)
    q["label_available_at"] = q.event_id.map(lookup.results_available_upper_bound)
    q["label_availability_basis"] = "reconstructed_race_date_plus_36h_not_publication"
    q["classification_status"] = q.classification_status.astype(str)
    numeric = q.label_usable
    q.loc[numeric, "classification_status"] = q.loc[numeric, "position"].astype(int).astype(str)
    q["missing_reason"] = np.where(
        q.label_usable, "", "reviewed_early_missing_or_conflicting_label"
    )
    q["new_admission_scope"] = "early_phase_replay_with_conservative_between_event_availability"
    q = q.rename(columns={c: "original_" + c for c in ORIGINAL_FLAGS if c in q})
    entries = results[["season", "round", "event_id", "driver_id", "constructor_id"]].copy()
    entries["prediction_cutoff"] = entries.event_id.map(lookup.prediction_cutoff)
    entries["entry_basis"] = "retrospective_reviewed_actual_roster_NOT_original_entry_snapshot"
    entries["entry_reference"] = "v4_reviewed_Q_race_grid_identity_union"
    entries["entry_known_at"] = pd.NaT
    entries["constructor_entry_count"] = entries.groupby(
        ["event_id", "constructor_id"]
    ).driver_id.transform("size")
    if not entries.constructor_entry_count.le(2).all():
        raise ValueError("More than two early teammates")
    entries["roster_status"] = np.where(
        entries.constructor_entry_count.eq(2), "resolved_pair", "documented_single_entry"
    )
    entries["entry_time_basis"] = "retrospective_identity_original_publication_unknown"
    keys = pd.MultiIndex.from_frame(entries[["event_id", "driver_id", "constructor_id"]])
    practice = []
    for n in (1, 2, 3):
        raw = db.table(f"races-free-practice-{n}-results")
        p = raw[raw.raceId.isin(events.event_id)].rename(
            columns={
                "raceId": "event_id",
                "year": "season",
                "driverId": "driver_id",
                "constructorId": "constructor_id",
                "driverNumber": "driver_number",
                "positionNumber": "position",
                "positionText": "classification_status",
            }
        )
        p = p.copy()
        p["session_type"], p["session_phase"] = f"FP{n}", n
        p["session_id"] = p.event_id.astype(str) + f":FP{n}"
        p["lap_seconds"] = pd.to_numeric(p.timeMillis, errors="coerce") / 1000
        p["lap_usable"] = p.lap_seconds.gt(0) & np.isfinite(p.lap_seconds) & p.laps.gt(0)
        p["missing_reason"] = np.where(p.lap_usable, "", "no_positive_time_or_lap_count")
        p["pre_first_q_order_reviewed"] = True
        p["session_cancelled"] = p.season.eq(2015) & p["round"].eq(16) & p.session_type.eq("FP2")
        if p.session_cancelled.any():
            raise ValueError("Recorded FP2 contradicts reviewed 2015 USA cancellation")
        p["scheduled_start_utc"], p["available_at"] = pd.NaT, pd.NaT
        p["time_source"] = "reviewed_rules_relative_session_order_actual_UTC_unknown"
        p["availability_basis"] = "completed_FP_phase_before_first_Q_not_a_UTC_publication_claim"
        p["schedule_conflict"] = False
        p["is_event_entry"] = pd.MultiIndex.from_frame(
            p[["event_id", "driver_id", "constructor_id"]]
        ).isin(keys)
        p["source_version"] = db.version
        columns = [*practice_columns, "session_phase", "pre_first_q_order_reviewed"]
        practice.append(p.reindex(columns=list(dict.fromkeys([*columns, "session_cancelled"]))))
    sessions = []
    for event in events.itertuples():
        for n, kind in enumerate(["FP1", "FP2", "FP3", "Q"], 1):
            cancelled = event.season == 2015 and event.round == 16 and kind == "FP2"
            sessions.append(
                {
                    "event_id": event.event_id,
                    "season": event.season,
                    "round": event.round,
                    "session_id": f"{event.event_id}:{kind}",
                    "session_type": kind,
                    "session_phase": n,
                    "scheduled_start_utc": pd.NaT,
                    "available_at": event.results_available_upper_bound if kind == "Q" else pd.NaT,
                    "actual_start_utc": pd.NaT,
                    "actual_end_utc": pd.NaT,
                    "session_status": "cancelled" if cancelled else "historical_phase_order_only",
                    "time_source": "rules_and_archived_exception_review_no_actual_clock",
                    "schedule_conflict": False,
                }
            )
    return {
        "events": events,
        "entries": entries,
        "qualifying": q,
        "practice": pd.concat(practice, ignore_index=True),
        "sessions": pd.DataFrame(sessions),
    }


def early_pair_timing(pairs, events):
    """Race-date +36 h is both the age proxy and the conservative availability."""
    lookup = events.set_index("event_id")
    pairs = pairs.rename(columns={c: "original_" + c for c in ORIGINAL_FLAGS if c in pairs})
    pairs["age_at"] = pairs.event_id.map(lookup.results_available_upper_bound)
    pairs["available_at"] = pairs.age_at
    pairs["prediction_cutoff"] = pairs.event_id.map(lookup.prediction_cutoff)
    pairs["evidence_at"] = pairs.age_at
    pairs["minimum_target_season"] = EARLY_YEARS[0]
    pairs["age_time_basis"] = "race_date_plus_36h_age_proxy"
    pairs["availability_basis"] = "conservative_between_event_reconstruction_not_publication"
    pairs["prediction_cutoff_basis"] = EARLY_CUTOFF_BASIS
    pairs["new_admission_scope"] = "reviewed_2010plus_relative_phase_replay"
    return pairs


def build_full_era(canonical, cache, output, *, offline=False):
    """Extend a canonical dataset (``f1-quali data``) to a sealed 2010--2026 dataset."""
    cache = Path(cache)
    modern, _ = load_dataset(canonical)
    db = F1DB(cache, resource("sources")["f1db"], offline=offline)
    pairs, _, _, results = official_warmup(db, cache, offline=offline, early_tables=True)
    supplement = json.loads(
        files("f1_quali").joinpath("resources/early_pace_supplement.json").read_text()
    )
    for source in supplement["sources"]:
        fetch_evidence(cache, source, offline)
    pairs = admit_early_pace(pairs, results, supplement)
    early = early_tables(db, results, list(modern.practice.columns))
    pairs = early_pair_timing(pairs, early["events"])
    held = set(
        modern.events.loc[modern.events.result_status.eq("historical_results_present"), "event_id"]
    )
    final = {}
    for name in ["events", "entries", "qualifying", "practice", "sessions"]:
        frame = getattr(modern, name)
        final[name] = _joined(early[name], frame[frame.event_id.isin(held)].copy())
    final["drivers"] = modern.drivers
    final["calendar"] = final["events"].copy()
    for frame in final.values():
        for column in DATETIME_COLUMNS:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
    provenance = {
        **modern.manifest,
        "kind": "full_era_reconstruction",
        "seasons": [EARLY_YEARS[0], int(modern.events.season.max())],
        "early_scope": "retrospective_phase_order_only_actual_clocks_unknown",
        "early_supplement": supplement["schema"],
        "modern_dataset_manifest_sha256": sha256((Path(canonical) / "manifest.json").read_bytes()),
    }
    data = Dataset(*[final[name] for name in TABLES], provenance)
    from f1_quali.data.experience import from_f1db

    return save_full_era(output, data, pairs, experience=from_f1db(db))


def save_full_era(directory, data, early_pairs, *, experience=None):
    """Seal a 2010+ dataset; ``early_pairs`` holds 2010--2015 teammate evidence."""
    validate(data)
    directory = Path(directory)
    if len(early_pairs) and not early_pairs.season.le(EARLY_YEARS[1]).all():
        raise ValueError("Early teammate evidence must precede 2016")
    for name in TABLES:
        write_frame(directory / f"{name}.parquet", getattr(data, name))
    write_frame(directory / "early_pairs.parquet", early_pairs)
    coverage = []
    for year, group in data.events.groupby("season"):
        ids = set(group.event_id)
        qual = data.qualifying[data.qualifying.event_id.isin(ids)]
        practice = data.practice[data.practice.event_id.isin(ids)]
        coverage.append(
            {
                "season": int(year),
                "events": len(group),
                "entrants": int(data.entries.event_id.isin(ids).sum()),
                "usable_final_labels": int(qual.label_usable.sum()),
                "practice_rows": len(practice),
                "usable_practice_rows": int(practice.lap_usable.sum()),
                "phase_order_only": bool(year <= EARLY_YEARS[1]),
            }
        )
    write_json(directory / "provenance.json", data.manifest)
    write_json(directory / "coverage.json", coverage)
    if experience is not None:
        from f1_quali.data.experience import save

        save(directory, *experience)
    return seal(
        directory,
        kind="dataset",
        metadata={
            "warmup": False,
            "full_era": True,
            "early_events": int(data.events.season.le(EARLY_YEARS[1]).sum()),
            "events": len(data.events),
            "early_pace_pairs": int(early_pairs.pace_usable.sum()) if len(early_pairs) else 0,
            "early_actual_clocks_unknown": True,
        },
    )


def load_full_era(directory):
    """Load a sealed 2010+ dataset and its early teammate evidence."""
    manifest = verify(directory, "dataset")
    if not manifest["metadata"].get("full_era"):
        raise ValueError("Expected a full-era dataset (run `f1-quali full-era`)")
    data, _ = load_dataset(directory)
    return data, pd.read_parquet(Path(directory) / "early_pairs.parquet")
