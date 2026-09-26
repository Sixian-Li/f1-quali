"""Source-referenced session, entry and qualifying corrections."""

import numpy as np
import pandas as pd

from f1_quali.data.canonical import INELIGIBLE_QUALIFYING_STATUSES, assert_unique
from f1_quali.sources.schedules import schedule_candidate


def mark_calendar_absences(
    sessions: pd.DataFrame, races: pd.DataFrame, schedules: dict
) -> pd.DataFrame:
    """An absent slot in a matched, pinned calendar is different from a cancellation."""
    out = sessions.copy()
    for race in races.to_dict("records"):
        candidate = schedule_candidate(schedules.get(race["year"]), race["date"])
        if not candidate:
            continue
        mask = (
            (out.event_id == race["id"])
            & out.session_type.isin(["FP1", "FP2", "FP3"])
            & ~out.session_type.isin(candidate)
            & out.scheduled_start_utc.isna()
            & out.session_status.eq("scheduled_reconstructed")
        )
        out.loc[mask, "session_status"] = "not_scheduled_in_pinned_calendar"
        out.loc[mask, "available_at"] = pd.NaT
        out.loc[mask, "availability_basis"] = "unavailable_session"
        out.loc[mask, "time_review_status"] = "matched_pinned_calendar_absence"
        out.loc[mask, "schedule_reference"] = f"f1schedule:schedule_{race['year']}.json"
    return out


def review_sessions(sessions: pd.DataFrame, rules: list[dict], buffer_minutes: int) -> pd.DataFrame:
    if buffer_minutes < 0 or any(r.get("post_session_buffer_minutes", 30) < 0 for r in rules):
        raise ValueError("Availability buffer cannot be negative")
    out = sessions.copy()
    out["original_scheduled_start_utc"] = out.scheduled_start_utc
    out["original_schedule_conflict"] = out.schedule_conflict
    out["scheduled_end_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["time_review_status"] = "unverified_actual_times"
    out["evidence_key"] = ""
    out["timing_note"] = ""
    out["session_status"] = "scheduled_reconstructed"
    out["publication_time_status"] = "original_publication_unknown"
    for rule in rules:
        mask = (
            (out.season == rule["season"])
            & (out["round"] == rule["round"])
            & (out.session_type == rule["session_type"])
        )
        if mask.sum() != 1:
            raise ValueError(f"Unknown or duplicate reviewed session: {rule}")
        out.loc[mask, "evidence_key"] = rule["evidence_key"]
        out.loc[mask, "time_review_status"] = rule["review_status"]
        out.loc[mask, "timing_note"] = rule.get("note", "")
        out.loc[mask, "schedule_reference"] = rule["source"]
        for field in [
            "scheduled_start_utc",
            "scheduled_end_utc",
            "actual_start_utc",
            "actual_end_utc",
        ]:
            if rule.get(field):
                out.loc[mask, field] = pd.Timestamp(rule[field])
        if rule.get("scheduled_start_utc"):
            out.loc[mask, "time_source"] = "reviewed_primary_document"
            out.loc[mask, "schedule_conflict"] = False
        if rule.get("actual_end_utc"):
            if not rule.get("actual_start_utc"):
                raise ValueError("Verified end requires a verified start")
            if pd.Timestamp(rule["actual_end_utc"]) < pd.Timestamp(rule["actual_start_utc"]):
                raise ValueError("Session end precedes start")
            out.loc[mask, "available_at"] = pd.Timestamp(rule["actual_end_utc"]) + pd.Timedelta(
                minutes=rule.get("post_session_buffer_minutes", 30)
            )
            out.loc[mask, "availability_basis"] = "observed_end_plus_buffer_reconstruction"
        elif rule.get("scheduled_start_utc"):
            out.loc[mask, "available_at"] = pd.Timestamp(
                rule["scheduled_start_utc"]
            ) + pd.Timedelta(minutes=buffer_minutes)
        if rule.get("session_status"):
            out.loc[mask, "session_status"] = rule["session_status"]
        if rule.get("session_status") in {"cancelled", "not_scheduled"}:
            out.loc[mask, "available_at"] = pd.NaT
            out.loc[mask, "availability_basis"] = "unavailable_session"
    out["time_quality_flags"] = np.where(
        (out.actual_start_utc.isna() | out.actual_end_utc.isna()),
        "actual_start_end_unverified",
        "publication_time_unverified",
    )
    out.loc[out.scheduled_start_utc.isna(), "time_quality_flags"] += ";scheduled_time_unknown"
    out.loc[out.schedule_conflict, "time_quality_flags"] += ";unresolved_schedule_conflict"
    return out


def review_entries(entries: pd.DataFrame, events: pd.DataFrame, rules: list[dict]) -> tuple:
    """Apply event-scoped announcements without accepting qualifying results as an input."""
    out = entries.copy()
    out["entry_evidence_key"] = ""
    out["entry_time_basis"] = "retrospective_membership_original_publication_unknown"
    changes = []
    single_entries = set()
    for rule in rules:
        selected = events[(events.season == rule["season"]) & (events["round"] == rule["round"])]
        if len(selected) != 1:
            raise ValueError(f"Entry review event not unique: {rule}")
        event = selected.iloc[0]
        known = pd.Timestamp(rule["effective_at"])
        if pd.isna(event.prediction_cutoff) or known > event.prediction_cutoff:
            continue
        eid, constructor = event.event_id, rule["constructor_id"]
        mask = (out.event_id == eid) & (out.constructor_id == constructor)
        removed = rule.get("remove_driver_id")
        if removed:
            out = out[~(mask & (out.driver_id == removed))].copy()
        added = rule.get("replacement_driver_id")
        if added:
            other_team = out[
                (out.event_id == eid)
                & (out.driver_id == added)
                & (out.constructor_id != constructor)
            ]
            if not other_team.empty:
                raise ValueError(f"Replacement already entered for another constructor: {rule}")
            out = out[~((out.event_id == eid) & (out.driver_id == added))].copy()
            row = {
                "event_id": eid,
                "season": event.season,
                "round": event["round"],
                "prediction_cutoff": event.prediction_cutoff,
                "driver_id": added,
                "constructor_id": constructor,
                "entry_basis": "reviewed_event_announcement",
                "entry_reference": rule["source"],
                "entry_known_at": known,
                "entry_evidence_key": rule["evidence_key"],
                "entry_time_basis": rule["time_basis"],
                "constructor_entry_count": 0,
            }
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        if rule.get("allow_single_entry"):
            single_entries.add((eid, constructor))
        changes.append({"event_id": eid, **rule})
    assert_unique(out, ["event_id", "driver_id"], "reviewed_entries")
    out["constructor_entry_count"] = out.groupby(
        ["event_id", "constructor_id"]
    ).driver_id.transform("size")
    out["roster_status"] = "resolved_pair"
    single_mask = out.apply(lambda r: (r.event_id, r.constructor_id) in single_entries, axis=1)
    out.loc[single_mask, "roster_status"] = "documented_single_entry"
    invalid = ~(
        (out.constructor_entry_count == 2) | ((out.constructor_entry_count == 1) & single_mask)
    )
    if invalid.any():
        ids = out.loc[invalid, ["season", "round", "constructor_id"]].drop_duplicates()
        raise ValueError(f"Unresolved roster:\n{ids.to_string(index=False)}")
    return out.sort_values(["season", "round", "driver_id"]).reset_index(drop=True), pd.DataFrame(
        changes
    )


def review_qualifying(q: pd.DataFrame, entries: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    out = q.copy()
    # CSV can infer numeric dtype when every source row happens to have a numeric status.
    out["classification_status"] = out.classification_status.astype("string")
    out["source_row_present"] = True
    out["classification_review_status"] = "source_only"
    out["classification_evidence_key"] = ""
    out["classification_note"] = ""
    out["decision_signed_at"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["lap_deletion_status"] = "not_audited_in_summary_source"
    # Preserve entrants with no source row. Never turn them into last-place numeric labels.
    roster = entries[["event_id", "season", "round", "driver_id", "constructor_id"]]
    missing = roster.merge(
        out[["event_id", "driver_id"]], how="left", on=["event_id", "driver_id"], indicator=True
    )
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    if not missing.empty:
        missing = missing.reindex(columns=out.columns)
        missing["source_row_present"] = False
        missing["source_variant_count"] = 0
        missing["classification_status"] = "SOURCE_ROW_ABSENT"
        missing["missing_reason"] = "source_row_absent_participation_unverified"
        missing["classification_review_status"] = "missing_source_row"
        missing["classification_evidence_key"] = ""
        missing["classification_note"] = ""
        missing["decision_signed_at"] = pd.Series(
            pd.NaT, index=missing.index, dtype="datetime64[ns, UTC]"
        )
        missing["lap_deletion_status"] = "not_audited_in_summary_source"
        missing["session_type"] = "Q"
        missing["source_version"] = out.source_version.iloc[0]
        # Match boolean dtypes before concatenation (pandas does not infer missing booleans).
        missing["label_usable"] = False
        missing["q1_usable"] = False
        out = pd.concat([out, missing], ignore_index=True)
    for rule in rules:
        mask = (
            (out.season == rule["season"])
            & (out["round"] == rule["round"])
            & (out.driver_id == rule["driver_id"])
        )
        if mask.sum() != 1:
            raise ValueError(f"Classification review requires one normalized row: {rule}")
        out.loc[mask, "classification_status"] = rule["classification_status"]
        out.loc[mask, "classification_review_status"] = "resolved_primary_document"
        out.loc[mask, "classification_evidence_key"] = rule["evidence_key"]
        out.loc[mask, "classification_note"] = rule["reason"]
        out.loc[mask, "missing_reason"] = rule["reason"]
        if rule.get("decision_signed_at"):
            out.loc[mask, "decision_signed_at"] = pd.Timestamp(rule["decision_signed_at"])
        if rule["classification_status"] in INELIGIBLE_QUALIFYING_STATUSES:
            out.loc[mask, ["position", "q1_seconds", "q2_seconds", "q3_seconds"]] = np.nan
    out["label_usable"] = (
        out.position.gt(0)
        & np.isfinite(out.position)
        & out.position.mod(1).eq(0)
        & ~out.classification_status.isin(INELIGIBLE_QUALIFYING_STATUSES)
    )
    out["q1_usable"] = (
        out.q1_seconds.gt(0)
        & np.isfinite(out.q1_seconds)
        & out.label_usable
        & ~out.classification_status.isin(INELIGIBLE_QUALIFYING_STATUSES)
    )
    out["quality_flags"] = ""
    out.loc[~out.source_row_present, "quality_flags"] = "missing_source_row"
    out.loc[out.classification_status.isin(["NC", "DNQ", "DNS"]), "quality_flags"] = (
        "non_numeric_classification;participation_or_107pct_review"
    )
    out.loc[out.classification_status.isin(["DSQ", "DQ", "EX"]), "quality_flags"] = (
        "disqualified;excluded_from_pace_evidence"
    )
    out.loc[~out.label_usable & out.q1_seconds.gt(0), "quality_flags"] += (
        ";time_without_rank_excluded_pending_review"
    )
    out.loc[out.classification_status.isin(["NOT_PARTICIPATED", "NO_TIME"]), "quality_flags"] += (
        ";participation_explained_primary_report;no_numeric_label_inferred"
    )
    out["quality_flags"] += ";lap_deletions_unverified"
    assert_unique(out, ["event_id", "driver_id"], "full_qualifying")
    return out.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)


def add_session_entries(tables: dict) -> pd.DataFrame:
    practice = tables["practice_results"]
    fp = practice[
        [
            "event_id",
            "session_id",
            "session_type",
            "driver_id",
            "constructor_id",
            "driver_number",
            "is_event_entry",
        ]
    ].copy()
    fp["entry_basis"] = "observed_practice_result_participant_not_complete_entry_list"
    fp["participation_status"] = "source_row_present"
    q = tables["entries"][["event_id", "driver_id", "constructor_id", "entry_basis"]].copy()
    q["session_id"] = q.event_id.astype(str) + ":Q"
    q["session_type"] = "Q"
    q["is_event_entry"] = True
    q = q.merge(
        tables["qualifying_results"][
            [
                "event_id",
                "driver_id",
                "driver_number",
                "source_row_present",
                "classification_status",
            ]
        ],
        on=["event_id", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    q["participation_status"] = np.where(
        q.source_row_present.eq(True), "source_row_present", "entered_participation_unverified"
    )
    reviewed = q.classification_status.isin(["NOT_PARTICIPATED", "NO_TIME"])
    q.loc[reviewed, "participation_status"] = q.loc[reviewed, "classification_status"]
    out = pd.concat(
        [fp, q.drop(columns=["source_row_present", "classification_status"])], ignore_index=True
    )
    assert_unique(out, ["session_id", "driver_id"], "session_entries")
    return out.sort_values(["event_id", "session_type", "driver_id"]).reset_index(drop=True)
