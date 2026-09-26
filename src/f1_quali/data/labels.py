"""Main qualifying stage eligibility, labels and teammate evidence."""

import numpy as np
import pandas as pd

from f1_quali.data.validation import _positive_time, _valid_label, participation

STAGES = ("Q1", "Q2", "Q3")


TASKS = ("q2", "q3", "top3", "pole")


EXCLUSIONS = {"EX", "DSQ", "DQ"}


def stage_limits(season, round_number, qualifying_format, eligible_field_size=None):
    """Return source-backed quotas, using eligible cars, never timed cars.

    Historical 2016 and 2017--25 championship sizes are fixed by their registered
    rules, including a race with only 19 actual entries. For 2026 callers may pass
    the officially eligible 20/22/24-car field; absent that information it is 22.
    """
    if season == 2016 and round_number <= 2:
        if qualifying_format != "ELIMINATION":
            raise ValueError("2016 elimination identity conflict")
        return 15, 8, "fia_2016_elimination"
    if qualifying_format not in {"KNOCKOUT", "SPRINT_RACE"}:
        raise ValueError(f"Unreviewed qualifying format: {qualifying_format}")
    if season == 2016:
        return 16, 10, "fia_2016_restoration"
    if 2017 <= season <= 2025:
        return 15, 10, "fia_2017_sporting;fia_2024_sporting"
    if season == 2026:
        field = 22 if eligible_field_size is None or pd.isna(eligible_field_size) else int(
            eligible_field_size
        )
        if field not in {20, 22, 24}:
            raise ValueError("2026 eligible field requires reviewed 20/22/24-car rule")
        eliminated = (field - 10) // 2
        return field - eliminated, 10, "fia_2026_sporting_issue08"
    raise ValueError(f"Unreviewed qualifying rule year: {season}")


def _checked_events(data):
    events = (
        data.events[data.events.result_status.eq("historical_results_present")]
        .sort_values(["season", "prediction_cutoff", "event_id"])
        .copy()
    )
    if events.event_id.duplicated().any() or events.prediction_cutoff.isna().any():
        raise ValueError("Unique held events with explicit cutoff required")
    events["season_event_ordinal"] = events.groupby("season").cumcount() + 1
    q = data.qualifying[data.qualifying.event_id.isin(events.event_id)].copy()
    if not q.session_type.eq("Q").all():
        raise ValueError("Main qualifying identity required")
    if q.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate result identity")
    if data.entries.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate entrant identity")
    return events, q


def _eligibility(row, size, limits, exclusion):
    # Entitlement to Q1 comes from the entrant roster, including DNS and no-time.
    eligible = [True, pd.NA, pd.NA]
    if row is None or not bool(row.get("source_row_present", False)):
        return eligible, "missing_source_row"
    if _valid_label(row, size) and not exclusion:
        reached, basis = participation(row, size, *limits, False)
        return [True, reached >= 2, reached >= 3], basis
    if row.classification_status == "NOT_PARTICIPATED":
        return [True, False, False], "source_explicit_not_participated"
    # Positive Q2 proves Q2 eligibility, but its absent Q3 time proves nothing.
    highest = max([s for s in (1, 2, 3) if _positive_time(row, s)], default=0)
    for index in range(1, highest):
        eligible[index] = True
    return eligible, "positive_time_only_classification_stage_ambiguous"


def build_labels(data):
    """Create event/driver/stage rows with nullable eligibility and task flags.

    ``complete_pace`` means all eligible entrants have an interpretable timing
    outcome (valid time or a known no-time), not that all have a valid time.
    Invalid final labels and source gaps remain unknown; they never become zeros.
    Every consumer must apply its own annual cutoff and first-five exclusion.
    """
    events, q = _checked_events(data)
    lookup = {(r.event_id, r.driver_id): r for _, r in q.iterrows()}
    records = []
    for _, event in events.iterrows():
        roster = data.entries[data.entries.event_id.eq(event.event_id)].sort_values("driver_id")
        size = len(roster)
        if size < 2:
            raise ValueError("Complete roster required")
        event_q = q[q.event_id.eq(event.event_id)]
        if not set(event_q.driver_id).issubset(set(roster.driver_id)):
            raise ValueError("Result outside actual entrant roster")
        exclusion = bool(event_q.classification_status.isin(EXCLUSIONS).any())
        q2, q3, evidence = stage_limits(
            int(event.season),
            int(event["round"]),
            event.qualifying_format,
            event.get("eligible_field_size"),
        )
        for entrant in roster.itertuples():
            row = lookup.get((event.event_id, entrant.driver_id))
            if row is not None and row.constructor_id != entrant.constructor_id:
                raise ValueError("Constructor identity differs from actual entrant")
            final_usable = _valid_label(row, size)
            eligibility, basis = _eligibility(row, size, (q2, q3), exclusion)
            source_present = row is not None and bool(row.get("source_row_present", False))
            available = row.label_available_at if row is not None else pd.NaT
            for stage_number, stage in enumerate(STAGES, 1):
                eligible = eligibility[stage_number - 1]
                positive = _positive_time(row, stage_number)
                # Preserve the canonical invalid-classification safeguards. A
                # recorded positive value alone is not a certified training lap.
                valid = bool(final_usable and positive and eligible is not pd.NA and eligible)
                outcome_known = bool(
                    final_usable
                    and source_present
                    and pd.notna(available)
                    and eligible is not pd.NA
                    and eligible
                )
                no_time = not positive if outcome_known else pd.NA
                reasons = []
                if not source_present:
                    reasons.append("missing_source_row")
                if not final_usable:
                    reasons.append("invalid_or_unavailable_final_label")
                if eligible is pd.NA:
                    reasons.append("stage_eligibility_unknown")
                elif not eligible:
                    reasons.append("not_eligible_no_counterfactual_lap_label")
                elif outcome_known and not positive:
                    reasons.append("eligible_without_valid_lap")
                if exclusion:
                    reasons.append("post_exclusion_stage_ambiguity")
                records.append(
                    {
                        "event_id": int(event.event_id),
                        "season": int(event.season),
                        "round": int(event["round"]),
                        "season_event_ordinal": int(event.season_event_ordinal),
                        "driver_id": entrant.driver_id,
                        "constructor_id": entrant.constructor_id,
                        "circuit_layout_id": event.circuit_layout_id,
                        "session_type": "Q",
                        "stage": stage,
                        "stage_number": stage_number,
                        "field_size": size,
                        "q2_quota": q2,
                        "q3_quota": q3,
                        "eligible": eligible,
                        "observed_positive_lap": positive,
                        "valid_lap": valid,
                        "no_valid_lap": no_time,
                        "lap_seconds": float(row[f"q{stage_number}_seconds"]) if valid else np.nan,
                        "final_position": float(row.position) if final_usable else np.nan,
                        "final_label_usable": final_usable,
                        "y_q2": eligibility[1],
                        "y_q3": eligibility[2],
                        "y_top3": bool(row.position <= 3) if final_usable else pd.NA,
                        "y_pole": bool(row.position == 1) if final_usable else pd.NA,
                        "prediction_cutoff": event.prediction_cutoff,
                        "label_available_at": available,
                        "source_row_present": source_present,
                        "stage_evidence": basis,
                        "rule_evidence": evidence,
                        "classification_evidence": row.get("classification_evidence_key", "")
                        if row is not None
                        else "",
                        "source_version": event.source_version,
                        "missing_reason": ";".join(reasons),
                        "quality_flags": "summary_conditions_lap_deletions_unverified;"
                        "reconstructed_availability_not_original_publication",
                        "normal_format_supported": not exclusion,
                    }
                )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    for col in ["eligible", "no_valid_lap", *[f"y_{t}" for t in TASKS]]:
        out[col] = out[col].astype("boolean")
    group = out.groupby(["event_id", "stage"], sort=False)
    out["stage_median_seconds"] = group.lap_seconds.transform("median")
    out["relative_gap"] = 100 * np.log(out.lap_seconds / out.stage_median_seconds)
    out["complete_stage"] = group.eligible.transform(lambda s: s.notna().all()).astype(bool)
    known = (~out.eligible.fillna(True)) | out.no_valid_lap.notna()
    out["complete_pace"] = (
        known.groupby([out.event_id, out.stage]).transform("all") & out.complete_stage
    )
    event_availability = out.groupby("event_id").label_available_at.transform("max")
    out["training_label_available_at"] = event_availability
    for task in TASKS:
        out[f"complete_{task}"] = (
            out.groupby("event_id")[f"y_{task}"].transform(lambda s: s.notna().all()).astype(bool)
        )
    final = event_labels(out)
    complete = {}
    for eid, part in final.groupby("event_id", sort=False):
        complete[eid] = bool(
            part.final_label_usable.all()
            and set(part.final_position) == set(range(1, len(part) + 1))
        )
        for task, quota in [
            ("q2", int(part.q2_quota.iloc[0])),
            ("q3", int(part.q3_quota.iloc[0])),
            ("top3", 3),
            ("pole", 1),
        ]:
            valid_task = bool(part[f"complete_{task}"].iloc[0])
            if valid_task and int(part[f"y_{task}"].sum()) != min(quota, len(part)):
                out.loc[out.event_id.eq(eid), f"complete_{task}"] = False
    out["complete_final"] = out.event_id.map(complete).astype(bool)
    out["complete_top3"] &= out.complete_final
    out["complete_pole"] &= out.complete_final
    out["complete_all_tasks"] = out[[f"complete_{task}" for task in TASKS]].all(axis=1)
    return out.sort_values(["season", "round", "driver_id", "stage_number"]).reset_index(drop=True)


def event_labels(labels):
    """One row per entrant, avoiding accidental triple weighting of pool targets."""
    return labels[labels.stage.eq("Q1")].reset_index(drop=True).copy()


def build_rating_observations(data, rating_config):
    """The frozen research teammate semantics, with the registered 2026 rule added."""
    events, q = _checked_events(data)
    lookup = {(r.event_id, r.driver_id): r for _, r in q.iterrows()}
    records = []
    for _, event in events.iterrows():
        roster = data.entries[data.entries.event_id.eq(event.event_id)]
        size = len(roster)
        if size < 2:
            raise ValueError("Complete roster required")
        event_q = q[q.event_id.eq(event.event_id)]
        if not set(event_q.driver_id).issubset(set(roster.driver_id)):
            raise ValueError("Result outside roster")
        exclusion = event_q.classification_status.isin(EXCLUSIONS).any()
        limits = stage_limits(
            int(event.season),
            int(event["round"]),
            event.qualifying_format,
            event.get("eligible_field_size"),
        )
        for team, entries in roster.groupby("constructor_id", sort=True):
            if len(entries) > 2:
                raise ValueError("More than two actual teammates")
            if len(entries) < 2:
                continue
            ids = sorted(entries.driver_id)
            a, b = (lookup.get((event.event_id, d)) for d in ids)
            if any(r is not None and r.constructor_id != team for r in (a, b)):
                raise ValueError("Constructor identity conflict")
            valid = bool(_valid_label(a, size) and _valid_label(b, size))
            valid = bool(valid and a.position != b.position) if valid else False
            si, bi = participation(a, size, *limits[:2], exclusion)
            sj, bj = participation(b, size, *limits[:2], exclusion)
            last = min(si, sj) if si and sj else 0
            if _positive_time(a, 3) and _positive_time(b, 3):
                last = 3
            reasons = [] if valid else ["invalid_or_missing_result_or_availability"]
            if not last:
                reasons.append("last_common_stage_unverified_after_classification_change")
            gaps = {}
            for stage in (1, 2, 3):
                gap = np.nan
                if valid and _positive_time(a, stage) and _positive_time(b, stage):
                    value = 100 * np.log(b[f"q{stage}_seconds"] / a[f"q{stage}_seconds"])
                    if abs(value) <= rating_config["maximum_absolute_gap_log_pct"]:
                        gap = float(value)
                    else:
                        reasons.append(f"q{stage}_gap_exceeds_fixed_quality_threshold")
                gaps[f"q{stage}_gap"] = gap
            pace = gaps.get(f"q{last}_gap", np.nan)
            if last and pd.isna(pace):
                reasons.append("last_common_stage_has_no_eligible_pace_no_downgrade")
            records.append(
                {
                    "event_id": int(event.event_id),
                    "season": int(event.season),
                    "round": int(event["round"]),
                    "season_event_ordinal": int(event.season_event_ordinal),
                    "session_type": "Q",
                    "circuit_layout_id": event.circuit_layout_id,
                    "constructor_id": team,
                    "driver_i": ids[0],
                    "driver_j": ids[1],
                    "field_size": size,
                    "prediction_cutoff": event.prediction_cutoff,
                    "available_at": max(a.label_available_at, b.label_available_at)
                    if valid
                    else pd.NaT,
                    "outcome_usable": valid,
                    "outcome": int(np.sign(b.position - a.position)) if valid else 0,
                    "rank_gap": float(b.position - a.position) if valid else np.nan,
                    "rank_gap_abs_normalized": abs(b.position - a.position) / (size - 1)
                    if valid
                    else np.nan,
                    "stage_i": si,
                    "stage_j": sj,
                    "stage_basis_i": bi,
                    "stage_basis_j": bj,
                    "last_common_stage": last,
                    "pace_gap": pace,
                    **gaps,
                    "pace_usable": bool(np.isfinite(pace)),
                    "rule_evidence": limits[2],
                    "source_version": event.source_version,
                    "quality_flags": ";".join(
                        reasons
                        + [
                            "summary_conditions_lap_deletions_unverified",
                            "reconstructed_availability_not_original_publication",
                        ]
                    ),
                }
            )
    return pd.DataFrame(records)
