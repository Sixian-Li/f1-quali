"""Reviewed 2010–2015 rating warmup; never predictor training rows."""

import hashlib
import json

import numpy as np
import pandas as pd

ARCHIVE_SHA256 = "8a92b898bc237bd3b0a86fe5665ebe0e8f8328f126e2422ca9946d175d66dfb4"


LIMITATIONS = (
    "reconstructed_season_end_availability_not_original_publication;"
    "race_date_upper_bound_is_age_proxy_not_actual_session_time;"
    "post_event_roster_not_pre_qualifying_entry_snapshot;"
    "summary_conditions_lap_deletions_unverified"
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def review_differences(differences, reviews):
    """Every observed field difference must exactly match one explicit decision."""
    keys = ["race_id", "year", "driver_id", "field", "f1db", "official"]
    expected = pd.DataFrame(reviews["difference_reviews"])

    def normalized(frame):
        return sorted(tuple(str(row[k]) for k in keys) for row in frame.to_dict("records"))

    if normalized(differences) != normalized(expected):
        raise ValueError("Unreviewed or changed official classification difference")
    if expected.duplicated(["race_id", "driver_id", "field"]).any():
        raise ValueError("Duplicate classification review")
    allowed = {
        "exclude_numeric_conflict",
        "retain_non_numeric_classification",
        "retain_raw_but_exclude_missing_official_row",
    }
    if not expected.decision.isin(allowed).all() or expected.reason.eq("").any():
        raise ValueError("Classification review lacks an explicit supported decision")
    return expected


def reconstructed_times(year, race_date):
    """Use a date upper bound for age; never pretend it is an actual Q cutoff."""
    date = pd.Timestamp(race_date)
    if date.tzinfo is not None or date != date.normalize() or date.year != int(year):
        raise ValueError("Race-date age proxy requires an unambiguous same-year local date")
    age = date.tz_localize("UTC") + pd.Timedelta(hours=36)
    available = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC")
    if age >= available:
        raise ValueError("Season-end availability does not follow the evidence age bound")
    return {
        "age_at": age,
        "evidence_at": age,
        "prediction_cutoff": age,
        "age_time_basis": "race_local_date_end_upper_bound_UTC_plus_12h_proxy",
        "prediction_cutoff_basis": "compatibility_order_proxy_not_pre_Q_cutoff",
        "available_at": available,
        "availability_basis": "reconstructed_next_January_1_not_original_publication",
        "actual_start_utc": pd.NaT,
        "actual_end_utc": pd.NaT,
        "original_publication_timestamp": pd.NaT,
        "minimum_target_season": 2016,
        "within_season_eligible": False,
        "predictor_training_eligible": False,
        "training_ready": False,
    }


def _positive(value):
    return bool(pd.notna(value) and np.isfinite(value) and value > 0)


def _numeric(position, text, size):
    return bool(
        _positive(position)
        and position == int(position)
        and position <= size
        and str(text).isdigit()
        and int(text) == position
    )


def reached_stage(row, q2_limit, q3_limit, exclusion_event, cancelled_q3=False):
    """A missing fastest lap never supplies evidence of elimination."""
    if _positive(row["q3_seconds"]):
        if cancelled_q3:
            raise ValueError("Positive Q3 contradicts reviewed cancellation")
        return 3, "positive_q3_time"
    if not row["label_usable"] or exclusion_event or q2_limit is None:
        return 0, "classification_or_historical_stage_rule_unresolved"
    stage = 3 if row["position"] <= q3_limit else 2 if row["position"] <= q2_limit else 1
    if cancelled_q3:
        stage = min(stage, 2)
    if any(_positive(row[f"q{s}_seconds"]) for s in range(stage + 1, 4)):
        raise ValueError("Positive later-stage time contradicts reviewed classification rules")
    return stage, (
        "reviewed_Q3_cancellation_and_classification"
        if cancelled_q3
        else "source_backed_rules_and_final_classification"
    )


def build_tables(races, qualifying, race_results, grid, official, official_events, reviews):
    """Construct a separate canonical extension from verified source tables."""
    races = races[races.year.between(2010, 2015)].sort_values(["year", "round", "id"])
    if races.id.duplicated().any() or not races.qualifyingFormat.eq("KNOCKOUT").all():
        raise ValueError("Unique 2010--2015 main knockout events required")
    for frame, columns in [
        (qualifying, ["raceId", "driverId"]),
        (official, ["race_id", "driver_id"]),
    ]:
        if frame.duplicated(columns).any():
            raise ValueError("Duplicate main qualifying identity")
    review_lookup = {}
    for row in reviews["difference_reviews"]:
        review_lookup.setdefault((row["race_id"], row["driver_id"]), []).append(row)
    event_rows, result_rows, pair_rows = [], [], []
    for ordinal, event in enumerate(races.itertuples(), 1):
        q = qualifying[qualifying.raceId.eq(event.id)]
        o = official[official.race_id.eq(event.id)]
        parts = [
            f[f.raceId.eq(event.id)][["driverId", "constructorId"]]
            for f in [qualifying, race_results, grid]
        ]
        roster = pd.concat(parts).drop_duplicates().sort_values("driverId")
        if roster.driverId.duplicated().any():
            raise ValueError("Constructor identity conflict in post-event roster")
        if not set(o.driver_id).issubset(set(roster.driverId)):
            raise ValueError("Official driver outside fixed-source roster union")
        size = len(roster)
        if size < 2:
            raise ValueError("Insufficient post-event roster")
        raw = {r.driverId: r for r in q.itertuples()}
        observed = {r.driver_id: r for r in o.itertuples()}
        source_event = official_events[official_events.race_id.eq(event.id)]
        if len(source_event) != 1:
            raise ValueError("Missing unique official event evidence")
        source_event = source_event.iloc[0]
        source_sha = digest(_json_bytes([ARCHIVE_SHA256, source_event.source_sha256]))
        times = reconstructed_times(event.year, event.date)
        common = {
            "event_id": int(event.id),
            "season": int(event.year),
            "round": int(event.round),
            "session_type": "Q",
            "circuit_layout_id": event.circuitLayoutId,
            "field_size": size,
            "source_version": "F1DB_v2026.14.0_plus_reviewed_official_Q",
            "source_sha256": source_sha,
            **times,
        }
        local = []
        for entry in roster.itertuples():
            a, b = raw.get(entry.driverId), observed.get(entry.driverId)
            for r, key in [(a, "constructorId"), (b, "constructor_id")]:
                if r is not None and getattr(r, key) != entry.constructorId:
                    raise ValueError("Qualifying constructor disagrees with actual driver identity")
            decisions = review_lookup.get((event.id, entry.driverId), [])
            blocked = any(r["decision"] != "retain_non_numeric_classification" for r in decisions)
            valid = bool(
                a is not None
                and b is not None
                and not blocked
                and _numeric(a.positionNumber, a.positionText, size)
                and _numeric(b.position_number, b.position_text, size)
                and a.positionNumber == b.position_number
            )
            flags = [LIMITATIONS]
            if source_event.official_header_date_conflict:
                flags.append("official_header_date_conflict_F1DB_race_date_proxy_retained")
            flags += [r["decision"] for r in decisions]
            if not valid:
                flags.append("numeric_qualifying_label_not_admitted")
            if a is None and b is None:
                flags.append("both_sources_missing_Q_row")
            status = (
                "classified"
                if valid
                else (
                    "DNS"
                    if event.id == 917 and entry.driverId in ["will-stevens", "roberto-merhi"]
                    else str(a.positionText)
                    if a is not None and not str(a.positionText).isdigit()
                    else "unresolved"
                    if a is not None
                    else "missing"
                )
            )
            row = {
                **common,
                "driver_id": entry.driverId,
                "constructor_id": entry.constructorId,
                "position": float(a.positionNumber) if valid else np.nan,
                "raw_f1db_position": a.positionNumber if a is not None else np.nan,
                "raw_f1db_classification": str(a.positionText) if a is not None else "",
                "official_position": b.position_number if b is not None else np.nan,
                "official_classification": str(b.position_text) if b is not None else "",
                "classification_status": status,
                "label_usable": valid,
                "f1db_source_row_present": a is not None,
                "official_source_row_present": b is not None,
                "rating_warmup_eligible": valid,
                "quality_flags": ";".join(flags),
            }
            for stage in [1, 2, 3]:
                av = getattr(a, f"q{stage}Millis") if a is not None else np.nan
                bv = getattr(b, f"q{stage}_millis") if b is not None else np.nan
                if _positive(av) and _positive(bv) and av != bv:
                    raise ValueError("Unreviewed official versus raw lap-time discrepancy")
                row[f"q{stage}_seconds"] = float(av) / 1000 if _positive(av) else np.nan
            local.append(row)
        complete = bool(
            len(local) == size
            and all(r["label_usable"] for r in local)
            and {r["position"] for r in local} == set(range(1, size + 1))
        )
        count = sum(r["label_usable"] for r in local)
        for row in local:
            row.update(
                {
                    "complete_numeric_final_field": complete,
                    "label_count": count,
                    "label_coverage": count / size,
                    "achievement_eligible": bool(complete and row["label_usable"]),
                }
            )
        result_rows.extend(local)
        rules = reviews["rules"][str(event.year)]
        q2_limit = rules["q2_limit"]
        if event.id in reviews["stage_inference_excluded_event_ids"]:
            q2_limit = None
        exclusion = any(r["classification_status"] in ["EX", "DSQ", "DQ", "RT"] for r in local)
        cancelled_q3 = event.id == reviews["q3_cancelled_event"]["event_id"]
        by_driver = {r["driver_id"]: r for r in local}
        for team, members in roster.groupby("constructorId", sort=True):
            if len(members) > 2:
                raise ValueError("More than two actual teammates")
            if len(members) != 2:
                continue
            ids = sorted(members.driverId)
            a, b = (by_driver[d] for d in ids)
            valid = a["label_usable"] and b["label_usable"] and a["position"] != b["position"]
            si, bi = reached_stage(a, q2_limit, rules["q3_limit"], exclusion, cancelled_q3)
            sj, bj = reached_stage(b, q2_limit, rules["q3_limit"], exclusion, cancelled_q3)
            last = min(si, sj) if si and sj else 0
            gaps, flags = {}, [LIMITATIONS]
            for stage in [1, 2, 3]:
                gap = np.nan
                if (
                    valid
                    and _positive(a[f"q{stage}_seconds"])
                    and _positive(b[f"q{stage}_seconds"])
                ):
                    value = 100 * np.log(b[f"q{stage}_seconds"] / a[f"q{stage}_seconds"])
                    if abs(value) <= 8:
                        gap = float(value)
                    else:
                        flags.append(f"q{stage}_gap_exceeds_fixed_quality_threshold")
                gaps[f"q{stage}_gap"] = gap
            pace = gaps.get(f"q{last}_gap", np.nan)
            if not last:
                flags.append("last_common_stage_unverified")
            if last and not np.isfinite(pace):
                flags.append("last_common_stage_has_no_eligible_pace_no_downgrade")
            if not valid:
                flags.append("invalid_or_unresolved_final_qualifying_classification")
            pair_rows.append(
                {
                    **common,
                    "constructor_id": team,
                    "driver_i": ids[0],
                    "driver_j": ids[1],
                    "season_event_ordinal": int(event.round),
                    "outcome_usable": bool(valid),
                    "outcome": int(np.sign(b["position"] - a["position"])) if valid else 0,
                    "rank_gap": b["position"] - a["position"] if valid else np.nan,
                    "rank_gap_abs_normalized": abs(b["position"] - a["position"]) / (size - 1)
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
                    "rule_evidence": f"v4_early_history_reviews_{event.year}",
                    "rating_warmup_eligible": bool(valid),
                    "quality_flags": ";".join(flags),
                }
            )
        event_rows.append(
            {
                **common,
                "race_date": event.date,
                "grand_prix_id": event.grandPrixId,
                "label_count": count,
                "label_coverage": count / size,
                "complete_numeric_final_field": complete,
                "rating_warmup_eligible": count > 1,
                "roster_basis": "post_event_Q_race_grid_union_not_pre_Q",
                "qualifying_format": event.qualifyingFormat,
                "official_header_date_conflict": bool(source_event.official_header_date_conflict),
                "quality_flags": LIMITATIONS,
            }
        )
    results = pd.DataFrame(result_rows)
    pairs = pd.DataFrame(pair_rows)
    events = pd.DataFrame(event_rows)
    achievements = results[results.achievement_eligible].copy().reset_index(drop=True)
    return pairs, achievements, events, results
