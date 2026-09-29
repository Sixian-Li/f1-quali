"""Current-season context with explicit cutoff and immutable ratings."""

import numpy as np
import pandas as pd

from f1_quali.data.core import Dataset, available_labels, fingerprint
from f1_quali.features.evidence import eligible_practice, recency_mean
from f1_quali.ratings.core import estimate, verify_bundle

CURRENT = [
    "practice_pct",
    "practice_missing",
    "practice_laps_log",
    "fp1_pct",
    "fp2_pct",
    "fp3_pct",
    "fp1_missing",
    "fp2_missing",
    "fp3_missing",
    "team_recent",
    "driver_recent",
    "driver_median",
    "team_track",
    "driver_history_log",
    "team_history_log",
    "changed_constructor",
]


FEATURE_GROUPS = {
    "current": CURRENT,
    "base": CURRENT + ["ability"],
    "layout": CURRENT + ["ability", "track_effect"],
}


def current_season(data, season):
    """Discard old raw details before any feature transform can inspect them."""

    def subset(frame):
        return frame[frame.season.eq(season)].copy() if "season" in frame else frame.copy()

    return Dataset(
        *(
            subset(frame)
            for frame in [
                data.events,
                data.entries,
                data.qualifying,
                data.practice,
                data.sessions,
            ]
        ),
        pd.DataFrame(),
        subset(data.calendar),
        {},
    )


def build_features(
    data, event, bundle, cfg, *, cutoff=None, roster=None, admit_practice=eligible_practice
):
    """``admit_practice`` selects weekend sessions; 2010--2015 use reviewed phase order."""
    verify_bundle(bundle)
    cutoff = pd.Timestamp(event.prediction_cutoff if cutoff is None else cutoff)
    if cutoff.tzinfo is None or pd.isna(cutoff):
        raise ValueError("Explicit timezone-aware prediction cutoff required")
    if cutoff > pd.Timestamp(event.prediction_cutoff):
        raise ValueError("Prediction cutoff is after target qualifying start")
    year = int(event.season)
    if bundle["target_year"] != year or pd.Timestamp(bundle["cutoff"]) >= cutoff:
        raise ValueError("Wrong rating bundle or cutoff")
    data = current_season(data, year)
    ent = (
        data.entries[data.entries.event_id.eq(event.event_id)].copy()
        if roster is None
        else roster.copy()
    )
    if ent.empty or ent.driver_id.duplicated().any():
        raise ValueError("Unique complete entrants required")
    if not ent.event_id.eq(event.event_id).all() or not ent.season.eq(year).all():
        raise ValueError("Roster event/season mismatch")
    if ent.entry_known_at.gt(cutoff).any():
        raise ValueError("Roster contains future entrant information")
    if not ent.roster_status.isin(
        [
            "resolved_pair",
            "documented_single_entry",
            "provisional_previous_event",
        ]
    ).all():
        raise ValueError("Unresolved roster")
    history = available_labels(data, cutoff, int(event.event_id))
    history = history[
        history.rank_percentile.notna() & history.prediction_cutoff.lt(cutoff)
    ].reset_index(drop=True)
    practice = admit_practice(data, int(event.event_id), cutoff)
    latest = practice.drop_duplicates(["driver_id", "constructor_id"], keep="last")
    out = ent[["event_id", "season", "round", "driver_id", "constructor_id", "roster_status"]]
    out = out.merge(
        latest[["driver_id", "constructor_id", "percentile", "laps", "session_id"]],
        on=["driver_id", "constructor_id"],
        how="left",
        validate="one_to_one",
    )
    out["practice_missing"] = out.percentile.isna().astype(float)
    out["missing_practice"] = out.percentile.isna()
    out["practice_pct"] = out.percentile.fillna(0.5)
    out["practice_laps_log"] = np.log1p(out.laps.fillna(0))
    out["selected_practice"] = out.session_id.fillna("none")
    out = out.drop(columns=["percentile", "laps", "session_id"])
    for kind in ["FP1", "FP2", "FP3"]:
        column = f"{kind.lower()}_pct"
        part = practice[practice.session_type.eq(kind)][
            ["driver_id", "constructor_id", "percentile"]
        ].rename(columns={"percentile": column})
        out = out.merge(part, on=["driver_id", "constructor_id"], how="left", validate="one_to_one")
        out[f"{kind.lower()}_missing"] = out[column].isna().astype(float)
        out[column] = out[column].fillna(0.5)
    team_history = (
        history.groupby(["constructor_id", "event_id"], sort=False)
        .agg(
            rank_percentile=("rank_percentile", "mean"),
            prediction_cutoff=("prediction_cutoff", "first"),
            circuit_layout_id=("circuit_layout_id", "first"),
        )
        .reset_index()
        .sort_values(["prediction_cutoff", "event_id"])
    )
    records = []
    for r in out.itertuples():
        team = team_history[team_history.constructor_id.eq(r.constructor_id)]
        driver = history[history.driver_id.eq(r.driver_id)]
        tm, _ = recency_mean(team.rank_percentile, cfg["history_half_life_events"])
        dm, _ = recency_mean(driver.rank_percentile, cfg["history_half_life_events"])
        track = team[team.circuit_layout_id.eq(event.circuit_layout_id)]
        lm, weight = recency_mean(track.rank_percentile, cfg["history_half_life_events"], tm)
        records.append(
            {
                "team_recent": tm,
                "driver_recent": dm,
                "driver_median": float(driver.rank_percentile.median()) if len(driver) else 0.5,
                "driver_history_log": np.log1p(len(driver)),
                "team_history_log": np.log1p(len(team)),
                "team_track": (lm - tm) * weight / (weight + 6),
                "changed_constructor": float(
                    bool(len(driver) and driver.constructor_id.iloc[-1] != r.constructor_id)
                ),
                **estimate(bundle, r.driver_id, event.circuit_layout_id),
            }
        )
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    out["circuit_id"] = event.circuit_id
    out["circuit_layout_id"] = event.circuit_layout_id
    out["prediction_cutoff"] = cutoff
    out["target_session_start_utc"] = event.prediction_cutoff
    held = data.events[
        data.events.result_status.eq("historical_results_present")
        & data.events.prediction_cutoff.lt(event.prediction_cutoff)
    ]
    out["season_event_ordinal"] = int(len(held) + 1)
    out["max_label_available_at"] = history.label_available_at.max()
    out["max_practice_available_at"] = practice.available_at.max()
    out["history_min_season"] = int(history.season.min()) if len(history) else 0
    out["history_max_season"] = int(history.season.max()) if len(history) else 0
    out["rating_cutoff"] = bundle["cutoff"]
    out["rating_bundle_sha256"] = bundle["bundle_sha256"]
    out["history_sha256"] = fingerprint(
        history,
        [
            "event_id",
            "driver_id",
            "constructor_id",
            "rank_percentile",
            "label_available_at",
        ],
    )
    out["time_basis"] = "reconstructed_historical_availability"
    out["fallback_reason"] = np.where(out.missing_practice, "no_eligible_practice_neutral_0.5", "")
    out["sprint_weekend"] = bool(
        data.sessions[data.sessions.event_id.eq(event.event_id)]
        .session_type.isin(["S", "SQ"])
        .any()
    )
    if not np.isfinite(out[FEATURE_GROUPS["layout"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite prediction features")
    return out.sort_values("driver_id").reset_index(drop=True)
