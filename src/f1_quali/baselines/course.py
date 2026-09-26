"""Cutoff-aware reconstruction of course-style OLS features."""

import numpy as np
import pandas as pd

from f1_quali.data.core import available_labels, fingerprint
from f1_quali.features.context import current_season
from f1_quali.features.evidence import eligible_practice

NUMERIC = ["fp_best_time", "driver_hist_avg", "track_specific", "prev_quali_pos"]


CATEGORICAL = ["driver_id", "constructor_id", "circuit_id"]


DEFAULTS = {
    "fp_best_time": 90.0,
    "driver_hist_avg": 90.0,
    "track_specific": 0.0,
    "prev_quali_pos": 10.5,
}


def _best_time(frame):
    values = frame[["q1_seconds", "q2_seconds", "q3_seconds"]].astype(float)
    values = values.where(np.isfinite(values) & values.gt(0))
    if "q1_usable" in frame:
        values.loc[~frame.q1_usable, "q1_seconds"] = np.nan
    return values.min(axis=1)


def build_course_features(data):
    """Rebuild the original four features without target or future raw details."""
    output = []
    events = data.events.sort_values(["prediction_cutoff", "event_id"])
    for event in events.itertuples(index=False):
        cutoff = pd.Timestamp(event.prediction_cutoff)
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("Explicit timezone-aware course prediction cutoff required")
        season_data = current_season(data, int(event.season))
        roster = season_data.entries[season_data.entries.event_id.eq(event.event_id)].copy()
        if roster.empty or roster.driver_id.duplicated().any():
            raise ValueError("Course baseline requires a complete unique roster")
        if roster.entry_known_at.gt(cutoff).any():
            raise ValueError("Future roster cannot enter course inputs")
        if not roster.roster_status.isin(
            [
                "resolved_pair",
                "documented_single_entry",
                "provisional_previous_event",
            ]
        ).all():
            raise ValueError("Unresolved roster cannot enter course inputs")
        history = available_labels(season_data, cutoff, int(event.event_id))
        history = history[
            history.prediction_cutoff.lt(cutoff)
            & history.label_usable
            & history.rank_percentile.notna()
        ].copy()
        history["quali_best_time"] = _best_time(history)
        # available_labels exposes layout identity; add the source circuit identity
        # used in the original one-hot and same-season track feature.
        history = history.merge(
            season_data.events[["event_id", "circuit_id"]], on="event_id", validate="many_to_one"
        )
        practice = eligible_practice(season_data, int(event.event_id), cutoff)
        practice["course_preference"] = practice.session_type.map({"FP2": 0, "FP3": 1, "FP1": 2})
        chosen = practice.sort_values(
            ["course_preference", "scheduled_start_utc", "session_id"],
            ascending=[True, False, True],
        ).drop_duplicates(["driver_id", "constructor_id"], keep="first")
        out = roster[
            ["event_id", "season", "round", "driver_id", "constructor_id", "roster_status"]
        ]
        out = out.merge(
            chosen[["driver_id", "constructor_id", "lap_seconds", "session_id"]],
            on=["driver_id", "constructor_id"],
            how="left",
            validate="one_to_one",
        )
        out = out.rename(columns={"lap_seconds": "fp_best_time", "session_id": "selected_practice"})
        out["selected_practice"] = out.selected_practice.fillna("none")
        out["missing_practice"] = out.fp_best_time.isna()
        rows = []
        for row in out.itertuples():
            driver = history[history.driver_id.eq(row.driver_id)]
            mean = (
                float(driver.quali_best_time.mean())
                if driver.quali_best_time.notna().any()
                else np.nan
            )
            track = driver[driver.circuit_id.eq(event.circuit_id)].quali_best_time
            track_mean = float(track.mean()) if track.notna().any() else np.nan
            rows.append(
                {
                    "driver_hist_avg": mean,
                    "track_specific": track_mean - mean,
                    "prev_quali_pos": float(driver.position.iloc[-1]) if len(driver) else np.nan,
                    "course_driver_history_count": int(driver.quali_best_time.notna().sum()),
                }
            )
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        out["circuit_id"], out["circuit_layout_id"] = event.circuit_id, event.circuit_layout_id
        out["prediction_cutoff"] = cutoff
        held = season_data.events[
            season_data.events.result_status.eq("historical_results_present")
            & season_data.events.prediction_cutoff.lt(cutoff)
        ]
        out["season_event_ordinal"] = len(held) + 1
        out["max_label_available_at"] = history.label_available_at.max()
        out["max_practice_available_at"] = chosen.available_at.max()
        out["history_min_season"] = int(history.season.min()) if len(history) else 0
        out["history_max_season"] = int(history.season.max()) if len(history) else 0
        out["history_sha256"] = fingerprint(
            history,
            [
                "event_id",
                "driver_id",
                "constructor_id",
                "quali_best_time",
                "position",
                "label_available_at",
            ],
        )
        out["time_basis"] = "reconstructed_historical_availability"
        output.append(out.sort_values("driver_id").reset_index(drop=True))
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _design(features, numeric_medians, categories):
    parts = [
        np.column_stack(
            [
                features[column].fillna(numeric_medians[column]).to_numpy(dtype=float)
                for column in NUMERIC
            ]
        )
    ]
    for column in CATEGORICAL:
        # Baseline category and genuinely unknown identities both map to zero;
        # vocabularies are fitted from previous-season training rows only.
        for category in categories[column][1:]:
            parts.append(features[column].eq(category).to_numpy(dtype=float)[:, None])
    return np.column_stack(parts)
