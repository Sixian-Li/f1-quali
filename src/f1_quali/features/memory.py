"""Selected v6 current-season weighted mean/median summaries."""

import numpy as np
import pandas as pd

from f1_quali.data.core import available_labels, fingerprint
from f1_quali.features.context import current_season

MODIFIED_FEATURES = ("driver_recent", "team_recent", "team_track")


CENTERS = ("weighted_mean", "mean_median_50_50")


HISTORY_COLUMNS = [
    "event_id",
    "driver_id",
    "constructor_id",
    "rank_percentile",
    "label_available_at",
]


def weighted_center(values, half_life=6.0, center="weighted_mean", fallback=0.5):
    """Use the last 24 valid observations and the registered weighted median.

    The median is the first value in stable value order whose cumulative weight
    reaches at least half the total. Its weight is also used by track shrinkage.
    """
    if not np.isfinite(half_life) or half_life <= 0 or center not in CENTERS:
        raise ValueError("Invalid memory half-life or center")
    v = pd.Series(values).dropna().to_numpy(dtype=float)[-24:]
    if not np.isfinite(v).all() or not np.isfinite(fallback):
        raise ValueError("Memory values and fallback must be finite")
    if not len(v):
        return float(fallback), 0.0
    weights = np.exp2(-np.arange(len(v) - 1, -1, -1) / half_life)
    mean = float(np.average(v, weights=weights))
    if center == "weighted_mean":
        return mean, float(weights.sum())
    order = np.argsort(v, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    median = float(v[order[index]])
    return 0.5 * mean + 0.5 * median, float(weights.sum())


def replace_summaries(data, base, *, half_life=6.0, center="weighted_mean"):
    """Reconstruct eligible evidence, change three columns, preserve every other value.

    Matching the original history fingerprint makes source drift an error. A
    poisoned target label remains excluded even if its availability is backdated.
    """
    weighted_center([], half_life, center)
    if base.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("One feature row per event entrant required")
    out = base.reset_index(drop=True).copy()
    seasons = {int(y): current_season(data, int(y)) for y in out.season.unique()}
    for eid, rows in out.groupby("event_id", sort=True):
        if rows.season.nunique() != 1 or rows.prediction_cutoff.nunique() != 1:
            raise ValueError("Mixed season or cutoff within target event")
        cutoff = pd.Timestamp(rows.prediction_cutoff.iloc[0])
        if cutoff.tzinfo is None or pd.isna(cutoff):
            raise ValueError("Explicit timezone-aware prediction cutoff required")
        local = seasons[int(rows.season.iloc[0])]
        target = local.events[local.events.event_id.eq(eid)]
        if len(target) != 1 or cutoff > pd.Timestamp(target.prediction_cutoff.iloc[0]):
            raise ValueError("Unknown target or cutoff after target qualifying")
        event = target.iloc[0]
        if not rows.circuit_layout_id.eq(event.circuit_layout_id).all():
            raise ValueError("Target layout differs from base features")
        history = available_labels(local, cutoff, int(eid))
        history = history[
            history.rank_percentile.notna() & history.prediction_cutoff.lt(cutoff)
        ].reset_index(drop=True)
        if not rows.history_sha256.eq(fingerprint(history, HISTORY_COLUMNS)).all():
            raise ValueError("Current-season history differs from immutable base evidence")
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
        for row in rows.itertuples():
            team = team_history[team_history.constructor_id.eq(row.constructor_id)]
            driver = history[history.driver_id.eq(row.driver_id)]
            tm, _ = weighted_center(team.rank_percentile, half_life, center)
            dm, _ = weighted_center(driver.rank_percentile, half_life, center)
            track = team[team.circuit_layout_id.eq(event.circuit_layout_id)]
            lm, weight = weighted_center(track.rank_percentile, half_life, center, tm)
            records.append((dm, tm, (lm - tm) * weight / (weight + 6)))
        out.loc[rows.index, list(MODIFIED_FEATURES)] = np.asarray(records)
    out.index = base.index.copy()
    return out
