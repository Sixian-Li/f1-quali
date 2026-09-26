"""Past-only practice and complete qualifying classifications."""

import numpy as np
import pandas as pd

from f1_quali.data.core import Dataset


def eligible_practice(data: Dataset, event_id: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    p = data.practice
    p = p[
        (p.event_id == event_id)
        & p.session_type.isin(["FP1", "FP2", "FP3"])
        & p.available_at.notna()
        & (p.available_at <= cutoff)
        & (p.scheduled_start_utc < cutoff)
        & ~p.schedule_conflict
        & p.lap_usable
        & p.lap_seconds.gt(0)
        & np.isfinite(p.lap_seconds)
        & p.laps.gt(0)
    ].copy()
    if len(p):
        g = p.groupby("session_id").lap_seconds
        count = g.transform("count")
        p["percentile"] = (g.rank(method="average") - 1) / (count - 1).clip(lower=1)
        p.loc[count < 2, "percentile"] = 0.5
    else:
        p["percentile"] = pd.Series(dtype=float)
    return p.sort_values(["scheduled_start_utc", "session_id", "driver_id"]).reset_index(drop=True)


def recency_mean(values: pd.Series, half_life: float, fallback=0.5) -> tuple[float, float]:
    v = values.dropna().to_numpy(dtype=float)[-24:]
    if not len(v):
        return fallback, 0.0
    weights = np.exp2(-np.arange(len(v) - 1, -1, -1) / half_life)
    return float(np.average(v, weights=weights)), float(weights.sum())


def complete_labels(data: Dataset) -> pd.DataFrame:
    rows = []
    for eid, group in data.qualifying.groupby("event_id"):
        ent = data.entries[data.entries.event_id == eid]
        full = (
            set(group.driver_id) == set(ent.driver_id)
            and group.label_usable.all()
            and set(group.position) == set(range(1, len(ent) + 1))
            and group.label_available_at.notna().all()
        )
        if full:
            for r in group.itertuples():
                rows.append(
                    {
                        "event_id": eid,
                        "driver_id": r.driver_id,
                        "training_target": (r.position - 1) / (len(ent) - 1) - 0.5,
                        "training_label_available_at": group.label_available_at.max(),
                    }
                )
    return pd.DataFrame(
        rows, columns=["event_id", "driver_id", "training_target", "training_label_available_at"]
    )
