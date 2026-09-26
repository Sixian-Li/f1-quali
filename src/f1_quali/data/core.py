"""Typed canonical tables and strictly available historical labels."""

from dataclasses import dataclass

import pandas as pd

from f1_quali.integrity import sha256


@dataclass
class Dataset:
    events: pd.DataFrame
    entries: pd.DataFrame
    qualifying: pd.DataFrame
    practice: pd.DataFrame
    sessions: pd.DataFrame
    drivers: pd.DataFrame
    calendar: pd.DataFrame
    manifest: dict

    def through_year(self, year: int):
        def subset(frame):
            return frame[frame.season <= year].copy() if "season" in frame else frame.copy()

        return Dataset(
            *(
                subset(f)
                for f in [self.events, self.entries, self.qualifying, self.practice, self.sessions]
            ),
            self.drivers.copy(),
            subset(self.calendar),
            self.manifest,
        )


def fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    return sha256(frame[columns].to_csv(index=False, float_format="%.12g").encode())


def available_labels(data: Dataset, cutoff: pd.Timestamp, target_id: int) -> pd.DataFrame:
    labels = data.qualifying
    labels = labels[
        (labels.event_id != target_id)
        & labels.label_available_at.notna()
        & (labels.label_available_at <= cutoff)
    ].copy()
    counts = data.entries.groupby("event_id").size()
    labels["field_size"] = labels.event_id.map(counts)
    labels["rank_percentile"] = ((labels.position - 1) / (labels.field_size - 1)).where(
        labels.label_usable & labels.position.gt(0) & labels.position.le(labels.field_size)
    )
    layouts = data.events[["event_id", "circuit_layout_id", "prediction_cutoff"]]
    return (
        labels.merge(layouts, on="event_id", validate="many_to_one")
        .sort_values(["prediction_cutoff", "event_id", "driver_id"])
        .reset_index(drop=True)
    )
