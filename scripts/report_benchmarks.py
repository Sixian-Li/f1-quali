"""Collect annual evaluation directories without reading research artifacts."""

import argparse
from pathlib import Path

import pandas as pd

from f1_quali.integrity import verify
from f1_quali.reporting import annual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames, raw_frames = [], []
    for path in sorted(args.evaluations.glob("*/event_metrics.parquet")):
        verify(path.parent, "evaluation")
        frames.append(pd.read_parquet(path))
        raw_frames.append(pd.read_parquet(path.parent / "raw_probability_metrics.parquet"))
    if not frames:
        raise ValueError("No evaluated model years found")
    events = pd.concat(frames, ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True)
    if events.duplicated(["model", "event_id"]).any():
        raise ValueError("Duplicate evaluation event")
    frames = []
    for name, frame in [
        ("from_sixth", events[events.season_event_ordinal.ge(6)]),
        ("first_five", events[events.season_event_ordinal.lt(6)]),
        ("all_events", events),
        ("from_sixth_raw_probabilities", raw[raw.season_event_ordinal.ge(6)]),
    ]:
        values = annual(frame)
        values.insert(0, "scope", name)
        frames.append(values)
    methods = set(events.model)
    if len(methods) != 1:
        raise ValueError("Evaluations mix methods")
    prefix = next(iter(methods))
    name = f"{prefix}_yearly.csv" if prefix in {"rm", "m1r1"} else "v6_yearly.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output / name, index=False)
    print(args.output / name)


if __name__ == "__main__":
    main()
