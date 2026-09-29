"""Deterministic, entirely synthetic offline example. Not a performance benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

from f1_quali import m1r1
from f1_quali.data.core import Dataset
from f1_quali.data.full_era import save_full_era
from f1_quali.data.portable import save_dataset
from f1_quali.integrity import write_json
from f1_quali.pipeline import predict_prepared, prepare, train


def synthetic_dataset(seed=20260926, history_events=20, target_events=6):
    rng = np.random.default_rng(seed)
    events, entries, qualifying, practice, sessions = [], [], [], [], []
    drivers = [f"synthetic-driver-{i:02}" for i in range(20)]
    teams = [f"synthetic-team-{i // 2:02}" for i in range(20)]
    skill = rng.normal(0, 0.15, 20) + np.repeat(rng.normal(0, 0.4, 10), 2)
    eid = 0
    for year, count in [(2023, history_events), (2024, target_events)]:
        for rnd in range(1, count + 1):
            eid += 1
            start = pd.Timestamp(f"{year}-03-01T12:00Z") + pd.Timedelta(days=12 * rnd)
            event = {
                "event_id": eid,
                "season": year,
                "round": rnd,
                "prediction_cutoff": start,
                "circuit_id": f"synthetic-circuit-{rnd % 3}",
                "circuit_layout_id": f"synthetic-layout-{rnd % 3}",
                "result_status": "historical_results_present",
                "qualifying_format": "KNOCKOUT",
                "source_version": "synthetic-v1",
                "eligible_field_size": 20,
            }
            events.append(event)
            order = np.argsort(skill + rng.normal(0, 0.3, 20), kind="stable")
            positions = np.argsort(order) + 1
            for kind, delta in [("FP1", 25), ("FP2", 21), ("FP3", 3), ("Q", 0)]:
                scheduled = start - pd.Timedelta(hours=delta)
                sessions.append(
                    {
                        "event_id": eid,
                        "season": year,
                        "round": rnd,
                        "session_id": f"{eid}:{kind}",
                        "session_type": kind,
                        "scheduled_start_utc": scheduled,
                        "available_at": scheduled + pd.Timedelta(hours=2),
                    }
                )
            for i, driver in enumerate(drivers):
                identity = {
                    "event_id": eid,
                    "season": year,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": teams[i],
                }
                entries.append(
                    {
                        **identity,
                        "entry_known_at": start - pd.Timedelta(days=2),
                        "roster_status": "resolved_pair",
                    }
                )
                rank = int(positions[i])
                qualifying.append(
                    {
                        **identity,
                        "position": rank,
                        "session_type": "Q",
                        "label_usable": True,
                        "source_row_present": True,
                        "classification_status": str(rank),
                        "label_available_at": start + pd.Timedelta(hours=2),
                        "q1_seconds": 90 + rank / 20,
                        "q2_seconds": 89 + rank / 20 if rank <= 15 else np.nan,
                        "q3_seconds": 88 + rank / 20 if rank <= 10 else np.nan,
                    }
                )
                for kind, delta in [("FP1", 25), ("FP2", 21), ("FP3", 3)]:
                    scheduled = start - pd.Timedelta(hours=delta)
                    practice.append(
                        {
                            **identity,
                            "session_id": f"{eid}:{kind}",
                            "session_type": kind,
                            "scheduled_start_utc": scheduled,
                            "available_at": scheduled + pd.Timedelta(hours=2),
                            "lap_seconds": 92 + skill[i] + rng.normal(0, 0.4),
                            "laps": 12,
                            "lap_usable": True,
                            "schedule_conflict": False,
                        }
                    )
    event_frame = pd.DataFrame(events)
    return Dataset(
        event_frame,
        pd.DataFrame(entries),
        pd.DataFrame(qualifying),
        pd.DataFrame(practice),
        pd.DataFrame(sessions),
        pd.DataFrame({"driver_id": drivers}),
        event_frame.copy(),
        {"kind": "synthetic", "seed": seed, "performance_evidence": False},
    )


def run_demo(output, progress=None, *, method="m1r1"):
    output = Path(output)
    data = synthetic_dataset()
    if method == "m1r1":
        save_full_era(output / "dataset", data, pd.DataFrame())
        m1r1.prepare(output / "dataset", output / "prepared", progress=progress)
        m1r1.train(output / "prepared", 2024, output / "model")
        m1r1.evaluate(output / "prepared", output / "model", output / "evaluation")
    elif method == "v6":
        save_dataset(output / "dataset", data)
        prepare(output / "dataset", output / "prepared", progress=progress)
        train(output / "prepared", 2024, output / "model")
        predict_prepared(output / "prepared", output / "model", output / "evaluation")
    else:
        raise ValueError(f"Unknown method: {method}")
    predictions = pd.read_parquet(output / "evaluation/predictions.parquet")
    last = predictions[predictions.event_id.eq(predictions.event_id.max())].sort_values(
        "predicted_rank"
    )
    summary = {
        "kind": "synthetic",
        "method": method,
        "year": 2024,
        "events": int(predictions.event_id.nunique()),
        "prediction_rows": len(predictions),
        "last_event_top3": last.head(3).driver_id.tolist(),
        "performance_evidence": False,
    }
    write_json(output / "summary.json", summary)
    return summary
