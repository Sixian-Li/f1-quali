"""Complete-event probability and membership evaluation."""

import numpy as np
import pandas as pd

TASKS = ("q2", "q3", "top3", "pole")


def phase(year):
    if year <= 2018:
        return "initial_2016_2018"
    if year <= 2022:
        return "selection_2019_2022"
    if year <= 2024:
        return "seen_check_2023_2024"
    return f"already_seen_replay_{year}"


def event_truth(labels):
    columns = [
        "event_id",
        "driver_id",
        "final_position",
        *[f"y_{k}" for k in TASKS],
        *[f"complete_{k}" for k in TASKS],
        "q2_quota",
        "q3_quota",
    ]
    out = labels[columns].drop_duplicates()
    if out.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Stage-dependent final classification or pool labels")
    return out


def score_events(predictions, labels, *, probability_prefix="p_"):
    truth = event_truth(labels)
    records = []
    for (model, eid), group in predictions.groupby(["model", "event_id"], sort=True):
        target = truth[truth.event_id.eq(eid)]
        aligned = group.merge(
            target,
            on=["event_id", "driver_id"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_label"),
        )
        same_roster = len(target) == len(group) and set(target.driver_id) == set(group.driver_id)
        n = len(group)
        first = group.iloc[0]
        row = {
            "model": model,
            "event_id": eid,
            "season": int(first.season),
            "round": int(first["round"]),
            "phase": phase(int(first.season)),
            "season_event_ordinal": int(first.season_event_ordinal),
            "entrants": n,
            "same_roster": same_roster,
            "missing_practice_fraction": float(group.missing_practice.mean())
            if "missing_practice" in group
            else np.nan,
            "sprint_weekend": bool(first.get("sprint_weekend", False)),
        }
        scores = []
        for task in TASKS:
            k = (
                int(target[f"{task}_quota"].iloc[0])
                if task in ("q2", "q3")
                else {"top3": 3, "pole": 1}[task]
            )
            y = pd.to_numeric(aligned[f"y_{task}"], errors="coerce").to_numpy(dtype=float)
            complete = same_roster and target[f"complete_{task}"].all() and np.isfinite(y).all()
            complete = bool(complete and np.isclose(y.sum(), k))
            row[f"complete_{task}"] = complete
            row[f"labels_{task}"] = int(np.isfinite(y).sum())
            row[f"quota_{task}"] = k
            for name in ("brier", "normalized_brier", "logloss", "overlap", "exact"):
                row[f"{task}_{name}"] = np.nan
            if not complete:
                continue
            p = aligned[probability_prefix + task].to_numpy(dtype=float)
            if not np.isfinite(p).all() or (p < -1e-7).any() or (p > 1 + 1e-7).any():
                raise ValueError("Invalid probabilities")
            p = np.clip(p, 0, 1)
            hard = aligned[f"hard_{task}"].to_numpy(dtype=bool)
            if hard.sum() != k:
                raise ValueError("Hard pool capacity mismatch")
            bs = float(np.mean((p - y) ** 2))
            ll = (
                -float(np.log(np.clip(p[y == 1], 1e-12, 1)).sum())
                if task == "pole"
                else -float(
                    np.mean(y * np.log(p.clip(1e-12, 1)) + (1 - y) * np.log((1 - p).clip(1e-12, 1)))
                )
            )
            row.update(
                {
                    f"{task}_brier": bs,
                    f"{task}_normalized_brier": bs / (k / n * (1 - k / n)),
                    f"{task}_logloss": ll,
                    f"{task}_overlap": float(y[hard].sum() / k),
                    f"{task}_exact": float(np.array_equal(hard, y.astype(bool))),
                }
            )
            scores.append(row[f"{task}_normalized_brier"])
        if same_roster:
            hard = aligned[[f"hard_{k}" for k in TASKS]].to_numpy(dtype=int)
            if (np.diff(hard, axis=1) > 0).any():
                raise ValueError("Hard pool nesting violated")
            if probability_prefix == "p_":
                probability = aligned[[f"p_{k}" for k in TASKS]].to_numpy(dtype=float)
                quotas = [row[f"quota_{k}"] for k in TASKS]
                if (
                    not np.allclose(probability.sum(axis=0), quotas, atol=1e-6)
                    or (np.diff(probability, axis=1) > 1e-6).any()
                ):
                    raise ValueError("Coordinated probability capacity or nesting violated")
        row["complete_four_tasks"] = len(scores) == 4
        row["score"] = float(np.mean(scores)) if len(scores) == 4 else np.nan
        positions = pd.to_numeric(aligned.final_position, errors="coerce")
        rank_complete = same_roster and set(positions) == set(range(1, n + 1))
        row["complete_rank"] = rank_complete
        row["rank_mae"] = (
            float((aligned.predicted_rank - positions).abs().mean()) if rank_complete else np.nan
        )
        records.append(row)
    return pd.DataFrame(records)


def summarize(metrics):
    results = []
    values = [
        "score",
        "rank_mae",
        *[
            f"{k}_{m}"
            for k in TASKS
            for m in ("brier", "normalized_brier", "logloss", "overlap", "exact")
        ],
    ]
    for scope, frame in [
        ("from_sixth", metrics[metrics.season_event_ordinal.ge(6)]),
        ("first_five_inference_only", metrics[metrics.season_event_ordinal.lt(6)]),
    ]:
        for (model, period), group in frame.groupby(["model", "phase"], sort=True):
            results.append(
                {
                    "model": model,
                    "phase": period,
                    "scope": scope,
                    "events": len(group),
                    "four_task_events": int(group.complete_four_tasks.sum()),
                    "entrants": int(group.entrants.sum()),
                    **{f"{k}_events": int(group[f"complete_{k}"].sum()) for k in TASKS},
                    **group[values].mean().to_dict(),
                }
            )
    return pd.DataFrame(results)


def _complete_prediction_rosters(predictions, labels):
    expected = labels.groupby("event_id").driver_id.agg(set).to_dict()
    good = []
    keys = ["model", "event_id"] if "model" in predictions else ["event_id"]
    for _, event in predictions.groupby(keys, sort=True):
        if not event.driver_id.duplicated().any() and set(event.driver_id) == expected.get(
            event.event_id.iloc[0], set()
        ):
            good.append(event)
    return pd.concat(good, ignore_index=True) if good else predictions.iloc[:0].copy()


def calibration_bins(predictions, labels):
    predictions = _complete_prediction_rosters(predictions, labels)
    truth = event_truth(labels)
    rows = []
    for (model, year), part in predictions[predictions.season_event_ordinal.ge(6)].groupby(
        ["model", "season"], sort=True
    ):
        joined = part.merge(truth, on=["event_id", "driver_id"], suffixes=("", "_label"))
        for task in TASKS:
            v = joined[joined[f"complete_{task}"] & joined[f"y_{task}"].notna()].copy()
            if v.empty:
                continue
            v["bin"] = np.minimum((v[f"p_{task}"] * 10).astype(int), 9)
            v["weight"] = 1 / v.groupby("event_id").driver_id.transform("size")
            for bucket, group in v.groupby("bin"):
                rows.append(
                    {
                        "model": model,
                        "season": year,
                        "phase": phase(year),
                        "task": task,
                        "bin": bucket,
                        "driver_rows": len(group),
                        "event_weight": group.weight.sum(),
                        "mean_probability": np.average(group[f"p_{task}"], weights=group.weight),
                        "observed_frequency": np.average(group[f"y_{task}"], weights=group.weight),
                    }
                )
    return pd.DataFrame(rows)
