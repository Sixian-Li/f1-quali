"""Course-style, same-season rolling OLS, with strict availability gates."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

from f1_quali.baselines.course import (
    CATEGORICAL,
    DEFAULTS,
    NUMERIC,
    _best_time,
    _design,
    build_course_features,
)
from f1_quali.data.labels import build_labels, event_labels
from f1_quali.integrity import seal, write_frame


def prior_training_rows(joined, target):
    """Original rolling origin, with both labels and features before the cutoff."""
    cutoff = pd.Timestamp(target.prediction_cutoff)
    return (
        joined.loc[
            joined.season.eq(int(target.season))
            & joined.season_event_ordinal.lt(int(target.season_event_ordinal))
            & pd.to_datetime(joined.prediction_cutoff, utc=True).lt(cutoff)
            & pd.to_datetime(joined.label_available_at, utc=True).lt(cutoff)
            & joined.course_target_seconds.notna()
            & joined.fp_best_time.notna()
        ]
        .sort_values(["prediction_cutoff", "event_id", "driver_id"])
        .copy()
    )


def fit_predict(train, target):
    if train.event_id.nunique() < 5:
        raise ValueError("The original course backtest begins after five races")
    if target.fp_best_time.isna().any():
        raise ValueError("Original OLS cannot score a full field with missing practice")
    medians = {
        col: float(train[col].median()) if train[col].notna().any() else DEFAULTS[col]
        for col in NUMERIC
    }
    categories = {col: sorted(train[col].dropna().unique().tolist()) for col in CATEGORICAL}
    x = _design(train, medians, categories)
    y = train.course_target_seconds.to_numpy(dtype=float)
    xm, ym = x.mean(axis=0), float(y.mean())
    beta, _, rank, _ = lstsq(x - xm, y - ym, cond=max(x.shape) * np.finfo(float).eps)
    pred = (_design(target, medians, categories) - xm) @ beta + ym
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite OLS prediction")
    out = target[["event_id", "season", "round", "season_event_ordinal", "driver_id"]].copy()
    out["predicted_seconds"] = pred
    out["predicted_rank"] = out.predicted_seconds.rank(method="first").astype(int)
    trace = {
        "training_events": int(train.event_id.nunique()),
        "training_rows": len(train),
        "training_max_round": int(train["round"].max()),
        "training_max_label_available_at": train.label_available_at.max().isoformat(),
        "fit_rank": int(rank),
        "imputation_medians": medians,
    }
    return out, trace


def run_baselines(data, years, output):
    features = build_course_features(data)
    q = data.qualifying[["event_id", "driver_id", "label_usable", "label_available_at"]].copy()
    q["course_target_seconds"] = _best_time(data.qualifying)
    q.loc[~q.label_usable, "course_target_seconds"] = np.nan
    joined = features.merge(q, on=["event_id", "driver_id"], validate="one_to_one")
    labels = event_labels(build_labels(data))
    predictions, metrics, skipped = [], [], []
    for eid, target in features[
        features.season.isin(years) & features.season_event_ordinal.ge(6)
    ].groupby("event_id", sort=True):
        training = prior_training_rows(joined, target.iloc[0])
        if training.event_id.nunique() < 5 or target.fp_best_time.isna().any():
            skipped.append({"event_id": eid, "reason": "insufficient_past_or_missing_practice"})
            continue
        ols, _ = fit_predict(training, target)
        practice = target[
            ["event_id", "season", "round", "season_event_ordinal", "driver_id"]
        ].copy()
        practice["predicted_rank"] = target.fp_best_time.rank(method="first").astype(int)
        for name, prediction in [("course_rolling_ols", ols), ("practice_time", practice)]:
            prediction = prediction.assign(model=name)
            predictions.append(prediction)
            truth = labels[labels.event_id.eq(eid)]
            aligned = prediction.merge(truth, on=["event_id", "driver_id"], validate="one_to_one")
            if len(aligned) != len(prediction) or set(truth.driver_id) != set(prediction.driver_id):
                raise ValueError("Baseline evaluation requires the complete matching roster")
            record = {
                "model": name,
                "season": int(target.season.iloc[0]),
                "event_id": int(eid),
                "entrants": len(target),
            }
            for task in ["q2", "q3", "top3", "pole"]:
                quota = (
                    int(truth[f"{task}_quota"].iloc[0])
                    if task in ["q2", "q3"]
                    else {"top3": 3, "pole": 1}[task]
                )
                complete = bool(truth[f"complete_{task}"].all())
                record[f"complete_{task}"] = complete
                record[f"{task}_overlap"] = (
                    float(aligned.loc[aligned.predicted_rank.le(quota), f"y_{task}"].sum() / quota)
                    if complete
                    else np.nan
                )
            record["complete_rank"] = set(aligned.final_position) == set(range(1, len(target) + 1))
            record["rank_mae"] = (
                float((aligned.predicted_rank - aligned.final_position).abs().mean())
                if record["complete_rank"]
                else np.nan
            )
            metrics.append(record)
    if not predictions:
        raise ValueError("No complete baseline prediction events with sufficient history")
    write_frame(Path(output) / "predictions.parquet", pd.concat(predictions, ignore_index=True))
    write_frame(Path(output) / "event_metrics.parquet", pd.DataFrame(metrics))
    from f1_quali.integrity import write_json

    write_json(Path(output) / "skipped.json", skipped)
    return seal(output, kind="baselines", metadata={"probabilistic": False})
