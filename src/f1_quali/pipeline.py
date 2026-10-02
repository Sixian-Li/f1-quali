"""Data-to-rating-to-prediction pipeline for the previous release method v6.

RM and m1r1 use :mod:`f1_quali.m1r1`; all methods share these engines.
"""

import json
from pathlib import Path

import pandas as pd

from f1_quali.config import FEATURES, METHOD, load_config
from f1_quali.data.labels import build_labels, stage_limits
from f1_quali.data.portable import load_dataset
from f1_quali.evaluation import score_events
from f1_quali.features.context import build_features as build_context
from f1_quali.features.memory import replace_summaries
from f1_quali.features.ratings import replace_ratings
from f1_quali.integrity import seal, sha256, verify, write_frame, write_json
from f1_quali.models.pools import fit_pool_models, predict_pools
from f1_quali.ratings.pipeline import evidence, rating_snapshot
from f1_quali.ratings.rolling import predictor_bundle


def event_features(data, event, bundle, config, *, cutoff=None, roster=None):
    base = build_context(
        data, event, predictor_bundle(bundle, composite=True), config, cutoff=cutoff, roster=roster
    )
    base = replace_ratings(
        base, {("composite_early", int(event.event_id)): bundle}, "composite_early"
    )
    return replace_summaries(
        data, base, half_life=config["memory_half_life"], center=config["memory_center"]
    )


def prepare(dataset, output, config=None, *, progress=None):
    config = config or load_config()
    data, early = load_dataset(dataset)
    pairs, achievements = evidence(data, config, early)
    frames = []
    for event in data.events.sort_values(["prediction_cutoff", "event_id"]).itertuples(index=False):
        event = pd.Series(event._asdict())
        bundle = rating_snapshot(event, pairs, achievements, config)
        frame = event_features(data, event, bundle, config)
        frames.append(frame)
        write_json(Path(output) / "ratings" / f"{event.event_id}.json", bundle)
        if progress:
            progress(
                f"Prepared {int(event.season)} event {int(event.event_id)} ({len(frame)} entrants)"
            )
    features = pd.concat(frames, ignore_index=True)
    labels = build_labels(data)
    write_frame(Path(output) / "features.parquet", features)
    write_frame(Path(output) / "labels.parquet", labels)
    write_json(Path(output) / "config.json", config)
    return seal(
        output,
        kind="prepared",
        metadata={
            "dataset_manifest_sha256": sha256((Path(dataset) / "manifest.json").read_bytes()),
            "warmup": early is not None,
            "method": METHOD,
        },
    )


def train(prepared, year, output):
    manifest = verify(prepared, "prepared")
    prepared, output = Path(prepared), Path(output)
    config = load_config(prepared / "config.json")
    features, labels = (
        pd.read_parquet(prepared / name) for name in ["features.parquet", "labels.parquet"]
    )
    model = fit_pool_models(
        features,
        labels,
        int(year),
        FEATURES,
        alpha=config["alpha"],
        minimum_events=config["minimum_events"],
    )
    write_json(output / "model.json", model)
    write_json(output / "config.json", config)
    return seal(
        output,
        kind="predictor",
        metadata={"method": METHOD, "year": int(year), "warmup": manifest["metadata"]["warmup"]},
    )


def load_predictor(directory):
    verify(directory, "predictor")
    model = json.loads((Path(directory) / "model.json").read_text())
    if model["columns"] != FEATURES or model["kind"] != "independent_pools":
        raise ValueError("Predictor schema/feature order differs from v6")
    return model


def predict_prepared(prepared, predictor, output, *, event_id=None):
    verify(prepared, "prepared")
    if (Path(prepared) / "config.json").read_bytes() != (
        Path(predictor) / "config.json"
    ).read_bytes():
        raise ValueError("Prepared features and model use different configurations")
    model = load_predictor(predictor)
    features = pd.read_parquet(Path(prepared) / "features.parquet")
    labels = pd.read_parquet(Path(prepared) / "labels.parquet")
    rows = features[features.season.eq(model["trace"]["year"])]
    if event_id is not None:
        rows = rows[rows.event_id.eq(event_id)]
    if rows.empty:
        raise ValueError("No matching prediction events")
    predictions = []
    for eid, group in rows.groupby("event_id", sort=True):
        target = labels[labels.event_id.eq(eid)]
        # Quotas are determined by reviewed rules, never target finishing order.
        if target.empty:
            raise ValueError("Use forecast for events without historical label schemas")
        q2, q3 = int(target.q2_quota.iloc[0]), int(target.q3_quota.iloc[0])
        predictions.append(predict_pools(model, group, q2, q3).assign(model=METHOD))
    frame = pd.concat(predictions, ignore_index=True)
    write_frame(Path(output) / "predictions.parquet", frame)
    write_frame(Path(output) / "event_metrics.parquet", score_events(frame, labels))
    write_frame(
        Path(output) / "raw_probability_metrics.parquet",
        score_events(frame, labels, probability_prefix="raw_p_"),
    )
    return seal(
        output, kind="evaluation", metadata={"method": METHOD, "scope": "historical_reconstruction"}
    )


def forecast(dataset, predictor, event_id, cutoff, output):
    data, early = load_dataset(dataset)
    model = load_predictor(predictor)
    config = load_config(Path(predictor) / "config.json")
    found = data.events[data.events.event_id.eq(event_id)]
    if len(found) != 1:
        raise ValueError("Exactly one target event required")
    event = found.iloc[0]
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(model["trace"]["training_cutoff"]):
        raise ValueError("Predictor was not yet available at cutoff")
    pairs, achievements = evidence(data, config, early, cutoff=cutoff, target_event_id=event_id)
    bundle = rating_snapshot(event, pairs, achievements, config, cutoff=cutoff)
    features = event_features(data, event, bundle, config, cutoff=cutoff)
    q2, q3, _ = stage_limits(
        int(event.season),
        int(event["round"]),
        event.qualifying_format,
        event.get("eligible_field_size"),
    )
    prediction = predict_pools(model, features, q2, q3).assign(model=METHOD)
    write_frame(Path(output) / "predictions.parquet", prediction)
    write_json(Path(output) / "ratings.json", bundle)
    from f1_quali.integrity import utc_now

    generated = utc_now()
    write_json(
        Path(output) / "forecast.json",
        {
            "event_id": int(event_id),
            "cutoff": pd.Timestamp(cutoff).isoformat(),
            "generated_at": generated,
            "prospective": pd.Timestamp(generated) < pd.Timestamp(event.prediction_cutoff),
            "model_manifest_sha256": sha256((Path(predictor) / "manifest.json").read_bytes()),
        },
    )
    return seal(output, kind="forecast", metadata={"method": METHOD})
