"""The selected m1r1 method: 2010--2026 data, ratings and four-pool prediction.

Pipeline, for every event in season/round order:

1. Refit the teammate network strictly before the event (fixed annual
   parameters; the achievement component uses complete sessions and beta 0.6).
   These anchor snapshots also define post-event teammate "innovations".
2. Rebuild only the achievement component with m1r1 memory: individually
   admitted results, no permanent floor, one-year half-lives, fixed beta 0.6.
3. Summarize each driver's earlier innovations; the annual model's shared rules
   turn them into bounded driver and driver-by-layout corrections.

The annual predictor for year Y trains only on 2010..Y-1 complete events from
the sixth round onward: an anchor fit supplies feature scales and Top3 quota
Brier seeds, then the Top3 head and the correction rules are fitted jointly and
Q2/Q3/pole are refitted on the corrected inputs.
"""

import json
from importlib.resources import files
from pathlib import Path

import pandas as pd

from f1_quali.config import FEATURES
from f1_quali.data.full_era import (
    EARLY_CUTOFF_BASIS,
    EARLY_TIME_BASIS,
    EARLY_YEARS,
    build_full_labels,
    load_full_era,
    practice_admission,
    split,
)
from f1_quali.data.labels import build_rating_observations, stage_limits
from f1_quali.evaluation import score_events
from f1_quali.features.context import build_features
from f1_quali.features.memory import replace_summaries
from f1_quali.features.ratings import replace_ratings
from f1_quali.integrity import seal, sha256, utc_now, verify, write_frame, write_json
from f1_quali.models.joint import (
    attach_history,
    changed_inputs,
    fit_fixed,
    fit_top3,
    fixed_achievement_features,
    make_innovations,
    predict_heads,
)
from f1_quali.models.pools import fit_pool_models
from f1_quali.ratings.achievements import qualifying_achievements
from f1_quali.ratings.memory import individual_achievements, rebuild_achievement
from f1_quali.ratings.pipeline import rating_snapshot
from f1_quali.ratings.rolling import fit_ratings, freeze_bundle, predictor_bundle
from f1_quali.ratings.rolling import with_achievement_bonus as add_bonus

METHOD = "m1r1"
VARIANT = "composite_early"


def load_m1r1_config(path=None):
    text = (
        Path(path).read_text()
        if path
        else files("f1_quali").joinpath("resources/m1r1.json").read_text()
    )
    config = json.loads(text)
    if config.get("method") != METHOD or config["first_optimized_ordinal"] != 6:
        raise ValueError("Expected the m1r1 configuration")
    betas = {row["achievement_beta"] for row in config["rating_years"].values()}
    if betas != {config["achievement_beta"]}:
        raise ValueError("m1r1 uses one absolute achievement beta in every year")
    if 3 in config["joint_rules"]["active"]:
        raise ValueError("The achievement beta delta is not trainable in m1r1")
    return config


def _progress(progress, message):
    if progress:
        progress(message)


def rating_evidence(data, early_pairs, config):
    """Teammate pairs (2010+) and complete/individual achievement evidence."""
    early_ids = data.events.loc[data.events.season.le(EARLY_YEARS[1]), "event_id"]
    early, modern = split(data, early_ids)
    rating_config = config["rating_years"][str(EARLY_YEARS[1] + 1)]["config"]
    pairs = build_rating_observations(modern, rating_config)
    pairs["age_at"] = pairs.prediction_cutoff
    if len(early_pairs):
        keep = [c for c in pairs if c in early_pairs]
        pairs = pd.concat([early_pairs[keep], pairs], ignore_index=True)
    for column in ["age_at", "available_at", "prediction_cutoff"]:
        pairs[column] = pd.to_datetime(pairs[column], utc=True)
    complete = qualifying_achievements(data)
    individual, quarantined = individual_achievements(data)
    if not early.events.empty:
        # Early results age from the conservative race-date bound, not the proxy cutoff.
        age = early.events.set_index("event_id").results_available_upper_bound
        for frame in [complete, individual]:
            old = frame.season.le(EARLY_YEARS[1])
            frame.loc[old, "age_at"] = frame.loc[old, "event_id"].map(age)
            frame.loc[old, "age_time_basis"] = "race_date_plus_36h_age_proxy"
    return pairs, complete, individual, quarantined


def _event_features(data, event, bundle, config, *, cutoff=None, roster=None):
    feature = build_features(
        data,
        event,
        predictor_bundle(bundle, composite=True),
        config,
        cutoff=cutoff,
        roster=roster,
        admit_practice=practice_admission(event),
    )
    if int(event.season) <= EARLY_YEARS[1]:
        feature["target_session_start_utc"] = pd.NaT
        feature["time_basis"] = EARLY_TIME_BASIS
        feature["prediction_cutoff_basis"] = EARLY_CUTOFF_BASIS
    else:
        feature["prediction_cutoff_basis"] = "existing_reviewed_modern_cutoff"
    return feature


def prepare(dataset, output, config=None, *, progress=None):
    """Causal per-event ratings and inputs for every 2010--2026 event."""
    config = config or load_m1r1_config()
    output = Path(output)
    data, early_pairs = load_full_era(dataset)
    labels = build_full_labels(data)
    pairs, complete, individual, quarantined = rating_evidence(data, early_pairs, config)
    write_frame(output / "labels.parquet", labels)
    write_frame(output / "pairs.parquet", pairs)
    write_frame(output / "complete_achievements.parquet", complete)
    write_frame(output / "individual_achievements.parquet", individual)
    write_frame(output / "quarantined_achievements.parquet", quarantined)
    evidence_sha256 = sha256((output / "individual_achievements.parquet").read_bytes())
    anchors, rows = {}, []
    events = data.events.sort_values(["season", "round"])
    for i, event in enumerate(events.itertuples(), 1):
        anchor = rating_snapshot(event, pairs, complete, config)
        anchors[int(event.event_id)] = anchor
        rows.append(_event_features(data, event, anchor, config))
        _progress(progress, f"Rated {i}/{len(events)}: {event.season} round {event.round}")
    features = pd.concat(rows, ignore_index=True)
    features = replace_summaries(
        data, features, half_life=config["memory_half_life"], center=config["memory_center"]
    )
    features = replace_ratings(features, {(VARIANT, k): v for k, v in anchors.items()}, VARIANT)
    cfg, beta = config["joint_rules"]["settings"], config["achievement_beta"]
    innovations, auxiliary = make_innovations(features, pairs, cfg)
    features = attach_history(features, anchors, innovations, cfg)
    anchor_features = fixed_achievement_features(features, beta)
    bundles = {}
    for eid in features.event_id.unique():
        bundle = rebuild_achievement(
            anchors[int(eid)], individual, config["achievement_memory"], evidence_sha256
        )
        bundles[int(eid)] = bundle
        write_json(output / "ratings" / f"{int(eid)}.json", bundle)
    rated = replace_ratings(features, {(VARIANT, k): v for k, v in bundles.items()}, VARIANT)
    rated = attach_history(rated, bundles, innovations, cfg)
    rated = fixed_achievement_features(rated, beta)
    write_frame(output / "innovations.parquet", innovations)
    write_frame(output / "auxiliary_pairs.parquet", auxiliary)
    write_frame(output / "anchor_features.parquet", anchor_features)
    write_frame(output / "features.parquet", rated)
    write_json(output / "config.json", config)
    return seal(
        output,
        kind="prepared",
        metadata={
            "method": METHOD,
            "dataset_manifest_sha256": sha256((Path(dataset) / "manifest.json").read_bytes()),
            "events": int(features.event_id.nunique()),
            "seasons": [int(features.season.min()), int(features.season.max())],
            "last_result_available_at": labels.label_available_at.max().isoformat(),
            "early_actual_clocks_unknown": True,
        },
    )


def _load_prepared(prepared):
    manifest = verify(prepared, "prepared")
    if manifest["metadata"].get("method") != METHOD:
        raise ValueError("Prepared inputs are not m1r1; use the v6 commands")
    config = load_m1r1_config(Path(prepared) / "config.json")
    return manifest, config


def train(prepared, year, output):
    """Annual m1r1 predictor using only 2010..year-1 complete events."""
    _, config = _load_prepared(prepared)
    prepared, output = Path(prepared), Path(output)
    labels = pd.read_parquet(prepared / "labels.parquet")
    anchor = pd.read_parquet(prepared / "anchor_features.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    pairs = pd.read_parquet(prepared / "auxiliary_pairs.parquet")
    base = fit_pool_models(
        anchor,
        labels,
        int(year),
        FEATURES,
        alpha=config["alpha"],
        minimum_events=config["minimum_events"],
    )
    base = fit_top3(base, anchor, labels)
    rules = config["joint_rules"]
    model = fit_fixed(
        base,
        features,
        labels,
        pairs,
        rules["active"],
        rules["settings"],
        config["achievement_beta"],
    )
    for task in model["tasks"].values():
        maximum = task["trace"]["training_max_year"]
        if maximum is not None and maximum >= int(year):
            raise RuntimeError("Annual predictor read target-year labels")
    model["method"] = METHOD
    write_json(output / "model.json", model)
    write_json(output / "config.json", config)
    return seal(output, kind="predictor", metadata={"method": METHOD, "year": int(year)})


def load_predictor(directory):
    verify(directory, "predictor")
    model = json.loads((Path(directory) / "model.json").read_text())
    if model.get("method") != METHOD or model["columns"] != FEATURES:
        raise ValueError("Predictor is not an m1r1 model")
    return model


def _check_pair(prepared, predictor):
    if (Path(prepared) / "config.json").read_bytes() != (
        Path(predictor) / "config.json"
    ).read_bytes():
        raise ValueError("Prepared inputs and model use different configurations")


def evaluate(prepared, predictor, output, *, event_id=None):
    """Replay the model's year on stored causal inputs; score complete events."""
    _load_prepared(prepared)
    _check_pair(prepared, predictor)
    model = load_predictor(predictor)
    config = load_m1r1_config(Path(prepared) / "config.json")
    features = pd.read_parquet(Path(prepared) / "features.parquet")
    labels = pd.read_parquet(Path(prepared) / "labels.parquet")
    rows = features[features.season.eq(model["trace"]["year"])]
    if event_id is not None:
        rows = rows[rows.event_id.eq(event_id)]
    if rows.empty:
        raise ValueError("No matching prediction events")
    rows = changed_inputs(rows, model, config["joint_rules"]["settings"])
    predictions = []
    for eid, group in rows.groupby("event_id", sort=True):
        target = labels[labels.event_id.eq(eid)]
        # Quotas come from reviewed rules, never from the target finishing order.
        q2, q3 = int(target.q2_quota.iloc[0]), int(target.q3_quota.iloc[0])
        predictions.append(predict_heads(model, group, q2, q3).assign(model=METHOD))
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


def _evidence(prepared):
    """Stored rating evidence; every consumer applies its own cutoff."""
    prepared = Path(prepared)
    pairs = pd.read_parquet(prepared / "pairs.parquet")
    complete = pd.read_parquet(prepared / "complete_achievements.parquet")
    individual = pd.read_parquet(prepared / "individual_achievements.parquet")
    innovations = pd.read_parquet(prepared / "innovations.parquet")
    evidence_sha256 = sha256((prepared / "individual_achievements.parquet").read_bytes())
    return pairs, complete, individual, innovations, evidence_sha256


def forecast(dataset, prepared, predictor, event_id, cutoff, output):
    """Predict one modern event at an explicit cutoff from a prepared history."""
    _load_prepared(prepared)
    _check_pair(prepared, predictor)
    config = load_m1r1_config(Path(prepared) / "config.json")
    model = load_predictor(predictor)
    data, _ = load_full_era(dataset)
    found = data.events[data.events.event_id.eq(event_id)]
    if len(found) != 1:
        raise ValueError("Exactly one target event required")
    event = found.iloc[0]
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff < pd.Timestamp(model["trace"]["training_cutoff"]):
        raise ValueError("Predictor was not yet available at cutoff")
    if int(event.season) != model["trace"]["year"] or int(event.season) <= EARLY_YEARS[1]:
        raise ValueError("Use the annual model for a modern target year")
    labels = pd.read_parquet(Path(prepared) / "labels.parquet")
    held = data.qualifying[
        data.qualifying.label_available_at.le(cutoff) & data.qualifying.event_id.ne(event_id)
    ]
    if not set(held.event_id).issubset(set(labels.event_id)):
        raise ValueError(
            "Prepared history is older than the dataset at this cutoff; re-run prepare"
        )
    pairs, complete, individual, innovations, digest = _evidence(prepared)
    anchor = rating_snapshot(event, pairs, complete, config, cutoff=cutoff)
    bundle = rebuild_achievement(anchor, individual, config["achievement_memory"], digest)
    features = _event_features(data, event, bundle, config, cutoff=cutoff)
    features = replace_summaries(
        data, features, half_life=config["memory_half_life"], center=config["memory_center"]
    )
    features = replace_ratings(features, {(VARIANT, int(event_id)): bundle}, VARIANT)
    cfg = config["joint_rules"]["settings"]
    features = attach_history(features, {int(event_id): bundle}, innovations, cfg)
    features = fixed_achievement_features(features, config["achievement_beta"])
    features = changed_inputs(features, model, cfg)
    q2, q3, _ = stage_limits(
        int(event.season),
        int(event["round"]),
        event.qualifying_format,
        event.get("eligible_field_size"),
    )
    prediction = predict_heads(model, features, q2, q3).assign(model=METHOD)
    write_frame(Path(output) / "predictions.parquet", prediction)
    write_json(Path(output) / "ratings.json", bundle)
    generated = utc_now()
    write_json(
        Path(output) / "forecast.json",
        {
            "event_id": int(event_id),
            "cutoff": cutoff.isoformat(),
            "generated_at": generated,
            "prospective": pd.Timestamp(generated) < pd.Timestamp(event.prediction_cutoff),
            "model_manifest_sha256": sha256((Path(predictor) / "manifest.json").read_bytes()),
        },
    )
    return seal(output, kind="forecast", metadata={"method": METHOD})


def driver_ratings(prepared, predictor, cutoff, *, layouts=True):
    """General (and optional per-layout) ratings after all results before ``cutoff``.

    The roster is the latest event with results before the cutoff. The network
    is refitted at the cutoff, the achievement memory rebuilt, and the model's
    shared correction rules applied, exactly as for a prediction snapshot.
    """
    _load_prepared(prepared)
    _check_pair(prepared, predictor)
    config = load_m1r1_config(Path(prepared) / "config.json")
    model = load_predictor(predictor)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Timezone-aware cutoff required")
    features = pd.read_parquet(Path(prepared) / "features.parquet")
    labels = pd.read_parquet(Path(prepared) / "labels.parquet")
    finished = labels[labels.label_available_at.le(cutoff)]
    held = features[features.event_id.isin(finished.event_id)]
    if held.empty:
        raise ValueError("No results before cutoff")
    last = held.sort_values(["prediction_cutoff", "event_id"]).event_id.iloc[-1]
    year = int(held[held.event_id.eq(last)].season.iloc[0])
    if model["trace"]["year"] != year or cutoff.year != year:
        raise ValueError("Use the annual model and a cutoff in the roster's season")
    pairs, complete, individual, innovations, digest = _evidence(prepared)
    parameters = config["rating_years"][str(year)]
    fit = fit_ratings(
        pairs,
        cutoff - pd.Timedelta(nanoseconds=1),
        parameters["config"],
        target_event_id=-year,
        penalty=parameters["penalty"],
        achievements=complete,
    )
    anchor = freeze_bundle(
        fit,
        year,
        parameters["calibration_slope"],
        variant="full_era_endpoint",
        selection_provenance={"annual_parameters_only": True},
    )
    anchor = add_bonus(
        anchor,
        config["achievement_beta"],
        calibration_slope=parameters["composite_calibration_slope"],
    )
    bundle = rebuild_achievement(anchor, individual, config["achievement_memory"], digest)
    cfg = config["joint_rules"]["settings"]
    target = features[features.event_id.eq(last)].copy()
    target["event_id"], target["prediction_cutoff"] = -year, cutoff
    target["circuit_layout_id"] = "overall_only"

    def transform(frame):
        frame = replace_ratings(frame, {(VARIANT, -year): bundle}, VARIANT)
        frame = attach_history(frame, {-year: bundle}, innovations, cfg)
        return changed_inputs(
            fixed_achievement_features(frame, config["achievement_beta"]), model, cfg
        )

    general = transform(target)
    general["rank"] = general.ability_score.rank(method="min", ascending=False).astype(int)
    general["last_result_event_id"] = int(last)
    grid = None
    if layouts:
        blocks = []
        for layout in sorted(features.circuit_layout_id.unique()):
            block = transform(target.assign(circuit_layout_id=layout))
            blocks.append(
                block[
                    [
                        "driver_id",
                        "circuit_layout_id",
                        "layout_score",
                        "track_effect",
                        "joint_layout_delta",
                        "joint_layout_events",
                    ]
                ]
            )
        grid = pd.concat(blocks, ignore_index=True)
    return general, grid, bundle


def export_ratings(prepared, predictor, cutoff, output, *, names=None):
    """Write a sealed leaderboard (JSON/CSV) and the driver-by-layout matrix."""
    general, grid, bundle = driver_ratings(prepared, predictor, cutoff)
    output = Path(output)
    columns = [
        "rank",
        "driver_id",
        "constructor_id",
        "ability_score",
        "joint_pure_score",
        "teammate_ability",
        "achievement_bonus",
        "joint_driver_delta",
        "joint_history_events",
    ]
    table = general.sort_values(["rank", "driver_id"])[columns].reset_index(drop=True)
    comparisons = {d: bundle["base"].get(d, {}).get("comparisons", 0) for d in table.driver_id}
    table["teammate_comparisons"] = table.driver_id.map(comparisons).astype(int)
    if names is not None:
        table.insert(2, "driver", table.driver_id.map(names))
    write_frame(output / "ratings.parquet", table)
    table.to_csv(output / "ratings.csv", index=False)
    write_frame(output / "layout_ratings.parquet", grid)
    write_json(
        output / "ratings.json",
        {
            "method": METHOD,
            "cutoff": pd.Timestamp(cutoff).isoformat(),
            "last_result_event_id": int(general.last_result_event_id.iloc[0]),
            "season": int(general.season.iloc[0]),
            "rule_training_max_year": load_predictor(predictor)["tasks"]["top3"]["trace"][
                "training_max_year"
            ],
            "rating_bundle_sha256": bundle["bundle_sha256"],
            "drivers": json.loads(table.to_json(orient="records")),
        },
    )
    return seal(output, kind="ratings", metadata={"method": METHOD})
