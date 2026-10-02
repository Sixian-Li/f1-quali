"""Refit rating states before each event using fixed annual parameters."""

from dataclasses import replace

import pandas as pd

from f1_quali.data.labels import build_rating_observations
from f1_quali.ratings.achievements import qualifying_achievements
from f1_quali.ratings.core import canonical_hash
from f1_quali.ratings.rolling import fit_ratings, freeze_bundle, with_achievement_bonus


def evidence(data, config, early=None, *, cutoff=None, target_event_id=None):
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        events = data.events[
            data.events.prediction_cutoff.lt(cutoff) & data.events.event_id.ne(target_event_id)
        ]
        qualifying = data.qualifying[
            data.qualifying.event_id.isin(events.event_id)
            & data.qualifying.label_available_at.le(cutoff)
        ].copy()
        data = replace(
            data,
            events=events.copy(),
            qualifying=qualifying,
            entries=data.entries[data.entries.event_id.isin(events.event_id)].copy(),
        )
    # Teammate observations are built once for all years, so the quality
    # threshold must be shared by every registered year rather than silently
    # taken from the first one.
    thresholds = {
        parameters["config"]["maximum_absolute_gap_log_pct"]
        for parameters in config["rating_years"].values()
    }
    if len(thresholds) != 1:
        raise ValueError("maximum_absolute_gap_log_pct must be identical across rating years")
    rating_config = next(iter(config["rating_years"].values()))["config"]
    pairs = build_rating_observations(data, rating_config)
    achievements = qualifying_achievements(data)
    if pairs.empty:
        pairs = pd.DataFrame(
            columns=[
                "event_id",
                "season",
                "constructor_id",
                "driver_i",
                "driver_j",
                "circuit_layout_id",
                "prediction_cutoff",
                "available_at",
                "outcome_usable",
                "outcome",
                "pace_gap",
                "rank_gap_abs_normalized",
                "session_type",
            ]
        )
    if early is not None:
        if not early[0].season.lt(2016).all() or not early[1].season.lt(2016).all():
            raise ValueError("Warmup cannot add current-period evidence")
        pairs = pd.concat([early[0], pairs], ignore_index=True)
        achievements = pd.concat([early[1], achievements], ignore_index=True)
    return pairs, achievements


def rating_snapshot(event, pairs, achievements, config, *, cutoff=None, annual_variance=None):
    year = int(event.season)
    parameters = config["rating_years"].get(str(year))
    if parameters is None:
        raise ValueError(f"No registered rating parameters for {year}; review a new version first")
    cutoff = pd.Timestamp(event.prediction_cutoff if cutoff is None else cutoff)
    if cutoff.tzinfo is None or pd.isna(cutoff) or cutoff > pd.Timestamp(event.prediction_cutoff):
        raise ValueError("Explicit cutoff no later than target qualifying required")
    if cutoff < pd.Timestamp(parameters["parameters_available_before"]):
        raise ValueError("Annual rating parameters were not yet available at cutoff")
    fit = fit_ratings(
        pairs,
        cutoff - pd.Timedelta(nanoseconds=1),
        parameters["config"],
        target_event_id=int(event.event_id),
        penalty=parameters["penalty"],
        achievements=achievements,
        annual_variance=annual_variance,
    )
    bundle = freeze_bundle(
        fit,
        year,
        parameters["calibration_slope"],
        variant="composite_early",
        selection_provenance={
            "parameters_sha256": canonical_hash(parameters),
            "downstream_metrics_used": False,
            "policy": "frozen_annual_parameters_eventwise_states",
        },
    )
    if annual_variance is not None:
        from f1_quali.ratings.rookie import adjust_forecast_variance

        bundle = adjust_forecast_variance(bundle, annual_variance, canonical_hash)
    return with_achievement_bonus(
        bundle,
        parameters["achievement_beta"],
        calibration_slope=parameters["composite_calibration_slope"],
    )
