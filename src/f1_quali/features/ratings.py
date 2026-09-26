"""Join immutable event ratings without downstream feedback."""

import pandas as pd

from f1_quali.ratings.core import estimate, verify_bundle
from f1_quali.ratings.rolling import predictor_bundle


def replace_ratings(features, bundles, variant):
    if features.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("One feature row per entrant required")
    parts = []
    composite = variant.startswith("composite")
    for eid, event in features.groupby("event_id", sort=True):
        raw = bundles[(variant, int(eid))]
        verify_bundle(raw)
        if int(raw["target_event_id"]) != int(eid):
            raise ValueError("Rating snapshot target event mismatch")
        if not event.season.eq(int(raw["target_year"])).all():
            raise ValueError("Rating snapshot year mismatch")
        cutoff = pd.Timestamp(raw["cutoff"])
        if (
            cutoff.tzinfo is None
            or not pd.to_datetime(event.prediction_cutoff, utc=True).gt(cutoff).all()
        ):
            raise ValueError("Rating snapshot reaches target qualifying cutoff")
        maximum = raw.get("diagnostics", {}).get("max_available_at")
        if maximum and pd.Timestamp(maximum) > cutoff:
            raise ValueError("Rating evidence was unavailable at snapshot cutoff")
        bundle = predictor_bundle(raw, composite=composite)
        event = event.copy().reset_index(drop=True)
        replacement = pd.DataFrame(
            [
                estimate(bundle, row.driver_id, row.circuit_layout_id)
                for row in event.itertuples(index=False)
            ]
        )
        for col in replacement:
            event[col] = replacement[col]
        event["rating_cutoff"] = cutoff
        event["rating_bundle_sha256"] = bundle["bundle_sha256"]
        event["rating_source_bundle_sha256"] = raw["bundle_sha256"]
        event["rating_variant"] = variant
        event["rating_target_event_id"] = int(eid)
        event["rating_score_scope"] = (
            "composite_including_car_environment" if composite else "teammate_ability"
        )
        event["rating_uncertainty_scope"] = (
            "ability_only_not_composite_interval" if composite else "conditional_ability"
        )
        event["teammate_ability"] = [
            raw["base"].get(d, {}).get("ability", 0.0) for d in event.driver_id
        ]
        event["achievement_bonus"] = [
            raw["base"].get(d, {}).get("achievement_bonus", 0.0) for d in event.driver_id
        ]
        parts.append(event)
    return pd.concat(parts, ignore_index=True)
