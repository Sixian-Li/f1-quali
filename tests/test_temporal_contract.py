from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from f1_quali.config import FEATURES
from f1_quali.features.evidence import eligible_practice
from f1_quali.pipeline import event_features
from f1_quali.ratings.pipeline import evidence, rating_snapshot
from f1_quali.ratings.rolling import recency_weights


def inputs(data, config, event, cutoff=None):
    cutoff = event.prediction_cutoff if cutoff is None else cutoff
    pairs, achievements = evidence(data, config, cutoff=cutoff, target_event_id=int(event.event_id))
    bundle = rating_snapshot(event, pairs, achievements, config, cutoff=cutoff)
    return bundle, event_features(data, event, bundle, config, cutoff=cutoff)


def test_target_and_future_results_do_not_change_input(data, config):
    event = data.events.iloc[-2]
    original, features = inputs(data, config, event)
    poisoned = deepcopy(data)
    target = poisoned.qualifying.event_id.ge(event.event_id)
    poisoned.qualifying.loc[target, "position"] = -999
    poisoned.qualifying.loc[target, ["q1_seconds", "q2_seconds", "q3_seconds"]] = 9999.0
    own = poisoned.qualifying.event_id.eq(event.event_id)
    poisoned.qualifying.loc[own, "label_available_at"] = event.prediction_cutoff - pd.Timedelta(
        days=10
    )
    repeated, actual = inputs(poisoned, config, event)
    assert original == repeated
    pd.testing.assert_frame_equal(features, actual, check_exact=True)


def test_late_past_labels_are_excluded(data, config):
    event = data.events.iloc[-1]
    prior_id = int(data.events.iloc[-2].event_id)
    data.qualifying.loc[data.qualifying.event_id.eq(prior_id), "label_available_at"] = (
        event.prediction_cutoff + pd.Timedelta(hours=1)
    )
    before, features = inputs(data, config, event)
    data.qualifying.loc[data.qualifying.event_id.eq(prior_id), "position"] = 999
    after, actual = inputs(data, config, event)
    assert before == after
    pd.testing.assert_frame_equal(features[FEATURES], actual[FEATURES], check_exact=True)


def test_practice_after_qualifying_and_sprint_are_not_features(data, config):
    event = data.events.iloc[-1]
    mask = data.practice.event_id.eq(event.event_id) & data.practice.session_type.eq("FP2")
    data.practice.loc[mask, "scheduled_start_utc"] = event.prediction_cutoff + pd.Timedelta(days=1)
    data.practice.loc[mask, "available_at"] = event.prediction_cutoff + pd.Timedelta(
        days=1, hours=2
    )
    _, before = inputs(data, config, event)
    data.practice.loc[mask, "lap_seconds"] = 0.001
    fake = data.practice[mask].assign(
        session_type="SQ",
        session_id="target:SQ",
        available_at=event.prediction_cutoff - pd.Timedelta(hours=1),
        scheduled_start_utc=event.prediction_cutoff - pd.Timedelta(hours=3),
    )
    data.practice = pd.concat([data.practice, fake], ignore_index=True)
    _, after = inputs(data, config, event)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES], check_exact=True)


def test_practice_substitute_is_not_the_regular_driver(data, config):
    event = data.events.iloc[-1]
    driver = data.entries.iloc[0].driver_id
    mask = data.practice.event_id.eq(event.event_id) & data.practice.driver_id.eq(driver)
    data.practice.loc[mask, "driver_id"] = "synthetic-substitute"
    _, features = inputs(data, config, event)
    assert features.loc[features.driver_id.eq(driver), "practice_missing"].item() == 1
    assert "synthetic-substitute" not in set(features.driver_id)


def test_old_season_practice_does_not_enter_current_features(data, config):
    event = data.events.iloc[-1]
    _, before = inputs(data, config, event)
    data.practice.loc[data.practice.season.lt(event.season), "lap_seconds"] = 9999.0
    _, after = inputs(data, config, event)
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES], check_exact=True)


def test_missing_laps_and_no_practice_use_explicit_fallback(data, config):
    event = data.events.iloc[-1]
    mask = data.practice.event_id.eq(event.event_id)
    data.practice.loc[mask, "lap_seconds"] = np.nan
    assert eligible_practice(data, int(event.event_id), event.prediction_cutoff).empty
    _, features = inputs(data, config, event)
    assert features.practice_pct.eq(0.5).all()
    assert features.practice_missing.eq(1).all()
    assert np.isfinite(features[FEATURES]).all().all()


def test_scores_update_per_event_and_keep_positive_old_weight(data, config):
    first, _ = inputs(data, config, data.events.iloc[0])
    second, _ = inputs(data, config, data.events.iloc[1])
    assert not first["base"]
    assert second["base"] and second["diagnostics"]["pairs"] == 10
    assert np.all(recency_weights([0, 1, 20, 100], 0.5, 4) >= 0.5)
    assert recency_weights([0], 0.5, 4)[0] == 1


@pytest.mark.parametrize("cutoff", ["2024-01-01", "2025-01-01T00:00Z", "2022-01-01T00:00Z"])
def test_bad_cutoff_rejected(data, config, cutoff):
    event = data.events.iloc[-1]
    pairs, achievements = evidence(data, config)
    with pytest.raises(ValueError):
        rating_snapshot(event, pairs, achievements, config, cutoff=cutoff)


def test_late_entry_rejected(data, config):
    event = data.events.iloc[-1]
    mask = data.entries.event_id.eq(event.event_id)
    data.entries.loc[mask, "entry_known_at"] = event.prediction_cutoff + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="future entrant"):
        inputs(data, config, event)
