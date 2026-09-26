from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from f1_quali.config import FEATURES
from f1_quali.data.labels import build_labels, build_rating_observations, stage_limits
from f1_quali.models.coordination import decode_nested, project_probabilities
from f1_quali.models.pools import fit_pool_models, predict_pools
from f1_quali.pipeline import event_features
from f1_quali.ratings.pipeline import evidence, rating_snapshot


def test_last_common_stage_and_final_result_both_preserved(data, config):
    cfg = config["rating_years"]["2024"]["config"]
    observations = build_rating_observations(data, cfg)
    for row in observations.itertuples():
        assert row.last_common_stage == min(row.stage_i, row.stage_j)
        assert row.outcome in [-1, 1]
        assert row.pace_gap == getattr(row, f"q{row.last_common_stage}_gap")


def test_missing_source_not_treated_as_elimination(data):
    target = data.qualifying.iloc[-1]
    data.qualifying = data.qualifying[
        ~(
            data.qualifying.event_id.eq(target.event_id)
            & data.qualifying.driver_id.eq(target.driver_id)
        )
    ]
    labels = build_labels(data)
    rows = labels[labels.event_id.eq(target.event_id)]
    assert len(rows) == 60
    assert not rows.complete_all_tasks.any()
    missing = rows[rows.driver_id.eq(target.driver_id)]
    assert missing.y_q3.isna().all()


def test_no_q3_lap_for_advanced_driver_remains_q3(data):
    mask = data.qualifying.position.eq(1)
    data.qualifying.loc[mask, "q3_seconds"] = np.nan
    labels = build_labels(data)
    pole = labels[labels.y_pole & labels.stage.eq("Q3")]
    assert pole.eligible.all() and pole.no_valid_lap.all()


def test_disqualification_keeps_uncertainty(data):
    data.qualifying.loc[data.qualifying.index[-1], "classification_status"] = "DSQ"
    labels = build_labels(data)
    target = labels[labels.event_id.eq(data.events.event_id.max())]
    assert not target.complete_all_tasks.any()


@pytest.mark.parametrize(
    "year,rnd,kind,field,expected",
    [
        (2016, 1, "ELIMINATION", 22, (15, 8)),
        (2016, 3, "KNOCKOUT", 22, (16, 10)),
        (2024, 3, "KNOCKOUT", 19, (15, 10)),
        (2026, 1, "KNOCKOUT", 22, (16, 10)),
    ],
)
def test_quotas_use_reviewed_rules(year, rnd, kind, field, expected):
    assert stage_limits(year, rnd, kind, field)[:2] == expected


def test_missing_2026_field_size_uses_registered_default():
    assert stage_limits(2026, 1, "KNOCKOUT", np.nan)[:2] == (16, 10)
    assert stage_limits(2026, 1, "KNOCKOUT", pd.NA)[:2] == (16, 10)


def test_rating_gap_threshold_must_match_across_years(data, config):
    config = deepcopy(config)
    config["rating_years"]["2024"]["config"]["maximum_absolute_gap_log_pct"] = 0.05
    with pytest.raises(ValueError, match="identical across rating years"):
        evidence(data, config)


def test_projection_and_hard_lists_have_nested_exact_quotas():
    rng = np.random.default_rng(9)
    probabilities = project_probabilities(rng.uniform(size=(22, 4)), [16, 10, 3, 1])
    np.testing.assert_allclose(probabilities.sum(axis=0), [16, 10, 3, 1], atol=1e-8)
    assert (np.diff(probabilities, axis=1) < 1e-8).all()
    frame = pd.DataFrame(
        {
            "event_id": 1,
            "driver_id": [f"d{i}" for i in range(22)],
            **{f"p_{t}": probabilities[:, i] for i, t in enumerate(["q2", "q3", "top3", "pole"])},
        }
    )
    decoded = decode_nested(frame, 16, 10)
    assert set(decoded.predicted_rank) == set(range(1, 23))
    assert [int(decoded[f"hard_{t}"].sum()) for t in ["q2", "q3", "top3", "pole"]] == [16, 10, 3, 1]


def test_early_season_fallback_and_rating_immutability(data, config):
    event = data.events.iloc[-1]
    pairs, achievements = evidence(data, config)
    bundle = rating_snapshot(event, pairs, achievements, config)
    before = deepcopy(bundle)
    features = event_features(data, event, bundle, config)
    labels = build_labels(data)
    model = fit_pool_models(features, labels, 2024, FEATURES, alpha=config["alpha"])
    assert all(v["kind"] == "quota_prior" for v in model["tasks"].values())
    pred = predict_pools(model, features, 15, 10)
    assert len(pred) == 20
    assert bundle == before


def test_model_training_ignores_future_and_first_five(data, config):
    event = data.events.iloc[2]
    pairs, achievements = evidence(data, config)
    features = event_features(
        data, event, rating_snapshot(event, pairs, achievements, config), config
    )
    labels = build_labels(data)
    model = fit_pool_models(features, labels, 2024, FEATURES)
    features.loc[:, "ability"] = 999.0
    labels.loc[labels.season.eq(2024), "relative_gap"] = 999.0
    assert fit_pool_models(features, labels, 2024, FEATURES) == model
