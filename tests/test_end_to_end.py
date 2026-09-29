"""Exercise fitted models and prediction from a dataset without target results."""

import json

import numpy as np
import pandas as pd
import pytest

from f1_quali.baselines.course import _best_time, build_course_features
from f1_quali.baselines.rolling import fit_predict, prior_training_rows
from f1_quali.config import FEATURES, load_config
from f1_quali.data.portable import save_dataset
from f1_quali.demo import run_demo, synthetic_dataset
from f1_quali.integrity import verify
from f1_quali.models.pools import fit_pool_models
from f1_quali.pipeline import forecast


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("fitted")
    run_demo(path, method="v6")
    return path


def test_fitted_model_is_independent_of_future_and_first_five(fitted):
    features = pd.read_parquet(fitted / "prepared/features.parquet")
    labels = pd.read_parquet(fitted / "prepared/labels.parquet")
    config = load_config()
    original = json.loads((fitted / "model/model.json").read_text())
    assert original["tasks"]["pole"]["kind"] == "softmax"
    ignored = features.season.ge(2024) | features.season_event_ordinal.lt(6)
    features.loc[ignored, FEATURES] = 999.0
    labels.loc[labels.event_id.isin(features.loc[ignored, "event_id"]), "y_pole"] = np.nan
    rebuilt = fit_pool_models(features, labels, 2024, FEATURES, alpha=config["alpha"])
    assert rebuilt == original


def test_truncated_training_roster_rejected(fitted):
    features = pd.read_parquet(fitted / "prepared/features.parquet")
    labels = pd.read_parquet(fitted / "prepared/labels.parquet")
    eligible = features.season.lt(2024) & features.season_event_ordinal.ge(6)
    features = features.drop(features[eligible].index[0])
    with pytest.raises(ValueError, match="complete roster"):
        fit_pool_models(features, labels, 2024, FEATURES)


def test_forecast_without_target_results(fitted, tmp_path):
    data = synthetic_dataset()
    event = data.events.iloc[-1]
    data.qualifying = data.qualifying[~data.qualifying.event_id.eq(event.event_id)].copy()
    data.events.loc[data.events.event_id.eq(event.event_id), "result_status"] = "future"
    save_dataset(tmp_path / "input", data)
    forecast(
        tmp_path / "input",
        fitted / "model",
        int(event.event_id),
        event.prediction_cutoff.isoformat(),
        tmp_path / "forecast",
    )
    verify(tmp_path / "forecast", "forecast")
    prediction = pd.read_parquet(tmp_path / "forecast/predictions.parquet")
    record = json.loads((tmp_path / "forecast/forecast.json").read_text())
    assert record["cutoff"] == event.prediction_cutoff.isoformat()
    assert record["prospective"] is False  # The synthetic event is historical.
    assert set(prediction.driver_id) == set(
        data.entries.loc[data.entries.event_id.eq(event.event_id), "driver_id"]
    )
    assert [int(prediction[f"hard_{task}"].sum()) for task in ["q2", "q3", "top3", "pole"]] == [
        15,
        10,
        3,
        1,
    ]


def test_rolling_ols_excludes_future_and_unavailable_labels():
    data = synthetic_dataset()
    features = build_course_features(data)
    labels = data.qualifying[["event_id", "driver_id", "label_available_at"]].copy()
    labels["course_target_seconds"] = _best_time(data.qualifying)
    joined = features.merge(labels, on=["event_id", "driver_id"], validate="one_to_one")
    target = features[features.event_id.eq(data.events.event_id.max())]
    original = prior_training_rows(joined, target.iloc[0])
    assert original.event_id.nunique() == 5
    prediction, _ = fit_predict(original, target)
    assert np.isfinite(prediction.predicted_seconds).all()
    excluded = ~joined.index.isin(original.index)
    joined.loc[excluded, "course_target_seconds"] = 9999.0
    pd.testing.assert_frame_equal(prior_training_rows(joined, target.iloc[0]), original)
    first = original.event_id.min()
    joined.loc[joined.event_id.eq(first), "label_available_at"] = target.prediction_cutoff.iloc[
        0
    ] + pd.Timedelta(seconds=1)
    delayed = prior_training_rows(joined, target.iloc[0])
    assert delayed.event_id.nunique() == 4
    with pytest.raises(ValueError, match="five races"):
        fit_predict(delayed, target)
