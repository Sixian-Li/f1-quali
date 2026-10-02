"""RM parity boundaries: experience, symmetric priors, temporal isolation and exports."""

import json
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from f1_quali import m1r1, rm
from f1_quali.data.experience import annual_profiles, from_f1db
from f1_quali.data.full_era import save_full_era
from f1_quali.demo import run_demo, synthetic_dataset
from f1_quali.integrity import verify
from f1_quali.models.joint import fit_fixed, fit_top3
from f1_quali.models.pools import fit_pool_models
from f1_quali.ratings.display import map_score
from f1_quali.ratings.rookie import adjusted_prior, gap_variance


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("rm")
    run_demo(path)
    return path


def test_experience_is_prior_main_qualifying_including_pre_2010():
    class DB:
        version, digest = "test", "test-sha"

        def table(self, name):
            return {
                "races": pd.DataFrame({"id": [1, 2, 3],
                                       "date": ["2009-06-01", "2025-05-01", "2026-03-01"]}),
                "drivers": pd.DataFrame({"id": ["a", "b", "reserve"]}),
                "races-qualifying-results": pd.DataFrame({
                    "raceId": [1, 1, 2, 2, 2, 3],
                    "driverId": ["a", "a", "a", "b", "reserve", "a"],
                    "positionText": ["1", "1", "2", "DNS", "NOT_PARTICIPATED", "3"]}),
            }[name]

    frame, source = from_f1db(DB())
    assert len(frame) == 3
    cfg = rm.load_rm_config()["rookie_variance"]
    profiles = annual_profiles(frame, source, ["a", "b"], [2010, 2026], cfg)
    assert profiles.prior_main_q_appearances.tolist() == [1, 0, 2, 0]
    assert profiles[profiles.driver_id.eq("b")].annual_variance_multiplier.eq(8).all()
    past = frame[frame.available_at.lt(pd.Timestamp("2026-01-01", tz="UTC"))]
    pd.testing.assert_frame_equal(
        annual_profiles(frame, source, ["a", "b"], [2026], cfg),
        annual_profiles(past, source, ["a", "b"], [2026], cfg), check_exact=True)
    with pytest.raises(ValueError, match="Missing career"):
        annual_profiles(frame, source, ["unknown"], [2026], cfg)
    with pytest.raises(ValueError, match="main-Q"):
        annual_profiles(frame.assign(session_type="SQ"), source, ["a"], [2026], cfg)


def test_experience_exact_year_boundary_is_excluded():
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({"event_id": [1, 2, 3], "driver_id": ["a"] * 3,
                          "available_at": [cutoff - pd.Timedelta(nanoseconds=1), cutoff,
                                           cutoff + pd.Timedelta(nanoseconds=1)],
                          "session_type": "Q"})
    profile = annual_profiles(frame, {"known_drivers": ["a"]}, ["a"], [2026],
                              rm.load_rm_config()["rookie_variance"])
    assert profile.prior_main_q_appearances.item() == 1


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_experience_timestamp_resolution_does_not_admit_future_rows(unit):
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame({"event_id": [1, 2, 3], "driver_id": ["a"] * 3,
                          "available_at": [cutoff - pd.Timedelta(seconds=1), cutoff,
                                           cutoff + pd.Timedelta(seconds=1)],
                          "session_type": "Q"})
    frame["available_at"] = frame.available_at.astype(f"datetime64[{unit}, UTC]")
    profile = annual_profiles(frame, {"known_drivers": ["a"]}, ["a"], [2026],
                              rm.load_rm_config()["rookie_variance"])
    assert profile.prior_main_q_appearances.item() == 1


def test_rookie_prior_is_symmetric_and_preserves_lifetime_layout():
    keys = [("lifetime", "a", 0), ("annual", "a", 2020),
            ("annual", "a", 2022), ("layout", "a", "track")]
    cfg = {"rho": .5, "innovation_sd": .25}
    precision = np.diag([3., 12., 0., 17.])
    edge = np.array([0., -.25, 1., 0.])
    precision += np.outer(edge, edge) / (.25**2 * 1.25)
    prior = sparse.csc_matrix(precision)
    responses = []
    for gain in [1., 8.]:
        factors = {("a", y): gain for y in range(2020, 2023)}
        actual = adjusted_prior(keys, prior, cfg, factors).toarray()
        np.testing.assert_array_equal(actual[[0, 3]], precision[[0, 3]])
        np.testing.assert_allclose(actual[1:3, 1:3], precision[1:3, 1:3] / gain)
        if gain == 1:
            np.testing.assert_array_equal(actual, precision)
        assert np.linalg.eigvalsh(actual).min() > 0
        design = np.array([0., 0., 1., 0.])
        covariance = np.linalg.inv(actual + np.outer(design, design))
        np.testing.assert_array_equal(covariance @ -design, -(covariance @ design))
        responses.append((covariance @ design)[2])
        factors[("a", 2026)] = 99999
        np.testing.assert_array_equal(adjusted_prior(keys, prior, cfg, factors).toarray(), actual)
    assert responses[1] > responses[0] > 0
    assert gap_variance("a", 2020, 2022, cfg, {("a", 2021): 2., ("a", 2022): 8.}) == (
        .25**2 * (.5**2 * 2 + 8))


def test_default_demo_and_display_export(fitted, tmp_path):
    assert json.loads((fitted / "summary.json").read_text())["method"] == "rm"
    rm.export_ratings(fitted / "prepared", fitted / "model", "2024-12-31T00:00:00Z",
                      tmp_path / "ratings")
    verify(tmp_path / "ratings", "ratings")
    table = pd.read_parquet(tmp_path / "ratings/ratings.parquet")
    scale = json.loads(files("f1_quali").joinpath("resources/rm_display_scale.json").read_text())
    assert (scale["mean"], scale["sd"], scale["cap"]) == (6.5, 2., None)
    assert scale["retrospectiveDisplayOnly"] is True
    np.testing.assert_array_equal(table.mapped_score,
                                  [map_score(v, scale["knots"]) for v in table.ability_score])
    assert table.mapped_score.is_monotonic_decreasing
    assert "mapped_score" not in rm.load_predictor(fitted / "model")["columns"]


def test_forecast_replays_without_target_results(fitted, tmp_path):
    data = synthetic_dataset()
    event = data.events.iloc[-1]
    data.qualifying = data.qualifying[~data.qualifying.event_id.eq(event.event_id)].copy()
    data.events.loc[data.events.event_id.eq(event.event_id), "result_status"] = "future"
    save_full_era(tmp_path / "input", data, pd.DataFrame())
    rm.forecast(tmp_path / "input", fitted / "prepared", fitted / "model", int(event.event_id),
                event.prediction_cutoff.isoformat(), tmp_path / "forecast")
    actual = pd.read_parquet(tmp_path / "forecast/predictions.parquet")
    expected = pd.read_parquet(fitted / "evaluation/predictions.parquet")
    expected = expected[expected.event_id.eq(event.event_id)].reset_index(drop=True)
    # A caller-supplied ISO cutoff may serialize at millisecond resolution;
    # compare its exact UTC instants in the stored history's nanosecond dtype.
    for column in expected:
        if isinstance(expected[column].dtype, pd.DatetimeTZDtype):
            actual[column] = actual[column].astype("datetime64[ns, UTC]")
            expected[column] = expected[column].astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_training_ignores_target_year_and_first_five(fitted):
    prepared = fitted / "prepared"
    cfg = rm.load_rm_config()
    model = rm.load_predictor(fitted / "model")
    labels = pd.read_parquet(prepared / "labels.parquet")
    anchor = pd.read_parquet(prepared / "anchor_features.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    pairs = pd.read_parquet(prepared / "auxiliary_pairs.parquet")
    columns = model["columns"] + ["joint_pace_fast", "joint_pace_slow", "joint_result",
                                  "joint_layout_pace", "joint_layout_result"]
    for frame in [anchor, features]:
        frame.loc[frame.season.ge(2024) | frame.season_event_ordinal.lt(6), columns] = 99.
    labels.loc[labels.season.ge(2024) | labels.season_event_ordinal.lt(6),
               ["y_q2", "y_q3", "y_top3", "y_pole"]] = np.nan
    pairs.loc[pairs.season.ge(2024), "outcome"] *= -1
    base = fit_pool_models(anchor, labels, 2024, model["columns"], alpha=cfg["alpha"])
    base = fit_top3(base, anchor, labels)
    rebuilt = fit_fixed(base, features, labels, pairs, cfg["joint_rules"]["active"],
                        cfg["joint_rules"]["settings"], .6)
    rebuilt["method"] = "rm"
    assert rebuilt == model


def test_missing_experience_rejected_instead_of_assuming_rookies(tmp_path):
    save_full_era(tmp_path / "data", synthetic_dataset(history_events=2), pd.DataFrame())
    with pytest.raises(ValueError, match="career history"):
        rm.prepare(tmp_path / "data", tmp_path / "prepared")


def test_prepared_model_method_mixing_rejected(fitted, tmp_path):
    model = tmp_path / "different-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(m1r1.load_m1r1_config()))
    with pytest.raises(ValueError, match="different configurations"):
        rm.evaluate(fitted / "prepared", model, tmp_path / "output")
