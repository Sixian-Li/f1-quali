"""m1r1: individual achievement memory, joint rules, causality and CLI flow."""

import json

import numpy as np
import pandas as pd
import pytest

from f1_quali import m1r1
from f1_quali.cli import main
from f1_quali.data.full_era import (
    admit_early_pace,
    early_limits,
    phase_practice,
    save_full_era,
)
from f1_quali.demo import run_demo, synthetic_dataset
from f1_quali.integrity import verify
from f1_quali.models.joint import (
    fit_fixed,
    fit_top3,
    history_summaries,
    quota_probabilities,
)
from f1_quali.models.pools import fit_pool_models
from f1_quali.ratings.memory import (
    achievement_statistics,
    eligible_individual,
    individual_achievements,
    rebuild_achievement,
)
from f1_quali.ratings.pipeline import rating_snapshot


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    path = tmp_path_factory.mktemp("m1r1")
    run_demo(path)
    return path


@pytest.fixture(scope="module")
def m1r1_config():
    return m1r1.load_m1r1_config()


def test_config_is_the_locked_selection(m1r1_config):
    assert m1r1_config["research_lock"] == "m1r1_full_era_de17567786cf"
    assert set(m1r1_config["rating_years"]) == {str(y) for y in range(2010, 2027)}
    assert m1r1_config["achievement_memory"] == {
        "admission": "individual",
        "mean_half_life": 1.0,
        "mass_half_life": 1.0,
        "floor": 0.0,
        "prior_mass": 20.0,
    }
    assert 3 not in m1r1_config["joint_rules"]["active"]


def test_other_entrant_disqualification_keeps_personal_result():
    data = synthetic_dataset(history_events=3, target_events=1)
    first = data.qualifying.event_id.min()
    row = data.qualifying.event_id.eq(first) & data.qualifying.position.eq(20)
    data.qualifying.loc[row, ["classification_status", "label_usable"]] = ["DSQ", False]
    rows, quarantined = individual_achievements(data)
    assert rows[rows.event_id.eq(first)].shape[0] == 19
    assert quarantined.empty
    duplicate = data.qualifying.event_id.eq(first) & data.qualifying.position.eq(2)
    data.qualifying.loc[duplicate, "position"] = 1
    rows, quarantined = individual_achievements(data)
    assert len(quarantined) == 2 and not rows[rows.event_id.eq(first)].position.eq(1).any()


def test_personal_evidence_excludes_target_and_future():
    data = synthetic_dataset(history_events=4, target_events=1)
    rows, _ = individual_achievements(data)
    target = data.events.iloc[2]
    admitted = eligible_individual(rows, target.prediction_cutoff, int(target.event_id))
    assert set(admitted.event_id) == set(data.events.event_id.iloc[:2])
    late = rows.copy()
    late.loc[late.event_id.eq(data.events.event_id.iloc[1]), "available_at"] = (
        target.prediction_cutoff + pd.Timedelta(seconds=1)
    )
    admitted = eligible_individual(late, target.prediction_cutoff, int(target.event_id))
    assert set(admitted.event_id) == {data.events.event_id.iloc[0]}


def test_memory_has_no_permanent_floor():
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    rows = pd.DataFrame(
        {
            "driver_id": ["a", "a"],
            "position": [1, 20],
            "field_size": [20, 20],
            "age_at": [cutoff - pd.Timedelta(days=3653), cutoff - pd.Timedelta(days=1)],
        }
    )
    summary, parts = achievement_statistics(
        rows, cutoff, floor=0.0, half_life_years=1.0, prior_mass=20.0
    )
    assert summary["a"]["weight_mass"] == pytest.approx(1.0, abs=2e-3)
    assert summary["a"]["bonus_unit"] == 0.0  # the old pole no longer lifts a recent last place


def test_rebuild_changes_only_the_achievement(m1r1_config):
    data = synthetic_dataset(history_events=5, target_events=1)
    pairs, complete, individual, _ = m1r1.rating_evidence(data, pd.DataFrame(), m1r1_config)
    event = data.events.iloc[-1]
    anchor = rating_snapshot(event, pairs, complete, m1r1_config)
    rebuilt = rebuild_achievement(anchor, individual, m1r1_config["achievement_memory"], "x")
    assert rebuilt["layout"] == anchor["layout"]
    assert {d: v["ability"] for d, v in rebuilt["base"].items()} == {
        d: v["ability"] for d, v in anchor["base"].items()
    }
    assert rebuilt["achievement_memory"]["admission"] == "individual"
    assert rebuilt["bundle_sha256"] != anchor["bundle_sha256"]


def test_quota_probabilities_sum_to_capacity():
    groups = np.repeat(np.arange(3), 20)
    logits = np.random.default_rng(1).normal(size=60) * 3
    p, _ = quota_probabilities(logits, groups, np.full(3, 3.0))
    np.testing.assert_allclose(np.bincount(groups, weights=p), 3, atol=1e-9)
    assert np.all((p > 0) & (p < 1))


def test_histories_ignore_target_and_future_innovations(fitted, m1r1_config):
    features = pd.read_parquet(fitted / "prepared/features.parquet")
    innovations = pd.read_parquet(fitted / "prepared/innovations.parquet")
    cfg = m1r1_config["joint_rules"]["settings"]
    target = features[features.event_id.eq(features.event_id.max())]
    cutoff = target.prediction_cutoff.iloc[0]
    poisoned = innovations.copy()
    mask = poisoned.age_at.ge(cutoff) | poisoned.event_id.eq(target.event_id.iloc[0])
    poisoned.loc[mask, ["pace_innovation", "result_innovation"]] = 12345.0
    poisoned.loc[mask, "available_at"] = cutoff - pd.Timedelta(days=5)
    pd.testing.assert_frame_equal(
        history_summaries(target, innovations, cfg),
        history_summaries(target, poisoned, cfg),
        check_exact=True,
    )


def test_annual_model_ignores_target_year_and_first_five(fitted, m1r1_config):
    prepared = fitted / "prepared"
    labels = pd.read_parquet(prepared / "labels.parquet")
    anchor = pd.read_parquet(prepared / "anchor_features.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    pairs = pd.read_parquet(prepared / "auxiliary_pairs.parquet")
    original = json.loads((fitted / "model/model.json").read_text())
    rules = m1r1_config["joint_rules"]
    columns = original["columns"] + [
        "joint_pace_fast",
        "joint_pace_slow",
        "joint_result",
        "joint_layout_pace",
        "joint_layout_result",
    ]
    for frame in [anchor, features]:
        ignored = frame.season.ge(2024) | frame.season_event_ordinal.lt(6)
        frame.loc[ignored, columns] = 9.0
    ignored = labels.season.ge(2024) | labels.season_event_ordinal.lt(6)
    labels.loc[ignored, ["y_q2", "y_q3", "y_top3", "y_pole"]] = np.nan
    pairs.loc[pairs.season.ge(2024), "outcome"] *= -1
    base = fit_pool_models(anchor, labels, 2024, original["columns"], alpha=m1r1_config["alpha"])
    base = fit_top3(base, anchor, labels)
    rebuilt = fit_fixed(base, features, labels, pairs, rules["active"], rules["settings"], 0.6)
    rebuilt["method"] = "m1r1"
    assert rebuilt == original
    assert original["joint_rule"]["parameters"][3] == 0.0


def test_beta_cannot_be_trained(fitted, m1r1_config):
    prepared = fitted / "prepared"
    labels = pd.read_parquet(prepared / "labels.parquet")
    features = pd.read_parquet(prepared / "features.parquet")
    model = json.loads((fitted / "model/model.json").read_text())
    rules = m1r1_config["joint_rules"]
    with pytest.raises(ValueError, match="fixed"):
        fit_fixed(model, features, labels, labels, [0, 3], rules["settings"], 0.6)


def test_forecast_without_target_results(fitted, tmp_path):
    data = synthetic_dataset()
    event = data.events.iloc[-1]
    data.qualifying = data.qualifying[~data.qualifying.event_id.eq(event.event_id)].copy()
    data.events.loc[data.events.event_id.eq(event.event_id), "result_status"] = "future"
    save_full_era(tmp_path / "input", data, pd.DataFrame())
    m1r1.forecast(
        tmp_path / "input",
        fitted / "prepared",
        fitted / "model",
        int(event.event_id),
        event.prediction_cutoff.isoformat(),
        tmp_path / "forecast",
    )
    verify(tmp_path / "forecast", "forecast")
    prediction = pd.read_parquet(tmp_path / "forecast/predictions.parquet")
    stored = pd.read_parquet(fitted / "evaluation/predictions.parquet")
    stored = stored[stored.event_id.eq(event.event_id)].reset_index(drop=True)
    # Same cutoff and history: identical to the historical replay of that event.
    np.testing.assert_allclose(prediction.p_top3, stored.p_top3, rtol=0, atol=1e-12)
    assert [int(prediction[f"hard_{t}"].sum()) for t in ["q2", "q3", "top3", "pole"]] == [
        15,
        10,
        3,
        1,
    ]


def test_forecast_rejects_stale_prepared_history(fitted, tmp_path):
    data = synthetic_dataset(target_events=8)
    event = data.events.iloc[-1]
    data.qualifying = data.qualifying[~data.qualifying.event_id.eq(event.event_id)].copy()
    save_full_era(tmp_path / "input", data, pd.DataFrame())
    with pytest.raises(ValueError, match="re-run prepare"):
        m1r1.forecast(
            tmp_path / "input",
            fitted / "prepared",
            fitted / "model",
            int(event.event_id),
            event.prediction_cutoff.isoformat(),
            tmp_path / "forecast",
        )


def test_ratings_export_and_cli(fitted, tmp_path, capsys):
    cutoff = "2024-12-31T00:00:00+00:00"
    main(
        [
            "ratings",
            "--prepared",
            str(fitted / "prepared"),
            "--model",
            str(fitted / "model"),
            "--cutoff",
            cutoff,
            "--output",
            str(tmp_path / "ratings"),
        ]
    )
    verify(tmp_path / "ratings", "ratings")
    table = pd.read_parquet(tmp_path / "ratings/ratings.parquet")
    assert len(table) == 20 and table["rank"].min() == 1
    assert table.ability_score.is_monotonic_decreasing
    layouts = pd.read_parquet(tmp_path / "ratings/layout_ratings.parquet")
    assert layouts.circuit_layout_id.nunique() == 3
    main(
        [
            "evaluate",
            "--prepared",
            str(fitted / "prepared"),
            "--model",
            str(fitted / "model"),
            "--output",
            str(tmp_path / "evaluation"),
        ]
    )
    assert "top3_overlap" in capsys.readouterr().out


def test_early_rules_and_phase_gate():
    assert early_limits(2013, 1, "KNOCKOUT")[:2] == (16, 10)
    assert early_limits(2014, 17, "KNOCKOUT")[:2] == (14, 10)
    with pytest.raises(ValueError):
        early_limits(2016, 1, "KNOCKOUT")
    data = synthetic_dataset(history_events=2, target_events=1)
    event = data.events.iloc[0]
    with pytest.raises(ValueError, match="modern"):
        phase_practice(data, int(event.event_id), event.prediction_cutoff)


def test_early_supplement_rejects_changed_pairs():
    from importlib.resources import files

    supplement = json.loads(
        files("f1_quali").joinpath("resources/early_pace_supplement.json").read_text()
    )
    assert len(supplement["admissions"]) == 136
    first = supplement["admissions"][0]
    pairs = pd.DataFrame(
        [
            {
                **{k: first[k] for k in ["event_id", "season", "constructor_id"]},
                "driver_i": first["driver_i"],
                "driver_j": first["driver_j"],
                "stage_i": first["previous_stage_i"],
                "stage_j": first["previous_stage_j"],
                "last_common_stage": first["previous_last_common_stage"],
                "pace_usable": True,  # already usable: the reviewed addition must refuse
                "outcome_usable": True,
                "rating_warmup_eligible": True,
                "source_version": "x",
                "source_sha256": "y",
            }
        ]
    )
    with pytest.raises(ValueError):
        admit_early_pace(pairs, pd.DataFrame(), supplement)


@pytest.mark.parametrize("name", ["m1r1", "v6"])
def test_repository_configs_match_packaged_resources(name):
    from importlib.resources import files
    from pathlib import Path

    repository = Path(__file__).resolve().parents[1] / "configs" / f"{name}.json"
    packaged = files("f1_quali").joinpath(f"resources/{name}.json").read_text()
    assert json.loads(repository.read_text()) == json.loads(packaged)
