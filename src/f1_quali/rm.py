"""RM configuration and entry point for the shared causal joint-rating pipeline."""

import json
from importlib.resources import files
from pathlib import Path

from f1_quali import m1r1

METHOD = "rm"


def load_rm_config(path=None):
    config = json.loads(Path(path).read_text() if path else
                        files("f1_quali").joinpath("resources/rm.json").read_text())
    if (config.get("method") != METHOD or config["first_optimized_ordinal"] != 6
            or config["achievement_beta"] != 0.6
            or {p["achievement_beta"] for p in config["rating_years"].values()} != {0.6}
            or config["joint_rules"]["active"] != [0, 1, 2, 4, 5]):
        raise ValueError("Expected the fixed-beta RM configuration")
    if config["rookie_variance"] != {"boost": 7.0, "half_life_appearances": 20.0,
                                      "timing": "before_January_1", "session_type": "Q"}:
        raise ValueError("RM uses the reviewed year-start rookie variance policy")
    if config["achievement_memory"] != {"admission": "individual", "mean_half_life": 0.25,
                                         "mass_half_life": 1.0, "floor": 0.0, "prior_mass": 20.0}:
        raise ValueError("RM uses three-month mean and one-year reliability memory")
    return config


def prepare(dataset, output, config=None, *, progress=None):
    return m1r1.prepare(dataset, output, config or load_rm_config(), progress=progress)


# Artifact metadata dispatches the remaining shared stages. Configuration checks
# reject mixing a prepared history and an annual model from different methods.
train = m1r1.train
evaluate = m1r1.evaluate
forecast = m1r1.forecast
driver_ratings = m1r1.driver_ratings
export_ratings = m1r1.export_ratings
load_predictor = m1r1.load_predictor
