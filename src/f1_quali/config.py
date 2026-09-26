"""The selected method and annual independently determined rating parameters."""

import json
from importlib.resources import files
from pathlib import Path

from f1_quali.features.context import CURRENT

FEATURES = CURRENT + ["ability", "track_effect"]
METHOD = "memory_h12_mean_median_50_50"


def load_config(path=None):
    text = (
        Path(path).read_text()
        if path
        else files("f1_quali").joinpath("resources/v6.json").read_text()
    )
    config = json.loads(text)
    if config["method"] != METHOD or config["first_optimized_ordinal"] != 6:
        raise ValueError("This release implements the frozen v6 method")
    return config
