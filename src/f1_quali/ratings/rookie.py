"""RM's zero-mean rookie variance policy; lifetime and layout priors are preserved."""

import copy
from itertools import pairwise

import numpy as np
from scipy import sparse


def gap_variance(driver, before, after, cfg, factors):
    if after < before or not 0 <= cfg["rho"] < 1 or cfg["innovation_sd"] <= 0:
        raise ValueError("Valid annual transition and positive innovation scale required")
    return cfg["innovation_sd"] ** 2 * sum(
        cfg["rho"] ** (2 * (after - year)) * factors[(driver, year)]
        for year in range(before + 1, after + 1)
    )


def adjusted_prior(keys, prior, cfg, factors):
    """Replace annual blocks only; preserve lifetime/layout entries exactly."""
    lookup = {key: i for i, key in enumerate(keys)}
    groups = {}
    for kind, driver, year in keys:
        if kind == "annual":
            groups.setdefault(driver, []).append(year)
    relevant = [
        factors[(d, y)] for d, years in groups.items() for y in range(min(years), max(years) + 1)
    ]
    if any(not np.isfinite(g) or g < 1 for g in relevant):
        raise ValueError("Finite variance multipliers at least one required")
    if all(g == 1 for g in relevant):
        return prior.copy()
    result = prior.tolil(copy=True)
    rho, sigma = cfg["rho"], cfg["innovation_sd"]
    for driver, years in groups.items():
        years.sort()
        ix = [lookup[("annual", driver, year)] for year in years]
        # This submatrix contains only this driver's AR prior, no other priors.
        for i in ix:
            for j in ix:
                result[i, j] = 0.0
        result[ix[0], ix[0]] = (1 - rho**2) / (sigma**2 * factors[(driver, years[0])])
        for before, after in pairwise(years):
            i, j = lookup[("annual", driver, before)], lookup[("annual", driver, after)]
            persistence = rho ** (after - before)
            precision = 1 / gap_variance(driver, before, after, cfg, factors)
            result[i, i] += persistence**2 * precision
            result[i, j] -= persistence * precision
            result[j, i] -= persistence * precision
            result[j, j] += precision
    return sparse.csc_matrix(result)


def adjust_forecast_variance(bundle, factors, canonical_hash):
    """Replace constant-variance AR gap uncertainty after the original freezer."""
    out = copy.deepcopy(bundle)
    cfg, target = out["config"], out["target_year"]
    lookup = {d: i for i, d in enumerate(out["base_joint_covariance"]["drivers"])}
    for driver, row in out["base"].items():
        variance = gap_variance(driver, row["last_observed_year"], target, cfg, factors)
        difference = variance - row["innovation_variance"]
        row["innovation_variance"] = float(variance)
        row["variance"] += difference
        row["annual_variance"] += difference
        index = lookup[driver]
        out["base_joint_covariance"]["matrix"][index][index] += difference
    out["diagnostics"]["prior"] = "original_lifetime_layout_plus_career_scaled_annual_variance"
    out["diagnostics"]["career_variance_policy"] = (
        "year_start_metadata;zero_mean;symmetric;per_year_gap_variances"
    )
    out.pop("bundle_sha256", None)
    out["bundle_sha256"] = canonical_hash(out)
    return out
