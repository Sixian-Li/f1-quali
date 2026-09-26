"""Prior-event lifetime/annual teammate ratings and bounded qualifying bonus."""

import copy
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import ndtr

from f1_quali.data.core import fingerprint
from f1_quali.ratings.core import canonical_hash, estimate, objective, verify_bundle
from f1_quali.ratings.core import compare as compare_reference

__all__ = ["compare", "estimate", "fit_ratings", "freeze_bundle", "verify_bundle"]


def transition_variance(rho, innovation_sd, years):
    """Innovation variance after `years` discrete annual transitions."""
    if not 0 <= rho < 1 or innovation_sd <= 0 or years < 0:
        raise ValueError(
            "A stationary AR prior requires 0 <= rho < 1 and positive scales and nonnegative gaps"
        )
    return innovation_sd**2 * (1 - rho ** (2 * years)) / (1 - rho**2)


def _structure(pairs, cfg):
    keys = set()
    for row in pairs.itertuples():
        for driver in (row.driver_i, row.driver_j):
            keys.update(
                [
                    ("lifetime", driver, 0),
                    ("annual", driver, int(row.season)),
                    ("layout", driver, row.circuit_layout_id),
                ]
            )
    keys = sorted(keys, key=str)
    lookup = {key: i for i, key in enumerate(keys)}
    rr, cc, vv = [], [], []
    for index, row in enumerate(pairs.itertuples()):
        for driver, sign in [(row.driver_i, 1), (row.driver_j, -1)]:
            for key in [
                ("lifetime", driver, 0),
                ("annual", driver, int(row.season)),
                ("layout", driver, row.circuit_layout_id),
            ]:
                rr.append(index)
                cc.append(lookup[key])
                vv.append(sign)
    design = sparse.csr_matrix((vv, (rr, cc)), shape=(len(pairs), len(keys)))
    diag = np.zeros(len(keys))
    by_driver = {}
    for index, (kind, driver, year) in enumerate(keys):
        if kind == "lifetime":
            diag[index] = 1 / cfg["lifetime_sd"] ** 2
        elif kind == "layout":
            diag[index] = cfg["prior_weight"] / cfg["layout_sd"] ** 2
        else:
            by_driver.setdefault(driver, []).append(("annual", driver, year))
    rho, innovation = cfg["rho"], cfg["innovation_sd"]
    stationary_variance = innovation**2 / (1 - rho**2)
    rr, cc, vv = [], [], []
    edge = 0
    for annual_keys in by_driver.values():
        annual_keys.sort(key=lambda key: key[2])
        diag[lookup[annual_keys[0]]] = 1 / stationary_variance
        for before, after in pairwise(annual_keys):
            years = after[2] - before[2]
            sd = np.sqrt(transition_variance(rho, innovation, years))
            rr.extend([edge, edge])
            cc.extend([lookup[before], lookup[after]])
            vv.extend([-(rho**years) / sd, 1 / sd])
            edge += 1
    edges = sparse.csr_matrix((vv, (rr, cc)), shape=(edge, len(keys)))
    prior = (sparse.diags(diag, format="csc") + edges.T @ edges).tocsc()
    return keys, lookup, design, prior


def fit_ratings(observations, cutoff, cfg, *, target_event_id, penalty=0.5, achievements=None):
    """Fit independent historical teammate evidence with a fixed result penalty."""
    cfg = copy.deepcopy(cfg)
    for key in ["lifetime_sd", "innovation_sd", "layout_sd", "prior_weight"]:
        if cfg[key] <= 0:
            raise ValueError(f"Rating prior scale must be positive: {key}")
    transition_variance(cfg["rho"], cfg["innovation_sd"], 1)
    if penalty <= 0:
        raise ValueError("Long-term rating must include final qualifying result")
    cutoff = _cutoff(cutoff)
    pairs = eligible_pairs(observations, cutoff, target_event_id)
    achievements = eligible_achievements(achievements, cutoff, target_event_id)
    floor = cfg.get("floor", 1.0)
    half_life = cfg.get("half_life_years", 4.0)
    ages = (cutoff - pairs.age_at).dt.total_seconds().to_numpy() / (365.25 * 86400)
    pairs["recency_weight"] = recency_weights(ages, floor, half_life)
    if len(pairs) and pairs.season.gt(cutoff.year).any():
        raise ValueError("A future season cannot enter a historical lifetime rating")
    keys, lookup, design, prior = _structure(pairs, cfg)
    event_weights = (
        pairs.recency_weight.to_numpy()
        / pairs.groupby("event_id").event_id.transform("size").to_numpy()
    )
    pairs["pair_weight"] = event_weights
    usable = pairs.pace_gap.notna().to_numpy()
    pace_rows = np.flatnonzero(usable)
    gaps = pairs.pace_gap.to_numpy(dtype=float)[usable]
    pace_weights = event_weights[usable]
    outcomes = pairs.outcome.to_numpy(dtype=float)
    result_weights = penalty * event_weights * (1 + pairs.rank_gap_abs_normalized.to_numpy())
    args = design, pace_rows, gaps, pace_weights, outcomes, result_weights, prior, cfg
    theta = np.zeros(len(keys))
    iterations, gradient_max, loss = 0, 0.0, 0.0
    hessian = prior
    for iterations in range(1, 301) if len(keys) else []:
        loss, gradient, hessian = objective(theta, *args)
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max < 2e-7:
            break
        step = splu(hessian).solve(-gradient)
        fraction, accepted = 1.0, False
        for _ in range(30):
            candidate = theta + fraction * step
            next_loss = objective(candidate, *args)[0]
            if next_loss <= loss + 1e-4 * fraction * float(gradient @ step):
                theta, accepted = candidate, True
                break
            fraction *= 0.5
        if not accepted:
            if gradient_max < 1e-5:
                break
            raise RuntimeError("Long-term rating objective line search failed")
    if len(keys):
        loss, gradient, hessian = objective(theta, *args)
        gradient_max = float(np.max(np.abs(gradient)))
    if gradient_max >= 1e-5:
        raise RuntimeError(f"Long-term rating optimizer not converged: {gradient_max}")
    return {
        "theta": theta,
        "keys": keys,
        "lookup": lookup,
        "hessian": hessian,
        "pairs": pairs,
        "achievements": achievements,
        "config": cfg,
        "target_event_id": int(target_event_id),
        "penalty": float(penalty),
        "cutoff": cutoff.isoformat(),
        "diagnostics": {
            "converged": True,
            "iterations": iterations,
            "max_gradient": gradient_max,
            "objective": loss,
            "pairs": len(pairs),
            "pace_terms": len(gaps),
            "events": int(pairs.event_id.nunique()),
            "latent_nodes": len(keys),
            "training_sha256": _frame_hash(pairs),
            "achievement_sha256": _frame_hash(achievements),
            "weight_sha256": fingerprint(
                pairs,
                ["event_id", "driver_i", "driver_j", "age_at", "recency_weight", "pair_weight"],
            ),
            "max_available_at": _max_time(pairs, achievements, "available_at"),
            "max_evidence_at": _max_time(pairs, achievements, "age_at"),
            "max_evidence_event": _last_event(pairs, achievements),
            "weighted_event_mass": float(pairs.groupby("event_id").recency_weight.first().sum()),
            "weighted_pair_mass": float(event_weights.sum()),
            "pair_effective_sample_size": _ess(event_weights),
            "evidence_by_season": [
                {
                    "season": int(year),
                    "pairs": len(group),
                    "event_weight_mass": float(
                        group.groupby("event_id").recency_weight.first().sum()
                    ),
                }
                for year, group in pairs.groupby("season")
            ],
            "uncertainty": "joint_conditional_positive_IRLS_curvature_plus_AR_forecast_innovation",
            "prior": "proper_lifetime_and_stationary_AR_initial_state_layout_v2_unchanged",
            "future_lifetime_backfill": False,
        },
    }


def freeze_bundle(
    fit,
    target_year,
    calibration_slope=None,
    *,
    selection_provenance=None,
    source_provenance=None,
    variant="rolling",
):
    """Forecast AR state, retaining μ/u covariance and cross-driver base covariance."""
    if pd.Timestamp(fit["cutoff"]).year > target_year:
        raise ValueError("Rating cutoff is after target year")
    if fit["pairs"].season.gt(target_year).any():
        raise ValueError("Rating evidence is after target year")
    cfg = copy.deepcopy(fit["config"])
    keys, lookup, theta = fit["keys"], fit["lookup"], fit["theta"]
    latest = {}
    for kind, driver, year in keys:
        if kind == "annual":
            latest[driver] = max(latest.get(driver, 0), year)
    rho, innovation = cfg["rho"], cfg["innovation_sd"]
    # The research estimator uses ability_sd**2/prior_weight for an unseen driver. Preserve
    # its API while making that default equal to the actual μ + stationary u prior.
    cfg["ability_sd"] = float(
        np.sqrt(cfg["prior_weight"] * (cfg["lifetime_sd"] ** 2 + innovation**2 / (1 - rho**2)))
    )
    drivers = sorted(latest)
    projection = np.zeros((len(keys), len(drivers)))
    for column, driver in enumerate(drivers):
        projection[lookup[("lifetime", driver, 0)], column] = 1
        projection[lookup[("annual", driver, latest[driver])], column] = rho ** (
            target_year - latest[driver]
        )
    lu = splu(fit["hessian"]) if len(keys) else None
    covariance_projection = lu.solve(projection) if len(keys) else projection
    base_covariance = projection.T @ covariance_projection
    base, layout = {}, {}
    for column, driver in enumerate(drivers):
        year = latest[driver]
        lifetime_index = lookup[("lifetime", driver, 0)]
        annual_index = lookup[("annual", driver, year)]
        years = target_year - year
        factor = rho**years
        innovation_var = transition_variance(rho, innovation, years)
        base_covariance[column, column] += innovation_var
        lifetime = float(theta[lifetime_index])
        annual = float(theta[annual_index] * factor)
        ability = lifetime + annual
        unit = np.zeros(len(keys))
        unit[lifetime_index] = 1
        lifetime_covariance = lu.solve(unit)
        unit[lifetime_index], unit[annual_index] = 0, 1
        annual_covariance = lu.solve(unit)
        base[driver] = {
            "ability": ability,
            "lifetime_ability": lifetime,
            "annual_deviation": annual,
            "last_observed_annual_deviation": float(theta[annual_index]),
            "last_observed_year": int(year),
            "forecast_years": int(years),
            "variance": float(base_covariance[column, column]),
            "lifetime_variance": float(lifetime_covariance[lifetime_index]),
            "annual_variance": float(factor**2 * annual_covariance[annual_index] + innovation_var),
            "lifetime_annual_covariance": float(factor * lifetime_covariance[annual_index]),
            "innovation_variance": float(innovation_var),
            "score": float(1 + 99 * ndtr(ability / cfg["display_ability_scale"])),
            "lifetime_score": float(1 + 99 * ndtr(lifetime / cfg["display_ability_scale"])),
            "comparisons": int(
                ((fit["pairs"].driver_i == driver) | (fit["pairs"].driver_j == driver)).sum()
            ),
        }
        for kind, other, circuit in keys:
            if kind != "layout" or other != driver:
                continue
            idx = lookup[(kind, driver, circuit)]
            unit = np.zeros(len(keys))
            unit[idx] = 1
            variance = float(lu.solve(unit)[idx])
            layout.setdefault(driver, {})[circuit] = {
                "effect": float(theta[idx]),
                "variance": variance,
                "base_covariance": float(covariance_projection[idx, column]),
                "score": float(1 + 9 * ndtr(theta[idx] / cfg["display_layout_scale"])),
                "comparisons": int(
                    (
                        ((fit["pairs"].driver_i == driver) | (fit["pairs"].driver_j == driver))
                        & fit["pairs"].circuit_layout_id.eq(circuit)
                    ).sum()
                ),
            }
    bundle = {
        "version": 4,
        "target_year": int(target_year),
        "cutoff": fit["cutoff"],
        "target_event_id": fit["target_event_id"],
        "variant": variant,
        "penalty": fit["penalty"],
        "config": cfg,
        "calibration_slope": float(
            calibration_slope if calibration_slope is not None else 1 / cfg["outcome_tau"]
        ),
        "base": base,
        "layout": layout,
        "diagnostics": fit["diagnostics"],
        "base_joint_covariance": {"drivers": drivers, "matrix": base_covariance.tolist()},
        "rating_input_policy": "teammate_history_only_no_downstream_optimization",
        "forecast_policy": "lifetime_plus_rho_power_gap_times_last_annual_state",
    }
    bundle["selection_provenance"] = copy.deepcopy(selection_provenance or {})
    bundle["source_provenance"] = copy.deepcopy(source_provenance or {})
    bundle["config_sha256"] = canonical_hash(cfg)
    bundle["rating_input_policy"] = "prior_event_evidence_only_independent_of_downstream"
    bundle["forecast_policy"] = "lifetime_plus_annual_AR_zero_gap_no_innovation"
    bundle["achievement"] = achievement_summary(fit["achievements"], fit["cutoff"], cfg)
    for driver, driver_base in base.items():
        weights = (
            fit["pairs"]
            .loc[
                (fit["pairs"].driver_i == driver) | (fit["pairs"].driver_j == driver), "pair_weight"
            ]
            .to_numpy()
        )
        driver_base["weighted_comparisons"] = float(weights.sum())
        driver_base["comparison_effective_sample_size"] = _ess(weights)
    bundle["bundle_sha256"] = canonical_hash(bundle)
    return with_achievement_bonus(bundle, cfg.get("achievement_beta", 0.0))


def compare(bundle, pairs):
    result = compare_reference(bundle, pairs)
    if result.empty:
        return result
    joint = bundle.get("base_joint_covariance", {"drivers": [], "matrix": []})
    lookup = {driver: index for index, driver in enumerate(joint["drivers"])}
    covariance = np.array(joint["matrix"])
    values = []
    for row in result.itertuples():
        a, b = estimate(bundle, row.driver_i, ""), estimate(bundle, row.driver_j, "")
        cov = (
            covariance[lookup[row.driver_i], lookup[row.driver_j]]
            if row.driver_i in lookup and row.driver_j in lookup
            else 0.0
        )
        values.append(np.sqrt(max(a["ability_sd"] ** 2 + b["ability_sd"] ** 2 - 2 * cov, 0)))
    result["base_delta_sd"] = values
    result["base_delta_sd_scope"] = "joint_event_base_only_excludes_layout_and_observation_noise"
    return result


def _cutoff(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Rating cutoff must be timezone-aware")
    return value.tz_convert("UTC")


def recency_weights(age_years, floor=0.5, half_life_years=4.0):
    """Zero-age anchor; never normalize to the newest observed race or old mass."""
    ages = np.asarray(age_years, dtype=float)
    if not 0 <= floor <= 1 or not np.isfinite(half_life_years) or half_life_years <= 0:
        raise ValueError("Memory requires floor in [0,1] and a positive half-life")
    if np.any(~np.isfinite(ages)) or np.any(ages < 0):
        raise ValueError("Evidence age must be finite and nonnegative")
    return floor + (1 - floor) * np.exp2(-ages / half_life_years)


def _age_column(frame):
    # Old warmup rows may have no actual session clock: age_at is explicitly a
    # conservative race-date bound, never silently represented as actual Q time.
    age = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for column in ["age_at", "evidence_at", "prediction_cutoff"]:
        if column in frame:
            age = age.fillna(pd.to_datetime(frame[column], utc=True))
    return age


def _eligible(frame, cutoff, target_event_id):
    selected = frame.loc[frame.event_id.ne(target_event_id)].copy()
    if "minimum_target_season" in selected:
        minimum = selected.minimum_target_season
        selected = selected.loc[minimum.isna() | minimum.le(cutoff.year)].copy()
    if "rating_warmup_eligible" in selected:
        selected = selected.loc[~selected.rating_warmup_eligible.eq(False)].copy()
    selected["available_at"] = pd.to_datetime(selected.available_at, utc=True)
    selected["age_at"] = _age_column(selected)
    return selected.loc[
        selected.available_at.notna()
        & selected.available_at.le(cutoff)
        & selected.age_at.notna()
        & selected.age_at.lt(cutoff)
    ].copy()


def eligible_pairs(observations, cutoff, target_event_id):
    selected = _eligible(observations, _cutoff(cutoff), target_event_id)
    selected = selected.loc[selected.outcome_usable].copy()
    if "session_type" in selected and not selected.session_type.eq("Q").all():
        raise ValueError("Only main qualifying can update driver ratings")
    if selected.duplicated(["event_id", "driver_i", "driver_j"]).any():
        raise ValueError("Duplicate teammate evidence")
    if (
        not selected.outcome.isin([-1, 1]).all()
        or not selected.rank_gap_abs_normalized.between(0, 1).all()
    ):
        raise ValueError("Invalid teammate outcome")
    if (selected.driver_i == selected.driver_j).any():
        raise ValueError("Driver cannot be their own teammate")
    return selected.sort_values(
        ["age_at", "event_id", "constructor_id", "driver_i", "driver_j"]
    ).reset_index(drop=True)


def eligible_achievements(achievements, cutoff, target_event_id):
    columns = [
        "event_id",
        "season",
        "driver_id",
        "position",
        "field_size",
        "age_at",
        "available_at",
    ]
    if achievements is None:
        empty = pd.DataFrame(columns=columns)
        for name in ["age_at", "available_at"]:
            empty[name] = pd.Series(dtype="datetime64[ns, UTC]")
        return empty
    selected = _eligible(achievements, _cutoff(cutoff), target_event_id)
    if "session_type" in selected and not selected.session_type.eq("Q").all():
        raise ValueError("Achievement must use main qualifying, not grid or race")
    if selected.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate achievement evidence")
    groups = []
    for _, group in selected.groupby("event_id", sort=True):
        n = len(group)
        if n < 2 or not group.field_size.eq(n).all() or set(group.position) != set(range(1, n + 1)):
            continue
        if (
            "classification_status" in group
            and group.classification_status.isin(
                ["EX", "DSQ", "DQ", "NC", "RT", "DNS", "DNQ"]
            ).any()
        ):
            continue
        if group.season.nunique() != 1:
            raise ValueError("Conflicting achievement season")
        groups.append(group)
    selected = pd.concat(groups, ignore_index=True) if groups else selected.iloc[:0].copy()
    return selected.sort_values(["age_at", "event_id", "driver_id"]).reset_index(drop=True)


def _frame_hash(frame):
    return fingerprint(frame, sorted(frame.columns))


def _ess(weights):
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum()) if np.any(weights) else 0.0


def _max_time(pairs, achievements, column):
    values = [frame[column].max() for frame in (pairs, achievements) if len(frame)]
    return max(values).isoformat() if values else None


def _last_event(pairs, achievements):
    rows = [frame[["event_id", "age_at"]] for frame in (pairs, achievements) if len(frame)]
    if not rows:
        return None
    return int(pd.concat(rows).sort_values(["age_at", "event_id"]).iloc[-1].event_id)


def achievement_summary(achievements, cutoff, cfg):
    prior = cfg.get("achievement_prior_mass", 20.0)
    if not np.isfinite(prior) or prior <= 0:
        raise ValueError("Achievement prior mass must be positive")
    rows = achievements.copy()
    ages = (_cutoff(cutoff) - rows.age_at).dt.total_seconds().to_numpy() / (365.25 * 86400)
    rows["weight"] = recency_weights(ages, cfg.get("floor", 1.0), cfg.get("half_life_years", 4.0))
    rows["q"] = (rows.field_size - rows.position) / (rows.field_size - 1)
    result = {}
    for driver, group in rows.groupby("driver_id", sort=True):
        weights = group.weight.to_numpy(dtype=float)
        q = float((0.5 * prior + np.dot(weights, group.q)) / (prior + weights.sum()))
        result[driver] = {
            "q_shrunk": q,
            "bonus_unit": max(0.0, 2 * (q - 0.5)),
            "events": len(group),
            "weight_mass": float(weights.sum()),
            "effective_sample_size": _ess(weights),
        }
    return result


def with_achievement_bonus(bundle, beta, *, calibration_slope=None):
    """Apply a bounded transparent component without changing teammate ability."""
    verify_bundle(bundle)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Achievement coefficient must be nonnegative")
    out = copy.deepcopy(bundle)
    out["achievement_beta"] = float(beta)
    out["composite_calibration_slope"] = float(
        calibration_slope if calibration_slope is not None else 1 / out["config"]["outcome_tau"]
    )
    out["composite_uncertainty"] = "not_estimated_overlap_with_ability_no_independent_variance_sum"
    # Drivers seen only in complete achievement data still have the pure ability
    # prior; achievement must not fabricate same-car evidence or confidence.
    for driver in sorted(set(out["base"]) | set(out["achievement"])):
        base = out["base"].setdefault(
            driver,
            {
                "ability": 0.0,
                "score": 50.5,
                "comparisons": 0,
                "variance": out["config"]["ability_sd"] ** 2 / out["config"]["prior_weight"],
                "weighted_comparisons": 0.0,
                "comparison_effective_sample_size": 0.0,
            },
        )
        achievement = out["achievement"].get(
            driver,
            {
                "q_shrunk": 0.5,
                "bonus_unit": 0,
                "events": 0,
                "weight_mass": 0,
                "effective_sample_size": 0,
            },
        )
        bonus = beta * achievement["bonus_unit"]
        base.update(
            achievement_bonus=float(bonus),
            composite_ability=base["ability"] + bonus,
            achievement=achievement,
        )
        base["composite_score"] = float(
            1 + 99 * ndtr(base["composite_ability"] / out["config"]["display_ability_scale"])
        )
    out["bundle_sha256"] = canonical_hash({k: v for k, v in out.items() if k != "bundle_sha256"})
    return out


def predictor_bundle(bundle, *, composite=False):
    """Explicit read-only feature projection; canonical bundle retains both parts."""
    verify_bundle(bundle)
    if not composite:
        return copy.deepcopy(bundle)
    out = copy.deepcopy(bundle)
    out["source_bundle_sha256"] = bundle["bundle_sha256"]
    out["input_score_component"] = "composite_including_car_environment"
    for base in out["base"].values():
        base["teammate_ability"] = base["ability"]
        base["ability"] = base["composite_ability"]
        base["score"] = base["composite_score"]
    out["calibration_slope"] = out["composite_calibration_slope"]
    out["bundle_sha256"] = canonical_hash({k: v for k, v in out.items() if k != "bundle_sha256"})
    return out
