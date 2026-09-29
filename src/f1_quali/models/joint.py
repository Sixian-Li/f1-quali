"""m1r1 predictor: quota-constrained Top3 Brier and anchored driver/layout rules.

The teammate network states remain evidence anchors. Six shared rule parameters
(five active in m1r1; the achievement-beta delta stays fixed at zero) turn each
driver's earlier, available main-Q innovations into bounded driver and
driver-by-layout corrections. There are no free per-driver parameters. The Top3
quota Brier loss differentiates through the corrections and the Top3 head
together; Q2, Q3 and pole are then refitted on the corrected inputs. The
original network solver and the final four-pool projection are not unrolled.
"""

import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp, ndtr

from f1_quali.models.coordination import decode_nested, project_probabilities
from f1_quali.models.pools import TASKS, _complete, _design, _logistic, _trace, _training_rows
from f1_quali.ratings.core import canonical_hash

DRIVER_COLUMNS = ("joint_pace_fast", "joint_pace_slow", "joint_result")
LAYOUT_COLUMNS = ("joint_layout_pace", "joint_layout_result")
PARAMETER_NAMES = (
    "driver_pace_fast",
    "driver_pace_slow",
    "driver_result",
    "achievement_beta_delta",
    "layout_pace",
    "layout_result",
)


# ----------------------------------------------------------------------------
# Quota-constrained probabilities


def group_counts(groups, quotas):
    groups = np.asarray(groups)
    quotas = np.asarray(quotas, dtype=float)
    if groups.ndim != 1 or not len(groups) or groups.dtype.kind not in "iu":
        raise ValueError("Nonempty integer event groups required")
    if quotas.ndim != 1 or not np.isfinite(quotas).all():
        raise ValueError("Finite event quotas required")
    if not np.array_equal(np.unique(groups), np.arange(len(quotas))):
        raise ValueError("Event groups must be contiguous and match quotas")
    counts = np.bincount(groups, minlength=len(quotas))
    if np.any(quotas <= 0) or np.any(quotas >= counts):
        raise ValueError("Pool quota must lie strictly inside the field size")
    return counts


def quota_probabilities(logits, groups, quotas):
    """Sigmoid probabilities with an implicit event intercept enforcing sum = quota."""
    logits = np.asarray(logits, dtype=float)
    groups = np.asarray(groups)
    quotas = np.asarray(quotas, dtype=float)
    counts = group_counts(groups, quotas)
    if logits.shape != groups.shape or not np.isfinite(logits).all():
        raise ValueError("Finite logits aligned with event groups required")
    centered = logits - (np.bincount(groups, weights=logits) / counts)[groups]
    low_z = np.full(len(quotas), np.inf)
    high_z = np.full(len(quotas), -np.inf)
    np.minimum.at(low_z, groups, centered)
    np.maximum.at(high_z, groups, centered)
    anchor = logit(quotas / counts)
    low, high = anchor - high_z - 1, anchor - low_z + 1
    for _ in range(64):
        shift = (low + high) / 2
        p = expit(centered + shift[groups])
        mass = np.bincount(groups, weights=p)
        low = np.where(mass < quotas, shift, low)
        high = np.where(mass >= quotas, shift, high)
    eta = centered + ((low + high) / 2)[groups]
    p = expit(eta)
    if np.max(np.abs(np.bincount(groups, weights=p) - quotas)) > 1e-9:
        raise RuntimeError("Pool capacity solver did not converge")
    return p, eta


def brier_objective(beta, x, y, groups, quotas, *, alpha):
    """Event-weighted quota Brier, scaled by 2q(1-q) to keep the ridge comparable."""
    counts = np.bincount(groups, minlength=len(quotas))
    q = (quotas / counts)[groups]
    weights = 1 / counts[groups]
    p, eta = quota_probabilities(x @ beta, groups, quotas)
    value = np.dot(weights, (p - y) ** 2 / (2 * q * (1 - q)))
    derivative = weights * (p - y) / (q * (1 - q))
    sensitivity = expit(eta) * expit(-eta)
    denominator = np.bincount(groups, weights=sensitivity)
    numerator = np.bincount(groups, weights=sensitivity * derivative)
    average = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    residual = sensitivity * (derivative - average[groups])
    value += 0.5 * alpha * np.dot(beta, beta)
    return float(value), x.T @ residual + alpha * beta


def fit_brier_head(x, y, groups, quotas, *, alpha, seed):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    groups, quotas = np.asarray(groups), np.asarray(quotas, dtype=float)
    counts = group_counts(groups, quotas)
    if x.ndim != 2 or len(x) != len(y) or y.shape != groups.shape:
        raise ValueError("Aligned design, outcomes and event groups required")
    if not np.isfinite(x).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Finite inputs and binary outcomes required")
    if not np.allclose(np.bincount(groups, weights=y), quotas, atol=1e-12, rtol=0):
        raise ValueError("Complete event outcomes must match pool quotas")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Positive regularization required")
    initial = np.asarray(seed, dtype=float)
    if initial.shape != (x.shape[1],):
        raise ValueError("Initial coefficient dimension mismatch")
    runs, fitted = [], []
    for name, start in [
        ("neutral", np.zeros_like(initial)),
        ("old_ce", initial),
        ("half_old_ce", initial / 2),
    ]:
        result = minimize(
            lambda theta: brier_objective(theta, x, y, groups, quotas, alpha=alpha),
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-7},
        )
        maximum_gradient = float(np.max(np.abs(result.jac)))
        accepted = bool(np.isfinite(result.fun) and maximum_gradient <= 1e-4)
        runs.append(
            {
                "start": name,
                "objective": float(result.fun),
                "max_gradient": maximum_gradient,
                "accepted": accepted,
                "converged": bool(result.success),
                "iterations": int(result.nit),
            }
        )
        if accepted:
            fitted.append((float(result.fun), len(runs) - 1, result.x))
    if not fitted:
        raise RuntimeError(f"No stationary probability fit: {runs}")
    value, best, theta = min(fitted, key=lambda item: (item[0], item[1]))
    return {
        "kind": "experimental_logistic",
        "loss": "brier",
        "quota_training": True,
        "quota_inference": True,
        "alpha": float(alpha),
        "beta": theta.tolist(),
        "intercept": 0.0,
        "objective": value,
        "chosen_start": runs[best]["start"],
        "starts": runs,
        "training_events": len(counts),
        "training_rows": len(y),
        "brier_scale": "1 / (2 * event_prevalence * (1-event_prevalence))",
    }


def fit_top3(base, features, labels):
    """Replace only the Top3 head with the quota Brier fit, keeping its trace."""
    out = copy.deepcopy(base)
    original = base["tasks"]["top3"]
    if original["kind"] == "quota_prior":
        return out
    year, columns = base["trace"]["year"], base["columns"]
    rows = _training_rows(features, labels, year, columns)
    rows = rows[rows.stage.eq("Q1")].copy()
    part = rows[_complete(rows, "complete_top3")].copy()
    if _trace(part, year, columns) != original["trace"]:
        raise RuntimeError("Top3 training rows differ from the anchor head")
    x = part[[f"_x_{c}" for c in columns]].to_numpy(float) / np.asarray(base["scale"])
    ids, groups = np.unique(part.event_id.to_numpy(), return_inverse=True)
    fitted = fit_brier_head(
        x,
        part.y_top3.to_numpy(float),
        groups,
        np.full(len(ids), 3.0),
        alpha=original["alpha"],
        seed=np.asarray(original["beta"]),
    )
    out["tasks"]["top3"] = {**fitted, "trace": original["trace"]}
    out["experiment"] = "top3_brier_quota"
    return out


# ----------------------------------------------------------------------------
# Causal rating histories and shared correction rules


def fixed_achievement_features(features, beta):
    """One absolute achievement beta for every snapshot; pure states unchanged."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("Fixed achievement beta must be finite and in [0, 1]")
    out = features.copy()
    h = out.joint_achievement_unit.to_numpy(float)
    if not np.isfinite(h).all() or np.any((h < 0) | (h > 1)):
        raise ValueError("Achievement units must be finite and in [0, 1]")
    np.testing.assert_allclose(
        out.ability, out.teammate_ability + out.achievement_bonus, rtol=0, atol=1e-12
    )
    out["fixed_achievement_original_beta"] = out.joint_base_beta
    out["achievement_bonus"] = float(beta) * h
    out["ability"] = out.teammate_ability + out.achievement_bonus
    out["ability_score"] = 1 + 99 * ndtr(out.ability / 0.4)
    out["joint_base_beta"] = float(beta)
    out["fixed_achievement_beta"] = float(beta)
    out["fixed_beta_source_bundle_sha256"] = out.rating_bundle_sha256
    out["rating_bundle_sha256"] = [
        canonical_hash({"original": h, "absolute_beta": beta})
        for h in out.fixed_beta_source_bundle_sha256
    ]
    return out


def make_innovations(features, pairs, cfg):
    """Signed post-event teammate surprises relative to each pre-event anchor."""
    lookup = features.set_index(["event_id", "driver_id"])
    rows, accepted = [], []
    for row in pairs[pairs.outcome_usable & pairs.available_at.notna()].itertuples():
        if (row.event_id, row.driver_i) not in lookup.index or (
            row.event_id,
            row.driver_j,
        ) not in lookup.index:
            continue
        a, b = lookup.loc[(row.event_id, row.driver_i)], lookup.loc[(row.event_id, row.driver_j)]
        if not a.constructor_id == b.constructor_id == row.constructor_id:
            raise ValueError("Teammate innovation constructor mismatch")
        if not pd.Timestamp(row.available_at) > a.prediction_cutoff:
            raise ValueError("Teammate evidence must postdate the anchor snapshot")
        delta = float(a.teammate_ability + a.track_effect - b.teammate_ability - b.track_effect)
        gap = float(row.pace_gap)
        pace = np.clip(gap - delta, -2 * cfg["pace_sigma"], 2 * cfg["pace_sigma"]) / 2
        result = (
            cfg["outcome_tau"]
            / 2
            * (row.outcome - np.tanh(delta / (2 * cfg["outcome_tau"])))
            * (1 + row.rank_gap_abs_normalized)
        )
        accepted.append(row._asdict())
        for driver, sign in [(row.driver_i, 1), (row.driver_j, -1)]:
            rows.append(
                {
                    "event_id": int(row.event_id),
                    "season": int(row.season),
                    "driver_id": driver,
                    "constructor_id": row.constructor_id,
                    "circuit_layout_id": row.circuit_layout_id,
                    "session_type": "Q",
                    "age_at": row.age_at,
                    "available_at": row.available_at,
                    "pace_innovation": float(sign * pace),
                    "result_innovation": float(sign * result),
                    "anchor_pair_delta": delta,
                    "last_common_stage": row.last_common_stage,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(accepted)


def history_summaries(targets, innovations, cfg):
    """Summaries of earlier, available main-Q innovations for each target driver.

    Layout contrasts subtract the driver's overall innovation mean, so the local
    correction describes only extra performance on that exact layout.
    """
    if innovations.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate driver innovation")
    if not innovations.session_type.eq("Q").all():
        raise ValueError("Only main qualifying may update rating histories")
    records = []
    histories = {
        driver: group.sort_values(["age_at", "event_id"])
        for driver, group in innovations.groupby("driver_id", sort=True)
    }
    for row in targets.itertuples():
        cutoff = pd.Timestamp(row.prediction_cutoff)
        if cutoff.tzinfo is None or pd.isna(cutoff):
            raise ValueError("Explicit timezone-aware cutoff required")
        history = histories.get(row.driver_id, innovations.iloc[:0])
        history = history[
            history.event_id.ne(row.event_id)
            & history.available_at.lt(cutoff)
            & history.age_at.lt(cutoff)
            & history.season.le(row.season)
        ]
        ages = (cutoff - history.age_at).dt.total_seconds().to_numpy() / (365.25 * 86400)
        same = history.circuit_layout_id.eq(row.circuit_layout_id).to_numpy()

        def summary(column, half_life, local=False, history=history, ages=ages, same=same):
            values = history[column].to_numpy(float)
            usable = np.isfinite(values)
            weights = cfg["memory_floor"] + (1 - cfg["memory_floor"]) * np.exp2(-ages / half_life)
            weights = weights * usable
            values = np.nan_to_num(values)
            mean = np.dot(weights, values) / (weights.sum() + cfg["driver_prior_mass"])
            if not local:
                return float(mean)
            local_weights = weights * same
            return float(
                np.dot(local_weights, values - mean)
                / (local_weights.sum() + cfg["layout_prior_mass"])
            )

        records.append(
            {
                "joint_pace_fast": summary("pace_innovation", cfg["fast_half_life"]),
                "joint_pace_slow": summary("pace_innovation", cfg["slow_half_life"]),
                "joint_result": summary("result_innovation", cfg["shared_half_life"]),
                "joint_layout_pace": summary("pace_innovation", cfg["shared_half_life"], True),
                "joint_layout_result": summary("result_innovation", cfg["shared_half_life"], True),
                "joint_history_events": len(history),
                "joint_layout_events": int(same.sum()),
                "joint_layout_pace_events": int((same & history.pace_innovation.notna()).sum()),
                "joint_history_max_available_at": history.available_at.max(),
                "joint_history_max_age_at": history.age_at.max(),
            }
        )
    out = targets.reset_index(drop=True).copy()
    for column, series in pd.DataFrame(records).items():
        out[column] = series
    return out


def attach_history(features, bundles, innovations, cfg):
    out = history_summaries(features, innovations, cfg)
    out["joint_achievement_unit"] = [
        bundles[int(row.event_id)]["achievement"].get(row.driver_id, {}).get("bonus_unit", 0.0)
        for row in out.itertuples()
    ]
    out["joint_base_beta"] = [bundles[int(e)]["achievement_beta"] for e in out.event_id]
    np.testing.assert_allclose(
        out.achievement_bonus, out.joint_base_beta * out.joint_achievement_unit, atol=1e-14, rtol=0
    )
    return out


def corrections(parameters, driver_history, layout_history, achievement_unit, cfg):
    """Internal-unit state changes and exact derivatives of the six shared rules."""
    p = np.asarray(parameters, dtype=float)
    za, zl, h = map(np.asarray, (driver_history, layout_history, achievement_unit))
    if p.shape != (6,) or za.shape != (len(h), 3) or zl.shape != (len(h), 2):
        raise ValueError("Invalid shared rating-rule dimensions")
    if not all(np.isfinite(v).all() for v in (p, za, zl, h)) or np.any((h < 0) | (h > 1)):
        raise ValueError("Finite history and bounded achievement units required")
    ta = np.tanh(za @ p[:3] / cfg["driver_cap"])
    tl = np.tanh(zl @ p[4:] / cfg["layout_cap"])
    da, dl, db = cfg["driver_cap"] * ta, cfg["layout_cap"] * tl, h * p[3]
    ja, jl, jb = (np.zeros((len(h), 6)) for _ in range(3))
    ja[:, :3] = (1 - ta**2)[:, None] * za
    jl[:, 4:] = (1 - tl**2)[:, None] * zl
    jb[:, 3] = h
    return da, dl, db, ja, jl, jb


def apply_rule(features, parameters, cfg, rule_id):
    """Corrected general and layout ratings; the anchor values are retained."""
    out = features.copy()
    da, dl, db, *_ = corrections(
        parameters,
        out[list(DRIVER_COLUMNS)].to_numpy(),
        out[list(LAYOUT_COLUMNS)].to_numpy(),
        out.joint_achievement_unit.to_numpy(),
        cfg,
    )
    for column in ["ability", "teammate_ability", "track_effect", "ability_score", "layout_score"]:
        out[f"anchor_{column}"] = out[column]
    out["joint_driver_delta"], out["joint_layout_delta"], out["joint_achievement_delta"] = (
        da,
        dl,
        db,
    )
    out["teammate_ability"] += da
    out["achievement_bonus"] += db
    out["ability"] += da + db
    out["track_effect"] += dl
    out["ability_score"] = 1 + 99 * ndtr(out.ability / 0.4)
    out["joint_pure_score"] = 1 + 99 * ndtr(out.teammate_ability / 0.4)
    out["layout_score"] = 1 + 9 * ndtr(out.track_effect / 0.1)
    out["joint_effective_beta"] = out.joint_base_beta + parameters[3]
    out["anchor_rating_bundle_sha256"] = out.rating_bundle_sha256
    out["rating_bundle_sha256"] = [
        canonical_hash({"anchor": h, "rule": rule_id}) for h in out.anchor_rating_bundle_sha256
    ]
    out["rating_variant"] = "anchored_joint_driver_layout_rule"
    out["rating_score_scope"] = "prediction_adapted_including_car_environment"
    out["rating_uncertainty_scope"] = (
        "anchor_uncertainty_only_joint_correction_uncertainty_not_estimated"
    )
    return out


def changed_inputs(features, model, cfg):
    rule = model["joint_rule"]
    return apply_rule(features, rule["parameters"], cfg, rule.get("rule_id", "prior_fallback"))


def _centered(values, groups, counts):
    sums = np.zeros((len(counts), *values.shape[1:]))
    np.add.at(sums, groups, values)
    denominator = counts.reshape((-1,) + (1,) * (values.ndim - 1))
    return values - (sums / denominator)[groups]


def joint_objective(theta, data, cfg, *, active, alpha):
    """Top3 quota Brier + teammate anchor + bounded drift, jointly differentiated."""
    m = data["x"].shape[1]
    beta = theta[:m]
    parameters = np.zeros(6)
    parameters[np.asarray(active, dtype=int)] = theta[m:]
    da, dl, db, ja, jl, jb = corrections(parameters, data["za"], data["zl"], data["h"], cfg)
    groups, counts = data["groups"], data["counts"]
    weights = 1 / counts[groups]
    a_col, l_col = data["ability_column"], data["layout_column"]
    a_scale, l_scale = data["ability_scale"], data["layout_scale"]
    x = data["x"].copy()
    x[:, a_col] += _centered(da + db, groups, counts) / a_scale
    x[:, l_col] += _centered(dl, groups, counts) / l_scale
    p, eta = quota_probabilities(x @ beta, groups, np.full(len(counts), 3.0))
    q = 3 / counts[groups]
    error = p - data["y"]
    loss_prediction = np.dot(weights, error**2 / (2 * q * (1 - q)))
    sensitivity = expit(eta) * expit(-eta)
    dp = weights * error / (q * (1 - q))
    denom = np.bincount(groups, weights=sensitivity)
    mean = np.divide(
        np.bincount(groups, weights=sensitivity * dp),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    dz = sensitivity * (dp - mean[groups])
    gradient_beta = x.T @ dz + alpha * beta
    gradient_rules_prediction = (
        _centered(ja + jb, groups, counts).T @ dz * beta[a_col] / a_scale
        + _centered(jl, groups, counts).T @ dz * beta[l_col] / l_scale
    )
    gradient_rules = gradient_rules_prediction.copy()
    loss_anchor = 0.0
    for delta, jacobian, scale in [
        (da, ja, cfg["driver_anchor_scale"]),
        (db, jb, cfg["driver_anchor_scale"]),
        (dl, jl, cfg["layout_anchor_scale"]),
    ]:
        loss_anchor += 0.5 * cfg["anchor_weight"] * np.dot(weights, (delta / scale) ** 2)
        gradient_rules += cfg["anchor_weight"] * jacobian.T @ (weights * delta / scale**2)
    # Same-car teammate evidence is scored at each original pre-Q snapshot; the
    # global achievement/car contribution is excluded from the comparison.
    pair = data["pairs"]
    i, j = pair["i"], pair["j"]
    delta = pair["base_delta"] + da[i] + dl[i] - da[j] - dl[j]
    jd = ja[i] + jl[i] - ja[j] - jl[j]
    valid = np.isfinite(pair["pace"])
    residual = (delta[valid] - pair["pace"][valid]) / cfg["pace_sigma"]
    nu = cfg["student_df"]
    pw = pair["weights"] * cfg["teammate_weight"]
    loss_pair = np.dot(pw[valid], (nu + 1) / 2 * np.log1p(residual**2 / nu))
    g_pair = np.zeros(len(i))
    g_pair[valid] = pw[valid] * (nu + 1) * residual / (nu + residual**2) / cfg["pace_sigma"]
    z = pair["outcome"] * delta / cfg["outcome_tau"]
    rw = pw * cfg["outcome_penalty"] * (1 + pair["rank_gap"])
    loss_pair += np.dot(rw, np.logaddexp(0, -z))
    g_pair -= rw * pair["outcome"] * expit(-z) / cfg["outcome_tau"]
    gradient_rules += jd.T @ g_pair
    loss_rules = 0.5 * cfg["rule_ridge"] * np.dot(parameters, parameters)
    gradient_rules += cfg["rule_ridge"] * parameters
    loss = loss_prediction + 0.5 * alpha * np.dot(beta, beta) + loss_anchor + loss_pair + loss_rules
    grad = np.r_[gradient_beta, gradient_rules[np.asarray(active, dtype=int)]]
    components = {
        "prediction": float(loss_prediction),
        "anchor": float(loss_anchor),
        "teammate": float(loss_pair),
        "rule_ridge": float(loss_rules),
        "prediction_rule_gradient": gradient_rules_prediction.tolist(),
    }
    return float(loss), grad, components


def fit_joint(data, seed_beta, cfg, *, active, alpha):
    m = data["x"].shape[1]
    active = list(active)
    limits = [(-2.0, 2.0)] * 6
    limits[3] = (-0.1, 0.6)
    bounds = [(None, None)] * m + [limits[i] for i in active]
    fits, diagnostics = [], []
    for name, multiplier in [("selected_head", 1.0), ("half_head", 0.5), ("neutral", 0.0)]:
        initial = np.r_[np.asarray(seed_beta) * multiplier, np.zeros(len(active))]
        result = minimize(
            lambda t: joint_objective(t, data, cfg, active=active, alpha=alpha)[:2],
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1600, "ftol": 1e-12, "gtol": 1e-7},
        )
        projected = result.jac.copy()
        for k, (low, high) in enumerate(bounds):
            if low is not None and result.x[k] <= low + 1e-10 and projected[k] > 0:
                projected[k] = 0
            if high is not None and result.x[k] >= high - 1e-10 and projected[k] < 0:
                projected[k] = 0
        maximum = float(np.max(np.abs(projected)))
        accepted = bool(np.isfinite(result.fun) and maximum <= 1e-4)
        diagnostics.append(
            {
                "start": name,
                "objective": float(result.fun),
                "accepted": accepted,
                "converged": bool(result.success),
                "max_projected_gradient": maximum,
                "iterations": int(result.nit),
            }
        )
        if accepted:
            fits.append((float(result.fun), len(diagnostics) - 1, result.x))
    if not fits:
        raise RuntimeError(f"Joint optimization has no stationary fit: {diagnostics}")
    objective, index, theta = min(fits, key=lambda value: (value[0], value[1]))
    parameters = np.zeros(6)
    parameters[active] = theta[m:]
    return {
        "beta": theta[:m].tolist(),
        "parameters": parameters.tolist(),
        "active": active,
        "objective": objective,
        "starts": diagnostics,
        "chosen_start": diagnostics[index]["start"],
        "components": joint_objective(theta, data, cfg, active=active, alpha=alpha)[2],
        "initial_prediction_rule_gradient": joint_objective(
            np.r_[seed_beta, np.zeros(len(active))], data, cfg, active=active, alpha=alpha
        )[2]["prediction_rule_gradient"],
    }


# ----------------------------------------------------------------------------
# Annual training


def training_data(base, features, labels, pairs):
    year, columns = base["trace"]["year"], base["columns"]
    rows = _training_rows(features, labels, year, columns)
    rows = rows[rows.stage.eq("Q1")].copy()
    rows = rows[_complete(rows, "complete_top3")].reset_index(drop=True)
    if not rows.y_top3.isin([0, 1]).all():
        raise ValueError("Complete binary Top3 labels required")
    _, groups = np.unique(rows.event_id.to_numpy(), return_inverse=True)
    counts = np.bincount(groups)
    if not np.allclose(np.bincount(groups, weights=rows.y_top3.to_numpy(float)), 3):
        raise ValueError("Each complete Top3 event needs exactly three positives")
    lookup = {(int(r.event_id), r.driver_id): i for i, r in enumerate(rows.itertuples())}
    auxiliary = []
    cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
    for p in pairs[pairs.available_at.lt(cutoff) & pairs.season.lt(year)].itertuples():
        i, j = lookup.get((int(p.event_id), p.driver_i)), lookup.get((int(p.event_id), p.driver_j))
        if i is not None and j is not None:
            delta = (
                rows.iloc[i].teammate_ability
                + rows.iloc[i].track_effect
                - rows.iloc[j].teammate_ability
                - rows.iloc[j].track_effect
            )
            auxiliary.append(
                {
                    "event_id": int(p.event_id),
                    "i": i,
                    "j": j,
                    "base_delta": delta,
                    "pace": float(p.pace_gap),
                    "outcome": float(p.outcome),
                    "rank_gap": p.rank_gap_abs_normalized,
                }
            )
    aux = pd.DataFrame(
        auxiliary, columns=["event_id", "i", "j", "base_delta", "pace", "outcome", "rank_gap"]
    )
    aux["weights"] = 1 / aux.groupby("event_id").event_id.transform("size")
    a_col, l_col = columns.index("ability"), columns.index("track_effect")
    data = {
        "x": rows[[f"_x_{c}" for c in columns]].to_numpy(float) / np.asarray(base["scale"]),
        "y": rows.y_top3.to_numpy(float),
        "groups": groups,
        "counts": counts,
        "za": rows[list(DRIVER_COLUMNS)].to_numpy(float),
        "zl": rows[list(LAYOUT_COLUMNS)].to_numpy(float),
        "h": rows.joint_achievement_unit.to_numpy(float),
        "ability_column": a_col,
        "layout_column": l_col,
        "ability_scale": base["scale"][a_col],
        "layout_scale": base["scale"][l_col],
        "pairs": {
            c: aux[c].to_numpy(dtype=int if c in ("i", "j") else float)
            for c in ["i", "j", "base_delta", "pace", "outcome", "rank_gap", "weights"]
        },
    }
    return rows, data


def refit_other_heads(base, features, labels):
    """Refit Q2/Q3 logistic and pole softmax heads on corrected inputs."""
    out = copy.deepcopy(base)
    year, columns = base["trace"]["year"], base["columns"]
    rows = _training_rows(features, labels, year, columns)
    rows = rows[rows.stage.eq("Q1")].copy()
    complete = rows[[f"complete_{task}" for task in TASKS]].eq(True).all(axis=1)
    complete &= _complete(rows, "complete_all_tasks") if "complete_all_tasks" in rows else True
    out["anchor_model_trace"] = copy.deepcopy(base["trace"])
    out["trace"] = _trace(rows[complete], year, columns)
    for task in ("q2", "q3", "pole"):
        part = rows[_complete(rows, f"complete_{task}")].copy()
        original = base["tasks"][task]
        if original["kind"] == "quota_prior":
            continue
        x = part[[f"_x_{c}" for c in columns]].to_numpy(float) / np.asarray(base["scale"])
        y = part[f"y_{task}"].to_numpy(float)
        counts = part.groupby("event_id").driver_id.transform("size").to_numpy()
        alpha = original["alpha"]
        if task != "pole":
            q = part[f"{task}_quota"].to_numpy(float) / counts
            fitted = _logistic(x, y, 1 / counts, alpha, np.log(q / (1 - q)))
            fitted["kind"] = "logistic"
        else:
            groups = list(part.groupby("event_id", sort=False).indices.values())

            def objective(beta, x=x, y=y, alpha=alpha, groups=groups):
                eta, residual = x @ beta, np.zeros(len(y))
                loss = 0.5 * alpha * np.dot(beta, beta)
                for g in groups:
                    normalizer = logsumexp(eta[g])
                    loss += normalizer - np.dot(y[g], eta[g])
                    residual[g] = np.exp(eta[g] - normalizer) - y[g]
                return loss, x.T @ residual + alpha * beta

            result = minimize(
                objective,
                np.zeros(len(columns)),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
            )
            if not result.success and np.max(np.abs(result.jac)) > 1e-4:
                raise RuntimeError(f"Pole softmax failed: {result.message}")
            fitted = {
                "kind": "softmax",
                "beta": result.x.tolist(),
                "objective": float(result.fun),
                "converged": bool(result.success),
            }
        out["tasks"][task] = {**fitted, "alpha": alpha, "trace": _trace(part, year, columns)}
    return out


def fit_candidate(base, features, labels, pairs, active, cfg):
    out = copy.deepcopy(base)
    if base["tasks"]["top3"]["kind"] == "quota_prior":
        out["joint_rule"] = {"parameters": [0.0] * 6, "active": active, "fallback": True}
        return out
    rows, data = training_data(base, features, labels, pairs)
    if _trace(rows, base["trace"]["year"], base["columns"]) != base["tasks"]["top3"]["trace"]:
        raise RuntimeError("Joint training rows differ from the Top3 anchor head")
    fitted = fit_joint(
        data,
        base["tasks"]["top3"]["beta"],
        cfg,
        active=active,
        alpha=base["tasks"]["top3"]["alpha"],
    )
    rule_id = canonical_hash(
        {"parameters": fitted["parameters"], "settings": cfg, "year": base["trace"]["year"]}
    )
    permitted = features.season.lt(base["trace"]["year"]) & features.season_event_ordinal.ge(6)
    changed = apply_rule(features[permitted].copy(), fitted["parameters"], cfg, rule_id)
    out = refit_other_heads(base, changed, labels)
    top_rows = _training_rows(changed, labels, base["trace"]["year"], base["columns"])
    top_rows = top_rows[top_rows.stage.eq("Q1")]
    top_rows = top_rows[_complete(top_rows, "complete_top3")]
    # The authoritative joint-fit diagnostics live in joint_rule.
    out["tasks"]["top3"] = {
        "beta": fitted["beta"],
        "intercept": 0.0,
        "kind": "experimental_logistic",
        "loss": "brier",
        "quota_inference": True,
        "quota_training": True,
        "alpha": base["tasks"]["top3"]["alpha"],
        "optimization_diagnostics": "joint_rule",
        "training_events": len(data["counts"]),
        "training_rows": len(rows),
        "brier_scale": "1 / (2 * event_prevalence * (1-event_prevalence))",
        "trace": _trace(top_rows, base["trace"]["year"], base["columns"]),
    }
    out["joint_rule"] = {
        **fitted,
        "rule_id": rule_id,
        "settings": cfg,
        "history_columns": [*DRIVER_COLUMNS, *LAYOUT_COLUMNS],
        "training_events": len(data["counts"]),
        "training_rows": len(rows),
        "auxiliary_pairs": len(data["pairs"]["i"]),
        "anchor_training_trace": base["tasks"]["top3"]["trace"],
    }
    out["experiment"] = "joint_driver_layout_rules"
    return out


def fit_fixed(base, features, labels, pairs, active, cfg, beta):
    """Joint fit with the absolute achievement beta excluded from optimization."""
    if 3 in active or not features.joint_base_beta.eq(beta).all():
        raise ValueError("The achievement beta must be fixed in every snapshot")
    source = copy.deepcopy(base)
    rows = _training_rows(features, labels, base["trace"]["year"], base["columns"])
    rows = rows[rows.stage.eq("Q1")]
    part = rows[_complete(rows, "complete_top3")]
    source["tasks"]["top3"]["trace"] = _trace(part, base["trace"]["year"], base["columns"])
    model = fit_candidate(source, features, labels, pairs, active, cfg)
    model["fixed_achievement"] = {
        "absolute_beta": beta,
        "trainable": False,
        "excluded_parameter_index": 3,
        "selected_model_trace": base["trace"],
        "selected_top3_trace": base["tasks"]["top3"]["trace"],
        "input_policy": "same_constant_in_all_training_and_prediction_snapshots",
    }
    if model["joint_rule"]["parameters"][3] != 0.0:
        raise RuntimeError("Fixed achievement beta moved during optimization")
    return model


def predict_heads(model, features, q2, q3):
    """Four raw probabilities, coordinated nested pools and a full-field order."""
    if features.event_id.nunique() != 1:
        raise ValueError("Pool prediction requires one full event")
    features = features.sort_values("driver_id").reset_index(drop=True)
    x = _design(model, features)
    quotas = [q2, q3, 3, 1]
    raw = np.empty((len(features), 4))
    for k, task in enumerate(TASKS):
        head = model["tasks"][task]
        if head["kind"] == "quota_prior":
            raw[:, k] = quotas[k] / len(features)
        elif head["kind"] == "experimental_logistic":
            linear = x @ np.asarray(head["beta"])
            raw[:, k] = quota_probabilities(linear, np.zeros(len(linear), dtype=int), [quotas[k]])[
                0
            ]
        elif task == "pole":
            eta = x @ np.asarray(head["beta"])
            raw[:, k] = np.exp(eta - logsumexp(eta))
        else:
            q = quotas[k] / len(features)
            raw[:, k] = expit(
                np.log(q / (1 - q)) + head["intercept"] + x @ np.asarray(head["beta"])
            )
    coordinated = project_probabilities(raw, quotas)
    out = features.copy()
    for k, task in enumerate(TASKS):
        out[f"raw_p_{task}"] = raw[:, k]
        out[f"p_{task}"] = coordinated[:, k]
    return decode_nested(out, q2, q3)
