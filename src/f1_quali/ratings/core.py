"""Robust teammate objective and immutable rating estimates."""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from f1_quali.integrity import sha256


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def objective(theta, design, pace_rows, gaps, pace_weights, outcomes, result_weights, prior, cfg):
    """Return loss, analytic gradient and positive IRLS/Newton curvature."""
    delta = np.asarray(design @ theta).ravel()
    residual = (delta[pace_rows] - gaps) / cfg["pace_sigma"]
    nu = cfg["student_df"]
    loss = float(np.sum(pace_weights * (nu + 1) / 2 * np.log1p(residual**2 / nu)))
    gradient_delta = np.zeros(len(delta))
    curvature = np.zeros(len(delta))
    np.add.at(
        gradient_delta,
        pace_rows,
        pace_weights * (nu + 1) * residual / (nu + residual**2) / cfg["pace_sigma"],
    )
    np.add.at(
        curvature, pace_rows, pace_weights * (nu + 1) / (nu + residual**2) / cfg["pace_sigma"] ** 2
    )
    z = outcomes * delta / cfg["outcome_tau"]
    loss += float(np.sum(result_weights * np.logaddexp(0, -z)))
    gradient_delta -= result_weights * outcomes * expit(-z) / cfg["outcome_tau"]
    curvature += result_weights * expit(z) * expit(-z) / cfg["outcome_tau"] ** 2
    penalty = np.asarray(prior @ theta).ravel()
    loss += float(theta @ penalty / 2)
    gradient = np.asarray(design.T @ gradient_delta).ravel() + penalty
    hessian = (design.T @ sparse.diags(curvature) @ design + prior).tocsc()
    return loss, gradient, hessian


def verify_bundle(bundle):
    if (
        canonical_hash({k: v for k, v in bundle.items() if k != "bundle_sha256"})
        != bundle["bundle_sha256"]
    ):
        raise ValueError("Frozen rating bundle was modified")


def estimate(bundle, driver, layout):
    cfg = bundle["config"]
    base = bundle["base"].get(driver, {})
    track = bundle["layout"].get(driver, {}).get(layout, {})
    ability = base.get("ability", 0.0)
    effect = track.get("effect", 0.0)
    base_var = base.get("variance", cfg["ability_sd"] ** 2 / cfg["prior_weight"])
    track_var = track.get("variance", cfg["layout_sd"] ** 2 / cfg["prior_weight"])
    return {
        "ability": ability,
        "track_effect": effect,
        "ability_sd": np.sqrt(base_var),
        "track_sd": np.sqrt(track_var),
        "combined_sd": np.sqrt(max(base_var + track_var + 2 * track.get("base_covariance", 0), 0)),
        "ability_score": base.get("score", 50.5),
        "layout_score": track.get("score", 5.5),
        "rating_comparisons": base.get("comparisons", 0),
        "layout_comparisons": track.get("comparisons", 0),
    }


def compare(bundle, pairs):
    verify_bundle(bundle)
    rows = []
    for r in pairs.itertuples():
        a = estimate(bundle, r.driver_i, r.circuit_layout_id)
        b = estimate(bundle, r.driver_j, r.circuit_layout_id)
        delta = a["ability"] + a["track_effect"] - b["ability"] - b["track_effect"]
        rows.append(
            {
                "event_id": r.event_id,
                "season": r.season,
                "driver_i": r.driver_i,
                "driver_j": r.driver_j,
                "pace_gap": r.pace_gap,
                "outcome": r.outcome,
                "outcome_usable": r.outcome_usable,
                "delta": delta,
                "p_i_ahead": float(expit(delta * bundle["calibration_slope"])),
                "bundle_sha256": bundle["bundle_sha256"],
            }
        )
    return pd.DataFrame(rows)
