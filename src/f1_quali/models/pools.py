"""Frozen v6 annual Q2/Q3/Top3 logistic and pole softmax predictors."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from f1_quali.data.core import fingerprint
from f1_quali.models.coordination import decode_nested, project_probabilities

TASKS = ("q2", "q3", "top3", "pole")


def _training_rows(features, labels, year, columns):
    cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
    if features.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Features require one row per event entrant")
    if labels.duplicated(["event_id", "driver_id", "stage"]).any():
        raise ValueError("Labels require one row per event entrant and stage")
    before = pd.to_datetime(features.prediction_cutoff, utc=True, errors="coerce")
    rows = (
        features[features.season.lt(year) & features.season_event_ordinal.ge(6) & before.lt(cutoff)]
        .sort_values(["event_id", "driver_id"])
        .copy()
    )
    # A truncated input field must not silently become a complete training event.
    rosters = labels[labels.stage.eq("Q1")].groupby("event_id").driver_id.agg(set)
    for event_id, group in rows.groupby("event_id"):
        if event_id not in rosters or set(group.driver_id) != rosters[event_id]:
            raise ValueError("Training features and labels require the same complete roster")
    rows["_roster_count"] = rows.groupby("event_id").driver_id.transform("size")
    # Center on the full pre-Q field, before selecting later-stage participants.
    for col in columns:
        if not np.isfinite(rows[col].astype(float)).all():
            raise ValueError(f"Nonfinite model input: {col}")
        rows[f"_x_{col}"] = rows[col] - rows.groupby("event_id")[col].transform("mean")
    keep = [c for c in labels if c not in rows or c in ("event_id", "driver_id")]
    rows = rows.merge(labels[keep], on=["event_id", "driver_id"], validate="one_to_many")
    availability = (
        "training_label_available_at"
        if "training_label_available_at" in rows
        else "label_available_at"
    )
    if availability not in rows:
        raise ValueError("Labels require a reconstructed availability timestamp")
    rows["_label_available_at"] = pd.to_datetime(rows[availability], utc=True, errors="coerce")
    rows = rows[rows._label_available_at.lt(cutoff)].copy()
    if "label_available_at" in rows:
        rows = rows[pd.to_datetime(rows.label_available_at, utc=True, errors="coerce").lt(cutoff)]
    return rows.sort_values(["event_id", "stage", "driver_id"]).reset_index(drop=True)


def _complete(rows, flag):
    if flag not in rows:
        raise ValueError(f"Missing task coverage flag: {flag}")
    known = rows[flag].eq(True)
    count = rows.groupby(["event_id", "stage"]).driver_id.transform("size")
    all_known = known.groupby([rows.event_id, rows.stage]).transform("all")
    return count.eq(rows._roster_count) & all_known


def _scale(rows, columns):
    unique = rows.drop_duplicates(["event_id", "driver_id"])
    x = unique[[f"_x_{c}" for c in columns]].to_numpy(float)
    if not len(x):
        return np.ones(len(columns))
    weights = 1 / unique.groupby("event_id").driver_id.transform("size").to_numpy()
    return np.sqrt(np.average(x * x, axis=0, weights=weights)).clip(min=0.05)


def _trace(rows, year, columns):
    hashes = (
        sorted(rows.rating_bundle_sha256.dropna().unique().tolist())
        if ("rating_bundle_sha256" in rows)
        else []
    )
    target_columns = [
        c
        for c in [
            "relative_gap",
            "eligible",
            "valid_lap",
            "no_valid_lap",
            *[f"y_{t}" for t in TASKS],
        ]
        if c in rows
    ]
    return {
        "year": int(year),
        "training_cutoff": f"{year}-01-01T00:00:00+00:00",
        "training_events": int(rows.event_id.nunique()),
        "training_rows": len(rows),
        "training_min_ordinal": int(rows.season_event_ordinal.min()) if len(rows) else None,
        "training_max_year": int(rows.season.max()) if len(rows) else None,
        "training_sha256": fingerprint(
            rows,
            ["event_id", "driver_id", "stage", "_label_available_at", *target_columns, *columns],
        ),
        "read_only_rating_bundles": hashes,
        "event_weighting": "equal event; equal stage where applicable; equal eligible driver",
    }


def _design(model, features):
    if not features.season.eq(model["trace"]["year"]).all():
        raise ValueError("Probability predictor year mismatch")
    if features.duplicated(["event_id", "driver_id"]).any():
        raise ValueError("Duplicate prediction entrant")
    x = features[model["columns"]].astype(float)
    x = x - features.groupby("event_id")[model["columns"]].transform("mean")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Prediction inputs must be finite")
    return x.to_numpy() / np.array(model["scale"])


def _logistic(x, y, weights, alpha, offset=None, prior_probability=None):
    """A weak proper intercept prior makes all-zero rare-event fits finite."""
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset)
    initial = np.zeros(x.shape[1] + 1)
    if prior_probability is not None:
        initial[0] = np.log(prior_probability / (1 - prior_probability))

    def objective(theta):
        eta = offset + theta[0] + x @ theta[1:]
        loss = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
        residual = weights * (expit(eta) - y)
        grad = np.r_[residual.sum(), x.T @ residual]
        loss += 0.5 * alpha * np.dot(theta[1:], theta[1:])
        grad[1:] += alpha * theta[1:]
        if prior_probability is not None:
            loss += np.logaddexp(0, theta[0]) - prior_probability * theta[0]
            grad[0] += expit(theta[0]) - prior_probability
        else:
            loss += 0.005 * theta[0] ** 2
            grad[0] += 0.01 * theta[0]
        return loss, grad

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success and np.max(np.abs(result.jac)) > 1e-4:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "beta": result.x[1:].tolist(),
        "converged": bool(result.success),
        "objective": float(result.fun),
    }


def fit_pool_models(features, labels, year, columns, alpha=3.0, minimum_events=12):
    """Fit independent Q2/Q3/Top3 logits and event-grouped pole softmax.

    ``alpha`` may be a scalar or a dict keyed by task, so each task can have its
    own causally selected regularization. Probabilities are coordinated only at
    prediction time; the raw independent estimates are always retained.
    """
    rows = _training_rows(features, labels, year, columns)
    rows = rows[rows.stage.eq("Q1")].copy()
    complete = rows[[f"complete_{task}" for task in TASKS]].eq(True).all(axis=1)
    complete &= (
        _complete(rows, "complete_all_tasks")
        if "complete_all_tasks" in rows
        else (
            rows.groupby(["event_id", "stage"]).driver_id.transform("size").eq(rows._roster_count)
        )
    )
    scale = _scale(rows[complete], columns)
    model = {
        "kind": "independent_pools",
        "columns": list(columns),
        "scale": scale.tolist(),
        "trace": _trace(rows[complete], year, columns),
        "tasks": {},
    }
    for task in TASKS:
        part = rows[_complete(rows, f"complete_{task}")].copy()
        if not part[f"y_{task}"].isin([0, 1]).all():
            raise ValueError(f"Invalid binary labels for {task}")
        regularization = float(alpha[task] if isinstance(alpha, dict) else alpha)
        trace = _trace(part, year, columns)
        if part.event_id.nunique() < minimum_events:
            model["tasks"][task] = {"kind": "quota_prior", "alpha": regularization, "trace": trace}
            continue
        x = part[[f"_x_{c}" for c in columns]].to_numpy(float) / scale
        y = part[f"y_{task}"].to_numpy(float)
        counts = part.groupby("event_id").driver_id.transform("size").to_numpy()
        weights = 1 / counts
        quota = (
            part[f"{task}_quota"].to_numpy(float)
            if task in ("q2", "q3")
            else np.full(len(part), 3 if task == "top3" else 1)
        )
        if not np.allclose(
            part.groupby("event_id")[f"y_{task}"].sum().to_numpy(),
            part.assign(_quota=quota).groupby("event_id")._quota.first(),
        ):
            raise ValueError(f"Complete {task} labels disagree with registered quota")
        if task != "pole":
            probability = quota / counts
            fitted = _logistic(
                x, y, weights, regularization, np.log(probability / (1 - probability))
            )
            fitted["kind"] = "logistic"
        else:
            groups = list(part.groupby("event_id", sort=False).indices.values())

            def objective(beta, x=x, y=y, groups=groups, regularization=regularization):
                eta = x @ beta
                residual = np.zeros(len(eta))
                loss = 0.5 * regularization * np.dot(beta, beta)
                for group in groups:
                    normalizer = logsumexp(eta[group])
                    loss += normalizer - np.dot(y[group], eta[group])
                    residual[group] = np.exp(eta[group] - normalizer) - y[group]
                return loss, x.T @ residual + regularization * beta

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
                "converged": bool(result.success),
                "objective": float(result.fun),
            }
        model["tasks"][task] = {**fitted, "alpha": regularization, "trace": trace}
    return model


def predict_pools(model, event_features, q2_quota, q3_quota):
    if event_features.event_id.nunique() != 1:
        raise ValueError("Pool prediction requires one full event")
    features = event_features.sort_values("driver_id").reset_index(drop=True)
    x = _design(model, features)
    n = len(features)
    quotas = [int(q2_quota), int(q3_quota), 3, 1]
    raw = np.empty((n, 4))
    for k, task in enumerate(TASKS):
        fitted = model["tasks"][task]
        if fitted["kind"] == "quota_prior":
            raw[:, k] = quotas[k] / n
        elif task == "pole":
            eta = x @ np.array(fitted["beta"])
            raw[:, k] = np.exp(eta - logsumexp(eta))
        else:
            base = quotas[k] / n
            raw[:, k] = expit(
                np.log(base / (1 - base)) + fitted["intercept"] + x @ np.array(fitted["beta"])
            )
    coordinated = project_probabilities(raw, quotas)
    out = features.copy()
    for k, task in enumerate(TASKS):
        out[f"raw_p_{task}"] = raw[:, k]
        out[f"p_{task}"] = coordinated[:, k]
    return decode_nested(out, q2_quota, q3_quota)
