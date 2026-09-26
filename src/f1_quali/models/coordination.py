"""Original v6 Euclidean probability coordination and nested pool decoding."""

import numpy as np
from scipy.optimize import linear_sum_assignment

TASKS = ("q2", "q3", "top3", "pole")


def _quotas(n, q2, q3):
    quotas = np.array([q2, q3, 3, 1], dtype=int)
    if not (n >= q2 >= q3 >= 3):
        raise ValueError("Invalid nested qualifying quotas")
    return quotas


def _isotonic_decreasing(values):
    levels, sizes = [], []
    for value in values:
        levels.append(float(value))
        sizes.append(1)
        while len(levels) > 1 and levels[-2] < levels[-1]:
            count = sizes[-2] + sizes[-1]
            mean = (levels[-2] * sizes[-2] + levels[-1] * sizes[-1]) / count
            levels[-2:] = [mean]
            sizes[-2:] = [count]
    return np.repeat(levels, sizes)


def project_probabilities(probabilities, quotas, tolerance=1e-10, max_iterations=3000):
    """Euclidean Dykstra projection onto row nesting and column capacities.

    Projection is a consistency operation; it is not a claim of calibration.
    """
    original = np.asarray(probabilities, dtype=float)
    capacities = np.asarray(quotas, dtype=float)
    if original.ndim != 2 or original.shape[1] != 4 or not np.isfinite(original).all():
        raise ValueError("Expected a finite entrant-by-four probability matrix")
    if (
        capacities.shape != (4,)
        or (capacities < 0).any()
        or (capacities > len(original)).any()
        or (np.diff(capacities) > 0).any()
    ):
        raise ValueError("Impossible nested capacities")
    x = original.copy()
    row_correction = np.zeros_like(x)
    col_correction = np.zeros_like(x)
    for _ in range(max_iterations):
        previous = x.copy()
        incoming = x + row_correction
        y = np.vstack([_isotonic_decreasing(row) for row in incoming]).clip(0, 1)
        row_correction = incoming - y
        incoming = y + col_correction
        # Bounded simplex projection: clip(v - lambda, 0, 1), sum == K.
        low = incoming.min(axis=0) - 1
        high = incoming.max(axis=0)
        for _ in range(48):
            middle = (low + high) / 2
            sums = np.clip(incoming - middle, 0, 1).sum(axis=0)
            low = np.where(sums > capacities, middle, low)
            high = np.where(sums > capacities, high, middle)
        x = np.clip(incoming - (low + high) / 2, 0, 1)
        col_correction = incoming - x
        if np.max(np.abs(x - previous)) < tolerance and np.max(np.diff(x, axis=1)) < tolerance:
            return x
    raise RuntimeError("Pool probability projection did not converge")


def decode_nested(predictions, q2_quota, q3_quota):
    """Maximize equal-task expected overlap with one joint rank assignment."""
    if predictions.event_id.nunique() != 1:
        raise ValueError("Nested decoding requires one event")
    out = predictions.sort_values("driver_id").reset_index(drop=True).copy()
    quotas = _quotas(len(out), q2_quota, q3_quota)
    p = out[[f"p_{task}" for task in TASKS]].to_numpy(float)
    membership = np.arange(1, len(out) + 1)[None, :] <= quotas[:, None]
    utility = (p / quotas) @ membership
    # Resolve ties within a membership bucket reproducibly; this auxiliary
    # ordering cannot materially change the primary expected-overlap objective.
    score = p @ np.array([1.0, 2.0, 3.0, 4.0])
    utility += 1e-12 * score[:, None] * np.arange(len(out), 0, -1)[None, :]
    rows, columns = linear_sum_assignment(-utility)
    ranks = np.empty(len(out), dtype=int)
    ranks[rows] = columns + 1
    out["predicted_rank"] = ranks
    for task, quota in zip(TASKS, quotas):
        out[f"hard_{task}"] = ranks <= quota
    return out
