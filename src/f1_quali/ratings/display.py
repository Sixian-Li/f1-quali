"""Frozen rank-normal display scale; never an input to the rating model."""

from bisect import bisect_left
from collections import Counter
from math import isfinite
from statistics import NormalDist


def normal_scale(values, *, mean=6.5, sd=2.0):
    """Return [original score, mapped score, midrank percentile] knots.

    Each driver-event carries one vote; ties share their average rank. The
    half-rank offset keeps the observed extrema finite without a score cap.
    """
    values = [float(value) for value in values]
    if len(values) < 2 or not all(isfinite(value) for value in values):
        raise ValueError("At least two finite reference scores are required")
    if not isfinite(mean) or not isfinite(sd) or sd <= 0:
        raise ValueError("Finite mean and positive finite standard deviation required")
    counts = Counter(values)
    if len(counts) < 2:
        raise ValueError("At least two distinct reference scores are required")
    distribution = NormalDist(mean, sd)
    below = 0
    knots = []
    for value, count in sorted(counts.items()):
        percentile = (below + count / 2) / len(values)
        knots.append([value, distribution.inv_cdf(percentile), percentile])
        below += count
    return knots


def map_score(value, knots):
    """Monotone linear interpolation between calibrated score knots.

    Outside the frozen reference range, continue the nearest segment's slope.
    This is a display extrapolation, not a claim about an empirical percentile.
    No clipping is applied at 10 or at the observed extrema.
    """
    if not isfinite(value):
        raise ValueError("Display score must be finite")
    upper = bisect_left(knots, value, key=lambda knot: knot[0])
    if upper < len(knots) and knots[upper][0] == value:
        return knots[upper][1]
    upper = min(max(upper, 1), len(knots) - 1)
    left, right = knots[upper - 1], knots[upper]
    return left[1] + (value - left[0]) / (right[0] - left[0]) * (right[1] - left[1])
