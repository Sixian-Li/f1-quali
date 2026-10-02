"""Mathematical and frozen-reference invariants for the presentation scale."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import norm, rankdata

from f1_quali.ratings.display import map_score, normal_scale


def test_average_rank_normal_scores_preserve_ties_and_order():
    scores = np.array([1., 8., 8., 8., 13., 21., 55.])
    knots = normal_scale(scores)
    expected = 6.5 + 2 * norm.ppf((rankdata(scores, method="average") - .5) / len(scores))
    actual = [map_score(value, knots) for value in scores]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert actual[1] == actual[2] == actual[3]
    assert all(a < b for a, b in pairwise(actual[3:]))


def test_no_ten_point_cap_and_finite_monotone_interpolation():
    knots = normal_scale(range(1000))
    assert map_score(999., knots) > 10
    assert map_score(1001., knots) > map_score(999., knots)
    assert map_score(-1., knots) < map_score(0., knots)
    grid = np.linspace(-1, 1001, 4001)
    mapped = [map_score(value, knots) for value in grid]
    assert np.isfinite(mapped).all()
    assert np.all(np.diff(mapped) > 0)
    assert map_score(123.5, knots) == pytest.approx((knots[123][1] + knots[124][1]) / 2)


def test_frozen_mapping_does_not_depend_on_display_series_or_mean_shift():
    scores = [10., 20., 30., 40., 50.]
    knots = normal_scale(scores)
    previous = [map_score(value, knots) for value in scores]
    displayed_with_future = [map_score(value, knots) for value in scores + [100.]]
    assert displayed_with_future[:-1] == previous
    shifted = normal_scale(scores, mean=6.)
    np.testing.assert_allclose(
        [map_score(value, knots) - map_score(value, shifted) for value in [5., 10., 25., 70.]],
        .5, rtol=0, atol=2e-15,
    )


@pytest.mark.parametrize("scores", [[], [1.], [1., 1.], [1., float("nan")]])
def test_invalid_reference_is_rejected(scores):
    with pytest.raises(ValueError):
        normal_scale(scores)
