"""Unit tests for the Poisson distribution family."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.distributions.families.poisson import PoissonFamily

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _as_array(xp, data):
    if xp is np:
        return np.asarray(data, dtype=float)
    if torch is not None and xp is torch:
        return torch.tensor(data, dtype=torch.float64)
    raise TypeError("Unsupported namespace")


def _to_scalar(value, xp):
    if xp is np:
        return float(value)
    if torch is not None and xp is torch:
        return value.item()
    raise TypeError("Unsupported namespace")


@pytest.mark.parametrize("xp", tuple(filter(None, (np, torch))))
def test_poisson_log_likelihood_matches_reference(xp):
    family = PoissonFamily()
    y = _as_array(xp, [0.0, 1.0, 3.0])
    mu = _as_array(xp, [0.6, 1.2, 2.8])
    result = family.log_likelihood(y, mu)
    expected = float(np.sum(np.asarray([0.0, 1.0, 3.0]) * np.log([0.6, 1.2, 2.8]) - np.asarray([0.6, 1.2, 2.8])))
    assert pytest.approx(expected) == _to_scalar(result, xp)


@pytest.mark.parametrize("xp", tuple(filter(None, (np, torch))))
def test_poisson_deviance_handles_zero_counts(xp):
    family = PoissonFamily()
    y_values = [0.0, 2.0, 4.0]
    mu_values = [0.5, 2.5, 3.5]
    y = _as_array(xp, y_values)
    mu = _as_array(xp, mu_values)
    result = family.deviance(y, mu)
    expected_terms = []
    for y_i, mu_i in zip(y_values, mu_values):
        if y_i == 0:
            expected_terms.append(2.0 * mu_i)
        else:
            expected_terms.append(2.0 * (y_i * np.log(y_i / mu_i) - (y_i - mu_i)))
    expected = float(sum(expected_terms))
    assert pytest.approx(expected) == _to_scalar(result, xp)


@pytest.mark.parametrize("xp", tuple(filter(None, (np, torch))))
def test_poisson_variance_equals_mean(xp):
    family = PoissonFamily()
    mu = _as_array(xp, [0.5, 1.5, 4.0])
    result = family.variance(mu)
    if xp is np:
        assert np.allclose(result, [0.5, 1.5, 4.0])
    else:
        assert torch.allclose(result, torch.tensor([0.5, 1.5, 4.0], dtype=result.dtype))


@pytest.mark.parametrize("xp", tuple(filter(None, (np, torch))))
def test_poisson_initialize_is_positive(xp):
    family = PoissonFamily()
    y = _as_array(xp, [0.0, 2.0, 5.0])
    init = family.initialize(y)
    min_allowed = 0.1
    if xp is np:
        assert np.all(init >= min_allowed)
    else:
        assert torch.all(init >= min_allowed)
