# SPDX-License-Identifier: MIT
"""Additional coverage tests for aurora.models.hurdle.truncated module.

Covers uncovered paths: variance functions, d_log_likelihood, prob_zero_untruncated,
theta override parameter, and edge cases for both TruncatedPoisson and TruncatedNegBin.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.models.hurdle.truncated import (
    TruncatedNegBinFamily,
    TruncatedPoissonFamily,
)


# ===========================================================================
# TruncatedPoissonFamily — uncovered branches
# ===========================================================================


class TestTruncatedPoissonVariance:
    """Cover TruncatedPoissonFamily.variance."""

    def test_variance_positive(self):
        fam = TruncatedPoissonFamily()
        mu = np.array([1.0, 2.0, 5.0])
        var = fam.variance(mu)
        assert np.all(var > 0)

    def test_variance_large_mu_approaches_mu(self):
        fam = TruncatedPoissonFamily()
        mu = np.array([50.0, 100.0])
        var = fam.variance(mu)
        # For large mu, truncated variance ~ mu
        assert_allclose(var, mu, rtol=0.01)


class TestTruncatedPoissonDLogLikelihood:
    """Cover TruncatedPoissonFamily.d_log_likelihood."""

    def test_finite_output(self):
        fam = TruncatedPoissonFamily()
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0, 2.0, 2.0])
        dll = fam.d_log_likelihood(y, mu)
        assert np.all(np.isfinite(dll))
        assert dll.shape == y.shape

    def test_zero_at_mean(self):
        """d_log_likelihood should cross zero near the MLE."""
        fam = TruncatedPoissonFamily()
        mu = np.array([3.0])
        y = np.array([3.0])
        dll = fam.d_log_likelihood(y, mu)
        assert np.isfinite(dll[0])


class TestTruncatedPoissonLogLikEdgeCases:
    """Edge cases for log_likelihood."""

    def test_rejects_zero_counts(self):
        fam = TruncatedPoissonFamily()
        with pytest.raises(ValueError, match="positive"):
            fam.log_likelihood(np.array([0.0, 1.0, 2.0]), np.array([2.0, 2.0, 2.0]))

    def test_rejects_negative_counts(self):
        fam = TruncatedPoissonFamily()
        with pytest.raises(ValueError, match="positive"):
            fam.log_likelihood(np.array([-1.0, 1.0]), np.array([2.0, 2.0]))


# ===========================================================================
# TruncatedNegBinFamily — uncovered branches
# ===========================================================================


class TestTruncatedNegBinVariance:
    """Cover TruncatedNegBinFamily.variance."""

    def test_variance_positive(self):
        fam = TruncatedNegBinFamily(theta=2.0)
        mu = np.array([1.0, 3.0, 5.0])
        var = fam.variance(mu)
        assert np.all(var > 0)

    def test_theta_override(self):
        fam = TruncatedNegBinFamily(theta=1.0)
        mu = np.array([2.0, 3.0])
        var_default = fam.variance(mu)
        var_override = fam.variance(mu, theta=5.0)
        # Different theta should give different variance
        assert not np.allclose(var_default, var_override)


class TestTruncatedNegBinThetaOverride:
    """Cover theta parameter override in all methods."""

    def test_log_likelihood_theta_override(self):
        fam = TruncatedNegBinFamily(theta=1.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0, 2.0, 2.0])
        ll1 = fam.log_likelihood(y, mu, theta=1.0)
        ll3 = fam.log_likelihood(y, mu, theta=3.0)
        assert np.isfinite(ll1)
        assert np.isfinite(ll3)
        assert ll1 != ll3

    def test_truncated_mean_theta_override(self):
        fam = TruncatedNegBinFamily(theta=1.0)
        mu = np.array([2.0, 3.0])
        tm1 = fam.truncated_mean(mu, theta=1.0)
        tm3 = fam.truncated_mean(mu, theta=3.0)
        assert np.all(np.isfinite(tm1))
        assert np.all(np.isfinite(tm3))

    def test_prob_zero_untruncated(self):
        fam = TruncatedNegBinFamily(theta=2.0)
        mu = np.array([1.0, 3.0, 5.0])
        p0 = fam.prob_zero_untruncated(mu)
        assert np.all((p0 >= 0) & (p0 <= 1))
        # Higher mu -> lower P(Y=0)
        assert p0[0] > p0[2]

    def test_prob_zero_theta_override(self):
        fam = TruncatedNegBinFamily(theta=1.0)
        mu = np.array([2.0])
        p0 = fam.prob_zero_untruncated(mu, theta=5.0)
        assert np.all((p0 >= 0) & (p0 <= 1))


class TestTruncatedNegBinLogLikEdgeCases:
    """Edge cases for TruncatedNegBinFamily.log_likelihood."""

    def test_rejects_zero_counts(self):
        fam = TruncatedNegBinFamily(theta=2.0)
        with pytest.raises(ValueError, match="positive"):
            fam.log_likelihood(np.array([0.0, 1.0]), np.array([2.0, 2.0]))

    def test_finite_for_various_theta(self):
        fam = TruncatedNegBinFamily(theta=0.5)
        y = np.array([1.0, 2.0, 5.0])
        mu = np.array([2.0, 2.0, 2.0])
        ll = fam.log_likelihood(y, mu)
        assert np.isfinite(ll)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
