"""Tests for aurora.models.zero_inflated.diagnostics module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aurora.models.zero_inflated.diagnostics import (
    rootogram,
    score_test_zero_inflation,
    vuong_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Seeded RNG for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def poisson_counts(rng):
    """Standard Poisson(3) counts -- no excess zeros."""
    return rng.poisson(3, size=100)


@pytest.fixture
def zip_like_counts(rng):
    """ZIP-like data with ~50% excess zeros."""
    return np.concatenate([np.zeros(50), rng.poisson(3, size=50)])


# ===========================================================================
# vuong_test
# ===========================================================================


class TestVuongTest:
    """Tests for the Vuong non-nested model comparison test."""

    # -- basic shape / type checks -------------------------------------------

    def test_returns_expected_keys(self, rng):
        n = 100
        ll1 = rng.randn(n)
        ll2 = rng.randn(n)
        result = vuong_test(ll1, ll2, n)
        for key in ("statistic", "p_value", "conclusion", "p_model1_better", "p_model2_better"):
            assert key in result, f"Missing key: {key}"

    def test_statistic_is_float(self, rng):
        n = 100
        result = vuong_test(rng.randn(n), rng.randn(n), n)
        assert isinstance(result["statistic"], float)

    def test_p_value_in_unit_interval(self, rng):
        n = 100
        result = vuong_test(rng.randn(n), rng.randn(n), n)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_one_sided_p_values_sum_to_two_sided(self, rng):
        """Two-sided p-value = 2 * min(p_model1_better, p_model2_better)."""
        n = 100
        result = vuong_test(rng.randn(n), rng.randn(n), n)
        min_one_sided = min(result["p_model1_better"], result["p_model2_better"])
        np.testing.assert_allclose(result["p_value"], 2 * min_one_sided, atol=1e-10)

    # -- model 1 clearly better (positive statistic) -------------------------

    def test_model1_clearly_better(self, rng):
        """When ll1 >> ll2, statistic should be large and positive."""
        n = 200
        ll1 = rng.randn(n) + 5.0  # systematically higher
        ll2 = rng.randn(n)
        result = vuong_test(ll1, ll2, n)
        assert result["statistic"] > 0
        assert result["p_model1_better"] < 0.05
        assert "Model 1" in result["conclusion"]

    # -- model 2 clearly better (negative statistic) -------------------------

    def test_model2_clearly_better(self, rng):
        """When ll2 >> ll1, statistic should be large and negative."""
        n = 200
        ll1 = rng.randn(n) - 5.0
        ll2 = rng.randn(n)
        result = vuong_test(ll1, ll2, n)
        assert result["statistic"] < 0
        assert result["p_model2_better"] < 0.05
        assert "Model 2" in result["conclusion"]

    # -- equal models (statistic near 0, large p) ----------------------------

    def test_equal_models(self, rng):
        """Identical log-likelihoods should give statistic = 0."""
        n = 100
        ll = rng.randn(n)
        result = vuong_test(ll, ll, n)
        assert abs(result["statistic"]) < 1e-10
        assert result["p_value"] > 0.99

    def test_similar_models_large_p(self, rng):
        """Very similar log-likelihoods -> p_value should be large."""
        n = 200
        ll1 = rng.randn(n)
        ll2 = ll1 + rng.randn(n) * 0.001  # tiny perturbation
        result = vuong_test(ll1, ll2, n)
        assert result["p_value"] > 0.05

    # -- correction: AIC -----------------------------------------------------

    def test_aic_correction_shifts_mean_diff(self, rng):
        """AIC correction should shift mean_diff by (k1 - k2) / n."""
        n = 100
        ll1 = rng.randn(n)
        ll2 = rng.randn(n)
        res_no = vuong_test(ll1, ll2, n, correction="none")
        res_aic = vuong_test(ll1, ll2, n, n_params1=10, n_params2=2, correction="aic")
        expected_shift = (10 - 2) / n
        np.testing.assert_allclose(
            res_no["mean_diff"] - res_aic["mean_diff"], expected_shift, rtol=1e-10
        )

    # -- correction: BIC -----------------------------------------------------

    def test_bic_correction_shifts_mean_diff(self, rng):
        """BIC correction should shift mean_diff by (k1 - k2) * ln(n) / (2n)."""
        n = 100
        ll1 = rng.randn(n)
        ll2 = rng.randn(n)
        res_no = vuong_test(ll1, ll2, n, correction="none")
        res_bic = vuong_test(ll1, ll2, n, n_params1=10, n_params2=2, correction="bic")
        expected_shift = (10 - 2) * np.log(n) / (2 * n)
        np.testing.assert_allclose(
            res_no["mean_diff"] - res_bic["mean_diff"], expected_shift, rtol=1e-10
        )

    # -- correction: none (no shift) -----------------------------------------

    def test_no_correction(self, rng):
        """With correction='none', mean_diff should equal raw mean(ll1 - ll2)."""
        n = 100
        ll1 = rng.randn(n)
        ll2 = rng.randn(n)
        result = vuong_test(ll1, ll2, n, correction="none")
        np.testing.assert_allclose(result["mean_diff"], np.mean(ll1 - ll2), rtol=1e-10)

    # -- scalar log-likelihoods ----------------------------------------------

    def test_scalar_loglik(self):
        """Scalar log-likelihoods should still produce valid results."""
        result = vuong_test(-150.0, -200.0, n=50)
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["conclusion"], str)
        # Model 1 has higher likelihood, so statistic should be positive
        assert result["statistic"] > 0

    def test_scalar_loglik_equal(self):
        """Equal scalar log-likelihoods -> m_mean = 0, statistic = 0."""
        result = vuong_test(-100.0, -100.0, n=50)
        assert result["statistic"] == 0.0

    # -- conclusion string checks -------------------------------------------

    def test_conclusion_is_string(self, rng):
        n = 50
        result = vuong_test(rng.randn(n), rng.randn(n), n)
        assert isinstance(result["conclusion"], str)
        assert len(result["conclusion"]) > 0

    def test_conclusion_not_significantly_different(self, rng):
        """Very similar models -> conclusion says not significantly different."""
        n = 200
        ll = rng.randn(n)
        result = vuong_test(ll, ll + rng.randn(n) * 0.001, n)
        assert "not significantly different" in result["conclusion"]


# ===========================================================================
# score_test_zero_inflation
# ===========================================================================


class TestScoreTestZeroInflation:
    """Tests for the score test for zero-inflation."""

    # -- basic return structure -----------------------------------------------

    def test_returns_expected_keys(self, poisson_counts):
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(poisson_counts, mu, family="poisson")
        for key in ("statistic", "p_value", "conclusion"):
            assert key in result, f"Missing key: {key}"

    def test_statistic_is_float(self, poisson_counts):
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(poisson_counts, mu, family="poisson")
        assert isinstance(result["statistic"], float)

    def test_p_value_in_unit_interval(self, poisson_counts):
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(poisson_counts, mu, family="poisson")
        assert 0.0 <= result["p_value"] <= 1.0

    # -- data without excess zeros -> non-significant -------------------------

    def test_no_excess_zeros_nonsignificant(self, rng):
        """Poisson data with no excess zeros should not flag zero-inflation."""
        mu = np.full(200, 5.0)
        y = rng.poisson(mu)
        result = score_test_zero_inflation(y, mu, family="poisson")
        # With mean=5, P(Y=0) ≈ 0.007 so very few zeros expected.
        # This should generally not be significant.
        assert result["p_value"] >= 0.0

    # -- data with excess zeros -> significant --------------------------------

    def test_excess_zeros_detected(self, rng):
        """Data with many excess zeros should give a significant test."""
        n = 200
        mu = np.full(n, 2.0)
        # Standard Poisson: ~13.5% zeros at mu=2
        y = rng.poisson(mu)
        # Inject many extra zeros
        zero_idx = rng.choice(n, 80, replace=False)
        y[zero_idx] = 0
        result = score_test_zero_inflation(y, mu, family="poisson")
        assert result["observed_zeros"] > result["expected_zeros"]
        assert result["statistic"] > 0

    def test_all_zeros_very_significant(self):
        """All-zero data should give strong evidence of zero-inflation."""
        y = np.zeros(100)
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(y, mu, family="poisson")
        assert result["statistic"] > 10.0
        assert result["p_value"] < 0.01
        assert "Strong" in result["conclusion"]

    # -- family variations ----------------------------------------------------

    def test_negative_binomial_family(self, rng):
        """Negative binomial family should work with theta provided."""
        n = 100
        mu = np.full(n, 3.0)
        y = rng.negative_binomial(5, 5 / (5 + 3), n)
        result = score_test_zero_inflation(y, mu, family="negbin", theta=5.0)
        assert isinstance(result["statistic"], float)
        assert result["p_value"] >= 0.0

    def test_nb_alias(self, rng):
        """'nb' is an alias for negative binomial."""
        n = 100
        mu = np.full(n, 3.0)
        y = rng.negative_binomial(5, 5 / (5 + 3), n)
        result = score_test_zero_inflation(y, mu, family="nb", theta=5.0)
        assert isinstance(result["statistic"], float)

    # -- error handling -------------------------------------------------------

    def test_nb_requires_theta(self):
        """Negative binomial family without theta should raise ValueError."""
        y = np.array([0, 1, 2, 3])
        mu = np.array([1.5, 1.5, 1.5, 1.5])
        with pytest.raises(ValueError, match="theta"):
            score_test_zero_inflation(y, mu, family="negbin")

    def test_unknown_family_raises(self):
        """Unknown family string should raise ValueError."""
        y = np.array([0, 1, 2])
        mu = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="family|Unknown"):
            score_test_zero_inflation(y, mu, family="gamma")

    # -- conclusion categories ------------------------------------------------

    def test_conclusion_contains_known_phrase(self, poisson_counts):
        """Conclusion should use one of the standard phrases."""
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(poisson_counts, mu, family="poisson")
        assert any(
            word in result["conclusion"]
            for word in ["Strong", "Moderate", "Weak", "No significant"]
        )

    # -- observed_zeros / expected_zeros --------------------------------------

    def test_observed_zeros_matches_data(self):
        """observed_zeros should equal the count of zeros in y."""
        y = np.array([0, 0, 0, 1, 2, 3, 4])
        mu = np.full(7, 2.0)
        result = score_test_zero_inflation(y, mu, family="poisson")
        assert result["observed_zeros"] == 3

    def test_expected_zeros_positive(self, poisson_counts):
        """expected_zeros should be positive for Poisson with mu > 0."""
        mu = np.full(100, 3.0)
        result = score_test_zero_inflation(poisson_counts, mu, family="poisson")
        assert result["expected_zeros"] > 0

    # -- edge: very small mu --------------------------------------------------

    def test_small_mu(self):
        """Very small mu should not cause numerical errors."""
        y = np.array([0, 0, 0, 0, 1])
        mu = np.full(5, 0.1)
        result = score_test_zero_inflation(y, mu, family="poisson")
        assert np.isfinite(result["statistic"])
        assert np.isfinite(result["p_value"])


# ===========================================================================
# rootogram
# ===========================================================================


class TestRootogram:
    """Tests for the rootogram diagnostic."""

    # -- basic Poisson rootogram ----------------------------------------------

    def test_poisson_basic_return_keys(self, rng):
        y = rng.poisson(3, 200)
        fitted = np.full(200, 3.0)
        result = rootogram(y, fitted, family="poisson")
        for key in ("counts", "observed", "expected", "sqrt_observed", "sqrt_expected"):
            assert key in result, f"Missing key: {key}"

    def test_poisson_observed_sums_to_n(self, rng):
        """Total observed frequencies should equal sample size."""
        n = 200
        y = rng.poisson(3, n)
        fitted = np.full(n, 3.0)
        result = rootogram(y, fitted, family="poisson")
        assert np.sum(result["observed"]) == n

    def test_poisson_observed_matches_bincount(self, rng):
        """observed should match np.bincount of y (up to max_count)."""
        y = rng.poisson(3, 200)
        fitted = np.full(200, 3.0)
        result = rootogram(y, fitted, family="poisson")
        expected_bincount = np.bincount(y, minlength=y.max() + 1)
        np.testing.assert_array_equal(result["observed"][: len(expected_bincount)], expected_bincount)

    def test_poisson_expected_positive(self, rng):
        """Expected frequencies should be non-negative."""
        y = rng.poisson(3, 200)
        fitted = np.full(200, 3.0)
        result = rootogram(y, fitted, family="poisson")
        assert np.all(result["expected"] >= 0)

    def test_poisson_lengths_match(self, rng):
        """observed, expected, counts should all have the same length."""
        y = rng.poisson(3, 200)
        fitted = np.full(200, 3.0)
        result = rootogram(y, fitted, family="poisson")
        k = len(result["counts"])
        assert len(result["observed"]) == k
        assert len(result["expected"]) == k
        assert len(result["sqrt_observed"]) == k
        assert len(result["sqrt_expected"]) == k

    # -- sqrt transform correctness -------------------------------------------

    def test_sqrt_observed_is_sqrt_of_observed(self):
        """sqrt_observed should equal np.sqrt(observed)."""
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 0, 1])
        fitted = np.full(10, 1.5)
        result = rootogram(y, fitted, family="poisson")
        np.testing.assert_allclose(result["sqrt_observed"], np.sqrt(result["observed"]))

    def test_sqrt_expected_is_sqrt_of_expected(self, rng):
        """sqrt_expected should equal np.sqrt(expected)."""
        y = rng.poisson(3, 100)
        fitted = np.full(100, 3.0)
        result = rootogram(y, fitted, family="poisson")
        np.testing.assert_allclose(result["sqrt_expected"], np.sqrt(result["expected"]))

    # -- negative binomial rootogram ------------------------------------------

    def test_negbin_basic(self, rng):
        y = rng.negative_binomial(5, 0.5, 200)
        fitted = np.full(200, 5.0)
        result = rootogram(y, fitted, family="negbin", theta=5.0)
        assert np.sum(result["observed"]) == 200
        assert np.all(result["expected"] >= 0)

    def test_negbin_requires_theta(self):
        y = np.array([0, 1, 2, 3])
        fitted = np.array([1.5, 1.5, 1.5, 1.5])
        with pytest.raises(ValueError, match="theta"):
            rootogram(y, fitted, family="negbin")

    # -- ZIP rootogram --------------------------------------------------------

    def test_zip_basic(self, rng):
        n = 200
        y = rng.poisson(2, n)
        y[:30] = 0  # excess zeros
        fitted = np.full(n, 2.0)
        result = rootogram(y, fitted, family="zip")
        assert np.sum(result["observed"]) == n

    def test_zip_expected_at_zero_higher(self, rng):
        """ZIP should predict more zeros than Poisson when excess zeros exist."""
        n = 200
        y = rng.poisson(2, n)
        y[:50] = 0  # many excess zeros
        fitted = np.full(n, 2.0)
        res_pois = rootogram(y, fitted, family="poisson")
        res_zip = rootogram(y, fitted, family="zip")
        assert res_zip["expected"][0] > res_pois["expected"][0]

    # -- ZINB rootogram -------------------------------------------------------

    def test_zinb_basic(self, rng):
        n = 200
        y = rng.negative_binomial(3, 0.4, n)
        y[:20] = 0
        fitted = np.full(n, 3.0)
        result = rootogram(y, fitted, family="zinb", theta=3.0)
        assert np.sum(result["observed"]) == n

    def test_zinb_requires_theta(self):
        y = np.array([0, 1, 2])
        fitted = np.array([1.5, 1.5, 1.5])
        with pytest.raises(ValueError, match="theta"):
            rootogram(y, fitted, family="zinb")

    # -- unknown family -------------------------------------------------------

    def test_unknown_family_raises(self):
        y = np.array([0, 1, 2])
        fitted = np.array([1.5, 1.5, 1.5])
        with pytest.raises(ValueError, match="family|Unknown"):
            rootogram(y, fitted, family="gamma")

    # -- max_count parameter --------------------------------------------------

    def test_max_count_limits_output(self, rng):
        y = rng.poisson(3, 100)
        fitted = np.full(100, 3.0)
        result = rootogram(y, fitted, family="poisson", max_count=5)
        # max_count = max(5, y.max()), so counts goes from 0..max(5, y.max())
        expected_len = max(5, int(y.max())) + 1
        assert len(result["counts"]) == expected_len

    def test_max_count_default_is_95th_percentile(self, rng):
        """Default max_count should be at least the 95th percentile of y."""
        y = rng.poisson(3, 200)
        fitted = np.full(200, 3.0)
        result = rootogram(y, fitted, family="poisson")
        # max_count is max(95th percentile, y.max()), so counts should cover all y values
        assert result["counts"][-1] >= y.max()

    # -- style variations (plotting) ------------------------------------------

    def test_standing_style(self, rng):
        y = rng.poisson(3, 100)
        fitted = np.full(100, 3.0)
        result = rootogram(y, fitted, family="poisson", style="standing")
        assert "observed" in result

    def test_hanging_style(self, rng):
        y = rng.poisson(3, 100)
        fitted = np.full(100, 3.0)
        result = rootogram(y, fitted, family="poisson", style="hanging")
        assert "observed" in result

    # -- custom axes ----------------------------------------------------------

    def test_with_custom_axes(self, rng):
        fig, ax = plt.subplots()
        y = rng.poisson(3, 100)
        fitted = np.full(100, 3.0)
        result = rootogram(y, fitted, family="poisson", ax=ax)
        assert "ax" in result
        assert result["ax"] is ax
        plt.close(fig)

    # -- typical count data range 0-10 ----------------------------------------

    def test_typical_count_range(self, rng):
        """Verify rootogram works for typical count data in the 0-10 range."""
        n = 500
        y = rng.poisson(4, n)
        fitted = np.full(n, 4.0)
        result = rootogram(y, fitted, family="poisson")
        # All counts from 0 to max should be present
        assert result["counts"][0] == 0
        assert result["counts"][-1] >= 8  # 4 + ~2*sd covers up to ~8-10

    # -- cleanup: close any figures created by rootogram plotting -------------

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        plt.close("all")
