"""Tests for Bayesian inference modules (result, priors, glm_bayes).

Covers:
- BayesianGLMResult: construction, properties, summary, credible intervals,
  predictions, posterior predictive, WAIC, factory methods
- BayesianGAMResult: smooth_summary
- Prior classes: Normal, Cauchy, HalfNormal, HalfCauchy, Exponential, Gamma,
  InverseGamma, Uniform, StudentT, Laplace, PriorSpec
- fit_glm_bayes: input validation, backend dispatch
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from aurora.models.bayes.glm_bayes import fit_glm_bayes
from aurora.models.bayes.priors import (
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    InverseGamma,
    Laplace,
    Normal,
    Prior,
    PriorSpec,
    StudentT,
    Uniform,
)
from aurora.models.bayes.result import BayesianGAMResult, BayesianGLMResult


# =============================================================================
# Helpers
# =============================================================================


def _make_result(
    family="gaussian",
    link="identity",
    n_chains=2,
    n_samples=50,
    n_obs=20,
    n_features=3,
    include_sigma=True,
    backend="numpyro",
):
    """Build a BayesianGLMResult with synthetic posterior samples."""
    np.random.seed(42)

    total = n_chains * n_samples
    beta = np.random.randn(n_chains, n_samples, n_features)
    samples = {"beta": beta}

    if include_sigma:
        samples["sigma"] = np.abs(np.random.randn(n_chains, n_samples)) + 0.5

    X = np.random.randn(n_obs, n_features)
    _coef = np.random.randn(n_features) * 0.5
    y = X @ _coef + np.random.randn(n_obs) * 0.5

    return BayesianGLMResult(
        posterior_samples_=samples,
        family=family,
        link=link,
        n_samples_=n_samples,
        n_chains_=n_chains,
        n_obs_=n_obs,
        n_features_=n_features,
        X_=X,
        y_=y,
        backend_=backend,
    )


# =============================================================================
# BayesianGLMResult — Construction & Properties
# =============================================================================


class TestBayesianGLMResultConstruction:
    """Test result object creation and basic properties."""

    def test_coef_samples_3d(self):
        """coef_samples_ should flatten chains: (C, S, p) -> (C*S, p)."""
        result = _make_result(n_chains=3, n_samples=40, n_features=4)
        assert result.coef_samples_.shape == (120, 4)

    def test_coef_samples_2d(self):
        """coef_samples_ should pass through 2D arrays."""
        np.random.seed(42)
        beta = np.random.randn(80, 3)  # Already flattened
        samples = {"beta": beta}
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        result = BayesianGLMResult(
            posterior_samples_=samples,
            family="gaussian",
            link="identity",
            n_samples_=80,
            n_chains_=1,
            n_obs_=20,
            n_features_=3,
            X_=X,
            y_=y,
            backend_="numpyro",
        )
        assert result.coef_samples_.shape == (80, 3)

    def test_coef_samples_missing_raises(self):
        """coef_samples_ should raise KeyError when no beta or coef key."""
        result = _make_result()
        result.posterior_samples_ = {"sigma": np.array([1.0])}
        with pytest.raises(KeyError, match="No coefficient samples"):
            _ = result.coef_samples_

    def test_coef_samples_coef_key(self):
        """coef_samples_ should accept 'coef' as alternative key."""
        np.random.seed(42)
        result = _make_result()
        beta_data = result.posterior_samples_.pop("beta")
        result.posterior_samples_["coef"] = beta_data
        assert result.coef_samples_.shape[1] == 3

    def test_coef_mean(self):
        """coef_ should return posterior mean of coefficients."""
        result = _make_result(n_chains=1, n_samples=1000, n_features=2)
        assert result.coef_.shape == (2,)

    def test_coef_std(self):
        """coef_std_ should return posterior std of coefficients."""
        result = _make_result(n_chains=1, n_samples=1000, n_features=2)
        assert result.coef_std_.shape == (2,)
        assert np.all(result.coef_std_ >= 0)

    def test_scale_samples_present(self):
        """scale_samples_ should return sigma when available."""
        result = _make_result(include_sigma=True)
        assert result.scale_samples_ is not None
        # Should be flattened
        assert result.scale_samples_.ndim == 1

    def test_scale_samples_absent(self):
        """scale_samples_ should return None when no scale parameter."""
        result = _make_result(include_sigma=False)
        assert result.scale_samples_ is None

    def test_scale_samples_alternate_keys(self):
        """scale_samples_ should find 'scale' and 'phi' keys too."""
        result = _make_result()
        sigma_data = result.posterior_samples_.pop("sigma")

        result.posterior_samples_["scale"] = sigma_data
        assert result.scale_samples_ is not None

        result.posterior_samples_.pop("scale")
        result.posterior_samples_["phi"] = sigma_data
        assert result.scale_samples_ is not None

    def test_scale_mean(self):
        """scale_ should return float mean of scale samples."""
        result = _make_result(include_sigma=True)
        assert isinstance(result.scale_, float)

    def test_scale_mean_none(self):
        """scale_ should return None when no scale parameter."""
        result = _make_result(include_sigma=False)
        assert result.scale_ is None


# =============================================================================
# BayesianGLMResult — Summary
# =============================================================================


class TestBayesianGLMResultSummary:
    """Test summary() method."""

    def test_summary_keys(self):
        """Summary should contain expected top-level keys."""
        result = _make_result()
        s = result.summary()
        for key in ["coef", "n_obs", "n_features", "n_samples", "n_chains", "family", "link", "backend"]:
            assert key in s

    def test_summary_coef_keys(self):
        """Coefficient summary should have mean, std, percentiles."""
        result = _make_result()
        s = result.summary()
        assert "mean" in s["coef"]
        assert "std" in s["coef"]
        assert "percentiles" in s["coef"]

    def test_summary_scale(self):
        """Summary should include scale when available."""
        result = _make_result(include_sigma=True)
        s = result.summary()
        assert "scale" in s

    def test_summary_no_scale(self):
        """Summary should not include scale when absent."""
        result = _make_result(include_sigma=False)
        s = result.summary()
        assert "scale" not in s

    def test_summary_custom_percentiles(self):
        """Summary should respect custom percentiles."""
        result = _make_result()
        s = result.summary(percentiles=(5, 50, 95))
        assert set(s["coef"]["percentiles"].keys()) == {5, 50, 95}

    def test_summary_with_rhat_ess(self):
        """Summary should include R-hat and ESS when present."""
        result = _make_result()
        result._r_hat = {"beta": np.array([1.0, 1.01, 0.99])}
        result._ess = {"beta": np.array([200, 180, 210])}
        s = result.summary()
        assert_allclose(s["coef"]["r_hat"], [1.0, 1.01, 0.99])
        assert_allclose(s["coef"]["ess"], [200, 180, 210])

    def test_summary_total_samples(self):
        """n_samples in summary should be n_samples_ * n_chains_."""
        result = _make_result(n_chains=3, n_samples=100)
        assert result.summary()["n_samples"] == 300


# =============================================================================
# BayesianGLMResult — Credible Intervals
# =============================================================================


class TestBayesianGLMResultCredibleIntervals:
    """Test credible_intervals() method."""

    def test_eti_shape(self):
        """ETI should return (n_features, 2) array for coef."""
        result = _make_result(n_features=4)
        ci = result.credible_intervals(level=0.95, method="eti")
        assert ci["coef"].shape == (4, 2)

    def test_hdi_shape(self):
        """HDI should return (n_features, 2) array for coef."""
        result = _make_result(n_features=4)
        ci = result.credible_intervals(level=0.95, method="hdi")
        assert ci["coef"].shape == (4, 2)

    def test_eti_contains_posterior_mean(self):
        """95% ETI should contain posterior mean for large samples."""
        np.random.seed(42)
        result = _make_result(n_chains=1, n_samples=10000, n_features=1)
        ci = result.credible_intervals(level=0.95, method="eti")
        mean = result.coef_[0]
        assert ci["coef"][0, 0] <= mean <= ci["coef"][0, 1]

    def test_hdi_narrower_than_eti(self):
        """HDI should be narrower than or equal to ETI."""
        result = _make_result(n_chains=1, n_samples=5000, n_features=2)
        hdi = result.credible_intervals(level=0.95, method="hdi")["coef"]
        eti = result.credible_intervals(level=0.95, method="eti")["coef"]
        hdi_width = hdi[:, 1] - hdi[:, 0]
        eti_width = eti[:, 1] - eti[:, 0]
        assert np.all(hdi_width <= eti_width + 1e-10)

    def test_scale_credible_interval(self):
        """Credible intervals should include scale when present."""
        result = _make_result(include_sigma=True)
        ci = result.credible_intervals(level=0.95, method="eti")
        assert "scale" in ci
        assert ci["scale"].shape == (2,)

    def test_hdi_scale(self):
        """HDI scale should return 2-element array."""
        result = _make_result(include_sigma=True)
        ci = result.credible_intervals(level=0.95, method="hdi")
        assert "scale" in ci
        assert ci["scale"].shape == (2,)


# =============================================================================
# BayesianGLMResult — Predictions
# =============================================================================


class TestBayesianGLMResultPredict:
    """Test predict() and _apply_inverse_link()."""

    def test_predict_mean_shape(self):
        """predict(type='mean') should return (n_new,) array."""
        result = _make_result(n_features=3)
        X_new = np.random.randn(10, 3)
        pred = result.predict(X_new, type="mean")
        assert pred.shape == (10,)

    def test_predict_samples_shape(self):
        """predict(type='samples') should return (n_samples, n_new) array."""
        result = _make_result(n_features=3)
        X_new = np.random.randn(10, 3)
        pred = result.predict(X_new, type="samples", n_samples=5)
        assert pred.shape == (5, 10)

    def test_predict_uses_training_data_when_none(self):
        """predict with X_new=None should use training data."""
        result = _make_result(n_features=3)
        pred = result.predict(None, type="mean")
        assert pred.shape == (20,)

    def test_predict_subsample(self):
        """predict with n_samples should subsample the posterior."""
        result = _make_result(n_chains=1, n_samples=100, n_features=2)
        pred = result.predict(np.random.randn(5, 2), type="samples", n_samples=10)
        assert pred.shape == (10, 5)

    def test_inverse_link_identity(self):
        """Identity link should pass through."""
        result = _make_result(link="identity")
        eta = np.array([1.0, -2.0, 3.0])
        assert_allclose(result._apply_inverse_link(eta), eta)

    def test_inverse_link_log(self):
        """Log link should apply exp()."""
        result = _make_result(link="log")
        eta = np.array([0.0, 1.0, -1.0])
        assert_allclose(result._apply_inverse_link(eta), np.exp(eta))

    def test_inverse_link_logit(self):
        """Logit link should apply sigmoid."""
        result = _make_result(link="logit")
        eta = np.array([0.0, 10.0, -10.0])
        expected = 1 / (1 + np.exp(-eta))
        assert_allclose(result._apply_inverse_link(eta), expected)

    def test_inverse_link_probit(self):
        """Probit link should apply norm.cdf."""
        result = _make_result(link="probit")
        from scipy.stats import norm

        eta = np.array([-1.0, 0.0, 1.0])
        assert_allclose(result._apply_inverse_link(eta), norm.cdf(eta))

    def test_inverse_link_inverse(self):
        """Inverse link should return 1/eta."""
        result = _make_result(link="inverse")
        eta = np.array([1.0, 2.0, 0.5])
        assert_allclose(result._apply_inverse_link(eta), 1.0 / eta)

    def test_inverse_link_sqrt(self):
        """Sqrt link should square the input."""
        result = _make_result(link="sqrt")
        eta = np.array([2.0, -3.0, 0.0])
        assert_allclose(result._apply_inverse_link(eta), eta**2)

    def test_inverse_link_unknown_raises(self):
        """Unknown link should raise ValueError."""
        result = _make_result(link="unknown_link")
        with pytest.raises(ValueError, match="Unknown link"):
            result._apply_inverse_link(np.array([1.0]))

    def test_inverse_link_none(self):
        """None link should behave like identity."""
        result = _make_result(link=None)
        eta = np.array([1.5, -2.0])
        assert_allclose(result._apply_inverse_link(eta), eta)


# =============================================================================
# BayesianGLMResult — Posterior Predictive
# =============================================================================


class TestBayesianGLMResultPosteriorPredictive:
    """Test posterior_predictive() method."""

    def test_gaussian_shape(self):
        """Gaussian posterior predictive should have correct shape."""
        result = _make_result(family="gaussian", include_sigma=True)
        y_pred = result.posterior_predictive(n_samples=50)
        assert y_pred.shape == (50, 20)

    def test_poisson_shape(self):
        """Poisson posterior predictive should have correct shape."""
        result = _make_result(family="poisson", link="log", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=30)
        assert y_pred.shape == (30, 20)

    def test_binomial_shape(self):
        """Binomial posterior predictive should have correct shape."""
        result = _make_result(family="binomial", link="logit", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=30)
        assert y_pred.shape == (30, 20)

    def test_gamma_shape(self):
        """Gamma posterior predictive should have correct shape."""
        result = _make_result(family="gamma", link="log", include_sigma=True)
        y_pred = result.posterior_predictive(n_samples=20)
        assert y_pred.shape == (20, 20)

    def test_negative_binomial_shape(self):
        """Negative binomial posterior predictive should have correct shape."""
        result = _make_result(family="negative_binomial", link="log", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=20)
        assert y_pred.shape == (20, 20)

    def test_unknown_family_defaults_gaussian(self):
        """Unknown family should fall back to Gaussian."""
        result = _make_result(family="cauchy", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=10)
        assert y_pred.shape == (10, 20)

    def test_with_custom_X(self):
        """Posterior predictive with custom X_new should work."""
        result = _make_result(n_features=3)
        X_new = np.random.randn(7, 3)
        y_pred = result.posterior_predictive(X_new=X_new, n_samples=15)
        assert y_pred.shape == (15, 7)


# =============================================================================
# BayesianGLMResult — WAIC
# =============================================================================


class TestBayesianGLMResultWAIC:
    """Test waic() method."""

    def test_waic_keys(self):
        """WAIC should return dict with 'waic', 'p_waic', 'se'."""
        result = _make_result(family="gaussian", include_sigma=True, n_samples=100, n_chains=2)
        w = result.waic()
        assert "waic" in w
        assert "p_waic" in w
        assert "se" in w

    def test_waic_finite(self):
        """WAIC components should be finite."""
        result = _make_result(family="gaussian", include_sigma=True, n_samples=100, n_chains=2)
        w = result.waic()
        assert np.isfinite(w["waic"])
        assert np.isfinite(w["p_waic"])
        assert np.isfinite(w["se"])

    def test_waic_poisson(self):
        """WAIC should work for Poisson family."""
        result = _make_result(family="poisson", link="log", include_sigma=False, n_samples=100)
        y_nonneg = np.abs(result.y_).astype(int)
        result.y_ = y_nonneg
        w = result.waic()
        assert np.isfinite(w["waic"])

    def test_waic_binomial(self):
        """WAIC should work for Binomial family."""
        result = _make_result(family="binomial", link="logit", include_sigma=False, n_samples=100)
        result.y_ = (result.y_ > 0).astype(float)
        w = result.waic()
        assert np.isfinite(w["waic"])


# =============================================================================
# BayesianGLMResult — Factory Methods
# =============================================================================


class TestBayesianGLMResultFactoryMethods:
    """Test from_numpyro() and from_pymc() class methods."""

    def test_from_numpyro(self):
        """from_numpyro should construct result from flat sample dict."""
        np.random.seed(42)
        n, p = 30, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        samples = {"beta": np.random.randn(400, p)}

        result = BayesianGLMResult.from_numpyro(samples, X, y, "gaussian", "identity", n_chains=1)
        assert result.n_samples_ == 400
        assert result.n_chains_ == 1
        assert result.backend_ == "numpyro"
        assert result.n_obs_ == 30
        assert result.n_features_ == 3

    def test_from_numpyro_multi_chain(self):
        """from_numpyro with multiple chains should compute per-chain samples."""
        np.random.seed(42)
        n, p = 20, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        samples = {"beta": np.random.randn(1000, p)}  # 4 chains * 250

        result = BayesianGLMResult.from_numpyro(samples, X, y, "gaussian", "identity", n_chains=4)
        assert result.n_samples_ == 250
        assert result.n_chains_ == 4

    def test_from_pymc(self):
        """from_pymc should construct result from arviz InferenceData."""
        np.random.seed(42)
        n, p = 20, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)

        # Mock arviz InferenceData
        mock_trace = MagicMock()
        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["beta", "sigma"]
        mock_posterior.__getitem__ = lambda self_dict, key: MagicMock(
            values=np.random.randn(4, 500, p) if key == "beta" else np.abs(np.random.randn(4, 500)) + 0.1
        )
        mock_trace.posterior = mock_posterior

        result = BayesianGLMResult.from_pymc(mock_trace, X, y, "gaussian", "identity")
        assert result.n_chains_ == 4
        assert result.n_samples_ == 500
        assert result.backend_ == "pymc"


# =============================================================================
# BayesianGAMResult
# =============================================================================


class TestBayesianGAMResult:
    """Test BayesianGAMResult smooth_summary()."""

    def test_smooth_summary(self):
        """smooth_summary should return info for specified smooth term."""
        np.random.seed(42)
        n_chains, n_samples, n_features = 2, 50, 5
        result = BayesianGAMResult(
            posterior_samples_={"beta": np.random.randn(n_chains, n_samples, n_features)},
            family="gaussian",
            link="identity",
            n_samples_=n_samples,
            n_chains_=n_chains,
            n_obs_=30,
            n_features_=n_features,
            X_=np.random.randn(30, n_features),
            y_=np.random.randn(30),
            backend_="numpyro",
            smooth_names_=["s(x1)", "s(x2)"],
            smooth_indices_={"s(x1)": (0, 3), "s(x2)": (3, 5)},
        )

        summary = result.smooth_summary("s(x1)")
        assert summary["name"] == "s(x1)"
        assert summary["n_basis"] == 3
        assert summary["coef_mean"].shape == (3,)
        assert summary["coef_std"].shape == (3,)

    def test_smooth_summary_unknown_raises(self):
        """smooth_summary should raise for unknown smooth term."""
        result = BayesianGAMResult(
            posterior_samples_={"beta": np.random.randn(2, 50, 3)},
            family="gaussian",
            link="identity",
            n_samples_=50,
            n_chains_=2,
            n_obs_=20,
            n_features_=3,
            X_=np.random.randn(20, 3),
            y_=np.random.randn(20),
            backend_="numpyro",
            smooth_names_=[],
            smooth_indices_={},
        )
        with pytest.raises(ValueError, match="Unknown smooth term"):
            result.smooth_summary("nonexistent")


# =============================================================================
# Prior Classes
# =============================================================================


class TestPriorBase:
    """Test Prior base class."""

    def test_to_numpyro_not_implemented(self):
        """Base Prior.to_numpyro should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            Prior().to_numpyro()

    def test_to_pymc_not_implemented(self):
        """Base Prior.to_pymc should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            Prior().to_pymc()


class TestNormal:
    """Test Normal prior."""

    def test_defaults(self):
        """Default Normal should be N(0, 10)."""
        p = Normal()
        assert p.mu == 0.0
        assert p.sigma == 10.0

    def test_custom(self):
        """Custom parameters should be stored."""
        p = Normal(5.0, 2.0)
        assert p.mu == 5.0
        assert p.sigma == 2.0

    def test_negative_sigma_raises(self):
        """Negative sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(0, -1.0)

    def test_zero_sigma_raises(self):
        """Zero sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            Normal(0, 0.0)


class TestCauchy:
    """Test Cauchy prior."""

    def test_defaults(self):
        p = Cauchy()
        assert p.loc == 0.0
        assert p.scale == 2.5

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            Cauchy(0, -1.0)


class TestHalfNormal:
    """Test HalfNormal prior."""

    def test_defaults(self):
        p = HalfNormal()
        assert p.sigma == 1.0

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            HalfNormal(0.0)


class TestHalfCauchy:
    """Test HalfCauchy prior."""

    def test_defaults(self):
        p = HalfCauchy()
        assert p.scale == 2.5

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            HalfCauchy(-1.0)


class TestExponential:
    """Test Exponential prior."""

    def test_defaults(self):
        p = Exponential()
        assert p.rate == 1.0

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            Exponential(0.0)


class TestGamma:
    """Test Gamma prior."""

    def test_defaults(self):
        p = Gamma()
        assert p.alpha == 1.0
        assert p.beta == 1.0

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            Gamma(-1.0, 1.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            Gamma(1.0, 0.0)


class TestInverseGamma:
    """Test InverseGamma prior."""

    def test_defaults(self):
        p = InverseGamma()
        assert p.alpha == 1.0
        assert p.beta == 1.0

    def test_negative_params_raises(self):
        with pytest.raises(ValueError, match="alpha and beta must be positive"):
            InverseGamma(0.0, 1.0)


class TestUniform:
    """Test Uniform prior."""

    def test_defaults(self):
        p = Uniform()
        assert p.lower == 0.0
        assert p.upper == 1.0

    def test_lower_gte_upper_raises(self):
        with pytest.raises(ValueError, match="lower must be less than upper"):
            Uniform(1.0, 1.0)

    def test_lower_gt_upper_raises(self):
        with pytest.raises(ValueError, match="lower must be less than upper"):
            Uniform(2.0, 1.0)


class TestStudentT:
    """Test StudentT prior."""

    def test_defaults(self):
        p = StudentT()
        assert p.df == 3.0
        assert p.loc == 0.0
        assert p.scale == 1.0

    def test_negative_df_raises(self):
        with pytest.raises(ValueError, match="df and scale must be positive"):
            StudentT(-1.0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="df and scale must be positive"):
            StudentT(3.0, 0.0, 0.0)


class TestLaplace:
    """Test Laplace prior."""

    def test_defaults(self):
        p = Laplace()
        assert p.loc == 0.0
        assert p.scale == 1.0

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            Laplace(0, 0.0)


# =============================================================================
# PriorSpec
# =============================================================================


class TestPriorSpec:
    """Test PriorSpec default and custom configuration."""

    def test_defaults(self):
        """Default PriorSpec should have standard weakly informative priors."""
        spec = PriorSpec()
        assert isinstance(spec.intercept_prior, Normal)
        assert isinstance(spec.coef_prior, Normal)
        assert isinstance(spec.scale_prior, HalfNormal)
        assert isinstance(spec.shape_prior, Exponential)

    def test_default_intercept_prior(self):
        """Default intercept prior should be Normal(0, 100)."""
        spec = PriorSpec()
        assert spec.intercept_prior.mu == 0.0
        assert spec.intercept_prior.sigma == 100.0

    def test_default_coef_prior(self):
        """Default coef prior should be Normal(0, 10)."""
        spec = PriorSpec()
        assert spec.coef_prior.mu == 0.0
        assert spec.coef_prior.sigma == 10.0

    def test_set_coef_prior_all(self):
        """set_coef_prior with no indices should set default."""
        spec = PriorSpec()
        new_prior = Normal(0, 1)
        spec.set_coef_prior(new_prior)
        assert spec.coef_prior is new_prior

    def test_set_coef_prior_specific(self):
        """set_coef_prior with indices should set per-index priors."""
        spec = PriorSpec()
        new_prior = Laplace(0, 1)
        spec.set_coef_prior(new_prior, indices=[0, 2])
        assert spec.get_coef_prior(0) is new_prior
        assert spec.get_coef_prior(2) is new_prior
        # Unset indices should return default
        assert isinstance(spec.get_coef_prior(1), Normal)

    def test_get_coef_prior_default(self):
        """get_coef_prior for unset index should return coef_prior."""
        spec = PriorSpec()
        result = spec.get_coef_prior(99)
        assert result is spec.coef_prior

    def test_to_dict(self):
        """to_dict should return dict with all four priors."""
        spec = PriorSpec()
        d = spec.to_dict()
        assert set(d.keys()) == {"intercept_prior", "coef_prior", "scale_prior", "shape_prior"}
        assert isinstance(d["intercept_prior"], Normal)
        assert isinstance(d["scale_prior"], HalfNormal)


# =============================================================================
# fit_glm_bayes — Input Validation & Dispatch
# =============================================================================


class TestFitGlmBayes:
    """Test fit_glm_bayes validation and backend dispatch."""

    def test_mismatched_rows_raises(self):
        """X and y with different lengths should raise ValueError."""
        X = np.random.randn(10, 3)
        y = np.random.randn(5)
        with pytest.raises(ValueError, match="X has 10 rows but y has 5"):
            fit_glm_bayes(X, y, family="gaussian")

    def test_invalid_family_raises(self):
        """Invalid family name should raise ValueError."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="Unknown family"):
            fit_glm_bayes(X, y, family="cauchy")

    def test_invalid_backend_raises(self):
        """Invalid backend name should raise ValueError."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="Unknown backend"):
            fit_glm_bayes(X, y, family="gaussian", backend="stan")

    @patch("aurora.models.bayes.glm_bayes._fit_numpyro")
    def test_numpyro_backend_dispatch(self, mock_fit):
        """backend='numpyro' should call _fit_numpyro."""
        mock_fit.return_value = MagicMock()
        import aurora.models.bayes.glm_bayes as gb

        original = gb.HAS_NUMPYRO
        gb.HAS_NUMPYRO = True
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            fit_glm_bayes(X, y, family="gaussian", backend="numpyro", seed=42)
            mock_fit.assert_called_once()
        finally:
            gb.HAS_NUMPYRO = original

    @patch("aurora.models.bayes.glm_bayes._fit_pymc")
    def test_pymc_backend_dispatch(self, mock_fit):
        """backend='pymc' should call _fit_pymc."""
        mock_fit.return_value = MagicMock()
        import aurora.models.bayes.glm_bayes as gb

        original = gb.HAS_PYMC
        gb.HAS_PYMC = True
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            fit_glm_bayes(X, y, family="gaussian", backend="pymc", seed=42)
            mock_fit.assert_called_once()
        finally:
            gb.HAS_PYMC = original

    def test_numpyro_not_installed_raises(self):
        """Should raise ImportError when numpyro is not installed."""
        import aurora.models.bayes.glm_bayes as gb

        original = gb.HAS_NUMPYRO
        gb.HAS_NUMPYRO = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            with pytest.raises(ImportError, match="NumPyro is not installed"):
                fit_glm_bayes(X, y, family="gaussian", backend="numpyro")
        finally:
            gb.HAS_NUMPYRO = original

    def test_pymc_not_installed_raises(self):
        """Should raise ImportError when pymc is not installed."""
        import aurora.models.bayes.glm_bayes as gb

        original = gb.HAS_PYMC
        gb.HAS_PYMC = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            with pytest.raises(ImportError, match="PyMC is not installed"):
                fit_glm_bayes(X, y, family="gaussian", backend="pymc")
        finally:
            gb.HAS_PYMC = original

    def test_no_backend_available_raises(self):
        """Should raise ImportError when no backend is available."""
        import aurora.models.bayes.glm_bayes as gb

        orig_np = gb.HAS_NUMPYRO
        orig_pm = gb.HAS_PYMC
        gb.HAS_NUMPYRO = False
        gb.HAS_PYMC = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            with pytest.raises(ImportError, match="No Bayesian backend"):
                fit_glm_bayes(X, y, family="gaussian")
        finally:
            gb.HAS_NUMPYRO = orig_np
            gb.HAS_PYMC = orig_pm

    def test_valid_families(self):
        """All valid family names should pass validation (no backend call)."""
        import aurora.models.bayes.glm_bayes as gb

        orig_np = gb.HAS_NUMPYRO
        gb.HAS_NUMPYRO = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            for family in ["gaussian", "poisson", "binomial", "gamma", "negative_binomial", "beta"]:
                # Should not raise ValueError for family
                try:
                    fit_glm_bayes(X, y, family=family)
                except (ImportError, ValueError) as e:
                    # ImportError from no backend is expected; ValueError is not
                    if isinstance(e, ValueError):
                        pytest.fail(f"Family '{family}' raised ValueError")
        finally:
            gb.HAS_NUMPYRO = orig_np

    def test_default_priors_created(self):
        """When priors=None, default PriorSpec should be created."""
        import aurora.models.bayes.glm_bayes as gb

        orig_np = gb.HAS_NUMPYRO
        gb.HAS_NUMPYRO = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            # Should not error on prior creation, only on backend
            with pytest.raises(ImportError):
                fit_glm_bayes(X, y, family="gaussian", priors=None)
        finally:
            gb.HAS_NUMPYRO = orig_np

    def test_custom_prior_passed(self):
        """Custom PriorSpec should be accepted."""
        import aurora.models.bayes.glm_bayes as gb

        orig_np = gb.HAS_NUMPYRO
        gb.HAS_NUMPYRO = False
        try:
            X = np.random.randn(10, 2)
            y = np.random.randn(10)
            spec = PriorSpec()
            spec.coef_prior = Normal(0, 1)
            with pytest.raises(ImportError):
                fit_glm_bayes(X, y, family="gaussian", priors=spec)
        finally:
            gb.HAS_NUMPYRO = orig_np


# =============================================================================
# BayesianGLMResult — Additional Coverage
# =============================================================================


class TestBayesianGLMResultAdditionalCoverage:
    """Tests for less-traversed code paths in BayesianGLMResult."""

    def test_waic_fallback_family(self):
        """WAIC with non-standard family should fall back to Gaussian log-lik."""
        result = _make_result(family="gamma", include_sigma=False, n_samples=100)
        w = result.waic()
        assert np.isfinite(w["waic"])

    def test_posterior_predictive_gaussian_no_scale(self):
        """Gaussian posterior_predictive without sigma should use scale=1.0."""
        result = _make_result(family="gaussian", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=10)
        assert y_pred.shape == (10, 20)
        assert np.isfinite(y_pred).all()

    def test_posterior_predictive_gamma_no_scale(self):
        """Gamma posterior_predictive without sigma should use scale=1.0."""
        result = _make_result(family="gamma", link="log", include_sigma=False)
        y_pred = result.posterior_predictive(n_samples=10)
        assert y_pred.shape == (10, 20)

    def test_posterior_predictive_negative_binomial_scalar_theta(self):
        """Negative binomial with scalar theta (not array) should work."""
        np.random.seed(42)
        n_chains, n_samples, n_obs, n_features = 2, 50, 15, 3
        samples = {
            "beta": np.random.randn(n_chains, n_samples, n_features),
            "theta": np.array(5.0),  # scalar, not array
        }
        result = BayesianGLMResult(
            posterior_samples_=samples,
            family="negative_binomial",
            link="log",
            n_samples_=n_samples,
            n_chains_=n_chains,
            n_obs_=n_obs,
            n_features_=n_features,
            X_=np.random.randn(n_obs, n_features),
            y_=np.abs(np.random.randn(n_obs)),
            backend_="numpyro",
        )
        y_pred = result.posterior_predictive(n_samples=5)
        assert y_pred.shape == (5, n_obs)

    def test_posterior_predictive_with_none_X(self):
        """Posterior predictive with X_new=None should use training data."""
        result = _make_result(family="gaussian", include_sigma=True, n_features=2, n_obs=10)
        y_pred = result.posterior_predictive(n_samples=5)
        assert y_pred.shape == (5, 10)

    def test_predict_mean_no_subsample(self):
        """predict(type='mean') without n_samples should use all posterior draws."""
        result = _make_result(n_chains=1, n_samples=50, n_features=2)
        pred = result.predict(np.random.randn(5, 2), type="mean")
        assert pred.shape == (5,)

    def test_credible_intervals_eti_no_scale(self):
        """ETI credible intervals without scale parameter should omit 'scale' key."""
        result = _make_result(include_sigma=False)
        ci = result.credible_intervals(level=0.90, method="eti")
        assert "scale" not in ci

    def test_credible_intervals_hdi_no_scale(self):
        """HDI credible intervals without scale parameter should omit 'scale' key."""
        result = _make_result(include_sigma=False)
        ci = result.credible_intervals(level=0.90, method="hdi")
        assert "scale" not in ci

    def test_compute_pointwise_ll_gaussian_no_scale(self):
        """Gaussian log-lik without scale should use scale=1.0."""
        result = _make_result(family="gaussian", include_sigma=False, n_samples=50)
        ll = result._compute_pointwise_ll()
        assert ll.shape == (100, 20)  # 2 chains * 50 samples
        assert np.isfinite(ll).all()

    def test_compute_pointwise_ll_fallback_family(self):
        """Non-standard family should fall back to basic Gaussian log-lik."""
        result = _make_result(family="beta", include_sigma=False, n_samples=30)
        ll = result._compute_pointwise_ll()
        assert ll.shape == (60, 20)
        assert np.isfinite(ll).all()

    def test_summary_no_rhat_ess(self):
        """Summary without R-hat/ESS should not include those keys in coef."""
        result = _make_result()
        s = result.summary()
        assert "r_hat" not in s["coef"]
        assert "ess" not in s["coef"]

    def test_from_numpyro_single_chain(self):
        """from_numpyro with n_chains=1 should set n_samples = total_samples."""
        np.random.seed(42)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        samples = {"beta": np.random.randn(200, 2)}
        result = BayesianGLMResult.from_numpyro(samples, X, y, "gaussian", "identity", n_chains=1)
        assert result.n_samples_ == 200
        assert result.n_chains_ == 1

    def test_scale_samples_1d_array(self):
        """scale_samples_ with 1D array should pass through directly."""
        np.random.seed(42)
        result = _make_result()
        result.posterior_samples_["sigma"] = np.abs(np.random.randn(100))
        samples = result.scale_samples_
        assert samples is not None
        assert samples.ndim == 1


class TestBayesianGAMResultAdditional:
    """Additional coverage for BayesianGAMResult."""

    def test_inherits_predict(self):
        """BayesianGAMResult should inherit predict from BayesianGLMResult."""
        np.random.seed(42)
        result = BayesianGAMResult(
            posterior_samples_={"beta": np.random.randn(2, 50, 3)},
            family="gaussian",
            link="identity",
            n_samples_=50,
            n_chains_=2,
            n_obs_=15,
            n_features_=3,
            X_=np.random.randn(15, 3),
            y_=np.random.randn(15),
            backend_="numpyro",
            smooth_names_=["s(x1)"],
            smooth_indices_={"s(x1)": (0, 3)},
        )
        pred = result.predict(np.random.randn(5, 3), type="mean")
        assert pred.shape == (5,)

    def test_inherits_waic(self):
        """BayesianGAMResult should inherit waic from BayesianGLMResult."""
        np.random.seed(42)
        result = BayesianGAMResult(
            posterior_samples_={
                "beta": np.random.randn(1, 100, 2),
                "sigma": np.abs(np.random.randn(1, 100)) + 0.5,
            },
            family="gaussian",
            link="identity",
            n_samples_=100,
            n_chains_=1,
            n_obs_=10,
            n_features_=2,
            X_=np.random.randn(10, 2),
            y_=np.random.randn(10),
            backend_="numpyro",
            smooth_names_=[],
            smooth_indices_={},
        )
        w = result.waic()
        assert np.isfinite(w["waic"])


class TestBayesPriorsExtraCoverage:
    """Cover a few more priors.py __repr__ paths."""

    def test_all_priors_repr(self):
        """repr() should work for all prior types."""
        for cls, kwargs in [
            (Normal, {"mu": 0, "sigma": 1}),
            (Cauchy, {"loc": 0, "scale": 1}),
            (HalfNormal, {"sigma": 1}),
            (HalfCauchy, {"scale": 1}),
            (Exponential, {"rate": 1}),
            (Gamma, {"alpha": 1, "beta": 1}),
            (InverseGamma, {"alpha": 1, "beta": 1}),
            (Uniform, {"lower": 0, "upper": 1}),
            (StudentT, {"df": 3, "loc": 0, "scale": 1}),
            (Laplace, {"loc": 0, "scale": 1}),
        ]:
            p = cls(**kwargs)
            r = repr(p)
            assert isinstance(r, str)
            assert len(r) > 0

    def test_all_priors_str(self):
        """str() should work for all prior types."""
        for cls, kwargs in [
            (Normal, {"mu": 0, "sigma": 1}),
            (Cauchy, {"loc": 0, "scale": 1}),
            (HalfNormal, {"sigma": 1}),
            (HalfCauchy, {"scale": 1}),
            (Exponential, {"rate": 1}),
            (Gamma, {"alpha": 1, "beta": 1}),
            (InverseGamma, {"alpha": 1, "beta": 1}),
            (Uniform, {"lower": 0, "upper": 1}),
            (StudentT, {"df": 3, "loc": 0, "scale": 1}),
            (Laplace, {"loc": 0, "scale": 1}),
        ]:
            p = cls(**kwargs)
            s = str(p)
            assert isinstance(s, str)

    def test_normal_equality(self):
        """Normal priors with same params should be equal."""
        p1 = Normal(0, 1)
        p2 = Normal(0, 1)
        assert p1 == p2

    def test_normal_inequality(self):
        """Normal priors with different params should not be equal."""
        p1 = Normal(0, 1)
        p2 = Normal(0, 2)
        assert p1 != p2








class TestBayesBackendModule:
    """Cover aurora.models.bayes.backends module."""

    def test_available_backends_empty(self):
        from aurora.models.bayes.backends import available_backends, get_default_backend, HAS_NUMPYRO, HAS_PYMC
        backends = available_backends()
        assert isinstance(backends, list)
        if not HAS_NUMPYRO and not HAS_PYMC:
            with pytest.raises(ImportError, match="No Bayesian backend"):
                get_default_backend()

    def test_available_backends_mocked(self):
        """Cover HAS_NUMPYRO=True and HAS_PYMC=True branches."""
        import aurora.models.bayes.backends as mod
        orig_np, orig_pmc = mod.HAS_NUMPYRO, mod.HAS_PYMC
        try:
            mod.HAS_NUMPYRO = True
            mod.HAS_PYMC = True
            assert "numpyro" in mod.available_backends()
            assert "pymc" in mod.available_backends()
            assert mod.get_default_backend() == "numpyro"
        finally:
            mod.HAS_NUMPYRO, mod.HAS_PYMC = orig_np, orig_pmc

    def test_module_dunder_all(self):
        from aurora.models.bayes.backends import __all__
        assert "HAS_NUMPYRO" in __all__
        assert "HAS_PYMC" in __all__
        assert "available_backends" in __all__
