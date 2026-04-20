"""Tests to improve coverage for aurora/models/gamm/fitting.py.

Targets uncovered lines:
- GAMMResult.predict(include_random=False) when _X_parametric exists
- GAMMResult.predict(include_random=False) when _X_parametric is None (error path)
- GAMMResult.summary() — output formatting, single and multivariate random effects
- Backend conversion branches in fit_gamm_gaussian
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm import RandomEffect, construct_Z_matrix
from aurora.models.gamm.fitting import (
    GAMMResult,
    fit_gamm_gaussian,
    predict_gamm,
)

try:
    import torch

    _torch_available = lambda: True
except ImportError:
    torch = None
    _torch_available = lambda: False

try:
    import jax  # noqa: F401

    _jax_available = lambda: True
except ImportError:
    _jax_available = lambda: False


# ---------------------------------------------------------------------------
# Reusable helpers (same pattern as test_fitting.py)
# ---------------------------------------------------------------------------


def _make_simple_result(**overrides) -> GAMMResult:
    """Build a minimal GAMMResult with sensible defaults for unit-testing."""
    defaults = dict(
        coefficients=np.array([2.0, 0.5]),
        beta_parametric=np.array([2.0, 0.5]),
        beta_smooth={},
        random_effects={"subject": {i: np.array([0.1 * i]) for i in range(3)}},
        variance_components=[np.array([[1.2]])],
        covariance_params=None,
        residual_variance=0.6,
        smoothing_parameters=None,
        edf_total=4.5,
        edf_parametric=2.0,
        edf_smooth={},
        fitted_values=np.random.randn(30),
        residuals=np.random.randn(30),
        log_likelihood=-45.3,
        aic=94.6,
        bic=102.1,
        converged=True,
        n_iterations=12,
        n_obs=30,
        n_groups=3,
        family="gaussian",
    )
    defaults.update(overrides)
    return GAMMResult(**defaults)


def _make_multivariate_result() -> GAMMResult:
    """Build a GAMMResult with 2x2 variance-covariance (random intercept+slope)."""
    vc = np.array([[1.0, 0.3], [0.3, 0.5]])
    return _make_simple_result(
        random_effects={"subject": {i: np.array([0.1 * i, 0.05 * i]) for i in range(3)}},
        variance_components=[vc],
    )


def _fit_simple_model(n_groups=5, n_per_group=10, seed=42, **kwargs):
    """Fit a simple random-intercept model and return the GAMMResult."""
    np.random.seed(seed)
    n = n_groups * n_per_group
    x = np.random.randn(n)
    X = np.column_stack([np.ones(n), x])
    groups = np.repeat(np.arange(n_groups), n_per_group)
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), groups] = 1
    beta_true = np.array([2.0, 0.5])
    b = np.random.randn(n_groups) * 0.8
    y = X @ beta_true + Z @ b + np.random.randn(n) * 0.4

    Z_info = [
        {
            "grouping": "subject",
            "n_effects": 1,
            "n_groups": n_groups,
            "groups": np.arange(n_groups),
            "start_col": 0,
            "end_col": n_groups,
        }
    ]
    return fit_gamm_gaussian(
        X_parametric=X, X_smooth=None, Z=Z, Z_info=Z_info, y=y, covariance="identity", **kwargs
    )


# ---------------------------------------------------------------------------
# GAMMResult.predict — include_random=False with _X_parametric stored
# ---------------------------------------------------------------------------


class TestGAMMResultPredictFixedOnly:
    """Cover lines 514-516: predict(include_random=False) via stored _X_parametric."""

    def test_predict_fixed_only_returns_correct_shape(self):
        result = _fit_simple_model()
        pred = result.predict(include_random=False)
        assert pred.shape == (result.n_obs,)

    def test_predict_fixed_only_equals_x_beta(self):
        result = _fit_simple_model()
        pred = result.predict(include_random=False)
        expected = result._X_parametric @ result.beta_parametric
        np.testing.assert_allclose(pred, expected)

    def test_predict_fixed_only_differs_from_fitted(self):
        """Population-level predictions should differ from fitted (which include RE)."""
        result = _fit_simple_model()
        pred_fixed = result.predict(include_random=False)
        pred_full = result.predict(include_random=True)
        # They should NOT be identical when random effects are nonzero
        assert not np.allclose(pred_fixed, pred_full)

    def test_predict_default_is_include_random_true(self):
        result = _fit_simple_model()
        pred_default = result.predict()
        pred_explicit = result.predict(include_random=True)
        np.testing.assert_array_equal(pred_default, pred_explicit)


# ---------------------------------------------------------------------------
# GAMMResult.predict — include_random=False when _X_parametric is None
# ---------------------------------------------------------------------------


class TestGAMMResultPredictNoXParametric:
    """Cover lines 518-519: error when _X_parametric is None."""

    def test_raises_when_x_parametric_not_stored(self):
        result = _make_simple_result(_X_parametric=None)
        with pytest.raises(ValueError, match="_X_parametric not stored"):
            result.predict(include_random=False)

    def test_include_random_true_still_works_without_x_parametric(self):
        result = _make_simple_result(_X_parametric=None)
        # Should return fitted_values, not raise
        pred = result.predict(include_random=True)
        assert pred.shape == (30,)


# ---------------------------------------------------------------------------
# GAMMResult.summary — single variance component (1x1)
# ---------------------------------------------------------------------------


class TestGAMMResultSummarySingleComponent:
    """Cover lines 536-601: summary() output for single variance component model."""

    def test_summary_returns_string(self):
        result = _fit_simple_model()
        s = result.summary()
        assert isinstance(s, str)

    def test_summary_contains_header(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "Generalized Additive Mixed Model" in s
        assert "GAMM" in s

    def test_summary_contains_family(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "gaussian" in s

    def test_summary_contains_obs_and_groups(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "Number of observations: 50" in s
        assert "Number of groups: 5" in s

    def test_summary_contains_fixed_effects(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "Fixed Effects" in s
        assert "\u03b20" in s  # β0
        assert "\u03b21" in s  # β1

    def test_summary_contains_random_effects_single_variance(self):
        """Single variance component should show Variance and Std.Dev."""
        result = _fit_simple_model()
        s = result.summary()
        assert "Random Effects" in s
        assert "Group: subject" in s
        assert "Variance:" in s
        assert "Std.Dev." in s

    def test_summary_contains_residual_sd(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "Residual Standard Deviation:" in s

    def test_summary_contains_model_fit_stats(self):
        result = _fit_simple_model()
        s = result.summary()
        assert "Model Fit Statistics" in s
        assert "Log-likelihood:" in s
        assert "AIC:" in s
        assert "BIC:" in s
        assert "Effective df" in s
        assert "Converged:" in s
        assert "Iterations:" in s

    def test_summary_captured_via_capsys(self, capsys):
        """Ensure print(result.summary()) works and capsys captures it."""
        result = _fit_simple_model()
        print(result.summary())
        captured = capsys.readouterr()
        assert "Generalized Additive Mixed Model" in captured.out

    def test_summary_separators(self):
        """Summary should contain = separators and - dividers."""
        result = _fit_simple_model()
        s = result.summary()
        assert "=" * 75 in s
        assert "-" * 75 in s


# ---------------------------------------------------------------------------
# GAMMResult.summary — multivariate random effects (2x2 VC)
# ---------------------------------------------------------------------------


class TestGAMMResultSummaryMultivariate:
    """Cover lines 566-584: multivariate variance-covariance and correlation display."""

    def test_summary_shows_variance_covariance_matrix(self):
        result = _make_multivariate_result()
        s = result.summary()
        assert "Variance-Covariance Matrix:" in s

    def test_summary_shows_standard_deviations(self):
        result = _make_multivariate_result()
        s = result.summary()
        assert "Standard Deviations:" in s
        assert "Component 0:" in s
        assert "Component 1:" in s

    def test_summary_shows_correlation_matrix(self):
        """Multivariate (dim > 1) should display correlation matrix."""
        result = _make_multivariate_result()
        s = result.summary()
        assert "Correlation Matrix:" in s

    def test_summary_multivariate_via_fitted_model(self):
        """Fit a random intercept+slope model and check summary output."""
        np.random.seed(42)
        n_groups = 5
        n_per_group = 10
        n = n_groups * n_per_group
        time = np.tile(np.arange(n_per_group), n_groups)
        groups = np.repeat(np.arange(n_groups), n_per_group)

        X = np.column_stack([np.ones(n), time])
        groups_data = {"subject": groups}
        re = RandomEffect(grouping="subject", variables=(1,))
        Z, Z_info = construct_Z_matrix(X, [re], groups_data)

        psi_true = np.array([[1.0, 0.3], [0.3, 0.5]])
        L = np.linalg.cholesky(psi_true)
        b_raw = np.random.randn(n_groups, 2)
        b = (L @ b_raw.T).T.flatten()
        y = X @ [2.0, 0.5] + Z @ b + np.random.randn(n) * 0.3

        result = fit_gamm_gaussian(
            X_parametric=X, X_smooth=None, Z=Z, Z_info=Z_info, y=y, covariance="unstructured"
        )
        s = result.summary()
        assert "Variance-Covariance Matrix:" in s
        assert "Correlation Matrix:" in s
        assert "Component 0:" in s
        assert "Component 1:" in s


# ---------------------------------------------------------------------------
# GAMMResult.summary — single component via dataclass (1x1 branch)
# ---------------------------------------------------------------------------


class TestGAMMResultSummarySingleComponentViaDataclass:
    """Cover the 1x1 branch (lines 562-565) via a manually-constructed result."""

    def test_summary_single_variance_shows_variance_and_stddev(self):
        vc = np.array([[2.5]])
        result = _make_simple_result(variance_components=[vc])
        s = result.summary()
        assert "Variance: 2.5000" in s
        assert "Std.Dev.: 1.5811" in s


# ---------------------------------------------------------------------------
# fit_gamm_gaussian — backend conversion branches
# ---------------------------------------------------------------------------


class TestFitGAMMGaussianBackend:
    """Cover lines 831-897: backend conversion paths in fit_gamm_gaussian."""

    def test_numpy_backend_explicit(self):
        result = _fit_simple_model(backend="numpy")
        assert result.converged

    def test_invalid_backend_falls_back_to_numpy(self):
        """An unrecognized backend name should still work (falls through to numpy)."""
        result = _fit_simple_model(backend="unknown_backend")
        assert result.converged

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not installed"
    )
    def test_torch_backend_converts_and_fits(self):
        import torch

        result = _fit_simple_model(backend="torch")
        assert result.converged
        # Internal arrays should still be numpy (converted back)
        assert isinstance(result.beta_parametric, np.ndarray)

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not installed"
    )
    def test_torch_backend_with_device_cpu(self):
        result = _fit_simple_model(backend="torch", device="cpu")
        assert result.converged

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not installed"
    )
    def test_pytorch_alias(self):
        """'pytorch' should be treated the same as 'torch'."""
        result = _fit_simple_model(backend="pytorch")
        assert result.converged

    @pytest.mark.skipif(
        not _jax_available(), reason="JAX not installed"
    )
    def test_jax_backend_converts_and_fits(self):
        result = _fit_simple_model(backend="jax")
        assert result.converged
        assert isinstance(result.beta_parametric, np.ndarray)



# ---------------------------------------------------------------------------
# predict_gamm — additional edge cases
# ---------------------------------------------------------------------------


class TestPredictGammEdgeCases:
    """Additional coverage for predict_gamm with various option combos."""

    def test_predict_gamm_no_smooth_no_random(self):
        """predict_gamm with no smooth terms and no random effects."""
        result = _fit_simple_model()
        n_new = 10
        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
        pred = predict_gamm(result, X_new, include_random=False)
        expected = X_new @ result.beta_parametric
        np.testing.assert_allclose(pred, expected)

    def test_predict_gamm_smooth_terms_included(self):
        """predict_gamm should include smooth term contributions."""
        np.random.seed(42)
        n_groups = 5
        n_per_group = 10
        n = n_groups * n_per_group

        x_smooth = np.linspace(0, 2 * np.pi, n)
        X_para = np.column_stack([np.ones(n), np.random.randn(n)])
        X_smooth = {"s1": np.column_stack([x_smooth, x_smooth**2, x_smooth**3])}
        S_smooth = {"s1": np.eye(3)}
        lambda_smooth = {"s1": 0.1}

        groups = np.repeat(np.arange(n_groups), n_per_group)
        Z = np.zeros((n, n_groups))
        Z[np.arange(n), groups] = 1
        y = 2.0 + np.sin(x_smooth) + Z @ (np.random.randn(n_groups) * 0.3) + np.random.randn(n) * 0.3

        Z_info = [
            {
                "grouping": "subject",
                "n_effects": 1,
                "n_groups": n_groups,
                "groups": np.arange(n_groups),
                "start_col": 0,
                "end_col": n_groups,
            }
        ]

        result = fit_gamm_gaussian(
            X_parametric=X_para,
            X_smooth=X_smooth,
            Z=Z,
            Z_info=Z_info,
            y=y,
            S_smooth=S_smooth,
            lambda_smooth=lambda_smooth,
            covariance="identity",
        )

        n_new = 8
        x_new = np.linspace(0, 1, n_new)
        X_para_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
        X_smooth_new = {"s1": np.column_stack([x_new, x_new**2, x_new**3])}

        pred = predict_gamm(result, X_para_new, X_smooth_new=X_smooth_new, include_random=False)

        # Verify manually
        expected = X_para_new @ result.beta_parametric
        expected += X_smooth_new["s1"] @ result.beta_smooth["s1"]
        np.testing.assert_allclose(pred, expected, rtol=1e-10)

    def test_predict_gamm_ignore_unknown_smooth_terms(self):
        """predict_gamm should silently skip unknown smooth term names."""
        result = _fit_simple_model()
        n_new = 5
        X_new = np.column_stack([np.ones(n_new), np.random.randn(n_new)])
        X_smooth_new = {"nonexistent": np.random.randn(n_new, 3)}
        # Should not raise, just skip
        pred = predict_gamm(result, X_new, X_smooth_new=X_smooth_new, include_random=False)
        expected = X_new @ result.beta_parametric
        np.testing.assert_allclose(pred, expected)


# ---------------------------------------------------------------------------
# GAMMResult.predict — include_random=True returns fitted_values
# ---------------------------------------------------------------------------


class TestGAMMResultPredictIncludeRandom:
    """Ensure predict(include_random=True) returns stored fitted_values."""

    def test_returns_fitted_values_array(self):
        result = _fit_simple_model()
        pred = result.predict(include_random=True)
        np.testing.assert_array_equal(pred, result.fitted_values)

    def test_fitted_values_shape_matches_n_obs(self):
        result = _fit_simple_model()
        assert result.predict().shape == (result.n_obs,)
