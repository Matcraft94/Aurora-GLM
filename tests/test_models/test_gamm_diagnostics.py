"""Tests for aurora.models.gamm.diagnostics module."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from aurora.models.gamm.diagnostics import (
    compute_r2_conditional_marginal,
    interpret_variance_components,
    plot_diagnostics,
    plot_random_effects,
)


class MockGAMMResult:
    """Mock GAMM result for testing diagnostics."""

    def __init__(
        self,
        variance_components,
        random_effects,
        fitted_values=None,
        residuals=None,
        residual_variance=None,
        X_parametric=None,
        beta_parametric=None,
    ):
        n = 50
        self.variance_components = variance_components
        self.random_effects = random_effects
        np.random.seed(42)
        self.fitted_values = fitted_values if fitted_values is not None else np.random.randn(n)
        self.residuals = residuals if residuals is not None else np.random.randn(n) * 0.5
        self.residual_variance = (
            residual_variance if residual_variance is not None else np.var(self.residuals)
        )
        self._X_parametric = X_parametric
        self.beta_parametric = beta_parametric


@pytest.fixture
def intercept_only_result():
    """Mock result with random intercept only."""
    np.random.seed(42)
    vc = [np.array([[2.5]])]  # 1x1 variance component
    re_inner = {f"subj_{i}": np.array([np.random.randn() * 1.5]) for i in range(10)}
    re = {"subject": re_inner}
    return MockGAMMResult(vc, re)


@pytest.fixture
def intercept_slope_result():
    """Mock result with random intercept + slope."""
    np.random.seed(42)
    vc = [np.array([[3.0, 0.5], [0.5, 1.0]])]
    re_inner = {f"subj_{i}": np.random.randn(2) for i in range(10)}
    re = {"subject": re_inner}
    return MockGAMMResult(vc, re)


# ---------------------------------------------------------------------------
# interpret_variance_components
# ---------------------------------------------------------------------------


class TestInterpretVarianceComponents:
    """Tests for interpret_variance_components."""

    def test_intercept_only_returns_string(self, intercept_only_result):
        """Output is a non-empty string."""
        vc = intercept_only_result.variance_components
        text = interpret_variance_components(vc)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_intercept_only_contains_variance(self, intercept_only_result):
        """1x1 case shows the variance value."""
        vc = intercept_only_result.variance_components
        text = interpret_variance_components(vc)
        assert "2.5000" in text

    def test_intercept_only_contains_sd(self, intercept_only_result):
        """1x1 case shows the standard deviation."""
        vc = intercept_only_result.variance_components
        text = interpret_variance_components(vc)
        # SD = sqrt(2.5) ~ 1.5811
        assert "1.58" in text

    def test_intercept_only_interpretation(self, intercept_only_result):
        """1x1 case includes the +-1.96*SD interpretation line."""
        vc = intercept_only_result.variance_components
        text = interpret_variance_components(vc)
        sd = np.sqrt(2.5)
        expected_bound = f"±{1.96 * sd:.2f}"
        assert expected_bound in text
        assert "Intercepto" in text

    def test_intercept_only_type_label(self, intercept_only_result):
        """1x1 case identifies as random-intercept-only."""
        vc = intercept_only_result.variance_components
        text = interpret_variance_components(vc)
        assert "Intercepto aleatorio" in text

    def test_intercept_slope_type_label(self, intercept_slope_result):
        """2x2 case identifies as intercept + slope."""
        vc = intercept_slope_result.variance_components
        text = interpret_variance_components(vc)
        assert "Intercepto y pendiente" in text

    def test_intercept_slope_contains_variance_and_sd(self, intercept_slope_result):
        """2x2 case shows variance and SD for both components."""
        vc = intercept_slope_result.variance_components
        text = interpret_variance_components(vc)
        assert "3.0000" in text  # var intercept
        assert "1.0000" in text  # var slope
        assert "0.5000" in text  # covariance
        assert "Correlación" in text

    def test_intercept_slope_correlation_value(self, intercept_slope_result):
        """2x2 case reports the correlation numerically."""
        vc = intercept_slope_result.variance_components
        text = interpret_variance_components(vc)
        # corr = 0.5 / (sqrt(3) * sqrt(1)) ~ 0.2887
        corr = 0.5 / (np.sqrt(3.0) * np.sqrt(1.0))
        assert f"{corr:.4f}" in text

    def test_weak_correlation(self):
        """Correlation |r| < 0.3 is labeled as weak."""
        vc = [np.array([[2.0, 0.1], [0.1, 1.0]])]  # correlation ~0.07
        text = interpret_variance_components(vc, group_names=["subject"])
        assert "débil" in text.lower() or "weak" in text.lower()

    def test_positive_correlation(self):
        """Correlation r > 0.3 is labeled as positive moderate/strong."""
        vc = [np.array([[2.0, 0.8], [0.8, 1.0]])]  # correlation ~0.57
        text = interpret_variance_components(vc, group_names=["subject"])
        assert "positiva" in text.lower()

    def test_negative_correlation(self):
        """Correlation r < -0.3 is labeled as negative moderate/strong."""
        vc = [np.array([[2.0, -0.8], [-0.8, 1.0]])]  # correlation ~-0.57
        text = interpret_variance_components(vc, group_names=["subject"])
        assert "negativa" in text.lower()

    def test_negative_correlation_compensatory(self):
        """Negative correlation mentions compensatory effect."""
        vc = [np.array([[2.0, -0.8], [-0.8, 1.0]])]
        text = interpret_variance_components(vc, group_names=["clinic"])
        assert "compensatorio" in text.lower()

    def test_custom_group_names(self):
        """Custom group names appear in the output."""
        vc = [np.array([[1.5]])]
        text = interpret_variance_components(vc, group_names=["patient"])
        assert "patient" in text
        assert "Group_1" not in text

    def test_default_group_names(self):
        """When group_names is None, default Group_1 etc. are used."""
        vc = [np.array([[1.5]])]
        text = interpret_variance_components(vc)
        assert "Group_1" in text

    def test_multiple_components_3x3(self):
        """3x3 matrix is handled with full matrix display."""
        vc = [np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]])]
        text = interpret_variance_components(vc)
        assert "3 componentes" in text
        assert "Matriz de varianza-covarianza" in text

    def test_multiple_components_shows_sds(self):
        """NxN (N>2) case shows standard deviations for each component."""
        vc = [np.array([[4.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 16.0]])]
        text = interpret_variance_components(vc)
        # SDs = 2.0, 3.0, 4.0
        assert "2.0000" in text
        assert "3.0000" in text
        assert "4.0000" in text

    def test_4x4_matrix(self):
        """4x4 matrix is also handled by the NxN path."""
        vc = [np.eye(4) * 2.0]
        text = interpret_variance_components(vc)
        assert "4 componentes" in text

    def test_multiple_groups(self):
        """Multiple variance components with different group names."""
        vc = [np.array([[1.5]]), np.array([[2.0]])]
        text = interpret_variance_components(vc, group_names=["subject", "item"])
        assert "subject" in text
        assert "item" in text

    def test_multiple_groups_default_names(self):
        """Multiple groups get Group_1, Group_2 by default."""
        vc = [np.array([[1.5]]), np.array([[2.0]])]
        text = interpret_variance_components(vc)
        assert "Group_1" in text
        assert "Group_2" in text

    def test_output_has_separator_lines(self):
        """Output includes separator bars."""
        vc = [np.array([[1.0]])]
        text = interpret_variance_components(vc)
        assert "=" * 75 in text
        assert "-" * 75 in text


# ---------------------------------------------------------------------------
# compute_r2_conditional_marginal
# ---------------------------------------------------------------------------


class TestComputeR2:
    """Tests for compute_r2_conditional_marginal."""

    def test_with_parametric_returns_tuple(self):
        """Returns a 2-element tuple of floats."""
        np.random.seed(42)
        n = 50
        X_param = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_param = np.array([2.0, 3.0])
        fitted = X_param @ beta_param
        residuals = np.random.randn(n) * 0.5
        result = MockGAMMResult(
            variance_components=[np.array([[1.0]])],
            random_effects={"group": {f"g{i}": np.array([0.1]) for i in range(10)}},
            fitted_values=fitted,
            residuals=residuals,
            X_parametric=X_param,
            beta_parametric=beta_param,
        )
        output = compute_r2_conditional_marginal(result)
        assert isinstance(output, tuple)
        assert len(output) == 2
        r2_m, r2_c = output
        assert isinstance(r2_m, float)
        assert isinstance(r2_c, float)

    def test_with_parametric_values_in_range(self):
        """Both R2 values are in [0, 1] and conditional >= marginal."""
        np.random.seed(42)
        n = 50
        X_param = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_param = np.array([2.0, 3.0])
        fitted = X_param @ beta_param
        residuals = np.random.randn(n) * 0.5
        result = MockGAMMResult(
            variance_components=[np.array([[1.0]])],
            random_effects={"group": {f"g{i}": np.array([0.1]) for i in range(10)}},
            fitted_values=fitted,
            residuals=residuals,
            X_parametric=X_param,
            beta_parametric=beta_param,
        )
        r2_m, r2_c = compute_r2_conditional_marginal(result)
        assert 0 <= r2_m <= 1
        assert 0 <= r2_c <= 1
        assert r2_c >= r2_m

    def test_without_parametric(self):
        """When _X_parametric is None, uses the approximate path."""
        np.random.seed(42)
        result = MockGAMMResult(
            variance_components=[np.array([[1.0]])],
            random_effects={"group": {f"g{i}": np.array([0.1]) for i in range(10)}},
        )
        result._X_parametric = None
        r2_m, r2_c = compute_r2_conditional_marginal(result)
        assert isinstance(r2_m, float)
        assert isinstance(r2_c, float)

    def test_without_parametric_conditional_gte_marginal(self):
        """Even on the approximate path, conditional >= marginal."""
        np.random.seed(42)
        result = MockGAMMResult(
            variance_components=[np.array([[1.0]])],
            random_effects={"group": {f"g{i}": np.array([0.1]) for i in range(10)}},
        )
        result._X_parametric = None
        r2_m, r2_c = compute_r2_conditional_marginal(result)
        assert r2_c >= r2_m

    def test_multiple_variance_components(self):
        """Works with multiple variance components."""
        np.random.seed(42)
        n = 50
        X_param = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_param = np.array([2.0, 3.0])
        result = MockGAMMResult(
            variance_components=[np.array([[1.0]]), np.array([[0.5]])],
            random_effects={
                "subject": {f"s{i}": np.array([0.1]) for i in range(10)},
                "item": {f"it{i}": np.array([0.05]) for i in range(5)},
            },
            X_parametric=X_param,
            beta_parametric=beta_param,
        )
        r2_m, r2_c = compute_r2_conditional_marginal(result)
        assert isinstance(r2_m, float)
        assert isinstance(r2_c, float)

    def test_zero_random_variance(self):
        """When random variance is zero, marginal == conditional."""
        np.random.seed(42)
        n = 50
        X_param = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_param = np.array([2.0, 3.0])
        result = MockGAMMResult(
            variance_components=[np.array([[0.0]])],
            random_effects={"group": {f"g{i}": np.array([0.0]) for i in range(10)}},
            X_parametric=X_param,
            beta_parametric=beta_param,
        )
        r2_m, r2_c = compute_r2_conditional_marginal(result)
        assert r2_c == pytest.approx(r2_m)


# ---------------------------------------------------------------------------
# plot_diagnostics
# ---------------------------------------------------------------------------


class TestPlotDiagnostics:
    """Tests for plot_diagnostics."""

    def test_returns_figure_and_axes(self, intercept_only_result):
        """Returns (fig, axes) with axes shape (2, 2)."""
        fig, axes = plot_diagnostics(intercept_only_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 2)
        for row in range(2):
            for col in range(2):
                assert isinstance(axes[row, col], matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_figsize(self, intercept_only_result):
        """Custom figsize is reflected in the figure."""
        fig, axes = plot_diagnostics(intercept_only_result, figsize=(8, 6))
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 6
        plt.close("all")

    def test_default_figsize(self, intercept_only_result):
        """Default figsize is (12, 10)."""
        fig, axes = plot_diagnostics(intercept_only_result)
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10
        plt.close("all")

    def test_panels_have_titles(self, intercept_only_result):
        """All four panels have non-empty titles."""
        fig, axes = plot_diagnostics(intercept_only_result)
        for row in range(2):
            for col in range(2):
                title = axes[row, col].get_title()
                assert len(title) > 0
        plt.close("all")

    def test_residuals_vs_fitted_panel(self, intercept_only_result):
        """Top-left panel is residuals vs fitted."""
        fig, axes = plot_diagnostics(intercept_only_result)
        ax = axes[0, 0]
        assert "Residuos" in ax.get_title()
        assert ax.get_xlabel() != "" or ax.get_ylabel() != ""
        plt.close("all")

    def test_qq_panel(self, intercept_only_result):
        """Top-right panel is the Q-Q plot."""
        fig, axes = plot_diagnostics(intercept_only_result)
        ax = axes[0, 1]
        assert "Q-Q" in ax.get_title()
        plt.close("all")

    def test_scale_location_panel(self, intercept_only_result):
        """Bottom-left panel is the scale-location plot."""
        fig, axes = plot_diagnostics(intercept_only_result)
        ax = axes[1, 0]
        assert "Scale-Location" in ax.get_title()
        plt.close("all")

    def test_histogram_panel(self, intercept_only_result):
        """Bottom-right panel is the histogram."""
        fig, axes = plot_diagnostics(intercept_only_result)
        ax = axes[1, 1]
        assert "Residuos" in ax.get_title() or "Distribución" in ax.get_title()
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_random_effects
# ---------------------------------------------------------------------------


class TestPlotRandomEffects:
    """Tests for plot_random_effects."""

    def test_intercept_only_returns_single_axes(self, intercept_only_result):
        """1 component returns (fig, ax) where ax is a single Axes."""
        fig, ax = plot_random_effects(intercept_only_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_intercept_only_has_zero_line(self, intercept_only_result):
        """Caterpillar plot has a horizontal zero line."""
        fig, ax = plot_random_effects(intercept_only_result)
        # Check that there is a horizontal line at y=0
        has_hline = any(
            getattr(line, "get_ydata", lambda: [None])()[0] == 0
            if hasattr(line, "get_ydata")
            else False
            for line in ax.get_lines()
        )
        assert has_hline
        plt.close("all")

    def test_intercept_slope_returns_three_axes(self, intercept_slope_result):
        """2 components returns (fig, axes) with 3 subplots."""
        fig, axes = plot_random_effects(intercept_slope_result)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes) == 3
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_intercept_slope_correlation_annotated(self, intercept_slope_result):
        """2-component scatter panel has a correlation annotation."""
        fig, axes = plot_random_effects(intercept_slope_result)
        scatter_ax = axes[2]
        texts = scatter_ax.texts
        assert any("r =" in t.get_text() for t in texts)
        plt.close("all")

    def test_custom_group_name(self, intercept_only_result):
        """Explicit group_name is used for labels."""
        result = intercept_only_result
        fig, ax = plot_random_effects(result, group_name="subject")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_default_group_name_uses_first_key(self, intercept_only_result):
        """When group_name is None, the first key in random_effects is used."""
        result = intercept_only_result
        fig, ax = plot_random_effects(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Title should contain the first key "subject"
        assert "subject" in ax.get_title()
        plt.close("all")

    def test_three_components_raises(self):
        """3 components raises NotImplementedError."""
        np.random.seed(42)
        vc = [np.eye(3)]
        re_inner = {f"g{i}": np.random.randn(3) for i in range(5)}
        re = {"group": re_inner}
        result = MockGAMMResult(vc, re)
        with pytest.raises(NotImplementedError, match="3"):
            plot_random_effects(result)
        plt.close("all")

    def test_four_components_raises(self):
        """4 components also raises NotImplementedError."""
        np.random.seed(42)
        vc = [np.eye(4)]
        re_inner = {f"g{i}": np.random.randn(4) for i in range(5)}
        re = {"group": re_inner}
        result = MockGAMMResult(vc, re)
        with pytest.raises(NotImplementedError, match="4"):
            plot_random_effects(result)
        plt.close("all")

    def test_custom_figsize_slope(self, intercept_slope_result):
        """Custom figsize is used for the 3-panel figure."""
        fig, axes = plot_random_effects(intercept_slope_result, figsize=(12, 4))
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 4
        plt.close("all")
