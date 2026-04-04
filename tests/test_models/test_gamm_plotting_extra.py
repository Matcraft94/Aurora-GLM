"""Tests for GAMM plotting functions not covered by existing tests.

Covers: plot_diagnostics_panel, plot_smooth_effect, plot_all_smooth_effects,
plus additional edge cases for plot_diagnostics.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from aurora.models.gamm.plotting import (
    plot_all_smooth_effects,
    plot_diagnostics,
    plot_diagnostics_panel,
    plot_smooth_effect,
)


# ---------------------------------------------------------------------------
# Helpers – lightweight mock GAMMResult
# ---------------------------------------------------------------------------


def _make_mock_gamm_result(
    n_obs=50,
    n_coefs=5,
    n_basis=10,
    with_smooth=True,
    with_covariance=False,
):
    """Build a minimal mock GAMMResult for plotting tests."""

    class MockGAMMResult:
        pass

    np.random.seed(42)
    result = MockGAMMResult()

    result.residuals = np.random.randn(n_obs)
    result.fitted_values = np.random.randn(n_obs) + 2.0
    result.coefficients = np.random.randn(n_coefs + n_basis)
    result.n_obs = n_obs

    if with_smooth:
        result.smooth_terms = [
            {
                "name": "s(x)",
                "variable": "x",
                "coef_start": n_coefs,
                "coef_end": n_coefs + n_basis,
                "n_basis": n_basis,
                "basis": "cr",
                "x_min": 0.0,
                "x_max": 10.0,
            }
        ]
    else:
        result.smooth_terms = None

    if with_covariance:
        total = n_coefs + n_basis
        result.covariance = np.eye(total) * 0.1
    else:
        result.covariance = None

    # _Z_info set to empty so random-effects code paths are skipped
    result._Z_info = None
    result.variance_components = []
    result.random_effects = {}

    return result


# ---------------------------------------------------------------------------
# plot_diagnostics_panel
# ---------------------------------------------------------------------------


class TestPlotDiagnosticsPanel:
    def test_basic_panel(self):
        result = _make_mock_gamm_result()
        fig = plot_diagnostics_panel(result)
        assert fig is not None
        assert len(fig.axes) == 4
        plt.close("all")

    def test_panel_custom_figsize(self):
        result = _make_mock_gamm_result()
        fig = plot_diagnostics_panel(result, figsize=(10, 8))
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        plt.close("all")

    def test_panel_axes_labels(self):
        result = _make_mock_gamm_result()
        fig = plot_diagnostics_panel(result)
        ax_resid = fig.axes[0]
        assert "Residuals" in ax_resid.get_ylabel()
        ax_qq = fig.axes[1]
        assert "Q-Q" in ax_qq.get_title()
        plt.close("all")

    def test_panel_small_sample(self):
        """With <= 10 observations, smooth lines should be skipped."""
        result = _make_mock_gamm_result(n_obs=8)
        fig = plot_diagnostics_panel(result)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_smooth_effect
# ---------------------------------------------------------------------------


class TestPlotSmoothEffect:
    def test_basic_smooth(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data)
        assert fig is not None
        assert "x" in ax.get_xlabel()
        plt.close("all")

    def test_smooth_no_data_uses_range(self):
        """When data is None, should use x_min/x_max from smooth_info."""
        result = _make_mock_gamm_result()
        fig, ax = plot_smooth_effect(result, term_name="x")
        assert fig is not None
        plt.close("all")

    def test_smooth_with_confidence_level(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data, level=0.99)
        assert fig is not None
        assert "99%" in ax.get_legend_handles_labels()[1][0]
        plt.close("all")

    def test_smooth_with_covariance(self):
        result = _make_mock_gamm_result(with_covariance=True)
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data)
        assert fig is not None
        plt.close("all")

    def test_smooth_no_rug(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data, show_data=False)
        assert fig is not None
        plt.close("all")

    def test_smooth_with_residuals(self):
        result = _make_mock_gamm_result(n_obs=50, n_basis=10)
        data = pd.DataFrame({"x": np.random.rand(50) * 10})
        fig, ax = plot_smooth_effect(
            result, term_name="x", data=data, show_residuals=True
        )
        assert fig is not None
        plt.close("all")

    def test_smooth_on_existing_axes(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig_orig, ax_orig = plt.subplots()
        fig2, ax2 = plot_smooth_effect(result, term_name="x", data=data, ax=ax_orig)
        assert fig2 is fig_orig
        assert ax2 is ax_orig
        plt.close("all")

    def test_smooth_no_smooth_terms_raises(self):
        result = _make_mock_gamm_result(with_smooth=False)
        with pytest.raises(ValueError, match="smooth terms"):
            plot_smooth_effect(result, term_name="x")

    def test_smooth_unknown_term_raises(self):
        result = _make_mock_gamm_result()
        with pytest.raises(ValueError, match="not found"):
            plot_smooth_effect(result, term_name="nonexistent")

    def test_smooth_s_name_format(self):
        """Should match s(varname) format in smooth term name."""
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data)
        assert fig is not None
        plt.close("all")

    def test_smooth_custom_n_points(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig, ax = plot_smooth_effect(result, term_name="x", data=data, n_points=200)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_all_smooth_effects
# ---------------------------------------------------------------------------


class TestPlotAllSmoothEffects:
    def test_basic_all_smooths(self):
        result = _make_mock_gamm_result()
        data = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        fig = plot_all_smooth_effects(result, data=data)
        assert fig is not None
        plt.close("all")

    def test_no_smooth_terms_raises(self):
        result = _make_mock_gamm_result(with_smooth=False)
        with pytest.raises(ValueError, match="smooth terms"):
            plot_all_smooth_effects(result)

    def test_empty_smooth_terms_raises(self):
        result = _make_mock_gamm_result(with_smooth=True)
        result.smooth_terms = []
        with pytest.raises(ValueError, match="no smooth terms"):
            plot_all_smooth_effects(result)

    def test_custom_ncols(self):
        result = _make_mock_gamm_result()
        # Add a second smooth term
        result.smooth_terms.append(
            {
                "name": "s(z)",
                "variable": "z",
                "coef_start": 15,
                "coef_end": 25,
                "n_basis": 10,
                "basis": "cr",
                "x_min": -5.0,
                "x_max": 5.0,
            }
        )
        data = pd.DataFrame({"x": np.linspace(0, 10, 50), "z": np.linspace(-5, 5, 50)})
        fig = plot_all_smooth_effects(result, data=data, n_cols=1)
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_diagnostics additional edge cases
# ---------------------------------------------------------------------------


class TestPlotDiagnosticsExtra:
    def test_fitted_small_sample(self):
        """fitted plot type with <=10 points skips smooth line."""
        result = _make_mock_gamm_result(n_obs=5)
        fig, ax = plot_diagnostics(result, plot_type="fitted")
        assert fig is not None
        plt.close("all")

    def test_scale_location_small_sample(self):
        result = _make_mock_gamm_result(n_obs=5)
        fig, ax = plot_diagnostics(result, plot_type="scale-location")
        assert fig is not None
        plt.close("all")

    def test_residuals_small_sample(self):
        result = _make_mock_gamm_result(n_obs=5)
        fig, ax = plot_diagnostics(result, plot_type="residuals")
        assert fig is not None
        plt.close("all")
