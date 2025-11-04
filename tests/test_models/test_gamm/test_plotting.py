"""Tests for GAMM visualization functions."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from aurora.models.gamm import (
    fit_gamm,
    plot_caterpillar,
    plot_diagnostics,
    plot_random_effects_density,
    plot_random_effects_qq,
    plot_random_effects_summary,
)


@pytest.fixture
def simple_gamm_result():
    """Create a simple GAMM result for testing plots."""
    np.random.seed(42)

    n_groups, n_per_group = 8, 20
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.zeros(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    # Generate with random intercepts
    b_true = np.random.randn(n_groups) * 0.5
    data['y'] = (
        2.0 + 0.5 * data['x1'] + b_true[data['subject']]
        + np.random.randn(n) * 0.3
    )

    result = fit_gamm(
        formula="y ~ x1 + (1 | subject)",
        data=data,
        covariance='identity',
    )

    return result


@pytest.fixture
def gamm_with_slopes():
    """Create GAMM with random slopes for testing."""
    np.random.seed(123)

    n_groups, n_per_group = 6, 15
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.zeros(n),
        'time': np.tile(np.arange(n_per_group), n_groups),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    b0 = np.random.randn(n_groups) * 0.3
    b1 = np.random.randn(n_groups) * 0.1

    for i in range(n):
        subj = data.loc[i, 'subject']
        data.loc[i, 'y'] = (
            2.0 + b0[subj] + (0.5 + b1[subj]) * data.loc[i, 'time']
            + np.random.randn() * 0.2
        )

    result = fit_gamm(
        formula="y ~ time + (1 + time | subject)",
        data=data,
        covariance='unstructured',
    )

    return result


@pytest.fixture
def gamm_crossed_effects():
    """Create GAMM with crossed random effects."""
    np.random.seed(42)

    n_subjects = 5
    n_items = 4
    n = n_subjects * n_items

    data = pd.DataFrame({
        'y': np.zeros(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_subjects), n_items),
        'item': np.tile(np.arange(n_items), n_subjects),
    })

    b_subj = np.random.randn(n_subjects) * 0.4
    b_item = np.random.randn(n_items) * 0.3

    for i in range(n):
        subj = data.loc[i, 'subject']
        item = data.loc[i, 'item']
        data.loc[i, 'y'] = (
            2.0 + 0.5 * data.loc[i, 'x1']
            + b_subj[subj] + b_item[item]
            + np.random.randn() * 0.2
        )

    result = fit_gamm(
        formula="y ~ x1 + (1 | subject) + (1 | item)",
        data=data,
        covariance='identity',
    )

    return result


def test_plot_caterpillar_basic(simple_gamm_result):
    """Test basic caterpillar plot creation."""
    fig, ax = plot_caterpillar(simple_gamm_result)

    assert fig is not None
    assert ax is not None
    assert len(ax.lines) > 0  # Should have horizontal lines for CIs

    plt.close(fig)


def test_plot_caterpillar_sorted(simple_gamm_result):
    """Test caterpillar plot with sorting."""
    fig, ax = plot_caterpillar(simple_gamm_result, sort=True)

    assert fig is not None
    plt.close(fig)


def test_plot_caterpillar_unsorted(simple_gamm_result):
    """Test caterpillar plot without sorting."""
    fig, ax = plot_caterpillar(simple_gamm_result, sort=False)

    assert fig is not None
    plt.close(fig)


def test_plot_caterpillar_custom_confidence(simple_gamm_result):
    """Test caterpillar plot with custom confidence level."""
    fig, ax = plot_caterpillar(simple_gamm_result, confidence=0.90)

    assert fig is not None
    assert '90%' in ax.get_title()
    plt.close(fig)


def test_plot_caterpillar_with_slopes(gamm_with_slopes):
    """Test caterpillar plot for random slopes."""
    # Plot intercepts
    fig1, ax1 = plot_caterpillar(gamm_with_slopes, effect_index=0)
    assert fig1 is not None
    assert 'Intercept' in ax1.get_title()
    plt.close(fig1)

    # Plot slopes
    fig2, ax2 = plot_caterpillar(gamm_with_slopes, effect_index=1)
    assert fig2 is not None
    assert 'Slope' in ax2.get_title()
    plt.close(fig2)


def test_plot_caterpillar_crossed_effects(gamm_crossed_effects):
    """Test caterpillar plot with crossed random effects."""
    # Must specify grouping when multiple groupings exist
    fig1, ax1 = plot_caterpillar(gamm_crossed_effects, grouping='subject')
    assert fig1 is not None
    assert 'subject' in ax1.get_title()
    plt.close(fig1)

    fig2, ax2 = plot_caterpillar(gamm_crossed_effects, grouping='item')
    assert fig2 is not None
    assert 'item' in ax2.get_title()
    plt.close(fig2)


def test_plot_caterpillar_requires_grouping_for_multiple(gamm_crossed_effects):
    """Test that grouping is required when multiple exist."""
    with pytest.raises(ValueError, match="grouping variables"):
        plot_caterpillar(gamm_crossed_effects)


def test_plot_caterpillar_invalid_effect_index(simple_gamm_result):
    """Test error for invalid effect index."""
    with pytest.raises(ValueError, match="effect_index"):
        plot_caterpillar(simple_gamm_result, effect_index=5)


def test_plot_caterpillar_custom_figsize(simple_gamm_result):
    """Test caterpillar plot with custom figure size."""
    fig, ax = plot_caterpillar(simple_gamm_result, figsize=(10, 8))

    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 8
    plt.close(fig)


def test_plot_random_effects_qq_basic(simple_gamm_result):
    """Test basic Q-Q plot creation."""
    fig, ax = plot_random_effects_qq(simple_gamm_result)

    assert fig is not None
    assert ax is not None
    assert 'Q-Q' in ax.get_title()

    plt.close(fig)


def test_plot_random_effects_qq_with_slopes(gamm_with_slopes):
    """Test Q-Q plot for random slopes."""
    fig1, ax1 = plot_random_effects_qq(gamm_with_slopes, effect_index=0)
    assert fig1 is not None
    plt.close(fig1)

    fig2, ax2 = plot_random_effects_qq(gamm_with_slopes, effect_index=1)
    assert fig2 is not None
    plt.close(fig2)


def test_plot_random_effects_qq_crossed(gamm_crossed_effects):
    """Test Q-Q plot with crossed effects."""
    fig, ax = plot_random_effects_qq(gamm_crossed_effects, grouping='subject')

    assert fig is not None
    plt.close(fig)


def test_plot_random_effects_density_basic(simple_gamm_result):
    """Test basic density plot creation."""
    fig, ax = plot_random_effects_density(simple_gamm_result)

    assert fig is not None
    assert ax is not None
    # Should have histogram patches
    assert len(ax.patches) > 0

    plt.close(fig)


def test_plot_random_effects_density_no_normal(simple_gamm_result):
    """Test density plot without normal overlay."""
    fig, ax = plot_random_effects_density(simple_gamm_result, show_normal=False)

    assert fig is not None
    # Fewer lines when normal overlay is off
    plt.close(fig)


def test_plot_random_effects_density_with_slopes(gamm_with_slopes):
    """Test density plot for random slopes."""
    fig1, ax1 = plot_random_effects_density(gamm_with_slopes, effect_index=0)
    assert fig1 is not None
    plt.close(fig1)

    fig2, ax2 = plot_random_effects_density(gamm_with_slopes, effect_index=1)
    assert fig2 is not None
    plt.close(fig2)


def test_plot_diagnostics_residuals(simple_gamm_result):
    """Test residuals vs fitted plot."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='residuals')

    assert fig is not None
    assert ax is not None
    assert 'Residuals' in ax.get_title()
    # Should have scatter plot
    assert len(ax.collections) > 0

    plt.close(fig)


def test_plot_diagnostics_fitted(simple_gamm_result):
    """Test fitted vs observed plot."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='fitted')

    assert fig is not None
    assert 'Fitted vs Observed' in ax.get_title()

    plt.close(fig)


def test_plot_diagnostics_qq(simple_gamm_result):
    """Test Q-Q plot of residuals."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='qq')

    assert fig is not None
    assert 'Q-Q' in ax.get_title()

    plt.close(fig)


def test_plot_diagnostics_scale_location(simple_gamm_result):
    """Test scale-location plot."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='scale-location')

    assert fig is not None
    assert 'Scale-Location' in ax.get_title()

    plt.close(fig)


def test_plot_diagnostics_invalid_type(simple_gamm_result):
    """Test error for invalid plot type."""
    with pytest.raises(ValueError, match="Unknown plot_type"):
        plot_diagnostics(simple_gamm_result, plot_type='invalid')


def test_plot_random_effects_summary(simple_gamm_result):
    """Test comprehensive summary plot."""
    fig = plot_random_effects_summary(simple_gamm_result)

    assert fig is not None
    # Should have 4 subplots
    assert len(fig.axes) == 4

    plt.close(fig)


def test_plot_random_effects_summary_with_slopes(gamm_with_slopes):
    """Test summary plot with random slopes."""
    fig = plot_random_effects_summary(gamm_with_slopes, effect_index=0)
    assert fig is not None
    plt.close(fig)

    fig = plot_random_effects_summary(gamm_with_slopes, effect_index=1)
    assert fig is not None
    plt.close(fig)


def test_plot_caterpillar_on_existing_axes(simple_gamm_result):
    """Test plotting on existing axes."""
    fig, ax = plt.subplots()
    fig2, ax2 = plot_caterpillar(simple_gamm_result, ax=ax)

    assert fig2 is fig  # Should return same figure
    assert ax2 is ax  # Should return same axes

    plt.close(fig)


def test_plot_qq_on_existing_axes(simple_gamm_result):
    """Test Q-Q plot on existing axes."""
    fig, ax = plt.subplots()
    fig2, ax2 = plot_random_effects_qq(simple_gamm_result, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    plt.close(fig)


def test_plot_density_on_existing_axes(simple_gamm_result):
    """Test density plot on existing axes."""
    fig, ax = plt.subplots()
    fig2, ax2 = plot_random_effects_density(simple_gamm_result, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    plt.close(fig)


def test_plot_diagnostics_on_existing_axes(simple_gamm_result):
    """Test diagnostics plot on existing axes."""
    fig, ax = plt.subplots()
    fig2, ax2 = plot_diagnostics(simple_gamm_result, plot_type='residuals', ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    plt.close(fig)


def test_plots_with_many_groups():
    """Test plots with large number of groups."""
    np.random.seed(42)

    n_groups, n_per_group = 50, 5
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    b_true = np.random.randn(n_groups) * 0.3
    data['y'] = 2.0 + 0.5*data['x1'] + b_true[data['subject']] + np.random.randn(n)*0.3

    result = fit_gamm(formula="y ~ x1 + (1 | subject)", data=data, covariance='identity')

    # Should handle many groups
    fig, ax = plot_caterpillar(result)
    assert fig is not None
    plt.close(fig)

    fig, ax = plot_random_effects_qq(result)
    assert fig is not None
    plt.close(fig)


def test_plots_with_few_groups():
    """Test plots with small number of groups."""
    np.random.seed(42)

    n_groups, n_per_group = 3, 10
    n = n_groups * n_per_group

    data = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'subject': np.repeat(np.arange(n_groups), n_per_group),
    })

    b_true = np.random.randn(n_groups) * 0.5
    data['y'] = 2.0 + 0.5*data['x1'] + b_true[data['subject']] + np.random.randn(n)*0.3

    result = fit_gamm(formula="y ~ x1 + (1 | subject)", data=data, covariance='identity')

    # Should handle few groups
    fig, ax = plot_caterpillar(result)
    assert fig is not None
    plt.close(fig)

    fig, ax = plot_random_effects_density(result)
    assert fig is not None
    plt.close(fig)


def test_caterpillar_plot_visual_elements(simple_gamm_result):
    """Test that caterpillar plot has expected visual elements."""
    fig, ax = plot_caterpillar(simple_gamm_result)

    # Should have reference line at zero
    vlines = [line for line in ax.lines if line.get_xdata()[0] == 0]
    assert len(vlines) > 0, "Should have reference line at x=0"

    # Should have horizontal lines (CIs)
    hlines = [line for line in ax.lines
              if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]]
    assert len(hlines) > 0, "Should have horizontal CI lines"

    plt.close(fig)


def test_density_plot_histogram_bins(simple_gamm_result):
    """Test that density plot has histogram."""
    fig, ax = plot_random_effects_density(simple_gamm_result)

    # Should have histogram patches
    assert len(ax.patches) > 0, "Should have histogram bars"

    # Should have normal curve when show_normal=True
    assert len(ax.lines) > 0, "Should have normal distribution line"

    plt.close(fig)


def test_residuals_plot_reference_line(simple_gamm_result):
    """Test that residuals plot has reference line at zero."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='residuals')

    # Should have horizontal line at y=0
    hlines = [line for line in ax.lines
              if len(set(line.get_ydata())) == 1 and line.get_ydata()[0] == 0]
    assert len(hlines) > 0, "Should have reference line at y=0"

    plt.close(fig)


def test_fitted_vs_observed_diagonal(simple_gamm_result):
    """Test that fitted vs observed has y=x line."""
    fig, ax = plot_diagnostics(simple_gamm_result, plot_type='fitted')

    # Should have y=x diagonal line
    diagonal_lines = [line for line in ax.lines
                      if len(line.get_xdata()) == 2
                      and np.allclose(line.get_xdata(), line.get_ydata())]
    assert len(diagonal_lines) > 0, "Should have y=x diagonal line"

    plt.close(fig)


def test_summary_plot_structure(simple_gamm_result):
    """Test that summary plot has correct structure."""
    fig = plot_random_effects_summary(simple_gamm_result)

    # Should have 4 subplots
    assert len(fig.axes) == 4

    # Check that each subplot has content
    for ax in fig.axes:
        # Each should have either lines or collections (scatter/histogram)
        has_content = len(ax.lines) > 0 or len(ax.collections) > 0 or len(ax.patches) > 0
        assert has_content, "Each subplot should have visual content"

    plt.close(fig)
