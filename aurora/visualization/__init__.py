"""Visualization helpers for Aurora-GLM models.

This module provides a centralized interface to all plotting functions
for GLM, GAM, and GAMM models.

GAM Smooth Effects
------------------
- plot_smooth: Plot individual smooth term with confidence bands
- plot_all_smooths: Grid of all smooth terms

GAMM Random Effects
-------------------
- plot_caterpillar: Random effects with CIs (caterpillar plot)
- plot_random_effects_qq: Q-Q plot for normality check
- plot_random_effects_density: Density plots of random effects
- plot_random_effects_summary: Summary panel of random effects

Model Diagnostics
-----------------
- plot_diagnostics: Residual diagnostics panel (like R's plot.lm)
- plot_diagnostics_panel: 2x2 diagnostic panel
- plot_smooth_effect: Single smooth term with confidence bands
- plot_all_smooth_effects: Grid of all smooth terms
"""
from __future__ import annotations

# GAM smooth term visualization
from aurora.models.gam.plotting import (
    plot_all_smooths,
    plot_smooth,
)

# GAMM random effects visualization
from aurora.models.gamm.plotting import (
    plot_caterpillar,
    plot_diagnostics,
    plot_random_effects_density,
    plot_random_effects_qq,
    plot_random_effects_summary,
    # Phase 5.4 additions
    plot_diagnostics_panel,
    plot_smooth_effect,
    plot_all_smooth_effects,
)

# GAMM diagnostics (additional)
from aurora.models.gamm.diagnostics import (
    plot_diagnostics as plot_gamm_diagnostics,
    plot_random_effects as plot_gamm_random_effects,
)

__all__ = [
    # GAM smooth terms
    "plot_smooth",
    "plot_all_smooths",
    # GAMM random effects
    "plot_caterpillar",
    "plot_random_effects_qq",
    "plot_random_effects_density",
    "plot_random_effects_summary",
    # Diagnostics
    "plot_diagnostics",
    "plot_diagnostics_panel",
    "plot_smooth_effect",
    "plot_all_smooth_effects",
    # GAMM specific
    "plot_gamm_diagnostics",
    "plot_gamm_random_effects",
]
