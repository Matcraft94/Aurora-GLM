# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Model diagnostic and smooth term plots for Aurora-GLM.

This submodule provides visualization functions for inspecting fitted
GLM, GAM, and GAMM model results.  Plotting is dispatched from the
top-level :mod:`aurora.visualization` package.

GAM Smooth Term Visualization
-----------------------------
plot_smooth
    Plot an individual smooth term with confidence bands.
plot_all_smooths
    Grid of all smooth terms in a GAM/GAMM.

GAMM Random Effects Visualization
---------------------------------
plot_caterpillar
    Random effects with confidence intervals (caterpillar plot).
plot_random_effects_qq
    Q-Q plot for normality check of random effects.
plot_random_effects_density
    Density plots of random effects.
plot_random_effects_summary
    Summary panel of random effects.

Model Diagnostics
-----------------
plot_diagnostics_panel
    2x2 diagnostic panel (residuals vs fitted, Q-Q, scale-location,
    residuals vs leverage), similar to R's ``plot.lm``.
plot_smooth_effect
    Single smooth term with confidence bands.
plot_all_smooth_effects
    Grid of all smooth terms.

GAMM-Specific Panels
--------------------
plot_gamm_diagnostics
    GAMM diagnostics panel.
plot_gamm_random_effects
    GAMM random effects plots.
"""
