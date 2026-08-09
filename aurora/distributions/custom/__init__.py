# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Custom user-defined distribution helpers.

.. note::
    **Not implemented.** Aurora-GLM currently has no quasi-likelihood
    families (quasi-Poisson, quasi-binomial) and no user-defined custom
    family support beyond subclassing
    :class:`aurora.distributions.base.Family` directly. This module is a
    placeholder documenting that known limitation; overdispersed count
    models should use
    :class:`aurora.distributions.families.NegativeBinomialFamily` instead
    of quasi-Poisson.
"""
