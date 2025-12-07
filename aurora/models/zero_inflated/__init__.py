# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Zero-Inflated Models for Count Data with Excess Zeros.

This module provides Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative
Binomial (ZINB) models for count data where the number of zeros exceeds
what would be expected from standard count distributions.

Models
------
- ZIP: Zero-Inflated Poisson for excess zeros without overdispersion
- ZINB: Zero-Inflated Negative Binomial for excess zeros with overdispersion

Mathematical Framework
----------------------
Zero-inflated models are two-component mixture models:

    P(Y = 0) = π + (1-π) × f(0)     (structural + sampling zeros)
    P(Y = k) = (1-π) × f(k)         for k > 0

where:
- π is the probability of a "structural zero" (from inflation process)
- f(k) is the count distribution (Poisson or Negative Binomial)
- (1-π) × f(0) is the probability of a "sampling zero" from the count process

Use Cases
---------
- Species counts (many sites with zero individuals)
- Insurance claims (many policies with zero claims)
- Healthcare utilization (many patients with zero visits)
- Manufacturing defects (many units with zero defects)

References
----------
.. [1] Lambert, D. (1992).
       "Zero-inflated Poisson regression, with an application to defects
       in manufacturing."
       Technometrics, 34(1), 1-14.
.. [2] Ridout, M., Demétrio, C. G., & Hinde, J. (1998).
       "Models for count data with many zeros."
       Proceedings of the XIXth International Biometric Conference, 179-192.
"""

from __future__ import annotations

from .zip import ZeroInflatedPoissonFamily, fit_zip, ZIPResult
from .zinb import ZeroInflatedNegBinFamily, fit_zinb, ZINBResult
from .diagnostics import vuong_test, score_test_zero_inflation

__all__ = [
    # ZIP
    "ZeroInflatedPoissonFamily",
    "fit_zip",
    "ZIPResult",
    # ZINB
    "ZeroInflatedNegBinFamily",
    "fit_zinb",
    "ZINBResult",
    # Diagnostics
    "vuong_test",
    "score_test_zero_inflation",
]
