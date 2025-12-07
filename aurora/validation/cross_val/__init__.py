# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Cross-validation utilities."""

from .evaluate import CrossValResult, cross_val_score
from .split import KFold, StratifiedKFold

__all__ = ["KFold", "StratifiedKFold", "CrossValResult", "cross_val_score"]
