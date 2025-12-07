# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Generalized Linear Model fitting routines."""

from __future__ import annotations

from ..base.result import GLMResult
from .fitting import fit_glm
from .prediction import predict_glm

__all__ = ["fit_glm", "predict_glm", "GLMResult"]
