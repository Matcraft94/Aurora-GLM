# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Input/output helpers for Aurora-GLM.

This module provides data loading and model serialization utilities.

Reading Data
------------
>>> from aurora.io import read_csv, read_design_matrix
>>>
>>> # Read CSV to dict of arrays
>>> data = read_csv("data.csv")
>>> X = np.column_stack([data['x1'], data['x2']])
>>> y = data['y']
>>>
>>> # Or directly extract X, y
>>> X, y = read_design_matrix("data.csv", response="y")

Saving Results
--------------
>>> from aurora.io import save_result, export_coefficients
>>>
>>> # Save full result
>>> save_result(result, "model.json")
>>>
>>> # Load it back
>>> loaded = load_result("model.json")
>>>
>>> # Export coefficients to CSV
>>> export_coefficients(result, "coef.csv")

Submodules
----------
readers
    Data ingestion from CSV, JSON, Excel, Stata

writers
    Result export and serialization

converters
    Data format conversion utilities (planned)
"""

from __future__ import annotations

# Readers
from .readers import (
    read_csv,
    read_design_matrix,
    read_json,
    read_excel,
    read_stata,
)

# Writers
from .writers import (
    save_result,
    load_result,
    export_coefficients,
    export_predictions,
)

__all__ = [
    # Readers
    "read_csv",
    "read_design_matrix",
    "read_json",
    "read_excel",
    "read_stata",
    # Writers
    "save_result",
    "load_result",
    "export_coefficients",
    "export_predictions",
]
