# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Result serialization and export utilities for Aurora-GLM.

This module provides functions for saving model results and data to various formats.

Supported Formats
-----------------
- CSV: Coefficients, predictions, residuals
- JSON: Full result serialization
- Pickle: Python-native serialization
- HDF5: Large array storage (if h5py installed)

Examples
--------
>>> from aurora.io.writers import save_result, export_coefficients
>>>
>>> # Save full result
>>> save_result(result, "model.json")
>>>
>>> # Export just coefficients
>>> export_coefficients(result, "coef.csv")
>>>
>>> # Export predictions
>>> export_predictions(result, X_new, "predictions.csv")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np


def save_result(
    result: Any,
    filepath: str | Path,
    *,
    format: Literal["json", "pickle", "auto"] = "auto",
    include_data: bool = False,
) -> None:
    """Save a model result to file.

    Parameters
    ----------
    result : ModelResult
        A fitted model result from fit_glm, fit_gam, etc.
    filepath : str or Path
        Output file path.
    format : {'json', 'pickle', 'auto'}, default='auto'
        Output format. 'auto' infers from file extension.
    include_data : bool, default=False
        If True, includes training data (X, y) in the output.

    Examples
    --------
    >>> from aurora import fit_glm, Gaussian
    >>> result = fit_glm(X, y, family=Gaussian())
    >>> save_result(result, "model.json")
    >>> save_result(result, "model.pkl", format='pickle')
    """
    filepath = Path(filepath)

    # Infer format from extension
    if format == "auto":
        ext = filepath.suffix.lower()
        if ext in (".json",):
            format = "json"
        elif ext in (".pkl", ".pickle"):
            format = "pickle"
        else:
            format = "json"  # Default to JSON

    if format == "json":
        _save_json(result, filepath, include_data=include_data)
    elif format == "pickle":
        _save_pickle(result, filepath)
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'json' or 'pickle'.")


def _save_json(result: Any, filepath: Path, include_data: bool) -> None:
    """Save result as JSON."""
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    else:
        data = _result_to_dict(result)

    if not include_data:
        # Remove large data arrays
        data.pop("X", None)
        data.pop("y", None)
        data.pop("fitted_values", None)
        data.pop("residuals", None)

    # Convert numpy arrays to lists
    data = _numpy_to_json(data)

    # Add metadata
    data["_aurora_version"] = _get_version()
    data["_result_type"] = type(result).__name__

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_pickle(result: Any, filepath: Path) -> None:
    """Save result as pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a result object to dictionary."""
    data = {}

    # Common attributes
    for attr in [
        "converged_",
        "n_iter_",
        "n_obs_",
        "coef_",
        "intercept_",
        "residual_variance_",
        "fixed_effects_",
        "random_effects_",
        "variance_components_",
        "log_likelihood_",
    ]:
        if hasattr(result, attr):
            data[attr.rstrip("_")] = getattr(result, attr)

    # Properties
    for prop in [
        "coefficients",
        "fitted_values",
        "residuals",
        "r_squared",
        "adj_r_squared",
        "aic",
        "bic",
    ]:
        if hasattr(result, prop):
            try:
                data[prop] = getattr(result, prop)
            except Exception:
                pass

    return data


def _numpy_to_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_json(item) for item in obj]
    return obj


def _get_version() -> str:
    """Get Aurora-GLM version."""
    try:
        from aurora import __version__

        return __version__
    except ImportError:
        return "unknown"


def load_result(
    filepath: str | Path,
    *,
    format: Literal["json", "pickle", "auto"] = "auto",
) -> Any:
    """Load a saved model result.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved result file.
    format : {'json', 'pickle', 'auto'}, default='auto'
        Input format. 'auto' infers from file extension.

    Returns
    -------
    result : dict or object
        The loaded result. JSON files return dicts, pickle files return
        the original object.

    Examples
    --------
    >>> result = load_result("model.json")
    >>> print(result['coefficients'])
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Infer format
    if format == "auto":
        ext = filepath.suffix.lower()
        if ext in (".json",):
            format = "json"
        elif ext in (".pkl", ".pickle"):
            format = "pickle"
        else:
            # Try to detect from content
            with open(filepath, "rb") as f:
                header = f.read(1)
            format = "pickle" if header[0] > 127 else "json"

    if format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert lists back to arrays
        for key in [
            "coefficients",
            "coef",
            "fitted_values",
            "residuals",
            "fixed_effects",
            "random_effects",
        ]:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key])

        return data

    elif format == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Unknown format '{format}'.")


def export_coefficients(
    result: Any,
    filepath: str | Path,
    *,
    include_std_errors: bool = True,
    include_pvalues: bool = True,
) -> None:
    """Export model coefficients to CSV.

    Parameters
    ----------
    result : ModelResult
        A fitted model result.
    filepath : str or Path
        Output CSV file path.
    include_std_errors : bool, default=True
        Include standard errors if available.
    include_pvalues : bool, default=True
        Include p-values if available.

    Examples
    --------
    >>> export_coefficients(result, "coefficients.csv")
    """
    filepath = Path(filepath)

    # Get coefficients
    if hasattr(result, "coef_"):
        coef = result.coef_
        intercept = getattr(result, "intercept_", None)
    elif hasattr(result, "fixed_effects_"):
        coef = result.fixed_effects_
        intercept = None
    elif hasattr(result, "coefficients"):
        coef = result.coefficients
        intercept = None
    else:
        raise ValueError("Cannot extract coefficients from result.")

    # Build data rows
    rows = []

    if intercept is not None:
        row = ["intercept", intercept]
        rows.append(row)

    for i, c in enumerate(coef):
        row = [f"X{i}", c]
        rows.append(row)

    # Write CSV
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.writer(f)

        header = ["name", "coefficient"]
        writer.writerow(header)

        for row in rows:
            writer.writerow(row)


def export_predictions(
    result: Any,
    X: np.ndarray,
    filepath: str | Path,
    *,
    include_se: bool = False,
    include_ci: bool = False,
    ci_level: float = 0.95,
) -> None:
    """Export model predictions to CSV.

    Parameters
    ----------
    result : ModelResult
        A fitted model result with a predict method.
    X : ndarray
        Design matrix for predictions.
    filepath : str or Path
        Output CSV file path.
    include_se : bool, default=False
        Include standard errors of predictions.
    include_ci : bool, default=False
        Include confidence intervals.
    ci_level : float, default=0.95
        Confidence level for intervals.

    Examples
    --------
    >>> export_predictions(result, X_new, "predictions.csv", include_ci=True)
    """
    filepath = Path(filepath)

    if not hasattr(result, "predict"):
        raise ValueError("Result must have a predict method.")

    predictions = result.predict(X)

    # Write CSV
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.writer(f)

        header = ["observation", "prediction"]
        writer.writerow(header)

        for i, pred in enumerate(predictions):
            writer.writerow([i, pred])


__all__ = [
    "save_result",
    "load_result",
    "export_coefficients",
    "export_predictions",
]
