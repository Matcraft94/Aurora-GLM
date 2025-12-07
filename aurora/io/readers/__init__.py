# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Lucy Eduardo Arias

"""Data ingestion interfaces for Aurora-GLM.

This module provides functions for reading data from various formats
into arrays suitable for Aurora-GLM model fitting.

Supported Formats
-----------------
- CSV files (via pandas or built-in csv)
- Excel files (.xlsx, .xls)
- JSON files
- Stata files (.dta)
- R data files (.rds, .rdata) - if pyreadr installed

Examples
--------
>>> from aurora.io.readers import read_csv, read_design_matrix
>>>
>>> # Simple CSV reading
>>> X, y = read_design_matrix("data.csv", response="y")
>>>
>>> # With pandas-style options
>>> data = read_csv("data.csv", index_col=0, na_values=['NA', '.'])
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np


def read_csv(
    filepath: str | Path,
    *,
    delimiter: str = ",",
    header: bool | int = True,
    columns: Sequence[str] | None = None,
    dtype: type | dict[str, type] | None = None,
    na_values: Sequence[str] | None = None,
    skip_rows: int = 0,
    max_rows: int | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read a CSV file into a dictionary of arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    delimiter : str, default=','
        Field delimiter character.
    header : bool or int, default=True
        If True, first row is used as column names.
        If int, specifies row number for headers (0-indexed).
        If False, columns are named 'col_0', 'col_1', etc.
    columns : sequence of str, optional
        Subset of columns to read.
    dtype : type or dict, optional
        Data type for columns. Can be a single type or dict mapping
        column names to types.
    na_values : sequence of str, optional
        Additional strings to recognize as NA/NaN.
    skip_rows : int, default=0
        Number of rows to skip at the beginning.
    max_rows : int, optional
        Maximum number of rows to read.
    **kwargs
        Additional arguments passed to pandas.read_csv if available.

    Returns
    -------
    data : dict[str, ndarray]
        Dictionary mapping column names to numpy arrays.

    Examples
    --------
    >>> data = read_csv("data.csv")
    >>> X = np.column_stack([data['x1'], data['x2'], data['x3']])
    >>> y = data['y']

    Notes
    -----
    If pandas is installed, uses pandas.read_csv for robust parsing.
    Otherwise, falls back to Python's csv module.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try pandas first (more robust)
    try:
        import pandas as pd

        header_arg = 0 if header is True else (None if header is False else header)

        df = pd.read_csv(
            filepath,
            sep=delimiter,
            header=header_arg,
            usecols=columns,
            dtype=dtype,
            na_values=na_values,
            skiprows=skip_rows,
            nrows=max_rows,
            **kwargs,
        )

        if header is False:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]

        return {col: df[col].values for col in df.columns}

    except ImportError:
        # Fallback to csv module
        import csv

        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Skip rows
            for _ in range(skip_rows):
                next(reader, None)

            # Read header
            if header is True or header == 0:
                col_names = next(reader)
            else:
                first_row = next(reader)
                col_names = [f"col_{i}" for i in range(len(first_row))]
                # Re-read the row since we consumed it
                rows = [first_row]

            if header is True or header == 0:
                rows = []

            # Read data
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(row)

        # Convert to arrays
        data = {}
        for j, name in enumerate(col_names):
            if columns is not None and name not in columns:
                continue

            col_data = [row[j] if j < len(row) else "" for row in rows]

            # Try to convert to numeric
            try:
                arr = np.array(col_data, dtype=float)
            except ValueError:
                arr = np.array(col_data, dtype=object)

            data[name] = arr

        return data


def read_design_matrix(
    filepath: str | Path,
    *,
    response: str,
    predictors: Sequence[str] | None = None,
    intercept: bool = True,
    delimiter: str = ",",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV file and extract design matrix X and response y.

    This is a convenience function for the common case of reading
    a data file for regression modeling.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    response : str
        Name of the response (y) column.
    predictors : sequence of str, optional
        Names of predictor (X) columns. If None, uses all non-response columns.
    intercept : bool, default=True
        If True, prepends a column of ones for the intercept.
    delimiter : str, default=','
        Field delimiter character.
    **kwargs
        Additional arguments passed to read_csv.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.
    y : ndarray of shape (n_samples,)
        Response vector.

    Examples
    --------
    >>> X, y = read_design_matrix("data.csv", response="y")
    >>> result = fit_glm(X, y)

    >>> # Select specific predictors
    >>> X, y = read_design_matrix("data.csv", response="price",
    ...                           predictors=["sqft", "bedrooms", "age"])
    """
    data = read_csv(filepath, delimiter=delimiter, **kwargs)

    if response not in data:
        raise KeyError(
            f"Response column '{response}' not found in data. "
            f"Available columns: {list(data.keys())}"
        )

    y = data[response]

    # Get predictor columns
    if predictors is None:
        predictors = [col for col in data.keys() if col != response]
    else:
        for pred in predictors:
            if pred not in data:
                raise KeyError(f"Predictor column '{pred}' not found in data.")

    if len(predictors) == 0:
        raise ValueError("No predictor columns found.")

    # Build design matrix
    X_cols = [data[col] for col in predictors]
    X = np.column_stack(X_cols).astype(float)

    if intercept:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

    return X, y.astype(float)


def read_json(
    filepath: str | Path,
    *,
    orient: Literal["records", "columns", "values"] = "records",
) -> dict[str, np.ndarray]:
    """Read a JSON file into a dictionary of arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file.
    orient : {'records', 'columns', 'values'}, default='records'
        Expected JSON structure:
        - 'records': List of {column: value} dicts
        - 'columns': {column: [values]} dict
        - 'values': List of lists (rows)

    Returns
    -------
    data : dict[str, ndarray]
        Dictionary mapping column names to numpy arrays.

    Examples
    --------
    >>> data = read_json("data.json")
    >>> X, y = data['X'], data['y']
    """
    import json

    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if orient == "columns":
        # {column: [values]} format
        return {k: np.array(v) for k, v in raw.items()}

    elif orient == "records":
        # [{column: value}, ...] format
        if not raw:
            return {}
        columns = list(raw[0].keys())
        return {col: np.array([row.get(col) for row in raw]) for col in columns}

    elif orient == "values":
        # [[row], [row], ...] format
        arr = np.array(raw)
        return {f"col_{i}": arr[:, i] for i in range(arr.shape[1])}

    else:
        raise ValueError(
            f"Unknown orient '{orient}'. Use 'records', 'columns', or 'values'."
        )


def read_excel(
    filepath: str | Path,
    *,
    sheet_name: str | int = 0,
    header: bool | int = True,
    columns: Sequence[str] | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read an Excel file into a dictionary of arrays.

    Requires openpyxl (for .xlsx) or xlrd (for .xls).

    Parameters
    ----------
    filepath : str or Path
        Path to the Excel file.
    sheet_name : str or int, default=0
        Sheet to read (name or 0-indexed position).
    header : bool or int, default=True
        Row to use for column names.
    columns : sequence of str, optional
        Subset of columns to read.
    **kwargs
        Additional arguments passed to pandas.read_excel.

    Returns
    -------
    data : dict[str, ndarray]
        Dictionary mapping column names to numpy arrays.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to read Excel files. "
            "Install with: pip install pandas openpyxl"
        )

    filepath = Path(filepath)
    header_arg = 0 if header is True else (None if header is False else header)

    df = pd.read_excel(
        filepath, sheet_name=sheet_name, header=header_arg, usecols=columns, **kwargs
    )

    if header is False:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    return {col: df[col].values for col in df.columns}


def read_stata(
    filepath: str | Path,
    *,
    columns: Sequence[str] | None = None,
    convert_categoricals: bool = True,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Read a Stata .dta file into a dictionary of arrays.

    Requires pandas.

    Parameters
    ----------
    filepath : str or Path
        Path to the Stata file.
    columns : sequence of str, optional
        Subset of columns to read.
    convert_categoricals : bool, default=True
        Whether to convert labeled integers to Categorical.
    **kwargs
        Additional arguments passed to pandas.read_stata.

    Returns
    -------
    data : dict[str, ndarray]
        Dictionary mapping column names to numpy arrays.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to read Stata files. Install with: pip install pandas"
        )

    df = pd.read_stata(
        filepath, columns=columns, convert_categoricals=convert_categoricals, **kwargs
    )

    return {col: df[col].values for col in df.columns}


__all__ = [
    "read_csv",
    "read_design_matrix",
    "read_json",
    "read_excel",
    "read_stata",
]
