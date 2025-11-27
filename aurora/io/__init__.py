"""Input/output helpers for Aurora-GLM.

This module provides data loading and model serialization utilities.

Submodules (Planned)
--------------------
converters
    Data format conversion (pandas, R, etc.)

readers
    Data ingestion from various formats (CSV, RDS, etc.)

writers
    Model export and serialization

Notes
-----
This module is planned for future development. Currently, Aurora-GLM
works directly with NumPy arrays and requires users to handle I/O
with standard tools like pandas or numpy.

Example usage (planned):
>>> from aurora.io import read_csv, to_dataframe
>>> data = read_csv("data.csv")
>>> result = fit_glm(data.X, data.y)
>>> to_dataframe(result).to_csv("results.csv")
"""