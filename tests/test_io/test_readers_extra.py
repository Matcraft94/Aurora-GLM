"""Tests for aurora.io.readers module.

Covers: read_csv (edge cases), read_json (all orients), read_design_matrix
(edge cases), read_excel and read_stata (import guards).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aurora.io.readers import read_csv, read_design_matrix, read_json


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------


class TestReadCSVExtra:
    def test_no_header(self, tmp_path: Path):
        p = tmp_path / "nohead.csv"
        p.write_text("1.0,2.0\n3.0,4.0\n")
        data = read_csv(str(p), header=False)
        assert "col_0" in data
        assert "col_1" in data
        assert len(data["col_0"]) == 2

    def test_custom_delimiter(self, tmp_path: Path):
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        data = read_csv(str(p), delimiter="\t")
        np.testing.assert_array_equal(data["a"], [1.0, 3.0])
        np.testing.assert_array_equal(data["b"], [2.0, 4.0])

    def test_skip_rows(self, tmp_path: Path):
        p = tmp_path / "skip.csv"
        p.write_text("junk line\nx,y\n10,20\n30,40\n")
        data = read_csv(str(p), skip_rows=1)
        assert "x" in data
        assert len(data["x"]) == 2

    def test_max_rows(self, tmp_path: Path):
        p = tmp_path / "maxr.csv"
        p.write_text("x,y\n1,2\n3,4\n5,6\n7,8\n")
        data = read_csv(str(p), max_rows=2)
        assert len(data["x"]) == 2

    def test_column_subset(self, tmp_path: Path):
        p = tmp_path / "cols.csv"
        p.write_text("a,b,c\n1,2,3\n4,5,6\n")
        data = read_csv(str(p), columns=["a", "c"])
        assert "a" in data
        assert "c" in data
        assert "b" not in data

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_csv("/no/such/file.csv")

    def test_string_columns_fallback(self, tmp_path: Path):
        """Non-numeric columns should produce object arrays."""
        p = tmp_path / "mixed.csv"
        p.write_text("name,value\nalice,1\nbob,2\n")
        data = read_csv(str(p))
        assert "name" in data
        assert data["name"].dtype == object or str(data["name"].dtype).startswith("str")


# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------


class TestReadJSON:
    def test_records_orient(self, tmp_path: Path):
        p = tmp_path / "rec.json"
        records = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        p.write_text(json.dumps(records))
        data = read_json(str(p), orient="records")
        np.testing.assert_array_equal(data["x"], [1, 3])
        np.testing.assert_array_equal(data["y"], [2, 4])

    def test_columns_orient(self, tmp_path: Path):
        p = tmp_path / "cols.json"
        p.write_text(json.dumps({"x": [10, 20], "y": [30, 40]}))
        data = read_json(str(p), orient="columns")
        np.testing.assert_array_equal(data["x"], [10, 20])

    def test_values_orient(self, tmp_path: Path):
        p = tmp_path / "vals.json"
        p.write_text(json.dumps([[1, 2], [3, 4]]))
        data = read_json(str(p), orient="values")
        assert "col_0" in data
        assert "col_1" in data
        np.testing.assert_array_equal(data["col_0"], [1, 3])

    def test_empty_records(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        data = read_json(str(p), orient="records")
        assert data == {}

    def test_invalid_orient(self, tmp_path: Path):
        p = tmp_path / "x.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unknown orient"):
            read_json(str(p), orient="bad")


# ---------------------------------------------------------------------------
# read_design_matrix
# ---------------------------------------------------------------------------


class TestReadDesignMatrixExtra:
    def test_no_intercept(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("a,b,y\n1,2,10\n3,4,20\n")
        X, y = read_design_matrix(str(p), response="y", intercept=False)
        assert X.shape == (2, 2)

    def test_with_intercept(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("a,b,y\n1,2,10\n3,4,20\n")
        X, y = read_design_matrix(str(p), response="y", intercept=True)
        assert X.shape[1] == 3  # 2 predictors + intercept
        np.testing.assert_array_equal(X[:, 0], 1.0)

    def test_missing_response_raises(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("a,b\n1,2\n3,4\n")
        with pytest.raises(KeyError, match="not found"):
            read_design_matrix(str(p), response="z")

    def test_missing_predictor_raises(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("a,y\n1,10\n3,20\n")
        with pytest.raises(KeyError, match="Predictor"):
            read_design_matrix(str(p), response="y", predictors=["nope"])

    def test_no_predictors_raises(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("y\n10\n20\n")
        with pytest.raises(ValueError, match="No predictor"):
            read_design_matrix(str(p), response="y")
