"""Tests for the CSV fallback path in aurora.io.readers (when pandas is not available)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aurora.io.readers import read_csv, read_design_matrix, read_excel, read_stata


# ---------------------------------------------------------------------------
# CSV fallback path (lines 119-164 of readers/__init__.py)
# ---------------------------------------------------------------------------


class TestCSVFallback:
    """Test read_csv when pandas is not available (csv module fallback)."""

    @pytest.fixture(autouse=True)
    def _hide_pandas(self):
        """Make pandas unavailable so read_csv uses the csv fallback."""
        with patch.dict("sys.modules", {"pandas": None}):
            yield

    def test_basic_read(self, tmp_path: Path):
        p = tmp_path / "basic.csv"
        p.write_text("x,y\n1.0,2.0\n3.0,4.0\n")
        data = read_csv(str(p))
        assert "x" in data
        assert "y" in data
        np.testing.assert_array_equal(data["x"], [1.0, 3.0])
        np.testing.assert_array_equal(data["y"], [2.0, 4.0])

    def test_no_header(self, tmp_path: Path):
        p = tmp_path / "nohead.csv"
        p.write_text("10.0,20.0\n30.0,40.0\n")
        data = read_csv(str(p), header=False)
        assert "col_0" in data
        assert "col_1" in data
        np.testing.assert_array_equal(data["col_0"], [10.0, 30.0])

    def test_skip_rows(self, tmp_path: Path):
        p = tmp_path / "skip.csv"
        p.write_text("junk\nx,y\n10,20\n")
        data = read_csv(str(p), skip_rows=1)
        assert "x" in data
        np.testing.assert_array_equal(data["x"], [10.0])

    def test_max_rows(self, tmp_path: Path):
        p = tmp_path / "maxr.csv"
        p.write_text("x,y\n1,2\n3,4\n5,6\n")
        data = read_csv(str(p), max_rows=2)
        assert len(data["x"]) == 2

    def test_column_subset(self, tmp_path: Path):
        p = tmp_path / "cols.csv"
        p.write_text("a,b,c\n1,2,3\n4,5,6\n")
        data = read_csv(str(p), columns=["a", "c"])
        assert "a" in data
        assert "c" in data
        assert "b" not in data

    def test_custom_delimiter(self, tmp_path: Path):
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\t2\n3\t4\n")
        data = read_csv(str(p), delimiter="\t")
        np.testing.assert_array_equal(data["a"], [1.0, 3.0])

    def test_non_numeric_produces_object_array(self, tmp_path: Path):
        p = tmp_path / "mixed.csv"
        p.write_text("name,val\nalice,1\nbob,2\n")
        data = read_csv(str(p))
        assert "name" in data
        assert data["name"].dtype == object

    def test_header_int_zero(self, tmp_path: Path):
        """header=0 should behave like header=True (first row is header)."""
        p = tmp_path / "h0.csv"
        p.write_text("x,y\n1,2\n3,4\n")
        data = read_csv(str(p), header=0)
        assert "x" in data
        np.testing.assert_array_equal(data["x"], [1.0, 3.0])

    def test_header_false_with_skip(self, tmp_path: Path):
        p = tmp_path / "hf.csv"
        p.write_text("junk\n10,20\n30,40\n")
        data = read_csv(str(p), header=False, skip_rows=1)
        assert "col_0" in data
        np.testing.assert_array_equal(data["col_0"], [10.0, 30.0])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_csv("/nonexistent/file.csv")


# ---------------------------------------------------------------------------
# read_excel and read_stata import guards
# ---------------------------------------------------------------------------


class TestExcelStataImportGuards:
    """Test that read_excel and read_stata raise ImportError without pandas."""

    def test_read_excel_no_pandas(self, tmp_path: Path):
        p = tmp_path / "test.xlsx"
        p.write_bytes(b"PK")  # minimal bytes so file exists
        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises(ImportError, match="pandas"):
                read_excel(str(p))

    def test_read_stata_no_pandas(self, tmp_path: Path):
        p = tmp_path / "test.dta"
        p.write_bytes(b"<Stata>")
        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises(ImportError, match="pandas"):
                read_stata(str(p))


# ---------------------------------------------------------------------------
# read_design_matrix via fallback
# ---------------------------------------------------------------------------


class TestDesignMatrixFallback:
    """Test read_design_matrix when pandas is unavailable."""

    @pytest.fixture(autouse=True)
    def _hide_pandas(self):
        with patch.dict("sys.modules", {"pandas": None}):
            yield

    def test_basic(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("x,y\n1.0,10.0\n2.0,20.0\n")
        X, y = read_design_matrix(str(p), response="y", intercept=True)
        assert X.shape == (2, 2)
        np.testing.assert_array_equal(X[:, 0], 1.0)
        np.testing.assert_array_equal(y, [10.0, 20.0])

    def test_no_intercept(self, tmp_path: Path):
        p = tmp_path / "dm.csv"
        p.write_text("x,y\n1.0,10.0\n2.0,20.0\n")
        X, y = read_design_matrix(str(p), response="y", intercept=False)
        assert X.shape == (2, 1)
