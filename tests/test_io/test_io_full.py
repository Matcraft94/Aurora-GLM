"""Tests for aurora.io.readers and aurora.io.writers modules."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from aurora.io.readers import read_csv, read_design_matrix, read_json
from aurora.io.writers import (
    export_coefficients,
    export_predictions,
    load_result,
    save_result,
)


# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------


class TestReadCSV:
    def test_basic_csv(self, tmp_path):
        """Read a basic CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        data = read_csv(csv_file)
        assert "x1" in data
        assert "x2" in data
        assert "y" in data
        np.testing.assert_array_almost_equal(data["x1"], [1.0, 4.0])
        np.testing.assert_array_almost_equal(data["y"], [3.0, 6.0])

    def test_no_header(self, tmp_path):
        """Read CSV without header."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("1.0,2.0\n3.0,4.0\n")
        data = read_csv(csv_file, header=False)
        assert "col_0" in data
        assert "col_1" in data
        np.testing.assert_array_almost_equal(data["col_0"], [1.0, 3.0])

    def test_custom_delimiter(self, tmp_path):
        """Read semicolon-delimited CSV."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a;b\n1;2\n3;4\n")
        data = read_csv(csv_file, delimiter=";")
        np.testing.assert_array_almost_equal(data["a"], [1.0, 3.0])

    def test_file_not_found(self):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            read_csv("/nonexistent/file.csv")

    def test_skip_rows(self, tmp_path):
        """Skip first N rows — pandas makes row 1 the new header."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n")
        data = read_csv(csv_file, skip_rows=1)
        # pandas treats first data row as header when skiprows=1
        assert len(data) == 2  # two columns

    def test_max_rows(self, tmp_path):
        """Limit number of rows read."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n")
        data = read_csv(csv_file, max_rows=2)
        assert len(data["a"]) == 2

    def test_column_subset(self, tmp_path):
        """Read only specific columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x,y,z\n1,2,3\n4,5,6\n")
        data = read_csv(csv_file, columns=["x", "z"])
        assert "x" in data
        assert "z" in data
        assert "y" not in data


# ---------------------------------------------------------------------------
# read_design_matrix
# ---------------------------------------------------------------------------


class TestReadDesignMatrix:
    def test_basic(self, tmp_path):
        """Read design matrix with response."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,x2,y\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        X, y = read_design_matrix(csv_file, response="y")
        assert X.shape == (2, 3)  # 2 rows, 2 predictors + intercept
        assert y.shape == (2,)
        assert X[:, 0].sum() == 2  # intercept column

    def test_no_intercept(self, tmp_path):
        """Read without intercept column."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,y\n1.0,3.0\n4.0,6.0\n")
        X, y = read_design_matrix(csv_file, response="y", intercept=False)
        assert X.shape == (2, 1)
        assert X[0, 0] == 1.0

    def test_missing_response_raises(self, tmp_path):
        """Should raise KeyError if response not in data."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,x2\n1.0,2.0\n")
        with pytest.raises(KeyError, match="Response column"):
            read_design_matrix(csv_file, response="y")

    def test_predictors(self, tmp_path):
        """Specify predictor columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,x2,x3,y\n1,2,3,4\n5,6,7,8\n")
        X, y = read_design_matrix(csv_file, response="y", predictors=["x1", "x3"])
        assert X.shape == (2, 3)  # 2 predictors + intercept

    def test_missing_predictor_raises(self, tmp_path):
        """Should raise KeyError if predictor not in data."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("x1,y\n1.0,3.0\n")
        with pytest.raises(KeyError, match="Predictor column"):
            read_design_matrix(csv_file, response="y", predictors=["x1", "x_missing"])


# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------


class TestReadJson:
    def test_records_orient(self, tmp_path):
        """Read JSON in records format."""
        json_file = tmp_path / "test.json"
        data = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]
        json_file.write_text(json.dumps(data))
        result = read_json(json_file, orient="records")
        np.testing.assert_array_almost_equal(result["x"], [1.0, 3.0])

    def test_columns_orient(self, tmp_path):
        """Read JSON in columns format."""
        json_file = tmp_path / "test.json"
        data = {"x": [1.0, 3.0], "y": [2.0, 4.0]}
        json_file.write_text(json.dumps(data))
        result = read_json(json_file, orient="columns")
        np.testing.assert_array_almost_equal(result["x"], [1.0, 3.0])

    def test_values_orient(self, tmp_path):
        """Read JSON in values format."""
        json_file = tmp_path / "test.json"
        data = [[1.0, 2.0], [3.0, 4.0]]
        json_file.write_text(json.dumps(data))
        result = read_json(json_file, orient="values")
        assert "col_0" in result
        assert "col_1" in result

    def test_empty_records(self, tmp_path):
        """Empty records list returns empty dict."""
        json_file = tmp_path / "test.json"
        json_file.write_text("[]")
        result = read_json(json_file, orient="records")
        assert result == {}

    def test_invalid_orient(self, tmp_path):
        """Invalid orient raises ValueError."""
        json_file = tmp_path / "test.json"
        json_file.write_text("{}")
        with pytest.raises(ValueError, match="Unknown orient"):
            read_json(json_file, orient="invalid")


# ---------------------------------------------------------------------------
# save_result / load_result
# ---------------------------------------------------------------------------


class MockResult:
    """Mock model result for testing serialization."""

    def __init__(self):
        self.converged_ = True
        self.n_iter_ = 10
        self.n_obs_ = 100
        self.coef_ = np.array([1.0, 2.0, -0.5])
        self.intercept_ = 3.0
        self.fitted_values = np.random.randn(100)
        self.residuals = np.random.randn(100)
        self.r_squared = 0.85


class TestSaveLoadResult:
    def test_save_json(self, tmp_path):
        """Save result as JSON."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath)
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["converged"] is True

    def test_save_pickle(self, tmp_path):
        """Save result as pickle."""
        result = MockResult()
        filepath = tmp_path / "model.pkl"
        save_result(result, filepath, format="pickle")
        assert filepath.exists()

    def test_save_auto_json(self, tmp_path):
        """Auto-detect JSON format from extension."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath, format="auto")
        assert filepath.exists()

    def test_save_auto_pickle(self, tmp_path):
        """Auto-detect pickle format from extension."""
        result = MockResult()
        filepath = tmp_path / "model.pkl"
        save_result(result, filepath, format="auto")
        assert filepath.exists()

    def test_save_include_data(self, tmp_path):
        """Include data arrays in JSON."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath, include_data=True)
        with open(filepath) as f:
            data = json.load(f)
        assert "fitted_values" in data

    def test_save_exclude_data(self, tmp_path):
        """Exclude data arrays from JSON."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath, include_data=False)
        with open(filepath) as f:
            data = json.load(f)
        assert "fitted_values" not in data
        assert "residuals" not in data

    def test_load_json(self, tmp_path):
        """Load JSON result."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath)
        loaded = load_result(filepath)
        assert loaded["converged"] is True
        np.testing.assert_array_almost_equal(loaded["coef"], [1.0, 2.0, -0.5])

    def test_load_pickle(self, tmp_path):
        """Load pickle result."""
        result = MockResult()
        filepath = tmp_path / "model.pkl"
        save_result(result, filepath)
        with pytest.warns(UserWarning, match="pickle"):
            loaded = load_result(filepath)
        assert loaded.converged_ is True

    def test_load_not_found(self):
        """Load non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_result("/nonexistent/file.json")

    def test_save_invalid_format(self, tmp_path):
        """Invalid format raises ValueError."""
        result = MockResult()
        with pytest.raises(ValueError, match="Unknown format"):
            save_result(result, tmp_path / "model.txt", format="xml")

    def test_save_with_to_dict(self, tmp_path):
        """Result with to_dict method uses it."""
        class DictResult:
            def to_dict(self):
                return {"custom": "data", "value": np.array([1.0, 2.0])}
        result = DictResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["custom"] == "data"

    def test_version_included(self, tmp_path):
        """JSON output includes aurora version."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert "_aurora_version" in data
        assert data["_aurora_version"] != "unknown"

    def test_save_auto_unknown_ext_defaults_json(self, tmp_path):
        """Auto format with unknown extension defaults to JSON."""
        result = MockResult()
        filepath = tmp_path / "model.xyz"
        save_result(result, filepath, format="auto")
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["converged"] is True

    def test_load_auto_unknown_ext_content_detection(self, tmp_path):
        """Auto format with unknown extension uses content detection."""
        result = MockResult()
        filepath = tmp_path / "model.dat"
        save_result(result, filepath)
        loaded = load_result(filepath)
        assert loaded["converged"] is True

    def test_load_invalid_format(self, tmp_path):
        """Load with invalid format raises ValueError."""
        result = MockResult()
        filepath = tmp_path / "model.json"
        save_result(result, filepath)
        with pytest.raises(ValueError, match="Unknown format"):
            load_result(filepath, format="xml")

    def test_numpy_to_json_types(self, tmp_path):
        """_numpy_to_json handles numpy integer, bool, and list types."""
        from aurora.io.writers import _numpy_to_json
        assert _numpy_to_json(np.int64(5)) == 5
        assert _numpy_to_json(np.bool_(True)) is True
        assert _numpy_to_json([np.float64(1.0), np.int32(2)]) == [1.0, 2]
        assert _numpy_to_json({"a": np.array([1, 2])}) == {"a": [1, 2]}



# ---------------------------------------------------------------------------
# export_coefficients
# ---------------------------------------------------------------------------


class TestExportCoefficients:
    def test_basic_export(self, tmp_path):
        """Export coefficients to CSV."""
        result = MockResult()
        filepath = tmp_path / "coef.csv"
        export_coefficients(result, filepath)
        assert filepath.exists()
        with open(filepath) as f:
            lines = f.readlines()
        assert "name" in lines[0]
        assert "intercept" in lines[1]

    def test_fixed_effects(self, tmp_path):
        """Export from result with fixed_effects_."""
        class FxResult:
            fixed_effects_ = np.array([1.0, 2.0])
        result = FxResult()
        filepath = tmp_path / "coef.csv"
        export_coefficients(result, filepath)
        assert filepath.exists()

    def test_coefficients_attr(self, tmp_path):
        """Export from result with coefficients attr."""
        class CoefResult:
            coefficients = np.array([1.0, 2.0])
        result = CoefResult()
        filepath = tmp_path / "coef.csv"
        export_coefficients(result, filepath)
        assert filepath.exists()

    def test_no_coefficients_raises(self, tmp_path):
        """Result without coefficients raises ValueError."""
        result = type("R", (), {})()
        with pytest.raises(ValueError, match="Cannot extract"):
            export_coefficients(result, tmp_path / "coef.csv")


# ---------------------------------------------------------------------------
# export_predictions
# ---------------------------------------------------------------------------


class TestExportPredictions:
    def test_basic_export(self, tmp_path):
        """Export predictions to CSV."""
        class PredResult:
            def predict(self, X):
                return np.array([1.0, 2.0, 3.0])
        result = PredResult()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        filepath = tmp_path / "pred.csv"
        export_predictions(result, X, filepath)
        assert filepath.exists()
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 4  # header + 3 predictions

    def test_no_predict_raises(self, tmp_path):
        """Result without predict method raises ValueError."""
        result = type("R", (), {})()
        X = np.array([[1, 2]])
        with pytest.raises(ValueError, match="predict"):
            export_predictions(result, X, tmp_path / "pred.csv")
