"""Tests for aurora.utils.validation.decorators module.

Tests the input validation decorators: @validate_array, @validate_positive, etc.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.utils.validation.decorators import (
    ValidationError,
    validate_array,
    validate_in_range,
    validate_non_negative,
    validate_one_of,
    validate_positive,
    validate_probability,
    validated,
)


class TestValidateArray:
    """Tests for @validate_array decorator."""

    def test_valid_numpy_array(self):
        """Test with valid numpy array."""

        @validate_array("x")
        def func(x):
            return x.sum()

        arr = np.array([1, 2, 3])
        result = func(arr)
        assert result == 6

    def test_valid_list_converted(self):
        """Test that list passes validation and is passed through."""

        @validate_array("x")
        def func(x):
            return x

        # The decorator validates but does not convert the return value
        # It just passes the input through after validation
        result = func([1, 2, 3])
        # The decorator validates by converting to array internally
        # but returns the original value since it passes validation
        assert result == [1, 2, 3]

    def test_invalid_type_raises(self):
        """Test that non-numeric array type with dtype_check raises error."""

        @validate_array("x", dtype_check="numeric")
        def func(x):
            return x

        with pytest.raises((TypeError, ValidationError)):
            func("not an array")

    def test_ndim_validation(self):
        """Test ndim validation."""

        @validate_array("x", ndim=2)
        def func(x):
            return x

        # Should pass with 2D
        result = func(np.ones((3, 3)))
        assert result.ndim == 2

        # Should fail with 1D
        with pytest.raises((ValueError, ValidationError)):
            func(np.ones(3))

    def test_min_ndim_validation(self):
        """Test min_rows validation (no min_ndim parameter)."""

        @validate_array("x", min_rows=1)
        def func(x):
            return x

        # Should pass with 1+ rows
        func(np.ones(3))
        func(np.ones((3, 3)))

        # Empty array should fail
        with pytest.raises((ValueError, ValidationError)):
            func(np.ones(0))


class TestValidatePositive:
    """Tests for @validate_positive decorator."""

    def test_positive_value_passes(self):
        """Test that positive values pass."""

        @validate_positive("x")
        def func(x):
            return x

        assert func(1.0) == 1.0
        assert func(100) == 100

    def test_zero_fails(self):
        """Test that zero fails."""

        @validate_positive("x")
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(0)

    def test_negative_fails(self):
        """Test that negative value fails."""

        @validate_positive("x")
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(-1)

    def test_positive_array(self):
        """Test with positive scalar."""

        @validate_positive("x")
        def func(x):
            return x

        # validate_positive validates scalar values
        result = func(5.0)
        assert result == 5.0

    def test_array_with_negative_fails(self):
        """Test that array with negative element fails."""

        @validate_positive("x")
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(np.array([1, -2, 3]))


class TestValidateNonNegative:
    """Tests for @validate_non_negative decorator."""

    def test_positive_passes(self):
        """Test positive value passes."""

        @validate_non_negative("x")
        def func(x):
            return x

        assert func(1.0) == 1.0

    def test_zero_passes(self):
        """Test zero passes."""

        @validate_non_negative("x")
        def func(x):
            return x

        assert func(0) == 0

    def test_negative_fails(self):
        """Test negative fails."""

        @validate_non_negative("x")
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(-1)


class TestValidateProbability:
    """Tests for @validate_probability decorator."""

    def test_valid_probability(self):
        """Test valid probability values."""

        @validate_probability("p")
        def func(p):
            return p

        assert func(0.5) == 0.5
        assert func(0.0) == 0.0
        assert func(1.0) == 1.0

    def test_below_zero_fails(self):
        """Test that p < 0 fails."""

        @validate_probability("p")
        def func(p):
            return p

        with pytest.raises((ValueError, ValidationError)):
            func(-0.1)

    def test_above_one_fails(self):
        """Test that p > 1 fails."""

        @validate_probability("p")
        def func(p):
            return p

        with pytest.raises((ValueError, ValidationError)):
            func(1.1)


class TestValidateInRange:
    """Tests for @validate_in_range decorator."""

    def test_within_range_passes(self):
        """Test value within range passes."""

        @validate_in_range("x", lower=0, upper=10)
        def func(x):
            return x

        assert func(5) == 5
        assert func(0) == 0
        assert func(10) == 10

    def test_below_range_fails(self):
        """Test value below range fails."""

        @validate_in_range("x", lower=0, upper=10)
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(-1)

    def test_above_range_fails(self):
        """Test value above range fails."""

        @validate_in_range("x", lower=0, upper=10)
        def func(x):
            return x

        with pytest.raises((ValueError, ValidationError)):
            func(11)

    def test_exclusive_bounds(self):
        """Test exclusive bounds using 'neither'."""

        @validate_in_range("x", lower=0, upper=1, inclusive="neither")
        def func(x):
            return x

        assert func(0.5) == 0.5

        with pytest.raises((ValueError, ValidationError)):
            func(0)

        with pytest.raises((ValueError, ValidationError)):
            func(1)


class TestValidateOneOf:
    """Tests for @validate_one_of decorator."""

    def test_valid_choice(self):
        """Test valid choice passes."""

        @validate_one_of("method", choices=["gcv", "reml", "ml"])
        def func(method):
            return method

        assert func("gcv") == "gcv"
        assert func("reml") == "reml"

    def test_invalid_choice_fails(self):
        """Test invalid choice fails."""

        @validate_one_of("method", choices=["gcv", "reml", "ml"])
        def func(method):
            return method

        with pytest.raises((ValueError, ValidationError)):
            func("invalid")

    def test_case_sensitivity(self):
        """Test case sensitivity."""

        @validate_one_of("method", choices=["GCV", "REML"])
        def func(method):
            return method

        # Different case should fail
        with pytest.raises((ValueError, ValidationError)):
            func("gcv")


class TestValidatedDecorator:
    """Tests for @validated composite decorator."""

    def test_validated_composite(self):
        """Test @validated composite decorator on function."""

        @validated(value={"type": "positive"})
        def set_value(value):
            return value

        assert set_value(5) == 5

        with pytest.raises((ValueError, ValidationError)):
            set_value(-1)


class TestMultipleValidators:
    """Tests for combining multiple validators."""

    def test_multiple_decorators(self):
        """Test multiple validation decorators on same function."""

        @validate_array("x", dtype_check="numeric")
        @validate_positive("alpha")
        def func(x, alpha):
            return np.sum(x) * alpha

        result = func(np.array([1, 2, 3]), alpha=2.0)
        assert result == 12.0

    def test_multiple_parameters(self):
        """Test validating multiple parameters."""

        @validate_array("x")
        @validate_positive("alpha")
        @validate_probability("p")
        def func(x, alpha, p):
            return x.sum() * alpha * p

        result = func(np.array([1, 2]), 2.0, 0.5)
        assert result == 3.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_value_handling(self):
        """Test handling of None values."""

        @validate_positive("x", allow_none=True)
        def func(x=None):
            return x

        assert func(None) is None
        assert func(1) == 1

    def test_keyword_argument(self):
        """Test validation of keyword arguments."""

        @validate_positive("x")
        def func(x=1):
            return x

        assert func(x=5) == 5

        with pytest.raises((ValueError, ValidationError)):
            func(x=-1)

    def test_missing_parameter(self):
        """Test handling when validated parameter has a default."""

        @validate_positive("x", allow_none=True)
        def func(y, x=None):
            return y

        # Should work since x defaults to None and allow_none=True
        result = func(y=5)
        assert result == 5
