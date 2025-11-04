"""Tests for random effects specifications."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm import (
    RandomEffect,
    count_random_effects,
    get_group_indices,
    validate_random_effects,
)


def test_random_effect_intercept_only():
    """RandomEffect should create random intercept (1 | group)."""
    re = RandomEffect(grouping='subject')

    assert re.grouping == 'subject'
    assert re.variables == ()
    assert re.include_intercept is True
    assert re.n_effects == 1
    assert re.covariance == 'unstructured'


def test_random_effect_intercept_plus_slope():
    """RandomEffect should handle random intercept + slope."""
    re = RandomEffect(grouping='subject', variables=('time',))

    assert re.grouping == 'subject'
    assert re.variables == ('time',)
    assert re.include_intercept is True
    assert re.n_effects == 2


def test_random_effect_slope_only():
    """RandomEffect should handle random slope without intercept."""
    re = RandomEffect(
        grouping='subject',
        variables=('time',),
        include_intercept=False
    )

    assert re.n_effects == 1
    assert re.include_intercept is False


def test_random_effect_multiple_slopes():
    """RandomEffect should handle multiple random slopes."""
    re = RandomEffect(
        grouping='subject',
        variables=('time', 'age'),
    )

    assert re.n_effects == 3  # intercept + 2 slopes
    assert re.variables == ('time', 'age')


def test_random_effect_integer_grouping():
    """RandomEffect should accept integer grouping."""
    re = RandomEffect(grouping=0)

    assert re.grouping == 0


def test_random_effect_integer_variables():
    """RandomEffect should accept integer variable indices."""
    re = RandomEffect(grouping='subject', variables=(0, 1))

    assert re.variables == (0, 1)
    assert re.n_effects == 3


def test_random_effect_diagonal_covariance():
    """RandomEffect should accept diagonal covariance structure."""
    re = RandomEffect(grouping='subject', covariance='diagonal')

    assert re.covariance == 'diagonal'


def test_random_effect_identity_covariance():
    """RandomEffect should accept identity covariance structure."""
    re = RandomEffect(grouping='subject', covariance='identity')

    assert re.covariance == 'identity'


def test_random_effect_no_grouping_raises():
    """RandomEffect should raise if grouping is None."""
    with pytest.raises(ValueError, match="grouping must be specified"):
        RandomEffect(grouping=None)


def test_random_effect_invalid_covariance_raises():
    """RandomEffect should raise for invalid covariance structure."""
    with pytest.raises(ValueError, match="covariance must be one of"):
        RandomEffect(grouping='subject', covariance='invalid')


def test_random_effect_no_effects_raises():
    """RandomEffect should raise if no intercept and no variables."""
    with pytest.raises(ValueError, match="must include either intercept"):
        RandomEffect(
            grouping='subject',
            variables=(),
            include_intercept=False
        )


def test_random_effect_variables_list_converted_to_tuple():
    """RandomEffect should convert list variables to tuple."""
    re = RandomEffect(grouping='subject', variables=['time', 'age'])

    assert isinstance(re.variables, tuple)
    assert re.variables == ('time', 'age')


def test_random_effect_single_variable_converted_to_tuple():
    """RandomEffect should convert single variable to tuple."""
    re = RandomEffect(grouping='subject', variables='time')

    assert isinstance(re.variables, tuple)
    assert re.variables == ('time',)


def test_random_effect_repr():
    """RandomEffect should have informative repr."""
    re = RandomEffect(grouping='subject', variables=('time',))

    repr_str = repr(re)
    assert 'RandomEffect' in repr_str
    assert '1 + time' in repr_str or 'time' in repr_str
    assert 'subject' in repr_str


def test_random_effect_repr_with_covariance():
    """RandomEffect repr should show non-default covariance."""
    re = RandomEffect(grouping='subject', covariance='diagonal')

    repr_str = repr(re)
    assert 'diagonal' in repr_str


def test_validate_random_effects_accepts_list():
    """validate_random_effects should accept list of RandomEffect."""
    re1 = RandomEffect(grouping='subject')
    re2 = RandomEffect(grouping='clinic')

    # Should not raise
    validate_random_effects([re1, re2])


def test_validate_random_effects_rejects_non_list():
    """validate_random_effects should reject non-list input."""
    re = RandomEffect(grouping='subject')

    with pytest.raises(TypeError, match="must be a list"):
        validate_random_effects(re)


def test_validate_random_effects_rejects_non_randomeffect_items():
    """validate_random_effects should reject non-RandomEffect items."""
    re = RandomEffect(grouping='subject')

    with pytest.raises(TypeError, match="must be RandomEffect instances"):
        validate_random_effects([re, "invalid"])


def test_get_group_indices_simple():
    """get_group_indices should return unique groups and indices."""
    groups = np.array([1, 1, 2, 2, 3])

    unique, indices = get_group_indices(groups)

    assert np.array_equal(unique, np.array([1, 2, 3]))
    assert np.array_equal(indices[1], np.array([0, 1]))
    assert np.array_equal(indices[2], np.array([2, 3]))
    assert np.array_equal(indices[3], np.array([4]))


def test_get_group_indices_unordered():
    """get_group_indices should work with unordered groups."""
    groups = np.array([2, 1, 3, 1, 2])

    unique, indices = get_group_indices(groups)

    assert np.array_equal(unique, np.array([1, 2, 3]))
    assert np.array_equal(indices[1], np.array([1, 3]))
    assert np.array_equal(indices[2], np.array([0, 4]))
    assert np.array_equal(indices[3], np.array([2]))


def test_count_random_effects_single_term():
    """count_random_effects should count effects for single term."""
    re = RandomEffect(grouping='subject', variables=('time',))

    total, per_term = count_random_effects([re])

    assert total == 2  # intercept + slope
    assert per_term['subject'] == 2


def test_count_random_effects_multiple_terms():
    """count_random_effects should count effects for multiple terms."""
    re1 = RandomEffect(grouping='subject', variables=('time',))
    re2 = RandomEffect(grouping='clinic')

    total, per_term = count_random_effects([re1, re2])

    assert total == 3  # 2 + 1
    assert per_term['subject'] == 2
    assert per_term['clinic'] == 1


def test_count_random_effects_no_intercept():
    """count_random_effects should handle terms without intercept."""
    re = RandomEffect(
        grouping='subject',
        variables=('time', 'age'),
        include_intercept=False
    )

    total, per_term = count_random_effects([re])

    assert total == 2  # Just slopes
    assert per_term['subject'] == 2
