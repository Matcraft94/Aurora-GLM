"""Tests for random effects design matrix construction."""
from __future__ import annotations

import numpy as np
import pytest

from aurora.models.gamm import RandomEffect, construct_Z_matrix, extract_random_effects


def test_construct_Z_random_intercept():
    """construct_Z_matrix should build Z for random intercept."""
    n = 6
    X = np.ones((n, 1))
    groups = np.array([1, 1, 2, 2, 3, 3])
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject')
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # Should have 3 groups, 1 effect per group
    assert Z.shape == (6, 3)

    # Check structure: indicator matrix for groups
    expected = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ], dtype=float)

    np.testing.assert_array_equal(Z, expected)

    # Check info
    assert len(Z_info) == 1
    assert Z_info[0]['grouping'] == 'subject'
    assert Z_info[0]['n_effects'] == 1
    assert Z_info[0]['n_groups'] == 3
    assert Z_info[0]['start_col'] == 0
    assert Z_info[0]['end_col'] == 3


def test_construct_Z_random_slope():
    """construct_Z_matrix should build Z for random intercept + slope."""
    n = 4
    X = np.column_stack([np.ones(n), np.array([0, 1, 2, 3])])
    groups = np.array([1, 1, 2, 2])
    groups_data = {'subject': groups}

    # Random intercept + slope on variable 1 (time)
    re = RandomEffect(grouping='subject', variables=(1,))
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # 2 groups, 2 effects per group = 4 columns
    assert Z.shape == (4, 4)

    # Expected: [intercept_group1, slope_group1, intercept_group2, slope_group2]
    # For group 1 (obs 0, 1): intercept=[1,1], slope=[0,1]
    # For group 2 (obs 2, 3): intercept=[1,1], slope=[2,3]
    expected = np.array([
        [1, 0, 0, 0],  # obs 0: group 1, time=0
        [1, 1, 0, 0],  # obs 1: group 1, time=1
        [0, 0, 1, 2],  # obs 2: group 2, time=2
        [0, 0, 1, 3],  # obs 3: group 2, time=3
    ], dtype=float)

    np.testing.assert_array_equal(Z, expected)

    # Check info
    assert Z_info[0]['n_effects'] == 2
    assert Z_info[0]['n_groups'] == 2


def test_construct_Z_slope_only():
    """construct_Z_matrix should handle random slope without intercept."""
    n = 4
    X = np.column_stack([np.ones(n), np.array([1, 2, 3, 4])])
    groups = np.array([1, 1, 2, 2])
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject', variables=(1,), include_intercept=False)
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # 2 groups, 1 effect per group = 2 columns
    assert Z.shape == (4, 2)

    expected = np.array([
        [1, 0],  # obs 0: group 1, time=1
        [2, 0],  # obs 1: group 1, time=2
        [0, 3],  # obs 2: group 2, time=3
        [0, 4],  # obs 3: group 2, time=4
    ], dtype=float)

    np.testing.assert_array_equal(Z, expected)

    assert Z_info[0]['n_effects'] == 1


def test_construct_Z_multiple_random_effects():
    """construct_Z_matrix should handle multiple random effect terms."""
    n = 4
    X = np.ones((n, 1))
    subject_groups = np.array([1, 1, 2, 2])
    clinic_groups = np.array([1, 2, 1, 2])
    groups_data = {
        'subject': subject_groups,
        'clinic': clinic_groups,
    }

    re1 = RandomEffect(grouping='subject')
    re2 = RandomEffect(grouping='clinic')

    Z, Z_info = construct_Z_matrix(X, [re1, re2], groups_data)

    # Subject: 2 groups, 1 effect = 2 cols
    # Clinic: 2 groups, 1 effect = 2 cols
    # Total: 4 cols
    assert Z.shape == (4, 4)

    # Check info
    assert len(Z_info) == 2
    assert Z_info[0]['grouping'] == 'subject'
    assert Z_info[0]['start_col'] == 0
    assert Z_info[0]['end_col'] == 2
    assert Z_info[1]['grouping'] == 'clinic'
    assert Z_info[1]['start_col'] == 2
    assert Z_info[1]['end_col'] == 4


def test_construct_Z_unbalanced_groups():
    """construct_Z_matrix should handle unbalanced group sizes."""
    n = 7
    X = np.ones((n, 1))
    groups = np.array([1, 1, 1, 2, 2, 3, 3])
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject')
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    assert Z.shape == (7, 3)

    # Group 1: 3 obs, Group 2: 2 obs, Group 3: 2 obs
    expected = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=float)

    np.testing.assert_array_equal(Z, expected)


def test_construct_Z_missing_grouping_raises():
    """construct_Z_matrix should raise if grouping variable not provided."""
    n = 4
    X = np.ones((n, 1))
    groups_data = {}  # Missing 'subject'

    re = RandomEffect(grouping='subject')

    with pytest.raises(ValueError, match="not found in groups_data"):
        construct_Z_matrix(X, [re], groups_data)


def test_construct_Z_wrong_length_groups_raises():
    """construct_Z_matrix should raise if groups length doesn't match."""
    n = 4
    X = np.ones((n, 1))
    groups = np.array([1, 1, 2])  # Length 3, not 4
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject')

    with pytest.raises(ValueError, match="does not match number of observations"):
        construct_Z_matrix(X, [re], groups_data)


def test_construct_Z_variable_out_of_bounds_raises():
    """construct_Z_matrix should raise if variable index out of bounds."""
    n = 4
    X = np.ones((n, 2))  # Only 2 columns
    groups = np.array([1, 1, 2, 2])
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject', variables=(5,))  # Index 5 out of bounds

    with pytest.raises(ValueError, match="out of bounds"):
        construct_Z_matrix(X, [re], groups_data)


def test_construct_Z_empty_random_effects():
    """construct_Z_matrix should handle empty random effects list."""
    n = 4
    X = np.ones((n, 1))
    groups_data = {}

    Z, Z_info = construct_Z_matrix(X, [], groups_data)

    assert Z.shape == (4, 0)
    assert len(Z_info) == 0


def test_extract_random_effects_intercept():
    """extract_random_effects should extract coefficients by group."""
    b = np.array([0.5, -0.3, 0.2])
    Z_info = [{
        'grouping': 'subject',
        'n_effects': 1,
        'n_groups': 3,
        'groups': np.array([1, 2, 3]),
        'start_col': 0,
        'end_col': 3,
    }]

    random_effects = extract_random_effects(b, Z_info)

    assert 'subject' in random_effects
    assert len(random_effects['subject']) == 3

    np.testing.assert_array_equal(random_effects['subject'][1], np.array([0.5]))
    np.testing.assert_array_equal(random_effects['subject'][2], np.array([-0.3]))
    np.testing.assert_array_equal(random_effects['subject'][3], np.array([0.2]))


def test_extract_random_effects_intercept_slope():
    """extract_random_effects should handle intercept + slope."""
    # 2 groups, 2 effects each = 4 coefficients
    b = np.array([0.5, 0.1, -0.3, 0.2])
    Z_info = [{
        'grouping': 'subject',
        'n_effects': 2,
        'n_groups': 2,
        'groups': np.array([1, 2]),
        'start_col': 0,
        'end_col': 4,
    }]

    random_effects = extract_random_effects(b, Z_info)

    np.testing.assert_array_equal(
        random_effects['subject'][1],
        np.array([0.5, 0.1])
    )
    np.testing.assert_array_equal(
        random_effects['subject'][2],
        np.array([-0.3, 0.2])
    )


def test_extract_random_effects_multiple_terms():
    """extract_random_effects should handle multiple random effect terms."""
    # Subject: 2 groups, 1 effect = 2 coefs
    # Clinic: 2 groups, 1 effect = 2 coefs
    b = np.array([0.5, -0.3, 0.1, 0.2])
    Z_info = [
        {
            'grouping': 'subject',
            'n_effects': 1,
            'n_groups': 2,
            'groups': np.array([1, 2]),
            'start_col': 0,
            'end_col': 2,
        },
        {
            'grouping': 'clinic',
            'n_effects': 1,
            'n_groups': 2,
            'groups': np.array([10, 20]),
            'start_col': 2,
            'end_col': 4,
        }
    ]

    random_effects = extract_random_effects(b, Z_info)

    assert 'subject' in random_effects
    assert 'clinic' in random_effects

    np.testing.assert_array_equal(random_effects['subject'][1], np.array([0.5]))
    np.testing.assert_array_equal(random_effects['subject'][2], np.array([-0.3]))
    np.testing.assert_array_equal(random_effects['clinic'][10], np.array([0.1]))
    np.testing.assert_array_equal(random_effects['clinic'][20], np.array([0.2]))


def test_construct_and_extract_roundtrip():
    """Constructing Z and extracting b should preserve structure."""
    n = 6
    X = np.column_stack([np.ones(n), np.arange(n)])
    groups = np.array([1, 1, 2, 2, 3, 3])
    groups_data = {'subject': groups}

    re = RandomEffect(grouping='subject', variables=(1,))
    Z, Z_info = construct_Z_matrix(X, [re], groups_data)

    # Create some random effect coefficients
    b = np.random.randn(Z.shape[1])

    # Extract
    random_effects = extract_random_effects(b, Z_info)

    # Check dimensions
    assert len(random_effects['subject']) == 3
    for group_id in [1, 2, 3]:
        assert random_effects['subject'][group_id].shape == (2,)
