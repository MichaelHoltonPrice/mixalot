"""Tests for cross-validation functionality."""
import numpy as np
import pytest

from mixalot.crossval import CrossValidationFoldsSpec


class TestCrossValidationFoldsSpec:
    """Tests for CrossValidationFoldsSpec class."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=5,
            random_state=42
        )
        assert cv_spec.n_splits == 5
        assert cv_spec.random_state == 42

    def test_validate_with_invalid_n_splits(self):
        """Test validation fails with n_splits < 2."""
        with pytest.raises(ValueError) as excinfo:
            CrossValidationFoldsSpec(
                n_splits=1,
                random_state=42
            )
        assert "n_splits must be at least 2" in str(excinfo.value)

    def test_create_folds_deterministic(self):
        """Test that created folds are deterministic with same random_state."""
        cv_spec1 = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        cv_spec2 = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        
        folds1 = cv_spec1.create_folds(10)
        folds2 = cv_spec2.create_folds(10)
        
        # Check that all folds are identical
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_create_folds_different_with_different_random_state(self):
        """Test that folds differ with different random_state."""
        cv_spec1 = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        cv_spec2 = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=43  # Different random state
        )
        
        folds1 = cv_spec1.create_folds(10)
        folds2 = cv_spec2.create_folds(10)
        
        # Check that at least one fold is different
        any_different = False
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            if not np.array_equal(train1, train2) or not np.array_equal(test1,
                                                                        test2):
                any_different = True
                break
        
        assert any_different, ("Folds should be different with different "
                               "random_state")

    def test_create_folds_shuffles_data(self):
        """Test that data is shuffled when creating folds."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=5,
            random_state=42
        )
        
        n_samples = 25
        folds = cv_spec.create_folds(n_samples)
        
        # With sequential indices and no shuffling, the test sets would be:
        # [0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19],
        # [20,21,22,23,24]
        # Check that at least one test set is different from these
        sequential_test_sets = [
            np.arange(i * 5, (i + 1) * 5) for i in range(5)
        ]
        
        any_different = False
        for (_, test), seq_test in zip(folds, sequential_test_sets):
            if not np.array_equal(np.sort(test), seq_test):
                any_different = True
                break
        
        assert any_different, ("Data should be shuffled with implicit "
                               "shuffle=True")

    def test_get_fold_indices_valid(self):
        """Test getting valid fold indices."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        
        n_samples = 10
        train_idx, test_idx = cv_spec.get_fold_indices(1, n_samples)
        
        # Check that indices are valid
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) + len(test_idx) == n_samples
        assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap
        
        # Test all indices are within range
        assert np.all((0 <= train_idx) & (train_idx < n_samples))
        assert np.all((0 <= test_idx) & (test_idx < n_samples))

    def test_get_fold_indices_invalid_fold_idx_too_high(self):
        """Test that an error is raised when fold_idx is too high."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        
        with pytest.raises(ValueError) as excinfo:
            cv_spec.get_fold_indices(3, 10)  # fold_idx is 3, but n_splits is 3
        
        assert "fold_idx must be between 0 and 2" in str(excinfo.value)

    def test_get_fold_indices_invalid_fold_idx_negative(self):
        """Test that an error is raised when fold_idx is negative."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=3,
            random_state=42
        )
        
        with pytest.raises(ValueError) as excinfo:
            cv_spec.get_fold_indices(-1, 10)  # fold_idx is negative
        
        assert "fold_idx must be between 0 and 2" in str(excinfo.value)

    def test_folds_have_expected_sizes(self):
        """Test that the folds have the expected sizes."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=4,
            random_state=42
        )
        
        n_samples = 20
        folds = cv_spec.create_folds(n_samples)
        
        for train_idx, test_idx in folds:
            # With 20 samples and 4 folds, each test set should have 5 samples
            assert len(test_idx) == 5
            assert len(train_idx) == 15

    def test_all_samples_used_once_in_test_sets(self):
        """Test that all samples appear exactly once in test sets."""
        cv_spec = CrossValidationFoldsSpec(
            n_splits=5,
            random_state=42
        )
        
        n_samples = 25
        folds = cv_spec.create_folds(n_samples)
        
        # Collect all test indices
        all_test_indices = np.concatenate([test for _, test in folds])
        
        # Sort them for comparison
        all_test_indices.sort()
        
        # Check that every sample index appears exactly once
        np.testing.assert_array_equal(all_test_indices, np.arange(n_samples))
