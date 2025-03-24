"""Tests for cross-validation functionality."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
import torch

from mixalot.crossval import (
    CrossValidationFoldsSpec,
    dataframe_to_mixed_dataset,
    fit_single_fold,
    load_cross_validation_folds_spec_from_json,
    run_cross_validation,
)
from mixalot.datasets import DatasetSpec, MixedDataset, VarSpec
from mixalot.models import RandomForestSpec


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


class TestLoadCrossValidationFoldsSpec:
    """Tests for load_cross_validation_folds_spec_from_json function."""
    
    def test_load_valid_cross_validation_spec(self, tmp_path):
        """Test loading a valid cross-validation specification."""
        # Create a temporary JSON file with valid data
        spec_path = tmp_path / "cv_spec.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "n_splits": 5,
                "random_state": 42
            }, f)
        
        # Load the spec
        cv_spec = load_cross_validation_folds_spec_from_json(str(spec_path))
        
        # Verify it has the correct values
        assert cv_spec.n_splits == 5
        assert cv_spec.random_state == 42
        
        # Verify it's the correct type
        assert isinstance(cv_spec, CrossValidationFoldsSpec)

    def test_load_missing_file(self):
        """Test that an error is raised for a non-existent file."""
        with pytest.raises(FileNotFoundError) as excinfo:
            load_cross_validation_folds_spec_from_json("nonexistent_file.json")
        
        assert "does not exist" in str(excinfo.value)

    def test_load_missing_n_splits(self, tmp_path):
        """Test that an error is raised when n_splits is missing."""
        # Create a temporary JSON file with missing n_splits
        spec_path = tmp_path / "missing_splits.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "random_state": 42
            }, f)
        
        with pytest.raises(ValueError) as excinfo:
            load_cross_validation_folds_spec_from_json(str(spec_path))
        
        assert "missing required field: 'n_splits'" in str(excinfo.value)

    def test_load_missing_random_state(self, tmp_path):
        """Test that an error is raised when random_state is missing."""
        # Create a temporary JSON file with missing random_state
        spec_path = tmp_path / "missing_random_state.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "n_splits": 5
            }, f)
        
        with pytest.raises(ValueError) as excinfo:
            load_cross_validation_folds_spec_from_json(str(spec_path))
        
        assert "missing required field: 'random_state'" in str(excinfo.value)

    def test_load_invalid_n_splits(self, tmp_path):
        """Test that an error is raised when n_splits is invalid."""
        # Create a temporary JSON file with invalid n_splits
        spec_path = tmp_path / "invalid_splits.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "n_splits": 1,  # Must be >= 2
                "random_state": 42
            }, f)
        
        with pytest.raises(ValueError) as excinfo:
            load_cross_validation_folds_spec_from_json(str(spec_path))
        
        assert "n_splits must be an integer >= 2" in str(excinfo.value)

    def test_load_non_integer_n_splits(self, tmp_path):
        """Test that an error is raised when n_splits is not an integer."""
        # Create a temporary JSON file with non-integer n_splits
        spec_path = tmp_path / "non_integer_splits.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "n_splits": "five",  # Must be an integer
                "random_state": 42
            }, f)
        
        with pytest.raises(ValueError) as excinfo:
            load_cross_validation_folds_spec_from_json(str(spec_path))
        
        assert "n_splits must be an integer >= 2" in str(excinfo.value)

    def test_load_non_integer_random_state(self, tmp_path):
        """Test that an error is raised when random_state is not an integer."""
        # Create a temporary JSON file with non-integer random_state
        spec_path = tmp_path / "non_integer_random_state.json"
        with open(spec_path, 'w') as f:
            json.dump({
                "n_splits": 5,
                "random_state": "forty-two"  # Must be an integer
            }, f)
        
        with pytest.raises(ValueError) as excinfo:
            load_cross_validation_folds_spec_from_json(str(spec_path))
        
        assert "random_state must be an integer" in str(excinfo.value)


@pytest.fixture
def simple_dataset_spec():
    """Fixture creating a realistic DatasetSpec instance."""
    cat_var_spec = VarSpec(
        var_name='var1',
        var_type='categorical',
        categorical_mapping=[{'a'}, {'b'}, {'c'}]
    )

    ord_var_spec = VarSpec(
        var_name='var2',
        var_type='ordinal',
        categorical_mapping=[{'low'}, {'medium'}, {'high'}]
    )

    num_var_spec = VarSpec(
        var_name='var3',
        var_type='numerical'
    )

    return DatasetSpec(
        cat_var_specs=[cat_var_spec],
        ord_var_specs=[ord_var_spec],
        num_var_specs=[num_var_spec]
    )


@pytest.fixture
def simple_model_spec():
    """Fixture creating a realistic RandomForestSpec instance."""
    return RandomForestSpec(
        y_var='var1',
        independent_vars=['var2', 'var3'],
        hyperparameters={
            'n_estimators': 10,
            'max_features': {'type': 'float', 'value': 0.5}
        }
    )


@pytest.fixture
def simple_numerical_model_spec():
    """Fixture creating a RandomForestSpec with numerical target variable."""
    return RandomForestSpec(
        y_var='var3',
        independent_vars=['var1', 'var2'],
        hyperparameters={
            'n_estimators': 10,
            'max_features': {'type': 'float', 'value': 0.5}
        }
    )


@pytest.fixture
def simple_cv_spec():
    """Fixture creating a simple CV spec."""
    return CrossValidationFoldsSpec(n_splits=2, random_state=42)


@pytest.fixture
def simple_dataframe():
    """Fixture creating a simple dataframe for testing."""
    return pd.DataFrame({
        'var1': ['a', 'b', 'a', 'c', 'b', 'c', 'a', 'b'],
        'var2': ['low', 'medium', 'high', 'medium', 'low', 'high', 'medium',
                 'high'],
        'var3': [0.5, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 0.7]
    })


class TestFitSingleFold:
    """Tests for fit_single_fold."""
    def test_fit_single_fold_categorical_target(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec
    ):
        """Test fitting a single fold with categorical target variable."""
        df = pd.DataFrame({
            'var1': ['a', 'a', 'b', 'b', 'c', 'c'],
            'var2': ['low', 'high', 'medium', 'low', 'high', 'medium'],
            'var3': [0, 1, 0, 1, 0, 1]
        })
    
        # Manually mock fold indices so that each fold has all categories
        with patch.object(simple_cv_spec,
                          'get_fold_indices') as mock_get_fold_indices:
            mock_get_fold_indices.return_value = (np.array([0, 2, 4]),
                                                  np.array([1, 3, 5]))
    
            result = fit_single_fold(
                df,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                fold_idx=0,
                random_seed=42
            )
    
        assert 'model' in result
        assert 'fold_loss' in result
        assert isinstance(result['fold_loss'], float)
        assert len(result['test_indices']) == 3
        assert len(result['predictions']) == 3

    def test_fit_single_fold_missing_categories(
        self, simple_dataset_spec, simple_model_spec
    ):
        """Test an error when test set has categories not in train set."""
        # Create a dataset where a category only appears in the test set
        df = pd.DataFrame({
            'var1': ['a', 'a', 'b', 'b', 'c'],  # 'c' will be only in test set
            'var2': ['low', 'high', 'medium', 'low', 'high'],
            'var3': [0.5, 0.6, 1.2, 1.3, 1.5]
        })
        
        # Create a CV spec that will place 'c' in the test set
        # We'll manually define the train/test split to ensure this
        class MockCVSpec:
            n_splits = 2
            def get_fold_indices(self, fold_idx, n_samples):
                # Fixed indices that ensure 'c' is only in test set
                if fold_idx == 0:
                    # 'c' is at index 4
                    return np.array([0, 1, 2, 3]), np.array([4])
                return np.array([1, 2, 3, 4]), np.array([0])
        
        mock_cv_spec = MockCVSpec()
        
        # Verify our test setup: 'c' should only be in test set
        train_indices, test_indices = mock_cv_spec.get_fold_indices(0, len(df))
        train_categories = set(df.iloc[train_indices]['var1'])
        test_categories = set(df.iloc[test_indices]['var1'])
        
        # Confirm test has a category not in train
        assert not test_categories.issubset(train_categories), \
            "Test should have categories not present in train set"
        
        # Now test that fit_single_fold raises an error
        with pytest.raises(ValueError, match="Test set contains categories"):
            fit_single_fold(
                df,
                simple_dataset_spec,
                mock_cv_spec,
                simple_model_spec,
                fold_idx=0,
                random_seed=42
            )

    def test_fit_single_fold_invalid_fold_idx(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test that an error is raised with invalid fold index."""
        with pytest.raises(ValueError,
                           match="fold_idx must be between 0 and 1"):
            fit_single_fold(
                simple_dataframe, 
                simple_dataset_spec, 
                simple_cv_spec,
                simple_model_spec, 
                fold_idx=2, 
                random_seed=42
            )

    def test_fit_single_fold_missing_columns(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec
    ):
        """Test that an error is raised when dataframe is missing columns."""
        # Create dataframe missing var2
        df_missing_column = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var3': [0.5, 1.2, 0.8]
        })

        with pytest.raises(ValueError,
                           match="Dataframe is missing required columns"):
            fit_single_fold(
                df_missing_column, 
                simple_dataset_spec, 
                simple_cv_spec,
                simple_model_spec, 
                fold_idx=0, 
                random_seed=42
            )

    def test_fit_single_fold_numerical_target(
        self, simple_dataset_spec, simple_numerical_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test fitting a single fold with numerical target variable."""
        # For numerical targets, we don't need to worry about missing
        # categories
        result = fit_single_fold(
            simple_dataframe, 
            simple_dataset_spec, 
            simple_cv_spec,
            simple_numerical_model_spec, 
            fold_idx=0, 
            random_seed=42
        )
    
        # Check all expected keys are present
        assert set(result.keys()) == {'model', 'fold_loss', 'test_indices',
                                      'predictions'}
        
        # For regression, the model should be a RandomForestRegressor
        assert isinstance(result['model'], RandomForestRegressor)
        
        # Check fold_loss is a float
        assert isinstance(result['fold_loss'], float)
        
        # Each prediction should be a single value, not an array
        for idx in result['predictions']:
            assert isinstance(result['predictions'][idx], (float, np.float32,
                                                           np.float64))

class TestDataframeToMixedDataset:
    """Tests for dataframe_to_mixed_dataset."""
    def test_dataframe_to_mixed_dataset_categorical_target(
        self, simple_dataset_spec, simple_model_spec
    ):
        """Test conversion of dataframe with categorical target."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.5, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_model_spec
        )

        assert isinstance(dataset, MixedDataset)
        
        # For categorical target (var1), Xcat should have 0 columns
        # as the target is extracted
        assert dataset.Xcat is None or dataset.Xcat.shape[1] == 0
        assert dataset.Xord is not None and dataset.Xord.shape == (3, 1)
        assert dataset.Xnum is not None and dataset.Xnum.shape == (3, 1)
        
        # Y_data should be the categorical target var1
        assert dataset.y_data is not None
        assert dataset.y_data.shape == (3,)
        
        # The dataset should have the model_spec
        assert dataset.model_spec == simple_model_spec

    def test_dataframe_to_mixed_dataset_numerical_target(
        self, simple_dataset_spec, simple_numerical_model_spec
    ):
        """Test conversion of dataframe with numerical target."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.5, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_numerical_model_spec
        )

        assert isinstance(dataset, MixedDataset)
        
        # For numerical target (var3), Xnum should have 0 columns
        # as the target is extracted
        assert dataset.Xcat is not None and dataset.Xcat.shape == (3, 1)
        assert dataset.Xord is not None and dataset.Xord.shape == (3, 1)
        assert dataset.Xnum is None or dataset.Xnum.shape[1] == 0
        
        # Y_data should be the numerical target var3
        assert dataset.y_data is not None
        assert dataset.y_data.shape == (3,)
        assert torch.is_floating_point(dataset.y_data)

    def test_dataframe_with_missing_values(
        self, simple_dataset_spec, simple_model_spec
    ):
        """Test handling of missing values in dataframe."""
        df = pd.DataFrame({
            'var1': ['a', None, 'c'],
            'var2': ['low', 'medium', None],
            'var3': [1.0, None, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_model_spec
        )
        
        # Check that missing values are properly encoded
        # Categorical and ordinal missing values are encoded as 0
        Xcat, Xord, Xnum, y_data = dataset.get_arrays()
        
        # Target is var1, so it's removed from Xcat
        assert y_data[1].item() == 0  # None in var1 should be encoded as 0
        
        # Check missing value in var2 (ordinal)
        assert Xord[2, 0].item() == 0  # None in var2 should be encoded as 0
        
        # Check missing value in var3 (numerical)
        assert torch.isnan(Xnum[1, 0])  # None in var3 should be encoded as NaN

    def test_dataframe_missing_column_raises_error(
        self, simple_dataset_spec, simple_model_spec
    ):
        """Test missing column raises ValueError."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            # 'var2' column is missing
            'var3': [1.0, 2.5, 3.8]
        })

        with pytest.raises(ValueError, match="Variable 'var2' .* not found"):
            dataframe_to_mixed_dataset(df, simple_dataset_spec,
                                       simple_model_spec)


class TestRunCrossValidation:
    """Tests for run_cross_validation function."""
    def test_run_cross_validation_basic(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path
    ):
        """Test basic functionality of run_cross_validation."""
        
        # Patch the fold indices to ensure a valid split
        with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
            # Create a split where all categories appear in both train and test
            # First 4 rows in training, last 4 in testing
            mock_indices.return_value = (
                np.array([0, 1, 2, 3]), 
                np.array([4, 5, 6, 7])
            )
            
            # Run cross-validation
            results = run_cross_validation(
                simple_dataframe,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                random_seed=42,
                output_folder=str(tmp_path)
            )
        
        # Check the structure of the results
        assert 'fold_results' in results
        assert 'average_loss' in results
        assert 'std_loss' in results
        assert 'all_predictions' in results
        assert 'loss_type' in results
        assert 'n_folds_completed' in results
        assert 'n_folds_total' in results
        
        # Check that the right number of folds were completed
        assert results['n_folds_completed'] == simple_cv_spec.n_splits
        assert results['n_folds_total'] == simple_cv_spec.n_splits
        
        # Check that fold results are as expected
        assert len(results['fold_results']) == simple_cv_spec.n_splits
        for fold_result in results['fold_results']:
            assert 'model' in fold_result
            assert 'fold_loss' in fold_result
            assert 'test_indices' in fold_result
            assert 'predictions' in fold_result
        
        # Check that output files were created
        for i in range(simple_cv_spec.n_splits):
            assert (tmp_path / f"fold_{i}_model.pkl").exists()
            assert (tmp_path / f"fold_{i}_results.csv").exists()
        
        assert (tmp_path / "cv_summary.csv").exists()
        assert (tmp_path / "all_predictions.csv").exists()    

    def test_run_cross_validation_no_output_folder(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test run_cross_validation without output folder."""
        
        # Patch the fold indices to ensure a valid split
        with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
            # Create a split where all categories appear in both train and test
            mock_indices.return_value = (
                np.array([0, 1, 2, 3]), 
                np.array([4, 5, 6, 7])
            )
            
            # Run cross-validation without output folder
            results = run_cross_validation(
                simple_dataframe,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                random_seed=42,
                output_folder=None
            )
        
        # Check the structure of the results
        assert 'fold_results' in results
        assert 'average_loss' in results
        assert 'std_loss' in results
        assert 'all_predictions' in results
        
        # Check that the right number of folds were completed
        assert len(results['fold_results']) == simple_cv_spec.n_splits

    def test_run_cross_validation_overwrite_files(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test overwrite_files flag in run_cross_validation."""
        
        # Patch the fold indices to ensure a valid split
        with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
            # Create a split where all categories appear in both train and test
            mock_indices.return_value = (
                np.array([0, 1, 2, 3]), 
                np.array([4, 5, 6, 7])
            )
            
            # First run to create initial files
            run_cross_validation(
                simple_dataframe,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                random_seed=42,
                output_folder=str(tmp_path)
            )
        
        # Modify one of the output files to detect if it gets overwritten
        fold0_results_path = tmp_path / "fold_0_results.csv"
        original_df = pd.read_csv(fold0_results_path)
        modified_df = original_df.copy()
        modified_df['fold_loss'] = 999.0  # Clearly modified value
        modified_df.to_csv(fold0_results_path, index=False)
        
        # Mock fit_single_fold to track calls
        original_fit_single_fold = fit_single_fold
        mock_calls = []
        
        def mock_fit_single_fold(*args, **kwargs):
            mock_calls.append((args, kwargs))
            return original_fit_single_fold(*args, **kwargs)
        
        monkeypatch.setattr(
            'mixalot.crossval.fit_single_fold', mock_fit_single_fold
        )
        
        # Patch the fold indices again for the second run
        with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
            mock_indices.return_value = (
                np.array([0, 1, 2, 3]), 
                np.array([4, 5, 6, 7])
            )
            
            # Run again with overwrite_files=False (default)
            run_cross_validation(
                simple_dataframe,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                random_seed=42,
                output_folder=str(tmp_path)
            )
        
        # Check that fit_single_fold was not called (files not overwritten)
        assert len(mock_calls) == 0
        
        # Verify the modified file was not changed
        df_after = pd.read_csv(fold0_results_path)
        assert df_after['fold_loss'].iloc[0] == 999.0
        
        # Reset mock and patch again for the third run
        mock_calls.clear()
        
        with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
            mock_indices.return_value = (
                np.array([0, 1, 2, 3]), 
                np.array([4, 5, 6, 7])
            )
            
            # Run again with overwrite_files=True
            run_cross_validation(
                simple_dataframe,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec,
                random_seed=42,
                output_folder=str(tmp_path),
                overwrite_files=True
            )
        
        # Check that fit_single_fold was called
        assert len(mock_calls) == simple_cv_spec.n_splits
        
        # Verify the file was overwritten
        df_after = pd.read_csv(fold0_results_path)
        assert df_after['fold_loss'].iloc[0] != 999.0

    def test_run_cross_validation_rerun_folds(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test rerun_folds flag in run_cross_validation."""
        # First run to create initial files
        run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Mock fit_single_fold to track calls
        original_fit_single_fold = fit_single_fold
        mock_calls = []
        
        def mock_fit_single_fold(*args, **kwargs):
            mock_calls.append((args, kwargs))
            return original_fit_single_fold(*args, **kwargs)
        
        monkeypatch.setattr(
            'mixalot.crossval.fit_single_fold', mock_fit_single_fold
        )
        
        # Run with rerun_folds=True
        run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_model_spec,
            random_seed=42,
            output_folder=str(tmp_path),
            rerun_folds=True
        )
        
        # Check that fit_single_fold was called for each fold
        assert len(mock_calls) == simple_cv_spec.n_splits

    def test_run_cross_validation_input_validation(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec
    ):
        """Test input validation in run_cross_validation."""
        # Create a dataframe missing a required column
        df_missing_column = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            # 'var2' is missing
            'var3': [0.5, 1.2, 0.8]
        })
        
        # Test with missing column
        with pytest.raises(ValueError, 
                          match="Dataframe is missing required columns"):
            run_cross_validation(
                df_missing_column,
                simple_dataset_spec,
                simple_cv_spec,
                simple_model_spec
            )
        
        # Test with invalid cv_spec
        invalid_cv_spec = type('InvalidCVSpec', (), {})()
        with pytest.raises(ValueError, 
                          match="cv_spec must be a CrossValidationFoldsSpec"):
            run_cross_validation(
                pd.DataFrame({
                    'var1': ['a'], 'var2': ['low'], 'var3': [0.5]
                }),
                simple_dataset_spec,
                invalid_cv_spec,
                simple_model_spec
            )
        
        # Test with invalid model_spec
        invalid_model_spec = type('InvalidModelSpec', (), {})()
        with pytest.raises(ValueError, 
                          match="model_spec must be a SingleTargetModelSpec"):
            run_cross_validation(
                pd.DataFrame({
                    'var1': ['a'], 'var2': ['low'], 'var3': [0.5]
                }),
                simple_dataset_spec,
                simple_cv_spec,
                invalid_model_spec
            )

    def test_run_cross_validation_fold_error_handling(
        self, simple_dataset_spec, simple_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test handling of errors in individual folds."""
        
        # Mock fit_single_fold to fail on the first fold only
        def mock_fit_single_fold(*args, **kwargs):
            if args[4] == 0:  # fold_idx is 0
                raise ValueError("Simulated error in first fold")
            
            # For fold_idx 1, we need to patch the fold indices
            # to ensure a valid split for the remaining fold
            mock_indices = patch.object(simple_cv_spec, 'get_fold_indices')
            with mock_indices as indices_mock:
                indices_mock.return_value = (
                    np.array([0, 1, 2, 3]), 
                    np.array([4, 5, 6, 7])
                )
                return fit_single_fold(*args, **kwargs)
        
        monkeypatch.setattr(
            'mixalot.crossval.fit_single_fold', mock_fit_single_fold
        )
        
        # Run cross-validation with the failing first fold
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Check that the function continued despite the first fold failing
        assert results['n_folds_completed'] == simple_cv_spec.n_splits - 1
        assert results['n_folds_total'] == simple_cv_spec.n_splits
        assert len(results['fold_results']) == simple_cv_spec.n_splits - 1
        
        # Check that only the successful fold has output files
        assert not (tmp_path / "fold_0_model.pkl").exists()
        assert not (tmp_path / "fold_0_results.csv").exists()
        assert (tmp_path / "fold_1_model.pkl").exists()
        assert (tmp_path / "fold_1_results.csv").exists()
        assert (tmp_path / "cv_summary.csv").exists()

    def test_run_cross_validation_numerical_target(
        self, simple_dataset_spec, simple_numerical_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path
    ):
        """Test run_cross_validation with numerical target variable."""
        # Run cross-validation with numerical target
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_numerical_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Check that loss type is mean_squared_error for regression
        assert results['loss_type'] == 'mean_squared_error'
        
        # Check format of predictions
        for idx, pred in results['all_predictions'].items():
            assert isinstance(pred, (float, int))
            
        # Check that the CSV files were created correctly
        fold0_results = pd.read_csv(tmp_path / "fold_0_results.csv")
        all_predictions = pd.read_csv(tmp_path / "all_predictions.csv")
        
        # Regression results should have 'prediction' column, not 'prob_*'
        assert 'prediction' in fold0_results.columns
        assert 'prediction' in all_predictions.columns
        assert not any(col.startswith('prob_') 
                      for col in fold0_results.columns)
