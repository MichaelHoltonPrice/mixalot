"""Tests for cross-validation functionality."""
import inspect
import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch

from mixalot.crossval import (
    _fit_model,
    _calculate_losses,
    _process_fold_observations,
    _validate_cv_inputs,
    CrossValidationFoldsSpec,
    dataframe_to_mixed_dataset,
    load_cross_validation_folds_spec_from_json,
    run_cross_validation,
)
from mixalot.datasets import DatasetSpec, MixedDataset, VarSpec
from mixalot.helpers import combine_features
from mixalot.models import ANNEnsembleSpec, RandomForestSpec
from mixalot.trainers import BasicAnn, EnsembleTorchModel


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


# Fixtures
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
def simple_rf_model_spec():
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
def simple_rf_numerical_model_spec():
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
def simple_ann_model_spec():
    """Fixture creating a realistic ANNEnsembleSpec instance."""
    return ANNEnsembleSpec(
        y_var='var1',
        independent_vars=['var2', 'var3'],
        hyperparameters={
            'hidden_sizes': [32, 16],
            'dropout_prob': 0.3,
            'num_models': 3,
            'batch_size': 16,
            'lr': 0.001,
            'final_lr': 0.0001,
            'epochs': 5  # Small value for faster tests
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


class TestDataframeToMixedDataset:
    """Tests for dataframe_to_mixed_dataset."""
    
    def test_dataframe_to_mixed_dataset_categorical_target_rf(
        self, simple_dataset_spec, simple_rf_model_spec
    ):
        """Test conversion of dataframe with categorical target (RF)."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.5, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_rf_model_spec
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
        assert dataset.model_spec == simple_rf_model_spec

    def test_dataframe_to_mixed_dataset_categorical_target_ann(
        self, simple_dataset_spec, simple_ann_model_spec
    ):
        """Test conversion of dataframe with categorical target (ANN)."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.5, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_ann_model_spec
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
        assert dataset.model_spec == simple_ann_model_spec

    def test_dataframe_to_mixed_dataset_numerical_target(
        self, simple_dataset_spec, simple_rf_numerical_model_spec
    ):
        """Test conversion of dataframe with numerical target (RF)."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.5, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_rf_numerical_model_spec
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
        self, simple_dataset_spec, simple_rf_model_spec
    ):
        """Test handling of missing values in dataframe."""
        df = pd.DataFrame({
            'var1': ['a', None, 'c'],
            'var2': ['low', 'medium', None],
            'var3': [1.0, None, 3.8]
        })

        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_rf_model_spec
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
        self, simple_dataset_spec, simple_rf_model_spec
    ):
        """Test missing column raises ValueError."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            # 'var2' column is missing
            'var3': [1.0, 2.5, 3.8]
        })

        with pytest.raises(ValueError, match="Variable 'var2' .* not found"):
            dataframe_to_mixed_dataset(df, simple_dataset_spec,
                                       simple_rf_model_spec)


class TestFitModel:
    """Tests for _fit_model function."""
    
    def test_fit_model_rf_classification(self, simple_dataset_spec,
                                         simple_rf_model_spec):
        """Test _fit_model with RandomForest classifier."""
        # Create a simple dataset
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.0, 3.0]
        })
        
        # Convert to MixedDataset
        dataset = dataframe_to_mixed_dataset(df, simple_dataset_spec, 
                                            simple_rf_model_spec)
        
        # Fit model
        model = _fit_model(dataset, simple_rf_model_spec, random_seed=42)
        
        # Check that the model has the expected type
        assert isinstance(model, RandomForestClassifier)

    def test_fit_model_rf_regression(self, simple_dataset_spec,
                                     simple_rf_numerical_model_spec):
        """Test _fit_model with RandomForest regressor."""
        # Create a simple dataset
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.0, 3.0]
        })
        
        # Convert to MixedDataset
        dataset = dataframe_to_mixed_dataset(df, simple_dataset_spec, 
                                            simple_rf_numerical_model_spec)
        
        # Fit model
        model = _fit_model(dataset, simple_rf_numerical_model_spec,
                           random_seed=42)
        
        # Check that the model has the expected type
        assert isinstance(model, RandomForestRegressor)

    def test_fit_model_ann_ensemble(self, simple_dataset_spec,
                                    simple_ann_model_spec):
        """Test _fit_model with ANN ensemble model."""
        # Create a simple dataset
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.0, 3.0]
        })
        
        # Convert to MixedDataset
        dataset = dataframe_to_mixed_dataset(df, simple_dataset_spec, 
                                            simple_ann_model_spec)
        
        # Use patch to avoid actual ANN training which is time-consuming
        with patch('mixalot.crossval.train_ann_ensemble') as mock_train:
            # Create a basic ensemble model to return
            ensemble_model = type('EnsembleModel', (), {
                'predict_prob': lambda self, features, device: torch.tensor([
                    [0.7, 0.2, 0.1], 
                    [0.3, 0.6, 0.1], 
                    [0.1, 0.3, 0.6]
                ])
            })()
            mock_train.return_value = ensemble_model
            
            # Fit model
            model = _fit_model(dataset, simple_ann_model_spec, random_seed=42)
            
            # Check that the correct training function was called
            mock_train.assert_called_once()
            mock_train.assert_called_with(dataset, simple_ann_model_spec, 42)
            
            # Check that the model has the expected predict_prob method
            assert hasattr(model, 'predict_prob')


class TestCalculateLosses:
    """Tests for _calculate_losses function."""
    
    def test_calculate_losses_rf_classification(self, simple_dataset_spec,
                                               simple_rf_model_spec):
        """Test _calculate_losses for classification with Random Forest."""
        # Create datasets for train and test
        train_df = pd.DataFrame({
            'var1': ['a', 'b', 'c', 'a', 'b'],
            'var2': ['low', 'medium', 'high', 'medium', 'low'],
            'var3': [1.0, 2.0, 3.0, 1.5, 2.5]
        })
        
        test_df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'high', 'medium'],
            'var3': [1.2, 2.2, 3.2]
        })
        
        # Convert to MixedDatasets
        train_dataset = dataframe_to_mixed_dataset(
            train_df, simple_dataset_spec, simple_rf_model_spec
        )
        test_dataset = dataframe_to_mixed_dataset(
            test_df, simple_dataset_spec, simple_rf_model_spec
        )
        
        # Fit model
        model = _fit_model(train_dataset, simple_rf_model_spec, random_seed=42)
        
        # Get test features and true values
        Xcat, Xord, Xnum, y = test_dataset.get_arrays()
        features = combine_features(Xcat, Xord, Xnum)
        true_values = test_dataset.y_data.cpu().numpy()
        
        # Calculate losses
        predictions, losses = _calculate_losses(
            model, features, true_values, is_classifier=True
        )
        
        # Check predictions and losses
        assert len(predictions) == len(test_df)
        assert len(losses) == len(test_df)
        
        # For classification, predictions should be arrays of probabilities
        assert isinstance(predictions[0], np.ndarray)
        assert predictions[0].ndim == 1
        assert np.all(predictions[0] >= 0) and np.all(predictions[0] <= 1)
        
        # Losses should be positive floats
        assert all(isinstance(loss, float) for loss in losses)
        assert all(loss >= 0 for loss in losses)

    def test_calculate_losses_rf_regression(self, simple_dataset_spec, 
                                           simple_rf_numerical_model_spec):
        """Test _calculate_losses for regression with Random Forest."""
        # Create datasets for train and test
        train_df = pd.DataFrame({
            'var1': ['a', 'b', 'c', 'a', 'b'],
            'var2': ['low', 'medium', 'high', 'medium', 'low'],
            'var3': [1.0, 2.0, 3.0, 1.5, 2.5]
        })
        
        test_df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'high', 'medium'],
            'var3': [1.2, 2.2, 3.2]
        })
        
        # Convert to MixedDatasets
        train_dataset = dataframe_to_mixed_dataset(
            train_df, simple_dataset_spec, simple_rf_numerical_model_spec
        )
        test_dataset = dataframe_to_mixed_dataset(
            test_df, simple_dataset_spec, simple_rf_numerical_model_spec
        )
        
        # Fit model
        model = _fit_model(train_dataset, simple_rf_numerical_model_spec, 
                          random_seed=42)
        
        # Get test features and true values
        Xcat, Xord, Xnum, y = test_dataset.get_arrays()
        features = combine_features(Xcat, Xord, Xnum)
        true_values = test_dataset.y_data.cpu().numpy()
        
        # Calculate losses
        predictions, losses = _calculate_losses(
            model, features, true_values, is_classifier=False
        )
        
        # Check predictions and losses
        assert len(predictions) == len(test_df)
        assert len(losses) == len(test_df)
        
        # For regression, predictions should be scalar values
        assert all(isinstance(pred, (float, np.float32, np.float64)) 
                  for pred in predictions)
        
        # Losses should be positive floats (squared errors)
        assert all(isinstance(loss, float) for loss in losses)
        assert all(loss >= 0 for loss in losses)
    

    def test_calculate_losses_ann_classification(self, simple_dataset_spec,
                                                 simple_ann_model_spec):
        """Test _calculate_losses for classification with ANN ensemble."""
        # Create an actual ensemble model with BasicAnn models
        ensemble_model = EnsembleTorchModel(
            num_models=3,
            lr=0.001,
            base_model_class=BasicAnn,
            num_x_var=2,  # Input features dimension
            num_cat=3,    # Number of categories in output
            hidden_sizes=[32, 16],
            dropout_prob=0.3
        )
        
        # Create sample features and true values
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        true_values = np.array([1, 2, 3])  # Classes 1, 2, 3
        
        # Calculate losses
        predictions, losses = _calculate_losses(
            ensemble_model, features, true_values, is_classifier=True
        )
        
        # Check predictions and losses
        assert len(predictions) == 3
        assert len(losses) == 3
        
        # For classification, predictions should be arrays of probabilities
        assert isinstance(predictions[0], np.ndarray)
        assert predictions[0].ndim == 1
        assert np.all(predictions[0] >= 0) and np.all(predictions[0] <= 1)
        
        # Losses should be positive floats
        assert all(isinstance(loss, float) for loss in losses)
        assert all(loss >= 0 for loss in losses)


class TestProcessFoldObservations:
    """Tests for _process_fold_observations function."""
    
    def test_process_fold_observations_rf_classification(
        self, simple_dataset_spec, simple_rf_model_spec
    ):
        """Test _process_fold_observations for classification with RF."""
        # Create a simple dataset
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c', 'a', 'b'],
            'var2': ['low', 'medium', 'high', 'medium', 'low'],
            'var3': [1.0, 2.0, 3.0, 1.5, 2.5]
        })
        
        # Convert to MixedDataset
        dataset = dataframe_to_mixed_dataset(df, simple_dataset_spec, 
                                            simple_rf_model_spec)
        
        # Fit model
        model = _fit_model(dataset, simple_rf_model_spec, random_seed=42)
        
        # Process observations for fold 0
        fold_idx = 0
        sample_type = 'test'
        indices = np.array([0, 1, 2])
        
        results = _process_fold_observations(
            model,
            dataset,
            indices,
            fold_idx,
            sample_type,
            df.index,
            is_classifier=True
        )
        
        # Check result structure
        assert len(results) == len(indices)
        
        for result in results:
            assert 'original_index' in result
            assert 'fold' in result and result['fold'] == fold_idx
            assert 'sample_type' in result and result['sample_type'] ==\
                sample_type
            assert 'actual_value' in result
            assert 'loss' in result and result['loss'] >= 0
            
            # Classification specific fields
            assert 'pred_class' in result
            assert any(key.startswith('pred_prob_class_') for key in result)

    def test_process_fold_observations_rf_regression(
        self, simple_dataset_spec, simple_rf_numerical_model_spec
    ):
        """Test _process_fold_observations for regression with RF."""
        # Create a simple dataset
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c', 'a', 'b'],
            'var2': ['low', 'medium', 'high', 'medium', 'low'],
            'var3': [1.0, 2.0, 3.0, 1.5, 2.5]
        })
        
        # Convert to MixedDataset
        dataset = dataframe_to_mixed_dataset(
            df, simple_dataset_spec, simple_rf_numerical_model_spec
        )
        
        # Fit model
        model = _fit_model(dataset, simple_rf_numerical_model_spec,
                           random_seed=42)
        
        # Process observations for fold 0
        fold_idx = 0
        sample_type = 'train'
        indices = np.array([0, 1, 2])
        
        results = _process_fold_observations(
            model,
            dataset,
            indices,
            fold_idx,
            sample_type,
            df.index,
            is_classifier=False
        )
        
        # Check result structure
        assert len(results) == len(indices)
        
        for result in results:
            assert 'original_index' in result
            assert 'fold' in result and result['fold'] == fold_idx
            assert 'sample_type' in result and result['sample_type'] ==\
                sample_type
            assert 'actual_value' in result
            assert 'loss' in result and result['loss'] >= 0
            
            # Regression specific fields
            assert 'prediction' in result
            assert not any(key.startswith('pred_prob_class_')
                           for key in result)

    def test_process_fold_observations_ann_classification(
            self, simple_dataset_spec, simple_ann_model_spec
        ):
            """Test _process_fold_obs. for classif. with ANN ensemble."""
            # Create a simple dataset
            df = pd.DataFrame({
                'var1': ['a', 'b', 'c'],
                'var2': ['low', 'medium', 'high'],
                'var3': [1.0, 2.0, 3.0]
            })
            
            # Convert to MixedDataset
            dataset = dataframe_to_mixed_dataset(df, simple_dataset_spec, 
                                                simple_ann_model_spec)
            
            # Create an actual ensemble model with BasicAnn models
            ensemble_model = EnsembleTorchModel(
                num_models=3,
                lr=0.001,
                base_model_class=BasicAnn,
                num_x_var=3,  # accounts for missing data dimension
                num_cat=3,    # Number of categories (matches our dataset)
                hidden_sizes=[32, 16],
                dropout_prob=0.3
            )
            
            # Process observations for fold 0
            fold_idx = 0
            sample_type = 'test'
            indices = np.array([0, 1, 2])
            
            # No need to patch _calculate_losses, let it use the real
            # implementation
            results = _process_fold_observations(
                ensemble_model,
                dataset,
                indices,
                fold_idx,
                sample_type,
                df.index,
                is_classifier=True
            )
            
            # Check result structure
            assert len(results) == len(indices)
            
            for result in results:
                assert 'original_index' in result
                assert 'fold' in result and result['fold'] == fold_idx
                assert 'sample_type' in result and result['sample_type'] ==\
                    sample_type
                assert 'actual_value' in result
                assert 'loss' in result and result['loss'] >= 0
                
                # Classification specific fields
                assert 'pred_class' in result
                assert any(
                    key.startswith('pred_prob_class_') for key in result
                )


class TestValidateCVInputs:
    """Tests for _validate_cv_inputs function."""
    
    def test_validate_cv_inputs_rf(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec
    ):
        """Test validation of inputs with Random Forest model spec."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.0, 3.0]
        })
        
        # This should not raise an error
        _validate_cv_inputs(df, simple_dataset_spec, simple_cv_spec, 
                           simple_rf_model_spec)
        
        # Test with missing variable
        bad_model_spec = RandomForestSpec(
            y_var='var1',
            independent_vars=['var2', 'missing_var'],  # non-existent variable
            hyperparameters=simple_rf_model_spec.hyperparameters
        )
        
        with pytest.raises(ValueError) as excinfo:
            _validate_cv_inputs(df, simple_dataset_spec, simple_cv_spec, 
                               bad_model_spec)
        
        assert "not found in dataset_spec" in str(excinfo.value)
    
    def test_validate_cv_inputs_ann(
        self, simple_dataset_spec, simple_ann_model_spec, simple_cv_spec
    ):
        """Test validation of inputs with ANN ensemble model spec."""
        df = pd.DataFrame({
            'var1': ['a', 'b', 'c'],
            'var2': ['low', 'medium', 'high'],
            'var3': [1.0, 2.0, 3.0]
        })
        
        # This should not raise an error
        _validate_cv_inputs(df, simple_dataset_spec, simple_cv_spec, 
                           simple_ann_model_spec)
        
        # Test with missing variable
        bad_model_spec = ANNEnsembleSpec(
            y_var='var1',
            independent_vars=['var2', 'missing_var'],  # non-existent variable
            hyperparameters=simple_ann_model_spec.hyperparameters
        )
        
        with pytest.raises(ValueError) as excinfo:
            _validate_cv_inputs(df, simple_dataset_spec, simple_cv_spec, 
                               bad_model_spec)
        
        assert "not found in dataset_spec" in str(excinfo.value)


class TestRunCrossValidation:
    """Tests for the run_cross_validation function."""
    
    def test_run_cross_validation_records_all_observations_rf(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path
    ):
        """Test that all observations are recorded with RF model."""
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
                simple_rf_model_spec,
                random_seed=42,
                output_folder=str(tmp_path)
            )
        
        # Check that observation_results contains all samples
        assert 'observation_results' in results
        obs_df = results['observation_results']
        
        # DataFrame should exist
        assert not obs_df.empty
        
        # DataFrame should have expected columns
        expected_cols = [
            'original_index', 'fold', 'sample_type', 'actual_value', 'loss'
        ]
        for col in expected_cols:
            assert col in obs_df.columns
        
        # Should have records for both 'train' and 'test' sample types
        assert 'train' in obs_df['sample_type'].values
        assert 'test' in obs_df['sample_type'].values
        
        # Should have the right total number of records
        # For 2 folds, each sample should appear once as train and once as test
        expected_records = len(simple_dataframe) * simple_cv_spec.n_splits
        assert len(obs_df) == expected_records
        
        # Output file should exist
        assert os.path.exists(os.path.join(tmp_path, "all_observations.csv"))

    def test_run_cross_validation_ann_ensemble(
            self, simple_dataset_spec, simple_ann_model_spec, simple_cv_spec,
            simple_dataframe, tmp_path
        ):
            """Test running cross-validation with an ANN ensemble model."""
            # Configure ANN model spec with minimal parameters for faster tests
            simple_ann_model_spec.hyperparameters['num_models'] = 2
            simple_ann_model_spec.hyperparameters['epochs'] = 3
            simple_ann_model_spec.hyperparameters['batch_size'] = 4
            
            # Patch the fold indices to ensure consistent splits
            with patch.object(simple_cv_spec, 'get_fold_indices') as mock_indices:
                # Create splits where all categories are represented in both sets
                mock_indices.side_effect = [
                    # First fold: rows 0-3 train, 4-7 test
                    (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
                    # Second fold: rows 4-7 train, 0-3 test
                    (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3]))
                ]
                
                # Run cross-validation with our fixed _calculate_losses function
                results = run_cross_validation(
                    simple_dataframe,
                    simple_dataset_spec,
                    simple_cv_spec,
                    simple_ann_model_spec,
                    random_seed=42,
                    output_folder=str(tmp_path),
                    overwrite_files=True  # Ensure fresh run
                )
                
                # Verify results
                assert 'observation_results' in results
                obs_df = results['observation_results']
                
                # DataFrame should exist with data
                assert not obs_df.empty
                
                # Check essential columns
                expected_cols = [
                    'original_index', 'fold', 'sample_type', 
                    'actual_value', 'loss', 'pred_class'
                ]
                for col in expected_cols:
                    assert col in obs_df.columns
                
                # Check probability columns
                prob_cols = [c for c in obs_df.columns 
                            if c.startswith('pred_prob_class_')]
                assert len(prob_cols) > 0
                
                # Should have both train and test samples
                assert 'train' in obs_df['sample_type'].values
                assert 'test' in obs_df['sample_type'].values
                
                # Should have all expected records
                expected_records = len(simple_dataframe) * simple_cv_spec.n_splits
                assert len(obs_df) == expected_records
                
                # Check loss type and statistics
                assert results['loss_type'] == 'log_loss'
                assert 'test_avg_loss' in results
                assert 'train_avg_loss' in results
                
                # Check fold completion
                assert results['n_folds_completed'] == simple_cv_spec.n_splits
                
                # Output file should exist
                assert os.path.exists(os.path.join(tmp_path, "all_observations.csv"))
   
    def test_run_cross_validation_saves_individual_losses(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path
    ):
        """Test that individual observation losses are saved."""
        # Run cross-validation
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Check that individual observation losses were calculated
        obs_df = results['observation_results']
        assert 'loss' in obs_df.columns
        
        # All loss values should be non-negative
        assert (obs_df['loss'] >= 0).all()
        
        # Check output file
        output_file = os.path.join(tmp_path, "all_observations.csv")
        assert os.path.exists(output_file)
        
        # Load and check the file contents
        saved_df = pd.read_csv(output_file)
        assert 'loss' in saved_df.columns
        assert 'sample_type' in saved_df.columns
        assert set(saved_df['sample_type'].unique()) == {'train', 'test'}
    
    def test_run_cross_validation_computes_average_losses(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test computation of average in-sample and out-of-sample losses."""
        # Run cross-validation without output folder
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42
        )
        
        # Check average losses
        assert 'test_avg_loss' in results
        assert 'train_avg_loss' in results
        
        # Average losses should match manual calculation from observation
        # results
        obs_df = results['observation_results']
        test_obs = obs_df[obs_df['sample_type'] == 'test']
        train_obs = obs_df[obs_df['sample_type'] == 'train']
        
        test_avg = test_obs['loss'].mean()
        train_avg = train_obs['loss'].mean()
        
        assert results['test_avg_loss'] == pytest.approx(test_avg)
        assert results['train_avg_loss'] == pytest.approx(train_avg)
    
    def test_run_cross_validation_records_predictions(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test that predictions are recorded for each observation."""
        # Run cross-validation
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42
        )
        
        # Check observation results for predictions
        obs_df = results['observation_results']
        
        # For classification, should have pred_class and pred_prob columns
        if simple_rf_model_spec.y_var == 'var1':  # Classification
            # Should have predicted class
            assert 'pred_class' in obs_df.columns
            
            # Should have probability columns
            prob_cols = [col for col in obs_df.columns 
                        if col.startswith('pred_prob_class_')]
            assert len(prob_cols) > 0
        else:  # Regression
            # Should have prediction column
            assert 'prediction' in obs_df.columns
    
    def test_run_cross_validation_stores_original_indices(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe
    ):
        """Test that original indices are preserved in results."""
        # Create a dataframe with non-sequential indices
        df = simple_dataframe.copy()
        df.index = [100, 101, 102, 103, 104, 105, 106, 107]
        
        # Run cross-validation
        results = run_cross_validation(
            df,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42
        )
        
        # Check original indices are preserved
        obs_df = results['observation_results']
        assert 'original_index' in obs_df.columns
        
        # All original indices from the dataframe should be present
        obs_indices = set(obs_df['original_index'].unique())
        expected_indices = set(df.index)
        assert obs_indices == expected_indices
    
    def test_run_cross_validation_loads_existing_results(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test loading of existing results with overwrite_files=False."""
        # First run to create initial files
        run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Mock _fit_model to track calls
        original_fit_model = _fit_model
        mock_calls = []
        
        def mock_fit(*args, **kwargs):
            mock_calls.append((args, kwargs))
            return original_fit_model(*args, **kwargs)
        
        monkeypatch.setattr('mixalot.crossval._fit_model', mock_fit)
        
        # Run again with overwrite_files=False (default)
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Check that _fit_model was not called (results were loaded)
        assert len(mock_calls) == 0
        
        # Results should still be complete
        assert 'observation_results' in results
        assert not results['observation_results'].empty
    
    def test_run_cross_validation_handles_fold_errors(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test that errors in individual folds are handled gracefully."""
        # Mock _fit_model to fail on the first fold
        original_fit_model = _fit_model
        
        def mock_fit(*args, **kwargs):
            # Get fold_idx from the calling context (hacky but effective)
            frame = inspect.currentframe()
            try:
                # Look for fold_idx in the caller's locals
                caller_locals = frame.f_back.f_locals
                if ('fold_idx' in caller_locals and
                    caller_locals['fold_idx'] == 0):
                    raise ValueError("Simulated error in first fold")
                return original_fit_model(*args, **kwargs)
            finally:
                del frame  # Avoid reference cycles
        
        monkeypatch.setattr('mixalot.crossval._fit_model', mock_fit)
        
        # Run cross-validation
        results = run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Should still have results, but for fewer folds
        assert results['n_folds_completed'] < simple_cv_spec.n_splits
        assert 'observation_results' in results
        
        # Should only have results for the successful fold(s)
        obs_df = results['observation_results']
        assert 0 not in obs_df['fold'].unique()
    

    def test_run_cross_validation_rerun_folds(
        self, simple_dataset_spec, simple_rf_model_spec, simple_cv_spec,
        simple_dataframe, tmp_path, monkeypatch
    ):
        """Test that rerun_folds=True forces recomputation of all folds."""
        # First run to create initial files
        run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path)
        )
        
        # Mock _fit_model to track calls
        original_fit_model = _fit_model
        mock_calls = []
        
        def mock_fit(*args, **kwargs):
            mock_calls.append((args, kwargs))
            return original_fit_model(*args, **kwargs)
        
        monkeypatch.setattr('mixalot.crossval._fit_model', mock_fit)
        
        # Run again with rerun_folds=True
        run_cross_validation(
            simple_dataframe,
            simple_dataset_spec,
            simple_cv_spec,
            simple_rf_model_spec,
            random_seed=42,
            output_folder=str(tmp_path),
            rerun_folds=True
        )
        
        # _fit_model should be called for each fold
        assert len(mock_calls) == simple_cv_spec.n_splits
