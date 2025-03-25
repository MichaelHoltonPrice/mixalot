"""Tests for model specification classes."""
import json
import os
import tempfile

import pytest

from mixalot.datasets import DatasetSpec, VarSpec
from mixalot.models import RandomForestSpec


@pytest.fixture
def sample_dataset_spec():
    """Create a sample dataset specification for testing.
    
    Returns:
        DatasetSpec: A sample dataset specification with categorical, ordinal,
            and numerical variables.
    """
    cat_var_specs = [
        VarSpec('cat_var1', 'categorical', 
                categorical_mapping=[{'yes'}, {'no'}]),
        VarSpec('cat_var2', 'categorical', 
                categorical_mapping=[{'a'}, {'b'}, {'c'}])
    ]
    
    ord_var_specs = [
        VarSpec('ord_var1', 'ordinal', 
                categorical_mapping=[{'low'}, {'medium'}, {'high'}])
    ]
    
    num_var_specs = [
        VarSpec('num_var1', 'numerical'),
        VarSpec('num_var2', 'numerical'),
        VarSpec('num_var3', 'numerical')
    ]
    
    return DatasetSpec(cat_var_specs, ord_var_specs, num_var_specs)


@pytest.fixture
def valid_rf_spec():
    """Create a valid RandomForestSpec for testing.
    
    Returns:
        RandomForestSpec: A valid random forest model specification.
    """
    return RandomForestSpec(
        y_var='cat_var1',
        independent_vars=['cat_var2', 'ord_var1', 'num_var1', 'num_var2'],
        hyperparameters={
            'n_estimators': 100,
            'max_features': {'type': 'float', 'value': 0.7}
        }
    )


class TestRandomForestSpec():
    """Test cases for the RandomForestSpec class."""
    
    def test_model_type(self, valid_rf_spec):
        """Test that the model_type property returns the correct value."""
        assert valid_rf_spec.model_type == 'random_forest'
    
    def test_init_with_valid_parameters(self):
        """Test that RandomForestSpec can be initialized with valid params."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        assert spec.y_var == 'target'
        assert spec.independent_vars == ['var1', 'var2']
        assert spec.hyperparameters == {
            'n_estimators': 100,
            'max_features': {'type': 'float', 'value': 0.7}
        }
    
    def test_validate_structure_with_empty_y_var(self):
        """Test that validation fails when y_var is empty."""
        with pytest.raises(ValueError) as excinfo:
            RandomForestSpec(
                y_var='',
                independent_vars=['var1', 'var2'],
                hyperparameters={
                    'n_estimators': 100,
                    'max_features': {'type': 'float', 'value': 0.7}
                }
            )
        assert "Dependent variable (y_var) must be specified"\
            in str(excinfo.value)
    
    def test_validate_structure_with_empty_independent_vars(self):
        """Test that validation fails when independent_vars is empty."""
        with pytest.raises(ValueError) as excinfo:
            RandomForestSpec(
                y_var='target',
                independent_vars=[],
                hyperparameters={
                    'n_estimators': 100,
                    'max_features': {'type': 'float', 'value': 0.7}
                }
            )
        assert "At least one independent variable must be specified"\
            in str(excinfo.value)
    
    def test_validate_structure_with_y_var_in_independent_vars(self):
        """Test that validation fails when y_var is in independent_vars."""
        with pytest.raises(ValueError) as excinfo:
            RandomForestSpec(
                y_var='target',
                independent_vars=['var1', 'target'],
                hyperparameters={
                    'n_estimators': 100,
                    'max_features': {'type': 'float', 'value': 0.7}
                }
            )
        assert "cannot be used as both dependent and independent"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_missing_n_estimators(self):
        """Test hyperparameter validation fails with missing n_estimators."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "n_estimators is required" in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_n_estimators(self):
        """Test validation fails with invalid n_estimators."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': -10,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "n_estimators must be a positive integer" in str(excinfo.value)
    
    def test_validate_hyperparameters_with_missing_max_features(self):
        """Test validation fails with missing max_features."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "max_features is required" in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_max_features_type(self):
        """Test validation fails with invalid max_features type."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': 'sqrt'  # Not a dict with type and value
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "max_features must be an object with" in str(excinfo.value)
    
    def test_validate_hyperparameters_with_incomplete_max_features(self):
        """Test validation fails with incomplete max_features dict."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'float'}  # Missing value
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "max_features must have 'type' and 'value'"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_int_value(self):
        """Test validation fails with invalid int value in max_features."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': -5}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "must have a positive integer value" in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_float_value(self):
        """Test validation fails with invalid float value in max_features."""
        spec = RandomForestSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'float', 'value': 1.5}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "must have a value between 0 and 1" in str(excinfo.value)
    
    def test_validate_model_specific(self, valid_rf_spec, sample_dataset_spec):
        """Test model-specific validation with dataset spec."""
        # This should pass without error
        valid_rf_spec._validate_model_specific(sample_dataset_spec)
    
    def test_validate_with_dataset_spec(self, valid_rf_spec,
                                        sample_dataset_spec):
        """Test validation against a dataset specification."""
        # This should pass without error
        valid_rf_spec.validate_with_dataset_spec(sample_dataset_spec)
    
    def test_validate_with_missing_y_var(self, sample_dataset_spec):
        """Test validation fails with y_var not in dataset spec."""
        invalid_spec = RandomForestSpec(
            y_var='missing_var',
            independent_vars=['cat_var2', 'ord_var1', 'num_var1'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            invalid_spec.validate_with_dataset_spec(sample_dataset_spec)
        assert "Dependent variable 'missing_var' not found"\
            in str(excinfo.value)
    
    def test_validate_with_missing_independent_var(self, sample_dataset_spec):
        """Test validation fails with independent var not in dataset spec."""
        invalid_spec = RandomForestSpec(
            y_var='cat_var1',
            independent_vars=['cat_var2', 'missing_var', 'num_var1'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        )
        with pytest.raises(ValueError) as excinfo:
            invalid_spec.validate_with_dataset_spec(sample_dataset_spec)
        assert "Independent variable 'missing_var' not found"\
            in str(excinfo.value)
    
    def test_to_dict(self, valid_rf_spec):
        """Test conversion to dictionary."""
        result = valid_rf_spec.to_dict()
        expected = {
            'model_type': 'random_forest',
            'y_var': 'cat_var1',
            'independent_vars': [
                'cat_var2', 'ord_var1', 'num_var1', 'num_var2'
            ],
            'hyperparameters': {
                'n_estimators': 100,
                'max_features': {'type': 'float', 'value': 0.7}
            }
        }
        assert result == expected
    
    def test_to_json(self, valid_rf_spec):
        """Test saving to a JSON file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            valid_rf_spec.to_json(tmp_path)
            
            # Verify the file exists and can be read
            assert os.path.exists(tmp_path)
            
            with open(tmp_path, 'r') as f:
                saved_data = json.load(f)
            
            expected = {
                'model_type': 'random_forest',
                'y_var': 'cat_var1',
                'independent_vars': [
                    'cat_var2', 'ord_var1', 'num_var1', 'num_var2'
                ],
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_features': {'type': 'float', 'value': 0.7}
                }
            }
            assert saved_data == expected
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

"""Tests for ANN Ensemble model specification."""
import json
import os
import tempfile

import pytest

from mixalot.datasets import DatasetSpec, VarSpec
from mixalot.models import ANNEnsembleSpec


@pytest.fixture
def valid_ann_ensemble_spec():
    """Create a valid ANNEnsembleSpec for testing.
    
    Returns:
        ANNEnsembleSpec: A valid ANN ensemble model specification.
    """
    return ANNEnsembleSpec(
        y_var='cat_var1',
        independent_vars=['cat_var2', 'ord_var1', 'num_var1', 'num_var2'],
        hyperparameters={
            'hidden_sizes': [32, 16],
            'dropout_prob': 0.5,
            'num_models': 5,
            'batch_size': 32,
            'lr': 0.001,
            'final_lr': 0.0001,
            'epochs': 100
        }
    )


class TestANNEnsembleSpec():
    """Test cases for the ANNEnsembleSpec class."""
    
    def test_model_type(self, valid_ann_ensemble_spec):
        """Test that the model_type property returns the correct value."""
        assert valid_ann_ensemble_spec.model_type == 'ann_ensemble'
    
    def test_init_with_valid_parameters(self):
        """Test that ANNEnsembleSpec can be initialized with valid params."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [64, 32],
                'dropout_prob': 0.5,
                'num_models': 10,
                'batch_size': 64,
                'lr': 0.002,
                'final_lr': 0.0002,
                'epochs': 200
            }
        )
        assert spec.y_var == 'target'
        assert spec.independent_vars == ['var1', 'var2']
        assert spec.hyperparameters['hidden_sizes'] == [64, 32]
        assert spec.hyperparameters['dropout_prob'] == 0.5
        assert spec.hyperparameters['num_models'] == 10
        assert spec.hyperparameters['batch_size'] == 64
        assert spec.hyperparameters['lr'] == 0.002
        assert spec.hyperparameters['final_lr'] == 0.0002
        assert spec.hyperparameters['epochs'] == 200
    
    def test_validate_structure_with_empty_y_var(self):
        """Test that validation fails when y_var is empty."""
        with pytest.raises(ValueError) as excinfo:
            ANNEnsembleSpec(
                y_var='',
                independent_vars=['var1', 'var2'],
                hyperparameters={
                    'hidden_sizes': [32, 16],
                    'dropout_prob': 0.5,
                    'num_models': 5
                }
            )
        assert "Dependent variable (y_var) must be specified"\
            in str(excinfo.value)
    
    def test_validate_structure_with_empty_independent_vars(self):
        """Test that validation fails when independent_vars is empty."""
        with pytest.raises(ValueError) as excinfo:
            ANNEnsembleSpec(
                y_var='target',
                independent_vars=[],
                hyperparameters={
                    'hidden_sizes': [32, 16],
                    'dropout_prob': 0.5,
                    'num_models': 5
                }
            )
        assert "At least one independent variable must be specified"\
            in str(excinfo.value)
    
    def test_validate_structure_with_y_var_in_independent_vars(self):
        """Test that validation fails when y_var is in independent_vars."""
        with pytest.raises(ValueError) as excinfo:
            ANNEnsembleSpec(
                y_var='target',
                independent_vars=['var1', 'target'],
                hyperparameters={
                    'hidden_sizes': [32, 16],
                    'dropout_prob': 0.5,
                    'num_models': 5
                }
            )
        assert "cannot be used as both dependent and independent"\
            in str(excinfo.value)
    
    def test_validate_with_numerical_target(self, sample_dataset_spec):
        """Test validation fails with numerical target variable."""
        invalid_spec = ANNEnsembleSpec(
            y_var='num_var1',
            independent_vars=['cat_var1', 'ord_var1', 'num_var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            invalid_spec._validate_model_specific(sample_dataset_spec)
        assert "Target variable 'num_var1' must be categorical or ordinal"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_hidden_sizes(self):
        """Test validation fails with invalid hidden_sizes."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': "not_a_list",
                'dropout_prob': 0.5,
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "hidden_sizes must be a list of positive integers"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_hidden_size_values(self):
        """Test validation fails with invalid values in hidden_sizes."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, -16],
                'dropout_prob': 0.5,
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "hidden_sizes must be a list of positive integers"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_dropout_prob(self):
        """Test validation fails with invalid dropout_prob."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 1.5,  # Must be between 0 and 1
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "dropout_prob must be a float between 0 and 1"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_num_models(self):
        """Test validation fails with invalid num_models."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': -5  # Must be positive
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "num_models must be a positive integer"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_batch_size(self):
        """Test validation fails with invalid batch_size."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5,
                'batch_size': 0  # Must be positive
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "batch_size must be a positive integer"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_lr(self):
        """Test validation fails with invalid learning rate."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5,
                'lr': -0.001  # Must be positive
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "lr must be a positive float"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_final_lr(self):
        """Test validation fails with invalid final learning rate."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5,
                'lr': 0.001,
                'final_lr': "not_a_number"  # Must be a number
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "final_lr must be a positive float"\
            in str(excinfo.value)
    
    def test_validate_hyperparameters_with_invalid_epochs(self):
        """Test validation fails with invalid epochs."""
        spec = ANNEnsembleSpec(
            y_var='target',
            independent_vars=['var1', 'var2'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5,
                'epochs': -100  # Must be positive
            }
        )
        with pytest.raises(ValueError) as excinfo:
            spec.validate_hyperparameters()
        assert "epochs must be a positive integer"\
            in str(excinfo.value)
    
    def test_validate_with_dataset_spec(self, valid_ann_ensemble_spec,
                                        sample_dataset_spec):
        """Test validation against a dataset specification."""
        # This should pass without error
        valid_ann_ensemble_spec.validate_with_dataset_spec(sample_dataset_spec)
    
    def test_validate_with_missing_y_var(self, sample_dataset_spec):
        """Test validation fails with y_var not in dataset spec."""
        invalid_spec = ANNEnsembleSpec(
            y_var='missing_var',
            independent_vars=['cat_var2', 'ord_var1', 'num_var1'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            invalid_spec.validate_with_dataset_spec(sample_dataset_spec)
        assert "Dependent variable 'missing_var' not found"\
            in str(excinfo.value)
    
    def test_validate_with_missing_independent_var(self, sample_dataset_spec):
        """Test validation fails with independent var not in dataset spec."""
        invalid_spec = ANNEnsembleSpec(
            y_var='cat_var1',
            independent_vars=['cat_var2', 'missing_var', 'num_var1'],
            hyperparameters={
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5
            }
        )
        with pytest.raises(ValueError) as excinfo:
            invalid_spec.validate_with_dataset_spec(sample_dataset_spec)
        assert "Independent variable 'missing_var' not found"\
            in str(excinfo.value)
    
    def test_to_dict(self, valid_ann_ensemble_spec):
        """Test conversion to dictionary."""
        result = valid_ann_ensemble_spec.to_dict()
        expected = {
            'model_type': 'ann_ensemble',
            'y_var': 'cat_var1',
            'independent_vars': [
                'cat_var2', 'ord_var1', 'num_var1', 'num_var2'
            ],
            'hyperparameters': {
                'hidden_sizes': [32, 16],
                'dropout_prob': 0.5,
                'num_models': 5,
                'batch_size': 32,
                'lr': 0.001,
                'final_lr': 0.0001,
                'epochs': 100
            }
        }
        assert result == expected
    
    def test_to_json(self, valid_ann_ensemble_spec):
        """Test saving to a JSON file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            valid_ann_ensemble_spec.to_json(tmp_path)
            
            # Verify the file exists and can be read
            assert os.path.exists(tmp_path)
            
            with open(tmp_path, 'r') as f:
                saved_data = json.load(f)
            
            expected = {
                'model_type': 'ann_ensemble',
                'y_var': 'cat_var1',
                'independent_vars': [
                    'cat_var2', 'ord_var1', 'num_var1', 'num_var2'
                ],
                'hyperparameters': {
                    'hidden_sizes': [32, 16],
                    'dropout_prob': 0.5,
                    'num_models': 5,
                    'batch_size': 32,
                    'lr': 0.001,
                    'final_lr': 0.0001,
                    'epochs': 100
                }
            }
            assert saved_data == expected
            
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main()
