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


class TestRandomForestSpec:
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


if __name__ == "__main__":
    pytest.main()
