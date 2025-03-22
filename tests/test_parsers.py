"""Tests for parser functions."""
import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mixalot.datasets import (
    DatasetSpec,
    VarSpec,
)
from mixalot.crossval import CrossValidationFoldsSpec
from mixalot.models import RandomForestSpec
from mixalot.parsers import (
    convert_categories_to_codes,
    extract_dependent_variable,
    load_cross_validation_folds_spec_from_json,
    load_mixed_data,
    load_model_spec,
    parse_numeric_variable,
    scale_numerical_variables,
)


class TestConvertCategoriesToCodes:
    """Test the convert_categories_to_codes method."""

    def test_convert_categories_to_codes(self):
        """Test with categorical and ordinal data."""
        data = pd.Series(['A', 'B', 'A', np.nan, 'B', '', 'D', 'C', 'A'])
        categorical_mapping = [{'A'}, {'B', 'C'}, {'D'}]
        var_spec = VarSpec('test_var', 'categorical', categorical_mapping)
        expected_output = np.array([1, 2, 1, 0, 2, 0, 3, 2, 1])
        np.testing.assert_array_equal(
            convert_categories_to_codes(data, var_spec), expected_output
        )

        # Test with ordinal data and explicit additional missing values
        data = pd.Series(['Low', 'Medium', 'High', np.nan,
                          '', 'Medium', 'Low', 'NA1',
                          'High', 'NA2'])
        ordinal_mapping = [{'Low'}, {'Medium'}, {'High'}]
        missing_values = {'NA1', 'NA2'}
        var_spec = VarSpec('test_var',
                           'ordinal',
                           ordinal_mapping, missing_values=missing_values)
        expected_output = np.array([1, 2, 3, 0, 0, 2, 1, 0, 3, 0])
        np.testing.assert_array_equal(
            convert_categories_to_codes(
                data,
                var_spec
            ),
            expected_output
        )

    def test_unobserved_category(self):
        """Test with unobserved category."""
        data = pd.Series(['A', 'B', 'D'])
        categorical_mapping = [{'A'}, {'B'}, {'C'}]
        var_spec = VarSpec('test_var', 'categorical', categorical_mapping)
        with pytest.raises(ValueError) as excinfo:
            convert_categories_to_codes(data, var_spec)
        assert str(excinfo.value) ==\
            "Variable test_var contains unobserved category D"


class TestParseNumericVariable:
    """Test the parse_numeric_variable method."""

    def test_parse_numeric_variable(self):
        """Test with numerical data."""
        # Test with missing values represented by np.nan
        data = pd.Series(['1.2', '3.4', np.nan, '5.6', '', '7.8'])
        var_spec = VarSpec('test_var', 'numerical')
        expected_output = np.array([1.2, 3.4, np.nan, 5.6, np.nan, 7.8])
        np.testing.assert_array_equal(parse_numeric_variable(data, var_spec),
                                      expected_output)

        # Test with additional missing values
        data = pd.Series(['1.2', '3.4', 'NA', '5.6', '', '7.8', 'missing'])
        missing_values = {'NA', 'missing'}
        var_spec = VarSpec('test_var', 'numerical',
                           missing_values=missing_values)
        expected_output = np.array([1.2, 3.4, np.nan, 5.6, np.nan, 7.8,
                                    np.nan])
        np.testing.assert_array_equal(parse_numeric_variable(data, var_spec),
                                      expected_output)

    def test_invalid_numeric_entry(self):
        """Test with invalid numeric entry."""
        data = pd.Series(['1.2', '3.4', 'invalid', '5.6'])
        var_spec = VarSpec('test_var', 'numerical')
        with pytest.raises(ValueError) as excinfo:
            parse_numeric_variable(data, var_spec)
        assert str(excinfo.value) == (
            "Invalid entry invalid for variable test_var cannot "
            "be converted to float"
        )


class TestScaleNumericalVariables:
    """Test the scale_numerical_variables method."""
    def test_scale_numerical_variables(self):
        """Test with numerical data."""
        # Single column of data
        Xnum = np.array([[1.0], [2.0], [3.0]])
        num_scalers = scale_numerical_variables(Xnum)
        assert len(num_scalers) == 1
        np.testing.assert_array_almost_equal(Xnum,
                                            np.array([[-1.22474487],
                                                      [0.0],
                                                      [1.22474487]]))

        # Multiple columns of data
        Xnum = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        num_scalers = scale_numerical_variables(Xnum)
        assert len(num_scalers) == 2
        np.testing.assert_array_almost_equal(
            Xnum,
            np.array([[-1.22474487, -1.22474487],
                      [0.0, 0.0],
                      [1.22474487, 1.22474487]])
        )

    def test_empty_array(self):
        """Test with empty array."""
        # Empty array
        Xnum = np.empty((0, 0))
        num_scalers = scale_numerical_variables(Xnum)
        assert len(num_scalers) == 0


class TestExtractDependentVariable:
    """Test the extract_dependent_variable method."""
    def test_extract_dependent_variable(self):
        """Test with various inputs."""
        # Test 1: categorical y variable
        cat_var_spec1 = VarSpec("cat_var1", "categorical", [{'0', '1', '2'}])
        cat_var_spec2 = VarSpec("y_var", "categorical", [{'0', '1', '2', '3'}])
        ord_var_spec1 = VarSpec("ord_var1", "ordinal",
                                [{'0', '1', '2', '3', '4'}])
        num_var_spec1 = VarSpec("num_var1", "numerical")
        
        dataset_spec = DatasetSpec(
            [cat_var_spec1, cat_var_spec2],
            [ord_var_spec1], 
            [num_var_spec1]
        )
        
        # Create a mock model_spec with categorical y_var
        model_spec = MagicMock()
        model_spec.y_var = "y_var"
        
        Xcat = np.array([[1, 3], [2, 1], [0, 2], [1, 0]])
        Xord = np.array([[0], [3], [2], [1]])
        Xnum = np.array([[1.1], [2.2], [3.3], [4.4]])
        
        Xcat, Xord, Xnum, y = extract_dependent_variable(
            dataset_spec, model_spec, Xcat, Xord, Xnum
        )
        
        np.testing.assert_array_equal(Xcat, np.array([[1], [2], [0], [1]]))
        np.testing.assert_array_equal(y, np.array([3, 1, 2, 0]))

        # Test 2: ordinal y variable
        ord_var_spec2 = VarSpec("y_var", "ordinal",
                                [{'0', '1', '2', '3', '4'}])
        dataset_spec = DatasetSpec(
            [cat_var_spec1],
            [ord_var_spec1, ord_var_spec2],
            [num_var_spec1]
        )
        
        # Update mock model_spec with ordinal y_var
        model_spec.y_var = "y_var"
        
        Xord = np.array([[0, 3], [3, 1], [2, 2], [1, 4]])
        
        Xcat, Xord, Xnum, y = extract_dependent_variable(
            dataset_spec, model_spec, Xcat, Xord, Xnum
        )
        
        np.testing.assert_array_equal(Xord, np.array([[0], [3], [2], [1]]))
        np.testing.assert_array_equal(y, np.array([3, 1, 2, 4]))

        # Test 3: numerical y variable
        num_var_spec2 = VarSpec("y_var", "numerical")
        dataset_spec = DatasetSpec(
            [cat_var_spec1], 
            [ord_var_spec1],
            [num_var_spec1, num_var_spec2]
        )
        
        # Update mock model_spec with numerical y_var
        model_spec.y_var = "y_var"
        
        Xnum = np.array([[1.1, 2.2], [2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])
        
        Xcat, Xord, Xnum, y = extract_dependent_variable(
            dataset_spec, model_spec, Xcat, Xord, Xnum
        )
        
        np.testing.assert_array_equal(
            Xnum, np.array([[1.1], [2.2], [3.3], [4.4]])
        )
        np.testing.assert_array_equal(y, np.array([2.2, 3.3, 4.4, 5.5]))

    def test_no_y_variable(self):
        """Test error cases when no y variable is specified."""
        # Test 4: Error when no y variable
        cat_var_spec1 = VarSpec("cat_var1", "categorical",
                                [{'0', '1', '2'}])
        ord_var_spec1 = VarSpec("ord_var1", "ordinal",
                                [{'0', '1', '2', '3', '4'}])
        num_var_spec1 = VarSpec("num_var1", "numerical")
        
        dataset_spec = DatasetSpec(
            [cat_var_spec1], 
            [ord_var_spec1],
            [num_var_spec1]
        )
        
        # Case 1: model_spec is None
        model_spec = None
        
        Xcat = np.array([[1], [2], [0], [1]])
        Xord = np.array([[0], [3], [2], [1]])
        Xnum = np.array([[1.1], [2.2], [3.3], [4.4]])
        
        with pytest.raises(ValueError) as excinfo:
            Xcat, Xord, Xnum, y = extract_dependent_variable(
                dataset_spec, model_spec, Xcat, Xord, Xnum
            )
            
        expected_error = (
            "This method should not be called if model_spec does not have "
            "a valid y_var"
        )
        assert str(excinfo.value) == expected_error
        
        # Case 2: model_spec.y_var is None
        model_spec = MagicMock()
        model_spec.y_var = None
        
        with pytest.raises(ValueError) as excinfo:
            Xcat, Xord, Xnum, y = extract_dependent_variable(
                dataset_spec, model_spec, Xcat, Xord, Xnum
            )
            
        assert str(excinfo.value) == expected_error


class TestLoadModelSpec:
    """Test cases for the load_model_spec function."""
    
    @pytest.fixture
    def create_temp_model_spec_file(self):
        """Create a temporary JSON file with the given model specification."""
        def _create_file(model_spec_dict):
            temp_file = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
            with open(temp_file.name, 'w') as f:
                json.dump(model_spec_dict, f)
            temp_file.close()
            return temp_file.name
        return _create_file
    
    def test_load_random_forest_spec(self, create_temp_model_spec_file):
        """Test loading a RandomForestSpec from a JSON file."""
        # Create a temporary model specification file for RandomForest
        model_spec_dict = {
            'model_type': 'random_forest',
            'y_var': 'target',
            'independent_vars': ['feature1', 'feature2', 'feature3'],
            'hyperparameters': {
                'n_estimators': 100,
                'max_features': {
                    'type': 'int',
                    'value': 2
                }
            }
        }
        model_spec_file = create_temp_model_spec_file(model_spec_dict)
        
        try:
            # Load the model specification
            model_spec = load_model_spec(model_spec_file)
            
            # Verify the model specification
            assert isinstance(model_spec, RandomForestSpec)
            assert model_spec.model_type == 'random_forest'
            assert model_spec.y_var == 'target'
            assert model_spec.independent_vars == ['feature1', 'feature2',
                                                   'feature3']
            assert model_spec.hyperparameters['n_estimators'] == 100
            assert model_spec.hyperparameters['max_features']['type'] == 'int'
            assert model_spec.hyperparameters['max_features']['value'] == 2
        finally:
            # Clean up
            os.unlink(model_spec_file)
    
    def test_missing_model_type(self, create_temp_model_spec_file):
        """Test error when model_type is missing."""
        # Create a temporary model specification file with missing model_type
        model_spec_dict = {
            'y_var': 'target',
            'independent_vars': ['feature1', 'feature2']
        }
        model_spec_file = create_temp_model_spec_file(model_spec_dict)
        
        try:
            # Verify that an error is raised
            with pytest.raises(ValueError) as excinfo:
                load_model_spec(model_spec_file)
            
            assert str(excinfo.value) ==\
                "Model specification is missing 'model_type' field"
        finally:
            # Clean up
            os.unlink(model_spec_file)
    
    def test_unsupported_model_type(self, create_temp_model_spec_file):
        """Test error when model_type is not supported."""
        # Create a temporary model specification file with unsupported
        # model_type
        model_spec_dict = {
            'model_type': 'unsupported_model',
            'y_var': 'target',
            'independent_vars': ['feature1', 'feature2']
        }
        model_spec_file = create_temp_model_spec_file(model_spec_dict)
        
        try:
            # Verify that an error is raised
            with pytest.raises(ValueError) as excinfo:
                load_model_spec(model_spec_file)
            
            assert str(excinfo.value) ==\
                "Unrecognized model type: unsupported_model"
        finally:
            # Clean up
            os.unlink(model_spec_file)
    
    def test_file_not_found(self):
        """Test error when model specification file does not exist."""
        # Verify that an error is raised for non-existent file
        with pytest.raises(FileNotFoundError) as excinfo:
            load_model_spec("non_existent_file.json")
        
        assert str(excinfo.value) ==\
            "Model specification file 'non_existent_file.json' does not exist."


class TestLoadMixedData:
    """Test cases for the load_mixed_data function."""
    
    @pytest.fixture
    def create_temp_data_file(self):
        """Create a temporary data file with the given data."""
        def _create_file(data_dict, file_type='csv'):
            data_df = pd.DataFrame(data_dict)
            temp_file = tempfile.NamedTemporaryFile(suffix=f'.{file_type}',
                                                    delete=False)
            if file_type == 'csv':
                data_df.to_csv(temp_file.name, index=False)
            elif file_type in ['xlsx', 'xls']:
                data_df.to_excel(temp_file.name, index=False)
            temp_file.close()  # Manually close the file
            return temp_file.name
        return _create_file

    @pytest.fixture
    def create_temp_dataset_spec_file(self):
        """Create a temporary dataset specification file."""
        def _create_file(dataset_spec_dict):
            temp_file = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
            with open(temp_file.name, 'w') as f:
                # Convert the sets to lists to make them serializable
                for cat_var_spec in dataset_spec_dict.get('cat_var_specs', []):
                    cat_var_spec['categorical_mapping'] = [
                        list(item) for item in 
                        cat_var_spec.get('categorical_mapping', [])
                    ]
                for ord_var_spec in dataset_spec_dict.get('ord_var_specs', []):
                    ord_var_spec['categorical_mapping'] = [
                        list(item) for item in
                        ord_var_spec.get('categorical_mapping', [])
                    ]
                json.dump(dataset_spec_dict, f)
            temp_file.close()
            return temp_file.name
        return _create_file
    
    @pytest.fixture
    def create_temp_model_spec_file(self):
        """Create a temporary model specification file."""
        def _create_file(model_spec_dict):
            temp_file = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
            with open(temp_file.name, 'w') as f:
                json.dump(model_spec_dict, f)
            temp_file.close()
            return temp_file.name
        return _create_file

    def test_load_mixed_data_without_model_spec(self, create_temp_data_file,
                                                create_temp_dataset_spec_file):
        """Test loading mixed data without a model specification."""
        # Create temporary data file
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
    
        # Create temporary dataset specification file
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
    
        # Define the expected output
        expected_Xcat = np.array([[1, 1], [1, 1], [2, 2]])
        expected_Xnum = np.array([[-1.2247449], [0.], [1.2247449]])
        
        try:
            for file_type in ['csv', 'xlsx']:
                # Create and load data file for each file type
                data_file = create_temp_data_file(data_dict,
                                                  file_type=file_type)
                try:
                    mixed_dataset, num_scalers = load_mixed_data(
                        dataset_spec_file, 
                        data_file
                    )
            
                    # Assert that the actual output matches the expected output
                    np.testing.assert_array_almost_equal(
                        mixed_dataset.Xcat.cpu().numpy(), 
                        expected_Xcat
                    )
                    assert mixed_dataset.Xord is None
                    np.testing.assert_array_almost_equal(
                        mixed_dataset.Xnum.cpu().numpy(), 
                        expected_Xnum
                    )
                    assert mixed_dataset.y_data is None
                    assert mixed_dataset.model_spec is None
            
                    # Assert that num_scalers has the correct length
                    assert len(num_scalers) == 1
                finally:
                    # Clean up temporary files
                    os.unlink(data_file)
        finally:
            os.unlink(dataset_spec_file)

    def test_load_mixed_data_with_model_spec(self, create_temp_data_file, 
                                             create_temp_dataset_spec_file,
                                             create_temp_model_spec_file):
        """Test loading mixed data with a model specification."""
        # Create temporary data file
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
    
        # Create temporary dataset specification file
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
        
        # Create temporary model specification file
        model_spec_dict = {
            'model_type': 'random_forest',
            'y_var': 'A',
            'independent_vars': ['B', 'C'],
            'hyperparameters': {
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 1}
            }
        }
        model_spec_file = create_temp_model_spec_file(model_spec_dict)
    
        # Define the expected output after A is extracted as y_var
        expected_Xcat = np.array([[1], [1], [2]])  # Just B column
        expected_Xnum = np.array([[-1.2247449], [0.], [1.2247449]])
        expected_y_data = np.array([1, 1, 2])  # A column values
        
        try:
            # Create and load data file
            data_file = create_temp_data_file(data_dict, file_type='csv')
            try:
                mixed_dataset, num_scalers = load_mixed_data(
                    dataset_spec_file, 
                    data_file, 
                    model_spec_file
                )
        
                # Assert that the actual output matches the expected output
                np.testing.assert_array_almost_equal(
                    mixed_dataset.Xcat.cpu().numpy(), 
                    expected_Xcat
                )
                assert mixed_dataset.Xord is None
                np.testing.assert_array_almost_equal(
                    mixed_dataset.Xnum.cpu().numpy(), 
                    expected_Xnum
                )
                
                # Assert that y_data contains the A column
                np.testing.assert_array_almost_equal(
                    mixed_dataset.y_data.cpu().numpy(), 
                    expected_y_data
                )
                
                # Assert that the model_spec is correctly loaded
                assert mixed_dataset.model_spec is not None
                assert mixed_dataset.model_spec.model_type == 'random_forest'
                assert mixed_dataset.model_spec.y_var == 'A'
                assert mixed_dataset.model_spec.independent_vars == ['B', 'C']
            finally:
                # Clean up temporary file
                os.unlink(data_file)
        finally:
            # Clean up temporary files
            os.unlink(dataset_spec_file)
            os.unlink(model_spec_file)

    def test_load_mixed_data_unsupported_file_type(self, create_temp_data_file, 
                                               create_temp_dataset_spec_file):
        """Test error when file type is unsupported."""
        # Create a file with unsupported type and test if ValueError is raised.
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
        data_file = create_temp_data_file(data_dict, 
                                        file_type='txt')  # Unsupported
    
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
        
        try:
            with pytest.raises(ValueError) as excinfo:
                _ , _ = load_mixed_data(dataset_spec_file, 
                                      data_file)
            assert str(excinfo.value) == "Unsupported file type"
        finally:
            os.unlink(data_file)
            os.unlink(dataset_spec_file)

    def test_load_mixed_data_invalid_specification(
            self,
            create_temp_data_file, 
            create_temp_dataset_spec_file
        ):
        """Test error when data does not match the dataset specification."""
        # Create a data file where 'A' column contains values not in mapping
        data_dict = {
            'A': ['x', 'y', 'z'],  # Not in categorical_mapping
            'B': ['d', 'e', 'f'], 
            'C': [1, 2, 3]
        }
        data_file = create_temp_data_file(data_dict, file_type='csv')
    
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
        
        try:
            with pytest.raises(ValueError) as excinfo:
                _ , _ = load_mixed_data(dataset_spec_file, 
                                      data_file)
            assert str(excinfo.value) ==\
                "Variable A contains unobserved category x"
        finally:
            os.unlink(data_file)
            os.unlink(dataset_spec_file)
    
    def test_load_mixed_data_file_not_found(self,
                                            create_temp_dataset_spec_file):
        """Test error when data file does not exist."""
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
        
        try:
            with pytest.raises(FileNotFoundError) as excinfo:
                _ , _ = load_mixed_data(
                    dataset_spec_file, 
                    "non_existent_file.csv"
                )
            assert str(excinfo.value) ==\
                "Data file 'non_existent_file.csv' does not exist."
        finally:
            os.unlink(dataset_spec_file)

    def test_model_spec_file_not_found(
        self, 
        create_temp_data_file, 
        create_temp_dataset_spec_file
    ):
        """Test error when model specification file does not exist."""
        # Create temporary data and dataset spec files
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
        data_file = create_temp_data_file(data_dict, file_type='csv')
        
        dataset_spec_dict = {
            'cat_var_specs': [
                {
                    'var_name': 'A', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'a', 'b'}, {'c'}]
                },
                {
                    'var_name': 'B', 
                    'var_type': 'categorical', 
                    'categorical_mapping': [{'d', 'e'}, {'f'}]
                }
            ],
            'ord_var_specs': [],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = create_temp_dataset_spec_file(dataset_spec_dict)
        
        try:
            # Test with non-existent model spec file
            with pytest.raises(FileNotFoundError) as excinfo:
                _ , _ = load_mixed_data(
                    dataset_spec_file,
                    data_file,
                    "non_existent_model_spec.json"
                )
            
            expected_error = (
                "Model specification file 'non_existent_model_spec.json' "
                "does not exist."
            )
            assert str(excinfo.value) == expected_error
        finally:
            # Clean up
            os.unlink(data_file)
            os.unlink(dataset_spec_file)


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
