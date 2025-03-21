# python -m unittest tests.test_datasets
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from mixalot.datasets import (
    DatasetSpec,
    VarSpec,
)
from mixalot.models import RandomForestSpec
from mixalot.parsers import (
    convert_categories_to_codes,
    extract_dependent_variable,
    load_mixed_data,
    load_model_spec,
    parse_numeric_variable,
    scale_numerical_variables,
)


class TestConvertCategoriesToCodes(unittest.TestCase):
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
        with self.assertRaises(ValueError) as cm:
            convert_categories_to_codes(data, var_spec)
        self.assertEqual(str(cm.exception),
                         "Variable test_var contains unobserved category D")


class TestParseNumericVariable(unittest.TestCase):
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
        with self.assertRaises(ValueError) as cm:
            parse_numeric_variable(data, var_spec)
        self.assertEqual(str(cm.exception),
                         ("Invalid entry invalid for variable test_var cannot "
                          "be converted to float")
                        )

class TestScaleNumericalVariables(unittest.TestCase):
    """Test the scale_numerical_variables method."""
    def test_scale_numerical_variables(self):
        """Test with numerical data."""
        # Single column of data
        Xnum = np.array([[1.0], [2.0], [3.0]])
        num_scalers = scale_numerical_variables(Xnum)
        self.assertEqual(len(num_scalers), 1)
        np.testing.assert_array_almost_equal(Xnum,
                                             np.array([[-1.22474487],
                                                       [0.0],
                                                       [1.22474487]]))

        # Multiple columns of data
        Xnum = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        num_scalers = scale_numerical_variables(Xnum)
        self.assertEqual(len(num_scalers), 2)
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
        self.assertEqual(len(num_scalers), 0)


class TestExtractDependentVariable(unittest.TestCase):
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
        
        with self.assertRaises(ValueError) as cm:
            Xcat, Xord, Xnum, y = extract_dependent_variable(
                dataset_spec, model_spec, Xcat, Xord, Xnum
            )
            
        expected_error = (
            "This method should not be called if model_spec does not have "
            "a valid y_var"
        )
        self.assertEqual(str(cm.exception), expected_error)
        
        # Case 2: model_spec.y_var is None
        model_spec = MagicMock()
        model_spec.y_var = None
        
        with self.assertRaises(ValueError) as cm:
            Xcat, Xord, Xnum, y = extract_dependent_variable(
                dataset_spec, model_spec, Xcat, Xord, Xnum
            )
            
        self.assertEqual(str(cm.exception), expected_error)


class TestLoadModelSpec(unittest.TestCase):
    """Test cases for the load_model_spec function."""
    
    def create_temp_model_spec_file(self, model_spec_dict):
        """Create a temporary JSON file with the given model specification."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(temp_file.name, 'w') as f:
            json.dump(model_spec_dict, f)
        temp_file.close()
        return temp_file.name
    
    def test_load_random_forest_spec(self):
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
        model_spec_file = self.create_temp_model_spec_file(model_spec_dict)
        
        # Load the model specification
        model_spec = load_model_spec(model_spec_file)
        
        # Verify the model specification
        self.assertIsInstance(model_spec, RandomForestSpec)
        self.assertEqual(model_spec.model_type, 'random_forest')
        self.assertEqual(model_spec.y_var, 'target')
        self.assertEqual(model_spec.independent_vars, 
                         ['feature1', 'feature2', 'feature3'])
        self.assertEqual(model_spec.hyperparameters['n_estimators'], 100)
        self.assertEqual(model_spec.hyperparameters['max_features']['type'],
                         'int')
        self.assertEqual(model_spec.hyperparameters['max_features']['value'],
                         2)
        
        # Clean up
        os.unlink(model_spec_file)
    
    def test_missing_model_type(self):
        """Test error when model_type is missing."""
        # Create a temporary model specification file with missing model_type
        model_spec_dict = {
            'y_var': 'target',
            'independent_vars': ['feature1', 'feature2']
        }
        model_spec_file = self.create_temp_model_spec_file(model_spec_dict)
        
        # Verify that an error is raised
        with self.assertRaises(ValueError) as cm:
            load_model_spec(model_spec_file)
        
        self.assertEqual(str(cm.exception), 
                         "Model specification is missing 'model_type' field")
        
        # Clean up
        os.unlink(model_spec_file)
    
    def test_unsupported_model_type(self):
        """Test error when model_type is not supported."""
        # Create a temporary model specification file with
        # unsupported model_type
        model_spec_dict = {
            'model_type': 'unsupported_model',
            'y_var': 'target',
            'independent_vars': ['feature1', 'feature2']
        }
        model_spec_file = self.create_temp_model_spec_file(model_spec_dict)
        
        # Verify that an error is raised
        with self.assertRaises(ValueError) as cm:
            load_model_spec(model_spec_file)
        
        self.assertEqual(str(cm.exception),
                         "Unrecognized model type: unsupported_model")
        
        # Clean up
        os.unlink(model_spec_file)
    
    def test_file_not_found(self):
        """Test error when model specification file does not exist."""
        # Verify that an error is raised for non-existent file
        with self.assertRaises(FileNotFoundError) as cm:
            load_model_spec("non_existent_file.json")
        
        self.assertEqual(
            str(cm.exception),
            "Model specification file 'non_existent_file.json' does not exist."
        )


class TestLoadMixedData(unittest.TestCase):
    """Test cases for the load_mixed_data function."""
    
    def create_temp_data_file(self, data_dict, file_type='csv'):
        """Create a temporary data file with the given data."""
        data_df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(suffix=f'.{file_type}',
                                                delete=False)
        if file_type == 'csv':
            data_df.to_csv(temp_file.name, index=False)
        elif file_type in ['xlsx', 'xls']:
            data_df.to_excel(temp_file.name, index=False)
        temp_file.close()  # Manually close the file
        return temp_file.name

    def create_temp_dataset_spec_file(self, dataset_spec_dict):
        """Create a temporary dataset specification file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
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
    
    def create_temp_model_spec_file(self, model_spec_dict):
        """Create a temporary model specification file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(temp_file.name, 'w') as f:
            json.dump(model_spec_dict, f)
        temp_file.close()
        return temp_file.name

    def test_load_mixed_data_without_model_spec(self):
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        # Define the expected output
        expected_Xcat = np.array([[1, 1], [1, 1], [2, 2]])
        expected_Xnum = np.array([[-1.2247449], [0.], [1.2247449]])

        for file_type in ['csv', 'xlsx']:
            # Create and load data file for each file type
            data_file = self.create_temp_data_file(data_dict, 
                                                   file_type=file_type)
            mixed_dataset, num_scalers = load_mixed_data(dataset_spec_file, 
                                                         data_file)
    
            # Assert that the actual output matches the expected output
            np.testing.assert_array_almost_equal(
                mixed_dataset.Xcat.cpu().numpy(), 
                expected_Xcat
            )
            self.assertIsNone(mixed_dataset.Xord)
            np.testing.assert_array_almost_equal(
                mixed_dataset.Xnum.cpu().numpy(), 
                expected_Xnum
            )
            self.assertIsNone(mixed_dataset.y_data)
            self.assertIsNone(mixed_dataset.model_spec)
    
            # Assert that num_scalers has the correct length
            self.assertEqual(len(num_scalers), 1)
    
            # Clean up temporary files
            os.unlink(data_file)
        os.unlink(dataset_spec_file)

    def test_load_mixed_data_with_model_spec(self):
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
        
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
        model_spec_file = self.create_temp_model_spec_file(model_spec_dict)
    
        # Define the expected output after A is extracted as y_var
        expected_Xcat = np.array([[1], [1], [2]])  # Just B column
        expected_Xnum = np.array([[-1.2247449], [0.], [1.2247449]])
        expected_y_data = np.array([1, 1, 2])  # A column values

        # Create and load data file
        data_file = self.create_temp_data_file(data_dict, file_type='csv')
        mixed_dataset, num_scalers = load_mixed_data(
            dataset_spec_file, 
            data_file, 
            model_spec_file
        )

        # Assert that the actual output matches the expected output
        np.testing.assert_array_almost_equal(mixed_dataset.Xcat.cpu().numpy(), 
                                             expected_Xcat)
        self.assertIsNone(mixed_dataset.Xord)
        np.testing.assert_array_almost_equal(mixed_dataset.Xnum.cpu().numpy(), 
                                             expected_Xnum)
        
        # Assert that y_data contains the A column
        np.testing.assert_array_almost_equal(
            mixed_dataset.y_data.cpu().numpy(), 
            expected_y_data
        )
        
        # Assert that the model_spec is correctly loaded
        self.assertIsNotNone(mixed_dataset.model_spec)
        self.assertEqual(mixed_dataset.model_spec.model_type, 'random_forest')
        self.assertEqual(mixed_dataset.model_spec.y_var, 'A')
        self.assertEqual(mixed_dataset.model_spec.independent_vars, ['B', 'C'])

        # Clean up temporary files
        os.unlink(data_file)
        os.unlink(dataset_spec_file)
        os.unlink(model_spec_file)

    def test_load_mixed_data_unsupported_file_type(self):
        """Test error when file type is unsupported."""
        # Create a file with unsupported type and test if ValueError is raised.
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
        data_file = self.create_temp_data_file(data_dict, 
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(ValueError) as cm:
            _ , _ = load_mixed_data(dataset_spec_file, 
                                    data_file)
        expected_error = "Unsupported file type"
        self.assertEqual(str(cm.exception), expected_error)

        os.unlink(data_file)
        os.unlink(dataset_spec_file)

    def test_load_mixed_data_invalid_specification(self):
        """Test error when data does not match the dataset specification."""
        # Create a data file where 'A' column contains values not in mapping
        data_dict = {
            'A': ['x', 'y', 'z'],  # Not in categorical_mapping
            'B': ['d', 'e', 'f'], 
            'C': [1, 2, 3]
        }
        data_file = self.create_temp_data_file(data_dict, file_type='csv')
    
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(ValueError) as cm:
            _ , _ = load_mixed_data(dataset_spec_file, 
                                    data_file)
        expected_error = "Variable A contains unobserved category x"
        self.assertEqual(str(cm.exception), expected_error)
    
        os.unlink(data_file)
        os.unlink(dataset_spec_file)
    
    def test_load_mixed_data_file_not_found(self):
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(FileNotFoundError) as cm:
            _ , _ = load_mixed_data(
                dataset_spec_file, 
                "non_existent_file.csv"
            )
        expected_error = "Data file 'non_existent_file.csv' does not exist."
        self.assertEqual(str(cm.exception), expected_error)
    
        os.unlink(dataset_spec_file)
    
    def test_model_spec_file_not_found(self):
        """Test error when model specification file does not exist."""
        # Create temporary data and dataset spec files
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'],
                     'C': [1, 2, 3]}
        data_file = self.create_temp_data_file(data_dict, file_type='csv')
        
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
        dataset_spec_file =\
            self.create_temp_dataset_spec_file(dataset_spec_dict)
        
        # Test with non-existent model spec file
        with self.assertRaises(FileNotFoundError) as cm:
            _ , _ = load_mixed_data(
                dataset_spec_file,
                data_file,
                "non_existent_model_spec.json"
            )
        
        expected_error = (
            "Model specification file 'non_existent_model_spec.json' does "
            "not exist."
        )
        self.assertEqual(str(cm.exception), expected_error)
        
        # Clean up
        os.unlink(data_file)
        os.unlink(dataset_spec_file)


if __name__ == "__main__":
    unittest.main()
