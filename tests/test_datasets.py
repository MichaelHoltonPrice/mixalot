# python -m unittest tests.test_datasets
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

from mixalot.datasets import (
    convert_categories_to_codes,
    DatasetSpec,
    extract_dependent_variable,
    load_mixed_data,
    load_model_spec,
    MixedDataset,
    parse_numeric_variable,
    RandomForestSpec,
    scale_numerical_variables,
    VarSpec,
)


class TestVarSpec(unittest.TestCase):
    """Tests for VarSpec class."""

    def test_valid_inputs(self):
        # Test valid inputs
        cat_var = VarSpec('cat_var',
                          'categorical',
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}],
                          column_name='other_name')
        self.assertEqual(cat_var.var_name, 'cat_var')
        self.assertEqual(cat_var.var_type, 'categorical')
        self.assertEqual(cat_var.categorical_mapping, [{'A', 'B'}, {'C', 'D'}])
        self.assertIsNone(cat_var.missing_values)
        self.assertEqual(cat_var.column_name, 'other_name')

        ord_var = VarSpec('ord_var',
                          'ordinal',
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}],
                          missing_values=['Also NA'])
        self.assertEqual(ord_var.var_name, 'ord_var')
        self.assertEqual(ord_var.var_type, 'ordinal')
        self.assertEqual(ord_var.categorical_mapping, [{'A', 'B'}, {'C', 'D'}])
        self.assertEqual(ord_var.missing_values, ['Also NA'])
        self.assertIsNone(ord_var.column_name)

        num_var = VarSpec('num_var',
                          'numerical')
        self.assertEqual(num_var.var_name, 'num_var')
        self.assertEqual(num_var.var_type, 'numerical')
        self.assertIsNone(num_var.categorical_mapping)
        self.assertIsNone(num_var.missing_values)
        self.assertIsNone(num_var.column_name)

    def test_invalid_inputs(self):
        # Test invalid inputs
        with self.assertRaises(ValueError) as cm:
            invalid_var_type = VarSpec('inv_var', 'invalid')
        self.assertEqual(
            str(cm.exception),
            ("Invalid 'type' field for variable inv_var. Expected "
             "'numerical', 'categorical', or 'ordinal'")
        )

        with self.assertRaises(ValueError) as cm:
            missing_mapping = VarSpec('cat_var', 'categorical')
        self.assertEqual(
            str(cm.exception),
            ("Missing 'categorical_mapping' field for variable cat_var of "
            "type categorical")
        )

        with self.assertRaises(ValueError) as cm:
            _ = VarSpec('cat_var',
                        'categorical',
                        categorical_mapping=[{'A', 'B'}, {'B', 'C'}])
            raise ValueError(
                f"Some values appear in more than one set for variable cat_var"
            )


class TestDatasetSpec(unittest.TestCase):
    """Tests for the DatasetSpec class."""

    def test_valid_inputs(self):
        """Test DatasetSpec with valid inputs."""
        cat_var = VarSpec('cat_var', 'categorical', 
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', 
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')
        
        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var])
        
        self.assertListEqual([var.var_name for var in dataset_spec.cat_var_specs], 
                             ['cat_var'])
        self.assertListEqual([var.var_name for var in dataset_spec.ord_var_specs], 
                             ['ord_var'])
        self.assertListEqual([var.var_name for var in dataset_spec.num_var_specs], 
                             ['num_var'])
        self.assertSetEqual(dataset_spec.all_var_names, 
                            {'cat_var', 'ord_var', 'num_var'})

    def test_all_var_names_property(self):
        """Test that all_var_names is properly calculated as a property."""
        cat_var = VarSpec('cat_var', 'categorical', 
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', 
                          categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        
        # Initial dataset with two variables
        dataset_spec = DatasetSpec([cat_var], [ord_var], [])
        self.assertSetEqual(dataset_spec.all_var_names, {'cat_var', 'ord_var'})
        
        # Add a new variable
        num_var = VarSpec('num_var', 'numerical')
        dataset_spec.num_var_specs.append(num_var)
        
        # Verify all_var_names property updates
        self.assertSetEqual(dataset_spec.all_var_names, 
                            {'cat_var', 'ord_var', 'num_var'})

    def test_invalid_inputs(self):
        """Test DatasetSpec with invalid inputs."""
        # Test with variable of wrong type
        wrong_var = VarSpec('wrong_var', 'categorical', [{'A', 'B'}])
        with self.assertRaises(ValueError) as cm:
            invalid_dataset_spec = DatasetSpec([], [wrong_var], [])
        self.assertEqual(str(cm.exception), 
                         "All variable specifications in ordinal_var_specs must "
                         "be instances of VarSpec of type ordinal")

        # Test with empty dataset spec
        with self.assertRaises(ValueError) as cm:
            empty_dataset_spec = DatasetSpec([], [], [])
        self.assertEqual(str(cm.exception), 
                         "At least one of cat_var_specs, ord_var_specs, or "
                         "num_var_specs must be non-empty")

        # Test with duplicate variable names
        cat_var1 = VarSpec('duplicate', 'categorical', [{'A', 'B'}])
        cat_var2 = VarSpec('duplicate', 'categorical', [{'C', 'D'}])
        with self.assertRaises(ValueError) as cm:
            duplicate_dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [])
        self.assertEqual(str(cm.exception), 
                         "Variable names must be unique across all variable types")

    def test_get_ordered_variables(self):
        """Test getting ordered variables for each type."""
        cat_var_1 = VarSpec('cat_var_1', 'categorical', 
                           categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        cat_var_2 = VarSpec('cat_var_2', 'categorical', 
                           categorical_mapping=[{'E', 'F'}, {'G', 'H'}])
        ord_var = VarSpec('ord_var', 'ordinal', 
                         categorical_mapping=[{'I', 'J'}, {'K', 'L'}])
        num_var = VarSpec('num_var', 'numerical')
        
        dataset_spec = DatasetSpec([cat_var_1, cat_var_2], [ord_var], [num_var])
        
        self.assertListEqual(dataset_spec.get_ordered_variables('categorical'), 
                             ['cat_var_1', 'cat_var_2'])
        self.assertListEqual(dataset_spec.get_ordered_variables('ordinal'), 
                             ['ord_var'])
        self.assertListEqual(dataset_spec.get_ordered_variables('numerical'), 
                             ['num_var'])
        
        # Test with invalid type
        with self.assertRaises(ValueError) as cm:
            dataset_spec.get_ordered_variables('invalid_type')
        self.assertEqual(str(cm.exception), 
                         "Invalid 'type' field. Expected 'numerical', "
                         "'categorical', or 'ordinal'")

    def test_get_var_spec(self):
        """Test retrieving a variable specification by name."""
        cat_var_1 = VarSpec('cat_var_1', 'categorical', 
                           categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        cat_var_2 = VarSpec('cat_var_2', 'categorical', 
                           categorical_mapping=[{'E', 'F'}, {'G', 'H'}])
        ord_var = VarSpec('ord_var', 'ordinal', 
                         categorical_mapping=[{'I', 'J'}, {'K', 'L'}])
        num_var = VarSpec('num_var', 'numerical')
        
        dataset_spec = DatasetSpec([cat_var_1, cat_var_2], [ord_var], [num_var])

        # Check that correct VarSpec is returned for valid variable names
        self.assertEqual(dataset_spec.get_var_spec('cat_var_1'), cat_var_1)
        self.assertEqual(dataset_spec.get_var_spec('ord_var'), ord_var)
        self.assertEqual(dataset_spec.get_var_spec('num_var'), num_var)

        # Check that ValueError is raised for invalid variable name
        with self.assertRaises(ValueError) as cm:
            dataset_spec.get_var_spec('nonexistent_var')
        self.assertEqual(str(cm.exception), 
                         "Variable name nonexistent_var is not found in the "
                         "provided variable specifications")

    def test_from_json(self):
        """Test creating a DatasetSpec from a JSON file."""
        # Prepare a dict that matches the expected structure of the json file
        dataset_spec_dict = {
            "cat_var_specs": [
                {
                    "var_name": "categorical_var_1",
                    "var_type": "categorical",
                    "categorical_mapping": [["a", "b", "c"], ["d", "e", "f"]],
                    "missing_values": "NA",
                    "column_name": "cat_1"
                }
            ],
            "ord_var_specs": [
                {
                    "var_name": "ordinal_var_1",
                    "var_type": "ordinal",
                    "categorical_mapping": [["1", "2", "3"], ["4", "5", "6"]],
                    "missing_values": "NA",
                    "column_name": "ord_1"
                }
            ],
            "num_var_specs": [
                {
                    "var_name": "numerical_var_1",
                    "var_type": "numerical",
                    "missing_values": "NA",
                    "column_name": "num_1"
                }
            ]
        }

        # Write the dict to a temporary json file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                         suffix='.json') as tmp:
            json.dump(dataset_spec_dict, tmp)
            tempname = tmp.name

        # Load the json file as a DatasetSpec object
        dataset_spec = DatasetSpec.from_json(tempname)

        # Validate the DatasetSpec object
        self.assertEqual(len(dataset_spec.cat_var_specs), 1)
        self.assertEqual(len(dataset_spec.ord_var_specs), 1)
        self.assertEqual(len(dataset_spec.num_var_specs), 1)
        
        # Also check some attributes of the first VarSpec in each list
        self.assertEqual(dataset_spec.cat_var_specs[0].var_name, 
                         "categorical_var_1")
        self.assertEqual(dataset_spec.ord_var_specs[0].var_name, 
                         "ordinal_var_1")
        self.assertEqual(dataset_spec.num_var_specs[0].var_name, 
                         "numerical_var_1")
        
        # Clean up the temporary file
        os.unlink(tempname)


class TestMixedDataset(unittest.TestCase):
    """Tests for the MixedDataset class."""

    def test_init_with_model_spec(self):
        """Test initialization using y_var from model_spec."""
        # Create variable specifications
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var1 = VarSpec(
            'ord_var1', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var2 = VarSpec(
            'ord_var2', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')

        # Create dataset specification without y_var
        dataset_spec = DatasetSpec(
            [cat_var1, cat_var2], 
            [ord_var1, ord_var2], 
            [num_var]
        )

        # Create model specification with num_var as y_var
        model_spec = RandomForestSpec(
            y_var='num_var',
            independent_vars=['cat_var1', 'cat_var2', 'ord_var1', 'ord_var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )

        # Create test data
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        Xord = np.array([[2, 2], [1, 1]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        expected_y_data = np.array([9, 10], dtype=np.float32)

        # Test with CPU device
        device = torch.device('cpu')
        mixed_dataset = MixedDataset(
            dataset_spec,
            Xcat=Xcat,
            Xord=Xord,
            Xnum=Xnum,
            model_spec=model_spec,
            device=device
        )

        # Verify dataset properties
        self.assertEqual(mixed_dataset.dataset_spec, dataset_spec)
        self.assertEqual(mixed_dataset.model_spec, model_spec)
        
        # Verify data tensors
        self.assertTrue(torch.all(
            mixed_dataset.Xcat == torch.tensor(Xcat, device=device)
        ))
        self.assertTrue(torch.all(
            mixed_dataset.Xord == torch.tensor(Xord, device=device)
        ))
        self.assertIsNone(mixed_dataset.Xnum)
        self.assertTrue(torch.all(
            mixed_dataset.y_data == torch.tensor(expected_y_data,
                                                 device=device)
        ))

    def test_init_categorical_target(self):
        """Test extracting a categorical target variable."""
        # Create variable specifications
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var = VarSpec(
            'ord_var', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')

        # Create dataset spec
        dataset_spec = DatasetSpec(
            [cat_var1, cat_var2], 
            [ord_var], 
            [num_var]
        )
        
        # Create model spec with categorical target
        model_spec = RandomForestSpec(
            y_var='cat_var1',
            independent_vars=['cat_var2', 'ord_var', 'num_var'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )

        # Create test data
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        Xord = np.array([[2], [1]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        
        # Expected y_data should be from cat_var1
        expected_y_data = np.array([1, 2], dtype=np.int32)

        # Initialize dataset
        mixed_dataset = MixedDataset(
            dataset_spec,
            Xcat=Xcat,
            Xord=Xord,
            Xnum=Xnum,
            model_spec=model_spec
        )
        
        # Verify target variable is correct
        self.assertTrue(torch.all(
            mixed_dataset.y_data == torch.tensor(expected_y_data)
        ))
        
        # Verify Xcat has only one column now (cat_var2) since cat_var1 was extracted
        self.assertEqual(mixed_dataset.Xcat.shape, (2, 1))
        self.assertTrue(torch.all(
            mixed_dataset.Xcat == torch.tensor([[2], [1]])
        ))

    def test_init_ordinal_target(self):
        """Test extracting an ordinal target variable."""
        # Create variable specifications
        cat_var = VarSpec(
            'cat_var', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var1 = VarSpec(
            'ord_var1', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var2 = VarSpec(
            'ord_var2', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')

        # Create dataset spec
        dataset_spec = DatasetSpec(
            [cat_var], 
            [ord_var1, ord_var2], 
            [num_var]
        )
        
        # Create model spec with ordinal target
        model_spec = RandomForestSpec(
            y_var='ord_var1',
            independent_vars=['cat_var', 'ord_var2', 'num_var'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )

        # Create test data
        Xcat = np.array([[1], [2]], dtype=np.int32)
        Xord = np.array([[2, 3], [1, 4]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        
        # Expected y_data should be from ord_var1
        expected_y_data = np.array([2, 1], dtype=np.int32)

        # Initialize dataset
        mixed_dataset = MixedDataset(
            dataset_spec,
            Xcat=Xcat,
            Xord=Xord,
            Xnum=Xnum,
            model_spec=model_spec
        )
        
        # Verify target variable is correct
        self.assertTrue(torch.all(
            mixed_dataset.y_data == torch.tensor(expected_y_data)
        ))
        
        # Verify Xord has only one column now (ord_var2) since ord_var1 was extracted
        self.assertEqual(mixed_dataset.Xord.shape, (2, 1))
        self.assertTrue(torch.all(
            mixed_dataset.Xord == torch.tensor([[3], [4]])
        ))

    def test_len(self):
        """Test the length calculation of the dataset."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [])

        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)

        # Test with default aug_mult=1
        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat)
        self.assertEqual(len(mixed_dataset), 2)

        # Test with aug_mult=3
        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, aug_mult=3)
        self.assertEqual(len(mixed_dataset), 6)

    def test_getitem(self):
        """Test retrieving items from the dataset."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var1 = VarSpec(
            'ord_var1', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var2 = VarSpec(
            'ord_var2', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')
    
        dataset_spec = DatasetSpec(
            [cat_var1, cat_var2], 
            [ord_var1, ord_var2], 
            [num_var]
        )
        
        model_spec = RandomForestSpec(
            y_var='num_var',
            independent_vars=['cat_var1', 'cat_var2', 'ord_var1', 'ord_var2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )
    
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        Xord = np.array([[2, 2], [1, 1]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        expected_y_data = torch.tensor([9], dtype=torch.float32)
    
        mixed_dataset = MixedDataset(
            dataset_spec, 
            Xcat=Xcat, 
            Xord=Xord, 
            Xnum=Xnum,
            model_spec=model_spec
        )
        
        # Get the first item
        x_cat, x_ord, x_num, m_num, y = mixed_dataset[0]
        
        # Verify item components
        self.assertTrue(torch.all(x_cat == torch.tensor([1, 2],
                                                        dtype=torch.long)))
        self.assertTrue(torch.all(x_ord == torch.tensor([2, 2],
                                                        dtype=torch.long)))
        self.assertIsNone(x_num)
        self.assertIsNone(m_num)
        self.assertTrue(torch.all(y == expected_y_data))

    def test_inconsistent_rows(self):
        """Test error when input arrays have inconsistent row counts."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var1 = VarSpec(
            'ord_var1', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var2 = VarSpec(
            'ord_var2', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')

        dataset_spec = DatasetSpec(
            [cat_var1, cat_var2], 
            [ord_var1, ord_var2], 
            [num_var]
        )

        # Create arrays with inconsistent row counts
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)  # 2 rows
        Xord = np.array([[2, 2], [1, 1]], dtype=np.int32)  # 2 rows
        Xnum = np.array([[9]], dtype=np.float32)          # 1 row

        with self.assertRaises(ValueError) as cm:
            mixed_dataset = MixedDataset(
                dataset_spec, 
                Xcat=Xcat, 
                Xord=Xord, 
                Xnum=Xnum
            )
        self.assertEqual(
            str(cm.exception), 
            "Input arrays do not have the same number of samples"
        )

    def test_inconsistent_columns(self):
        """Test error when input arrays have inconsistent column counts."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var1 = VarSpec(
            'ord_var1', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var2 = VarSpec(
            'ord_var2', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')

        # Dataset spec expects 1 categorical variable
        dataset_spec = DatasetSpec(
            [cat_var1], 
            [ord_var1, ord_var2], 
            [num_var]
        )

        # But Xcat has 2 columns (variables)
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        Xord = np.array([[2, 2], [1, 1]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)

        with self.assertRaises(ValueError) as cm:
            mixed_dataset = MixedDataset(
                dataset_spec, 
                Xcat=Xcat, 
                Xord=Xord, 
                Xnum=Xnum
            )
        self.assertEqual(
            str(cm.exception), 
            "Xcat has 2 columns but dataset_spec has 1 categorical variables"
        )

    def test_get_arrays_with_target(self):
        """Test get_arrays method when a target variable is defined."""
        cat_var1 = VarSpec('cat1', 'categorical', [{'A','B'}])  
        num_var1 = VarSpec('num1', 'numerical')   
        num_var2 = VarSpec('num2', 'numerical')
    
        dataset_spec = DatasetSpec([cat_var1], [], [num_var1, num_var2])
    
        model_spec = RandomForestSpec(
            y_var='num1',
            independent_vars=['cat1', 'num2'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )
    
        Xcat = np.random.randint(1, 3, (10, 1), dtype=np.int32)  
        Xnum = np.random.randn(10, 2).astype(np.float32)
    
        dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xnum=Xnum, 
                              model_spec=model_spec)
    
        x_cat, x_ord, x_num, y = dataset.get_arrays()
    
        # Verify that arrays match (with num1 removed from Xnum)
        np.testing.assert_array_almost_equal(x_cat.cpu().numpy(), Xcat)
        self.assertIsNone(x_ord)
        np.testing.assert_array_almost_equal(x_num.cpu().numpy(), Xnum[:,1:])  
        np.testing.assert_array_almost_equal(y.cpu().numpy(),
                                             dataset.y_data.cpu().numpy())
    
    def test_get_arrays_no_target(self):
        """Test get_arrays method when no target variable is defined."""
        cat_var1 = VarSpec('cat1', 'categorical', [{'A','B'}])
        cat_var2 = VarSpec('cat2', 'categorical', [{'C','D'}])
    
        num_var1 = VarSpec('num1', 'numerical')
        num_var2 = VarSpec('num2', 'numerical') 
        num_var3 = VarSpec('num3', 'numerical')
    
        dataset_spec = DatasetSpec(
            [cat_var1, cat_var2], 
            [], 
            [num_var1, num_var2, num_var3]
        )
    
        Xcat = np.random.randint(1, 3, (10, 2), dtype=np.int32)
        Xnum = np.random.randn(10, 3).astype(np.float32)
    
        dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xnum=Xnum)
    
        x_cat, x_ord, x_num, y = dataset.get_arrays()
    
        np.testing.assert_array_almost_equal(x_cat.cpu().numpy(), Xcat)
        self.assertIsNone(x_ord)
        np.testing.assert_array_almost_equal(x_num.cpu().numpy(), Xnum)
        self.assertIsNone(y)

    def test_aug_mult(self):
        """Test data augmentation multiplier."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [])

        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)

        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, aug_mult=3)
        self.assertEqual(len(mixed_dataset), 6)

    def test_mask_prob(self):
        """Test masking probability."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [])
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        
        # Set mask probability to 0.2
        mixed_dataset = MixedDataset(
            dataset_spec, 
            Xcat=Xcat, 
            mask_prob=0.2, 
            aug_mult=1
        )
    
        # Run many samples to verify mask probability is close to expected
        num_samples = 10000
        mask_fractions = []
    
        for _ in range(num_samples):
            item = mixed_dataset[0]
            x_cat = item[0]
    
            # If there are unmasked elements, ensure they match original values
            if (x_cat != 0).any():
                self.assertTrue(np.all(
                    (x_cat[x_cat != 0].cpu().numpy() == 
                     Xcat[0, x_cat != 0])
                ))
            
            # Record the fraction of masked values
            mask_fractions.append((x_cat == 0).float().mean().item())
    
        avg_mask_fraction = np.mean(mask_fractions)
    
        # Verify mask fraction is close to expected probability
        self.assertTrue(
            np.abs(avg_mask_fraction - mixed_dataset.mask_prob) < 0.01
        )

    def test_require_input(self):
        """Test the require_input functionality."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        ord_var = VarSpec(
            'ord_var', 'ordinal', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')
    
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [ord_var], [num_var])
        
        model_spec = RandomForestSpec(
            y_var='ord_var',
            independent_vars=['cat_var1', 'cat_var2', 'num_var'],
            hyperparameters={
                'n_estimators': 100,
                'max_features': {'type': 'int', 'value': 2}
            }
        )
    
        # Create random data
        Xcat = np.random.randint(1, 3, (100, 2), dtype=np.int32)
        Xord = np.random.randint(1, 3, (100, 1), dtype=np.int32)
        Xnum = np.random.random((100, 1)).astype(np.float32)
    
        # Create dataset with high mask probability but require_input=True
        mixed_dataset = MixedDataset(
            dataset_spec, 
            Xcat=Xcat, 
            Xord=Xord, 
            Xnum=Xnum,
            model_spec=model_spec,
            mask_prob=0.99, 
            require_input=True
        )
    
        # Verify that every sample has at least one unmasked variable
        for i in range(len(mixed_dataset)):
            sample = mixed_dataset[i]
            sample_Xcat, sample_Xord, sample_Xnum, sample_Mnum, _ = sample
            all_masked = True
    
            # Check if any categorical variables are unmasked
            self.assertIsNotNone(sample_Xcat)
            if np.any(sample_Xcat.cpu().numpy() != 0):
                all_masked = False

            # Ordinal should be None as it was the target
            self.assertIsNone(sample_Xord)

            # Check if any numerical variables are unmasked using the mask
            self.assertIsNotNone(sample_Xnum)
            if np.any(sample_Mnum.cpu().numpy() == True):
                all_masked = False
    
            # Verify that not all variables are masked
            self.assertFalse(all_masked)
    
    def test_no_y_var(self):
        """Test that dataset works without any y_var specified."""
        cat_var1 = VarSpec(
            'cat_var1', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        cat_var2 = VarSpec(
            'cat_var2', 'categorical', 
            categorical_mapping=[{'A', 'B'}, {'C', 'D'}]
        )
        num_var = VarSpec('num_var', 'numerical')
    
        # Create dataset spec without y_var
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [num_var])
        
        # No model_spec provided
        Xcat = np.array([[1, 2], [2, 1]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        
        # Initialize dataset without model_spec
        mixed_dataset = MixedDataset(
            dataset_spec,
            Xcat=Xcat,
            Xnum=Xnum
        )
        
        # Verify no target variable was extracted
        self.assertIsNone(mixed_dataset.y_data)
        
        # Verify data tensors are intact
        self.assertTrue(torch.all(
            mixed_dataset.Xcat == torch.tensor(Xcat)
        ))
        self.assertTrue(torch.all(
            mixed_dataset.Xnum == torch.tensor(Xnum, dtype=torch.float)
        ))


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
        var_spec = VarSpec('test_var', 'numerical', missing_values=missing_values)
        expected_output = np.array([1.2, 3.4, np.nan, 5.6, np.nan, 7.8, np.nan])
        np.testing.assert_array_equal(parse_numeric_variable(data, var_spec), expected_output)

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
        ord_var_spec1 = VarSpec("ord_var1", "ordinal", [{'0', '1', '2', '3', '4'}])
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
        self.assertEqual(model_spec.hyperparameters['max_features']['type'], 'int')
        self.assertEqual(model_spec.hyperparameters['max_features']['value'], 2)
        
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
        # Create a temporary model specification file with unsupported model_type
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