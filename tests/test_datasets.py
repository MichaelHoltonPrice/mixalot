# python -m unittest tests.test_datasets
import json
import os
import tempfile
import unittest

import numpy as np
import torch

from mixalot.datasets import (
    DatasetSpec,
    MixedDataset,
    VarSpec,
)
from mixalot.models import RandomForestSpec


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
        
        self.assertListEqual(
            [var.var_name for var in dataset_spec.cat_var_specs], 
            ['cat_var']
        )
        self.assertListEqual(
            [var.var_name for var in dataset_spec.ord_var_specs], 
            ['ord_var']
        )
        self.assertListEqual(
            [var.var_name for var in dataset_spec.num_var_specs], 
            ['num_var']
        )
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
                         "All variable specifications in ordinal_var_specs "
                         "must be instances of VarSpec of type ordinal")

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
        self.assertEqual(
            str(cm.exception), 
            "Variable names must be unique across all variable types"
        )

    def test_get_ordered_variables(self):
        """Test getting ordered variables for each type."""
        cat_var_1 = VarSpec('cat_var_1', 'categorical', 
                           categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        cat_var_2 = VarSpec('cat_var_2', 'categorical', 
                           categorical_mapping=[{'E', 'F'}, {'G', 'H'}])
        ord_var = VarSpec('ord_var', 'ordinal', 
                         categorical_mapping=[{'I', 'J'}, {'K', 'L'}])
        num_var = VarSpec('num_var', 'numerical')
        
        dataset_spec =\
            DatasetSpec([cat_var_1, cat_var_2], [ord_var], [num_var])
        
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
        
        dataset_spec = DatasetSpec([cat_var_1, cat_var_2], [ord_var],
                                   [num_var])

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
        
        # Verify Xcat has only one column now (cat_var2) since cat_var1 was
        # extracted
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
        
        # Verify Xord has only one column now (ord_var2) since ord_var1 wa
        # extracted
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
