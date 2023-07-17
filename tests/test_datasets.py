# python -m unittest tests.test_datasets
import tempfile
import json
import unittest
from mixalot.datasets import DatasetSpec, VarSpec, MixedDataset
from mixalot.datasets import convert_categories_to_codes
from mixalot.datasets import parse_numeric_variable
from mixalot.datasets import scale_numerical_variables
from mixalot.datasets import extract_dependent_variable
from mixalot.datasets import load_mixed_data
import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TestVarSpec(unittest.TestCase):

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
        self.assertEqual(str(cm.exception), "Invalid 'type' field for variable inv_var. Expected 'numerical', 'categorical', or 'ordinal'")

        with self.assertRaises(ValueError) as cm:
            missing_mapping = VarSpec('cat_var', 'categorical')
        self.assertEqual(str(cm.exception), "Missing 'categorical_mapping' field for variable cat_var of type categorical")

        with self.assertRaises(ValueError) as cm:
            duplicate_mapping = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'B', 'C'}])
            raise ValueError(f"Some values appear in more than one set for variable cat_var")


class TestDatasetSpec(unittest.TestCase):

    def test_valid_inputs(self):
        # Test valid inputs
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')
        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var])
        self.assertListEqual([var.var_name for var in dataset_spec.cat_var_specs], ['cat_var'])
        self.assertListEqual([var.var_name for var in dataset_spec.ord_var_specs], ['ord_var'])
        self.assertListEqual([var.var_name for var in dataset_spec.num_var_specs], ['num_var'])
        self.assertSetEqual(dataset_spec.all_var_names, {'cat_var', 'ord_var', 'num_var'})

    def test_invalid_inputs(self):
        # Test invalid inputs
        wrong_var = VarSpec('wrong_var', 'categorical', [{'A', 'B'}])
        with self.assertRaises(ValueError) as cm:
            invalid_dataset_spec = DatasetSpec([], [wrong_var], [])
        self.assertEqual(str(cm.exception), "All variable specifications in ordinal_var_specs must be instances of VarSpec of type ordinal")

        with self.assertRaises(ValueError) as cm:
            empty_dataset_spec = DatasetSpec([], [], [])
        self.assertEqual(str(cm.exception), "At least one of cat_var_specs, ord_var_specs, or num_var_specs must be non-empty")

    def test_get_ordered_variables(self):
        # Test getting ordered variables
        cat_var_1 = VarSpec('cat_var_1', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        cat_var_2 = VarSpec('cat_var_2', 'categorical', categorical_mapping=[{'E', 'F'}, {'G', 'H'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'I', 'J'}, {'K', 'L'}])
        num_var = VarSpec('num_var', 'numerical')
        dataset_spec = DatasetSpec([cat_var_1, cat_var_2], [ord_var], [num_var])
        self.assertListEqual(dataset_spec.get_ordered_variables('categorical'), ['cat_var_1', 'cat_var_2'])
        self.assertListEqual(dataset_spec.get_ordered_variables('ordinal'), ['ord_var'])
        self.assertListEqual(dataset_spec.get_ordered_variables('numerical'), ['num_var'])

    def test_y_var(self):
        # Test valid y_var
        cat_var = VarSpec('cat_var', 'categorical', [{'A', 'B'}])
        ord_var = VarSpec('ord_var', 'ordinal', [{'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')
        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var], 'num_var')
        self.assertEqual(dataset_spec.y_var, 'num_var')

        # Test invalid y_var
        with self.assertRaises(ValueError) as cm:
            wrong_y_var_dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var], 'wrong_var')
        self.assertEqual(str(cm.exception), "y_var wrong_var is not found in the provided variable specifications")

    def test_get_var_spec(self):
        # Test getting VarSpec
        cat_var_1 = VarSpec('cat_var_1', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        cat_var_2 = VarSpec('cat_var_2', 'categorical', categorical_mapping=[{'E', 'F'}, {'G', 'H'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'I', 'J'}, {'K', 'L'}])
        num_var = VarSpec('num_var', 'numerical')
        dataset_spec = DatasetSpec([cat_var_1, cat_var_2], [ord_var], [num_var])

        # Check that correct VarSpec is returned for valid variable name
        self.assertEqual(dataset_spec.get_var_spec('cat_var_1'), cat_var_1)
        self.assertEqual(dataset_spec.get_var_spec('ord_var'), ord_var)
        self.assertEqual(dataset_spec.get_var_spec('num_var'), num_var)

        # Check that ValueError is raised for invalid variable name
        with self.assertRaises(ValueError) as cm:
            dataset_spec.get_var_spec('nonexistent_var')
        self.assertEqual(str(cm.exception), "Variable name nonexistent_var is not found in the provided variable specifications")

    def test_from_json(self):
        # Prepare a dict that matches the expected structure of the json file
        dataset_spec_dict = {
            "cat_var_specs": [
                {
                    "var_name": "categorical_var_1",
                    "var_type": "categorical",
                    "categorical_mapping": [["a","b","c"],["d","e","f"]],
                    "missing_values": "NA",
                    "column_name": "cat_1"
                }
            ],
            "ord_var_specs": [
                {
                    "var_name": "ordinal_var_1",
                    "var_type": "ordinal",
                    "categorical_mapping": [["1","2","3"],["4","5","6"]],
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
            ],
            "y_var": "ordinal_var_1"
        }

        # Write the dict to a temporary json file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(dataset_spec_dict, tmp)
            tempname = tmp.name

        # Load the json file as a DatasetSpec object
        dataset_spec = DatasetSpec.from_json(tempname)

        # Validate the DatasetSpec object
        self.assertEqual(len(dataset_spec.cat_var_specs), 1)
        self.assertEqual(len(dataset_spec.ord_var_specs), 1)
        self.assertEqual(len(dataset_spec.num_var_specs), 1)
        self.assertEqual(dataset_spec.y_var, "ordinal_var_1")

        # Also check some attributes of the first VarSpec in each list
        self.assertEqual(dataset_spec.cat_var_specs[0].var_name, "categorical_var_1")
        self.assertEqual(dataset_spec.ord_var_specs[0].var_name, "ordinal_var_1")
        self.assertEqual(dataset_spec.num_var_specs[0].var_name, "numerical_var_1")

    def tearDown(self):
        try:
            os.remove(self.tempname)
        except:
            pass

class TestMixedDataset(unittest.TestCase):

    def test_init(self):
        # Test initialization with and without y_var.
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')

        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var], y_var='num_var')

        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)
        Xord = np.array([[5, 6], [7, 8]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        expected_y_data = np.array([9, 10], dtype=np.float32)

        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xord=Xord, Xnum=Xnum)

        self.assertEqual(mixed_dataset.dataset_spec, dataset_spec)
        self.assertTrue((mixed_dataset.Xcat == Xcat).all())
        self.assertTrue((mixed_dataset.Xord == Xord).all())
        self.assertTrue(mixed_dataset.Xnum is None)
        self.assertTrue((mixed_dataset.y_data == expected_y_data).all())

    def test_len(self):
        # Test length
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        dataset_spec = DatasetSpec([cat_var], [], [])

        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)

        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat)
        self.assertEqual(len(mixed_dataset), 2)

    def test_getitem(self):
        # Test item fetching
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')
    
        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var], y_var='num_var')
    
        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)
        Xord = np.array([[5, 6], [7, 8]], dtype=np.int32)
        Xnum = np.array([[9], [10]], dtype=np.float32)
        expected_y_data = torch.from_numpy(np.array([9], dtype=np.float32))
    
        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xord=Xord, Xnum=Xnum)
        
        x_cat, x_ord, x_num, y = mixed_dataset[0]
        # Check the value of x_cat
        self.assertTrue((x_cat == torch.from_numpy(Xcat[0])).all())
        # Check the value of x_ord
        self.assertTrue((x_ord == torch.from_numpy(Xord[0])).all())
        # Check the value of x_num
        self.assertEqual(x_num, None)
        # Check the value of y
        self.assertTrue((y == expected_y_data).all())

    def test_inconsistent_shapes(self):
        # Test inconsistent shapes
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        ord_var = VarSpec('ord_var', 'ordinal', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        num_var = VarSpec('num_var', 'numerical')

        dataset_spec = DatasetSpec([cat_var], [ord_var], [num_var], y_var='num_var')

        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)
        Xord = np.array([[5, 6], [7, 8]], dtype=np.int32)
        Xnum = np.array([[9]])

        with self.assertRaises(ValueError) as cm:
            mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xord=Xord, Xnum=Xnum)
        self.assertEqual(str(cm.exception), "Input arrays do not have the same number of samples")

    def test_get_arrays_with_y(self):
        cat_var1 = VarSpec('cat1', 'categorical', [{'A','B'}])  
        num_var1 = VarSpec('num1', 'numerical')   
        num_var2 = VarSpec('num2', 'numerical')
    
        dataset_spec = DatasetSpec([cat_var1], [], [num_var1, num_var2], y_var='num1')
    
        Xcat = np.random.randint(0, 2, (10, 1), dtype=np.int32)  
        Xnum = np.random.randn(10, 2).astype(np.float32)
    
        dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xnum=Xnum)
    
        x_cat, x_ord, x_num, y = dataset.get_arrays()
    
        np.testing.assert_array_almost_equal(x_cat, Xcat)
        self.assertIsNone(x_ord)
        np.testing.assert_array_almost_equal(x_num, Xnum[:,1:])  
        np.testing.assert_array_almost_equal(y, dataset.y_data)
    
    def test_get_arrays_no_y(self):
        cat_var1 = VarSpec('cat1', 'categorical', [{'A','B'}])
        cat_var2 = VarSpec('cat2', 'categorical', [{'C','D'}])
    
        num_var1 = VarSpec('num1', 'numerical')
        num_var2 = VarSpec('num2', 'numerical') 
        num_var3 = VarSpec('num3', 'numerical')
    
        dataset_spec = DatasetSpec([cat_var1, cat_var2], [], [num_var1, num_var2, num_var3])
    
        Xcat = np.random.randint(0, 2, (10, 2), dtype=np.int32)
        Xnum = np.random.randn(10, 3).astype(np.float32)
    
        dataset = MixedDataset(dataset_spec, Xcat=Xcat, Xnum=Xnum)
    
        x_cat, x_ord, x_num, y = dataset.get_arrays()
    
        np.testing.assert_array_almost_equal(x_cat, Xcat)
        self.assertIsNone(x_ord)
        np.testing.assert_array_almost_equal(x_num, Xnum)
        self.assertIsNone(y)

    def test_aug_mult(self):
        # Test data augmentation
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        dataset_spec = DatasetSpec([cat_var], [], [])

        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)

        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, aug_mult=2)
        self.assertEqual(len(mixed_dataset), 4)

    def test_mask_prob(self):
        cat_var = VarSpec('cat_var', 'categorical', categorical_mapping=[{'A', 'B'}, {'C', 'D'}])
        dataset_spec = DatasetSpec([cat_var], [], [])
        Xcat = np.array([[1, 2], [3, 4]], dtype=np.int32)
        mixed_dataset = MixedDataset(dataset_spec, Xcat=Xcat, mask_prob=0.2, aug_mult=1)
        for idx in range(len(mixed_dataset)):
            item = mixed_dataset[idx]
            if mixed_dataset.y_data is not None:
                x_cat, x_ord, x_num, y = item
                self.assertTrue((x_cat == 0).sum() <= x_cat.numel() * 0.2)
                self.assertIsNone(x_ord)
                self.assertIsNone(x_num)
                self.assertIsNone(y)
            else:
                x_cat, x_ord, x_num = item
                self.assertTrue((x_cat == 0).sum() <= x_cat.numel() * 0.2)
                self.assertIsNone(x_ord)
                self.assertIsNone(x_num)

  
class TestConvertCategoriesToCodes(unittest.TestCase):

    def test_convert_categories_to_codes(self):
        # Test with categorical data
        data = pd.Series(['A', 'B', 'A', np.nan, 'B', '', 'D', 'C', 'A'])
        categorical_mapping = [{'A'}, {'B', 'C'}, {'D'}]
        var_spec = VarSpec('test_var', 'categorical', categorical_mapping)
        expected_output = np.array([1, 2, 1, 0, 2, 0, 3, 2, 1])
        np.testing.assert_array_equal(convert_categories_to_codes(data, var_spec), expected_output)

        # Test with ordinal data and explicit additional missing values
        data = pd.Series(['Low', 'Medium', 'High', np.nan,
                          '', 'Medium', 'Low', 'NA1',
                          'High', 'NA2'])
        ordinal_mapping = [{'Low'}, {'Medium'}, {'High'}]
        missing_values = {'NA1', 'NA2'}
        var_spec = VarSpec('test_var', 'ordinal', ordinal_mapping, missing_values=missing_values)
        expected_output = np.array([1, 2, 3, 0, 0, 2, 1, 0, 3, 0])
        np.testing.assert_array_equal(convert_categories_to_codes(data, var_spec), expected_output)

    def test_unobserved_category(self):
        data = pd.Series(['A', 'B', 'D'])
        categorical_mapping = [{'A'}, {'B'}, {'C'}]
        var_spec = VarSpec('test_var', 'categorical', categorical_mapping)
        with self.assertRaises(ValueError) as cm:
            convert_categories_to_codes(data, var_spec)
        self.assertEqual(str(cm.exception), "Variable test_var contains unobserved category D")


class TestParseNumericVariable(unittest.TestCase):

    def test_parse_numeric_variable(self):
        # Test with missing values represented by np.nan
        data = pd.Series(['1.2', '3.4', np.nan, '5.6', '', '7.8'])
        var_spec = VarSpec('test_var', 'numerical')
        expected_output = np.array([1.2, 3.4, np.nan, 5.6, np.nan, 7.8])
        np.testing.assert_array_equal(parse_numeric_variable(data, var_spec), expected_output)

        # Test with additional missing values
        data = pd.Series(['1.2', '3.4', 'NA', '5.6', '', '7.8', 'missing'])
        missing_values = {'NA', 'missing'}
        var_spec = VarSpec('test_var', 'numerical', missing_values=missing_values)
        expected_output = np.array([1.2, 3.4, np.nan, 5.6, np.nan, 7.8, np.nan])
        np.testing.assert_array_equal(parse_numeric_variable(data, var_spec), expected_output)

    def test_invalid_numeric_entry(self):
        data = pd.Series(['1.2', '3.4', 'invalid', '5.6'])
        var_spec = VarSpec('test_var', 'numerical')
        with self.assertRaises(ValueError) as cm:
            parse_numeric_variable(data, var_spec)
        self.assertEqual(str(cm.exception), "Invalid entry invalid for variable test_var cannot be converted to float")

class TestScaleNumericalVariables(unittest.TestCase):

    def test_scale_numerical_variables(self):
        # Single column of data
        Xnum = np.array([[1.0], [2.0], [3.0]])
        num_scalers = scale_numerical_variables(Xnum)
        self.assertEqual(len(num_scalers), 1)
        np.testing.assert_array_almost_equal(Xnum, np.array([[-1.22474487], [0.0], [1.22474487]]))

        # Multiple columns of data
        Xnum = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        num_scalers = scale_numerical_variables(Xnum)
        self.assertEqual(len(num_scalers), 2)
        np.testing.assert_array_almost_equal(
            Xnum,
            np.array([[-1.22474487, -1.22474487], [0.0, 0.0], [1.22474487, 1.22474487]])
        )

    def test_empty_array(self):
        # Empty array
        Xnum = np.empty((0, 0))
        num_scalers = scale_numerical_variables(Xnum)
        self.assertEqual(len(num_scalers), 0)


class TestExtractDependentVariable(unittest.TestCase):

    def test_extract_dependent_variable(self):
        # Test 1: categorical y variable
        cat_var_spec1 = VarSpec("cat_var1", "categorical", [{0, 1, 2}])
        cat_var_spec2 = VarSpec("y_var", "categorical", [{0, 1, 2, 3}])
        ord_var_spec1 = VarSpec("ord_var1", "ordinal", [{0, 1, 2, 3, 4}])
        num_var_spec1 = VarSpec("num_var1", "numerical")
        dataset_spec = DatasetSpec([cat_var_spec1, cat_var_spec2], [ord_var_spec1], [num_var_spec1], "y_var")
        Xcat = np.array([[1, 3], [2, 1], [0, 2], [1, 0]])
        Xord = np.array([[0], [3], [2], [1]])
        Xnum = np.array([[1.1], [2.2], [3.3], [4.4]])
        Xcat, Xord, Xnum, y = extract_dependent_variable(dataset_spec, Xcat, Xord, Xnum)
        np.testing.assert_array_equal(Xcat, np.array([[1], [2], [0], [1]]))
        np.testing.assert_array_equal(y, np.array([3, 1, 2, 0]))

        # Test 2: ordinal y variable
        ord_var_spec2 = VarSpec("y_var", "ordinal", [{0, 1, 2, 3, 4}])
        dataset_spec = DatasetSpec([cat_var_spec1], [ord_var_spec1, ord_var_spec2], [num_var_spec1], "y_var")
        Xord = np.array([[0, 3], [3, 1], [2, 2], [1, 4]])
        Xcat, Xord, Xnum, y = extract_dependent_variable(dataset_spec, Xcat, Xord, Xnum)
        np.testing.assert_array_equal(Xord, np.array([[0], [3], [2], [1]]))
        np.testing.assert_array_equal(y, np.array([3, 1, 2, 4]))

        # Test 3: numerical y variable
        num_var_spec2 = VarSpec("y_var", "numerical")
        dataset_spec = DatasetSpec([cat_var_spec1], [ord_var_spec1], [num_var_spec1, num_var_spec2], "y_var")
        Xnum = np.array([[1.1, 2.2], [2.2, 3.3], [3.3, 4.4], [4.4, 5.5]])
        Xcat, Xord, Xnum, y = extract_dependent_variable(dataset_spec, Xcat, Xord, Xnum)
        np.testing.assert_array_equal(Xnum, np.array([[1.1], [2.2], [3.3], [4.4]]))
        np.testing.assert_array_equal(y, np.array([2.2, 3.3, 4.4, 5.5]))

    def test_no_y_variable(self):
        # Test 4: Error when no y variable
        cat_var_spec1 = VarSpec("cat_var1", "categorical", [{0, 1, 2}])
        ord_var_spec1 = VarSpec("ord_var1", "ordinal", [{0, 1, 2, 3, 4}])
        num_var_spec1 = VarSpec("num_var1", "numerical")
        dataset_spec = DatasetSpec([cat_var_spec1], [ord_var_spec1], [num_var_spec1], None)
        Xcat = np.array([[1], [2], [0], [1]])
        Xord = np.array([[0], [3], [2], [1]])
        Xnum = np.array([[1.1], [2.2], [3.3], [4.4]])
        with self.assertRaises(ValueError) as cm:
            Xcat, Xord, Xnum, y = extract_dependent_variable(dataset_spec, Xcat, Xord, Xnum)
        expected_error = "This method should not be called if there is no y-variable in the dataset_spec"
        self.assertEqual(str(cm.exception), expected_error)


class TestLoadMixedData(unittest.TestCase):
    def create_temp_data_file(self, data_dict, file_type='csv'):
        data_df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=False)
        if file_type == 'csv':
            data_df.to_csv(temp_file.name, index=False)
        elif file_type in ['xlsx', 'xls']:
            data_df.to_excel(temp_file.name, index=False)
        temp_file.close()  # Manually close the file
        return temp_file.name

    def create_temp_dataset_spec_file(self, dataset_spec_dict):
        temp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(temp_file.name, 'w') as f:
            # Convert the sets to lists to make them serializable
            for cat_var_spec in dataset_spec_dict.get('cat_var_specs', []):
                cat_var_spec['categorical_mapping'] = [list(item) for item in cat_var_spec.get('categorical_mapping', [])]
            for ord_var_spec in dataset_spec_dict.get('ord_var_specs', []):
                ord_var_spec['categorical_mapping'] = [list(item) for item in ord_var_spec.get('categorical_mapping', [])]
            json.dump(dataset_spec_dict, f)
        temp_file.close()
        return temp_file.name

    def test_load_mixed_data(self):
        # Create temporary data file
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'], 'C': [1, 2, 3]}
    
        # Create temporary dataset specification file
        dataset_spec_dict = {
            'cat_var_specs': [
                {'var_name': 'A', 'var_type': 'categorical', 'categorical_mapping': [{'a', 'b'}, {'c'}]},
                {'var_name': 'B', 'var_type': 'categorical', 'categorical_mapping': [{'d', 'e'}, {'f'}]}
            ],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        # Define the expected output
        expected_Xcat = np.array([[1, 1], [1, 1], [2, 2]])
        expected_Xnum = np.array([[-1.2247449], [0.], [1.2247449]])

        for file_type in ['csv', 'xlsx']:
            # Create and load data file for each file type
            data_file = self.create_temp_data_file(data_dict, file_type=file_type)
            mixed_dataset, num_scalers = load_mixed_data(dataset_spec_file, data_file)
    
            # Assert that the actual output matches the expected output
            np.testing.assert_array_almost_equal(mixed_dataset.Xcat, expected_Xcat)
            self.assertIsNone(mixed_dataset.Xord)
            np.testing.assert_array_almost_equal(mixed_dataset.Xnum, expected_Xnum)
    
            # Assert that num_scalers has the correct length
            self.assertEqual(len(num_scalers), 1)
    
            # Clean up temporary files
            os.unlink(data_file)
        os.unlink(dataset_spec_file)

    def test_load_mixed_data_unsupported_file_type(self):
        # Create a file with unsupported type and test if ValueError is raised.
        data_dict = {'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f'], 'C': [1, 2, 3]}
        data_file = self.create_temp_data_file(data_dict, file_type='txt')  # Unsupported file type
    
        dataset_spec_dict = {
            'cat_var_specs': [
                {'var_name': 'A', 'var_type': 'categorical', 'categorical_mapping': [{'a', 'b'}, {'c'}]},
                {'var_name': 'B', 'var_type': 'categorical', 'categorical_mapping': [{'d', 'e'}, {'f'}]}
            ],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(ValueError) as cm:
            mixed_dataset, num_scalers = load_mixed_data(dataset_spec_file, data_file)
        expected_error = "Unsupported file type"
        self.assertEqual(str(cm.exception), expected_error)

        os.unlink(data_file)
        os.unlink(dataset_spec_file)

    def test_load_mixed_data_invalid_specification(self):
        # Test for cases where the data file and dataset specification file do not align.
        # Create a data file where 'A' column contains values not specified in the categorical_mapping
        data_dict = {'A': ['x', 'y', 'z'], 'B': ['d', 'e', 'f'], 'C': [1, 2, 3]}  # 'x', 'y' and 'z' are not in the categorical_mapping
        data_file = self.create_temp_data_file(data_dict, file_type='csv')
    
        dataset_spec_dict = {
            'cat_var_specs': [
                {'var_name': 'A', 'var_type': 'categorical', 'categorical_mapping': [{'a', 'b'}, {'c'}]},  # Does not match data in 'A'
                {'var_name': 'B', 'var_type': 'categorical', 'categorical_mapping': [{'d', 'e'}, {'f'}]}
            ],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(ValueError) as cm:  # Assuming a ValueError is raised
            mixed_dataset, num_scalers = load_mixed_data(dataset_spec_file, data_file)
        expected_error = "Variable A contains unobserved category x"
        self.assertEqual(str(cm.exception), expected_error)
    
        os.unlink(data_file)
        os.unlink(dataset_spec_file)
    
    def test_load_mixed_data_file_not_found(self):
        # Test for cases where the data file or the dataset specification file does not exist.
        dataset_spec_dict = {
            'cat_var_specs': [
                {'var_name': 'A', 'var_type': 'categorical', 'categorical_mapping': [{'a', 'b'}, {'c'}]},
                {'var_name': 'B', 'var_type': 'categorical', 'categorical_mapping': [{'d', 'e'}, {'f'}]}
            ],
            'num_var_specs': [
                {'var_name': 'C', 'var_type': 'numerical'}
            ]
        }
        dataset_spec_file = self.create_temp_dataset_spec_file(dataset_spec_dict)
    
        with self.assertRaises(FileNotFoundError) as cm:
            mixed_dataset, num_scalers = load_mixed_data(dataset_spec_file, "non_existent_file.csv")  # Non-existent file
        expected_error = "Data file 'non_existent_file.csv' does not exist."
        self.assertEqual(str(cm.exception), expected_error)
    
        os.unlink(dataset_spec_file)


if __name__ == "__main__":
    unittest.main()