# python -m unittest tests.test_datasets
import tempfile
import json
import unittest
from mixalot.datasets import DatasetSpec, VarSpec
import os

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
        self.assertEqual(str(cm.exception), "Some strings appear in more than one set for variable cat_var")


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


if __name__ == "__main__":
    unittest.main()