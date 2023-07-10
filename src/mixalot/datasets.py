from typing import List, Optional
import json

class VarSpec:
    """
    Represents the specifications of a variable in a dataset.

    Each VarSpec object captures the variable name, its type (numerical, categorical, or ordinal), 
    a mapping for categorical variables (if applicable), any missing values, and the order of the variable.

    Args:
        var_name (str): Name of the variable.
        var_type (str): Type of the variable. Must be 'numerical', 'categorical', or 'ordinal'.
        categorical_mapping (list[set], optional): List of sets representing the categories of a categorical or ordinal variable.
                                                   Each set includes strings that map to the respective category. 
                                                   This parameter is required if var_type is 'categorical' or 'ordinal'.
        missing_values (Any, optional): Specifies the representation of missing values in the variable. Default is None.
        column_name (str, optional): Specifies a distinct column name in an input file (by default var_name is assumed)

    Raises:
        ValueError: If var_type is not 'numerical', 'categorical', or 'ordinal'.
        ValueError: If var_type is 'categorical' or 'ordinal' but no categorical_mapping is provided.
        ValueError: If the same string appears in more than one set of categorical_mapping.
    """
 
    def __init__(self,
                 var_name,
                 var_type,
                 categorical_mapping=None,
                 missing_values=None,
                 column_name=None):
        self.var_name = var_name
        self.var_type = var_type.lower()
        self.categorical_mapping = categorical_mapping
        self.missing_values = missing_values
        self.column_name = column_name
        self._validate_spec()

    def _validate_spec(self):
        if self.var_type not in ['numerical', 'categorical', 'ordinal']:
            raise ValueError(f"Invalid 'type' field for variable {self.var_name}. Expected 'numerical', 'categorical', or 'ordinal'")
        if self.var_type in ['categorical', 'ordinal'] and self.categorical_mapping is None:
            raise ValueError(f"Missing 'categorical_mapping' field for variable {self.var_name} of type {self.var_type}")
        if self.var_type in ['categorical', 'ordinal']:
            self._validate_categorical_mapping()

    def _validate_categorical_mapping(self):
        # Check that all entries are sets
        if not all(isinstance(cat_map, set) for cat_map in self.categorical_mapping):
            raise ValueError("All entries in 'categorical_mapping' should be sets")
        # Check for string duplication across different sets
        all_values = [item for cat_map in self.categorical_mapping for item in cat_map]
        if len(all_values) != len(set(all_values)):
            raise ValueError(f"Some strings appear in more than one set for variable {self.var_name}")


class DatasetSpec:
    """
    Class to specify a dataset. This includes ordered lists of categorical, ordinal, and numerical variables.

    Args:
        cat_var_specs: List of VarSpecs for categorical variables
        ord_var_specs: List of VarSpecs for ordinal variables
        num_var_specs: List of VarSpecs for numerical variables
        y_var: String specifying the y variable. It should match a variable name in one of the VarSpecs.
               Defaults to None.
    """
    def __init__(self,
                 cat_var_specs: List[VarSpec],
                 ord_var_specs: List[VarSpec],
                 num_var_specs: List[VarSpec],
                 y_var: Optional[str] = None):
        self.cat_var_specs = cat_var_specs
        self.ord_var_specs = ord_var_specs
        self.num_var_specs = num_var_specs
        self.y_var = y_var
        self._validate_specs()

    def _validate_specs(self):
        """
        Private method to validate the variable specifications provided. Raises ValueError with an appropriate
        message in case of any of the following conditions:
        1. All of cat_var_specs, ord_var_specs, and num_var_specs are empty
        2. Any variable name is repeated across different types
        3. The y_var is specified but doesn't match a variable name in the VarSpecs
        4. The type of each variable in the respective var_specs list is not as expected
        """
        if not self.cat_var_specs and not self.ord_var_specs and not self.num_var_specs:
            raise ValueError("At least one of cat_var_specs, ord_var_specs, or num_var_specs must be non-empty")
    
        all_var_names = [var.var_name for var in self.cat_var_specs + self.ord_var_specs + self.num_var_specs]
        if len(all_var_names) != len(set(all_var_names)):
            raise ValueError("Variable names must be unique across all variable types")
    
        if self.y_var and self.y_var not in all_var_names:
            raise ValueError(f"y_var {self.y_var} is not found in the provided variable specifications")
    
        for var_type, var_specs in [('categorical', self.cat_var_specs), 
                                    ('ordinal', self.ord_var_specs), 
                                    ('numerical', self.num_var_specs)]:
            if not all(isinstance(var_spec, VarSpec) and var_spec.var_type == var_type for var_spec in var_specs):
                raise ValueError(f"All variable specifications in {var_type}_var_specs must be instances of VarSpec of type {var_type}")

    def get_ordered_variables(self, var_type):
        """
        Returns a list of ordered variable names of a certain type.

        Args:
        - var_type: A string representing the type of the variables to be returned.
                    This should be one of 'numerical', 'categorical', or 'ordinal'.

        Returns:
        - ordered_vars: A list of variable names in the order they appear in the dataset.

        Raises:
        - ValueError: If the var_type is not 'numerical', 'categorical', or 'ordinal'.
        """
        var_specs = []
        if var_type == 'categorical':
            var_specs = self.cat_var_specs
        elif var_type == 'ordinal':
            var_specs = self.ord_var_specs
        elif var_type == 'numerical':
            var_specs = self.num_var_specs
        else:
            raise ValueError(f"Invalid 'type' field. Expected 'numerical', 'categorical', or 'ordinal'")
        
        # get variable names in the order they are listed
        ordered_vars = [var_spec.var_name for var_spec in var_specs]
        return ordered_vars

    @classmethod
    def from_json(cls, json_path):
        """
        Static method to create a DatasetSpec object from a JSON file.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            DatasetSpec object
        """
        with open(json_path, 'r') as file:
            json_dict = json.load(file)

        cat_var_specs = [VarSpec(**{**spec, 'categorical_mapping': [set(cat_map) for cat_map in spec.get('categorical_mapping', [])]}) for spec in json_dict.get('cat_var_specs', [])]
        ord_var_specs = [VarSpec(**{**spec, 'categorical_mapping': [set(cat_map) for cat_map in spec.get('categorical_mapping', [])]}) for spec in json_dict.get('ord_var_specs', [])]
        num_var_specs = [VarSpec(**spec) for spec in json_dict.get('num_var_specs', [])]
        y_var = json_dict.get('y_var', None)
        return cls(cat_var_specs, ord_var_specs, num_var_specs, y_var)