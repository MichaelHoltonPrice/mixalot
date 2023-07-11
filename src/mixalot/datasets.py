from typing import List, Optional, Dict
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

class VarSpec:
    """
    Represents the specifications of a variable in a dataset.

    Each VarSpec object captures the variable name, its type (numerical, ttttttttttt, or ordinal), 
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
        if not all(isinstance(cat_set, set) for cat_set in self.categorical_mapping):
            raise ValueError("All entries in 'categorical_mapping' should be sets")

        all_values = [value for cat_set in self.categorical_mapping for value in cat_set]
        if len(all_values) != len(set(all_values)):
            raise ValueError(f"Some values appear in more than one set for variable {self.var_name}")


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

        self.all_var_names = set([var.var_name for var_list in [self.cat_var_specs, self.ord_var_specs, self.num_var_specs] for var in var_list])

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
    
        all_var_names = [var.var_name for var_list in [self.cat_var_specs, self.ord_var_specs, self.num_var_specs] for var in var_list]

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

    def get_var_spec(self, var_name):
        """
        Returns the VarSpec corresponding to the provided variable name.

        Args:
        - var_name: The name of the variable for which the VarSpec is to be returned.

        Returns:
        - var_spec: The VarSpec corresponding to the provided variable name.

        Raises:
        - ValueError: If the var_name is not found in the variable specifications.
        """
        for var_type in ['cat', 'ord', 'num']:
            for var_spec in getattr(self, var_type + "_var_specs"):
                if var_spec.var_name == var_name:
                    return var_spec

        raise ValueError(f"Variable name {var_name} is not found in the provided variable specifications")


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

        cat_var_specs = [
            VarSpec(
                **{
                    **spec,
                    'categorical_mapping': [
                        set(cat_map) for cat_map in spec.get('categorical_mapping', [])
                    ]
                }
            )
            for spec in json_dict.get('cat_var_specs', [])
        ]
        
        ord_var_specs = [
            VarSpec(
                **{
                    **spec,
                    'categorical_mapping': [
                        set(cat_map) for cat_map in spec.get('categorical_mapping', [])
                    ]
                }
            )
            for spec in json_dict.get('ord_var_specs', [])
        ]


        num_var_specs = [VarSpec(**spec) for spec in json_dict.get('num_var_specs', [])]
        y_var = json_dict.get('y_var', None)
        return cls(cat_var_specs, ord_var_specs, num_var_specs, y_var)


class MixedDataset(Dataset):
    """
    A class to hold and manage mixed types of data (categorical, ordinal, and numerical).

    Args:
        dataset_spec (DatasetSpec): Specification of the dataset.
        Xcat (np.ndarray, optional): Numpy array holding categorical data.
        Xord (np.ndarray, optional): Numpy array holding ordinal data.
        Xnum (np.ndarray, optional): Numpy array holding numerical data.

    Attributes:
        dataset_spec (DatasetSpec): Specification of the dataset.
        Xcat (np.ndarray): Categorical data.
        Xord (np.ndarray): Ordinal data.
        Xnum (np.ndarray): Numerical data.
        y_data (np.ndarray): Y variable data if it exists.
        num_obs (int): Number of observations in the dataset.
    """

    def __init__(self, dataset_spec: DatasetSpec,
                 Xcat: Optional[np.ndarray] = None,
                 Xord: Optional[np.ndarray] = None,
                 Xnum: Optional[np.ndarray] = None):

        self.dataset_spec = dataset_spec

        if Xcat is None and Xord is None and Xnum is None:
            raise ValueError("Xcat, Xord, and Xnum cannot all be None")
        
        # Ensure all arrays have the same number of samples
        num_obs_list = set([len(X) for X in [Xcat, Xord, Xnum] if X is not None])

        if len(num_obs_list) > 1:
            raise ValueError("Input arrays do not have the same number of samples")
        self.num_obs = num_obs_list.pop()

        # Validate and enforce data types of inputs
        if Xcat is not None and Xcat.dtype != np.int32:
            raise ValueError("Xcat should have dtype int32")
        if Xord is not None and Xord.dtype != np.int32:
            raise ValueError("Xord should have dtype int32")
        if Xnum is not None and Xnum.dtype != np.float32:
            raise ValueError("Xnum should have dtype float32")

        # Extract y from the relevant dataset and store separately, if it exists
        if dataset_spec.y_var is not None:
            if dataset_spec.y_var in dataset_spec.get_ordered_variables('categorical'):
                y_idx = dataset_spec.get_ordered_variables('categorical').index(dataset_spec.y_var)
                self.y_data = Xcat[:, y_idx].copy()
                if Xcat.shape[1] == 1:
                    Xcat = None
                else:
                    Xcat = np.delete(Xcat, y_idx, axis=1)
            elif dataset_spec.y_var in dataset_spec.get_ordered_variables('ordinal'):
                y_idx = dataset_spec.get_ordered_variables('ordinal').index(dataset_spec.y_var)
                self.y_data = Xord[:, y_idx].copy()
                if Xord.shape[1] == 1:
                    Xord = None
                else:
                    Xord = np.delete(Xord, y_idx, axis=1)
            elif dataset_spec.y_var in dataset_spec.get_ordered_variables('numerical'):
                y_idx = dataset_spec.get_ordered_variables('numerical').index(dataset_spec.y_var)
                self.y_data = Xnum[:, y_idx].copy()
                if Xnum.shape[1] == 1:
                    Xnum = None
                else:
                    Xnum = np.delete(Xnum, y_idx, axis=1)
        else:
            self.y_data = None

        self.Xcat = Xcat
        self.Xord = Xord
        self.Xnum = Xnum

    def __len__(self):
        """
        Returns:
            Number of observations in the dataset.
        """
        return self.num_obs

    def __getitem__(self, idx):
        """
        Given an index, retrieves the corresponding item from the MixedDataset. The item is a list consisting of
        corresponding elements (rows) from Xcat, Xord, Xnum, and y_data (if they exist). 

        Args:
            idx (int): The index of the item to be fetched.

        Returns:
            list: A list containing elements from Xcat, Xord, Xnum and y_data at the given index. The length of
                  the list can be either 3 or 4 depending on whether y_data exists.
        """

        # Initialize an empty list to store the item
        item = []

        # TODO: determine whether it is best/faster to store Xcat (etc.) as
        #       torch tensors rather than numpy arrays.
        # Append x_cat (or None)
        if self.Xcat is not None:
            item.append(torch.from_numpy(self.Xcat[idx]))
        else:
            item.append(None)
        
        # Append x_ord (or None)
        if self.Xord is not None:
            item.append(torch.from_numpy(self.Xord[idx]))
        else:
            item.append(None)

        # Append x_num (or None)
        if self.Xnum is not None:
            item.append(torch.from_numpy(self.Xnum[idx]))
        else:
            item.append(None)

        # If y_data exists, append the corresponding value to the item
        # We do not append None if it does not exist.
        if self.y_data is not None:
            item.append(torch.from_numpy(np.array([self.y_data[idx]],
                                                  dtype=self.y_data.dtype)))

        # Return the item
        return item

def convert_categories_to_codes(data, var_spec):
    """
    Function to convert categorical or ordinal data to coded format.
    
    Parameters
    ----------
    data: array
        The data (column) to be converted.
    var_spec: VarSpec
        The variable specification for this column
        
    Returns
    -------
    array
        Integer codes of the converted data as specified by
        var_spec.categorical_mapping, where 0 is reserved for NA

    Raises
    ------
    ValueError
        If the input column contains an unobserved category.
    """
    # Create a mapping from category to code
    category_to_code = {cat: i+1 for i, cat_set in enumerate(var_spec.categorical_mapping)\
                        for cat in cat_set}

    codes = []
    for entry in data:
        # If the entry is null, this is a missing value (coded as 0)
        if pd.isnull(entry):
            codes.append(0)
        else:
            # The entry is not null, so strip it of whitespace
            entry = entry.strip()
            if entry == '':
                # We interpret blank entries as missing
                codes.append(0)
            elif var_spec.missing_values is not None and entry in var_spec.missing_values:
                # This is an NA explicitly specified in var_spec.missing_values
                codes.append(0)
            else:
                # Try to get the code for this category
                code = category_to_code.get(entry)
                if code is None:
                    # This entry doesn't map to any category in category_to_code
                    raise ValueError(f"Variable {var_spec.var_name} contains unobserved category {entry}")
                else:
                    codes.append(code)
    return np.array(codes)

def parse_numeric_variable(data, var_spec):
    """
    Function to handle missing values in numerical data.
    
    Parameters
    ----------
    data: array
        The data in which to handle missing values.
    var_spec: VarSpec
        The variable specification for this column.
        
    Returns
    -------
    array
        The data with missing values handled.
    """
    handled_data = []
    for entry in data:
        # If the entry is null, this is a missing value (coded as NaN)
        if pd.isnull(entry):
            handled_data.append(np.nan)
        else:
            # The entry is not null, so strip it of whitespace
            entry = entry.strip()
            if entry == '':
                # We interpret blank entries as missing
                handled_data.append(np.nan)
            elif var_spec.missing_values is not None and entry in var_spec.missing_values:
                # This is an NA explicitly specified in var_spec.missing_values
                handled_data.append(np.nan)
            else:
                try:
                    # This is a valid number! Convert it to float.
                    handled_data.append(float(entry))
                except ValueError:
                    raise ValueError(f"Invalid entry {entry} for variable {var_spec.var_name} cannot be converted to float")
    return np.array(handled_data, dtype=float)


def scale_numerical_variables(Xnum):
    """
    Function to scale numerical variables.
    
    Parameters
    ----------
    Xnum: array
        The numerical data to be scaled.
        
    Returns
    -------
    dict
        A dictionary of scalers for each numerical variable, where the keys
        are the integer column index (starting from 0)
    """
    num_scalers = {}
    if Xnum.shape[1] > 0:
        for i in range(Xnum.shape[1]):
            col = Xnum[:, i]
            scaler = StandardScaler()
            scaled_col = scaler.fit_transform(col.reshape(-1, 1))
            Xnum[:, i] = scaled_col.flatten()
            num_scalers[i] = scaler
    return num_scalers


def extract_dependent_variable(y_var_name, var_spec, Xcat, Xord, Xnum):
    """
    Function to extract the dependent variable from the data.
    
    Parameters
    ----------
    y_var_name: str
        The name of the dependent variable.
    var_spec: object
        An object containing a dictionary of variable specifications and functions for obtaining
        the ordered variables of a specific type (categorical, ordinal, numerical).
    Xcat: array
        The categorical data.
    Xord: array
        The ordinal data.
    Xnum: array
        The numerical data.
        
    Returns
    -------
    tuple
        A tuple consisting of the updated matrices for categorical, ordinal, and numerical data,
        and the extracted dependent variable.
    
    Raises
    ------
    ValueError
        If the dependent variable is not found in any variables.
    """
    for var_type in ['categorical', 'ordinal', 'numerical']:
        if y_var_name in var_spec.get_ordered_variables(var_type):
            idx = var_spec.get_ordered_variables(var_type).index(y_var_name)
            if var_type == 'categorical':
                y = Xcat[:, idx]
                Xcat = np.delete(Xcat, idx, axis=1)
            elif var_type == 'ordinal':
                y = Xord[:, idx]
                Xord = np.delete(Xord, idx, axis=1)
            else:  # var_type == 'numerical'
                y = Xnum[:, idx]
                Xnum = np.delete(Xnum, idx, axis=1)
            return Xcat, Xord, Xnum, y

    raise ValueError(f"y_var_name {y_var_name} not found in any variables")