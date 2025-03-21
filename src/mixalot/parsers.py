"""Methods for parsing and loading data."""
from __future__ import annotations
import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mixalot.datasets import DatasetSpec, MixedDataset
from mixalot.models import RandomForestSpec


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
    category_to_code = {
        cat: i+1 for i, cat_set in enumerate(
            var_spec.categorical_mapping
        ) for cat in cat_set
        }

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
            elif (var_spec.missing_values is not None and entry in
                  var_spec.missing_values):
                # This is an NA explicitly specified in var_spec.missing_values
                codes.append(0)
            else:
                # Try to get the code for this category
                code = category_to_code.get(entry)
                if code is None:
                    # This entry doesn't map to any category in
                    # category_to_code
                    raise ValueError(
                        f"Variable {var_spec.var_name} contains unobserved "
                        f"category {entry}"
                    )
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
            elif (var_spec.missing_values is not None
                  and entry in var_spec.missing_values):
                # This is an NA explicitly specified in var_spec.missing_values
                handled_data.append(np.nan)
            else:
                try:
                    # This is a valid number! Convert it to float.
                    handled_data.append(float(entry))
                except ValueError:
                    raise ValueError(
                        f"Invalid entry {entry} for variable "
                        f"{var_spec.var_name} cannot be converted to float"
                    )
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


def extract_dependent_variable(dataset_spec, model_spec, Xcat, Xord, Xnum):
    """Function to extract the dependent variable from the data.
    
    Parameters
    ----------
    dataset_spec: DatasetSpec
        A DatasetSpec object containing the specifications of the dataset.
    model_spec: SingleTargetModelSpec
        A model specification containing the y_var name.
    Xcat: array
        The categorical data.
    Xord: array
        The ordinal data.
    Xnum: array
        The numerical data.
        
    Returns
    -------
    tuple
        A tuple consisting of the updated matrices for categorical, ordinal,
        and numerical data, and the extracted dependent variable.
        
    Raises
    ------
    ValueError
        If the y-variable is not specified in the model_spec.
    """
    if (model_spec is None
            or not hasattr(model_spec, 'y_var')
            or model_spec.y_var is None):
        raise ValueError(
            "This method should not be called if model_spec does not have a "
            "valid y_var"
        )

    y_var_name = model_spec.y_var
    var_spec = dataset_spec.get_var_spec(y_var_name)

    if var_spec.var_type == 'categorical':
        idx = dataset_spec.get_ordered_variables(
            'categorical'
        ).index(y_var_name)
        y = Xcat[:, idx]
        Xcat = np.delete(Xcat, idx, axis=1)
    elif var_spec.var_type == 'ordinal':
        idx = dataset_spec.get_ordered_variables('ordinal').index(y_var_name)
        y = Xord[:, idx]
        Xord = np.delete(Xord, idx, axis=1)
    else:  # var_spec.var_type == 'numerical'
        idx = dataset_spec.get_ordered_variables('numerical').index(y_var_name)
        y = Xnum[:, idx]
        Xnum = np.delete(Xnum, idx, axis=1)
        
    return Xcat, Xord, Xnum, y


def load_model_spec(model_spec_file: str) -> 'SingleTargetModelSpec':
    """Load a model specification from a JSON file.
    
    Args:
        model_spec_file (str): Path to a JSON file containing the model
            specification.
            
    Returns:
        SingleTargetModelSpec: A model specification object of the appropriate
            type based on the 'model_type' field in the file.
            
    Raises:
        FileNotFoundError: If the provided model specification file does not
            exist.
        ValueError: If the model type specified in the file is not recognized.
    """
    # Ensure the model specification file exists
    if not os.path.isfile(model_spec_file):
        raise FileNotFoundError(
            f"Model specification file '{model_spec_file}' does not exist."
        )
    
    # Load the model specification JSON
    with open(model_spec_file, 'r') as file:
        model_dict = json.load(file)
    
    # Extract the model type
    model_type = model_dict.get('model_type')
    if not model_type:
        raise ValueError("Model specification is missing 'model_type' field")
    
    # Create the appropriate model specification based on model_type
    if model_type == 'random_forest':
        return RandomForestSpec(
            y_var=model_dict['y_var'],
            independent_vars=model_dict['independent_vars'],
            hyperparameters=model_dict.get('hyperparameters', {})
        )
    # Add more model types here as they are implemented
    else:
        raise ValueError(f"Unrecognized model type: {model_type}")


def load_mixed_data(
    dataset_spec_file: str,
    data_file: str,
    model_spec_file: Optional[str] = None
) -> Tuple[MixedDataset, StandardScaler]:
    """Load mixed dataset from file, optionally with a model specification.
    
    Args:
        dataset_spec_file (str): Path to a JSON file containing the
            specifications for the dataset.
        data_file (str): Path to the dataset file (.csv, .xlsx, or .xls
            format).
        model_spec_file (str, optional): Path to a JSON file containing the
            model specification. Default is None.
            
    Returns:
        mixed_dataset (MixedDataset): A MixedDataset object generated from the
            inputs files
        num_scalers (StandardScaler): Scaler objects used to scale numerical
            variables.
            
    Raises:
        FileNotFoundError: If the provided dataset specification or data file
            does not exist.
        ValueError: If the provided file type is unsupported, or if there are
            inconsistencies between the data file and the dataset
            specification.
    """
    # Ensure that dataset specification file exists
    if not os.path.isfile(dataset_spec_file):
        raise FileNotFoundError(
            f"Dataset specification file '{dataset_spec_file}' does not exist."
        )
    
    # Ensure that data file exists
    if not os.path.isfile(data_file):
        raise FileNotFoundError(
            f"Data file '{data_file}' does not exist."
        )
    
    # Load the dataset specification from file
    dataset_spec = DatasetSpec.from_json(dataset_spec_file)
    
    # Load the model specification if provided
    model_spec = None
    if model_spec_file:
        model_spec = load_model_spec(model_spec_file)
    
    # Load the dataframe from the file, with all columns loaded as strings
    if data_file.endswith('.csv'):
        dataframe = pd.read_csv(data_file, dtype=str)
    elif data_file.endswith('.xlsx') or data_file.endswith('.xls'):
        dataframe = pd.read_excel(data_file, dtype=str)
    else:
        raise ValueError('Unsupported file type')
    
    # Ensure that the dataset specification matches the data file
    dataframe_columns = set(dataframe.columns)
    for var_name in dataset_spec.all_var_names:
        var_spec = dataset_spec.get_var_spec(var_name)
        column_name =\
            var_spec.column_name if var_spec.column_name else var_name
        if column_name not in dataframe_columns:
            raise ValueError(
                f"Variable '{var_name}' (column '{column_name}') is specified "
                f"in dataset specification but not found in data file."
            )
    
    # Initialize empty lists for storing data
    Xcat, Xord, Xnum = [], [], []
    
    # Process the data according to its type
    for var_type, X_list in zip(
        ['categorical', 'ordinal', 'numerical'], 
        [Xcat, Xord, Xnum]
    ):
        ordered_variables = dataset_spec.get_ordered_variables(var_type)
        
        for var_name in ordered_variables:
            # Get the variable specification for this variable
            var_spec = dataset_spec.get_var_spec(var_name)
            column_name = (var_spec.column_name 
                          if var_spec.column_name 
                          else var_spec.var_name)
            
            # Get the pandas Series object for this column
            column_data = dataframe[column_name]
            
            # Process the column according to its type
            if var_type in {'categorical', 'ordinal'}:
                parsed_column = convert_categories_to_codes(column_data,
                                                            var_spec)
            else:  # var_type == 'numerical'
                parsed_column = parse_numeric_variable(column_data, var_spec)
            
            # Append the processed column to the list
            X_list.append(parsed_column)
    
    # Convert lists of columns into 2D arrays
    Xcat = np.column_stack(Xcat).astype(np.int32) if len(Xcat) != 0 else None
    Xord = np.column_stack(Xord).astype(np.int32) if len(Xord) != 0 else None
    Xnum = np.column_stack(Xnum).astype(np.float32) if len(Xnum) != 0 else None
    
    # Scale numerical variables and get the scaler object
    if Xnum is not None:
        num_scalers = scale_numerical_variables(Xnum)
    else:
        num_scalers = None
    
    # Create the mixed dataset object
    mixed_dataset = MixedDataset(
        dataset_spec=dataset_spec,
        Xcat=Xcat,
        Xord=Xord,
        Xnum=Xnum,
        model_spec=model_spec
    )
    
    return mixed_dataset, num_scalers
