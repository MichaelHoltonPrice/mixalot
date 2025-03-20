"""Core classes for dataset and model specification."""
from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class VarSpec:
    """Represents the specifications of a variable in a dataset.

    Each VarSpec object captures the variable name, its type (numerical,
    categorical, or ordinal), a mapping for categorical variables (if
    applicable), any missing values, and the order of the variable.

    Args:
        var_name (str): Name of the variable.
        var_type (str): Type of the variable. Must be 'numerical',
            'categorical', or 'ordinal'.
        categorical_mapping (list[set], optional): List of sets representing
            the categories of a categorical or ordinal variable. Each set
            includes strings that map to the respective category. 
            This parameter is required if var_type is 'categorical' or
            'ordinal'.
        missing_values (Any, optional): Specifies the representation of missing
            values in the variable. Default is None.
        column_name (str, optional): Specifies a distinct column name in an
            input file (by default var_name is assumed)

    Raises:
        ValueError: If var_type is not 'numerical', 'categorical', or
            'ordinal'.
        ValueError: If var_type is 'categorical' or 'ordinal' but no
            categorical_mapping is provided.
        ValueError: If the same string appears in more than one set of
            categorical_mapping.
    """
 
    def __init__(self,
                 var_name,
                 var_type,
                 categorical_mapping=None,
                 missing_values=None,
                 column_name=None):
        """Initialize a VarSpec object."""
        self.var_name = var_name
        self.var_type = var_type.lower()
        self.categorical_mapping = categorical_mapping
        self.missing_values = missing_values
        self.column_name = column_name
        self._validate_spec()

    def _validate_spec(self):
        """Validate the variable specification."""
        if self.var_type not in ['numerical', 'categorical', 'ordinal']:
            raise ValueError(
                f"Invalid 'type' field for variable {self.var_name}. "
                "Expected 'numerical', 'categorical', or 'ordinal'"
            )
        if (self.var_type in ['categorical', 'ordinal']
            and self.categorical_mapping is None):
            raise ValueError(
                "Missing 'categorical_mapping' field for variable "
                f"{self.var_name} of type {self.var_type}"
            )
        if self.var_type in ['categorical', 'ordinal']:
            self._validate_categorical_mapping()

    def _validate_categorical_mapping(self):
        """Validate mapping for a categorical (or ordinal) variable."""
        if not all(isinstance(cat_set, set)
                   for cat_set in self.categorical_mapping):
            raise ValueError(
                "All entries in 'categorical_mapping' should be sets"
            )

        all_values = [value for cat_set in self.categorical_mapping
                      for value in cat_set]
        if len(all_values) != len(set(all_values)):
            raise ValueError(
                "Some values appear in more than one set for "
                f"variable {self.var_name}"
            )


class DatasetSpec:
    """Class to specify a dataset's variables and their characteristics.
    
    This class maintains ordered lists of categorical, ordinal, and numerical 
    variables, along with their specifications. It provides methods to validate
    the specifications and retrieve variable information.
    
    Args:
        cat_var_specs: List of VarSpecs for categorical variables
        ord_var_specs: List of VarSpecs for ordinal variables
        num_var_specs: List of VarSpecs for numerical variables
    """
    def __init__(self,
                 cat_var_specs: List[VarSpec],
                 ord_var_specs: List[VarSpec],
                 num_var_specs: List[VarSpec]):
        """Initializes a DatasetSpec object."""
        self.cat_var_specs = cat_var_specs
        self.ord_var_specs = ord_var_specs
        self.num_var_specs = num_var_specs
        self._validate_specs()

    def _validate_specs(self):
        """Validates the variable specifications provided.
        
        Raises:
            ValueError: If any of the following conditions are met:
                1. All of cat_var_specs, ord_var_specs, and num_var_specs are 
                   empty
                2. Any variable name is repeated across different types
                3. The type of each variable in the respective var_specs list
                   is not as expected
        """
        if (not self.cat_var_specs and not self.ord_var_specs
            and not self.num_var_specs):
            raise ValueError("At least one of cat_var_specs, ord_var_specs, "
                             "or num_var_specs must be non-empty")
    
        all_var_names = [var.var_name for var_list in 
                         [self.cat_var_specs, self.ord_var_specs, 
                          self.num_var_specs] for var in var_list]

        if len(all_var_names) != len(set(all_var_names)):
            raise ValueError("Variable names must be unique across all "
                             "variable types")
    
        for var_type, var_specs in [('categorical', self.cat_var_specs), 
                                    ('ordinal', self.ord_var_specs), 
                                    ('numerical', self.num_var_specs)]:
            if not all(isinstance(var_spec, VarSpec) and 
                       var_spec.var_type ==
                       var_type for var_spec in var_specs):
                raise ValueError(
                    f"All variable specifications in {var_type}_var_specs "
                    f"must be instances of VarSpec of type {var_type}")

    @property
    def all_var_names(self):
        """Returns the set of all variable names across all variable types."""
        return set(var.var_name for var_list in 
                  [self.cat_var_specs, self.ord_var_specs, self.num_var_specs] 
                  for var in var_list)

    def get_ordered_variables(self, var_type):
        """Returns a list of ordered variable names of a certain type.
        
        Args:
            var_type: A string representing the type of the variables to be 
                      returned. This should be one of 'numerical', 
                      'categorical', or 'ordinal'.
        
        Returns:
            ordered_vars: A list of variable names in the order they appear in 
                          the dataset.
        
        Raises:
            ValueError: If the var_type is not 'numerical', 'categorical', or 
                        'ordinal'.
        """
        var_specs = []
        if var_type == 'categorical':
            var_specs = self.cat_var_specs
        elif var_type == 'ordinal':
            var_specs = self.ord_var_specs
        elif var_type == 'numerical':
            var_specs = self.num_var_specs
        else:
            raise ValueError(f"Invalid 'type' field. Expected 'numerical', "
                             f"'categorical', or 'ordinal'")
        
        # get variable names in the order they are listed
        ordered_vars = [var_spec.var_name for var_spec in var_specs]
        return ordered_vars

    def get_var_spec(self, var_name):
        """Returns the VarSpec corresponding to the provided variable name.
        
        Args:
            var_name: The name of the variable for which the VarSpec is to be 
                      returned.
        
        Returns:
            var_spec: The VarSpec corresponding to the provided variable name.
        
        Raises:
            ValueError: If the var_name is not found in the variable 
                        specifications.
        """
        for var_type in ['cat', 'ord', 'num']:
            for var_spec in getattr(self, var_type + "_var_specs"):
                if var_spec.var_name == var_name:
                    return var_spec

        raise ValueError(f"Variable name {var_name} is not found in the "
                         f"provided variable specifications")

    @classmethod
    def from_json(cls, json_path):
        """Creates a DatasetSpec object from a JSON file.
        
        Args:
            json_file: Path to the JSON file containing the dataset 
                       specifications.
        
        Returns:
            DatasetSpec: A new DatasetSpec object created from the JSON file.
        """
        with open(json_path, 'r') as file:
            json_dict = json.load(file)

        cat_var_specs = [
            VarSpec(
                **{
                    **spec,
                    'categorical_mapping': [
                        set(cat_map) for cat_map in 
                        spec.get('categorical_mapping', [])
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
                        set(cat_map) for cat_map in 
                        spec.get('categorical_mapping', [])
                    ]
                }
            )
            for spec in json_dict.get('ord_var_specs', [])
        ]

        num_var_specs = [VarSpec(**spec) 
                         for spec in json_dict.get('num_var_specs', [])]
        
        return cls(cat_var_specs, ord_var_specs, num_var_specs)

class SingleTargetModelSpec(ABC):
    """Abstract class for model specifications with a single target variable.
    
    This class defines the common structure and functionality for supervised 
    learning models with a single dependent variable. It handles basic
    validation and serialization while requiring subclasses to implement
    model-specific validation and behavior.
    
    Args:
        y_var: String specifying the dependent variable name.
        independent_vars: List of strings specifying the names of independent 
                          variables to include in the model.
        hyperparameters: Optional dictionary containing model-specific 
                         hyperparameters.
    """
    def __init__(self,
                 y_var: str,
                 independent_vars: List[str],
                 hyperparameters: Optional[Dict[str, Any]] = None):
        self.y_var = y_var
        self.independent_vars = independent_vars
        self.hyperparameters = hyperparameters or {}
        
        # Basic validations that don't require DatasetSpec
        self._validate_structure()
    
    def _validate_structure(self):
        """Validates the basic structure of the model specification.
        
        Checks that:
        1. The y_var is not empty
        2. There is at least one independent variable
        3. No variable is used as both dependent and independent
        
        Raises:
            ValueError: If any of the basic validation checks fail.
        """
        if not self.y_var:
            raise ValueError("Dependent variable (y_var) must be specified")
        
        if not self.independent_vars:
            raise ValueError(
                "At least one independent variable must be specified"
            )
        
        if self.y_var in self.independent_vars:
            raise ValueError(f"Variable '{self.y_var}' cannot be used as both "
                             f"dependent and independent variable")
    
    def validate_with_dataset_spec(self, dataset_spec):
        """Validates that the model specification is compatible with a dataset.
        
        Performs common validations and then calls model-specific validation.
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If y_var or any independent variable is not found in
                the dataset_spec or if model-specific validations fail.
        """
        # Check that y_var exists in dataset_spec
        all_var_names = dataset_spec.all_var_names
        if self.y_var not in all_var_names:
            raise ValueError(f"Dependent variable '{self.y_var}' not found in "
                             f"dataset specification")
        
        # Check that all independent variables exist in dataset_spec
        for var in self.independent_vars:
            if var not in all_var_names:
                raise ValueError(f"Independent variable '{var}' not found in "
                                 f"dataset specification")
        
        # Call model-specific validation
        self._validate_model_specific(dataset_spec)
    
    @abstractmethod
    def _validate_model_specific(self, dataset_spec):
        """Performs model-specific validation against a dataset specification.
        
        This method must be implemented by subclasses to check that the model
        specification is compatible with the dataset for the specific model
        type.
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If model-specific validations fail.
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Returns the type of model as a string.
        
        This property must be implemented by subclasses to identify the model
        type.
        
        Returns:
            str: A string identifier for the model type.
        """
        pass
    
    @abstractmethod
    def validate_hyperparameters(self):
        """Validates that the hyperparameters are appropriate for model type.
        
        This method must be implemented by subclasses to validate the
        hyperparameters specific to each model type.
        
        Raises:
            ValueError: If the hyperparameters are invalid for the model type.
        """
        pass

    def to_dict(self):
        """Converts the model specification to a dictionary.
        
        Returns:
            dict: A dictionary representation of the model specification.
        """
        return {
            'model_type': self.model_type,
            'y_var': self.y_var,
            'independent_vars': self.independent_vars,
            'hyperparameters': self.hyperparameters
        }
    
    def to_json(self, json_path):
        """Saves the model specification to a JSON file.
        
        Args:
            json_path: Path where the JSON file will be saved.
        """
        with open(json_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)
    
    @classmethod
    def from_dict(cls, model_dict):
        """Create appropriate model specification object from a dictionary.
        
        This is a factory method that creates the appropriate subclass instance
        based on the model_type field in the dictionary.
        
        Args:
            model_dict: Dictionary containing model specification.
            
        Returns:
            SingleTargetModelSpec: A new model specification object.
            
        Raises:
            ValueError: If the model_type is not recognized.
        """
        # This method needs to be implemented after subclasses are defined
        # It would use a registry pattern to map model_type strings to
        # subclasses
        raise NotImplementedError(
            "Subclasses should implement this method or use "
            "a registry pattern to handle different model types"
        )
    
    @classmethod
    def from_json(cls, json_path):
        """Creates a model specification object from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing the model
                specification.
            
        Returns:
            SingleTargetModelSpec: A new model specification object.
        """
        with open(json_path, 'r') as file:
            model_dict = json.load(file)
        
        return cls.from_dict(model_dict)


class RandomForestSpec(SingleTargetModelSpec):
    """Model specification for Random Forest models.
    
    Args:
        y_var: String specifying the dependent variable name.
        independent_vars: List of strings specifying the names of independent 
            variables to include in the model.
        hyperparameters: Optional dictionary containing RandomForest-specific 
            hyperparameters. Supported parameters include:
                - n_estimators: Number of trees (int)
                - max_features: Features to consider per split.
                    Can be one of:
                        * A string ('sqrt', 'log2', 'auto')
                        * An object with 'type': 'int' and 'value': <int>
                        * An object with 'type': 'float' and 'value': <float>
    """
    
    @property
    def model_type(self) -> str:
        """Returns the type of model.
        
        Returns:
            str: 'random_forest'
        """
        return 'random_forest'
    
    def _validate_model_specific(self, dataset_spec):
        """Performs Random Forest specific validation.
        
        For Random Forest, checks that:
        1. The hyperparameters are valid for Random Forest
        
        Args:
            dataset_spec: A DatasetSpec object containing variable
                specifications.
            
        Raises:
            ValueError: If model-specific validations fail.
        """
        # Validate hyperparameters
        self.validate_hyperparameters()

    def validate_hyperparameters(self):
        """Validates Random Forest specific hyperparameters.
        
        Checks that:
        1. n_estimators is present and a positive integer
        2. max_features is present with specified type and valid value
        
        Raises:
            ValueError: If any hyperparameter is invalid.
        """
        # Check required n_estimators
        if 'n_estimators' not in self.hyperparameters:
            raise ValueError("n_estimators is required for RandomForestSpec")
            
        n_estimators = self.hyperparameters['n_estimators']
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        
        # Check required max_features
        if 'max_features' not in self.hyperparameters:
            raise ValueError("max_features is required for RandomForestSpec")
            
        max_features = self.hyperparameters['max_features']
        
        # max_features must be a dictionary with type and value
        if not isinstance(max_features, dict):
            raise ValueError("max_features must be an object with 'type' and "
                            "'value' fields")
        
        if 'type' not in max_features or 'value' not in max_features:
            raise ValueError(
                "max_features must have 'type' and 'value' fields"
            )
        
        feat_type = max_features['type']
        feat_value = max_features['value']
        
        if feat_type == 'int':
            if not isinstance(feat_value, int) or feat_value <= 0:
                raise ValueError("max_features of type 'int' must have a "
                               "positive integer value")
        elif feat_type == 'float':
            if not isinstance(feat_value,
                              (int, float)) or not 0 < feat_value <= 1:
                raise ValueError("max_features of type 'float' must have a "
                               "value between 0 and 1")
        elif feat_type == 'string':
            raise ValueError("max_features 'type' must be 'int', 'float', "
                            "or 'string'")


class MixedDataset(Dataset):
    """A dataset to hold and manage mixed variable types.

    Categorical, ordinal, and numerical variables are supported. The class 
    accommodates optional data augmentation and masking.

    Args:
        dataset_spec (DatasetSpec): Specification of the dataset.
        Xcat (np.ndarray, optional): Numpy array holding categorical data.
        Xord (np.ndarray, optional): Numpy array holding ordinal data.
        Xnum (np.ndarray, optional): Numpy array holding numerical data.
        model_spec (SingleTargetModelSpec, optional): Model specification with
            target variable. If provided, the y_var will be extracted from this 
            instead of from dataset_spec.
        mask_prob (float, optional): Probability that a given input is masked.
            Default is 0.
        aug_mult (int, optional): Multiplier for data augmentation. Default is
            1.
        device (torch.device or str, optional): The torch device object or
            string to specify the device to use. Default is 'cpu'.
        require_input (bool, optional): If True, ensures that at least one
            variable remains unmasked in every item. Default is False.

    Attributes:
        dataset_spec (DatasetSpec): Specification of the dataset.
        model_spec (SingleTargetModelSpec, optional): Model specification.
        Xcat (torch.Tensor): Categorical data.
        Xord (torch.Tensor): Ordinal data.
        Xnum (torch.Tensor): Numerical data.
        y_data (torch.Tensor): Y variable data if it exists.
        num_obs (int): Number of observations in the dataset.
        mask_prob (float): Probability that a given input is masked.
        aug_mult (int): Multiplier for data augmentation.
        require_input (bool): If True, ensures that at least one variable
            remains unmasked in every item.
    """
    def __init__(
            self, 
            dataset_spec: 'DatasetSpec',
            Xcat: Optional[np.ndarray] = None,
            Xord: Optional[np.ndarray] = None,
            Xnum: Optional[np.ndarray] = None,
            model_spec: Optional['SingleTargetModelSpec'] = None,
            mask_prob: float = 0,
            aug_mult: int = 1,
            require_input: bool = False,
            device=None
        ):
        """Initialize a MixedDataset object."""
        self.dataset_spec = dataset_spec
        self.model_spec = model_spec
        self.mask_prob = mask_prob
        self.aug_mult = aug_mult
        self.require_input = require_input
        self.device = (torch.device('cpu') if device is None
                       else torch.device(device))

        if Xcat is None and Xord is None and Xnum is None:
            raise ValueError("Xcat, Xord, and Xnum cannot all be None")

        # Convert tensors to numpy if they are tensors
        Xcat = Xcat.cpu().numpy() if isinstance(Xcat, torch.Tensor) else Xcat
        Xord = Xord.cpu().numpy() if isinstance(Xord, torch.Tensor) else Xord
        Xnum = Xnum.cpu().numpy() if isinstance(Xnum, torch.Tensor) else Xnum

        # Check that all input arrays have the same number of samples
        num_obs_list = {len(X) for X in [Xcat, Xord, Xnum] if X is not None}
        if len(num_obs_list) > 1:
            raise ValueError(
                "Input arrays do not have the same number of samples"
            )
        self.num_obs = num_obs_list.pop()

        # Ensure that each input has the correct number of columns
        for var_type, X, s in zip(
                ['categorical', 'ordinal', 'numerical'],
                [Xcat, Xord, Xnum],
                ['Xcat', 'Xord', 'Xnum']
            ):
            num_var_ds = len(dataset_spec.get_ordered_variables(var_type))
            if X is None:
                num_var_mat = 0
            else:
                num_var_mat = X.shape[1]

            if num_var_ds != num_var_mat:
                raise ValueError(
                    f'{s} has {num_var_mat} columns but dataset_spec has '
                    f'{num_var_ds} {var_type} variables'
                )

        # Extract y_data before converting to tensor and moving to device
        y_var = (model_spec.y_var if model_spec is not None
                 else None)
        
        if y_var is not None:
            if y_var in dataset_spec.get_ordered_variables('categorical'):
                y_var_type = 'categorical'
                y_idx = dataset_spec.get_ordered_variables(
                    'categorical'
                ).index(y_var)
                self.y_data = Xcat[:, y_idx].copy()
                if Xcat.shape[1] > 1:
                    Xcat = np.delete(Xcat, y_idx, axis=1) 
                else:
                    Xcat = None
            elif y_var in dataset_spec.get_ordered_variables('ordinal'):
                y_var_type = 'ordinal'
                y_idx = dataset_spec.get_ordered_variables(
                    'ordinal'
                ).index(y_var)
                self.y_data = Xord[:, y_idx].copy()
                if Xord.shape[1] > 1:
                    Xord = np.delete(Xord, y_idx, axis=1)
                else:
                    Xord = None
            elif y_var in dataset_spec.get_ordered_variables('numerical'):
                y_var_type = 'numerical'
                y_idx = dataset_spec.get_ordered_variables(
                    'numerical'
                ).index(y_var)
                self.y_data = Xnum[:, y_idx].copy()
                if Xnum.shape[1] > 1:
                    Xnum = np.delete(Xnum, y_idx, axis=1)
                else:
                    Xnum = None
            else:
                raise ValueError(
                    f"y_var '{y_var}' not found in any variable type in "
                    "dataset_spec"
                )
        else:
            self.y_data = None

        # Convert numpy arrays to tensors and move to device
        self.Xcat = torch.tensor(
            Xcat, device=self.device
        ).long() if Xcat is not None else None
        self.Xord = torch.tensor(
            Xord, device=self.device
        ).long() if Xord is not None else None
        self.Xnum = torch.tensor(
            Xnum, device=self.device
        ).float() if Xnum is not None else None

        # Convert y_data to tensor and move to device
        if self.y_data is not None:
            if y_var_type == 'numerical':
                self.y_data = torch.tensor(self.y_data,
                                           device=self.device).float()
            else:
                assert y_var_type in ['categorical', 'ordinal']
                self.y_data = torch.tensor(self.y_data,
                                           device=self.device).long()

        # Create masks as tensors and move to device
        if Xcat is not None:
            self.Xcat_mask = torch.tensor(
                Xcat == 0, dtype=torch.bool, device=self.device
            )
        else:
            self.Xcat_mask = None

        if Xord is not None:
            self.Xord_mask = torch.tensor(
                Xord == 0, dtype=torch.bool, device=self.device
            )
        else:
            self.Xord_mask = None

        if Xnum is not None:
            self.Xnum_mask = torch.tensor(
                np.isnan(Xnum), dtype=torch.bool, device=self.device
            )
        else:
            self.Xnum_mask = None

    def __len__(self):
        """Return the number of items in the dataset.
        
        Returns:
            int: Number of observations, accounting for data augmentation.
        """
        return self.num_obs * self.aug_mult

    def __getitem__(self, idx):
        """Retrieve a dataset item with optional masking.
        
        Given an index, retrieves the corresponding item from the MixedDataset.
        The item consists of corresponding elements from Xcat, Xord, Xnum,
        a mask for numerical variables, and y_data (if they exist). 
        
        Incorporates artificial masking of inputs. If require_input is True,
        ensures that at least one variable remains unmasked.
    
        Args:
            idx (int): The index of the item to be fetched.
    
        Returns:
            list: A list containing elements from Xcat, Xord, Xnum, a mask for
                numerical variables, and y_data at the given index, with
                artificial masking applied.
        """
        orig_idx = idx // self.aug_mult
        item = []
        all_masked = True
        mask_collection = []
    
        # Apply artificial masking and append elements to the item
        for i, X in enumerate([self.Xcat, self.Xord, self.Xnum]):
            pre_mask = [self.Xcat_mask, self.Xord_mask, self.Xnum_mask][i]
            if X is not None:
                x = X[orig_idx, :].clone()
                if pre_mask is not None:
                    x[pre_mask[orig_idx, :]] = 0
                    
                mask_float = torch.rand_like(
                    x.to(torch.float32), 
                    device=self.device
                )
                mask = mask_float < self.mask_prob
                mask_collection.append(mask)
                x[mask] = 0
                item.append(x)
    
                # Check if all variables in the item are masked
                if (torch.any(mask == False) and 
                    torch.any(pre_mask[orig_idx, :] == False)):
                    all_masked = False
            else:
                item.append(None)
                mask_collection.append(None)
    
        # If all variables are masked and require_input is True, unmask a
        # random variable
        if self.require_input and all_masked:
            while all_masked:
                # Choose a random dataset (Xcat, Xord, Xnum)
                X_idx = torch.randint(
                    len([self.Xcat, self.Xord, self.Xnum]), 
                    (1,)
                ).item()
    
                # Skip if the chosen dataset is None
                if [self.Xcat, self.Xord, self.Xnum][X_idx] is None:
                    continue
    
                # Choose a random variable within the chosen dataset
                variable_idx = torch.randint(
                    [self.Xcat, self.Xord, self.Xnum][X_idx].shape[1], 
                    (1,)
                ).item()
    
                # Skip if the chosen variable is pre-masked
                masks = [self.Xcat_mask, self.Xord_mask, self.Xnum_mask]
                if masks[X_idx][orig_idx, variable_idx]:
                    continue
    
                # Unmask the chosen variable
                mask_collection[X_idx][variable_idx] = False
                data_sources = [self.Xcat, self.Xord, self.Xnum]
                item[X_idx][variable_idx] =\
                    data_sources[X_idx][orig_idx, variable_idx]
    
                all_masked = False
    
        # Append mask of numerical variables to item
        if self.Xnum is not None:
            num_mask = ~mask_collection[2]
            item.append(num_mask)
        else:
            item.append(None)
    
        # If y_data exists, append the corresponding value to the item
        if self.y_data is not None:
            item.append(self.y_data[orig_idx].unsqueeze(0))
    
        return item

    def get_arrays(self):
        """Return the input arrays and target data.
    
        Returns:
            tuple: A four element tuple containing:
                - Xcat (torch.Tensor or None): Categorical input data 
                - Xord (torch.Tensor or None): Ordinal input data
                - Xnum (torch.Tensor or None): Numerical input data
                - y_data (torch.Tensor or None): Target data
        """
        return (self.Xcat, self.Xord, self.Xnum, self.y_data)


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
